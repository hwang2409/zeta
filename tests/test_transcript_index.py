from __future__ import annotations

import asyncio
import io
import json
import multiprocessing
import os
import queue
import selectors
import sqlite3
import subprocess
import sys
import tarfile
import threading
import time
import traceback
from pathlib import Path

import pytest

import zeta.transcript_search.index as index_module
import zeta.transcript_search.refresh as refresh_module
from tests.support.branch_reads import branch_entries
from tests.support.processes import reap_process
from zeta.cli.main import build_parser
from zeta.cli.project import run as run_project
from zeta.core.session import SessionManager
from zeta.project_registry import ProjectRegistry
from zeta.transcript_search.index import (
    SANITIZER_VERSION,
    TranscriptIndex,
    TranscriptIndexUnavailable,
    TranscriptSource,
)
from zeta.transcript_search.refresh import (
    refresh_transcript_index,
    transcript_index_refresh_failure,
)

PROJECT_A = "p_" + "a" * 32
PROJECT_B = "p_" + "b" * 32
V4_REVISION = "9b8b7ea4c2e6873c22ad2f365c66e717873a4cce"


def _row(seq: int, role: str, text: str, *, state: str | None = None) -> dict:
    metadata = {"zeta.origin": "user", "origin": "human"} if role == "user" else {}
    if state is not None:
        metadata["response_state"] = state
    return {
        "seq": seq,
        "id": f"r-{seq}",
        "parent_id": f"r-{seq - 1}" if seq > 1 else None,
        "lane": "main",
        "type": "message",
        "data": {
            "message": {
                "role": role,
                "content": [{"type": "text", "text": text}],
                "metadata": metadata,
            }
        },
    }


def _source(root: Path, session_id: str, rows: list[dict]) -> TranscriptSource:
    session = root / session_id
    session.mkdir(parents=True)
    (session / "conversation.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return TranscriptSource(session_id, session)


def _result_ids(index: TranscriptIndex, query: str) -> list[str]:
    return [hit.unit_id for hit in index.search(query, limit=20)]


def _write_rows(source: TranscriptSource, rows: list[dict]) -> None:
    source.conversation_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _construct_index_process(child_index: int, project_dir: str, start, results) -> None:
    results.put((child_index, "ready", None))
    try:
        assert start.wait(60), "timed out waiting 60s for the parent start signal"
        TranscriptIndex(Path(project_dir), PROJECT_A)
    except BaseException:  # noqa: BLE001 - child must report every failure
        results.put((child_index, "error", traceback.format_exc()))
    else:
        results.put((child_index, "ok", None))


def _append_index_process(
    child_index: int,
    project_dir: str,
    session_dir: str,
    session_id: str,
    start,
    results,
) -> None:
    results.put((child_index, "ready", None))
    try:
        assert start.wait(60), "timed out waiting 60s for the parent start signal"
        index = TranscriptIndex(Path(project_dir), PROJECT_A)
        index.append(TranscriptSource(session_id, Path(session_dir)))
    except BaseException:  # noqa: BLE001 - child must report every failure
        results.put((child_index, "error", traceback.format_exc()))
    else:
        results.put((child_index, "ok", None))


def _open_project_index_process(
    child_index: int,
    projects_root: str,
    project_id: str,
    start,
    results,
) -> None:
    results.put((child_index, "ready", None))
    try:
        assert start.wait(60), "timed out waiting 60s for the parent start signal"
        registry = ProjectRegistry(Path(projects_root))
        index_module.open_project_index(registry, project_id)
    except BaseException:  # noqa: BLE001 - child must report every failure
        results.put((child_index, "error", traceback.format_exc()))
    else:
        results.put((child_index, "ok", None))


def _await_phase(children, results, phase: str, deadline: float) -> dict[int, tuple[str, object]]:
    messages: dict[int, tuple[str, object]] = {}
    while len(messages) < len(children):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return messages
        try:
            child_index, message_phase, payload = results.get(
                timeout=min(0.1, remaining)
            )
        except queue.Empty:
            for child_index, child in enumerate(children):
                child.is_alive()
                if child_index not in messages and child.exitcode not in (None, 0):
                    raise AssertionError(
                        f"child {child_index} exited before {phase}: "
                        f"exitcode={child.exitcode}"
                    )
            continue
        assert message_phase == phase or (
            phase == "ok" and message_phase == "error"
        ), f"child {child_index} reported {message_phase!r}, expected {phase!r}"
        messages[child_index] = (message_phase, payload)
    return messages


def _diagnostic_process(child_index: int, start, results) -> None:
    results.put((child_index, "ready", None))
    start.wait(60)
    try:
        if child_index == 3:
            raise RuntimeError("child 3 mutation")
    except BaseException:  # noqa: BLE001 - child must report every failure
        results.put((child_index, "error", traceback.format_exc()))
    else:
        results.put((child_index, "ok", None))


def _exit_before_ready_process(child_index: int, start, results) -> None:
    if child_index == 2:
        os._exit(7)
    results.put((child_index, "ready", None))
    start.wait(60)
    results.put((child_index, "ok", None))


def _run_index_processes(target, arguments: list[tuple]) -> list[str | None]:
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(target=target, args=(index, *items, start, results))
        for index, items in enumerate(arguments)
    ]
    messages: dict[int, tuple[str, object]] = {}

    def diagnostics() -> str:
        return "\n".join(
            f"child {number}: exitcode={process.exitcode}, "
            f"last_phase={messages.get(number, (None, None))[0]!r}, "
            f"traceback={messages.get(number, (None, None))[1]!r}"
            for number, process in enumerate(processes)
        )

    try:
        for process in processes:
            process.start()

        messages.update(
            _await_phase(
                processes,
                results,
                "ready",
                time.monotonic() + 60,
            )
        )
        assert len(messages) == len(processes), (
            "timed out waiting 60s for all children to report ready;\n"
            + diagnostics()
        )
        start.set()

        messages.update(
            _await_phase(
                processes,
                results,
                "ok",
                time.monotonic() + 60,
            )
        )
        assert len(messages) == len(processes), (
            "timed out waiting 60s for all children to finish;\n" + diagnostics()
        )
        for process in processes:
            process.join(60)
        assert all(message[0] == "ok" for message in messages.values()), diagnostics()
        assert all(process.exitcode == 0 for process in processes), diagnostics()
        return [None] * len(processes)
    finally:
        start.set()
        for process in processes:
            reap_process(process, timeout=60)


def test_process_diagnostics_keep_child_identity_and_traceback() -> None:
    with pytest.raises(AssertionError, match=r"child 3:.*child 3 mutation"):
        _run_index_processes(_diagnostic_process, [()] * 8)


def test_process_exit_before_ready_reports_exitcode() -> None:
    with pytest.raises(AssertionError, match=r"child 2 exited before ready: exitcode=7"):
        _run_index_processes(_exit_before_ready_process, [()] * 8)


def test_concurrent_processes_initialize_one_valid_database(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"

    errors = _run_index_processes(
        _construct_index_process, [(str(project_dir),) for _ in range(8)]
    )

    assert errors == [None] * 8
    assert TranscriptIndex(project_dir, PROJECT_A).status().ready


def test_concurrent_processes_initialize_and_append_all_sessions(
    tmp_path: Path,
) -> None:
    sessions_dir = tmp_path / "sessions"
    sources = [
        _source(
            sessions_dir,
            f"session-{number}",
            [
                _row(1, "user", f"canary {number}"),
                _row(2, "assistant", "answer", state="completed"),
            ],
        )
        for number in range(8)
    ]

    errors = _run_index_processes(
        _append_index_process,
        [
            (str(tmp_path / "project"), str(source.session_dir), source.session_id)
            for source in sources
        ],
    )

    assert errors == [None] * 8
    index = TranscriptIndex(tmp_path / "project", PROJECT_A)
    assert index.status().session_count == 8
    assert all(index.search(f"canary {number}") for number in range(8))


def test_rebuild_equals_incremental_append(tmp_path: Path) -> None:
    sources = [
        _source(
            tmp_path / "sessions",
            "one",
            [
                _row(1, "user", "alpha question"),
                _row(2, "assistant", "beta answer", state="completed"),
            ],
        ),
        _source(
            tmp_path / "sessions",
            "two",
            [
                _row(1, "user", "gamma question"),
                _row(2, "assistant", "delta answer", state="completed"),
            ],
        ),
    ]
    rebuilt = TranscriptIndex(tmp_path / "rebuilt", PROJECT_A)
    rebuilt.rebuild(sources)
    incremental = TranscriptIndex(tmp_path / "incremental", PROJECT_A)
    for source in sources:
        incremental.append(source)

    for query in ("alpha beta", "delta", "question"):
        assert _result_ids(rebuilt, query) == _result_ids(incremental, query)
    assert rebuilt.status().unit_count == incremental.status().unit_count == 2


def test_rebuild_skips_corrupt_session_and_names_it(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    good = _source(
        sessions,
        "good",
        [
            _row(1, "user", "healthy canary question"),
            _row(2, "assistant", "answer", state="completed"),
        ],
    )
    bad = _source(sessions, "bad", [_row(1, "user", "doomed")])
    # A newline-terminated invalid row after a valid one is the corruption that
    # blocked the whole project's rebuild on main with a raw "Expecting property
    # name".
    with bad.conversation_path.open("a", encoding="utf-8") as handle:
        handle.write("{not json}\n")
    index = TranscriptIndex(tmp_path / "project", PROJECT_A)

    try:
        status = index.rebuild((bad, good))
    except index_module.TranscriptIndexError:
        status = None
    assert status is not None, "rebuild aborted instead of skipping the corrupt session"
    assert _result_ids(index, "canary"), "good session not searchable after rebuild"
    assert any("bad" in message for message in status.skipped)


def test_incremental_state_keeps_only_sanitized_search_evidence(tmp_path: Path) -> None:
    private_key = (
        "-----BEGIN PRIVATE KEY-----\n"
        "TOP_SECRET_KEY_MATERIAL\n"
        "-----END PRIVATE KEY-----"
    )
    user = _row(1, "user", private_key)
    assistant = _row(2, "assistant", "safe answer", state="completed")
    assistant["data"]["message"]["content"].insert(
        0, {"type": "thinking", "thinking": "PRIVATE_REASONING_MARKER"}
    )
    source = _source(tmp_path / "sessions", "one", [user, assistant])
    index = TranscriptIndex(tmp_path / "project", PROJECT_A)

    index.append(source)

    with sqlite3.connect(index.path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        stored = "\n".join(
            row[0] for row in connection.execute("SELECT text FROM units")
        )
    indexed = "\n".join(hit.unit.text for hit in index.search("safe answer"))
    assert "source_rows" not in tables
    assert "TOP_SECRET_KEY_MATERIAL" not in stored
    assert "END PRIVATE KEY" not in stored
    assert "PRIVATE_REASONING_MARKER" not in stored
    assert "TOP_SECRET_KEY_MATERIAL" not in indexed
    assert "PRIVATE_REASONING_MARKER" not in indexed


def test_deletion_and_reassignment(tmp_path: Path) -> None:
    source = _source(
        tmp_path / "sessions",
        "one",
        [
            _row(1, "user", "moving canary"),
            _row(2, "assistant", "answer", state="completed"),
        ],
    )
    old = TranscriptIndex(tmp_path / "old", PROJECT_A)
    new = TranscriptIndex(tmp_path / "new", PROJECT_B)
    old.append(source)
    old.delete_session(source.session_id)
    new.append(source)
    assert old.search("moving canary") == ()
    assert new.search("moving canary")


def test_v4_and_v5_processes_search_the_same_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(
        tmp_path / "sessions",
        "one",
        [
            _row(1, "user", "mixed version canary"),
            _row(2, "assistant", "answer", state="completed"),
        ],
    )
    base_checkout = tmp_path / "base"
    archive = subprocess.run(
        ["git", "archive", V4_REVISION, "src/zeta"],
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        bundle.extractall(base_checkout, filter="data")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(base_checkout / "src")
    script = """
import sys
from pathlib import Path
from zeta.transcript_search.index import TranscriptIndex, TranscriptSource

project_dir, session_dir = map(Path, sys.argv[1:])
index = TranscriptIndex(project_dir, 'p_' + 'a' * 32)
index.append(TranscriptSource('one', session_dir))
assert index.search('mixed version canary')
print(index.path.name, flush=True)
sys.stdin.readline()
assert index.search('mixed version canary')
print('v4-found', flush=True)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path / "project"), str(source.session_dir)],
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            assert selector.select(10), "v4 process did not build its index"
        assert process.stdout.readline().strip() == "transcript-index.sqlite3"

        v5 = TranscriptIndex(tmp_path / "project", PROJECT_A)
        v5.append(source)
        assert v5.search("mixed version canary")

        lock_held = threading.Event()
        release_append = threading.Event()
        append_errors: list[BaseException] = []
        original_commit = v5._commit_prepared_locked

        def hold_commit(prepared):
            lock_held.set()
            assert release_append.wait(10), "test did not release v5 append"
            return original_commit(prepared)

        def append_with_held_lock() -> None:
            try:
                v5.append(source)
            except BaseException as exc:  # noqa: BLE001 - assert thread failure
                append_errors.append(exc)

        # test-contract: seam hold append after it acquires the production lock
        monkeypatch.setattr(v5, "_commit_prepared_locked", hold_commit)
        append_thread = threading.Thread(target=append_with_held_lock)
        append_thread.start()
        try:
            assert lock_held.wait(10), "v5 append did not acquire its lock"
            assert process.stdin is not None
            process.stdin.write("search\n")
            process.stdin.flush()
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                assert selector.select(10), "v4 search blocked on the v5 append lock"
            result = process.stdout.readline().strip()
        finally:
            release_append.set()
            append_thread.join(timeout=10)
        assert not append_thread.is_alive(), "v5 append did not finish"
        assert not append_errors
        returncode = process.wait(timeout=10)
        assert process.stderr is not None
        error = process.stderr.read()
        assert result == "v4-found", error
        assert returncode == 0, error
        assert v5.path.name == "transcript-index-v5.sqlite3"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()


def test_malformed_schema_metadata_constructs_for_rebuild(tmp_path: Path) -> None:
    source = _source(
        tmp_path / "sessions",
        "one",
        [
            _row(1, "user", "malformed metadata canary"),
            _row(2, "assistant", "answer", state="completed"),
        ],
    )
    index = TranscriptIndex(tmp_path / "project", PROJECT_A)
    with sqlite3.connect(index.path) as connection:
        connection.execute(
            "UPDATE metadata SET value = 'malformed' WHERE key = 'schema_version'"
        )
        connection.commit()

    recovered = TranscriptIndex(tmp_path / "project", PROJECT_A)
    status = recovered.status()
    assert not status.ready
    assert status.detail is not None and "rebuild" in status.detail
    recovered.rebuild((source,))
    assert recovered.search("malformed metadata canary")


def test_missing_cursor_table_constructs_for_rebuild(tmp_path: Path) -> None:
    source = _source(
        tmp_path / "sessions",
        "one",
        [
            _row(1, "user", "missing cursor canary"),
            _row(2, "assistant", "answer", state="completed"),
        ],
    )
    index = TranscriptIndex(tmp_path / "project", PROJECT_A)
    with sqlite3.connect(index.path) as connection:
        connection.execute("DROP TABLE session_cursors")
        connection.commit()

    recovered = TranscriptIndex(tmp_path / "project", PROJECT_A)
    status = recovered.status()
    assert not status.ready
    assert status.detail is not None and "rebuild" in status.detail
    recovered.rebuild((source,))
    assert recovered.search("missing cursor canary")


def test_cursor_lag_resumes_without_duplicates(tmp_path: Path) -> None:
    source = _source(
        tmp_path / "sessions",
        "one",
        [
            _row(1, "user", "crash recovery"),
            _row(2, "assistant", "idempotent", state="completed"),
        ],
    )
    index = TranscriptIndex(tmp_path / "project", PROJECT_A)
    index.append(source)
    with sqlite3.connect(index.path) as connection:
        connection.execute(
            "UPDATE session_cursors SET last_seq = 0 WHERE session_id = ?",
            (source.session_id,),
        )
        connection.commit()
    index.append(source)
    status = index.status()
    assert status.unit_count == 1
    assert status.cursors[source.session_id] == 2


def test_relaxed_search_uses_distinctive_partial_terms(tmp_path: Path) -> None:
    source = _source(
        tmp_path / "sessions",
        "one",
        [
            _row(1, "user", "common common platypus"),
            _row(2, "assistant", "answer", state="completed"),
        ],
    )
    other = _source(
        tmp_path / "sessions",
        "two",
        [
            _row(1, "user", "common ordinary"),
            _row(2, "assistant", "answer", state="completed"),
        ],
    )
    index = TranscriptIndex(tmp_path / "project", PROJECT_A)
    index.rebuild((source, other))
    hits = index.search("missing common platypus")
    assert hits
    assert "platypus" in hits[0].unit.text
    assert hits[0].match == "partial"


def test_missing_receipts_verify_bounded_prefix_then_read_only_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [_row(1, "user", "prefix canary")]
    source = _source(tmp_path / "sessions", "one", rows)
    index = TranscriptIndex(tmp_path / "project", PROJECT_A)
    index.append(source)
    observed: list[object] = []
    original_read = index_module._read_transcript

    def observe(path: Path, cursor, receipts=None, *, session_id):
        result = original_read(path, cursor, receipts, session_id=session_id)
        observed.append(result)
        return result

    # test-contract: seam count bounded transcript reads through the index
    monkeypatch.setattr(index_module, "_read_transcript", observe)
    with source.conversation_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(_row(2, "assistant", "tail canary", state="completed"))
            + "\n"
        )
    index.append(source)

    read = observed[-1]
    assert read.full is False
    assert read.bytes_read < source.conversation_path.stat().st_size
    assert index.search("tail canary")


def test_prefix_mutation_without_receipts_forces_full_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(
        tmp_path / "sessions",
        "one",
        [
            _row(1, "user", "oldcanary"),
            _row(2, "assistant", "answer", state="completed"),
        ],
    )
    index = TranscriptIndex(tmp_path / "project", PROJECT_A)
    index.append(source)
    original_read = index_module._read_transcript
    observed: list[object] = []

    def observe(path: Path, cursor, receipts=None, *, session_id):
        result = original_read(path, cursor, receipts, session_id=session_id)
        observed.append(result)
        return result

    # test-contract: seam observe fallback through the index reader
    monkeypatch.setattr(index_module, "_read_transcript", observe)
    _write_rows(
        source,
        [
            _row(1, "user", "newcanary"),
            _row(2, "assistant", "answer", state="completed"),
            _row(3, "user", "tail"),
        ],
    )
    index.append(source)
    assert observed[-1].full is True
    assert index.search("newcanary")
    assert not index.search("oldcanary")


def test_missing_receipts_incremental_results_equal_clean_rebuild(tmp_path: Path) -> None:
    source = _source(
        tmp_path / "sessions",
        "one",
        [
            _row(1, "user", "first"),
            _row(2, "assistant", "answer", state="completed"),
        ],
    )
    incremental = TranscriptIndex(tmp_path / "incremental", PROJECT_A)
    incremental.append(source)
    with source.conversation_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_row(3, "user", "second")) + "\n")
        handle.write(json.dumps(_row(4, "assistant", "reply", state="completed")) + "\n")
    incremental.append(source)
    rebuilt = TranscriptIndex(tmp_path / "rebuilt", PROJECT_A)
    rebuilt.rebuild((source,))
    for query in ("first", "second", "reply"):
        assert _result_ids(incremental, query) == _result_ids(rebuilt, query)


def test_unverified_append_rereads_and_matches_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(
        tmp_path / "sessions",
        "one",
        [
            _row(1, "user", "first question"),
            _row(2, "assistant", "first answer", state="completed"),
        ],
    )
    index = TranscriptIndex(tmp_path / "incremental", PROJECT_A)
    index.append(source)
    first_offset = index.path.stat().st_size
    original_read = index_module._read_transcript
    reads: list[object] = []

    def observe(path: Path, cursor, receipts=None, *, session_id):
        result = original_read(path, cursor, receipts, session_id=session_id)
        reads.append(result)
        return result

    # test-contract: seam count bounded transcript reads through the index
    monkeypatch.setattr(index_module, "_read_transcript", observe)
    with source.conversation_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_row(3, "user", "second question")) + "\n")
        handle.write(
            json.dumps(_row(4, "assistant", "second answer", state="completed")) + "\n"
        )
    index.append(source)

    rebuilt = TranscriptIndex(tmp_path / "rebuilt", PROJECT_A)
    rebuilt.rebuild((source,))
    assert len(reads) == 2
    assert reads[0].full is False
    assert first_offset > 0
    for query in ("first answer", "second question", "second answer"):
        assert _result_ids(index, query) == _result_ids(rebuilt, query)


def test_incomplete_turn_uses_only_bounded_active_tail(tmp_path: Path) -> None:
    source = _source(
        tmp_path / "sessions", "one", [_row(1, "user", "pending question")]
    )
    index = TranscriptIndex(tmp_path / "project", PROJECT_A)
    index.append(source)

    with sqlite3.connect(index.path) as connection:
        tail = json.loads(
            connection.execute(
                "SELECT active_tail_json FROM session_cursors WHERE session_id = ?",
                (source.session_id,),
            ).fetchone()[0]
        )
    assert [row["seq"] for row in tail] == [1]
    assert index.status().unit_count == 0

    with source.conversation_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(_row(2, "assistant", "completed answer", state="completed"))
            + "\n"
        )
    index.append(source)

    with sqlite3.connect(index.path) as connection:
        tail_json = connection.execute(
            "SELECT active_tail_json FROM session_cursors WHERE session_id = ?",
            (source.session_id,),
        ).fetchone()[0]
    assert tail_json == "[]"
    assert index.search("pending question")
    assert index.search("completed answer")


def test_equal_length_same_inode_rewrite_forces_full_reread(tmp_path: Path) -> None:
    old_rows = [
        _row(1, "user", "oldcanary"),
        _row(2, "assistant", "answer", state="completed"),
    ]
    source = _source(tmp_path / "sessions", "one", old_rows)
    index = TranscriptIndex(tmp_path / "project", PROJECT_A)
    index.append(source)
    inode = source.conversation_path.stat().st_ino

    new_rows = [
        _row(1, "user", "newcanary"),
        _row(2, "assistant", "answer", state="completed"),
    ]
    _write_rows(source, new_rows)
    assert source.conversation_path.stat().st_ino == inode
    assert source.conversation_path.stat().st_size == len(
        "".join(json.dumps(row) + "\n" for row in old_rows).encode()
    )

    index.append(source)

    assert index.search("oldcanary") == ()
    assert index.search("newcanary")


def test_truncate_and_regrow_larger_forces_full_reread(tmp_path: Path) -> None:
    source = _source(
        tmp_path / "sessions",
        "one",
        [
            _row(1, "user", "oldcanary"),
            _row(2, "assistant", "old answer", state="completed"),
        ],
    )
    index = TranscriptIndex(tmp_path / "project", PROJECT_A)
    index.append(source)

    replacement = [
        _row(1, "user", "newcanary " + "x" * 800),
        _row(2, "assistant", "new answer", state="completed"),
        _row(3, "user", "regrown tail"),
        _row(4, "assistant", "tail answer", state="completed"),
    ]
    _write_rows(source, replacement)
    index.append(source)

    assert index.search("oldcanary") == ()
    assert index.search("newcanary")
    assert index.search("regrown tail")


def test_rewrite_before_cursor_with_appended_tail_forces_full_reread(
    tmp_path: Path,
) -> None:
    source = _source(
        tmp_path / "sessions",
        "one",
        [
            _row(1, "user", "oldcanary"),
            _row(2, "assistant", "first answer", state="completed"),
        ],
    )
    index = TranscriptIndex(tmp_path / "project", PROJECT_A)
    index.append(source)

    _write_rows(
        source,
        [
            _row(1, "user", "newcanary"),
            _row(2, "assistant", "first answer", state="completed"),
            _row(3, "user", "appended canary"),
            _row(4, "assistant", "second answer", state="completed"),
        ],
    )
    index.append(source)

    assert index.search("oldcanary") == ()
    assert index.search("newcanary")
    assert index.search("appended canary")


def test_database_size_is_bounded_against_indexed_text(tmp_path: Path) -> None:
    rows = []
    paragraph = " ".join(f"representative-token-{index % 80}" for index in range(240))
    for turn in range(800):
        rows.extend(
            (
                _row(2 * turn + 1, "user", f"question {turn} {paragraph}"),
                _row(
                    2 * turn + 2,
                    "assistant",
                    f"answer {turn} {paragraph}",
                    state="completed",
                ),
            )
        )
    source = _source(tmp_path / "sessions", "one", rows)
    index = TranscriptIndex(tmp_path / "project", PROJECT_A)

    index.rebuild((source,))
    with sqlite3.connect(index.path) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        indexed_bytes = int(
            connection.execute(
                "SELECT coalesce(sum(length(cast(text AS blob))), 0) FROM units"
            ).fetchone()[0]
        )
    database_bytes = index.path.stat().st_size

    assert database_bytes <= indexed_bytes * 1.5, (
        f"database={database_bytes}, indexed_text={indexed_bytes}, "
        f"ratio={database_bytes / indexed_bytes:.3f}"
    )


def test_first_version_backfill_reports_building_and_keeps_concurrent_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    projects_root = tmp_path / "projects"
    registry = ProjectRegistry(projects_root)
    project = registry.create_project("demo", "demo", str(tmp_path))
    historical = _source(
        tmp_path / "sessions",
        "historical",
        [
            _row(1, "user", "historical build canary"),
            _row(2, "assistant", "answer", state="completed"),
        ],
    )
    (historical.session_dir / "meta.json").write_text(
        json.dumps({"project_id": project.project_id}), encoding="utf-8"
    )
    registry.record_session(
        project.project_id,
        session_id=historical.session_id,
        transcript_path=str(historical.session_dir),
    )
    entered = threading.Event()
    release = threading.Event()
    original = index_module.render_transcript_units
    calls = 0

    def controlled(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert release.wait(10), "test did not release version backfill"
        return original(*args)

    monkeypatch.setattr(index_module, "render_transcript_units", controlled)
    errors: list[BaseException] = []

    def backfill() -> None:
        try:
            index_module.open_project_index(registry, project.project_id)
        except BaseException as exc:  # noqa: BLE001 - assert thread failure
            errors.append(exc)

    backfill_thread = threading.Thread(target=backfill)
    backfill_thread.start()
    append_thread: threading.Thread | None = None
    try:
        assert entered.wait(10), "version backfill did not start"
        building = index_module.open_project_index(registry, project.project_id)
        status = building.status()
        assert not status.ready
        assert status.detail == "index building"
        with pytest.raises(TranscriptIndexUnavailable, match="index building"):
            building.search("historical")

        active = _source(
            tmp_path / "sessions",
            "active",
            [
                _row(1, "user", "concurrent append canary"),
                _row(2, "assistant", "answer", state="completed"),
            ],
        )
        (active.session_dir / "meta.json").write_text(
            json.dumps({"project_id": project.project_id}), encoding="utf-8"
        )
        registry.record_session(
            project.project_id,
            session_id=active.session_id,
            transcript_path=str(active.session_dir),
        )

        def append() -> None:
            try:
                refresh_module._refresh_sources(
                    registry, project.project_id, (active,)
                )
            except BaseException as exc:  # noqa: BLE001 - assert thread failure
                errors.append(exc)

        append_thread = threading.Thread(target=append)
        append_thread.start()
    finally:
        release.set()
        backfill_thread.join(10)
        if append_thread is not None:
            append_thread.join(10)

    assert not backfill_thread.is_alive()
    assert append_thread is not None and not append_thread.is_alive()
    assert not errors
    complete = index_module.open_project_index(registry, project.project_id)
    assert complete.search("historical build canary")
    assert complete.search("concurrent append canary")


def test_concurrent_processes_backfill_the_version_once(tmp_path: Path) -> None:
    projects_root = tmp_path / "projects"
    registry = ProjectRegistry(projects_root)
    project = registry.create_project("demo", "demo", str(tmp_path))
    for number in range(4):
        source = _source(
            tmp_path / "sessions",
            f"session-{number}",
            [
                _row(1, "user", f"backfill canary {number}"),
                _row(2, "assistant", "answer", state="completed"),
            ],
        )
        (source.session_dir / "meta.json").write_text(
            json.dumps({"project_id": project.project_id}), encoding="utf-8"
        )
        registry.record_session(
            project.project_id,
            session_id=source.session_id,
            transcript_path=str(source.session_dir),
        )

    errors = _run_index_processes(
        _open_project_index_process,
        [(str(projects_root), project.project_id) for _ in range(8)],
    )

    assert errors == [None] * 8
    index = index_module.open_project_index(registry, project.project_id)
    status = index.status()
    assert status.ready
    assert status.session_count == 4
    # generation increments once per build; a value of 1 proves exactly one of the
    # eight racing processes built the version and the rest deferred.
    assert status.generation == 1
    assert all(index.search(f"backfill canary {number}") for number in range(4))


def test_stale_building_marker_from_crashed_backfill_is_recovered(
    tmp_path: Path,
) -> None:
    source = _source(
        tmp_path / "sessions",
        "one",
        [
            _row(1, "user", "stale marker canary"),
            _row(2, "assistant", "answer", state="completed"),
        ],
    )
    index = TranscriptIndex(tmp_path / "project", PROJECT_A)
    index.append(source)
    assert index.search("stale marker canary")
    # a backfill killed mid-run leaves the marker behind with no live write lock.
    index.building_path.touch()
    assert index.status().ready
    assert index.status().detail != "index building"
    assert index.search("stale marker canary")
    assert not index.building_path.exists()


def test_duplicate_append_never_regresses_cursor(tmp_path: Path) -> None:
    source = _source(
        tmp_path / "sessions",
        "one",
        [
            _row(1, "user", "old question"),
            _row(2, "assistant", "old answer", state="completed"),
            _row(3, "user", "new canary"),
            _row(4, "assistant", "new answer", state="completed"),
        ],
    )
    index = TranscriptIndex(tmp_path / "project", PROJECT_A)
    index.append(source)
    index.append(source)

    assert index.status().cursors["one"] == 4
    assert index.search("new canary")


def test_rebuild_concurrent_with_append_loses_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(
        tmp_path / "sessions",
        "one",
        [
            _row(1, "user", "old question"),
            _row(2, "assistant", "old answer", state="completed"),
        ],
    )
    index = TranscriptIndex(tmp_path / "project", PROJECT_A)
    entered = threading.Event()
    release = threading.Event()
    original = index_module.render_transcript_units
    calls = 0

    def controlled(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(index_module, "render_transcript_units", controlled)
    rebuild = threading.Thread(target=index.rebuild, args=((source,),))
    rebuild.start()
    assert entered.wait(5)
    with source.conversation_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_row(3, "user", "new canary")) + "\n")
        handle.write(
            json.dumps(_row(4, "assistant", "new answer", state="completed")) + "\n"
        )
    append = threading.Thread(target=index.append, args=(source,))
    append.start()
    release.set()
    rebuild.join(5)
    append.join(5)

    assert not rebuild.is_alive() and not append.is_alive()
    assert index.status().cursors["one"] == 4
    assert index.search("new canary")


@pytest.mark.parametrize("action", ["reassign", "delete"])
def test_inflight_refresh_cannot_repopulate_unbound_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    source = _source(
        tmp_path / "sessions",
        "one",
        [
            _row(1, "user", "binding canary"),
            _row(2, "assistant", "answer", state="completed"),
        ],
    )
    metadata = source.session_dir / "meta.json"
    metadata.write_text(json.dumps({"project_id": PROJECT_A}), encoding="utf-8")
    bound = TranscriptSource("one", source.session_dir, PROJECT_A)
    index = TranscriptIndex(tmp_path / "projects" / PROJECT_A, PROJECT_A)
    entered = threading.Event()
    release = threading.Event()
    original = index_module.render_transcript_units

    def controlled(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(index_module, "render_transcript_units", controlled)
    refresh = threading.Thread(target=index.append, args=(bound,))
    refresh.start()
    assert entered.wait(5)
    if action == "reassign":
        metadata.write_text(json.dumps({"project_id": PROJECT_B}), encoding="utf-8")
    else:
        metadata.unlink()
    release.set()
    refresh.join(5)

    assert not refresh.is_alive()
    assert index.search("binding canary") == ()


def test_search_progresses_while_append_projects_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(
        tmp_path / "sessions",
        "one",
        [
            _row(1, "user", "old searchable canary"),
            _row(2, "assistant", "old answer", state="completed"),
        ],
    )
    index = TranscriptIndex(tmp_path / "project", PROJECT_A)
    index.append(source)
    _write_rows(
        source,
        [
            _row(1, "user", "old searchable canary"),
            _row(2, "assistant", "old answer", state="completed"),
            _row(3, "user", "new question"),
            _row(4, "assistant", "new answer", state="completed"),
        ],
    )
    projection_entered = threading.Event()
    release_projection = threading.Event()
    search_done = threading.Event()
    original = index_module.render_transcript_units
    hits = ()

    def controlled(*args):
        projection_entered.set()
        assert release_projection.wait(10)
        return original(*args)

    def search() -> None:
        nonlocal hits
        search_index = TranscriptIndex(tmp_path / "project", PROJECT_A)
        hits = search_index.search("old searchable canary")
        search_done.set()

    monkeypatch.setattr(index_module, "render_transcript_units", controlled)
    append = threading.Thread(target=index.append, args=(source,))
    search_thread = threading.Thread(target=search)
    try:
        append.start()
        assert projection_entered.wait(10)
        search_thread.start()
        assert search_done.wait(10), "public search blocked on source projection"
        assert [hit.unit.session_id for hit in hits] == ["one"]
    finally:
        release_projection.set()
        append.join(10)
        search_thread.join(10)
    assert not append.is_alive() and not search_thread.is_alive()


def test_append_retries_three_cursor_cas_losses_without_regressing_winners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        _row(1, "user", "initial question"),
        _row(2, "assistant", "initial answer", state="completed"),
    ]
    source = _source(tmp_path / "sessions", "one", rows)
    stale = TranscriptIndex(tmp_path / "project", PROJECT_A)
    winner = TranscriptIndex(tmp_path / "project", PROJECT_A)
    entered = [threading.Event() for _ in range(3)]
    releases = [threading.Event() for _ in range(3)]
    original = index_module.render_transcript_units
    stale_thread_id: int | None = None
    attempts = 0
    error: BaseException | None = None
    winner_threads: list[threading.Thread] = []

    def controlled(*args):
        nonlocal attempts
        if threading.get_ident() == stale_thread_id:
            attempt = attempts
            attempts += 1
            entered[attempt].set()
            assert releases[attempt].wait(10)
        return original(*args)

    def append_stale() -> None:
        nonlocal stale_thread_id, error
        stale_thread_id = threading.get_ident()
        try:
            stale.append(source)
        except index_module.TranscriptIndexError as exc:
            error = exc

    monkeypatch.setattr(index_module, "render_transcript_units", controlled)
    thread = threading.Thread(target=append_stale)
    try:
        thread.start()
        for attempt in range(3):
            assert entered[attempt].wait(10)
            seq = len(rows) + 1
            rows.extend(
                [
                    _row(seq, "user", f"winner {attempt} question"),
                    _row(seq + 1, "assistant", "winner answer", state="completed"),
                ]
            )
            _write_rows(source, rows)
            winner_done = threading.Event()

            def append_winner(done: threading.Event = winner_done) -> None:
                winner.append(source)
                done.set()

            winner_thread = threading.Thread(target=append_winner)
            winner_threads.append(winner_thread)
            winner_thread.start()
            assert winner_done.wait(10), "competing public append blocked on projection"
            winner_thread.join(10)
            releases[attempt].set()
    finally:
        for release in releases:
            release.set()
        thread.join(10)
        for winner_thread in winner_threads:
            winner_thread.join(10)
    assert not thread.is_alive()
    assert attempts == 3
    assert isinstance(error, index_module.TranscriptIndexError)
    assert str(error) == "transcript changed during index refresh; retry required"
    assert stale.status().cursors["one"] == 8
    assert len(stale.search("winner question", limit=20)) == 3


def test_unbound_only_project_backfills_once_then_stays_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("demo", "demo", str(tmp_path))
    # a linked session whose dir has no meta.json is indexable (no
    # attention_fork.json) but unbound, so the rebuild publishes zero cursors
    # and the index stays ready-but-empty.
    session_dir = home / "sessions" / "ghost"
    session_dir.mkdir(parents=True)
    (session_dir / "conversation.jsonl").write_text(
        json.dumps(_row(1, "user", "ghost canary")) + "\n", encoding="utf-8"
    )
    registry.record_session(
        project.project_id,
        session_id="ghost",
        transcript_path=str(session_dir),
    )

    first = index_module.open_project_index(registry, project.project_id)
    assert first.status().session_count == 0
    assert first.status().generation == 1

    second = index_module.open_project_index(registry, project.project_id)
    assert second.status().session_count == 0
    # on base 545bd02a the ready-but-empty re-trigger rebuilds on every open,
    # so generation would be 2 here; the generation-0 gate keeps it at 1.
    assert second.status().generation == 1


def test_legacy_generation_zero_empty_file_backfills_bound_source_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("demo", "demo", str(tmp_path))
    project_dir = registry.root / project.project_id
    # a legacy #1123 build left an empty v5 file at generation 0.
    TranscriptIndex(project_dir, project.project_id)

    session_dir = home / "sessions" / "bound"
    session_dir.mkdir(parents=True)
    (session_dir / "conversation.jsonl").write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                _row(1, "user", "bound canary"),
                _row(2, "assistant", "answer", state="completed"),
            )
        ),
        encoding="utf-8",
    )
    (session_dir / "meta.json").write_text(
        json.dumps({"project_id": project.project_id}), encoding="utf-8"
    )
    registry.record_session(
        project.project_id,
        session_id="bound",
        transcript_path=str(session_dir),
    )

    first = index_module.open_project_index(registry, project.project_id)
    assert first.search("bound canary")
    assert first.status().generation == 1

    second = index_module.open_project_index(registry, project.project_id)
    assert second.status().generation == 1


def test_version_mismatch_refuses_old_rows_and_rebuild_restores(tmp_path: Path) -> None:
    source = _source(
        tmp_path / "sessions",
        "one",
        [
            _row(1, "user", "version canary"),
            _row(2, "assistant", "answer", state="completed"),
        ],
    )
    index = TranscriptIndex(tmp_path / "project", PROJECT_A)
    index.append(source)
    with sqlite3.connect(index.path) as connection:
        connection.execute(
            "UPDATE metadata SET value = ? WHERE key = 'sanitizer_version'",
            (str(SANITIZER_VERSION - 1),),
        )
        connection.commit()

    assert not index.status().ready
    with pytest.raises(TranscriptIndexUnavailable, match="rebuild"):
        index.search("version canary")
    index.rebuild((source,))
    assert index.status().ready
    assert index.search("version canary")


def test_rebuild_recovers_schema_mismatch_and_corrupt_database(tmp_path: Path) -> None:
    source = _source(
        tmp_path / "sessions",
        "one",
        [
            _row(1, "user", "repair canary"),
            _row(2, "assistant", "answer", state="completed"),
        ],
    )
    project_dir = tmp_path / "project"
    index = TranscriptIndex(project_dir, PROJECT_A)
    with sqlite3.connect(index.path) as connection:
        connection.execute(
            "UPDATE metadata SET value = '0' WHERE key = 'schema_version'"
        )
        connection.commit()
    index.rebuild((source,))
    assert index.search("repair canary")

    index.path.write_bytes(b"not sqlite")
    repaired = TranscriptIndex(project_dir, PROJECT_A)
    assert not repaired.status().ready
    repaired.rebuild((source,))
    assert repaired.search("repair canary")


def test_rebuild_publishes_over_existing_wal_sidecars(tmp_path: Path) -> None:
    source = _source(
        tmp_path / "sessions",
        "one",
        [
            _row(1, "user", "published canary"),
            _row(2, "assistant", "answer", state="completed"),
        ],
    )
    index = TranscriptIndex(tmp_path / "project", PROJECT_A)
    reader = sqlite3.connect(index.path)
    try:
        reader.execute("PRAGMA journal_mode=WAL")
        reader.execute("SELECT count(*) FROM metadata").fetchone()
        index.rebuild((source,))
        assert index.search("published canary")
    finally:
        reader.close()


def test_index_uses_wal(tmp_path: Path) -> None:
    index = TranscriptIndex(tmp_path / "project", PROJECT_A)
    with sqlite3.connect(index.path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_operator_cli_rebuild_status_and_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("demo", "demo", str(tmp_path))
    opened = SessionManager(home).create(
        provider="codex",
        model="test",
        cwd=tmp_path,
        project_id=project.project_id,
        auto_project=False,
    )
    source = TranscriptSource(opened.metadata.session_id, opened.store.session_dir)
    opened.store.close()
    _write_rows(
        source,
        [
            _row(1, "user", "operator canary"),
            _row(2, "assistant", "answer", state="completed"),
        ],
    )
    for arguments in (
        ["project", "index", "demo", "rebuild"],
        ["project", "index", "demo", "status"],
        ["project", "index", "demo", "search", "operator", "canary"],
    ):
        assert run_project(build_parser().parse_args(arguments)) == 0
    output = capsys.readouterr().out
    assert "operator canary" in output
    assert '"generation": 1' in output


def test_first_v5_search_backfills_linked_sessions_from_v4_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("demo", "demo", str(tmp_path))
    source = _source(
        home / "sessions",
        "historical",
        [
            _row(1, "user", "historical backfill canary"),
            _row(2, "assistant", "answer", state="completed"),
        ],
    )
    (source.session_dir / "meta.json").write_text(
        json.dumps({"project_id": project.project_id}), encoding="utf-8"
    )
    registry.record_session(
        project.project_id,
        session_id=source.session_id,
        transcript_path=str(source.session_dir),
    )
    base_checkout = tmp_path / "base"
    archive = subprocess.run(
        ["git", "archive", V4_REVISION, "src/zeta"],
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        bundle.extractall(base_checkout, filter="data")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(base_checkout / "src")
    built = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from pathlib import Path
from zeta.transcript_search.index import TranscriptIndex, TranscriptSource
index = TranscriptIndex(Path(sys.argv[1]), sys.argv[2])
index.append(TranscriptSource('historical', Path(sys.argv[3])))
assert index.search('historical backfill canary')
print(index.path.name)
""",
            str(registry.root / project.project_id),
            project.project_id,
            str(source.session_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert built.stdout.strip() == "transcript-index.sqlite3"

    search_args = build_parser().parse_args(
        ["project", "index", "demo", "search", "historical", "backfill"]
    )
    assert run_project(search_args) == 0
    first_search = json.loads(capsys.readouterr().out)
    assert first_search, "first v5 search returned [] instead of linked v4 results"

    assert run_project(search_args) == 0
    second_search = json.loads(capsys.readouterr().out)
    assert second_search == first_search
    assert run_project(
        build_parser().parse_args(["project", "index", "demo", "status"])
    ) == 0
    assert json.loads(capsys.readouterr().out)["generation"] == 1

    assert run_project(
        build_parser().parse_args(["project", "index", "demo", "rebuild"])
    ) == 0
    capsys.readouterr()
    assert run_project(search_args) == 0
    rebuilt_search = json.loads(capsys.readouterr().out)
    assert rebuilt_search == first_search


@pytest.mark.asyncio
async def test_refreshes_coalesce_per_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = ProjectRegistry(tmp_path / "projects")
    source = _source(
        tmp_path / "sessions",
        "one",
        [
            _row(1, "user", "coalesce"),
            _row(2, "assistant", "answer", state="completed"),
        ],
    )
    (source.session_dir / "meta.json").write_text(
        json.dumps({"project_id": PROJECT_A}), encoding="utf-8"
    )
    batches: list[tuple[str, ...]] = []
    original = refresh_module._refresh_sources

    def observe(root: Path, project_id: str, sources) -> None:
        batches.append(tuple(item.session_id for item in sources))
        original(root, project_id, sources)

    # test-contract: seam inject the refresh worker boundary
    monkeypatch.setattr(refresh_module, "_refresh_sources", observe)
    await asyncio.wait_for(
        asyncio.gather(
            *(
                refresh_transcript_index(
                    registry,
                    PROJECT_A,
                    source.session_id,
                    source.session_dir,
                )
                for _ in range(10)
            )
        ),
        10,
    )

    assert batches == [(source.session_id,)]


@pytest.mark.asyncio
async def test_refresh_sqlite_failure_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def fail(*args, **kwargs) -> None:
        raise sqlite3.OperationalError("disk full")

    # test-contract: seam inject the refresh worker boundary
    monkeypatch.setattr(refresh_module, "_refresh_sources", fail)
    with pytest.raises(index_module.TranscriptIndexError, match="disk full"):
        await refresh_transcript_index(
            ProjectRegistry(tmp_path / "projects"), PROJECT_A, "one", tmp_path / "missing"
        )
    assert "could not refresh transcript index" in caplog.text


@pytest.mark.asyncio
async def test_transient_refresh_failure_retries_pending_session_on_next_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = ProjectRegistry(tmp_path / "projects")
    first = _source(
        tmp_path / "sessions",
        "one",
        [
            _row(1, "user", "first canary"),
            _row(2, "assistant", "answer", state="completed"),
        ],
    )
    second = _source(
        tmp_path / "sessions",
        "two",
        [
            _row(1, "user", "second canary"),
            _row(2, "assistant", "answer", state="completed"),
        ],
    )
    for source in (first, second):
        (source.session_dir / "meta.json").write_text(
            json.dumps({"project_id": PROJECT_A}), encoding="utf-8"
        )
    original = refresh_module._refresh_sources
    calls = 0

    def fail_once(*args) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise sqlite3.OperationalError("database is locked")
        original(*args)

    # test-contract: seam inject the refresh worker boundary
    monkeypatch.setattr(refresh_module, "_refresh_sources", fail_once)
    with pytest.raises(index_module.TranscriptIndexError, match="database is locked"):
        await refresh_transcript_index(
            registry, PROJECT_A, first.session_id, first.session_dir
        )
    await refresh_transcript_index(
        registry, PROJECT_A, second.session_id, second.session_dir
    )

    index = TranscriptIndex(tmp_path / "projects" / PROJECT_A, PROJECT_A)
    assert index.status().session_count == 2
    assert index.search("first canary")
    assert index.search("second canary")


@pytest.mark.asyncio
async def test_failed_durable_append_refresh_stays_degraded_until_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.core.store import ConversationStore
    from zeta.protocol.types import Message, MessageRole, TextContent

    projects_root = tmp_path / "projects"
    store = ConversationStore(tmp_path / "sessions", session_id="one")
    store.enable_persisted_append_tracking()
    metadata = {"project_id": PROJECT_A}
    (store.session_dir / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")

    def message(role: MessageRole, text: str) -> Message:
        key = "zeta.origin" if role == MessageRole.USER else "response_state"
        value = "user" if role == MessageRole.USER else "completed"
        return Message(role, [TextContent(text)], metadata={key: value})

    store.append_message(message(MessageRole.USER, "first indexed question"))
    store.append_message(message(MessageRole.ASSISTANT, "first indexed answer"))
    await refresh_transcript_index(
        ProjectRegistry(projects_root),
        PROJECT_A,
        store.session_id,
        store.session_dir,
        store.take_persisted_appends(),
    )

    await store.append_message_async(message(MessageRole.USER, "durableretrycanary"))
    await store.append_message_async(
        message(MessageRole.ASSISTANT, "durable retry answer")
    )
    receipts = store.take_persisted_appends()
    original_refresh = refresh_module._refresh_sources
    refresh_calls = 0

    def fail_once(*args) -> None:
        nonlocal refresh_calls
        refresh_calls += 1
        if refresh_calls == 1:
            raise sqlite3.OperationalError("disk full")
        original_refresh(*args)

    # test-contract: seam inject the refresh worker boundary
    monkeypatch.setattr(refresh_module, "_refresh_sources", fail_once)
    with pytest.raises(index_module.TranscriptIndexError, match="disk full"):
        await refresh_transcript_index(
            ProjectRegistry(projects_root),
            PROJECT_A,
            store.session_id,
            store.session_dir,
            receipts,
        )

    index = TranscriptIndex(projects_root / PROJECT_A, PROJECT_A)
    assert any(
        entry.data["message"]["content"][0]["text"] == "durableretrycanary"
        for entry in branch_entries(store)
        if entry.type == "message"
    )
    assert index.search("first indexed question")
    assert index.search("durableretrycanary") == ()
    failure = transcript_index_refresh_failure(
        projects_root, PROJECT_A, store.session_id
    )
    assert failure == (
        "disk full; pending changes will retry after the next transcript refresh"
    )

    await refresh_transcript_index(
        ProjectRegistry(projects_root),
        PROJECT_A,
        store.session_id,
        store.session_dir,
    )

    assert refresh_calls == 2
    assert len(index.search("durableretrycanary")) == 1
    assert (
        transcript_index_refresh_failure(projects_root, PROJECT_A, store.session_id)
        is None
    )


@pytest.mark.asyncio
async def test_failed_refresh_receipts_fall_back_to_bounded_full_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.core.store import PersistedAppend
    from zeta.core.store._log import MAX_PENDING_APPEND_RECEIPTS

    projects_root = tmp_path / "projects"
    registry = ProjectRegistry(projects_root)
    source = _source(
        tmp_path / "sessions",
        "one",
        [
            _row(1, "user", "boundedretrycanary"),
            _row(2, "assistant", "answer", state="completed"),
        ],
    )
    (source.session_dir / "meta.json").write_text(
        json.dumps({"project_id": PROJECT_A}), encoding="utf-8"
    )
    original_refresh = refresh_module._refresh_sources

    def fail(*args) -> None:
        raise sqlite3.OperationalError("disk full")

    # test-contract: seam inject the refresh worker boundary
    monkeypatch.setattr(refresh_module, "_refresh_sources", fail)
    for offset in range(MAX_PENDING_APPEND_RECEIPTS + 1):
        receipt = PersistedAppend(
            start_offset=offset,
            end_offset=offset + 1,
            digest=f"{offset:064x}",
            source_device=1,
            source_inode=1,
            before_mtime_ns=offset,
            before_ctime_ns=offset,
            after_mtime_ns=offset + 1,
            after_ctime_ns=offset + 1,
        )
        with pytest.raises(index_module.TranscriptIndexError, match="disk full"):
            await refresh_transcript_index(
                registry,
                PROJECT_A,
                source.session_id,
                source.session_dir,
                (receipt,),
            )

    states = refresh_module._REFRESH_STATES[asyncio.get_running_loop()]
    pending = states[(projects_root.resolve(), PROJECT_A)].pending[source.session_id]
    assert pending.append_receipts is None

    # test-contract: seam inject the refresh worker boundary
    monkeypatch.setattr(refresh_module, "_refresh_sources", original_refresh)
    await refresh_transcript_index(
        registry,
        PROJECT_A,
        source.session_id,
        source.session_dir,
        (receipt,),
    )

    index = TranscriptIndex(projects_root / PROJECT_A, PROJECT_A)
    assert len(index.search("boundedretrycanary")) == 1


@pytest.mark.asyncio
async def test_append_runs_off_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = ProjectRegistry(tmp_path / "projects")
    source = _source(
        tmp_path / "sessions",
        "one",
        [_row(1, "user", "ticker"), _row(2, "assistant", "answer", state="completed")],
    )
    (source.session_dir / "meta.json").write_text(
        json.dumps({"project_id": PROJECT_A}), encoding="utf-8"
    )
    event_loop_thread = threading.get_ident()
    refresh_threads: list[int] = []
    original_refresh = refresh_module._refresh_sources

    def observed_refresh(*args) -> None:
        refresh_threads.append(threading.get_ident())
        original_refresh(*args)

    # test-contract: seam inject the refresh worker boundary
    monkeypatch.setattr(refresh_module, "_refresh_sources", observed_refresh)
    await refresh_transcript_index(
        registry, PROJECT_A, source.session_id, source.session_dir
    )

    assert refresh_threads
    assert event_loop_thread not in refresh_threads


def test_mid_file_rewrite_after_digest_window_is_a_known_limit(tmp_path: Path) -> None:
    filler = "x" * (384 * 1024)
    original = [
        _row(1, "user", filler + " oldcanary " + filler),
        _row(2, "assistant", "first answer", state="completed"),
    ]
    source = _source(tmp_path / "sessions", "one", original)
    index = TranscriptIndex(tmp_path / "project", PROJECT_A)
    index.append(source)

    replacement = [
        _row(1, "user", filler + " newcanary " + filler),
        _row(2, "assistant", "first answer", state="completed"),
        _row(3, "user", "appended question"),
        _row(4, "assistant", "appended answer", state="completed"),
    ]
    _write_rows(source, replacement)
    index.append(source)

    # The bounded digest cannot detect edits outside its 64 KiB window.
    assert index.search("oldcanary")
    assert index.search("newcanary") == ()
    assert index.search("appended question")


@pytest.mark.asyncio
async def test_verified_appends_stay_incremental(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.core.store import ConversationStore
    from zeta.protocol.types import Message, MessageRole, TextContent

    root = tmp_path / "sessions"
    store = ConversationStore(root, session_id="one")
    store.enable_persisted_append_tracking()
    store.append_message(
        Message(
            MessageRole.USER,
            [TextContent("first question")],
            metadata={"zeta.origin": "user"},
        )
    )
    store.append_message(
        Message(
            MessageRole.ASSISTANT,
            [TextContent("first answer")],
            metadata={"response_state": "completed"},
        )
    )
    initial_receipts = store.take_persisted_appends()
    source = TranscriptSource(
        "one", store.session_dir, append_receipts=initial_receipts
    )
    index = TranscriptIndex(tmp_path / "project", PROJECT_A)
    index.append(source)

    await store.append_message_async(
        Message(
            MessageRole.USER,
            [TextContent("second question")],
            metadata={"zeta.origin": "user"},
        )
    )
    await store.append_message_async(
        Message(
            MessageRole.ASSISTANT,
            [TextContent("second answer")],
            metadata={"response_state": "completed"},
        )
    )
    receipts = store.take_persisted_appends()
    appended_bytes = sum(
        receipt.end_offset - receipt.start_offset for receipt in receipts
    )
    observed: list[object] = []
    original_read = index_module._read_transcript

    def observe(*args, **kwargs):
        result = original_read(*args, **kwargs)
        observed.append(result)
        return result

    monkeypatch.setattr(index_module, "_read_transcript", observe)
    index.append(TranscriptSource("one", store.session_dir, append_receipts=receipts))

    assert len(observed) == 1
    assert observed[0].full is False
    assert observed[0].bytes_read == appended_bytes
    assert index.search("second question")


def test_append_receipt_overflow_forces_correct_full_reread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.core.store import ConversationStore
    from zeta.core.store._log import MAX_PENDING_APPEND_RECEIPTS
    from zeta.protocol.types import Message, MessageRole, TextContent

    root = tmp_path / "sessions"
    store = ConversationStore(root, session_id="one")
    store.enable_persisted_append_tracking()
    store.append_message(
        Message(
            MessageRole.USER,
            [TextContent("first question")],
            metadata={"zeta.origin": "user"},
        )
    )
    store.append_message(
        Message(
            MessageRole.ASSISTANT,
            [TextContent("first answer")],
            metadata={"response_state": "completed"},
        )
    )
    index = TranscriptIndex(tmp_path / "project", PROJECT_A)
    index.append(
        TranscriptSource(
            "one", store.session_dir, append_receipts=store.take_persisted_appends()
        )
    )

    for turn in range(MAX_PENDING_APPEND_RECEIPTS // 2 + 1):
        store.append_message(
            Message(
                MessageRole.USER,
                [TextContent(f"overflow question {turn}")],
                metadata={"zeta.origin": "user"},
            )
        )
        store.append_message(
            Message(
                MessageRole.ASSISTANT,
                [TextContent(f"overflow answer {turn}")],
                metadata={"response_state": "completed"},
            )
        )

    assert len(store._persisted_appends) <= MAX_PENDING_APPEND_RECEIPTS
    receipts = store.take_persisted_appends()
    assert receipts is None
    observed: list[object] = []
    original_read = index_module._read_transcript

    def observe(*args, **kwargs):
        result = original_read(*args, **kwargs)
        observed.append(result)
        return result

    monkeypatch.setattr(index_module, "_read_transcript", observe)
    index.append(TranscriptSource("one", store.session_dir, append_receipts=receipts))

    assert len(observed) == 1
    assert observed[0].full is False
    assert observed[0].bytes_read < store.path.stat().st_size
    assert index.search(f"overflow question {MAX_PENDING_APPEND_RECEIPTS // 2}")


def test_unexplained_mtime_change_forces_reread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(
        tmp_path / "sessions",
        "one",
        [
            _row(1, "user", "first question"),
            _row(2, "assistant", "first answer", state="completed"),
        ],
    )
    index = TranscriptIndex(tmp_path / "project", PROJECT_A)
    index.append(source)
    with source.conversation_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_row(3, "user", "later question")) + "\n")
        handle.write(
            json.dumps(_row(4, "assistant", "later answer", state="completed")) + "\n"
        )

    observed: list[object] = []
    original_read = index_module._read_transcript

    def observe(*args, **kwargs):
        result = original_read(*args, **kwargs)
        observed.append(result)
        return result

    monkeypatch.setattr(index_module, "_read_transcript", observe)
    index.append(source)

    assert len(observed) == 1
    assert observed[0].full is False
    assert observed[0].bytes_read < source.conversation_path.stat().st_size


def test_other_process_append_without_receipt_rereads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.core.store import ConversationStore
    from zeta.protocol.types import Message, MessageRole, TextContent

    root = tmp_path / "sessions"
    writer = ConversationStore(root, session_id="one")
    writer.append_message(
        Message(
            MessageRole.USER,
            [TextContent("first question")],
            metadata={"zeta.origin": "user"},
        )
    )
    writer.append_message(
        Message(
            MessageRole.ASSISTANT,
            [TextContent("first answer")],
            metadata={"response_state": "completed"},
        )
    )
    source = TranscriptSource("one", writer.session_dir)
    index = TranscriptIndex(tmp_path / "project", PROJECT_A)
    index.append(source)

    other_process = ConversationStore(root, session_id="one")
    other_process.append_message(
        Message(
            MessageRole.USER,
            [TextContent("external question")],
            metadata={"zeta.origin": "user"},
        )
    )
    other_process.append_message(
        Message(
            MessageRole.ASSISTANT,
            [TextContent("external answer")],
            metadata={"response_state": "completed"},
        )
    )

    observed: list[object] = []
    original_read = index_module._read_transcript

    def observe(*args, **kwargs):
        result = original_read(*args, **kwargs)
        observed.append(result)
        return result

    monkeypatch.setattr(index_module, "_read_transcript", observe)
    index.append(source)

    assert len(observed) == 1
    assert observed[0].full is False
    assert observed[0].bytes_read < source.conversation_path.stat().st_size
    assert index.search("external question")
