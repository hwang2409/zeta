from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from pathlib import Path

import pytest

import zeta.transcript_search.index as index_module
from zeta.cli.main import build_parser
from zeta.cli.project import run as run_project
from zeta.project_registry import ProjectRegistry
from zeta.transcript_search.index import (
    SANITIZER_VERSION,
    TranscriptIndex,
    TranscriptIndexUnavailable,
    TranscriptSource,
    refresh_transcript_index,
)

PROJECT_A = "p_" + "a" * 32
PROJECT_B = "p_" + "b" * 32


def _row(seq: int, role: str, text: str, *, state: str | None = None) -> dict:
    metadata = (
        {"zeta.origin": "user", "origin": "human"} if role == "user" else {}
    )
    if state is not None:
        metadata["response_state"] = state
    return {
        "seq": seq,
        "id": f"r-{seq}",
        "parent_id": f"r-{seq - 1}" if seq > 1 else None,
        "lane": "main",
        "type": "message",
        "data": {"message": {"role": role, "content": [{"type": "text", "text": text}], "metadata": metadata}},
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


def test_rebuild_equals_incremental_append(tmp_path: Path) -> None:
    sources = [
        _source(tmp_path / "sessions", "one", [_row(1, "user", "alpha question"), _row(2, "assistant", "beta answer", state="completed")]),
        _source(tmp_path / "sessions", "two", [_row(1, "user", "gamma question"), _row(2, "assistant", "delta answer", state="completed")]),
    ]
    rebuilt = TranscriptIndex(tmp_path / "rebuilt", PROJECT_A)
    rebuilt.rebuild(sources)
    incremental = TranscriptIndex(tmp_path / "incremental", PROJECT_A)
    for source in sources:
        incremental.append(source)

    for query in ("alpha beta", "delta", "question"):
        assert _result_ids(rebuilt, query) == _result_ids(incremental, query)
    assert rebuilt.status().unit_count == incremental.status().unit_count == 2


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
        stored = "\n".join(row[0] for row in connection.execute("SELECT text FROM units"))
    indexed = "\n".join(hit.unit.text for hit in index.search("safe answer"))
    assert "source_rows" not in tables
    assert "TOP_SECRET_KEY_MATERIAL" not in stored
    assert "END PRIVATE KEY" not in stored
    assert "PRIVATE_REASONING_MARKER" not in stored
    assert "TOP_SECRET_KEY_MATERIAL" not in indexed
    assert "PRIVATE_REASONING_MARKER" not in indexed


def test_project_scope_rejects_wrong_project_canary(tmp_path: Path) -> None:
    source = _source(tmp_path / "sessions", "one", [_row(1, "user", "private canary"), _row(2, "assistant", "answer", state="completed")])
    index_a = TranscriptIndex(tmp_path / "a", PROJECT_A)
    index_b = TranscriptIndex(tmp_path / "b", PROJECT_B)
    index_a.append(source)
    assert index_a.search("private canary")
    assert index_b.search("private canary") == ()

    unit = index_a.search("private canary")[0].unit
    with pytest.raises(ValueError, match="project"):
        index_b.append_units("foreign", (unit,), cursor=2)


def test_deletion_and_reassignment(tmp_path: Path) -> None:
    source = _source(tmp_path / "sessions", "one", [_row(1, "user", "moving canary"), _row(2, "assistant", "answer", state="completed")])
    old = TranscriptIndex(tmp_path / "old", PROJECT_A)
    new = TranscriptIndex(tmp_path / "new", PROJECT_B)
    old.append(source)
    old.delete_session(source.session_id)
    new.append(source)
    assert old.search("moving canary") == ()
    assert new.search("moving canary")


def test_cursor_lag_resumes_without_duplicates(tmp_path: Path) -> None:
    source = _source(tmp_path / "sessions", "one", [_row(1, "user", "crash recovery"), _row(2, "assistant", "idempotent", state="completed")])
    index = TranscriptIndex(tmp_path / "project", PROJECT_A)
    index.append(source)
    with sqlite3.connect(index.path) as connection:
        connection.execute("UPDATE session_cursors SET last_seq = 0 WHERE session_id = ?", (source.session_id,))
        connection.commit()
    index.append(source)
    status = index.status()
    assert status.unit_count == 1
    assert status.cursors[source.session_id] == 2


def test_relaxed_search_uses_distinctive_partial_terms(tmp_path: Path) -> None:
    source = _source(tmp_path / "sessions", "one", [_row(1, "user", "common common platypus"), _row(2, "assistant", "answer", state="completed")])
    other = _source(tmp_path / "sessions", "two", [_row(1, "user", "common ordinary"), _row(2, "assistant", "answer", state="completed")])
    index = TranscriptIndex(tmp_path / "project", PROJECT_A)
    index.rebuild((source, other))
    hits = index.search("missing common platypus")
    assert hits
    assert "platypus" in hits[0].unit.text
    assert hits[0].match == "partial"


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

    def observe(path: Path, cursor, receipts=None):
        result = original_read(path, cursor, receipts)
        reads.append(result)
        return result

    monkeypatch.setattr(index_module, "_read_transcript", observe)
    with source.conversation_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_row(3, "user", "second question")) + "\n")
        handle.write(
            json.dumps(_row(4, "assistant", "second answer", state="completed"))
            + "\n"
        )
    index.append(source)

    rebuilt = TranscriptIndex(tmp_path / "rebuilt", PROJECT_A)
    rebuilt.rebuild((source,))
    assert len(reads) == 2
    assert reads[0].full is True
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
    paragraph = " ".join(
        f"representative-token-{index % 80}" for index in range(240)
    )
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


def test_out_of_order_unit_commit_never_regresses(tmp_path: Path) -> None:
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
    rows = [
        json.loads(line)
        for line in source.conversation_path.read_text(encoding="utf-8").splitlines()
    ]
    units = index_module.render_transcript_units(PROJECT_A, "one", rows)
    index.append_units("one", units, cursor=4)
    index.append_units("one", units[:1], cursor=2)

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
        handle.write(json.dumps(_row(4, "assistant", "new answer", state="completed")) + "\n")
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


def test_version_mismatch_refuses_old_rows_and_rebuild_restores(
    tmp_path: Path
) -> None:
    source = _source(
        tmp_path / "sessions",
        "one",
        [_row(1, "user", "version canary"), _row(2, "assistant", "answer", state="completed")],
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
        [_row(1, "user", "repair canary"), _row(2, "assistant", "answer", state="completed")],
    )
    project_dir = tmp_path / "project"
    index = TranscriptIndex(project_dir, PROJECT_A)
    with sqlite3.connect(index.path) as connection:
        connection.execute("UPDATE metadata SET value = '0' WHERE key = 'schema_version'")
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
        [_row(1, "user", "published canary"), _row(2, "assistant", "answer", state="completed")],
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
    source = _source(
        home / "sessions",
        "one",
        [_row(1, "user", "operator canary"), _row(2, "assistant", "answer", state="completed")],
    )
    (source.session_dir / "meta.json").write_text(
        json.dumps({"project_id": project.project_id}), encoding="utf-8"
    )
    registry.record_session(
        project.project_id,
        session_id=source.session_id,
        transcript_path=str(source.session_dir),
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


@pytest.mark.asyncio
async def test_refreshes_coalesce_per_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(
        tmp_path / "sessions",
        "one",
        [_row(1, "user", "coalesce"), _row(2, "assistant", "answer", state="completed")],
    )
    (source.session_dir / "meta.json").write_text(
        json.dumps({"project_id": PROJECT_A}), encoding="utf-8"
    )
    batches: list[tuple[str, ...]] = []
    original = index_module._refresh_sources

    def observe(root: Path, project_id: str, sources) -> None:
        batches.append(tuple(item.session_id for item in sources))
        original(root, project_id, sources)

    monkeypatch.setattr(index_module, "_refresh_sources", observe)
    await asyncio.gather(
        *(
            refresh_transcript_index(
                tmp_path / "projects",
                PROJECT_A,
                source.session_id,
                source.session_dir,
            )
            for _ in range(10)
        )
    )

    assert batches == [(source.session_id,)]


@pytest.mark.asyncio
async def test_refresh_sqlite_failure_is_best_effort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def fail(*args, **kwargs) -> None:
        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(index_module, "_refresh_sources", fail)
    await refresh_transcript_index(
        tmp_path / "projects", PROJECT_A, "one", tmp_path / "missing"
    )
    assert "could not refresh transcript index" in caplog.text


@pytest.mark.asyncio
async def test_append_runs_off_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path / "sessions", "one", [_row(1, "user", "ticker"), _row(2, "assistant", "answer", state="completed")])
    (source.session_dir / "meta.json").write_text(
        json.dumps({"project_id": PROJECT_A}), encoding="utf-8"
    )
    original_init = TranscriptIndex.__init__

    def slow_init(self, *args, **kwargs):
        import time

        time.sleep(0.06)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(TranscriptIndex, "__init__", slow_init)
    gaps: list[float] = []

    async def ticker() -> None:
        previous = asyncio.get_running_loop().time()
        for _ in range(15):
            await asyncio.sleep(0.002)
            now = asyncio.get_running_loop().time()
            gaps.append(now - previous)
            previous = now

    ticker_task = asyncio.create_task(ticker())
    await asyncio.sleep(0)
    await asyncio.gather(
        refresh_transcript_index(
            tmp_path / "projects", PROJECT_A, source.session_id, source.session_dir
        ),
        ticker_task,
    )
    assert max(gaps) < 0.01


def test_mid_file_rewrite_outside_any_window_forces_reread(tmp_path: Path) -> None:
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

    assert index.search("oldcanary") == ()
    assert index.search("newcanary")
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
        Message(MessageRole.USER, [TextContent("first question")], metadata={"zeta.origin": "user"})
    )
    store.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("first answer")], metadata={"response_state": "completed"})
    )
    initial_receipts = store.take_persisted_appends()
    source = TranscriptSource("one", store.session_dir, append_receipts=initial_receipts)
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
    appended_bytes = sum(receipt.end_offset - receipt.start_offset for receipt in receipts)
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
    from zeta.core.store._store import MAX_PENDING_APPEND_RECEIPTS
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
    assert observed[0].full is True
    assert observed[0].bytes_read == store.path.stat().st_size
    assert index.search(f"overflow question {MAX_PENDING_APPEND_RECEIPTS // 2}")


def test_unexplained_mtime_change_forces_reread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(
        tmp_path / "sessions",
        "one",
        [_row(1, "user", "first question"), _row(2, "assistant", "first answer", state="completed")],
    )
    index = TranscriptIndex(tmp_path / "project", PROJECT_A)
    index.append(source)
    with source.conversation_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_row(3, "user", "later question")) + "\n")
        handle.write(json.dumps(_row(4, "assistant", "later answer", state="completed")) + "\n")

    observed: list[object] = []
    original_read = index_module._read_transcript

    def observe(*args, **kwargs):
        result = original_read(*args, **kwargs)
        observed.append(result)
        return result

    monkeypatch.setattr(index_module, "_read_transcript", observe)
    index.append(source)

    assert len(observed) == 1
    assert observed[0].full is True
    assert observed[0].bytes_read == source.conversation_path.stat().st_size


def test_other_process_append_without_receipt_rereads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.core.store import ConversationStore
    from zeta.protocol.types import Message, MessageRole, TextContent

    root = tmp_path / "sessions"
    writer = ConversationStore(root, session_id="one")
    writer.append_message(
        Message(MessageRole.USER, [TextContent("first question")], metadata={"zeta.origin": "user"})
    )
    writer.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("first answer")], metadata={"response_state": "completed"})
    )
    source = TranscriptSource("one", writer.session_dir)
    index = TranscriptIndex(tmp_path / "project", PROJECT_A)
    index.append(source)

    other_process = ConversationStore(root, session_id="one")
    other_process.append_message(
        Message(MessageRole.USER, [TextContent("external question")], metadata={"zeta.origin": "user"})
    )
    other_process.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("external answer")], metadata={"response_state": "completed"})
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
    assert observed[0].full is True
    assert observed[0].bytes_read == source.conversation_path.stat().st_size
    assert index.search("external question")
