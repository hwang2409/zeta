from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from pathlib import Path

import pytest

from zeta.cli.main import build_parser
from zeta.cli.project import run as run_project
from zeta.project_registry import ProjectRegistry
from zeta.transcript_search.background import append_transcript_off_loop
from zeta.transcript_search.index import (
    TranscriptIndex,
    TranscriptSource,
    rebuild_project_index,
)

PROJECT_A = "p_" + "a" * 32
PROJECT_B = "p_" + "b" * 32


def _row(seq: int, role: str, text: str, *, state: str | None = None) -> dict:
    metadata = {"origin": "human"} if role == "user" else {}
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


def test_rebuild_equals_incremental_append(tmp_path: Path) -> None:
    sources = [
        _source(tmp_path / "sessions", "one", [_row(1, "user", "alpha question"), _row(2, "assistant", "beta answer", state="completed")]),
        _source(tmp_path / "sessions", "two", [_row(1, "user", "gamma question"), _row(2, "assistant", "delta answer", state="completed")]),
    ]
    rebuilt = TranscriptIndex(tmp_path / "rebuilt", PROJECT_A)
    rebuild_project_index(rebuilt, sources)
    incremental = TranscriptIndex(tmp_path / "incremental", PROJECT_A)
    for source in sources:
        incremental.append(source)

    for query in ("alpha beta", "delta", "question"):
        assert _result_ids(rebuilt, query) == _result_ids(incremental, query)
    assert rebuilt.status().unit_count == incremental.status().unit_count == 2


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
async def test_append_runs_off_event_loop(tmp_path: Path) -> None:
    source = _source(tmp_path / "sessions", "one", [_row(1, "user", "ticker"), _row(2, "assistant", "answer", state="completed")])
    index = TranscriptIndex(tmp_path / "project", PROJECT_A)
    original = index.append

    def slow_append(item: TranscriptSource) -> object:
        time.sleep(0.06)
        return original(item)

    index.append = slow_append  # type: ignore[method-assign]
    gaps: list[float] = []

    async def ticker() -> None:
        previous = asyncio.get_running_loop().time()
        for _ in range(15):
            await asyncio.sleep(0.002)
            now = asyncio.get_running_loop().time()
            gaps.append(now - previous)
            previous = now

    await asyncio.gather(append_transcript_off_loop(index, source), ticker())
    assert max(gaps) < 0.01
