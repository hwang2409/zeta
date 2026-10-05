"""Tests for the read-only ``zeta session stats --compaction`` report."""

from __future__ import annotations

import json
import os
import stat
import threading
import time
import tracemalloc
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pytest

from zeta.cli.compaction_stats import compaction_report, render_report, scan_log
from zeta.cli.main import main
from zeta.compaction import fallback_summary
from zeta.context_eviction import recall_history
from zeta.core.context import ContextAssembler
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    Message,
    MessageRole,
    TextContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
)
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog

NOW = datetime(2026, 10, 5, 12, tzinfo=UTC)


def text(role: MessageRole, value: str) -> Message:
    return Message(role, [TextContent(value)])


def tool_pair(name: str, call_id: str, output: str, **arguments: object) -> list[Message]:
    return [
        Message(
            MessageRole.ASSISTANT,
            [ToolUseContent(ToolCall(call_id, name, arguments or {"path": "a.py"}))],
        ),
        Message(MessageRole.TOOL_RESULT, tool_result=ToolResult(call_id, output)),
    ]


def write_meta(
    home: Path,
    session_id: str,
    *,
    mode: str | None,
    budget: int,
    updated_at: datetime = NOW,
) -> None:
    meta = {"session_id": session_id, "compaction_budget": budget}
    meta["updated_at"] = updated_at.isoformat()
    if mode is not None:
        meta["compaction"] = mode
    (home / "sessions" / session_id / "meta.json").write_text(json.dumps(meta))


async def evicted_store(root: Path, session_id: str) -> ConversationStore:
    """Build a log with one real eviction marker from the context assembler."""

    store = ConversationStore(root, session_id=session_id)
    store.append_message(text(MessageRole.USER, "old request"))
    for message in tool_pair("read", f"{session_id}-old", "old output\n" * 3000):
        store.append_message(message)
    store.append_message(text(MessageRole.USER, "latest request"))
    await ContextAssembler(
        store, token_budget=1000, retained_tail=8, compaction="evict"
    ).assemble_context()
    return store


def append_recall(store: ConversationStore, call_id: str, query: str) -> None:
    result = recall_history(store, query=query)
    for message in tool_pair("recall_history", call_id, result, query=query):
        store.append_message(message)


async def run_over_budget(store: ConversationStore) -> None:
    loop = AgentLoop(
        FakeBackend([ScriptedTurn([TextContent("unused")])]),
        store,
        token_budget=5,
        compaction="summary",
        skill_catalog=SkillCatalog.empty(),
        max_turns=1,
    )
    _ = [event async for event in loop.run_turn("over budget")]


async def build_home(home: Path) -> None:
    sessions = home / "sessions"
    # Evict session: real eviction, recall hit and miss, an empty-summary
    # fallback marker, and nested child agents with their own eviction.
    store = await evicted_store(sessions, "evict1")
    append_recall(store, "recall-hit", "old output")
    append_recall(store, "recall-miss", "zzzqqqx")
    store.append_message(
        Message(
            MessageRole.ASSISTANT,
            [ToolUseContent(ToolCall("recall-open", "recall_history", {"query": "x"}))],
        )
    )
    last = store.replay()[-1].seq
    store.append_compaction_marker(fallback_summary("[]", max_chars=500), 1, last)
    write_meta(home, "evict1", mode="evict", budget=1000)
    await evicted_store(sessions / "evict1" / "agents", "1")
    grandchild = ConversationStore(sessions / "evict1" / "agents" / "1" / "agents", session_id="1")
    grandchild.append_message(text(MessageRole.USER, "quiet child"))

    # Evict session at the 1M default that never compacts.
    idle = ConversationStore(sessions, session_id="evict2")
    idle.append_message(text(MessageRole.USER, "short"))
    write_meta(home, "evict2", mode="evict", budget=1_000_000)

    # Legacy summary session with an incremental compaction that replaces
    # the first marker and a real compaction failure; its child agent records
    # a real BudgetExceeded turn error.
    summary = ConversationStore(sessions, session_id="summary1")
    for index in range(6):
        summary.append_message(text(MessageRole.USER, f"message {index} " + "x" * 400))
    first = summary.append_compaction_marker("first summary", 1, 3)
    summary.append_compaction_marker("second summary", 1, 6, replaces=[first.id])
    await run_over_budget(summary)
    await run_over_budget(ConversationStore(sessions / "summary1" / "agents", session_id="1"))
    write_meta(home, "summary1", mode=None, budget=200_000)

    # Old session outside the default window.
    old = ConversationStore(sessions, session_id="old1")
    old.append_message(text(MessageRole.USER, "old"))
    old.append_compaction_marker("old summary", 1, 1)
    write_meta(home, "old1", mode="summary", budget=200_000, updated_at=NOW - timedelta(days=30))


def snapshot(root: Path) -> dict[str, tuple[int, int, bytes | None]]:
    result = {}
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        digest = sha256(path.read_bytes()).digest() if path.is_file() else None
        result[str(path.relative_to(root))] = (info.st_mtime_ns, info.st_size, digest)
    return result


@pytest.mark.asyncio
async def test_report_counts_both_modes_children_and_recall(tmp_path: Path) -> None:
    await build_home(tmp_path)

    report = compaction_report(tmp_path, now=NOW)

    assert report["sessions_scanned"] == 3
    assert report["logs_scanned"] == 6
    evict = report["modes"]["evict"]
    assert evict["sessions"] == 2
    assert evict["sessions_compacted"] == 1
    assert evict["child_logs"] == 2
    assert evict["child_logs_compacted"] == 1
    assert evict["evictions"] == 2
    assert evict["evictions_without_telemetry"] == 0
    assert evict["items_evicted"] >= 2
    assert evict["evict_tokens_before"] > evict["evict_tokens_after"] > 0
    assert 0 < evict["evict_saved_pct"] < 100
    assert evict["summary_compactions"] == 1
    assert evict["summary_fallbacks_from_evict"] == 1
    assert evict["empty_summary_fallbacks"] == 1
    assert evict["recall_calls"] == 3
    assert evict["recall_with_content"] == 1
    assert evict["recall_no_match"] == 1
    assert evict["recall_unanswered"] == 1
    assert evict["budget_exceeded"] == 0

    summary = report["modes"]["summary"]
    assert summary["sessions"] == 1
    assert summary["sessions_compacted"] == 1
    assert summary["summary_compactions"] == 2
    assert summary["summary_fallbacks_from_evict"] is None
    assert summary["empty_summary_fallbacks"] == 0
    assert summary["child_logs"] == 1
    assert summary["budget_exceeded"] == 1
    assert summary["compaction_errors"] == 1
    assert summary["evictions"] == 0

    budgets = {(item["mode"], item["budget"]): item for item in report["budgets"]}
    assert budgets[("evict", 1_000_000)]["sessions"] == 1
    assert budgets[("evict", 1_000_000)]["sessions_compacted"] == 0
    assert budgets[("evict", 1_000_000)]["sessions_history_over_budget"] == 0
    assert budgets[("evict", 1000)]["sessions_history_over_budget"] == 1
    top = [item["session_id"] for item in report["top_sessions"]]
    assert top == ["evict1", "summary1"]
    assert report["integrity"] == {
        "torn_final_lines": 0,
        "unparseable_lines": 0,
        "unreadable_logs": 0,
    }


@pytest.mark.asyncio
async def test_summary_estimate_does_not_double_count_replaced_markers(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="s")
    for index in range(6):
        store.append_message(text(MessageRole.USER, f"message {index} " + "x" * 400))
    first = store.append_compaction_marker("first", 1, 3)
    store.append_compaction_marker("second", 1, 6, replaces=[first.id])
    log = tmp_path / "s" / "conversation.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    sizes = {
        row["seq"]: -(-len(json.dumps(row, separators=(",", ":"), sort_keys=True)) // 4)
        for row in rows
        if row["type"] == "message"
    }

    tally = scan_log(log)

    # First marker covers seqs 1-3; the second covers the first summary plus
    # seqs 4-6 only.
    expected_first = sizes[1] + sizes[2] + sizes[3]
    expected_second = -(-len("first") // 4) + sizes[4] + sizes[5] + sizes[6]
    assert tally.summary_est_tokens_before == expected_first + expected_second
    assert tally.summary_est_tokens_after == 4


def test_torn_final_line_and_corrupt_rows_are_skipped(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions", session_id="torn")
    store.append_message(text(MessageRole.USER, "hello"))
    store.append_compaction_marker("summary", 1, 1)
    log = tmp_path / "sessions" / "torn" / "conversation.jsonl"
    with open(log, "ab") as handle:
        handle.write(b"not json\n")
        handle.write(b'{"data":{"summary":"half')

    tally = scan_log(log)

    assert tally.summary_compactions == 1
    assert tally.torn_final_lines == 1
    assert tally.unparseable_lines == 1
    report = compaction_report(tmp_path, since="all")
    assert report["integrity"]["torn_final_lines"] == 1
    assert "1 torn final lines, 1 unparseable lines" in render_report(report)


def test_empty_home_header_only_and_missing_meta(tmp_path: Path) -> None:
    assert compaction_report(tmp_path, since="all")["sessions_scanned"] == 0
    assert "no sessions in range" in render_report(compaction_report(tmp_path))
    ConversationStore(tmp_path / "sessions", session_id="bare")
    empty = tmp_path / "sessions" / "empty"
    empty.mkdir()
    (empty / "conversation.jsonl").write_bytes(b"")

    report = compaction_report(tmp_path, since="all")

    assert report["sessions_scanned"] == 2
    assert report["modes"]["unknown"]["sessions"] == 2
    assert report["top_sessions"] == []
    assert "(no compaction activity)" in render_report(report)


@pytest.mark.asyncio
async def test_since_and_session_selection(tmp_path: Path) -> None:
    await build_home(tmp_path)

    assert compaction_report(tmp_path, since="all")["sessions_scanned"] == 4
    assert compaction_report(tmp_path, since="2026-09-01", now=NOW)["sessions_scanned"] == 4
    assert compaction_report(tmp_path, since="12h", now=NOW)["sessions_scanned"] == 3
    only_old = compaction_report(tmp_path, session="old", now=NOW)
    assert only_old["sessions_scanned"] == 1
    assert only_old["since"] is None
    with pytest.raises(ValueError, match="ambiguous session prefix"):
        compaction_report(tmp_path, session="evict")
    with pytest.raises(ValueError, match="no session matches"):
        compaction_report(tmp_path, session="nope")
    with pytest.raises(ValueError, match="invalid --since"):
        compaction_report(tmp_path, since="yesterday")
    with pytest.raises(ValueError, match="--top must be at least 1"):
        compaction_report(tmp_path, top=0)


@pytest.mark.asyncio
async def test_cli_is_read_only_and_prints_text_and_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    await build_home(tmp_path)
    monkeypatch.setenv("ZETA_HOME", str(tmp_path))
    before = snapshot(tmp_path)
    # Remove write permission everywhere: any lock, repair, or metadata write
    # would fail instead of passing silently.
    for path in tmp_path.rglob("*"):
        path.chmod(stat.S_IRUSR | (stat.S_IXUSR if path.is_dir() else 0))
    try:
        assert main(["session", "stats", "--compaction", "--since", "all"]) == 0
        text_output = capsys.readouterr().out
        assert main(["session", "stats", "--compaction", "--json", "--session", "evict1"]) == 0
        json_output = json.loads(capsys.readouterr().out)
        assert main(["session", "stats", "--compaction", "--since", "bad"]) == 1
        assert "invalid --since" in capsys.readouterr().err
    finally:
        for path in tmp_path.rglob("*"):
            path.chmod(stat.S_IRWXU if path.is_dir() else stat.S_IRUSR | stat.S_IWUSR)

    assert snapshot(tmp_path) == before
    assert "compaction report: 4 sessions" in text_output
    assert "evictions" in text_output and "BudgetExceeded" in text_output
    assert "1,000,000" in text_output
    assert "top sessions by compaction activity" in text_output
    assert json_output["modes"]["evict"]["evictions"] == 2
    assert json_output["top_sessions"][0]["session_id"] == "evict1"


def test_concurrent_writer_never_produces_mid_file_corruption(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="live")
    store.append_message(text(MessageRole.USER, "start"))
    log = tmp_path / "live" / "conversation.jsonl"
    row = (
        '{"data":{"message":{"content":[{"text":"' + "y" * 2000 + '","type":"text"}],'
        '"role":"user"}},"id":"%d","lane":"main","parent_id":null,"seq":%d,"type":"message"}\n'
    )
    stop = threading.Event()

    def write() -> None:
        seq = 100
        with open(log, "ab", buffering=0) as handle:
            while not stop.is_set():
                encoded = (row % (seq, seq)).encode()
                handle.write(encoded[:700])
                time.sleep(0.0005)
                handle.write(encoded[700:])
                seq += 1

    writer = threading.Thread(target=write)
    writer.start()
    try:
        tallies = [scan_log(log) for _ in range(30)]
    finally:
        stop.set()
        writer.join()

    assert all(tally.unparseable_lines == 0 for tally in tallies)
    assert all(tally.torn_final_lines <= 1 for tally in tallies)
    assert not (tmp_path / "live" / "conversation.jsonl.lock").exists()


def test_large_log_is_fast_and_memory_bounded(tmp_path: Path) -> None:
    log = tmp_path / "conversation.jsonl"
    filler = "z" * 400
    with open(log, "w") as handle:
        handle.write('{"data":{"schema":"zeta.conversation.v1"},"type":"header"}\n')
        for seq in range(1, 100_001):
            handle.write(
                f'{{"data":{{"message":{{"content":[{{"text":"{filler}","type":"text"}}],'
                f'"role":"user"}}}},"id":"m{seq}","lane":"main","parent_id":null,'
                f'"seq":{seq},"type":"message"}}\n'
            )
            if seq % 10_000 == 0:
                handle.write(
                    json.dumps(
                        {
                            "data": {
                                "replaces": [],
                                "source_seq_end": seq,
                                "source_seq_start": seq - 9_999,
                                "summary": "chunk summary",
                            },
                            "id": f"c{seq}",
                            "type": "compaction",
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                )
    size = os.path.getsize(log)
    assert size > 45_000_000

    started = time.perf_counter()
    tracemalloc.start()
    try:
        tally = scan_log(log)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    elapsed = time.perf_counter() - started

    assert tally.summary_compactions == 10
    assert tally.summary_est_tokens_before > 10 * 10_000 * 100
    assert peak < 8_000_000, peak
    assert elapsed < 20, elapsed
