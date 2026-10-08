"""Switching an existing session's compaction mode (TUI, serve, headless)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from zeta.cli.main import build_parser
from zeta.core.context import CompactionPolicy, ContextAssembler
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.session import SessionError, SessionManager
from zeta.core.slash import create_slash_registry
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    Message,
    MessageOrigin,
    MessageRole,
    TextContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
    with_message_origin,
)
from zeta.runtime.compaction_mode import apply_compaction
from zeta.runtime.headless import run_headless
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry
from zeta.tui.app import create_app


def _text(role: MessageRole, value: str) -> Message:
    return Message(role, [TextContent(value)])


def _tool_pair(call_id: str, output: str) -> tuple[Message, Message]:
    call = Message(
        MessageRole.ASSISTANT,
        [ToolUseContent(ToolCall(call_id, "read", {"path": "RULES.md"}))],
    )
    result = Message(
        MessageRole.TOOL_RESULT, tool_result=ToolResult(call_id, output)
    )
    return call, result


def _rendered(messages) -> str:  # type: ignore[no-untyped-def]
    values: list[str] = []
    for message in messages:
        values.extend(
            block.text for block in message.content if isinstance(block, TextContent)
        )
        if message.tool_result is not None:
            values.append(message.tool_result.content)
    return "\n".join(values)


def _meta(home: Path, session_id: str) -> dict:
    return json.loads((home / "sessions" / session_id / "meta.json").read_text())


async def _slash(app, text: str) -> str:  # type: ignore[no-untyped-def]
    output = await create_slash_registry(
        skill_catalog=SkillCatalog.empty()
    ).dispatch_async(app, text)
    assert isinstance(output, str)
    return output


def _tool_names(schemas) -> set[str]:  # type: ignore[no-untyped-def]
    return {schema["name"] for schema in schemas}


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(path))
    monkeypatch.chdir(tmp_path)
    return path


async def test_tui_switch_persists_across_reopen(home: Path) -> None:
    app = create_app(build_parser().parse_args(["--provider", "fake"]))
    session_id = app.loop.store.session_id
    assert app.loop.context_assembler.compaction == "evict"

    output = await _slash(app, "/compaction summary")

    assert "evict -> summary" in output
    assert app.loop.context_assembler.compaction == "summary"
    assert "recall_history" not in _tool_names(app.loop._active_tool_schemas())
    assert _meta(home, session_id)["compaction"] == "summary"
    assert _meta(home, session_id)["compaction_pinned"] is True
    assert app.loop.session_metadata.compaction == "summary"

    # Switching back and forth in one process keeps metadata coherent.
    assert "summary -> evict" in await _slash(app, "/compaction evict")
    assert "evict -> summary" in await _slash(app, "/compaction summary")
    assert _meta(home, session_id)["compaction"] == "summary"
    await app.close()

    reopened = create_app(build_parser().parse_args(["--resume", session_id]))
    assert reopened.loop.context_assembler.compaction == "summary"
    assert "recall_history" not in reopened.loop.tool_registry.registered_names

    assert "summary -> evict" in await _slash(reopened, "/compaction evict")
    await reopened.close()

    again = create_app(build_parser().parse_args(["--resume", session_id]))
    assert again.loop.context_assembler.compaction == "evict"
    assert "recall_history" in _tool_names(again.loop._active_tool_schemas())
    await again.close()


async def test_tui_compaction_reports_mode_budget_and_last_stats(home: Path) -> None:
    app = create_app(build_parser().parse_args(["--provider", "fake"]))
    try:
        shown = await _slash(app, "/compaction")
        assert "compaction: evict" in shown
        budget = app.loop.context_assembler.token_budget
        assert f"budget: {budget:,} tokens" in shown
        assert "recall_history: advertised" in shown
        assert "last compaction: none" in shown

        store = app.loop.store
        first = store.append_message(with_message_origin(_text(MessageRole.USER, "old " * 400), MessageOrigin.USER))
        store.append_compaction_marker("short summary", first.seq, first.seq)
        shown = await _slash(app, "/compaction")
        assert "last compaction: summary" in shown
        assert "1 entries folded" in shown

        assert "unchanged" in await _slash(app, "/compaction evict")
        assert "usage: /compaction" in await _slash(app, "/compaction bogus")
        assert app.loop.context_assembler.compaction == "evict"
    finally:
        await app.close()


async def test_switch_toggles_recall_history_for_next_request(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn([TextContent("one")]),
            ScriptedTurn([TextContent("two")]),
            ScriptedTurn([TextContent("three")]),
        ]
    )
    loop = AgentLoop(
        backend,
        ConversationStore(tmp_path / "sessions"),
        skill_catalog=SkillCatalog.empty(),
        max_turns=1,
    )
    assert loop.context_assembler.compaction == "summary"

    async for _ in loop.run_turn("first", origin=MessageOrigin.USER):
        pass
    apply_compaction(loop, "evict")
    async for _ in loop.run_turn("second", origin=MessageOrigin.USER):
        pass
    apply_compaction(loop, "summary")
    async for _ in loop.run_turn("third", origin=MessageOrigin.USER):
        pass

    names = [_tool_names(tools) for _messages, tools in backend.calls]
    assert "recall_history" not in names[0]
    assert "recall_history" in names[1]
    assert "recall_history" not in names[2]
    assert names[0] == names[2]
    assert loop.tool_registry.compaction == "summary"
    await loop.close()


class _StaticSummaryPolicy(CompactionPolicy):
    async def summarize_chunked(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        return "static summary"


async def test_mixed_markers_replay_identically_across_switches(
    tmp_path: Path,
) -> None:
    sessions = tmp_path / "sessions"
    store = ConversationStore(sessions, session_id="mixed")
    first = store.append_message(with_message_origin(_text(MessageRole.USER, "early requirement"), MessageOrigin.USER))
    store.append_compaction_marker("summary of the early work", first.seq, first.seq)
    call, result = _tool_pair("read-1", "large output\n" * 1500)
    store.append_message(call)
    store.append_message(result)
    store.append_message(_text(MessageRole.ASSISTANT, "old reasoning " * 20))
    store.append_message(with_message_origin(_text(MessageRole.USER, "latest request verbatim"), MessageOrigin.USER))
    assembler = ContextAssembler(
        store,
        token_budget=900,
        retained_tail=1,
        compaction="evict",
        compaction_policy=_StaticSummaryPolicy(),
    )

    evicted = await assembler.assemble_context()
    kinds = [
        entry.data.get("kind", "summary")
        for entry in store.replay()
        if entry.type == "compaction"
    ]
    assert kinds == ["summary", "evict"]
    marker_count = store.compaction_marker_count()

    views = []
    for mode in ("summary", "evict", "summary", "evict"):
        assembler.compaction = mode
        views.append(await assembler.assemble_context())
        reopened = ContextAssembler(
            ConversationStore(sessions, session_id="mixed"),
            token_budget=900,
            retained_tail=1,
            compaction=mode,
        )
        views.append(await reopened.assemble_context())

    assert store.compaction_marker_count() == marker_count
    expected = [message.to_dict() for message in evicted.messages]
    for view in views:
        assert [message.to_dict() for message in view.messages] == expected
        assert view.digest == evicted.digest
    assert _rendered(evicted.messages).count("latest request verbatim") == 1

    # A later summary compaction replaces the eviction view instead of
    # stacking on it; switching back evicts over the summary without
    # duplicating the pinned request.
    store.append_message(_text(MessageRole.ASSISTANT, "more reasoning " * 200))
    store.append_message(with_message_origin(_text(MessageRole.USER, "newest request"), MessageOrigin.USER))
    assembler.compaction = "summary"
    summarized = await assembler.assemble_context(force=True)
    markers = [entry for entry in store.replay() if entry.type == "compaction"]
    assert markers[-1].data.get("kind", "summary") == "summary"
    active = ContextAssembler._active_markers(store.replay())
    assert [marker.id for marker in active] == [markers[-1].id]
    assert _rendered(summarized.messages).count("newest request") == 1

    store.append_message(_text(MessageRole.ASSISTANT, "even more " * 300))
    store.append_message(with_message_origin(_text(MessageRole.USER, "final request"), MessageOrigin.USER))
    assembler.compaction = "evict"
    final = await assembler.assemble_context(force=True)
    active = ContextAssembler._active_markers(store.replay())
    assert len(active) == 1
    assert _rendered(final.messages).count("final request") == 1
    assert _rendered(final.messages).count("newest request") <= 1


def _agent_call() -> ToolCall:
    return ToolCall(
        "agent-1",
        "agent",
        {"prompt": "inspect the task", "description": "task research"},
    )


@pytest.mark.parametrize(("start", "target"), [("summary", "evict"), ("evict", "summary")])
async def test_children_spawned_after_switch_inherit_mode(
    tmp_path: Path, start: str, target: str
) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call()]),
            ScriptedTurn([TextContent("child done")]),
            ScriptedTurn([TextContent("parent done")]),
        ]
    )
    loop = AgentLoop(
        backend,
        ConversationStore(tmp_path / "sessions"),
        skill_catalog=SkillCatalog.empty(),
        compaction=start,
        max_turns=2,
    )
    apply_compaction(loop, target)
    async for _ in loop.run_turn("start", origin=MessageOrigin.USER):
        pass

    child_tools = _tool_names(backend.calls[1][1])
    assert ("recall_history" in child_tools) is (target == "evict")
    await loop.close()


def test_tool_policy_hides_recall_history_in_evict_mode(tmp_path: Path) -> None:
    registry = ToolRegistry(
        tmp_path,
        session_store=ConversationStore(tmp_path / "sessions"),
        skill_catalog=SkillCatalog.empty(),
        compaction="summary",
        tool_deny=("recall_history",),
    )
    registry.set_compaction("evict")
    assert registry.compaction == "evict"
    assert "recall_history" not in _tool_names(registry.schemas)
    assert "recall_history" not in registry.registered_names

    allowlisted = ToolRegistry(
        tmp_path,
        session_store=ConversationStore(tmp_path / "other"),
        skill_catalog=SkillCatalog.empty(),
        compaction="summary",
        tool_allow=("read",),
    )
    allowlisted.set_compaction("evict")
    assert _tool_names(allowlisted.schemas) == {"read"}


async def test_tui_compaction_reports_policy_block_and_required_tool(
    home: Path,
) -> None:
    blocked = create_app(
        build_parser().parse_args(
            ["--provider", "fake", "--disallowed-tools", "recall_history"]
        )
    )
    try:
        assert "recall_history: blocked by tool policy" in await _slash(
            blocked, "/compaction"
        )
        assert "recall_history" not in _tool_names(blocked.loop._active_tool_schemas())
    finally:
        await blocked.close()

    required = create_app(
        build_parser().parse_args(
            ["--provider", "fake", "--tools", "read,recall_history"]
        )
    )
    try:
        session_id = required.loop.store.session_id
        output = await _slash(required, "/compaction summary")
        assert "compaction unchanged" in output
        assert "recall_history" in output
        assert required.loop.context_assembler.compaction == "evict"
        assert "recall_history" in required.loop.tool_registry.registered_names
        assert _meta(home, session_id)["compaction"] == "evict"
    finally:
        await required.close()


async def test_tui_refuses_switch_while_turn_is_active(home: Path) -> None:
    app = create_app(build_parser().parse_args(["--provider", "fake"]))
    pending = asyncio.get_running_loop().create_future()
    try:
        app._active_task = pending  # type: ignore[assignment]
        assert app.active
        output = await _slash(app, "/compaction summary")
        assert "compaction unchanged" in output
        assert app.loop.context_assembler.compaction == "evict"
    finally:
        pending.cancel()
        app._active_task = None
        await app.close()


def test_headless_resume_flag_switches_persisted_mode(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    parser = build_parser()
    first = parser.parse_args(["--provider", "fake", "-p", "first"])
    assert run_headless(first, first.prompt) == 0
    session_id = SessionManager(home).list_sessions()[0].session_id
    assert _meta(home, session_id)["compaction"] == "evict"

    second = parser.parse_args(
        ["--resume", session_id, "--compaction", "summary", "-p", "second"]
    )
    assert run_headless(second, second.prompt) == 0
    assert _meta(home, session_id)["compaction"] == "summary"
    assert _meta(home, session_id)["compaction_pinned"] is True

    third = parser.parse_args(["--resume", session_id, "-p", "third"])
    assert run_headless(third, third.prompt) == 0
    assert _meta(home, session_id)["compaction"] == "summary"
    capsys.readouterr()


def test_settings_file_compaction_does_not_switch_resumed_session(
    home: Path,
) -> None:
    app = create_app(build_parser().parse_args(["--provider", "fake"]))
    session_id = app.loop.store.session_id
    asyncio.run(app.close())
    home.mkdir(exist_ok=True)
    (home / "settings.toml").write_text('compaction = "summary"\n')

    resumed = create_app(build_parser().parse_args(["--resume", session_id]))
    try:
        assert resumed.loop.context_assembler.compaction == "evict"
        assert _meta(home, session_id)["compaction"] == "evict"
    finally:
        asyncio.run(resumed.close())


def test_default_behaviour_unchanged_for_new_and_legacy_sessions(home: Path) -> None:
    fresh = create_app(build_parser().parse_args(["--provider", "fake"]))
    session_id = fresh.loop.store.session_id
    assert fresh.loop.context_assembler.compaction == "evict"
    assert _meta(home, session_id)["compaction_pinned"] is False
    asyncio.run(fresh.close())

    path = home / "sessions" / session_id / "meta.json"
    metadata = json.loads(path.read_text())
    metadata.pop("compaction")
    metadata.pop("compaction_pinned")
    path.write_text(json.dumps(metadata), encoding="utf-8")
    legacy = create_app(build_parser().parse_args(["--resume", session_id]))
    try:
        assert legacy.loop.context_assembler.compaction == "summary"
        assert "recall_history" not in legacy.loop.tool_registry.registered_names
    finally:
        asyncio.run(legacy.close())


@pytest.mark.parametrize("require_tools", [False, True])
def test_headless_resume_switch_refused_by_required_tool_leaves_metadata(
    home: Path, capsys: pytest.CaptureFixture[str], require_tools: bool
) -> None:
    parser = build_parser()
    first = parser.parse_args(
        ["--provider", "fake", "--tools", "recall_history", "-p", "first"]
    )
    assert run_headless(first, first.prompt) == 0
    session_id = SessionManager(home).list_sessions()[0].session_id
    before = _meta(home, session_id)
    before_bytes = (home / "sessions" / session_id / "meta.json").read_bytes()
    assert before["compaction"] == "evict"
    capsys.readouterr()

    argv = ["--resume", session_id, "--compaction", "summary"]
    if require_tools:
        argv.append("--require-tools")
    second = parser.parse_args([*argv, "-p", "second"])
    assert run_headless(second, second.prompt) == 1

    err = capsys.readouterr().err
    assert "recall_history" in err
    after = _meta(home, session_id)
    assert (after["compaction"], after["compaction_pinned"]) == (
        before["compaction"],
        before["compaction_pinned"],
    )
    assert (home / "sessions" / session_id / "meta.json").read_bytes() == before_bytes


def test_tui_resume_switch_refused_by_required_tool_leaves_metadata(
    home: Path,
) -> None:
    app = create_app(
        build_parser().parse_args(["--provider", "fake", "--tools", "recall_history"])
    )
    session_id = app.loop.store.session_id
    asyncio.run(app.close())
    before = _meta(home, session_id)
    before_bytes = (home / "sessions" / session_id / "meta.json").read_bytes()

    with pytest.raises(SessionError, match="recall_history"):
        create_app(
            build_parser().parse_args(
                ["--resume", session_id, "--compaction", "summary"]
            )
        )

    after = _meta(home, session_id)
    assert (after["compaction"], after["compaction_pinned"]) == (
        before["compaction"],
        before["compaction_pinned"],
    )
    assert (home / "sessions" / session_id / "meta.json").read_bytes() == before_bytes


def test_tui_resume_switch_persists_after_startup_validation(home: Path) -> None:
    app = create_app(build_parser().parse_args(["--provider", "fake"]))
    session_id = app.loop.store.session_id
    asyncio.run(app.close())

    args = build_parser().parse_args(["--resume", session_id, "--compaction", "summary"])
    resumed = create_app(args)
    try:
        assert resumed.loop.context_assembler.compaction == "summary"
        assert "recall_history" not in resumed.loop.tool_registry.registered_names
        assert _meta(home, session_id)["compaction"] == "summary"
        assert _meta(home, session_id)["compaction_pinned"] is True
        assert resumed.loop.session_metadata.compaction == "summary"
    finally:
        asyncio.run(resumed.close())

    reopened = create_app(build_parser().parse_args(["--resume", session_id]))
    try:
        assert reopened.loop.context_assembler.compaction == "summary"
    finally:
        asyncio.run(reopened.close())
