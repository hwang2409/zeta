from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from zeta.cli import build_parser
from zeta.core.approval import ApprovalDecision, ApprovalPolicy
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.slash import create_slash_registry
from zeta.core.store import ConversationStore
from zeta.loop import AgentLoop
from zeta.tools.plan_mode import EXIT_PLAN_MODE, PLAN_MODE_TOOLS
from zeta.tui.app import create_app
from zeta.tui.render import format_status
from zeta.types import StreamEvent, TextContent, ToolCall


async def collect(events: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [event async for event in events]


def schema_names(schemas: list[dict]) -> set[str]:
    return {schema["name"] for schema in schemas}


def build_loop(tmp_path: Path, turns: list[ScriptedTurn]) -> AgentLoop:
    backend = FakeBackend(turns)
    store = ConversationStore(tmp_path, cwd=str(tmp_path))
    return AgentLoop(
        backend,
        store,
        approval_policy=ApprovalPolicy(
            store=store, default=ApprovalDecision.ALLOW, always_ask=()
        ),
        skip_mcp_mount=True,
    )


# --- schema exposure -------------------------------------------------------


async def test_exit_plan_mode_is_hidden_outside_plan_mode(tmp_path: Path) -> None:
    loop = build_loop(tmp_path, [ScriptedTurn(content=[TextContent("ok")])])
    await collect(loop.run_turn("hi"))
    _, schemas = loop.backend.calls[0]
    assert EXIT_PLAN_MODE not in schema_names(schemas)
    assert "bash" in schema_names(schemas)


async def test_plan_mode_narrows_the_schemas_the_provider_sees(
    tmp_path: Path,
) -> None:
    loop = build_loop(tmp_path, [ScriptedTurn(content=[TextContent("ok")])])
    loop.set_plan_mode(True)
    await collect(loop.run_turn("hi"))
    _, schemas = loop.backend.calls[0]
    names = schema_names(schemas)
    assert names == PLAN_MODE_TOOLS | {EXIT_PLAN_MODE}
    assert not names & {"bash", "edit", "write", "exec", "agent"}


async def test_leaving_plan_mode_restores_the_full_schemas(tmp_path: Path) -> None:
    loop = build_loop(
        tmp_path,
        [
            ScriptedTurn(content=[TextContent("one")]),
            ScriptedTurn(content=[TextContent("two")]),
        ],
    )
    loop.set_plan_mode(True)
    await collect(loop.run_turn("hi"))
    loop.set_plan_mode(False)
    await collect(loop.run_turn("again"))
    assert "bash" not in schema_names(loop.backend.calls[0][1])
    assert "bash" in schema_names(loop.backend.calls[1][1])
    assert EXIT_PLAN_MODE not in schema_names(loop.backend.calls[1][1])


def test_plan_mode_is_idempotent(tmp_path: Path) -> None:
    loop = build_loop(tmp_path, [])
    original = loop.context_assembler.system_prompt
    loop.set_plan_mode(True)
    composed = loop.context_assembler.system_prompt
    loop.set_plan_mode(True)
    assert loop.context_assembler.system_prompt is composed
    loop.set_plan_mode(False)
    assert loop.context_assembler.system_prompt is original


# --- system prompt ---------------------------------------------------------


def test_plan_mode_wraps_and_restores_the_system_prompt(tmp_path: Path) -> None:
    loop = build_loop(tmp_path, [])
    original = loop.context_assembler.system_prompt
    loop.set_plan_mode(True)
    text = "".join(
        block.text
        for block in loop.context_assembler.system_prompt.content
        if isinstance(block, TextContent)
    )
    assert "PLAN MODE" in text
    assert EXIT_PLAN_MODE in text
    loop.set_plan_mode(False)
    assert loop.context_assembler.system_prompt is original


# --- execution gate --------------------------------------------------------


def test_plan_mode_denies_the_mutating_tools(tmp_path: Path) -> None:
    loop = build_loop(tmp_path, [])
    policy = loop.tool_registry.approval_policy
    assert policy is not None
    assert policy.decide("bash", {}) is ApprovalDecision.ALLOW

    loop.set_plan_mode(True)
    for name in ("bash", "edit", "write", "exec", "agent"):
        assert policy.decide(name, {}) is ApprovalDecision.DENY, name
    for name in sorted(PLAN_MODE_TOOLS):
        assert policy.decide(name, {}) is not ApprovalDecision.DENY, name
    assert policy.decide(EXIT_PLAN_MODE, {}) is not ApprovalDecision.DENY

    loop.set_plan_mode(False)
    assert policy.decide("bash", {}) is ApprovalDecision.ALLOW


# --- the approve / reject transition ---------------------------------------


async def test_approved_plan_leaves_plan_mode_and_restores_tools(
    tmp_path: Path,
) -> None:
    loop = build_loop(
        tmp_path,
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall("call-1", EXIT_PLAN_MODE, {"plan": "1. do the thing"})
                ]
            ),
            ScriptedTurn(content=[TextContent("done")]),
        ],
    )
    loop.set_plan_mode(True)
    await collect(loop.run_turn("plan it"))
    assert loop.plan_mode is False
    # The turn after approval carries the plan out with the full tool set.
    assert "bash" in schema_names(loop.backend.calls[1][1])


async def test_a_rejected_plan_keeps_plan_mode_on(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn(
                tool_calls=[ToolCall("call-1", EXIT_PLAN_MODE, {"plan": "x"})]
            ),
            ScriptedTurn(content=[TextContent("revising")]),
        ]
    )
    store = ConversationStore(tmp_path, cwd=str(tmp_path))
    loop = AgentLoop(
        backend,
        store,
        approval_policy=ApprovalPolicy(store=store, default=ApprovalDecision.DENY),
        skip_mcp_mount=True,
    )
    loop.set_plan_mode(True)
    await collect(loop.run_turn("plan it"))
    assert loop.plan_mode is True
    assert "bash" not in schema_names(backend.calls[1][1])


async def test_an_empty_plan_does_not_leave_plan_mode(tmp_path: Path) -> None:
    loop = build_loop(
        tmp_path,
        [
            ScriptedTurn(
                tool_calls=[ToolCall("call-1", EXIT_PLAN_MODE, {"plan": "   "})]
            ),
            ScriptedTurn(content=[TextContent("retry")]),
        ],
    )
    loop.set_plan_mode(True)
    await collect(loop.run_turn("plan it"))
    assert loop.plan_mode is True


# --- the /plan command -----------------------------------------------------


class PlanSession:
    def __init__(self) -> None:
        self.plan_mode = False
        self.active = False
        self.pending_approvals: tuple[object, ...] = ()
        self.loop = self

    def set_plan_mode(self, enabled: bool) -> None:
        self.plan_mode = enabled

    def _invalidate_prompt(self) -> None:
        return None


def dispatch(session: object, value: str) -> str | None:
    return create_slash_registry().dispatch(session, value)


def test_plan_command_reports_and_toggles() -> None:
    from zeta.tui.slash_handlers import SlashHandlerMixin

    class Session(SlashHandlerMixin, PlanSession):
        pass

    session = Session()
    assert dispatch(session, "/plan") == "plan mode: off"
    assert dispatch(session, "/plan on") == (
        "plan mode: on (read-only tools until you approve a plan)"
    )
    assert session.plan_mode is True
    assert dispatch(session, "/plan") == "plan mode: on"
    assert dispatch(session, "/plan toggle") == "plan mode: off"
    assert session.plan_mode is False


def test_plan_command_rejects_an_unknown_argument() -> None:
    from zeta.tui.slash_handlers import SlashHandlerMixin

    class Session(SlashHandlerMixin, PlanSession):
        pass

    session = Session()
    assert dispatch(session, "/plan sideways") == (
        "plan mode unchanged: use /plan on, /plan off, or /plan toggle"
    )
    assert session.plan_mode is False


def test_plan_command_refuses_while_a_turn_is_active() -> None:
    from zeta.tui.slash_handlers import SlashHandlerMixin

    class Session(SlashHandlerMixin, PlanSession):
        pass

    session = Session()
    session.active = True
    assert dispatch(session, "/plan on") == (
        "plan mode unchanged: cannot change plan mode while a turn or "
        "approval is active"
    )
    assert session.plan_mode is False


def test_plan_command_is_listed_in_help() -> None:
    assert "/plan" in create_slash_registry().help_text()


# --- surfaces --------------------------------------------------------------


def test_status_bar_shows_and_styles_the_plan_segment() -> None:
    rendered = format_status(
        "claude", "claude-sonnet-4-6", "idle", None, None, plan_state="PLAN"
    )
    assert rendered.plain.startswith("PLAN")
    assert any(span.style for span in rendered.spans)


def test_status_bar_omits_the_plan_segment_when_off() -> None:
    rendered = format_status("claude", "claude-sonnet-4-6", "idle", None, None)
    assert "PLAN" not in rendered.plain


def test_yolo_still_asks_before_leaving_plan_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZETA_HOME", str(tmp_path / "zeta-home"))
    app = create_app(
        build_parser().parse_args(["--provider", "fake", "--yolo"])
    )
    policy = app.loop.tool_registry.approval_policy
    assert policy is not None
    assert policy.decide("bash", {}) is ApprovalDecision.ALLOW
    assert policy.decide(EXIT_PLAN_MODE, {}) is ApprovalDecision.ASK


def test_status_command_reports_plan_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZETA_HOME", str(tmp_path / "zeta-home"))
    app = create_app(build_parser().parse_args(["--provider", "fake"]))
    assert "plan_mode: off" in dispatch(app, "/status")
    app.loop.set_plan_mode(True)
    assert "plan_mode: on" in dispatch(app, "/status")


def test_plan_mode_does_not_touch_session_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    app = create_app(build_parser().parse_args(["--provider", "fake"]))
    app.loop.set_plan_mode(True)
    session_id = app.loop.store.session_id
    metadata = json.loads((home / "sessions" / session_id / "meta.json").read_text())
    assert "plan_mode" not in metadata
