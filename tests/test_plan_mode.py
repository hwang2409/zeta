from __future__ import annotations

import json
from collections.abc import AsyncIterator
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from zeta.cli import build_parser
from zeta.core.approval import ApprovalDecision, ApprovalPolicy
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.slash import create_slash_registry
from zeta.core.store import ConversationStore
from zeta.loop import AgentLoop
from zeta.mcp.prompt_commands import SlashModelInput
from zeta.tools.plan_mode import PLAN_MODE_TOOLS
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


def build_app(tmp_path: Path, turns: list[ScriptedTurn]):
    from zeta.tui.app import TUIApp

    return TUIApp(
        build_loop(tmp_path, turns),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=True),
    )


# --- schema exposure -------------------------------------------------------


async def test_plan_mode_has_no_exit_tool(tmp_path: Path) -> None:
    loop = build_loop(tmp_path, [ScriptedTurn(content=[TextContent("ok")])])
    await collect(loop.run_turn("hi"))
    _, schemas = loop.backend.calls[0]
    assert "exit_plan_mode" not in schema_names(schemas)
    assert "bash" in schema_names(schemas)


async def test_plan_mode_narrows_the_schemas_the_provider_sees(
    tmp_path: Path,
) -> None:
    loop = build_loop(tmp_path, [ScriptedTurn(content=[TextContent("ok")])])
    loop.set_plan_mode(True)
    await collect(loop.run_turn("hi"))
    _, schemas = loop.backend.calls[0]
    names = schema_names(schemas)
    assert names == PLAN_MODE_TOOLS | {"agent"}
    assert not names & {"bash", "edit", "write", "exec"}


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
    assert "exit_plan_mode" not in schema_names(loop.backend.calls[1][1])


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
    assert "deliver" in text
    assert "approve" not in text
    loop.set_plan_mode(False)
    assert loop.context_assembler.system_prompt is original


# --- execution gate --------------------------------------------------------


def test_plan_mode_checks_the_allowlist_without_mutating_approval_policy(
    tmp_path: Path,
) -> None:
    loop = build_loop(tmp_path, [])
    policy = loop.tool_registry.approval_policy
    assert policy is not None
    assert policy.decide("bash", {}) is ApprovalDecision.ALLOW

    loop.set_plan_mode(True)
    for name in ("bash", "edit", "write", "exec", "agent"):
        if name == "agent":
            assert loop.plan_mode_allows(name), name
        else:
            assert not loop.plan_mode_allows(name), name
    for name in sorted(PLAN_MODE_TOOLS):
        assert loop.plan_mode_allows(name), name
    assert policy.decide("bash", {}) is ApprovalDecision.ALLOW

    loop.set_plan_mode(False)
    assert policy.decide("bash", {}) is ApprovalDecision.ALLOW


# --- turn boundaries and persistence --------------------------------------


async def test_plan_delivery_ends_the_turn_without_implementation(
    tmp_path: Path,
) -> None:
    loop = build_loop(
        tmp_path,
        [ScriptedTurn(content=[TextContent("1. do the thing")])],
    )
    loop.set_plan_mode(True)
    await collect(loop.run_turn("plan it"))
    assert loop.plan_mode is True
    assert len(loop.backend.calls) == 1


async def test_plan_mode_persists_across_follow_up_turns(tmp_path: Path) -> None:
    loop = build_loop(
        tmp_path,
        [
            ScriptedTurn(content=[TextContent("first plan")]),
            ScriptedTurn(content=[TextContent("revised plan")]),
        ],
    )
    loop.set_plan_mode(True)
    await collect(loop.run_turn("plan it"))
    await collect(loop.run_turn("revise it"))
    assert loop.plan_mode is True
    assert all(
        schema_names(call[1]) == PLAN_MODE_TOOLS | {"agent"}
        for call in loop.backend.calls
    )


async def test_plan_prompt_enters_mode_and_submits_in_one_action(
    tmp_path: Path,
) -> None:
    app = build_app(tmp_path, [ScriptedTurn(content=[TextContent("the plan")])])
    await app._handle_prompt_value("/plan inspect the repository")
    assert app._active_task is not None
    await app._active_task

    assert app.loop.plan_mode is True
    assert len(app.loop.backend.calls) == 1
    user_message = next(
        message
        for message in app.loop.backend.calls[0][0]
        if message.role.value == "user"
    )
    assert user_message.content[0].text == "inspect the repository"


async def test_implement_exits_plan_mode_on_a_later_turn(
    tmp_path: Path,
) -> None:
    app = build_app(
        tmp_path,
        [
            ScriptedTurn(content=[TextContent("the plan")]),
            ScriptedTurn(content=[TextContent("implemented")]),
        ],
    )
    await app._handle_prompt_value("/plan inspect the repository")
    assert app._active_task is not None
    await app._active_task
    assert app.loop.plan_mode is True

    await app._handle_prompt_value("/implement")
    assert app._active_task is not None
    await app._active_task

    assert app.loop.plan_mode is False
    assert len(app.loop.backend.calls) == 2
    second_user_message = [
        message
        for message in app.loop.backend.calls[1][0]
        if message.role.value == "user"
    ][-1]
    assert second_user_message.content[0].text == (
        "implement the plan you proposed above"
    )
    assert "bash" in schema_names(app.loop.backend.calls[1][1])


async def test_general_sub_agents_are_rejected_in_plan_mode(tmp_path: Path) -> None:
    loop = build_loop(
        tmp_path,
        [],
    )
    loop.set_plan_mode(True)
    result = await loop.tool_registry.execute(
        ToolCall(
            "call-1",
            "agent",
            {
                "prompt": "inspect this",
                "description": "inspect this",
                "agent_type": "general",
            },
        )
    )
    assert result["isError"] is True
    assert "general agents are unavailable in plan mode" in result["content"][0][
        "text"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_type", ["explore", "plan"])
async def test_allowed_sub_agents_complete_a_turn_in_plan_mode(
    tmp_path: Path, agent_type: str
) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[ToolCall(
                "agent-call",
                "agent",
                {
                    "prompt": "inspect this",
                    "description": "task research",
                    "agent_type": agent_type,
                },
            )]),
            ScriptedTurn([TextContent("child complete")]),
        ]
    )
    store = ConversationStore(tmp_path)
    loop = AgentLoop(
        backend,
        store,
        max_turns=1,
        approval_policy=ApprovalPolicy(
            store=store, default=ApprovalDecision.ALLOW
        ),
    )
    loop.set_plan_mode(True)

    await collect(loop.run_turn("start"))

    child_result = next(
        message.tool_result
        for message in store.messages()
        if message.tool_result is not None
    )
    assert child_result.is_error is False
    assert child_result.content.startswith("child complete")
    assert len(backend.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("approval_default", [ApprovalDecision.ALLOW, ApprovalDecision.ASK])
async def test_late_mounted_tool_is_rejected_in_plan_mode(
    tmp_path: Path, approval_default: ApprovalDecision
) -> None:
    executed = False

    async def late_tool(arguments: dict[str, object]) -> str:
        del arguments
        nonlocal executed
        executed = True
        return "must not run"

    call = ToolCall("late-call", "remote:write", {})
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[call]), ScriptedTurn([TextContent("done")])]
    )
    store = ConversationStore(tmp_path)
    loop = AgentLoop(
        backend,
        store,
        approval_policy=ApprovalPolicy(store=store, default=approval_default),
    )
    loop.set_plan_mode(True)
    loop.tool_registry.register(
        "remote:write",
        late_tool,
        parameters={"type": "object"},
    )

    await collect(loop.run_turn("start"))

    result = next(
        message.tool_result
        for message in store.messages()
        if message.tool_result is not None
    )
    assert result.is_error is True
    assert "remote:write is not allowed" in result.content
    assert executed is False
    assert store.pending_approvals() == []


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


def dispatch(session: object, value: str) -> str | SlashModelInput | None:
    return create_slash_registry().dispatch(session, value)


def test_plan_command_reports_and_toggles() -> None:
    from zeta.tui.slash_handlers import SlashHandlerMixin

    class Session(SlashHandlerMixin, PlanSession):
        pass

    session = Session()
    assert dispatch(session, "/plan") == (
        "plan mode: on (read-only tools; deliver the plan as your answer)"
    )
    assert session.plan_mode is True
    assert dispatch(session, "/plan") == "plan mode: off"
    assert dispatch(session, "/plan on") == (
        "plan mode: on (read-only tools; deliver the plan as your answer)"
    )
    assert dispatch(session, "/plan off") == "plan mode: off"
    assert session.plan_mode is False


def test_plan_command_submits_prompt_when_text_is_present() -> None:
    from zeta.tui.slash_handlers import SlashHandlerMixin

    class Session(SlashHandlerMixin, PlanSession):
        pass

    session = Session()
    result = dispatch(session, "/plan inspect the repository")
    assert isinstance(result, SlashModelInput)
    assert result.text == "inspect the repository"
    assert session.plan_mode is True


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


@pytest.mark.asyncio
async def test_plan_command_refuses_live_background_work_then_allows_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    monkeypatch.setenv("ZETA_HOME", str(tmp_path / "zeta-home"))
    app = build_app(tmp_path, [])
    watcher = asyncio.create_task(asyncio.sleep(0))
    app.loop._background_owner.register(
        "macro:test",
        lambda: None,
        watcher,
        description="/build",
    )

    blocked = dispatch(app, "/plan on")
    assert blocked == (
        "plan mode unchanged: background work is active: /build; stop it or wait"
    )
    assert app.loop.plan_mode is False

    await watcher
    app.loop._background_owner.unregister("macro:test")
    assert dispatch(app, "/plan on") == (
        "plan mode: on (read-only tools; deliver the plan as your answer)"
    )
    assert app.loop.plan_mode is True


@pytest.mark.asyncio
async def test_plan_command_refuses_running_background_process_then_allows_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZETA_HOME", str(tmp_path / "zeta-home"))
    app = build_app(tmp_path, [])
    task_id, _ = await app.loop.tool_registry.background_tasks.start(
        "sleep 0.2", tmp_path
    )

    assert dispatch(app, "/plan on") == (
        "plan mode unchanged: background work is active: sleep 0.2; stop it or wait"
    )
    assert app.loop.plan_mode is False

    await app.loop.tool_registry.background_tasks.wait(task_id)
    assert dispatch(app, "/plan on") == (
        "plan mode: on (read-only tools; deliver the plan as your answer)"
    )
    assert app.loop.plan_mode is True


def test_plan_command_is_listed_in_help() -> None:
    assert "/plan" in create_slash_registry().help_text()
    assert "/implement" in create_slash_registry().help_text()


def test_implement_exits_and_submits_the_explicit_request() -> None:
    from zeta.tui.slash_handlers import SlashHandlerMixin

    class Session(SlashHandlerMixin, PlanSession):
        pass

    session = Session()
    session.set_plan_mode(True)
    result = dispatch(session, "/implement")
    assert isinstance(result, SlashModelInput)
    assert result.text == "implement the plan you proposed above"
    assert session.plan_mode is False


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


def test_yolo_has_no_plan_mode_approval_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZETA_HOME", str(tmp_path / "zeta-home"))
    app = create_app(
        build_parser().parse_args(["--provider", "fake", "--yolo"])
    )
    policy = app.loop.tool_registry.approval_policy
    assert policy is not None
    assert policy.decide("bash", {}) is ApprovalDecision.ALLOW


def test_status_command_reports_plan_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZETA_HOME", str(tmp_path / "zeta-home"))
    app = create_app(build_parser().parse_args(["--provider", "fake"]))
    assert "plan_mode: off" in dispatch(app, "/status")
    app.loop.set_plan_mode(True)
    assert "plan_mode: on" in dispatch(app, "/status")


def test_plan_mode_persists_and_restores_session_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    app = create_app(build_parser().parse_args(["--provider", "fake"]))
    app.loop.set_plan_mode(True)
    session_id = app.loop.store.session_id
    metadata = json.loads((home / "sessions" / session_id / "meta.json").read_text())
    assert metadata["plan_mode"] is True

    resumed = create_app(
        build_parser().parse_args(["--resume", session_id, "--provider", "fake"])
    )
    assert resumed.loop.plan_mode is True


# --- the shift+tab binding -------------------------------------------------


async def test_shift_tab_toggles_plan_mode() -> None:
    import asyncio

    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from zeta.tui.key_bindings import build_key_bindings

    toggles: list[str] = []
    with create_pipe_input() as pipe:
        session = PromptSession(
            input=pipe,
            output=DummyOutput(),
            key_bindings=build_key_bindings(
                on_interrupt=lambda: None,
                on_exit=lambda: None,
                on_plan_toggle=lambda: toggles.append("toggle"),
            ),
            multiline=True,
        )
        task = asyncio.create_task(session.prompt_async(" > "))
        await asyncio.sleep(0)
        pipe.send_text("\x1b[Z")  # shift+tab
        await asyncio.sleep(0.05)
        assert toggles == ["toggle"]

        # It is not a printable key, so it works with text in the composer too.
        pipe.send_text("hello\x1b[Z")
        await asyncio.sleep(0.05)
        assert toggles == ["toggle", "toggle"]
        assert session.default_buffer.text == "hello"

        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def test_shift_tab_is_inert_without_a_handler() -> None:
    from zeta.tui.key_bindings import build_key_bindings

    bindings = build_key_bindings(
        on_interrupt=lambda: None,
        on_exit=lambda: None,
    )
    assert not [
        binding
        for binding in bindings.bindings
        if any(str(key).endswith("BackTab") for key in binding.keys)
    ]


def test_key_toggle_reports_through_the_same_path_as_the_command() -> None:
    from zeta.tui.slash_handlers import SlashHandlerMixin

    printed: list[str] = []

    class Session(SlashHandlerMixin, PlanSession):
        def _print_system(self, output: str) -> None:
            printed.append(output)

    session = Session()
    session.toggle_plan_mode()
    assert session.plan_mode is True
    assert printed == [
        "plan mode: on (read-only tools; deliver the plan as your answer)"
    ]
    session.toggle_plan_mode()
    assert session.plan_mode is False
    assert printed[-1] == "plan mode: off"
