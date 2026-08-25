import asyncio
from dataclasses import dataclass, replace
from io import StringIO
from pathlib import Path

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from zeta.core.approval import ApprovalPolicy
from zeta.core.context import ContextAssembler
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.slash import SlashStatus, create_slash_registry
from zeta.core.store import ConversationStore
from zeta.loop import AgentLoop
from zeta.tui.app import TUIApp
from zeta.tui.composer import build_key_bindings
from zeta.types import Message, MessageRole, TextContent, ToolCall, ToolUseContent


@dataclass(frozen=True, slots=True)
class FakeSlashSession:
    status: SlashStatus

    def slash_status(self) -> SlashStatus:
        return self.status


def session() -> FakeSlashSession:
    return FakeSlashSession(
        SlashStatus(
            session_id="session-1",
            provider="fake",
            model="offline",
            retained_tail=8,
            tokens_used_this_session=123,
            tokens_in_current_context=45,
            compaction_marker_count=2,
            pending_approvals=("approval-1 (write)",),
            cache_read_input_tokens=50,
            cache_creation_input_tokens=25,
            uncached_input_tokens=25,
            output_tokens_this_session=4,
        )
    )


@pytest.mark.asyncio
async def test_status_returns_live_required_fields(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions", session_id="test-xyz-123")
    store.append_message(Message(MessageRole.USER, [TextContent("old")]))
    store.append_compaction_marker("summary", 1, 1)
    store.append_compaction_marker("summary 2", 1, 1)
    backend = FakeBackend([])
    policy = ApprovalPolicy(store=store)
    loop = AgentLoop(
        backend,
        store,
        approval_policy=policy,
        retained_tail=17,
    )
    await loop.context_assembler.assemble()
    loop.context_assembler.record_usage(
        {
            "total_tokens": 321,
            "cache_read_input_tokens": 50,
            "cache_creation_input_tokens": 25,
            "input_tokens": 25,
            "output_tokens": 4,
        }
    )
    calls = [
        ToolCall("approval-live-1", "write", {}),
        ToolCall("approval-live-2", "write", {}),
    ]
    store.append_message_with_approval_requests(
        Message(
            MessageRole.ASSISTANT,
            [ToolUseContent(call) for call in calls],
        ),
        [(call.id, call) for call in calls],
    )
    app = TUIApp(
        loop,
        provider="provider-live",
        model="model-live",
        approval_policy=policy,
    )
    output = create_slash_registry().dispatch(app, "/status")

    assert output is not None
    assert f"session_id: {store.session_id}" in output
    assert f"provider: {app.provider}" in output
    assert f"model: {app.model}" in output
    assert f"retained_tail: {loop.context_assembler.retained_tail}" in output
    assert (
        "tokens_used_this_session: "
        f"{loop.context_assembler.tokens_used_this_session}"
    ) in output
    assert (
        "tokens_in_current_context: "
        f"{loop.context_assembler.token_count}"
    ) in output
    assert f"compaction_marker_count: {store.compaction_marker_count()}" in output
    assert "compaction_marker_count: 2" in output
    assert "live_pending_approvals: 2" in output
    assert "prompt_cache_read: 50" in output
    assert "prompt_cache_write: 25" in output
    assert "prompt_cache_uncached_input: 25" in output
    assert "prompt_cache_hit_rate: 50.0%" in output
    assert "output_tokens_this_session: 4" in output


@pytest.mark.asyncio
async def test_status_counts_compaction_usage(tmp_path: Path) -> None:
    def token_count(message: Message) -> int:
        if message.role is MessageRole.COMPACTION or message.metadata.get(
            "compaction_summary"
        ):
            return 1
        return 40

    backend = FakeBackend(
        [
            ScriptedTurn([TextContent("first")], usage={"total_tokens": 20}),
            ScriptedTurn([TextContent("summary")], usage={"total_tokens": 10}),
            ScriptedTurn([TextContent("second")], usage={"total_tokens": 5}),
        ]
    )
    store = ConversationStore(tmp_path / "sessions")
    assembler = ContextAssembler(
        store,
        backend=backend,
        token_budget=80,
        retained_tail=1,
        token_counter=token_count,
    )
    loop = AgentLoop(backend, store, context_assembler=assembler)

    async for _ in loop.run_turn("first"):
        pass
    async for _ in loop.run_turn("second"):
        pass

    app = TUIApp(loop, provider="fake", model="offline")
    output = create_slash_registry().dispatch(app, "/status")

    assert output is not None
    assert "tokens_used_this_session: 35" in output


def test_status_renders_cache_hit_rate_as_na_without_usage() -> None:
    empty_session = FakeSlashSession(
        replace(
            session().status,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            uncached_input_tokens=0,
        )
    )
    output = create_slash_registry().dispatch(empty_session, "/status")

    assert output is not None
    assert "prompt_cache_hit_rate: n/a" in output


def test_unknown_command_passes_through_unchanged() -> None:
    registry = create_slash_registry()

    assert registry.dispatch(session(), "/unknown arg") is None
    assert registry.input_for_model("/unknown arg") == "/unknown arg"


def test_double_slash_escapes_registered_command() -> None:
    registry = create_slash_registry()

    assert registry.dispatch(session(), "//status") is None
    assert registry.input_for_model("//status") == "/status"


def test_multiline_known_command_consumes_the_whole_message() -> None:
    output = create_slash_registry().dispatch(
        session(), "/status\nmodel must not see this"
    )

    assert output is not None
    assert "model must not see this" not in output


def test_empty_message_does_nothing() -> None:
    registry = create_slash_registry()

    assert registry.dispatch(session(), "") is None
    assert registry.input_for_model("") == ""


def test_model_command_shows_and_changes_the_model() -> None:
    class ModelSession:
        def __init__(self) -> None:
            self.model = "offline"

        def slash_status(self) -> SlashStatus:
            return session().status

        def slash_model(self, args: str) -> str:
            if args:
                self.model = args
            return f"model: {self.model}"

        async def slash_compact(self) -> str:
            return "compact: nothing to compact"

    model_session = ModelSession()
    registry = create_slash_registry()

    assert registry.dispatch(model_session, "/model") == "model: offline"
    assert registry.dispatch(model_session, "/model faster") == "model: faster"


@pytest.mark.asyncio
async def test_compact_command_uses_async_dispatch() -> None:
    class CompactSession:
        def slash_status(self) -> SlashStatus:
            return session().status

        def slash_model(self, args: str) -> str:
            del args
            return "model: offline"

        async def slash_compact(self) -> str:
            return "compacted entries 1–2; tokens after: 3"

    output = await create_slash_registry().dispatch_async(
        CompactSession(), "/compact"
    )

    assert output == "compacted entries 1–2; tokens after: 3"


@pytest.mark.asyncio
async def test_tui_renders_status_without_calling_the_model(tmp_path: Path) -> None:
    backend = FakeBackend([])
    output = StringIO()
    app = TUIApp(
        AgentLoop(backend, ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
        console=Console(file=output, force_terminal=False),
    )
    with create_pipe_input() as pipe:
        prompt = PromptSession(
            input=pipe,
            output=DummyOutput(),
            key_bindings=build_key_bindings(
                on_interrupt=app.abort_active,
                on_exit=app.request_exit,
            ),
            multiline=True,
        )
        run_task = asyncio.create_task(app.run(prompt))
        pipe.send_text("/status\r")
        for _ in range(100):
            if "session_id:" in output.getvalue():
                break
            await asyncio.sleep(0.01)
        pipe.send_text("\x04")
        await run_task

    assert "session_id:" in output.getvalue()
    assert "provider: fake" in output.getvalue()
    assert backend.calls == []
