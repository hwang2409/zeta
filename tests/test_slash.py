import asyncio
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from zeta.core.fake import FakeBackend
from zeta.core.slash import SlashStatus, create_slash_registry
from zeta.core.store import ConversationStore
from zeta.loop import AgentLoop
from zeta.tui.app import TUIApp
from zeta.tui.composer import build_key_bindings


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
        )
    )


def test_status_returns_all_required_fields() -> None:
    output = create_slash_registry().dispatch(session(), "/status")

    assert output is not None
    assert "session_id: session-1" in output
    assert "provider: fake" in output
    assert "model: offline" in output
    assert "retained_tail: 8" in output
    assert "tokens_used_this_session: 123" in output
    assert "tokens_in_current_context: 45" in output
    assert "compaction_marker_count: 2" in output
    assert "live_pending_approvals: 1 (approval-1 (write))" in output


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
