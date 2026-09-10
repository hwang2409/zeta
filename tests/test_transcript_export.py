"""``/copy``: plain-text export of the visible transcript."""

from __future__ import annotations

import shutil
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from zeta.core.fake import FakeBackend
from zeta.core.slash import create_slash_registry
from zeta.core.store import ConversationStore
from zeta.loop import AgentLoop
from zeta.tui import theme
from zeta.tui.app import TUIApp
from zeta.tui.render import render_error_card, render_event, render_markdown
from zeta.tui.slash_handlers import transcript_export
from zeta.tui.slash_handlers.transcript_export import (
    ClipboardError,
    clipboard_command,
    copy_to_clipboard,
    plain_transcript,
)
from zeta.tui.transcript import TranscriptWidget
from zeta.tui.transcript_presenter import TranscriptPresenter
from zeta.types import ErrorInfo, StreamEvent, StreamEventType, ToolCall, ToolResult

BOX_GLYPHS = "╭╮╰╯│─┏┓┗┛┃"
LONG_REASON = "Anthropic HTTP 404: " + "x" * 1_000


def _user(text: str) -> Text:
    return Text.assemble(("▌ ", theme.USER_ROLE), (text, theme.BODY))


def _error_event(message: str = LONG_REASON) -> StreamEvent:
    return StreamEvent(
        StreamEventType.ERROR, error=ErrorInfo("http_error", message)
    )


def _populated_transcript() -> TranscriptWidget:
    transcript = TranscriptWidget()
    presenter = TranscriptPresenter(
        transcript,
        Console(file=StringIO(), force_terminal=False),
        lambda: True,
        transcript.append,
    )
    presenter.print_user(_user("first question\nsecond line"))
    presenter.print_unit(render_markdown("# Title\n\n- item one\n- item two"))
    call = ToolCall("call-1", "bash", {"command": "ls"})
    presenter.handle_tool_event(
        StreamEvent(StreamEventType.TOOL_EXECUTION_START, tool_call=call),
        aborted=False,
    )
    presenter.handle_tool_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=call,
            tool_result=ToolResult(call.id, "README.md\npyproject.toml"),
        ),
        aborted=False,
    )
    presenter.print_user(_user("/model opus"))
    presenter.print_unit(Text("system · model: opus", style=theme.CHROME))
    presenter.print_unit(render_event(_error_event()))
    return transcript


def test_plain_transcript_strips_chrome_and_keeps_full_text() -> None:
    transcript = _populated_transcript()

    text = plain_transcript(
        transcript.export_entries(), header="zeta · fake · offline · session abc12345"
    )

    assert text.startswith("zeta · fake · offline · session abc12345\n\n")
    assert "> first question\n> second line\n" in text
    assert "▌" not in text
    assert not any(glyph in text for glyph in BOX_GLYPHS)
    # Assistant markdown is exported as its source, not the painted form.
    assert "# Title\n\n- item one\n- item two" in text
    assert "bash" in text
    assert "README.md" in text
    assert "system · model: opus" in text
    assert "provider failure · http_error\nreason: " + LONG_REASON in text
    assert "retry: ctrl+y" not in text
    assert "\n\n\n" not in text
    assert text.endswith("\n")


def test_plain_transcript_last_turn_starts_at_newest_user_message() -> None:
    transcript = _populated_transcript()

    text = plain_transcript(transcript.export_entries(), last_turn=True)

    assert text.startswith("> /model opus\n")
    assert "first question" not in text
    assert "# Title" not in text
    assert "provider failure · http_error" in text


def test_plain_transcript_is_empty_when_nothing_was_shown() -> None:
    transcript = TranscriptWidget()
    transcript.append_blank()

    assert plain_transcript(transcript.export_entries(), header="zeta") == ""


def test_error_card_wraps_on_screen_and_exports_untruncated() -> None:
    rendered = render_error_card(_error_event())

    assert isinstance(rendered, Panel)
    reason = rendered.renderable.renderables[1]
    assert isinstance(reason, Text)
    assert reason.no_wrap is False
    assert reason.overflow == "fold"
    assert len(reason.plain) < 420
    assert rendered.plain_export == (
        "provider failure · http_error\nreason: " + LONG_REASON
    )


def test_error_card_export_keeps_full_json_payload() -> None:
    payload = '{"message": "' + "y" * 800 + '"}'
    rendered = render_error_card(
        StreamEvent(StreamEventType.ERROR, error=ErrorInfo("stream_error", payload))
    )

    assert rendered.plain_export == (
        "provider failure · stream_error\npayload · json\n" + payload
    )


def _app(tmp_path: Path) -> TUIApp:
    app = TUIApp(
        AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
        history_path=tmp_path / "history",
    )
    app._active_session = app._make_session()
    return app


def test_slash_copy_puts_the_plain_chat_on_the_clipboard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app(tmp_path)
    app._print_user("hello there")
    app._print_system("model: offline")
    app._print_unit(render_event(_error_event()))
    copied: list[str] = []

    def fake_copy(text: str) -> str:
        copied.append(text)
        return "pbcopy"

    monkeypatch.setattr(transcript_export, "copy_to_clipboard", fake_copy)

    output = create_slash_registry().dispatch(app, "/copy")

    assert len(copied) == 1
    text = copied[0]
    session_id = app.loop.store.session_id[:8]
    assert text.startswith(f"zeta · fake · offline · session {session_id}\n\n")
    assert "> hello there\n" in text
    assert "system · model: offline" in text
    assert "reason: " + LONG_REASON in text
    assert not any(glyph in text for glyph in BOX_GLYPHS)
    assert output == f"copied chat: {text.count(chr(10))} lines to the clipboard via pbcopy"


def test_slash_copy_last_scopes_to_the_newest_user_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app(tmp_path)
    app._print_user("first")
    app._print_system("answer one")
    app._print_user("second")
    app._print_system("answer two")
    copied: list[str] = []
    monkeypatch.setattr(
        transcript_export, "copy_to_clipboard", lambda text: copied.append(text) or "pbcopy"
    )

    output = create_slash_registry().dispatch(app, "/copy last")

    assert output.startswith("copied last turn: ")
    assert "> second\n" in copied[0]
    assert "answer two" in copied[0]
    assert "first" not in copied[0].split("\n\n", 1)[1]


def test_slash_copy_falls_back_to_a_session_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app(tmp_path)
    app._print_user("hello")

    def no_clipboard(text: str) -> str:
        raise ClipboardError("no clipboard tool found")

    monkeypatch.setattr(transcript_export, "copy_to_clipboard", no_clipboard)

    output = create_slash_registry().dispatch(app, "/copy")

    written = sorted(app.loop.store.session_dir.glob("transcript-*.txt"))
    assert len(written) == 1
    assert output == (
        f"copied chat: 3 lines to {written[0]} (no clipboard tool found)"
    )
    assert "> hello" in written[0].read_text(encoding="utf-8")


def test_slash_copy_reports_an_empty_chat_and_bad_arguments(tmp_path: Path) -> None:
    app = _app(tmp_path)
    registry = create_slash_registry()

    assert registry.dispatch(app, "/copy") == "copy: nothing to copy yet"
    assert registry.dispatch(app, "/copy everything") == (
        "copy unchanged: use /copy or /copy last"
    )


def test_copy_is_listed_for_completion_and_help() -> None:
    registry = create_slash_registry()

    assert ("copy", "copy the chat as plain text: /copy [last]", "") in (
        registry.completion_entries
    )
    assert "/copy" in registry.help_text()


@pytest.mark.parametrize(
    ("system", "wayland", "available", "expected"),
    [
        ("Darwin", "", {"pbcopy"}, ["/bin/pbcopy"]),
        ("Darwin", "", set(), None),
        ("Windows", "", {"clip"}, ["/bin/clip"]),
        ("Linux", "wayland-0", {"wl-copy", "xclip"}, ["/bin/wl-copy"]),
        ("Linux", "", {"wl-copy", "xclip"}, ["/bin/xclip", "-selection", "clipboard", "-in"]),
        ("Linux", "", {"xsel"}, ["/bin/xsel", "--clipboard", "--input"]),
        ("Linux", "", set(), None),
    ],
)
def test_clipboard_command_picks_the_platform_tool(
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    wayland: str,
    available: set[str],
    expected: list[str] | None,
) -> None:
    monkeypatch.setattr(transcript_export.platform, "system", lambda: system)
    monkeypatch.setattr(
        transcript_export.shutil,
        "which",
        lambda tool: f"/bin/{tool}" if tool in available else None,
    )
    if wayland:
        monkeypatch.setenv("WAYLAND_DISPLAY", wayland)
    else:
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    assert clipboard_command() == expected


def test_copy_to_clipboard_pipes_text_and_reports_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cat = shutil.which("cat")
    sh = shutil.which("sh")
    assert cat is not None and sh is not None

    monkeypatch.setattr(transcript_export, "clipboard_command", lambda: [cat])
    assert copy_to_clipboard("hello") == "cat"

    monkeypatch.setattr(
        transcript_export,
        "clipboard_command",
        lambda: [sh, "-c", "echo nope >&2; exit 3"],
    )
    with pytest.raises(ClipboardError, match="sh exited 3: nope"):
        copy_to_clipboard("hello")

    monkeypatch.setattr(transcript_export, "clipboard_command", lambda: None)
    with pytest.raises(ClipboardError, match="no clipboard tool found"):
        copy_to_clipboard("hello")
