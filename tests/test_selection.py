"""Mouse selection over the transcript, and the clipboard it copies to."""

from __future__ import annotations

import shutil
from io import StringIO
from pathlib import Path

import pytest
from prompt_toolkit.application.current import set_app
from prompt_toolkit.data_structures import Point
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from zeta.core.fake import FakeBackend
from zeta.core.store import ConversationStore
from zeta.loop import AgentLoop
from zeta.tui import app as app_module
from zeta.tui import composer as composer_module
from zeta.tui.app import TUIApp
from zeta.tui.composer import ClipboardError, clipboard_command, copy_to_clipboard
from zeta.tui.key_bindings import MOUSE_OFF
from zeta.tui.render import render_error_card
from zeta.tui.theme import active_palette
from zeta.tui.transcript import TranscriptWidget
from zeta.tui.transcript_search import Selection, highlight_fragments
from zeta.types import ErrorInfo, StreamEvent, StreamEventType

SELECTION_STYLE = f"bg:{active_palette().search_bg}"


def _event(
    event_type: MouseEventType, x: int, y: int, button: MouseButton = MouseButton.LEFT
) -> MouseEvent:
    return MouseEvent(
        position=Point(x=x, y=y),
        event_type=event_type,
        button=button,
        modifiers=frozenset(),
    )


def _drag(
    transcript: TranscriptWidget, start: tuple[int, int], end: tuple[int, int]
) -> object:
    """Press at ``start``, drag to ``end``, release; cells are ``(x, y)``."""

    transcript.mouse_handler(_event(MouseEventType.MOUSE_DOWN, *start))
    transcript.mouse_handler(_event(MouseEventType.MOUSE_MOVE, *end))
    return transcript.mouse_handler(_event(MouseEventType.MOUSE_UP, *end))


def _transcript(*lines: str, height: int = 10) -> tuple[TranscriptWidget, int]:
    transcript = TranscriptWidget()
    for line in lines:
        transcript.append(Text(line))
    transcript.create_content(40, height)
    return transcript, transcript._prefix_lines


def _selected(fragments: list[tuple[str, str]]) -> str:
    return "".join(text for style, text in fragments if SELECTION_STYLE in style)


def _plain(fragments: list[tuple[str, str]]) -> str:
    return "".join(text for _, text in fragments)


def test_selection_geometry_is_inclusive_and_order_independent() -> None:
    for selection in (Selection((1, 2), (2, 4)), Selection((2, 4), (1, 2))):
        assert (selection.start, selection.end) == ((1, 2), (2, 4))
        assert selection.line_span(0, 10) is None
        assert selection.line_span(1, 10) == (2, 10)
        assert selection.line_span(2, 10) == (0, 5)
        assert selection.line_span(3, 10) is None
    assert Selection((0, 5), (0, 5)).is_click
    assert Selection((0, 8), (0, 9)).line_span(0, 4) is None
    assert Selection((0, 0), (0, 9)).line_span(0, 4) == (0, 4)
    assert Selection((0, 0), (0, 0)).released((3, 3)) == Selection((0, 0), (3, 3), False)


def test_selection_text_trims_lines_and_skips_missing_ones() -> None:
    lines = ["line 0", "line 1  ", "", "line 3"]

    def line_text(index: int) -> str | None:
        return lines[index] if 0 <= index < len(lines) else None

    assert Selection((1, 2), (3, 3)).text(line_text) == "ne 1\n\nline"
    assert Selection((-2, 0), (0, 3)).text(line_text) == "line"
    assert Selection((2, 0), (2, 5)).text(line_text) == ""


def test_highlight_fragments_splits_at_the_span_edges() -> None:
    fragments = [("fg:#fff", "hello "), ("bold", "world")]

    assert highlight_fragments(fragments, (2, 8), "bg:#111") == [
        ("fg:#fff", "he"),
        ("fg:#fff bg:#111", "llo "),
        ("bold bg:#111", "wo"),
        ("bold", "rld"),
    ]
    assert highlight_fragments(fragments, (20, 25), "bg:#111") == fragments
    assert highlight_fragments([("", "abc")], (0, 3), "bg:#111") == [("bg:#111", "abc")]


def test_drag_highlights_and_copies_the_covered_text() -> None:
    transcript, prefix = _transcript("line 0", "line 1", "line 2")
    copied: list[str] = []
    transcript.set_copy_handler(lambda text: copied.append(text) or "copied 2 lines")

    result = _drag(transcript, (2, prefix + 1), (4, prefix + 2))

    assert result is None
    assert copied == ["ne 1\nline"]
    assert transcript.copy_notice == "copied 2 lines"
    assert transcript.selection is not None and not transcript.selection.dragging
    content = transcript.create_content(40, 10)
    assert _selected(content.get_line(prefix)) == ""
    assert _selected(content.get_line(prefix + 1)) == "ne 1"
    assert _plain(content.get_line(prefix + 1)) == "line 1"
    assert _selected(content.get_line(prefix + 2)) == "line "


def test_backward_drag_selects_the_same_text() -> None:
    transcript, prefix = _transcript("alpha", "beta", "gamma")
    copied: list[str] = []
    transcript.set_copy_handler(lambda text: copied.append(text) or None)

    _drag(transcript, (2, prefix + 2), (1, prefix))

    assert copied == ["lpha\nbeta\ngam"]
    assert transcript.copy_notice is None


def test_click_without_drag_clears_and_copies_nothing() -> None:
    transcript, prefix = _transcript("line 0", "line 1")
    copied: list[str] = []
    transcript.set_copy_handler(lambda text: copied.append(text) or "copied")

    assert _drag(transcript, (1, prefix), (1, prefix)) is None

    assert copied == []
    assert transcript.selection is None
    assert transcript.selection_text() == ""
    assert transcript.copy_notice is None


def test_transcript_change_drops_the_selection() -> None:
    transcript, prefix = _transcript("line 0", "line 1")
    transcript.set_copy_handler(lambda text: "copied 1 line")
    _drag(transcript, (0, prefix), (3, prefix))
    assert transcript.selection is not None

    transcript.append(Text("line 2"))

    assert transcript.selection is None
    assert transcript.copy_notice is None
    # A move or release with nothing in flight is not ours to handle.
    assert transcript.mouse_handler(_event(MouseEventType.MOUSE_MOVE, 1, 1)) is NotImplemented
    assert transcript.mouse_handler(_event(MouseEventType.MOUSE_UP, 1, 1)) is NotImplemented


def test_wheel_still_scrolls_and_other_buttons_pass_through() -> None:
    transcript, _ = _transcript(*(f"line {index}" for index in range(30)))
    tail = transcript.scroll_offset

    assert transcript.mouse_handler(_event(MouseEventType.SCROLL_UP, 0, 0, MouseButton.NONE)) is None
    assert transcript.scroll_offset == tail - 3
    assert (
        transcript.mouse_handler(_event(MouseEventType.MOUSE_DOWN, 0, 0, MouseButton.RIGHT))
        is NotImplemented
    )
    assert transcript.selection is None


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


def test_session_enables_drag_tracking_without_pointer_motion(tmp_path: Path) -> None:
    app = _app(tmp_path)
    written: list[str] = []
    app._active_session.app.output.write_raw = written.append

    app._active_session.app.output.enable_mouse_support()

    assert "\x1b[?1000h" in written
    assert "\x1b[?1002h" in written  # motion while a button is held
    assert "\x1b[?1003h" not in written  # never the idle pointer stream
    assert b"\x1b[?1002l" in MOUSE_OFF


def test_app_copies_a_selection_to_both_clipboards_and_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app(tmp_path)
    app._print_system("alpha one")
    app._print_system("beta two")
    copied: list[str] = []
    monkeypatch.setattr(
        app_module, "copy_to_clipboard", lambda text: copied.append(text) or "pbcopy"
    )
    session = app._active_session

    with set_app(session.app):
        app._transcript.create_content(60, 10)
        prefix = app._transcript._prefix_lines
        _drag(app._transcript, (9, prefix), (12, prefix + 2))
        footer = "".join(text for _, text in app._status_toolbar())

    assert copied == ["alpha one\n\nsystem · beta"]
    assert session.app.clipboard.get_data().text == "alpha one\n\nsystem · beta"
    assert app._transcript.copy_notice == "copied 3 lines"
    assert "copied 3 lines" in footer


def test_app_reports_a_failed_system_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app(tmp_path)
    app._print_system("alpha one")

    def failing(text: str) -> str:
        raise ClipboardError("no clipboard tool found")

    monkeypatch.setattr(app_module, "copy_to_clipboard", failing)
    session = app._active_session

    with set_app(session.app):
        app._transcript.create_content(60, 10)
        prefix = app._transcript._prefix_lines
        _drag(app._transcript, (9, prefix), (13, prefix))

    assert app._transcript.copy_notice == "copy failed: no clipboard tool found"
    assert session.app.clipboard.get_data().text == "alpha"


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
    monkeypatch.setattr(composer_module.platform, "system", lambda: system)
    monkeypatch.setattr(
        composer_module.shutil,
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

    monkeypatch.setattr(composer_module, "clipboard_command", lambda: [cat])
    assert copy_to_clipboard("hello") == "cat"

    monkeypatch.setattr(
        composer_module,
        "clipboard_command",
        lambda: [sh, "-c", "echo nope >&2; exit 3"],
    )
    with pytest.raises(ClipboardError, match="sh exited 3: nope"):
        copy_to_clipboard("hello")

    monkeypatch.setattr(composer_module, "clipboard_command", lambda: None)
    with pytest.raises(ClipboardError, match="no clipboard tool found"):
        copy_to_clipboard("hello")


def test_error_card_wraps_its_reason_instead_of_cutting_it() -> None:
    rendered = render_error_card(
        StreamEvent(
            StreamEventType.ERROR,
            error=ErrorInfo("http_error", "Anthropic HTTP 404: " + "x" * 1_000),
        )
    )

    assert isinstance(rendered, Panel)
    reason = rendered.renderable.renderables[1]
    assert isinstance(reason, Text)
    assert reason.no_wrap is False
    assert reason.overflow == "fold"
    assert reason.plain.startswith("reason: Anthropic HTTP 404: ")
    assert len(reason.plain) < 420
