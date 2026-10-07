"""Render before/after SVGs of transcript search in the full-screen TUI.

Run under ``uv run --frozen python scripts/render_finder_screenshots.py``.

* "before" shows the old inline exact search: matched substrings highlighted in
  place in the transcript with a single footer status line.
* "after" shows the new fzf-style finder overlay: a query line, a ranked result
  list with role labels, turn markers, highlighted excerpts, and a preview of
  the selected message.

Both images use the real gruvbox-dark role colours so the comparison is honest.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console, Group
from rich.text import Text as RichText

from zeta.protocol.types import ToolCall
from zeta.tui import theme
from zeta.tui.transcript import TranscriptWidget
from zeta.tui.transcript.finder_overlay import FinderControl
from zeta.tui.transcript.streaming_text import StreamingText

OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"
WIDTH = 92


def _console() -> Console:
    return Console(
        force_terminal=True,
        color_system="truecolor",
        no_color=False,
        width=WIDTH,
        record=True,
    )


def _seed(transcript: TranscriptWidget) -> None:
    messages = [
        ("user", "run the full pytest suite and fix any failing fuzzy tests"),
        ("assistant", "Running pytest now; two cases in test_fuzzy.py fail on scoring."),
        ("tool", "read src/zeta/tui/fuzzy.py"),
        ("assistant", "The camelCase bonus was double-counted. Patching the DP."),
        ("notice", "context compacted · kept the last 12k tokens"),
        ("user", "re-run just the fuzzy matcher tests please"),
        ("assistant", "pytest tests/test_fuzzy.py — 21 passed in 0.3s. All green."),
        ("tool", "bash pytest -q tests/test_fuzzy.py"),
        ("user", "now wire the finder overlay to Ctrl+F"),
        ("assistant", "Ctrl+F now opens the finder; typing filters messages live."),
    ]
    for index, (role, text) in enumerate(messages):
        if role == "tool":
            name, _, rest = text.partition(" ")
            call = ToolCall(f"c{index}", name, {"target": rest})
            transcript.start_tool(f"c{index}", call, RichText(text))
            transcript.finish_tool(f"c{index}", RichText(text))
        elif role == "assistant":
            streaming = StreamingText(theme.BODY, palette_role="body")
            streaming.append(text)
            transcript.append(streaming)
        else:
            unit = transcript.append(RichText(text))
            if role == "user":
                transcript.mark_user(unit)
    transcript.create_content(WIDTH - 2, 18)


def _before() -> Group:
    match = f"black on {theme.active_palette().accent}"
    dim_match = f"{theme.BODY} on {theme.active_palette().search_bg}"
    lines = [
        RichText("  run the full ", style=theme.BODY).append("pyt", style=match).append(
            "est suite and fix any failing fuzzy tests"
        ),
        RichText("  Running ", style=theme.BODY).append("pyt", style=dim_match).append(
            "est now; two cases in test_fuzzy.py fail on scoring."
        ),
        RichText("  re-run just the fuzzy matcher tests please", style=theme.DIM),
        RichText("  ", style=theme.BODY).append("pyt", style=dim_match).append(
            "est tests/test_fuzzy.py — 21 passed in 0.3s. All green."
        ),
    ]
    footer = RichText("⌕ pyt", style=theme.ACCENT).append(
        "   1/3 matches · n/N to step · Esc to close", style=theme.DIM
    )
    return Group(RichText(""), *lines, RichText(""), footer)


def _to_rich(line: list[tuple[str, str]]) -> RichText:
    rendered = RichText()
    for style, text in line:
        rendered.append(text, style=style or None)
    return rendered


def _after() -> Group:
    transcript = TranscriptWidget()
    _seed(transcript)
    transcript.open_finder()
    transcript.finder_set_query("pyt")
    while not transcript.finder_rank_more():
        pass
    transcript.finder_move(1)
    control = FinderControl(transcript.finder_state)
    content = control.create_content(WIDTH, None)
    lines = [_to_rich(content.get_line(index)) for index in range(content.line_count)]
    return Group(*lines)


def _render(path: Path, title: str, body: Group) -> None:
    console = _console()
    console.print(body)
    console.save_svg(str(path), title=title)
    print(f"wrote {path}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    theme.set_active_palette(theme.GRUVBOX_DARK)
    _render(
        OUT / "tui-fuzzy-finder-before.svg",
        "zeta · inline exact search (before)",
        _before(),
    )
    _render(
        OUT / "tui-fuzzy-finder-after.svg",
        "zeta · fzf-style message finder (after)",
        _after(),
    )


if __name__ == "__main__":
    main()
