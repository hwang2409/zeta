"""Render before/after images of the recoloured, non-tool-card TUI surfaces.

Run under ``uv run --frozen python scripts/render_role_screenshots.py``. The
"after" image uses the real gruvbox role colours wired into the visible chrome
(the main agent reads blue, its subagents read green, warnings are orange and
notices aqua); the "before" image patches those same call sites back to their
pre-change values (agent identity and the status spinner in the accent/chrome,
warnings in error red, notices in chrome) so the two images show exactly the
surfaces this change touches. Tool cards are rendered from the real code in
both images and must look identical.

SVGs are written for a faithful vector record; PNGs are produced with
ImageMagick (``magick``) when it is installed so the pair can embed in a PR.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console, Group
from rich.rule import Rule
from rich.text import Text

from zeta.protocol.types import StreamEvent, StreamEventType, ToolCall, ToolResult
from zeta.tui import theme
from zeta.tui.cards.agent import AgentCard
from zeta.tui.cards.approval_card import render_approval_card
from zeta.tui.render import render_event

OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"
WIDTH = 92


@dataclass(frozen=True)
class Roles:
    """The colours each touched call site uses, so before/after stay honest."""

    agent_main: str
    agent_child: str
    warning: str
    notice: str
    approval_border: str


def _console() -> Console:
    return Console(
        force_terminal=True,
        color_system="truecolor",
        no_color=False,
        width=WIDTH,
        record=True,
    )


def _breadcrumb(roles: Roles) -> Text:
    # main agent (blue) > subagent > subagent (green); separators stay dim.
    return Text.assemble(
        ("main", f"bold {roles.agent_main}"),
        (" > ", theme.CHROME),
        ("research", f"bold {roles.agent_child}"),
        (" > ", theme.CHROME),
        ("inspect theme.py", f"bold {roles.agent_child}"),
    )


def _status_bar(roles: Roles) -> Text:
    # The main agent working: the spinner and loop state read as agent_main.
    bar = Text(no_wrap=True)
    bar.append("gruvbox-dark  ~/zeta  ", style=theme.CHROME)
    bar.append("● streaming  4.2K (2%)", style=roles.agent_main)
    bar.append("    /status · ctrl+c interrupt · ctrl+d quit", style=theme.CHROME)
    return bar


def _assistant_reply() -> Text:
    return Text(
        "I split the role colours into one owner and wired them into the "
        "status bar, the subagent list, and the breadcrumb.",
        style=theme.BODY,
    )


def _todo(roles: Roles) -> Group:
    return Group(
        Text.assemble(("[>] ", roles.agent_main), ("wire role colors", theme.BODY)),
        Text.assemble(("[ ] ", theme.DIM), ("screenshots", theme.DIM)),
    )


def _agent_list(roles: Roles) -> Group:
    # Two running subagents plus one finished one, as the navigator shows them.
    return Group(
        Text("> research · explorer · running", style=f"bold {roles.agent_child}"),
        Text("  inspect · code · running", style=roles.agent_child),
        Text("  summarize · general · done", style=roles.agent_child),
    )


def _read_card():
    call = ToolCall("c-read", "read", {"path": "src/app.py"})
    result = ToolResult("c-read", "def main() -> None:\n    return None\n")
    return render_event(
        StreamEvent(StreamEventType.TOOL_EXECUTION_END, tool_call=call, tool_result=result)
    )


def _bash_card():
    call = ToolCall("c-bash", "bash", {"command": "pytest -q"})
    result = ToolResult("c-bash", "863 passed\n", structured_content={"exit_code": 0})
    return render_event(
        StreamEvent(StreamEventType.TOOL_EXECUTION_END, tool_call=call, tool_result=result)
    )


def _agent_card():
    call = ToolCall("c-agent", "agent", {"description": "inspect repo", "prompt": "look"})
    return AgentCard.render_progress(
        call, "turn 2: reading src/zeta/tui/theme.py", elapsed_seconds=3.4, turns_used=2
    )


def _scene(roles: Roles) -> Group:
    # The approval card reads theme.WARNING internally; set it for this render.
    theme.WARNING = roles.approval_border
    return Group(
        _breadcrumb(roles),
        _status_bar(roles),
        Rule("main agent reply", style=theme.DIM),
        _assistant_reply(),
        Rule("todo", style=theme.DIM),
        _todo(roles),
        Rule("subagents", style=theme.DIM),
        _agent_list(roles),
        Rule("tool cards (unchanged)", style=theme.DIM),
        _read_card(),
        _bash_card(),
        _agent_card(),
        Rule("severity", style=theme.DIM),
        Text("warning · approval rule 'todo(*)' declares no subject", style=roles.warning),
        Text("[aborted]", style=roles.warning),
        Text("⏺ message from neenerair", style=roles.notice),
        Text("⏺ inspect repository · failed · reason: timeout", style=theme.ERROR),
        Rule("approval", style=theme.DIM),
        render_approval_card("bash", {"command": "rm -rf build"}),
    )


def _to_png(svg: Path) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ModuleNotFoundError:
        print(f"(skip PNG for {svg.name}: playwright not installed)")
        return
    png = svg.with_suffix(".png")
    with sync_playwright() as play:
        browser = play.chromium.launch()
        page = browser.new_page(device_scale_factor=2)
        page.goto(svg.resolve().as_uri())
        element = page.query_selector("svg") or page
        element.screenshot(path=str(png))
        browser.close()
    print(f"wrote {png}")


def _render(path: Path, title: str, roles: Roles) -> None:
    console = _console()
    console.print(_scene(roles))
    console.save_svg(str(path), title=title)
    print(f"wrote {path}")
    _to_png(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    theme.set_active_palette(theme.GRUVBOX_DARK)

    after = Roles(
        agent_main=theme.GRUVBOX_DARK.agent_main,
        agent_child=theme.GRUVBOX_DARK.agent_child,
        warning=theme.GRUVBOX_DARK.warning,
        notice=theme.GRUVBOX_DARK.notice,
        approval_border=theme.GRUVBOX_DARK.warning,
    )
    _render(OUT / "tui-role-colors-after.svg", "zeta · gruvbox role colors (after)", after)

    # Pre-change look: agent identity and the status spinner reused the accent
    # or chrome; warnings and the aborted marker were error red; notices chrome.
    before = Roles(
        agent_main=theme.ACCENT,
        agent_child=theme.ACCENT,
        warning=theme.ERROR,
        notice=theme.RECEIPT,
        approval_border=theme.ACCENT,
    )
    _render(
        OUT / "tui-role-colors-before.svg",
        "zeta · before (reddish/accent reuse)",
        before,
    )


if __name__ == "__main__":
    main()
