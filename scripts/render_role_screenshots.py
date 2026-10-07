"""Render before/after SVGs of the recoloured, non-tool-card TUI surfaces.

Run under ``uv run --frozen python scripts/render_role_screenshots.py``. The
"after" image uses the real gruvbox role colours; the "before" image patches
the role constants back to their pre-change values (error red for warnings,
accent yellow for agent identity, chrome for notices) so the two images show
exactly the call sites this change touched.
"""

from __future__ import annotations

from pathlib import Path

from dataclasses import dataclass

from rich.console import Console, Group
from rich.rule import Rule
from rich.text import Text

from zeta.protocol.types import StreamEvent, StreamEventType, ToolCall, ToolResult
from zeta.tui import theme
from zeta.tui.cards.agent import AgentCard
from zeta.tui.cards.approval_card import render_approval_card
from zeta.tui.render import render_event

OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"
WIDTH = 84


@dataclass(frozen=True)
class Roles:
    """The colours each touched call site used, so before/after stay honest."""

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


def _todo_line(roles: Roles) -> Text:
    return Text.assemble(
        ("[>] ", roles.agent_main),
        ("wire semantic role colors", theme.BODY),
    )


def _agent_list(roles: Roles) -> Group:
    return Group(
        Text("> research · explorer · running", style=roles.agent_child),
        Text("  summarize · general · done", style=theme.DIM),
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
        Rule("tool cards (unchanged)", style=theme.DIM),
        _read_card(),
        _bash_card(),
        _agent_card(),
        Rule("agent identity", style=theme.DIM),
        Text("todo", style=theme.DIM),
        _todo_line(roles),
        Text("subagents", style=theme.DIM),
        _agent_list(roles),
        Rule("severity", style=theme.DIM),
        Text("warning · approval rule 'todo(*)' declares no subject", style=roles.warning),
        Text("[aborted]", style=roles.warning),
        Text("⏺ message from neenerair", style=roles.notice),
        Text("⏺ inspect repository · failed · reason: timeout", style=theme.ERROR),
        Rule("approval", style=theme.DIM),
        render_approval_card("bash", {"command": "rm -rf build"}),
    )


def _render(path: Path, title: str, roles: Roles) -> None:
    console = _console()
    console.print(_scene(roles))
    console.save_svg(str(path), title=title)
    print(f"wrote {path}")


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

    # Pre-change look: identity and approval reused the accent; warnings,
    # notices, and the aborted marker were drawn in error red / chrome.
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
