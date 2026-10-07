"""Representative tool-card renders used to guard byte-identical output.

The gruvbox role-color work must not change how tool call/result cards look.
These helpers render a fixed set of tool cards under the ``gruvbox-dark``
palette to a deterministic ANSI string so a regression test can assert the
bytes stay identical to ``origin/main`` (5d8adf6e).
"""

from __future__ import annotations

from collections import OrderedDict

from rich.console import Console

from zeta.protocol.types import StreamEvent, StreamEventType, ToolCall, ToolResult
from zeta.tui import theme
from zeta.tui.cards.agent import AgentCard
from zeta.tui.render import render_event

SNAPSHOT_WIDTH = 80


def _console() -> Console:
    return Console(
        force_terminal=True,
        color_system="truecolor",
        no_color=False,
        width=SNAPSHOT_WIDTH,
    )


def _ansi(renderable: object) -> str:
    console = _console()
    with console.capture() as capture:
        console.print(renderable)
    return capture.get()


def _completed(call: ToolCall, result: ToolResult) -> StreamEvent:
    return StreamEvent(
        StreamEventType.TOOL_EXECUTION_END,
        tool_call=call,
        tool_result=result,
    )


def _render(call: ToolCall, result: ToolResult) -> str:
    rendered = render_event(_completed(call, result))
    assert rendered is not None
    return _ansi(rendered)


def representative_cards() -> OrderedDict[str, str]:
    """Render the frozen tool-card set to ANSI under the gruvbox palette."""

    previous = theme.active_palette()
    theme.set_active_palette(theme.GRUVBOX_DARK)
    try:
        cards: OrderedDict[str, str] = OrderedDict()

        read_call = ToolCall("c-read", "read", {"path": "src/app.py"})
        read_result = ToolResult(
            "c-read",
            "def main() -> None:\n    return None\n",
        )
        cards["read"] = _render(read_call, read_result)

        edit_call = ToolCall(
            "c-edit",
            "edit",
            {
                "path": "src/app.py",
                "old_string": "def main() -> None:\n    return None\n",
                "new_string": "def main() -> int:\n    return 0\n",
            },
        )
        edit_result = ToolResult("c-edit", "edited src/app.py")
        cards["edit"] = _render(edit_call, edit_result)

        bash_call = ToolCall("c-bash", "bash", {"command": "echo hi"})
        bash_result = ToolResult(
            "c-bash",
            "hi\n",
            structured_content={"exit_code": 0},
        )
        cards["bash"] = _render(bash_call, bash_result)

        bash_fail_call = ToolCall("c-bash2", "bash", {"command": "false"})
        bash_fail_result = ToolResult(
            "c-bash2",
            "boom\n",
            is_error=True,
            structured_content={"exit_code": 1},
        )
        cards["bash_failed"] = _render(bash_fail_call, bash_fail_result)

        websearch_call = ToolCall("c-web", "websearch", {"query": "rich library"})
        websearch_result = ToolResult(
            "c-web",
            "1 matches\nresult line\n",
        )
        cards["websearch"] = _render(websearch_call, websearch_result)

        fetch_call = ToolCall("c-fetch", "fetch", {"url": "https://example.com"})
        fetch_result = ToolResult("c-fetch", "fetched 10 lines of text\n")
        cards["fetch"] = _render(fetch_call, fetch_result)

        mcp_call = ToolCall(
            "c-mcp",
            "mcp__files__list",
            {"directory": "/tmp"},
        )
        mcp_result = ToolResult("c-mcp", "a.txt\nb.txt\n")
        cards["mcp_generic"] = _render(mcp_call, mcp_result)

        todo_call = ToolCall("c-todo", "todo", {})
        todo_result = ToolResult(
            "c-todo",
            "updated",
            structured_content={
                "counts": {"in_progress": 1, "completed": 2},
                "items": [
                    {"content": "wire roles", "status": "in_progress"},
                    {"content": "write tests", "status": "completed"},
                ],
            },
        )
        cards["todo"] = _render(todo_call, todo_result)

        skill_call = ToolCall("c-skill", "skill", {"name": "frontend-design"})
        skill_result = ToolResult("c-skill", "prompt loaded")
        cards["skill"] = _render(skill_call, skill_result)

        agent_call = ToolCall(
            "c-agent",
            "agent",
            {"description": "inspect repo", "prompt": "look"},
        )
        agent_progress = AgentCard.render_progress(
            agent_call,
            "turn 1: reading files",
            elapsed_seconds=1.2,
            turns_used=1,
            depth=1,
        )
        assert agent_progress is not None
        cards["agent_progress"] = _ansi(agent_progress)

        return cards
    finally:
        theme.set_active_palette(previous)
