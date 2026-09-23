"""Agent tool card presentation and state."""

from __future__ import annotations

import os
import re
import time
from collections import deque
from collections.abc import Callable
from difflib import unified_diff
from pathlib import Path, PurePath
from typing import Any, NamedTuple

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from ..agent_receipt import (
    agent_stats,
    ensure_agent_receipt_text,
    has_agent_receipt_suffix,
    terminal_state,
)
from ..core.checkpoints import ConversationIntegrityError, load_session_json
from ..core.session_files import SessionError, open_session_file, session_directory
from ..tools.agent import send_to_run
from ..tools.agent_presets import GENERAL_PRESET, get_agent_preset
from ..types import StreamEvent, StreamEventType, ToolCall, flatten_tool_content
from . import theme

MAX_ARGUMENTS = 140
MAX_RESULT = 180
MAX_TAIL_LINES = 20


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def _readable_argument(value: Any) -> str:
    if isinstance(value, dict):
        pairs = " ".join(
            f"{key}={_readable_argument(nested)}"
            for key, nested in sorted(value.items(), key=lambda item: str(item[0]))
        )
        return "{" + pairs + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_readable_argument(item) for item in value) + "]"
    return str(value).replace("\r\n", " ").replace("\r", " ").replace("\n", " ")


def _arguments(arguments: dict[str, Any]) -> str:
    parts = [
        f"{key}={_readable_argument(arguments[key])}" for key in sorted(arguments)
    ]
    return _truncate(" ".join(parts), MAX_ARGUMENTS)


def _read_lifecycle(path: str) -> dict[str, Any]:
    if not path:
        return {}
    try:
        value = load_session_json(Path(path) / "agent_lifecycle.json")
    except ConversationIntegrityError:
        return {}
    return value if type(value) is dict else {}


class AgentCard:
    """Own one agent card's rendering state, including its bounded tail."""

    def __init__(self, call: ToolCall) -> None:
        self.call = call
        self._supported = call.name.casefold() == "agent"
        self._disclosure_supported = not self._supported
        self._output: list[str] = []
        self._finished = False
        self._started_at = time.monotonic()
        self._elapsed_seconds = 0.0
        self._turns = 0
        self._child_session_path = ""
        self._expanded = False
        self._receipt: RenderableType | None = None
        self._depth = 1

    @property
    def supported(self) -> bool:
        return self._supported

    @property
    def active(self) -> bool:
        return self._supported and not self._finished

    @classmethod
    def _description(cls, call: ToolCall) -> str:
        description = call.arguments.get("description")
        return str(description) if description is not None else call.name

    @classmethod
    def _agent_type(cls, call: ToolCall) -> str:
        preset = get_agent_preset(call.arguments.get("agent_type"))
        if preset is None or preset.name == GENERAL_PRESET.name:
            return ""
        return preset.name

    @classmethod
    def _turns_from_content(cls, content: str) -> int:
        turns = [
            int(match.group(1))
            for match in re.finditer(r"\bturn\s+(\d+)\b", content, re.IGNORECASE)
        ]
        return max(turns, default=0)

    @classmethod
    def _step(cls, content: str) -> str:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            return "thinking"
        line = lines[-1]
        line = re.sub(r"^↳\s+(?:\[stdout\]|\[stderr\])\s+", "", line)
        line = re.sub(r"^.*?:\s+turn\s+\d+:\s+", "", line, flags=re.IGNORECASE)
        line = re.sub(r"^turn\s+\d+:\s+", "", line, flags=re.IGNORECASE)
        return _truncate(line, MAX_RESULT)

    @classmethod
    def _header(
        cls,
        call: ToolCall,
        *,
        elapsed_seconds: float,
        turns_used: int,
        depth: int = 1,
        expanded: bool = False,
    ) -> Text:
        affordance = "collapse: ctrl+x ctrl+o" if expanded else "expand: ctrl+x ctrl+o"
        agent_type = cls._agent_type(call)
        prefix = f"{agent_type} · " if agent_type else ""
        return Text(
            f"{prefix}{cls._description(call)} · {elapsed_seconds:.1f}s · "
            f"{turns_used} turns · depth {depth} · {affordance}",
            style=theme.COMMAND,
            no_wrap=True,
            overflow="ellipsis",
        )

    @classmethod
    def render_progress(
        cls,
        call: ToolCall,
        content: str,
        *,
        elapsed_seconds: float = 0.0,
        turns_used: int | None = None,
        depth: int = 1,
    ) -> Panel | None:
        if call.name.casefold() != "agent":
            return None
        turns = cls._turns_from_content(content) if turns_used is None else turns_used
        body = Text(cls._step(content), style=theme.BODY, no_wrap=True, overflow="ellipsis")
        return Panel(
            Group(
                cls._header(
                    call,
                    elapsed_seconds=elapsed_seconds,
                    turns_used=turns,
                    depth=depth,
                ),
                body,
            ),
            border_style=theme.CARD_BORDER,
            style=theme.CARD_BG,
            padding=(0, 1),
            expand=True,
        )

    @classmethod
    def _tail_lines(
        cls,
        child_session_path: str,
        limit: int,
        seen: set[str] | None = None,
    ) -> list[str]:
        seen = set() if seen is None else seen
        if child_session_path in seen or limit < 1:
            return []
        seen.add(child_session_path)
        path = Path(child_session_path) / "conversation.jsonl"
        lines: deque[str] = deque(maxlen=limit)
        try:
            with session_directory(path.parent.parent, path.parent.name) as (_, directory_fd), os.fdopen(open_session_file(directory_fd, path.name, os.O_RDONLY), "rb") as handle:
                for raw_line in handle:
                    try:
                        row = load_session_json(raw_line)
                    except ConversationIntegrityError:
                        continue
                    if not isinstance(row, dict) or row.get("type") != "message":
                        continue
                    data = row.get("data")
                    message = data.get("message") if isinstance(data, dict) else None
                    if not isinstance(message, dict):
                        continue
                    role = message.get("role", "message")
                    content = message.get("content")
                    if not isinstance(content, list):
                        continue
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        block_type = block.get("type")
                        if block_type == "text" and isinstance(block.get("text"), str):
                            for text_line in block["text"].splitlines() or [""]:
                                lines.append(f"{role}: {text_line}")
                        elif block_type == "tool_use" and isinstance(block.get("tool_call"), dict):
                            tool_call = block["tool_call"]
                            name = tool_call.get("name", "tool")
                            arguments = tool_call.get("arguments", {})
                            if isinstance(name, str) and isinstance(arguments, dict):
                                lines.append(f"tool: {name} {_arguments(arguments)}")
                    tool_result = message.get("tool_result")
                    structured = (
                        tool_result.get("structured_content")
                        if isinstance(tool_result, dict)
                        else None
                    )
                    nested_path = (
                        structured.get("child_session_path")
                        if isinstance(structured, dict)
                        else None
                    )
                    if isinstance(nested_path, str) and nested_path:
                        nested_tail = cls._tail_lines(
                            nested_path,
                            max(1, limit - len(lines)),
                            seen,
                        )
                        lines.extend(f"  {line}" for line in nested_tail)
        except (OSError, SessionError):
            return []
        return list(lines)

    @classmethod
    def render_expanded(
        cls,
        call: ToolCall,
        *,
        elapsed_seconds: float,
        turns_used: int,
        child_session_path: str,
        limit: int = MAX_TAIL_LINES,
        depth: int = 1,
    ) -> Panel | None:
        if call.name.casefold() != "agent":
            return None
        tail = cls._tail_lines(child_session_path, limit)
        body = Text(
            "\n".join(tail) if tail else "child transcript unavailable",
            style=theme.BODY if tail else theme.DIM,
            overflow="ellipsis",
            no_wrap=True,
        )
        return Panel(
            Group(
                cls._header(
                    call,
                    elapsed_seconds=elapsed_seconds,
                    turns_used=turns_used,
                    depth=depth,
                    expanded=True,
                ),
                body,
            ),
            border_style=theme.CARD_BORDER,
            style=theme.CARD_BG,
            padding=(0, 1),
            expand=True,
        )

    @classmethod
    def render_receipt(
        cls,
        event: StreamEvent,
        *,
        elapsed_seconds: float | None = None,
        turns_used: int | None = None,
        depth: int | None = None,
    ) -> Text | None:
        call = event.tool_call
        result = event.tool_result
        if call is None or result is None or call.name.casefold() != "agent":
            return None
        structured = result.structured_content or {}
        child_path = structured.get("child_session_path")
        lifecycle = _read_lifecycle(child_path if type(child_path) is str else "")
        if lifecycle:
            lifecycle_elapsed = lifecycle.get("elapsed")
            if type(lifecycle_elapsed) in {int, float} and lifecycle_elapsed >= 0:
                elapsed_seconds = float(lifecycle_elapsed)
        elapsed = elapsed_seconds
        if elapsed is None:
            value = event.data.get("elapsed_seconds")
            if isinstance(value, (int, float)):
                elapsed = max(0.0, float(value))
            else:
                value = event.data.get("elapsed_ms")
                elapsed = max(0.0, float(value) / 1000) if isinstance(value, (int, float)) else 0.0
        turns = turns_used
        event_depth = event.data.get("depth")
        display_depth = 1
        if depth is not None:
            display_depth = depth
        elif type(event_depth) is int and event_depth >= 1:
            display_depth = event_depth
        if turns is None and result.structured_content is not None:
            value = result.structured_content.get("turns_used")
            turns = value if type(value) is int and value >= 0 else 0
        if result.structured_content is not None:
            value = result.structured_content.get("depth")
            if (
                depth is None
                and not (type(event_depth) is int and event_depth >= 1)
                and type(value) is int
                and value >= 1
            ):
                display_depth = value
        turns = turns or 0
        structured_status = structured.get("status")
        receipt_status = terminal_state(
            error=result.is_error,
            canceled=result.is_canceled,
            status=(
                structured_status
                if structured_status in {"completed", "error", "canceled"}
                else None
            ),
        )
        status = (
            structured_status
            if structured_status in {"completed", "error", "canceled"}
            else "canceled"
            if receipt_status == "canceled"
            else "fail"
            if receipt_status == "failed"
            else "ok"
        )
        stats = agent_stats(
            lifecycle,
            status=receipt_status,
            turns_used=turns,
        )
        receipt_text = ensure_agent_receipt_text(
            result.content,
            receipt_status,
            stats,
        )
        if has_agent_receipt_suffix(result.content):
            return Text(
                receipt_text,
                style=theme.ERROR if receipt_status in {"failed", "canceled"} else theme.RECEIPT,
                no_wrap=True,
                overflow="ellipsis",
            )
        agent_type = cls._agent_type(call)
        prefix = f"{agent_type} · " if agent_type else ""
        return Text(
            f"{prefix}{cls._description(call)} · {turns} turns · "
            f"{max(0.0, elapsed or 0.0):.1f}s · {status} · "
            f"depth {display_depth} · {receipt_text} · expand: ctrl+x ctrl+o",
            style=theme.ERROR if receipt_status in {"failed", "canceled"} else theme.RECEIPT,
            no_wrap=True,
            overflow="ellipsis",
        )

    @classmethod
    def render_start(cls, event: StreamEvent) -> RenderableType | None:
        call = event.tool_call
        if event.type is not StreamEventType.TOOL_EXECUTION_START or call is None:
            return None
        depth = event.data.get("depth")
        return cls.render_progress(
            call,
            "",
            depth=depth if type(depth) is int and depth >= 1 else 1,
        )

    def start(self, event: StreamEvent) -> None:
        if event.type is not StreamEventType.TOOL_EXECUTION_START:
            return
        depth = event.data.get("depth")
        if type(depth) is int and depth >= 1:
            self._depth = depth

    @classmethod
    def render_end(cls, event: StreamEvent) -> RenderableType | None:
        if event.type is not StreamEventType.TOOL_EXECUTION_END:
            return None
        return cls.render_receipt(event)

    def _elapsed(self) -> float:
        return max(0.0, time.monotonic() - self._started_at)

    def _progress(self) -> Panel | None:
        return type(self).render_progress(
            self.call,
            "\n".join(self._output),
            elapsed_seconds=self._elapsed(),
            turns_used=self._turns,
            depth=self._depth,
        )

    def current(self) -> RenderableType | None:
        return self._progress() if self.active else None

    def set_child_session_path(self, path: str) -> None:
        self._child_session_path = path

    def update(self, rendered: RenderableType, event: StreamEvent | None = None) -> RenderableType | None:
        if not self._supported:
            return None
        text = getattr(rendered, "plain", None)
        if isinstance(text, str):
            self._output.append(text)
            self._turns = max(self._turns, self._turns_from_content("\n".join(self._output)))
        if event is not None:
            path = event.data.get("child_session_path")
            if isinstance(path, str) and path:
                self._child_session_path = path
            depth = event.data.get("depth")
            if type(depth) is int and depth >= 1:
                self._depth = depth
        if self._expanded:
            return self._expanded_render()
        return self._progress()

    def refresh(self) -> RenderableType | None:
        if not self.active:
            return None
        return self._expanded_render() if self._expanded else self._progress()

    def finish(
        self,
        event: StreamEvent | None,
        rendered: RenderableType | None = None,
    ) -> RenderableType | None:
        if event is None:
            return None
        if not self._supported and not self._disclosure_supported:
            return None
        if not self._supported:
            if rendered is None or not isinstance(rendered, Panel):
                return None
            self._receipt = rendered
            return _compact_tool_card(rendered)
        self._finished = True
        self._elapsed_seconds = self._elapsed()
        result = event.tool_result
        structured = result.structured_content if result is not None else None
        turns = structured.get("turns_used") if structured else None
        if type(turns) is int and turns >= 0:
            self._turns = turns
        depth = structured.get("depth") if structured else None
        if type(depth) is int and depth >= 1:
            self._depth = depth
        path = structured.get("child_session_path") if structured else None
        if isinstance(path, str):
            self._child_session_path = path
        self._receipt = type(self).render_receipt(
            event,
            elapsed_seconds=self._elapsed_seconds,
            turns_used=self._turns,
            depth=self._depth,
        )
        return self._expanded_render() if self._expanded else self._receipt

    def _expanded_render(self) -> Panel | None:
        return type(self).render_expanded(
            self.call,
            elapsed_seconds=self._elapsed_seconds if self._finished else self._elapsed(),
            turns_used=self._turns,
            child_session_path=self._child_session_path,
            depth=self._depth,
        )

    def toggle(self) -> RenderableType | None:
        if not self._supported and not self._disclosure_supported:
            return None
        if not self._supported:
            if self._receipt is None:
                return None
            self._expanded = not self._expanded
            return self._receipt if self._expanded else _compact_tool_card(self._receipt)
        self._expanded = not self._expanded
        if self._expanded:
            return self._expanded_render()
        return self._receipt if self._finished else self._progress()


def render_agent_progress(
    call: ToolCall,
    content: str,
    *,
    elapsed_seconds: float = 0.0,
    turns_used: int | None = None,
    depth: int = 1,
) -> Panel | None:
    return AgentCard.render_progress(
        call,
        content,
        elapsed_seconds=elapsed_seconds,
        turns_used=turns_used,
        depth=depth,
    )


def render_agent_expanded(
    call: ToolCall,
    *,
    elapsed_seconds: float,
    turns_used: int,
    child_session_path: str,
    limit: int = MAX_TAIL_LINES,
    depth: int = 1,
) -> Panel | None:
    return AgentCard.render_expanded(
        call,
        elapsed_seconds=elapsed_seconds,
        turns_used=turns_used,
        child_session_path=child_session_path,
        limit=limit,
        depth=depth,
    )


def render_agent_receipt(
    event: StreamEvent,
    *,
    elapsed_seconds: float | None = None,
    turns_used: int | None = None,
    depth: int | None = None,
) -> Text | None:
    return AgentCard.render_receipt(
        event,
        elapsed_seconds=elapsed_seconds,
        turns_used=turns_used,
        depth=depth,
    )


class AgentRunCommandMixin:
    """Let the user list and steer live agent runs from the composer.

    Lives here rather than in app.py, which is at the module line cap, and
    reaches a run through the same seam the agent_send tool uses.
    """

    def slash_runs(self, args: str) -> str:
        del args
        children = self.loop.store.agent_children()
        runs = [
            (marker_key, marker)
            for marker_key, marker in children.items()
            if marker.get("background") and marker.get("agent_type") == "run"
        ]
        if not runs:
            return "no live runs"
        lines = []
        for marker_key, marker in sorted(runs):
            turns = marker.get("turns_used", 0)
            lines.append(
                f"{marker_key}  {marker['description']}  ({turns} turns)"
            )
        return "\n".join(lines)

    def slash_send(self, args: str) -> str:
        parts = args.strip().split(maxsplit=1)
        if len(parts) != 2:
            return "use /send <run-id> <message>; /runs lists the live ones"
        run_id, message = parts
        error = send_to_run(self.loop.store, run_id, message)
        if error is not None:
            return error
        return f"queued for {run_id}; it arrives at the run's next turn boundary"


# Per-tool cards share this module with AgentCard to keep the TUI module count
# within the repository limit. New cards only need one registry declaration.
ToolCardRenderer = Callable[[StreamEvent, bool], RenderableType]
TOOL_CARD_REGISTRY: dict[str, ToolCardRenderer] = {}
LANGUAGE_BY_EXTENSION = {
    ".bash": "bash",
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".css": "css",
    ".go": "go",
    ".h": "c",
    ".hpp": "cpp",
    ".html": "html",
    ".ini": "ini",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "jsx",
    ".json": "json",
    ".md": "markdown",
    ".py": "python",
    ".rs": "rust",
    ".sh": "bash",
    ".sql": "sql",
    ".toml": "toml",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".xml": "xml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".zsh": "bash",
}
MAX_CARD_LINES = 15
MAX_CARD_COLUMNS = 240
MAX_TOOL_SCAN_LINES = MAX_CARD_LINES * 4
MAX_TOOL_SCAN_BYTES = 64 * 1024


class _BoundedToolOutput(NamedTuple):
    lines: tuple[str, ...]
    total_lines: int | None
    truncated: bool


def _scan_tool_output(content: str) -> _BoundedToolOutput:
    """Read only a bounded prefix of tool output without building line lists."""

    lines: list[str] = []
    start = 0
    scan_end = min(len(content), MAX_TOOL_SCAN_BYTES)
    truncated = False
    while start < len(content) and len(lines) < MAX_TOOL_SCAN_LINES:
        newline = content.find("\n", start, scan_end)
        if newline < 0:
            end = scan_end
            line = content[start:end]
            lines.append(line.removesuffix("\r")[: MAX_RESULT + 1])
            if end < len(content):
                truncated = True
            start = len(content)
            break
        lines.append(content[start:newline].removesuffix("\r")[: MAX_RESULT + 1])
        start = newline + 1
    if start < len(content):
        truncated = True
    return _BoundedToolOutput(
        tuple(lines),
        None if truncated else len(lines),
        truncated,
    )


def _compact_tool_card(rendered: RenderableType) -> RenderableType:
    if not isinstance(rendered, Panel):
        return rendered
    content = rendered.renderable
    if isinstance(content, Group) and content.renderables:
        header = content.renderables[0]
    elif isinstance(content, Text):
        header = content.split("\n")[0]
    else:
        return rendered
    if isinstance(header, Text):
        header = header.copy()
        header.append(" · expand: ctrl+x ctrl+o", style=theme.DIM)
    else:
        header = Group(
            header,
            Text("expand: ctrl+x ctrl+o", style=theme.DIM),
        )
    return Panel(
        header,
        border_style=rendered.border_style,
        style=rendered.style,
        padding=rendered.padding,
        expand=rendered.expand,
    )


def _base_render():
    from . import render

    return render


def register_tool_card(*tool_names: str) -> Callable[[ToolCardRenderer], ToolCardRenderer]:
    """Register one renderer for one or more normalized tool names."""

    def register(renderer: ToolCardRenderer) -> ToolCardRenderer:
        for tool_name in tool_names:
            TOOL_CARD_REGISTRY[tool_name.strip().lower()] = renderer
        return renderer

    return register


def infer_language(path: str) -> str:
    """Return the syntax lexer for a path, with text as the safe fallback."""

    return LANGUAGE_BY_EXTENSION.get(PurePath(path).suffix.lower(), "text")


def _card_path(call: ToolCall, result: Any | None = None) -> str:
    arguments_path = call.arguments.get("path")
    if isinstance(arguments_path, str) and arguments_path:
        return arguments_path
    structured = getattr(result, "structured_content", None)
    result_path = structured.get("path") if isinstance(structured, dict) else None
    return result_path if isinstance(result_path, str) else "<unknown>"


def _read_content(event: StreamEvent) -> str:
    """Return text for a read card while preserving image read receipts."""

    result = event.tool_result
    blocks = result.content_blocks if result is not None else None
    structured = result.structured_content if result is not None else None
    if (
        blocks
        and any(block.get("type") == "image" for block in blocks)
        and isinstance(structured, dict)
        and structured.get("format") in {"png", "jpeg", "gif", "webp"}
    ):
        return flatten_tool_content(
            [block for block in blocks if block.get("type") == "text"]
        )
    return _base_render()._tool_content(event)


def _card_line_count(value: str) -> int:
    if not value:
        return 0
    return value.count("\n") + (0 if value.endswith("\n") else 1)


def _bounded_card_lines(value: str) -> tuple[list[str], int]:
    visible: list[str] = []
    start = 0
    while start < len(value) and len(visible) < MAX_CARD_LINES:
        end = value.find("\n", start)
        if end < 0:
            line = value[start:]
            start = len(value)
        else:
            line = value[start:end]
            start = end + 1
        line = line.removesuffix("\r")
        visible.append(line[:MAX_CARD_COLUMNS])
    total = _card_line_count(value)
    return visible, max(0, total - len(visible))


def _read_header(call: ToolCall, content: str, result: Any | None) -> Text:
    path = _card_path(call, result)
    offset = call.arguments.get("offset", 0)
    line_start = offset + 1 if type(offset) is int and offset >= 0 else 1
    line_count = max(1, _card_line_count(content))
    line_end = line_start + line_count - 1
    return Text.assemble(
        (call.name, theme.COMMAND),
        (f" {path}", theme.BODY),
        (f" · lines {line_start}-{line_end}", theme.DIM),
    )


def _read_tool_card(event: StreamEvent, running: bool) -> RenderableType:
    base = _base_render()
    call = event.tool_call
    if call is None:
        return base._tool_card(event, running=running)
    if running or event.tool_result is None:
        return Text(
            f"⏺ {call.name} {_card_path(call)} · running",
            style=theme.RECEIPT,
            overflow="ellipsis",
            no_wrap=True,
        )
    result = event.tool_result
    image_read = (
        result.content_blocks
        and any(block.get("type") == "image" for block in result.content_blocks)
        and isinstance(result.structured_content, dict)
        and result.structured_content.get("format") in {"png", "jpeg", "gif", "webp"}
    )
    if result.is_error or (
        any(block.get("type") != "text" for block in result.content_blocks or [])
        and not image_read
    ):
        return base._tool_card(event)
    content = _read_content(event)
    raw_visible, omitted = _bounded_card_lines(content)
    visible = [base._strip_terminal_controls(line) for line in raw_visible]
    syntax = Syntax(
        "\n".join(visible),
        infer_language(_card_path(call, result)),
        theme=theme.CODE_THEME,
        word_wrap=True,
        background_color="default",
    )
    body: RenderableType = syntax
    if omitted:
        body = Group(
            syntax,
            Text(f"… +{omitted} lines", style=theme.AFFORDANCE),
        )
    return base._tool_panel(
        call,
        body,
        header=_read_header(call, content, result),
    )


def _cap_diff_lines(lines: list[str]) -> tuple[list[str], int]:
    visible = [line[:MAX_CARD_COLUMNS] for line in lines[:MAX_CARD_LINES]]
    return visible, max(0, len(lines) - len(visible))


DIFF_CONTEXT_LINES = 3


def _iter_card_lines(value: str, *, reverse: bool = False):
    if not reverse:
        start = 0
        while start < len(value):
            end = value.find("\n", start)
            if end < 0:
                end = len(value)
            yield value[start:end].removesuffix("\r")
            start = end + 1
        return

    end = len(value)
    if end and value[end - 1] == "\n":
        end -= 1
    while end > 0:
        start = value.rfind("\n", 0, end)
        yield value[start + 1 : end].removesuffix("\r")
        if start < 0:
            return
        end = start


def _common_line_prefix(old: str, new: str, limit: int) -> int:
    count = 0
    for old_line, new_line in zip(
        _iter_card_lines(old), _iter_card_lines(new), strict=False
    ):
        if old_line != new_line:
            break
        count += 1
        if count >= limit:
            break
    return count


def _common_line_suffix(old: str, new: str, limit: int) -> int:
    count = 0
    for old_line, new_line in zip(
        _iter_card_lines(old, reverse=True),
        _iter_card_lines(new, reverse=True),
        strict=False,
    ):
        if old_line != new_line:
            break
        count += 1
        if count >= limit:
            break
    return count


def _card_line_window(value: str, start: int, end: int) -> list[str]:
    lines: list[str] = []
    for index, line in enumerate(_iter_card_lines(value)):
        if index >= end:
            break
        if index >= start:
            lines.append(line[:MAX_CARD_COLUMNS])
            if len(lines) >= MAX_CARD_LINES:
                break
    return lines


def _diff_window_bounds(total: int, change_start: int, change_end: int) -> tuple[int, int]:
    if total == 0:
        return 0, 0
    start = max(0, change_start - DIFF_CONTEXT_LINES)
    preview_end = min(change_end, start + MAX_CARD_LINES - DIFF_CONTEXT_LINES)
    end = min(total, max(start + 1, preview_end + DIFF_CONTEXT_LINES))
    return start, min(end, start + MAX_CARD_LINES)


def _bounded_unified_diff(old: str, new: str, path: str) -> tuple[list[str], int]:
    old_count = _card_line_count(old)
    new_count = _card_line_count(new)
    prefix = _common_line_prefix(old, new, min(old_count, new_count))
    suffix_limit = min(old_count - prefix, new_count - prefix)
    suffix = _common_line_suffix(old, new, suffix_limit)
    old_change_end = old_count - suffix
    new_change_end = new_count - suffix
    if prefix == old_change_end == new_change_end:
        return [], 0

    old_start, old_end = _diff_window_bounds(old_count, prefix, old_change_end)
    new_start, new_end = _diff_window_bounds(new_count, prefix, new_change_end)
    old_window = _card_line_window(old, old_start, old_end)
    new_window = _card_line_window(new, new_start, new_end)
    lines = list(
        unified_diff(
            old_window,
            new_window,
            fromfile=path,
            tofile=path,
            lineterm="",
        )
    )
    if old_start or new_start:
        lines.insert(3, "  … unchanged lines omitted")
    if old_end < old_change_end or new_end < new_change_end or suffix:
        lines.append("  … unchanged lines omitted")
    return _cap_diff_lines(lines)


def _diff_from_structured(result: Any, path: str) -> tuple[list[str], str, int] | None:
    structured = result.structured_content
    if not isinstance(structured, dict):
        return None
    for key in ("diff", "unified_diff"):
        value = structured.get(key)
        if isinstance(value, str):
            lines, omitted = _bounded_card_lines(value)
            return lines, "structured diff", omitted
    old = next(
        (structured.get(key) for key in ("old_content", "before", "pre_image")),
        None,
    )
    new = next(
        (structured.get(key) for key in ("new_content", "after")),
        None,
    )
    if isinstance(old, str) and isinstance(new, str):
        visible, omitted = _bounded_unified_diff(old, new, path)
        return visible, "structured pre-image", omitted
    return None


def _diff_lines(event: StreamEvent, path: str) -> tuple[list[str], str, int]:
    call = event.tool_call
    result = event.tool_result
    assert call is not None and result is not None
    if call.name.strip().lower() == "edit":
        old = call.arguments.get("old_string")
        new = call.arguments.get("new_string")
        if isinstance(old, str) and isinstance(new, str):
            visible, omitted = _bounded_unified_diff(old, new, path)
            return visible, "", omitted
    structured_diff = _diff_from_structured(result, path)
    if structured_diff is not None:
        return structured_diff
    content = call.arguments.get("content")
    if not isinstance(content, str):
        return [], "", 0
    content_lines, content_omitted = _bounded_card_lines(content)
    if not content_lines:
        return [], "", 0
    total = len(content_lines) + content_omitted
    lines = [
        "--- /dev/null",
        f"+++ {path}",
        f"@@ -0,0 +1,{total} @@",
        *(f"+{line}" for line in content_lines),
    ]
    visible = [line[:MAX_CARD_COLUMNS] for line in lines[:MAX_CARD_LINES]]
    omitted = total + 3 - len(visible)
    return visible, "new file · pre-image unavailable", omitted


def _render_diff(lines: list[str], note: str, omitted: int = 0) -> Text:
    rendered = Text(overflow="ellipsis", no_wrap=True)
    base = _base_render()
    for index, line in enumerate(lines):
        if index:
            rendered.append("\n")
        if line.startswith("+"):
            style = theme.DIFF_ADD
        elif line.startswith("-"):
            style = theme.DIFF_REMOVE
        elif line.startswith(("@@", " ")):
            style = theme.DIFF_CONTEXT
        else:
            style = theme.DIM
        rendered.append(
            base._strip_terminal_controls(line)[:MAX_CARD_COLUMNS], style=style
        )
    if omitted > 0:
        if rendered:
            rendered.append("\n")
        rendered.append(f"… +{omitted} diff lines", style=theme.AFFORDANCE)
    if note:
        if rendered:
            rendered.append("\n")
        rendered.append(note, style=theme.DIM)
    return rendered


@register_tool_card("write", "edit")
def _write_edit_tool_card(event: StreamEvent, running: bool) -> RenderableType:
    base = _base_render()
    call = event.tool_call
    if call is None:
        return base._tool_card(event, running=running)
    if running or event.tool_result is None:
        return base._tool_panel(call, Text("running…", style=theme.DIM))
    result = event.tool_result
    if result.is_error:
        return base._tool_card(event)
    path = _card_path(call, result)
    lines, note, omitted = _diff_lines(event, path)
    if not lines:
        return base._tool_card(event)
    header = Text.assemble(
        (call.name, theme.COMMAND),
        (f" {path}", theme.BODY),
        (" · diff", theme.DIM),
    )
    return base._tool_panel(
        call,
        _render_diff(lines, note, omitted),
        header=header,
    )


@register_tool_card("read")
def _registered_read_tool_card(event: StreamEvent, running: bool) -> RenderableType:
    return _read_tool_card(event, running)
