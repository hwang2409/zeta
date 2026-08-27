"""Agent tool card presentation and state."""

from __future__ import annotations

import json
import re
import time
from collections import deque
from pathlib import Path
from typing import Any

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.text import Text

from ..types import StreamEvent, StreamEventType, ToolCall
from .theme import BODY, CARD_BG, CARD_BORDER, COMMAND, DIM, ERROR, RECEIPT


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


class AgentCard:
    """Own one agent card's rendering state, including its bounded tail."""

    def __init__(self, call: ToolCall) -> None:
        self.call = call
        self._supported = call.name.casefold() == "agent"
        self._output: list[str] = []
        self._finished = False
        self._started_at = time.monotonic()
        self._elapsed_seconds = 0.0
        self._turns = 0
        self._child_session_path = ""
        self._expanded = False
        self._receipt: RenderableType | None = None

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
        expanded: bool = False,
    ) -> Text:
        affordance = "collapse: ctrl+x ctrl+o" if expanded else "expand: ctrl+x ctrl+o"
        return Text(
            f"{cls._description(call)} · {elapsed_seconds:.1f}s · "
            f"{turns_used} turns · {affordance}",
            style=COMMAND,
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
    ) -> Panel | None:
        if call.name.casefold() != "agent":
            return None
        turns = cls._turns_from_content(content) if turns_used is None else turns_used
        body = Text(cls._step(content), style=BODY, no_wrap=True, overflow="ellipsis")
        return Panel(
            Group(cls._header(call, elapsed_seconds=elapsed_seconds, turns_used=turns), body),
            border_style=CARD_BORDER,
            style=CARD_BG,
            padding=(0, 1),
            expand=True,
        )

    @classmethod
    def _tail_lines(cls, child_session_path: str, limit: int) -> list[str]:
        path = Path(child_session_path) / "conversation.jsonl"
        lines: deque[str] = deque(maxlen=limit)
        try:
            with path.open(encoding="utf-8") as handle:
                for raw_line in handle:
                    try:
                        row = json.loads(raw_line)
                    except (json.JSONDecodeError, RecursionError):
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
        except OSError:
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
    ) -> Panel | None:
        if call.name.casefold() != "agent":
            return None
        tail = cls._tail_lines(child_session_path, limit)
        body = Text(
            "\n".join(tail) if tail else "child transcript unavailable",
            style=BODY if tail else DIM,
            overflow="ellipsis",
            no_wrap=True,
        )
        return Panel(
            Group(
                cls._header(
                    call,
                    elapsed_seconds=elapsed_seconds,
                    turns_used=turns_used,
                    expanded=True,
                ),
                body,
            ),
            border_style=CARD_BORDER,
            style=CARD_BG,
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
    ) -> Text | None:
        call = event.tool_call
        result = event.tool_result
        if call is None or result is None or call.name.casefold() != "agent":
            return None
        elapsed = elapsed_seconds
        if elapsed is None:
            value = event.data.get("elapsed_seconds")
            if isinstance(value, (int, float)):
                elapsed = max(0.0, float(value))
            else:
                value = event.data.get("elapsed_ms")
                elapsed = max(0.0, float(value) / 1000) if isinstance(value, (int, float)) else 0.0
        turns = turns_used
        if turns is None and result.structured_content is not None:
            value = result.structured_content.get("turns_used")
            turns = value if type(value) is int and value >= 0 else 0
        turns = turns or 0
        status = "canceled" if result.content == "tool execution canceled" else (
            "fail" if result.is_error else "ok"
        )
        return Text(
            f"{cls._description(call)} · {turns} turns · {max(0.0, elapsed or 0.0):.1f}s · "
            f"{status} · expand: ctrl+x ctrl+o",
            style=ERROR if status == "fail" else RECEIPT,
            no_wrap=True,
            overflow="ellipsis",
        )

    @classmethod
    def render_start(cls, event: StreamEvent) -> RenderableType | None:
        call = event.tool_call
        if event.type is not StreamEventType.TOOL_EXECUTION_START or call is None:
            return None
        return cls.render_progress(call, "")

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
        )

    def current(self) -> RenderableType | None:
        return self._progress() if self.active else None

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
        if self._expanded:
            return self._expanded_render()
        return self._progress()

    def refresh(self) -> RenderableType | None:
        if not self.active:
            return None
        return self._expanded_render() if self._expanded else self._progress()

    def finish(self, event: StreamEvent | None) -> RenderableType | None:
        if not self._supported or event is None:
            return None
        self._finished = True
        self._elapsed_seconds = self._elapsed()
        result = event.tool_result
        structured = result.structured_content if result is not None else None
        turns = structured.get("turns_used") if structured else None
        if type(turns) is int and turns >= 0:
            self._turns = turns
        path = structured.get("child_session_path") if structured else None
        if isinstance(path, str):
            self._child_session_path = path
        self._receipt = type(self).render_receipt(
            event,
            elapsed_seconds=self._elapsed_seconds,
            turns_used=self._turns,
        )
        return self._expanded_render() if self._expanded else self._receipt

    def _expanded_render(self) -> Panel | None:
        return type(self).render_expanded(
            self.call,
            elapsed_seconds=self._elapsed_seconds if self._finished else self._elapsed(),
            turns_used=self._turns,
            child_session_path=self._child_session_path,
        )

    def toggle(self) -> RenderableType | None:
        if not self._supported:
            return None
        self._expanded = not self._expanded
        if self._expanded:
            return self._expanded_render()
        return self._receipt


def render_agent_progress(
    call: ToolCall,
    content: str,
    *,
    elapsed_seconds: float = 0.0,
    turns_used: int | None = None,
) -> Panel | None:
    return AgentCard.render_progress(
        call,
        content,
        elapsed_seconds=elapsed_seconds,
        turns_used=turns_used,
    )


def render_agent_expanded(
    call: ToolCall,
    *,
    elapsed_seconds: float,
    turns_used: int,
    child_session_path: str,
    limit: int = MAX_TAIL_LINES,
) -> Panel | None:
    return AgentCard.render_expanded(
        call,
        elapsed_seconds=elapsed_seconds,
        turns_used=turns_used,
        child_session_path=child_session_path,
        limit=limit,
    )


def render_agent_receipt(
    event: StreamEvent,
    *,
    elapsed_seconds: float | None = None,
    turns_used: int | None = None,
) -> Text | None:
    return AgentCard.render_receipt(
        event,
        elapsed_seconds=elapsed_seconds,
        turns_used=turns_used,
    )
