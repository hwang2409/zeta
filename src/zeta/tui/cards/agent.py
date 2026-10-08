"""Agent tool card presentation and state."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.text import Text

from ...agent.presets import GENERAL_PRESET, get_agent_preset
from ...agent.receipt import (
    agent_stats,
    ensure_agent_receipt_text,
    has_agent_receipt_suffix,
    terminal_state,
)
from ...core.checkpoints import ConversationIntegrityError, load_session_json
from ...protocol.types import StreamEvent, StreamEventType, ToolCall
from ...tools.agent import send_to_run
from .. import theme
from .agent_sync import (
    AgentTranscriptSource,
    AgentTranscriptTreeSnapshot,
)
from .base import compact_tool_card

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
        self._expanded = True
        self._receipt: RenderableType | None = None
        self._depth = 1
        self._tail: tuple[str, ...] = ()
        self._tail_loaded = False
        self._transcript_source: AgentTranscriptSource | None = None
        self._final_tail_pending = False

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
    ) -> Text:
        agent_type = cls._agent_type(call)
        prefix = f"{agent_type} · " if agent_type else ""
        return Text(
            f"{prefix}{cls._description(call)} · {elapsed_seconds:.1f}s · "
            f"{turns_used} turns · depth {depth}",
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
            box=box.MINIMAL if theme.AGENT_BG else box.ROUNDED,
            border_style=theme.CARD_BORDER,
            style=theme.AGENT_BG,
            padding=(0, 1),
            expand=True,
        )

    @classmethod
    def _tail_lines(
        cls,
        tree: AgentTranscriptTreeSnapshot,
        path: Path,
        limit: int,
        seen: set[Path] | None = None,
    ) -> list[str]:
        """Project a bounded recursive card tail from immutable snapshots."""

        seen = set() if seen is None else seen
        if path in seen or limit < 1:
            return []
        seen.add(path)
        snapshot = tree.transcript(path)
        if snapshot is None:
            return []
        if snapshot.unavailable_reason is not None:
            return [snapshot.unavailable_reason]
        lines: list[str] = []
        tool_names: dict[str, str] = {}
        for _entry_id, message in snapshot.messages:
            role = message.get("role", "message")
            if role == "assistant":
                tool_names.clear()
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if (
                    block_type == "text"
                    and role != "tool_result"
                    and isinstance(block.get("text"), str)
                ):
                    for text_line in block["text"].splitlines() or [""]:
                        lines.append(f"{role}: {text_line}")
                elif block_type == "tool_use" and isinstance(
                    block.get("tool_call"), dict
                ):
                    tool_call = block["tool_call"]
                    name = tool_call.get("name", "tool")
                    arguments = tool_call.get("arguments", {})
                    if isinstance(name, str) and isinstance(arguments, dict):
                        call_id = tool_call.get("id")
                        if isinstance(call_id, str):
                            tool_names[call_id] = name
                        lines.append(f"tool: {name} {_arguments(arguments)}")
            tool_result = message.get("tool_result")
            if isinstance(tool_result, dict):
                call_id = tool_result.get("tool_call_id")
                name = (
                    tool_names.get(call_id, "tool")
                    if isinstance(call_id, str)
                    else "tool"
                )
                result_data = tool_result.get("structured_content")
                exit_code = (
                    result_data.get("exit_code")
                    if isinstance(result_data, dict)
                    else None
                )
                if type(exit_code) is int:
                    status = f"exit {exit_code}"
                elif tool_result.get("is_error") is True:
                    status = "failed"
                else:
                    status = "done"
                lines.append(f"{name}: {status}")
                nested_path = (
                    result_data.get("child_session_path")
                    if isinstance(result_data, dict)
                    else None
                )
                if isinstance(nested_path, str) and nested_path:
                    nested_tail = cls._tail_lines(
                        tree, Path(nested_path), max(1, limit - len(lines)), seen
                    )
                    lines.extend(f"  {line}" for line in nested_tail)
            if len(lines) > limit:
                lines = lines[-limit:]
        return lines[-limit:]

    @classmethod
    def _expanded_panel(
        cls,
        call: ToolCall,
        tail: Sequence[str],
        *,
        elapsed_seconds: float,
        turns_used: int,
        depth: int,
    ) -> Panel | None:
        if call.name.casefold() != "agent":
            return None
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
                ),
                body,
            ),
            box=box.MINIMAL if theme.AGENT_BG else box.ROUNDED,
            border_style=theme.CARD_BORDER,
            style=theme.AGENT_BG,
            padding=(0, 1),
            expand=True,
        )

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
        tail: list[str] = []
        if child_session_path:
            source = AgentTranscriptSource(Path(child_session_path))
            try:
                snapshot = source.refresh(recursive=True)
                tail = cls._tail_lines(snapshot, snapshot.root, limit)
            except (ConversationIntegrityError, OSError, ValueError):
                pass
            finally:
                source.close()
        return cls._expanded_panel(
            call,
            tail,
            elapsed_seconds=elapsed_seconds,
            turns_used=turns_used,
            depth=depth,
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
            f"depth {display_depth} · {receipt_text}",
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
        if not self.active:
            return None
        return self._active_render()

    @property
    def transcript_source(self) -> AgentTranscriptSource | None:
        return self._transcript_source

    @property
    def final_tail_pending(self) -> bool:
        return self._final_tail_pending

    @property
    def refresh_eligible(self) -> bool:
        return bool(
            self._expanded
            and self._child_session_path
            and (self.active or self._final_tail_pending)
        )

    def set_child_session_path(self, path: str) -> None:
        if path == self._child_session_path:
            return
        previous = self._transcript_source
        self._tail = ()
        self._tail_loaded = False
        self._child_session_path = path
        self._transcript_source = (
            AgentTranscriptSource(Path(path), message_limit=64) if path else None
        )
        if previous is not None:
            previous.close()
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            source = self._transcript_source
            if source is not None:
                try:
                    self.apply_transcript_snapshot(source.refresh(recursive=True))
                except (ConversationIntegrityError, OSError, ValueError):
                    pass

    def apply_transcript_snapshot(
        self, snapshot: AgentTranscriptTreeSnapshot
    ) -> bool:
        """Publish one worker-produced snapshot without storage access."""

        if Path(self._child_session_path) != snapshot.root:
            return False
        tail = tuple(type(self)._tail_lines(snapshot, snapshot.root, MAX_TAIL_LINES))
        changed = not self._tail_loaded or tail != self._tail
        self._tail = tail
        self._tail_loaded = True
        return changed

    async def refresh_tail(self, *, final: bool = False) -> bool:
        """Refresh this card through the shared serialized snapshot module."""

        source = self._transcript_source
        if source is None or (not final and not self.refresh_eligible):
            return False
        try:
            snapshot = await asyncio.to_thread(source.refresh, recursive=True)
        except (ConversationIntegrityError, OSError, RuntimeError, ValueError):
            return False
        return self.apply_transcript_snapshot(snapshot)

    def release_transcript_source(self) -> None:
        """Detach this card's source and close it outside the owner loop."""

        source, self._transcript_source = self._transcript_source, None
        self._final_tail_pending = False
        if source is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            source.close()
        else:
            loop.create_task(asyncio.to_thread(source.close))

    async def finish_tail(self) -> RenderableType | None:
        """Publish one final tail, then close its source off-loop."""

        if not self._final_tail_pending:
            return None
        source, self._transcript_source = self._transcript_source, None
        self._final_tail_pending = False
        if source is not None:
            try:
                snapshot = await asyncio.to_thread(source.refresh, recursive=True)
                self.apply_transcript_snapshot(snapshot)
            except (ConversationIntegrityError, OSError, RuntimeError, ValueError):
                pass
            finally:
                await asyncio.to_thread(source.close)
        return self.terminal_render()

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
                self.set_child_session_path(path)
            depth = event.data.get("depth")
            if type(depth) is int and depth >= 1:
                self._depth = depth
        return self._active_render()

    def refresh(self) -> RenderableType | None:
        if not self.active:
            return None
        return self._active_render()

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
            return (
                self._receipt if self._expanded else compact_tool_card(rendered)
            )
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
            self.set_child_session_path(path)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            can_refresh_final_tail = False
        else:
            can_refresh_final_tail = True
        self._final_tail_pending = bool(
            can_refresh_final_tail
            and self._expanded
            and self._transcript_source is not None
        )
        if not self._final_tail_pending:
            self.release_transcript_source()
        self._receipt = type(self).render_receipt(
            event,
            elapsed_seconds=self._elapsed_seconds,
            turns_used=self._turns,
            depth=self._depth,
        )
        return (
            self._progress()
            if self._final_tail_pending
            else self.terminal_render()
        )

    def terminal_render(self) -> RenderableType | None:
        self._final_tail_pending = False
        return (
            self._expanded_render()
            if self._expanded and self._child_session_path
            else self._receipt
        )

    def _active_render(self) -> Panel | None:
        if self._expanded and self._child_session_path:
            return self._expanded_render()
        return self._progress()

    def _expanded_render(self) -> Panel | None:
        return type(self)._expanded_panel(
            self.call,
            self._tail,
            elapsed_seconds=self._elapsed_seconds if self._finished else self._elapsed(),
            turns_used=self._turns,
            depth=self._depth,
        )

    def toggle(self) -> RenderableType | None:
        if not self._supported and not self._disclosure_supported:
            return None
        if not self._supported:
            if self._receipt is None:
                return None
            self._expanded = not self._expanded
            return self._receipt if self._expanded else compact_tool_card(self._receipt)
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
