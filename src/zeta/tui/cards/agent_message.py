"""Readable cards for messages between parent and child agents."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.text import Text

from ...protocol.types import StreamEvent, ToolCall
from .. import theme
from .base import render_tool_output, safe_text, tool_panel
from .shared import MAX_CARD_LINES, MAX_TOOL_SCAN_BYTES, scan_tool_output

_ANSWER_RE = re.compile(r"^Re\s+([^:]+):", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class AgentIdentity:
    description: str
    short_handle: str
    agent_type: str | None = None
    model: str | None = None

    @property
    def label(self) -> str:
        return f"{self.description} {self.short_handle}"


class AgentIdentityIndex:
    """Resolve agent handles from transcript results and in-memory markers."""

    def __init__(
        self,
        marker_snapshot: Callable[[], Mapping[str, Mapping[str, Any]]] | None = None,
    ) -> None:
        self._marker_snapshot = marker_snapshot
        self._identities: dict[str, AgentIdentity] = {}

    def observe(self, event: StreamEvent) -> None:
        call = event.tool_call
        result = event.tool_result
        if (
            call is None
            or result is None
            or not isinstance(result.structured_content, dict)
        ):
            return
        name = call.name.casefold()
        if name == "agent":
            item = dict(result.structured_content)
            item.setdefault("model", call.arguments.get("model"))
            self._observe_item(item)
        elif name == "agent_status":
            children = result.structured_content.get("children")
            if isinstance(children, list):
                for child in children:
                    if isinstance(child, dict):
                        self._observe_item(child)

    def _observe_item(self, item: Mapping[str, Any]) -> None:
        handle = item.get("child_instance_id", item.get("handle"))
        description = item.get("description")
        if (
            not isinstance(handle, str)
            or not isinstance(description, str)
            or not description
        ):
            return
        self._identities[handle] = AgentIdentity(
            description,
            short_agent_handle(handle),
            _optional_text(item.get("agent_type")),
            _optional_text(item.get("model")),
        )

    def resolve(self, handle: str) -> AgentIdentity:
        observed = self._identities.get(handle)
        if observed is not None:
            return observed
        markers = self._marker_snapshot() if self._marker_snapshot is not None else {}
        marker = markers.get(handle)
        if isinstance(marker, Mapping):
            description = marker.get("description")
            if isinstance(description, str) and description:
                tool_call = marker.get("tool_call")
                arguments = (
                    tool_call.get("arguments") if isinstance(tool_call, dict) else None
                )
                model = arguments.get("model") if isinstance(arguments, dict) else None
                identity = AgentIdentity(
                    description,
                    short_agent_handle(handle),
                    _optional_text(marker.get("agent_type")),
                    _optional_text(model),
                )
                self._identities[handle] = identity
                return identity
        return AgentIdentity("child", short_agent_handle(handle))


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _state_style(state: str) -> str:
    if state == "error":
        return theme.ERROR
    if state in {"queued", "delivered", "sent"}:
        return theme.SUCCESS
    return theme.NOTICE


def short_agent_handle(handle: str) -> str:
    separator = handle.find(":")
    return handle[separator:] if separator >= 0 else handle


def answer_question_id(message: str) -> str | None:
    match = _ANSWER_RE.match(message)
    if match is None:
        return None
    question_id = match.group(1).strip()
    if len(question_id) <= 12:
        return question_id
    return question_id[:8]


def agent_header(
    verb: str,
    direction: str,
    identity: AgentIdentity,
    *,
    state: str | None = None,
    answer_to: str | None = None,
) -> Text:
    header = Text.assemble(
        (f"{verb} {direction} ", theme.AGENT_MAIN),
        (identity.label, theme.AGENT_CHILD),
    )
    if answer_to is not None:
        header.append(f" · answer to {answer_to}", style=theme.NOTICE)
    if state is not None:
        header.append(f" · {state}", style=_state_style(state))
    return header


def render_agent_tool_message(
    event: StreamEvent,
    running: bool,
    identity_index: AgentIdentityIndex | None,
) -> Panel | None:
    call = event.tool_call
    if call is None:
        return None
    name = call.name.casefold()
    if name == "agent_send":
        handle = call.arguments.get("child_instance_id")
        message = call.arguments.get("message")
        if not isinstance(handle, str) or not isinstance(message, str):
            return None
        identity = (identity_index or AgentIdentityIndex()).resolve(handle)
        answer_to = answer_question_id(message)
        result = event.tool_result
        if running or result is None:
            state = "sending"
        elif result.is_error:
            state = "error"
        elif result.content.casefold().startswith("delivered"):
            state = "delivered"
        else:
            state = "queued"
        return render_agent_message(
            agent_header("send", "→", identity, state=state, answer_to=answer_to),
            body=message,
            result=result.content if result is not None and result.is_error else None,
            state="error" if result is not None and result.is_error else state,
        )
    if name == "ask_parent":
        question = call.arguments.get("question")
        options = call.arguments.get("options", ())
        if (
            not isinstance(question, str)
            or not isinstance(options, (list, tuple))
            or any(not isinstance(option, str) for option in options)
        ):
            return None
        result = event.tool_result
        if running or result is None:
            state = "pending"
            receipt = None
        elif result.is_error:
            state = "error"
            receipt = result.content
        else:
            state = "sent"
            structured = result.structured_content
            question_id = (
                structured.get("question_id") if isinstance(structured, dict) else None
            )
            receipt = (
                f"question {question_id}"
                if isinstance(question_id, str)
                else "question sent"
            )
        header = Text("ask → parent", style=theme.AGENT_MAIN)
        header.append(f" · {state}", style=_state_style(state))
        return render_agent_message(
            header,
            body=question,
            options=options,
            result=receipt,
            state="error" if state == "error" else state,
        )
    return None


def render_agent_message(
    header: RenderableType,
    *,
    body: str | None = None,
    options: Iterable[str] = (),
    result: str | None = None,
    state: str | None = None,
) -> Panel:
    """Render one bounded, wrapping agent communication card."""

    content: list[RenderableType] = [header]
    if body is not None:
        content.append(_message_body(body))
    option_lines = [f"{index}. {option}" for index, option in enumerate(options, 1)]
    if option_lines:
        content.append(_message_body("\n".join(option_lines)))
    if result is not None:
        style = theme.ERROR if state == "error" else theme.NOTICE
        content.append(safe_text(result, style=style, wrap=True))
    # Group keeps wrapping enabled for bodies; tool_panel's Text fast path is no-wrap.
    return tool_panel(
        ToolCall("agent-message", "agent_message", {}),
        Group(*content[1:]) if len(content) > 1 else None,
        header=content[0],
    )


def _message_body(body: str) -> Text:
    scan = scan_tool_output(body)
    if scan.truncated and len(body.encode("utf-8")) <= MAX_TOOL_SCAN_BYTES:
        total_lines = body.count("\n") + (0 if body.endswith("\n") else 1)
        scan = scan._replace(total_lines=total_lines)
    if scan.truncated or scan.total_lines is None or scan.total_lines > MAX_CARD_LINES:
        return render_tool_output(body, scan=scan)
    return safe_text(body, style=theme.BODY, wrap=True)
