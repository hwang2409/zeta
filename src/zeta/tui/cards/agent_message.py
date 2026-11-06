"""Readable cards for messages between parent and child agents."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.text import Text

from ...agent.receipt import terminal_state
from ...protocol.types import StreamEvent, ToolCall
from .. import theme
from .agent import AGENT_TOOL_NAME
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
        if name == AGENT_TOOL_NAME:
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
            _identity_description(description),
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
                    _identity_description(description),
                    short_agent_handle(handle),
                    _optional_text(marker.get("agent_type")),
                    _optional_text(model),
                )
                self._identities[handle] = identity
                return identity
        identity = AgentIdentity("child", short_agent_handle(handle))
        self._identities[handle] = identity
        return identity


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _identity_description(value: str) -> str:
    return " ".join(safe_text(value, style=theme.BODY, wrap=True).plain.split())


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
    renderer = {
        "agent_send": _render_agent_send,
        "ask_parent": _render_ask_parent,
        "agent_status": _render_agent_status,
        "agent_output": _render_agent_output,
        "agent_cancel": _render_agent_cancel,
    }.get(call.name.casefold())
    if renderer is None:
        return None
    return renderer(event, running, identity_index or AgentIdentityIndex())


def _render_agent_send(
    event: StreamEvent,
    running: bool,
    identities: AgentIdentityIndex,
) -> Panel | None:
    call = event.tool_call
    if call is None:
        return None
    handle = call.arguments.get("child_instance_id")
    message = call.arguments.get("message")
    if not isinstance(handle, str) or not isinstance(message, str):
        return None
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
        agent_header(
            "send",
            "→",
            identities.resolve(handle),
            state=state,
            answer_to=answer_question_id(message),
        ),
        body=message,
        result=result.content if result is not None and result.is_error else None,
        state="error" if result is not None and result.is_error else state,
    )


def _render_ask_parent(
    event: StreamEvent,
    running: bool,
    _identities: AgentIdentityIndex,
) -> Panel | None:
    call = event.tool_call
    if call is None:
        return None
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
        state, receipt = "pending", None
    elif result.is_error:
        state, receipt = "error", result.content
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


def _control_handle(event: StreamEvent) -> str | None:
    result = event.tool_result
    structured = result.structured_content if result is not None else None
    value = structured.get("handle") if isinstance(structured, dict) else None
    if not isinstance(value, str) and event.tool_call is not None:
        value = event.tool_call.arguments.get("handle")
    return value if isinstance(value, str) else None


def _render_agent_status(
    event: StreamEvent,
    running: bool,
    identities: AgentIdentityIndex,
) -> Panel:
    result = event.tool_result
    structured = result.structured_content if result is not None else None
    children = structured.get("children") if isinstance(structured, dict) else None
    if running or not isinstance(children, list):
        state = "error" if result is not None and result.is_error else "running"
        return render_agent_message(
            Text(f"agent status · {state}", style=_state_style(state)),
            result=result.content if result is not None and result.is_error else None,
            state=state,
        )
    rows: list[str] = []
    for child in children:
        if not isinstance(child, Mapping):
            continue
        handle = child.get("handle")
        if not isinstance(handle, str):
            continue
        identity = identities.resolve(handle)
        fields = [identity.label]
        for value in (
            child.get("agent_type", identity.agent_type),
            child.get("model", identity.model),
            child.get("state"),
        ):
            if isinstance(value, str) and value:
                fields.append(value)
        rows.append(" · ".join(fields))
    noun = "child" if len(rows) == 1 else "children"
    header = Text(f"agent status · {len(rows)} {noun}", style=theme.AGENT_MAIN)
    return render_agent_message(header, body="\n".join(rows) or "no children")


def _render_agent_output(
    event: StreamEvent,
    running: bool,
    identities: AgentIdentityIndex,
) -> Panel | None:
    handle = _control_handle(event)
    if handle is None:
        return None
    identity = identities.resolve(handle)
    result = event.tool_result
    if running or result is None:
        return render_agent_message(
            agent_header("output", "←", identity, state="reading")
        )
    structured = result.structured_content
    offset = structured.get("offset") if isinstance(structured, dict) else None
    total = structured.get("total") if isinstance(structured, dict) else None
    header = agent_header(
        "output", "←", identity, state="error" if result.is_error else None
    )
    if type(offset) is int and type(total) is int:
        end = min(total, offset + len(result.content))
        page = f"page {offset + 1}-{end} of {total}" if end > offset else f"page {offset} of {total}"
        header.append(f" · {page}", style=theme.NOTICE)
    return render_agent_message(
        header,
        body=result.content,
        state="error" if result.is_error else None,
    )


def _render_agent_cancel(
    event: StreamEvent,
    running: bool,
    identities: AgentIdentityIndex,
) -> Panel | None:
    handle = _control_handle(event)
    if handle is None:
        return None
    identity = identities.resolve(handle)
    result = event.tool_result
    if running or result is None:
        outcome = "canceling"
    elif result.is_error:
        outcome = "error"
    else:
        structured = result.structured_content
        status = structured.get("status") if isinstance(structured, dict) else None
        outcome = {
            "cancellation_confirmed": "canceled",
            "already_finished": "already_finished",
        }.get(status, status if isinstance(status, str) else "canceled")
    return render_agent_message(
        agent_header("cancel", "→", identity, state=outcome),
        result=result.content if result is not None and result.is_error else None,
        state="error" if result is not None and result.is_error else outcome,
    )


def render_agent_notification_message(
    event: StreamEvent,
    identity_index: AgentIdentityIndex | None,
) -> Panel | Text | None:
    """Render child notification kinds owned by the shared agent card."""

    kind = event.data.get("kind", "agent_completion")
    identities = identity_index or AgentIdentityIndex()
    if kind == "child_question":
        return _render_child_question(event, identities)
    if kind == "child_question_withdrawn":
        return _render_child_question_withdrawn(event, identities)
    if kind == "child_idle":
        return _render_child_idle(event, identities)
    if kind == "agent_completion":
        return _render_child_completion(event, identities)
    return None


def _notification_identity(
    event: StreamEvent,
    identities: AgentIdentityIndex,
) -> AgentIdentity | None:
    handle = event.data.get("child_instance_id")
    if not isinstance(handle, str):
        return None
    resolved = identities.resolve(handle)
    description = event.data.get("description")
    if resolved.description == "child" and isinstance(description, str) and description:
        return AgentIdentity(
            _identity_description(description),
            resolved.short_handle,
            resolved.agent_type,
            resolved.model,
        )
    return resolved


def _unavailable_notification() -> Text:
    return Text("background agent notification unavailable", style=theme.ERROR)


def _render_child_question(
    event: StreamEvent,
    identities: AgentIdentityIndex,
) -> Panel | Text:
    identity = _notification_identity(event, identities)
    question = event.data.get("question")
    question_id = event.data.get("question_id")
    options = event.data.get("options", ())
    if (
        identity is None
        or not isinstance(question, str)
        or not isinstance(question_id, str)
        or not isinstance(options, (list, tuple))
        or any(not isinstance(option, str) for option in options)
    ):
        return _unavailable_notification()
    return render_agent_message(
        agent_header("ask", "←", identity, state="pending"),
        body=question,
        options=options,
        result=f"question {question_id}",
        state="pending",
    )


def _render_child_question_withdrawn(
    event: StreamEvent,
    identities: AgentIdentityIndex,
) -> Panel | Text:
    identity = _notification_identity(event, identities)
    question_id = event.data.get("question_id")
    if identity is None or not isinstance(question_id, str):
        return _unavailable_notification()
    return render_agent_message(
        agent_header("withdraw", "←", identity, state="withdrawn"),
        result=f"question {question_id}",
        state="withdrawn",
    )


def _render_child_idle(
    event: StreamEvent,
    identities: AgentIdentityIndex,
) -> Panel | Text:
    identity = _notification_identity(event, identities)
    report = event.data.get("text")
    if identity is None or not isinstance(report, str):
        return _unavailable_notification()
    return render_agent_message(
        agent_header("idle", "←", identity, state="idle"),
        body=report,
        state="idle",
    )


def _notification_stats(data: Mapping[str, Any]) -> list[str]:
    stats = data.get("stats")
    if not isinstance(stats, Mapping):
        return []
    parts: list[str] = []
    elapsed = stats.get("elapsed")
    if type(elapsed) in {int, float} and elapsed >= 0:
        seconds = round(float(elapsed), 1)
        if seconds >= 3600:
            parts.append(f"{int(seconds // 3600)}h{int(seconds // 60) % 60:02d}m")
        elif seconds >= 60:
            minutes, remainder = divmod(int(seconds), 60)
            parts.append(f"{minutes}m{remainder:02d}s")
        else:
            parts.append(f"{seconds:.1f}s")
    turns = stats.get("turns_used")
    if type(turns) is int and turns >= 0:
        noun = "turn" if turns == 1 else "turns"
        parts.append(f"{turns} {noun}")
    return parts


def _completion_state(data: Mapping[str, Any]) -> tuple[str, bool]:
    status = data.get("status")
    if not isinstance(status, str):
        raise TypeError
    state = terminal_state(status=status)
    stats = data.get("stats")
    if not isinstance(stats, Mapping) or not ({"error", "canceled"} & stats.keys()):
        return state, False
    error = stats.get("error")
    canceled = stats.get("canceled")
    conflict = (
        type(error) is not bool
        or type(canceled) is not bool
        or (error and canceled)
        or error != (state == "failed")
        or canceled != (state == "canceled")
    )
    return ("failed" if conflict else state), conflict


def _render_child_completion(
    event: StreamEvent,
    identities: AgentIdentityIndex,
) -> Panel | Text:
    identity = _notification_identity(event, identities)
    if identity is None:
        return _unavailable_notification()
    try:
        state, state_conflict = _completion_state(event.data)
    except TypeError:
        return _unavailable_notification()
    header = agent_header("complete", "←", identity, state=state)
    for stat in _notification_stats(event.data):
        header.append(f" · {stat}", style=theme.NOTICE)
    text = event.data.get("text")
    if state in {"failed", "canceled"}:
        reason = "conflicting completion state" if state_conflict else text
        body = f"reason: {reason}" if isinstance(reason, str) and reason else None
    else:
        lines: list[str] = []
        verdict = event.data.get("verdict")
        summary = event.data.get("summary")
        if isinstance(verdict, str) and verdict:
            lines.append(f"verdict: {verdict}")
        if isinstance(summary, str) and summary:
            lines.append(f"summary: {summary}")
        elif isinstance(text, str) and text:
            lines.append(text)
        body = "\n".join(lines) or None
    return render_agent_message(header, body=body, state=state)

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
