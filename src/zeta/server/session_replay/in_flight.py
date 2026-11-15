"""The foreground reply that is streaming now, for an `in_flight_replay` join.

History has no row for a reply until its `assistant_message`, and a running
tool's streamed output has no row at all. A replay join with `in_flight_replay`
does not hold these live frames while it reads history; it sends the merged
frames kept here instead, so a client that joins mid-reply gets the whole partial reply once, then the
live deltas after it.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..protocol import EventDescriptor

# a longer in-flight reply or tool output keeps only its newest parts; the
# final assistant_message or tool_end still carries all of it
IN_FLIGHT_MAX_BYTES = 256 * 1024
IN_FLIGHT_MAX_PARTS = 16  # per reply and per tool call
_MERGE_MAX_BYTES = 64 * 1024


@dataclass(slots=True)
class _Part:
    event: EventDescriptor
    group: object  # the reply or one tool call; the part cap counts per group
    fields: dict[str, object]  # the frame without its text
    text_key: str
    texts: list[str]
    size: int


def _tool_call_id(fields: Mapping[str, object]) -> object:
    call = fields.get("tool_call")
    return call.get("id") if isinstance(call, Mapping) else None


_REPLY = object()


class InFlightReply:
    """The active session's unfinished reply text and running tools' output."""

    def __init__(self) -> None:
        self._session_id: str | None = None
        self._parts: list[_Part] = []
        # foreground tool calls of this run, so a join drops their held output
        self._tools: set[object] = set()

    def start_run(self, session_id: str | None) -> None:
        self._session_id = session_id
        self._parts.clear()
        self._tools.clear()

    def end_run(self) -> None:
        self._parts.clear()

    def delta(self, session_id: str | None, delta: str, kind: str) -> None:
        self._add(session_id, EventDescriptor.ASSISTANT_DELTA, _REPLY, {"kind": kind}, "delta", delta)

    def tool_output(self, session_id: str | None, fields: dict[str, object]) -> None:
        tool_call_id = _tool_call_id(fields)
        self._tools.add(tool_call_id)
        rest = {key: value for key, value in fields.items() if key != "output"}
        self._add(
            session_id, EventDescriptor.TOOL_OUTPUT, tool_call_id, rest, "output",
            str(fields.get("output", "")),
        )

    def end_message(self) -> None:
        """The reply was committed, reset, or ended; history or a live frame has it."""
        self._parts = [p for p in self._parts if p.group is not _REPLY]

    def end_tool(self, fields: Mapping[str, object]) -> None:
        tool_call_id = _tool_call_id(fields)
        self._parts = [
            p for p in self._parts
            if p.event is not EventDescriptor.TOOL_OUTPUT or p.group != tool_call_id
        ]

    def owns(self, event: object, fields: Mapping[str, object]) -> bool:
        """True for a live frame that a join's in-flight frames replace."""
        if event is EventDescriptor.ASSISTANT_DELTA:
            return True
        return event is EventDescriptor.TOOL_OUTPUT and _tool_call_id(fields) in self._tools

    def frames(self, session_id: str) -> list[tuple[EventDescriptor, dict[str, object]]]:
        if session_id != self._session_id:
            return []
        return [
            (p.event, {**p.fields, p.text_key: "".join(p.texts)}) for p in self._parts
        ]

    def _add(
        self,
        session_id: str | None,
        event: EventDescriptor,
        group: object,
        fields: dict[str, object],
        text_key: str,
        text: str,
    ) -> None:
        if session_id != self._session_id:
            self.start_run(session_id)
        size = len(text.encode("utf-8"))
        # parts of different groups are independent, so a group merges into its own last part
        last = next((p for p in reversed(self._parts) if p.group == group), None)
        if (
            last is not None
            and last.event is event
            and last.fields == fields
            and last.size + size <= _MERGE_MAX_BYTES
        ):
            last.texts.append(text)
            last.size += size
        else:
            self._parts.append(_Part(event, group, fields, text_key, [text], size))
            in_group = [p for p in self._parts if p.group == group]
            if len(in_group) > IN_FLIGHT_MAX_PARTS:
                self._parts.remove(in_group[0])
        while len(self._parts) > 1 and sum(p.size for p in self._parts) > IN_FLIGHT_MAX_BYTES:
            self._parts.pop(0)
