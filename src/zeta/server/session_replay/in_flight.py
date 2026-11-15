"""The foreground reply that is streaming now, for an `in_flight_replay` join.

History has no row for a reply until its `assistant_message`, and a running
tool's streamed output has no row at all. A replay join with `in_flight_replay`
replaces its held copies of these live frames with the merged frames kept here,
so a client that joins mid-reply gets the whole partial reply once, then the
live deltas after it.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..protocol import EventDescriptor

# shortcut: a longer in-flight reply keeps only its newest frames; the final
# assistant_message still carries all of it. Raise it if replies grow past this.
IN_FLIGHT_MAX_BYTES = 256 * 1024
IN_FLIGHT_MAX_FRAMES = 16
_MERGE_MAX_BYTES = 64 * 1024


@dataclass(slots=True)
class _Part:
    event: EventDescriptor
    fields: dict[str, object]
    size: int


def _tool_call_id(fields: Mapping[str, object]) -> object:
    call = fields.get("tool_call")
    return call.get("id") if isinstance(call, Mapping) else None


class InFlightReply:
    """The active session's unfinished reply text and running tools' output."""

    def __init__(self) -> None:
        self._session_id: str | None = None
        self._frames: list[_Part] = []
        # foreground tool calls of this run, so a join drops their held output
        self._tools: set[object] = set()

    def start_run(self, session_id: str | None) -> None:
        self._session_id = session_id
        self._frames.clear()
        self._tools.clear()

    def end_run(self) -> None:
        self._frames.clear()

    def delta(self, session_id: str | None, delta: str, kind: str) -> None:
        self._add(
            session_id,
            EventDescriptor.ASSISTANT_DELTA,
            {"delta": delta, "kind": kind},
            "delta",
        )

    def tool_output(self, session_id: str | None, fields: dict[str, object]) -> None:
        self._tools.add(_tool_call_id(fields))
        self._add(session_id, EventDescriptor.TOOL_OUTPUT, dict(fields), "output")

    def end_message(self) -> None:
        """The reply was committed, reset, or ended; history or a live frame has it."""
        self._frames = [f for f in self._frames if f.event is not EventDescriptor.ASSISTANT_DELTA]

    def end_tool(self, fields: Mapping[str, object]) -> None:
        tool_call_id = _tool_call_id(fields)
        self._frames = [
            f for f in self._frames
            if f.event is not EventDescriptor.TOOL_OUTPUT or _tool_call_id(f.fields) != tool_call_id
        ]

    def owns(self, event: object, fields: Mapping[str, object]) -> bool:
        """True for a live frame that a join's in-flight frames replace."""
        if event is EventDescriptor.ASSISTANT_DELTA:
            return True
        return event is EventDescriptor.TOOL_OUTPUT and _tool_call_id(fields) in self._tools

    def frames(self, session_id: str) -> list[tuple[EventDescriptor, dict[str, object]]]:
        if session_id != self._session_id:
            return []
        return [(f.event, dict(f.fields)) for f in self._frames]

    def _add(
        self,
        session_id: str | None,
        event: EventDescriptor,
        fields: dict[str, object],
        text_key: str,
    ) -> None:
        if session_id != self._session_id:
            self.start_run(session_id)
        text = str(fields[text_key])
        size = len(text.encode("utf-8"))
        last = self._frames[-1] if self._frames else None
        if (
            last is not None
            and last.event is event
            and last.size + size <= _MERGE_MAX_BYTES
            and {k: v for k, v in last.fields.items() if k != text_key}
            == {k: v for k, v in fields.items() if k != text_key}
        ):
            last.fields[text_key] = str(last.fields[text_key]) + text
            last.size += size
        else:
            self._frames.append(_Part(event, fields, size))
        while len(self._frames) > IN_FLIGHT_MAX_FRAMES or (
            len(self._frames) > 1 and sum(f.size for f in self._frames) > IN_FLIGHT_MAX_BYTES
        ):
            self._frames.pop(0)
