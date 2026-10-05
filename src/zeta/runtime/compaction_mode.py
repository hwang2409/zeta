"""Switch and describe the compaction mode of a live session.

This module owns one rule for every frontend: the session metadata, the
context assembler, and the tool registry agree on the compaction mode. A
switch changes only how the next compaction is made. Durable compaction
markers of either kind stay valid history views, because replay does not
depend on the mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..compaction import COMPACTION_MODES
from ..core.session import SessionError
from ..core.slash import compaction_history
from ..tools.recall_history import TOOL_NAME as RECALL_TOOL

USAGE = "usage: /compaction [evict|summary]"


@dataclass(frozen=True, slots=True)
class CompactionSwitch:
    """The result of one switch request."""

    previous: str
    current: str
    recall_history: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "compaction": self.current,
            "previous": self.previous,
            "recall_history": self.recall_history,
        }


def apply_compaction(loop: Any, mode: str) -> None:
    """Apply ``mode`` to the live loop for its next request; do not persist.

    Children spawned later inherit the mode through the loop's assembler and a
    clone of its registry. A switch that would remove a tool the session's
    policy requires is refused and leaves the loop unchanged.
    """

    if mode not in COMPACTION_MODES:
        raise ValueError(f"unknown compaction mode {mode!r}; use evict or summary")
    registry = loop.tool_registry
    previous = registry.compaction
    missing_before = set(registry.missing_required_tools)
    registry.set_compaction(mode)
    newly_missing = set(registry.missing_required_tools) - missing_before
    if newly_missing:
        registry.set_compaction(previous)
        raise ValueError(
            "tool policy requires " + ", ".join(sorted(newly_missing))
        )
    loop.context_assembler.compaction = mode
    loop.tool_schemas = list(registry.schemas)


def switch_compaction(loop: Any, mode: str) -> CompactionSwitch:
    """Persist ``mode`` in session metadata and apply it to the live loop."""

    previous = loop.context_assembler.compaction
    if mode != previous:
        apply_compaction(loop, mode)
        try:
            persist_compaction(loop)
        except BaseException:
            apply_compaction(loop, previous)
            raise
    return CompactionSwitch(previous, mode, _recall_advertised(loop))


def persist_compaction(loop: Any) -> None:
    """Pin the live loop's mode in session metadata.

    Call this only after ``apply_compaction`` and every other startup or
    request validation passed, so a refused switch never reaches disk.
    """

    metadata = loop.session_metadata
    mode = loop.context_assembler.compaction
    if (metadata.compaction, metadata.compaction_pinned) != (mode, True):
        loop.manager.record_compaction(metadata, compaction=mode, pinned=True)


def run_compaction_command(loop: Any, args: str, *, busy: str | None) -> str:
    """Serve ``/compaction [evict|summary]`` for any frontend.

    ``busy`` names the reason a frontend cannot switch now, or is ``None``.
    """

    requested = args.strip().lower()
    if not requested:
        return describe_compaction(loop)
    if requested not in COMPACTION_MODES:
        return USAGE
    if busy is not None:
        return f"compaction unchanged: {busy}"
    try:
        result = switch_compaction(loop, requested)
    except (ValueError, SessionError, OSError) as exc:
        return f"compaction unchanged: {exc}"
    if result.previous == result.current:
        return f"compaction: {result.current} (unchanged)"
    return (
        f"compaction: {result.previous} -> {result.current} "
        f"(applies from the next request; recall_history: {_recall_state(loop)})"
    )


def describe_compaction(loop: Any) -> str:
    """Show the mode, the budget, and the last durable compaction."""

    assembler = loop.context_assembler
    return "\n".join(
        [
            f"compaction: {assembler.compaction}",
            f"budget: {assembler.token_budget:,} tokens",
            f"recall_history: {_recall_state(loop)}",
            _last_compaction(loop),
        ]
    )


def _recall_advertised(loop: Any) -> bool:
    return RECALL_TOOL in loop.tool_registry.registered_names


def _recall_state(loop: Any) -> str:
    registry = loop.tool_registry
    if registry.compaction != "evict":
        return "not registered (summary mode)"
    if not registry.tool_is_allowed(RECALL_TOOL):
        return "blocked by tool policy"
    if _recall_advertised(loop):
        return "advertised"
    return "unavailable"


def _last_compaction(loop: Any) -> str:
    entries = loop.store.replay()
    markers = [entry for entry in entries if entry.type == "compaction"]
    if not markers:
        return "last compaction: none"
    marker = markers[-1]
    summary = compaction_history(entries, loop.context_assembler.token_counter)[-1]
    text = (
        f"last compaction: {marker.data.get('kind', 'summary')} at turn "
        f"{summary.turn}, {summary.entries_folded} entries folded, "
        f"{summary.tokens_saved:,} tokens saved"
    )
    telemetry = marker.data.get("telemetry")
    if isinstance(telemetry, dict) and "items_evicted" in telemetry:
        text += (
            f", {telemetry['items_evicted']} items evicted "
            f"({telemetry.get('tokens_before', '?')} -> "
            f"{telemetry.get('tokens_after', '?')} tokens)"
        )
    return text


__all__ = [
    "CompactionSwitch",
    "apply_compaction",
    "describe_compaction",
    "persist_compaction",
    "run_compaction_command",
    "switch_compaction",
]
