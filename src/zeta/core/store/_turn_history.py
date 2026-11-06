"""Durable served-turn end facts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ._store import ConversationStore

TurnEndReason = Literal["completed", "user_stop", "error", "disconnect"]
_TURN_END_REASONS = {"completed", "user_stop", "error", "disconnect"}


def validate_turn_end_data(data: dict[str, object]) -> None:
    reason = data.get("reason")
    if data == {}:
        return
    if set(data) != {"reason"} or reason not in _TURN_END_REASONS:
        raise ValueError("invalid turn end")


class TurnHistoryMixin:
    async def append_turn_end_async(
        self: ConversationStore, reason: TurnEndReason
    ) -> None:
        if reason not in _TURN_END_REASONS:
            raise ValueError("invalid turn end reason")
        await self._to_thread_durable(
            self._append_row, "turn_end", {"reason": reason}
        )
