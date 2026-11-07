"""Durable served-turn end facts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ._store import ConversationStore

TurnEndReason = Literal["completed", "user_stop", "error", "disconnect"]
_TURN_END_REASONS = {"completed", "user_stop", "error", "disconnect"}


def validate_turn_end_data(data: Mapping[str, object]) -> None:
    if set(data) != {"reason"} or data.get("reason") not in _TURN_END_REASONS:
        raise ValueError("invalid turn end")


class TurnHistoryMixin:
    async def append_turn_end_async(
        self: ConversationStore, reason: TurnEndReason
    ) -> None:
        if reason not in _TURN_END_REASONS:
            raise ValueError("invalid turn end reason")
        await self._to_thread_durable(self._append_row, "turn_end", {"reason": reason})
