"""Durable served-turn end facts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ._store import ConversationStore

TurnEndReason = Literal["completed", "user_stop", "error", "disconnect"]
_TURN_END_REASONS = {"completed", "user_stop", "error", "disconnect"}
_TURN_ID = re.compile(r"[0-9a-f]{32}").fullmatch


def validate_turn_end_data(data: Mapping[str, object]) -> None:
    turn_id = data.get("turn_id")
    if (
        set(data) != {"turn_id", "reason"}
        or not isinstance(turn_id, str)
        or _TURN_ID(turn_id) is None
        or data.get("reason") not in _TURN_END_REASONS
    ):
        raise ValueError("invalid turn end")


class TurnHistoryMixin:
    async def append_turn_end_async(
        self: ConversationStore, turn_id: str, reason: TurnEndReason
    ) -> None:
        data = {"turn_id": turn_id, "reason": reason}
        validate_turn_end_data(data)
        await self._to_thread_durable(self._append_row, "turn_end", data)
