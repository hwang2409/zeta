"""Ordered admission and cancellation recovery for approval frames."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

from ..connections import Admission
from ..protocol import EventDescriptor

if TYPE_CHECKING:
    from .approval_lifecycle import ApprovalRecord

Admit = Callable[..., Admission]


def admit_end_then_terminal(
    admit: Admit,
    session_id: str | None,
    end: tuple[str, dict[str, object]] | None,
    terminal: tuple[str, dict[str, object]] | None,
) -> tuple[Admission | None, Admission | None]:
    """Admit the end and matching terminal without an await between them."""
    end_result = admit(EventDescriptor.APPROVAL_END, session_id, **end[1]) if end else None
    terminal_result = (
        admit(EventDescriptor.TOOL_END, session_id, **terminal[1]) if terminal else None
    )
    return end_result, terminal_result


def reset_cancelled_settlement(
    record: ApprovalRecord,
    release_gate: Callable[..., None],
    put: Callable[[ApprovalRecord], None],
    token: int,
) -> None:
    """Return a cancelled durable settlement to retryable state with a fresh gate."""
    release_gate(record, error=asyncio.CancelledError())
    retry = replace(
        record,
        admitted=asyncio.get_running_loop().create_future(),
        token=token + 1,
        phase="pending",
        disposition="active",
        settlement=None,
        publication_owner=None,
    )
    put(retry)
