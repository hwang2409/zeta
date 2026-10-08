"""Dormant format-2 project-memory reconciliation.

The module owns bounded provider requests, proposal validation, evidence policy,
dependency rejection, and CAS regeneration behind one reconciliation interface.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import date

from .reconciler import ReconciliationResponse, Transcript

EntryInvokeResult = str | ReconciliationResponse
EntryInvoke = Callable[[str], EntryInvokeResult | Awaitable[EntryInvokeResult]]


@dataclass(frozen=True, slots=True)
class EntryReconciliationResult:
    """Result of one complete scheduled format-2 reconciliation attempt."""

    changed_entry_ids: tuple[str, ...]
    rejected_groups: tuple[str, ...]
    usage: Mapping[str, int]


async def reconcile_entry_range(
    *,
    registry: object,
    project_id: str,
    transcript: Transcript,
    reconciliation_key: str,
    invoke: EntryInvoke,
    cas_retries: int,
    as_of: date,
    now: str,
) -> EntryReconciliationResult:
    """Reconcile one durable transcript fragment into dormant format-2 state."""
    raise NotImplementedError
