"""Plan a format-2 transaction undo against the current memory state.

The interface accepts immutable snapshots and receipts. It either returns one
validated replacement state that preserves independent later work, or rejects
the plan before publication when later work depends on the target transaction.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from zeta.memory.entry_store import (
    MemoryState,
    OperationReceipt,
    canonical_state_bytes,
)
from zeta.project_errors import ProjectRegistryError


@dataclass(frozen=True, slots=True)
class EntryUndoPlan:
    state: MemoryState
    target_ids: tuple[str, ...]


def plan_entry_transaction_undo(
    *,
    current: MemoryState,
    before: MemoryState,
    after: MemoryState,
    target_receipts: tuple[OperationReceipt, ...],
    later_receipts: tuple[OperationReceipt, ...],
) -> EntryUndoPlan:
    """Invert only the target transaction while preserving independent changes."""
    if before.project_id != current.project_id or after.project_id != current.project_id:
        raise ProjectRegistryError("memory undo snapshots target different projects")
    schema_changed = before.schema != after.schema
    if schema_changed and current.schema != after.schema:
        raise ProjectRegistryError("memory undo has dependent later schema changes")

    changed_ids = {
        entry_id
        for entry_id in before.entries.keys() | after.entries.keys()
        if before.entries.get(entry_id) != after.entries.get(entry_id)
    }
    dependent_ids = {
        entry_id
        for entry_id in changed_ids
        if current.entries.get(entry_id) != after.entries.get(entry_id)
    }
    dependencies = [
        receipt
        for receipt in later_receipts
        if dependent_ids.intersection((*receipt.target_ids, *receipt.result_ids))
    ]
    if dependencies:
        details = "; ".join(
            f"{item.type} {item.operation_id} "
            f"({', '.join(dict.fromkeys((*item.target_ids, *item.result_ids)))})"
            for item in dependencies
        )
        raise ProjectRegistryError(
            f"memory undo has dependent later operations: {details}"
        )

    # Exact state comparison makes reversed operations inactive dependencies.
    # Any difference without a retained receipt still fails closed.
    if dependent_ids:
        raise ProjectRegistryError(
            "memory undo has dependent later operations touching: "
            + ", ".join(sorted(dependent_ids))
        )

    entries = dict(current.entries)
    for entry_id in changed_ids:
        prior = before.entries.get(entry_id)
        if prior is None:
            entries.pop(entry_id, None)
        else:
            entries[entry_id] = prior
    state = replace(
        current,
        generation=current.generation + 1,
        schema=before.schema if schema_changed else current.schema,
        entries=entries,
    )
    canonical_state_bytes(state)
    target_ids = tuple(
        dict.fromkeys(
            entry_id
            for receipt in target_receipts
            for entry_id in (*receipt.target_ids, *receipt.result_ids)
        )
    )
    return EntryUndoPlan(state=state, target_ids=target_ids)


__all__ = ["EntryUndoPlan", "plan_entry_transaction_undo"]
