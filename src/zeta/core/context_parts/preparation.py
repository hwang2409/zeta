"""Prepare one context snapshot without blocking the event loop."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from ..store import ConversationEntry, ConversationStore
from .types import _AssemblyPreparation

PreparationKey = tuple[str | None, bool, int, str]
RetainPreparation = Callable[
    [PreparationKey, tuple[ConversationEntry, ...], _AssemblyPreparation], None
]


class ContextPreparation:
    """Own one cancellable, stale-head-checked assembly preparation."""

    def __init__(
        self,
        *,
        store: ConversationStore,
        prepare: Callable[..., _AssemblyPreparation],
        cooperative_call: Callable[..., _AssemblyPreparation],
        token_budget: Callable[[], int],
        estimator_budget: Callable[[], int],
        retain: RetainPreparation,
        invalidate: Callable[[], None],
    ) -> None:
        self._store = store
        self._prepare = prepare
        self._cooperative_call = cooperative_call
        self._token_budget = token_budget
        self._estimator_budget = estimator_budget
        self._retain = retain
        self._invalidate = invalidate

    def budget(self, reserved_estimator_tokens: int) -> int:
        if reserved_estimator_tokens < 0:
            raise ValueError("reserved estimator tokens must be non-negative")
        return self._token_budget() if reserved_estimator_tokens == 0 else max(1, self._estimator_budget() - reserved_estimator_tokens)

    def key(self, branch_id: str | None, *, force: bool, assembly_budget: int, system_append: str) -> PreparationKey:
        return branch_id, force, assembly_budget, system_append

    async def prepare(
        self,
        *,
        force: bool,
        reserved_estimator_tokens: int,
        system_append: str,
    ) -> bool:
        assembly_budget = self.budget(reserved_estimator_tokens)
        try:
            while True:
                branch = self._store.active_branch_snapshot()
                branch_id = branch[-1].id if branch else None
                preparation = await asyncio.to_thread(
                    self._cooperative_call,
                    self._prepare,
                    branch,
                    force=force,
                    assembly_budget=assembly_budget,
                    system_append=system_append,
                )
                await asyncio.sleep(0)
                if self._store.active_branch_head_id() == branch_id:
                    self._retain(
                        self.key(
                            branch_id,
                            force=force,
                            assembly_budget=assembly_budget,
                            system_append=system_append,
                        ),
                        tuple(branch),
                        preparation,
                    )
                    return preparation.should_compact
        except asyncio.CancelledError:
            self._invalidate()
            raise
