"""One-shot asynchronous context preparation."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Protocol

from ...context_accounting import cooperative_call
from ..store import ConversationEntry, ConversationStore
from .types import _AssemblyPreparation

PreparationKey = tuple[str | None, bool, int, str]
Prepared = tuple[tuple[ConversationEntry, ...], _AssemblyPreparation]


class PrepareAssembly(Protocol):
    def __call__(
        self,
        branch: Sequence[ConversationEntry],
        *,
        force: bool,
        assembly_budget: int,
        system_append: str,
    ) -> _AssemblyPreparation: ...


class ContextPreparation:
    """Own one cancellable, stale-head-checked assembly preparation."""

    def __init__(
        self,
        store: ConversationStore,
        prepare: PrepareAssembly,
        token_budget: int,
        estimator_budget: int,
    ) -> None:
        self._store = store
        self._prepare = prepare
        self._token_budget = token_budget
        self._estimator_budget = estimator_budget
        self._prepared: tuple[PreparationKey, Prepared] | None = None

    @staticmethod
    def key(
        branch_id: str | None,
        *,
        force: bool,
        assembly_budget: int,
        system_append: str,
    ) -> PreparationKey:
        return branch_id, force, assembly_budget, system_append

    def update_budgets(self, token_budget: int, estimator_budget: int) -> None:
        self._token_budget = token_budget
        self._estimator_budget = estimator_budget
        self.discard()

    def budget(self, reserved_estimator_tokens: int) -> int:
        if reserved_estimator_tokens < 0:
            raise ValueError("reserved estimator tokens must be non-negative")
        return (
            self._token_budget
            if reserved_estimator_tokens == 0
            else max(1, self._estimator_budget - reserved_estimator_tokens)
        )

    async def prepare_and_take(
        self,
        *,
        force: bool,
        reserved_estimator_tokens: int,
        system_append: str,
    ) -> tuple[tuple[ConversationEntry, ...], _AssemblyPreparation]:
        while True:
            assembly_budget = self.budget(reserved_estimator_tokens)
            key = self.key(
                self._store.active_branch_head_id(),
                force=force,
                assembly_budget=assembly_budget,
                system_append=system_append,
            )
            prepared = self.take(key)
            if prepared is not None:
                return prepared
            await self.prepare(
                force=force,
                reserved_estimator_tokens=reserved_estimator_tokens,
                system_append=system_append,
            )

    async def prepare(
        self,
        *,
        force: bool,
        reserved_estimator_tokens: int,
        system_append: str,
    ) -> bool:
        self.discard()
        try:
            while True:
                assembly_budget = self.budget(reserved_estimator_tokens)
                branch = self._store.active_branch_snapshot()
                branch_id = branch[-1].id if branch else None
                preparation = await asyncio.to_thread(
                    cooperative_call,
                    self._prepare,
                    branch,
                    force=force,
                    assembly_budget=assembly_budget,
                    system_append=system_append,
                )
                await asyncio.sleep(0)
                current_budget = self.budget(reserved_estimator_tokens)
                if (
                    self._store.active_branch_head_id() == branch_id
                    and current_budget == assembly_budget
                ):
                    key = self.key(
                        branch_id,
                        force=force,
                        assembly_budget=assembly_budget,
                        system_append=system_append,
                    )
                    self._prepared = (key, (tuple(branch), preparation))
                    return preparation.should_compact
        except asyncio.CancelledError:
            self.discard()
            raise

    def take(self, key: PreparationKey) -> Prepared | None:
        prepared = self._prepared
        self._prepared = None
        if prepared is None or prepared[0] != key:
            return None
        branch, preparation = prepared[1]
        if self._store.active_branch_head_id() != (branch[-1].id if branch else None):
            return None
        return branch, preparation

    def discard(self) -> None:
        self._prepared = None

    invalidate = discard
