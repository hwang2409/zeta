"""One-shot asynchronous context preparation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Protocol

from ...context_accounting import cooperative_call
from ...protocol.types import Message
from ..store import ConversationEntry, ConversationStore
from .types import _AssemblyPreparation

PreparationKey = tuple[str | None, bool, int, str, int]
Prepared = tuple[tuple[ConversationEntry, ...], _AssemblyPreparation]


class PrepareAssembly(Protocol):
    def __call__(
        self,
        branch: Sequence[ConversationEntry],
        *,
        force: bool,
        assembly_budget: int,
        system_append: str,
        system_prompt: Message,
    ) -> _AssemblyPreparation: ...


class ContextPreparation:
    """Own one cancellable, stale-head-checked assembly preparation."""

    def __init__(
        self,
        store: ConversationStore,
        prepare: PrepareAssembly,
        token_budget: int,
        estimator_budget: int,
        prompt_state: Callable[[], tuple[int, Message]],
    ) -> None:
        self._store = store
        self._prepare = prepare
        self._token_budget = token_budget
        self._estimator_budget = estimator_budget
        self._prompt_state = prompt_state
        self._prepared: tuple[PreparationKey, Prepared] | None = None

    @staticmethod
    def key(
        branch_id: str | None,
        *,
        force: bool,
        assembly_budget: int,
        system_append: str,
        prompt_generation: int,
    ) -> PreparationKey:
        return branch_id, force, assembly_budget, system_append, prompt_generation

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
            prompt_generation, _ = self._prompt_state()
            key = self.key(
                self._store.active_branch_head_id(),
                force=force,
                assembly_budget=assembly_budget,
                system_append=system_append,
                prompt_generation=prompt_generation,
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
                branch = self._tuple(store.entry(ref) for ref in store.branch_refs())
                branch_id = branch[-1].id if branch else None
                prompt_generation, system_prompt = self._prompt_state()
                preparation = await asyncio.to_thread(
                    cooperative_call,
                    self._prepare,
                    branch,
                    force=force,
                    assembly_budget=assembly_budget,
                    system_append=system_append,
                    system_prompt=system_prompt,
                )
                await asyncio.sleep(0)
                current_budget = self.budget(reserved_estimator_tokens)
                if (
                    self._store.active_branch_head_id() == branch_id
                    and current_budget == assembly_budget
                    and self._prompt_state()[0] == prompt_generation
                ):
                    key = self.key(
                        branch_id,
                        force=force,
                        assembly_budget=assembly_budget,
                        system_append=system_append,
                        prompt_generation=prompt_generation,
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
        if self._prompt_state()[0] != key[-1]:
            return None
        return branch, preparation

    def discard(self) -> None:
        self._prepared = None

    invalidate = discard
