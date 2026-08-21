"""Monotonic cooperative abort generations."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass(frozen=True, slots=True, init=False)
class AbortSignal:
    """Immutable token for one abort generation."""

    _registry: AbortGenerationRegistry
    generation: int

    def __init__(
        self,
        registry: AbortGenerationRegistry | None = None,
        generation: int | None = None,
    ) -> None:
        if registry is None:
            registry = AbortGenerationRegistry()
            generation = registry._allocate_generation()
        if generation is None:
            raise ValueError("abort signal requires a generation")
        object.__setattr__(self, "_registry", registry)
        object.__setattr__(self, "generation", generation)

    def abort(self) -> None:
        self._registry.abort(self.generation)

    def is_set(self) -> bool:
        return self._registry.is_aborted(self.generation)

    async def wait(self) -> None:
        await self._registry.wait(self.generation)

    @property
    def aborted(self) -> bool:
        return self.is_set()

    @property
    def registry(self) -> AbortGenerationRegistry:
        return self._registry


class AbortGenerationRegistry:
    """Own generation ids and sticky abort state."""

    def __init__(self) -> None:
        self._next_generation = 0
        self._aborted: set[int] = set()
        self._waiters: dict[int, asyncio.Future[None]] = {}

    def _allocate_generation(self) -> int:
        self._next_generation += 1
        return self._next_generation

    def new_generation(self) -> AbortSignal:
        return AbortSignal(self, self._allocate_generation())

    def abort(self, generation: int) -> None:
        if generation in self._aborted:
            return
        self._aborted.add(generation)
        waiter = self._waiters.pop(generation, None)
        if waiter is not None and not waiter.done():
            waiter.set_result(None)

    def is_aborted(self, generation: int) -> bool:
        return generation in self._aborted

    async def wait(self, generation: int) -> None:
        if generation in self._aborted:
            return
        loop = asyncio.get_running_loop()
        waiter = self._waiters.get(generation)
        if waiter is None:
            waiter = loop.create_future()
            self._waiters[generation] = waiter
        await asyncio.shield(waiter)
