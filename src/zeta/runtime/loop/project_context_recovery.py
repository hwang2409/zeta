"""Non-blocking recovery for damaged project context."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass

from ...core.project_context import ProjectContext
from ...project_context.projection import (
    ProjectContextHealth,
    ProjectContextProjection,
    _health,
)
from ...project_registry import ProjectRegistry
from ..offload import await_joined_offload
from ..project_registry_state import ProjectRegistryState
from ..prompt import RuntimePrompt, replace_runtime_prompt


@dataclass(frozen=True, slots=True)
class _PreparedProjectContext:
    generation: int
    base: RuntimePrompt
    prompt: RuntimePrompt
    context: ProjectContext


class ProjectContextRecoveryMixin:
    """Advance one runtime-owned recovery pipeline only at turn boundaries."""

    def _configure_project_context_recovery(
        self,
        *,
        registry: ProjectRegistry | None,
        project_id: str | None,
        health: ProjectContextHealth | None,
        prepare: Callable[[ProjectContextProjection], ProjectContext] | None,
    ) -> None:
        self.project_context_health = health
        self._project_context_registry = registry
        self._project_context_project_id = project_id
        self._project_context_prepare = prepare
        self._project_context_read_task: asyncio.Task[ProjectContextProjection] | None = None
        self._project_context_prepare_task: asyncio.Task[_PreparedProjectContext] | None = None
        self._project_context_pending: ProjectContextProjection | None = None

    async def _read_project_context_projection(self) -> ProjectContextProjection:
        assert self._project_context_registry is not None
        return await await_joined_offload(
            self._project_context_registry.project_context_projection,
            self._project_context_project_id,
        )

    async def _prepare_project_context_projection(
        self, projection: ProjectContextProjection
    ) -> _PreparedProjectContext:
        assert self._project_context_prepare is not None
        generation = self._prompt_generation
        base = self._skill_prompt_base()
        context = await await_joined_offload(self._project_context_prepare, projection)
        return _PreparedProjectContext(
            generation,
            base,
            replace_runtime_prompt(
                base, context.system_prompt, context.prompt_components
            ),
            context,
        )

    def _advance_project_context_recovery(self) -> None:
        read = self._project_context_read_task
        if read is not None and read.done():
            self._project_context_read_task = None
            try:
                self._project_context_pending = read.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                logging.getLogger(__name__).warning(
                    "project context projection failed", exc_info=True
                )
                self.project_context_health = _health("project_registry_unavailable")

        prepared = self._project_context_prepare_task
        if prepared is not None and prepared.done():
            self._project_context_prepare_task = None
            try:
                candidate = prepared.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                logging.getLogger(__name__).warning(
                    "project context preparation failed", exc_info=True
                )
                self.project_context_health = _health("project_registry_unavailable")
            else:
                if self._commit_prepared_prompt(
                    candidate.generation, candidate.base, candidate.prompt
                ):
                    self.project_context_health = candidate.context.health
                    if self._project_context_registry is not None and (
                        candidate.context.health is None
                        or candidate.context.health.code
                        != "project_registry_unavailable"
                    ):
                        state = ProjectRegistryState(self._project_context_registry)
                        self.tool_registry.project_registry_state = state
                        manager = getattr(self, "manager", None)
                        if manager is not None:
                            manager.project_registry_state = state
                            manager.project_registry_unavailable = None
                    self._project_context_pending = None

        if (
            self._project_context_pending is not None
            and self._project_context_prepare_task is None
        ):
            self._project_context_prepare_task = asyncio.create_task(
                self._prepare_project_context_projection(self._project_context_pending)
            )
        elif (
            self.project_context_health is not None
            and self._project_context_pending is None
            and self._project_context_read_task is None
            and self._project_context_registry is not None
            and self._project_context_prepare is not None
        ):
            self._project_context_read_task = asyncio.create_task(
                self._read_project_context_projection()
            )

    async def _close_project_context_recovery(self) -> None:
        tasks = tuple(
            task
            for task in (
                self._project_context_read_task,
                self._project_context_prepare_task,
            )
            if task is not None
        )
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logging.getLogger(__name__).debug(
                    "project context recovery ended during close", exc_info=True
                )


__all__ = ["ProjectContextRecoveryMixin"]
