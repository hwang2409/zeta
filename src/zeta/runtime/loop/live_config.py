"""Live catalog refresh at agent turn boundaries."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ...config.live import (
    LiveConfigBase,
    LiveConfigListener,
    LiveConfigScope,
    LiveConfigWatcher,
)
from ...protocol.types import StreamEvent, StreamEventType, harness_notice
from ...skills import SkillCatalog
from ...skills.agent_catalog import AgentCatalog
from ..prompt import RuntimePrompt


class LiveConfigMixin:
    """Keep one running loop's catalogs, prompt, schemas, and surfaces atomic."""

    tool_registry: Any
    context_assembler: Any
    tool_schemas: tuple[dict[str, Any], ...]
    _provided_tool_schemas: bool
    _plan_mode: bool
    _plan_mode_prior_prompt: RuntimePrompt | None

    def _configure_live_config(
        self,
        *,
        skill_catalog: SkillCatalog,
        agent_catalog: AgentCatalog,
        home: Path | None,
        project_dir: Path | None,
        frozen: bool,
        runtime_prompt: RuntimePrompt | None,
    ) -> None:
        self._live_config_scope = (
            LiveConfigScope(home, project_dir)
            if not frozen and home is not None and project_dir is not None
            else None
        )
        self._live_config_watcher = (
            LiveConfigWatcher(
                self._live_config_scope,
                skill_catalog=skill_catalog,
                agent_catalog=agent_catalog,
            )
            if self._live_config_scope is not None
            else None
        )
        self._live_config_lock = asyncio.Lock()
        self._live_config_generation = 0
        self._prompt_generation = 0
        self._live_config_notices: list[str] = []
        self._live_config_listeners: list[LiveConfigListener] = []
        self._live_skill_catalog_snapshot: list[dict[str, object]] | None = None
        self._live_agent_catalog_snapshot: list[dict[str, object]] | None = None
        self._runtime_prompt = RuntimePrompt(
            self.context_assembler.system_prompt,
            (
                {
                    key: dict(component)
                    for key, component in runtime_prompt.prompt_components.items()
                }
                if runtime_prompt is not None
                else {}
            ),
        )
        self.tool_registry.set_live_config_refresher(self.refresh_live_config)

    def _set_runtime_prompt(self, runtime_prompt: RuntimePrompt) -> None:
        self._runtime_prompt = runtime_prompt
        self.context_assembler.system_prompt = runtime_prompt.system_prompt
        self._prompt_generation += 1

    def _live_prompt_base(self) -> RuntimePrompt:
        if self._plan_mode and self._plan_mode_prior_prompt is not None:
            return self._plan_mode_prior_prompt
        return self._runtime_prompt

    def add_live_config_listener(self, listener: LiveConfigListener) -> None:
        self._live_config_listeners.append(listener)

    async def refresh_live_config(
        self, force: bool = False, *, publish: bool = True
    ) -> None:
        """Prepare one complete candidate off-loop and commit it if still current."""

        watcher = self._live_config_watcher
        if watcher is None:
            return
        self._live_config_generation += 1
        generation = self._live_config_generation
        base = LiveConfigBase(
            generation=generation,
            prompt_generation=self._prompt_generation,
            runtime_prompt=self._runtime_prompt,
            prompt_base=self._live_prompt_base(),
            plan_mode=self._plan_mode,
            skill_catalog=self.tool_registry.skill_catalog,
            agent_catalog=self.tool_registry.agent_catalog,
            agent_definition=self.tool_registry.definition_reference("agent"),
            tool_schemas=self.tool_schemas,
            explicit_tool_schemas=self._provided_tool_schemas,
            listeners=tuple(self._live_config_listeners),
        )
        async with self._live_config_lock:
            candidate = await asyncio.to_thread(watcher.prepare, base, force=force)
            if candidate is None or not self._live_config_base_is_current(base):
                return
            watcher.commit(candidate)
            if not candidate.changed:
                return
            self.tool_registry.skill_catalog = candidate.skill_catalog
            self.tool_registry.agent_catalog = candidate.agent_catalog
            self.agent_catalog = candidate.agent_catalog
            if candidate.agent_definition is not None:
                self.tool_registry.commit_prepared_definition(
                    base.agent_definition, candidate.agent_definition
                )
            self.tool_schemas = candidate.tool_schemas
            if candidate.runtime_prompt is not base.runtime_prompt:
                if self._plan_mode:
                    self._plan_mode_prior_prompt = candidate.prompt_base
                self._set_runtime_prompt(candidate.runtime_prompt)
            for listener, prepared in candidate.listener_updates:
                listener.commit_live_config(prepared)
            self._live_skill_catalog_snapshot = candidate.skill_snapshot
            self._live_agent_catalog_snapshot = candidate.agent_snapshot
            if publish and candidate.notice:
                self._live_config_notices.append(candidate.notice)
            if publish and candidate.payload is not None:
                self._publish_background_event(
                    StreamEvent(StreamEventType.CONFIG_CHANGED, data=candidate.payload)
                )

    def _live_config_base_is_current(self, base: LiveConfigBase) -> bool:
        return (
            base.generation == self._live_config_generation
            and base.prompt_generation == self._prompt_generation
            and base.runtime_prompt is self._runtime_prompt
            and base.prompt_base is self._live_prompt_base()
            and base.skill_catalog is self.tool_registry.skill_catalog
            and base.agent_catalog is self.tool_registry.agent_catalog
            and base.agent_definition
            is self.tool_registry.definition_reference("agent")
            and base.tool_schemas is self.tool_schemas
            and base.listeners == tuple(self._live_config_listeners)
        )

    async def _flush_live_config_notices(self) -> None:
        """Persist queued notices only at a new-turn transcript boundary."""

        if not self._live_config_notices:
            return
        count = len(self._live_config_notices)
        notice = "\n".join(self._live_config_notices[:count])
        await self._append_turn_message(
            harness_notice("skill refresh", notice, event="skill_refresh")
        )
        del self._live_config_notices[:count]


__all__ = ["LiveConfigMixin"]
