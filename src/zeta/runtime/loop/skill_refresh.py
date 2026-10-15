"""Live skill discovery at agent turn boundaries."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ...agent.plan_mode import plan_mode_prompt
from ...protocol.types import Message, MessageOrigin, MessageRole, TextContent
from ...skills import (
    SkillCatalog,
    SkillCatalogWatcher,
    SkillRefresh,
    replace_skill_index_metadata,
)


@dataclass(frozen=True, slots=True)
class RuntimePrompt:
    """One live prompt and the metadata that locates its owned spans."""

    system_prompt: str | Message
    prompt_components: dict[str, dict[str, int | str]]


class SkillRefreshMixin:
    """Keep one running loop's catalog, prompt, and listeners synchronized."""

    tool_registry: Any
    context_assembler: Any
    _plan_mode: bool
    _plan_mode_prior_prompt: str | Message | None

    def _configure_skill_refresh(
        self,
        *,
        catalog: SkillCatalog,
        home: Path | None,
        project_dir: Path | None,
        frozen: bool,
        runtime_prompt: RuntimePrompt | None,
    ) -> None:
        self._skill_watcher = (
            SkillCatalogWatcher(catalog, home=home, project_dir=project_dir)
            if not frozen and home is not None and project_dir is not None
            else None
        )
        self._skill_refresh_lock = asyncio.Lock()
        self._skill_refresh_generation = 0
        self._skill_refresh_notices: list[str] = []
        self._skill_catalog_listeners: list[Callable[[SkillCatalog], None]] = []
        self._runtime_prompt = (
            RuntimePrompt(
                self.context_assembler.system_prompt,
                {
                    key: dict(component)
                    for key, component in runtime_prompt.prompt_components.items()
                },
            )
            if runtime_prompt is not None
            else None
        )
        self._legacy_skill_prompt_components: (
            dict[str, dict[str, int | str]] | None
        ) = None
        self.tool_registry.set_skill_refresher(self.refresh_skills)

    def _set_runtime_prompt(self, runtime_prompt: RuntimePrompt) -> None:
        self._runtime_prompt = runtime_prompt
        self.context_assembler.system_prompt = runtime_prompt.system_prompt

    def add_skill_catalog_listener(
        self, listener: Callable[[SkillCatalog], None]
    ) -> None:
        self._skill_catalog_listeners.append(listener)

    async def refresh_skills(self, force: bool = False) -> None:
        """Scan off-thread and commit one ordered catalog snapshot on the loop."""

        watcher = self._skill_watcher
        if watcher is None:
            return
        self._skill_refresh_generation += 1
        generation = self._skill_refresh_generation
        async with self._skill_refresh_lock:
            scan = await asyncio.to_thread(watcher.scan, force=force)
            if scan is None or generation != self._skill_refresh_generation:
                return
            refresh = watcher.commit(scan)
            if refresh is None:
                return
            self._commit_skill_refresh(refresh)

    def _commit_skill_refresh(self, refresh: SkillRefresh) -> None:
        """Apply all live catalog state synchronously on the event loop."""

        self.tool_registry.skill_catalog = refresh.catalog
        if self._runtime_prompt is not None:
            runtime_prompt = (
                self._plan_mode_prior_prompt
                if self._plan_mode and self._plan_mode_prior_prompt is not None
                else self._runtime_prompt
            )
            prompt = runtime_prompt.system_prompt
            components = runtime_prompt.prompt_components
        else:
            prompt = (
                self._plan_mode_prior_prompt
                if self._plan_mode and self._plan_mode_prior_prompt is not None
                else self.context_assembler.system_prompt
            )
            if self._legacy_skill_prompt_components is None:
                metadata = getattr(self, "session_metadata", None)
                components = (
                    {
                        key: dict(value)
                        for key, value in metadata.prompt_components.items()
                    }
                    if metadata is not None
                    else {}
                )
                if (
                    metadata is not None
                    and "project_memory" not in components
                    and metadata.project_memory_offset is not None
                    and metadata.project_memory_length is not None
                    and metadata.project_memory_digest is not None
                ):
                    components["project_memory"] = {
                        "offset": metadata.project_memory_offset,
                        "length": metadata.project_memory_length,
                        "digest": metadata.project_memory_digest,
                    }
            else:
                components = self._legacy_skill_prompt_components
        if isinstance(prompt, Message):
            prompt_text = "".join(
                block.text
                for block in prompt.content
                if isinstance(block, TextContent)
            )
            prompt_text, components = replace_skill_index_metadata(
                prompt_text, refresh.catalog, components
            )
            prompt = replace(prompt, content=[TextContent(prompt_text)])
            if self._runtime_prompt is not None:
                runtime_prompt = RuntimePrompt(prompt, components)
                if self._plan_mode:
                    self._plan_mode_prior_prompt = runtime_prompt
                    self._set_runtime_prompt(
                        RuntimePrompt(
                            plan_mode_prompt(prompt), runtime_prompt.prompt_components
                        )
                    )
                else:
                    self._set_runtime_prompt(runtime_prompt)
            else:
                self._legacy_skill_prompt_components = components
                if self._plan_mode:
                    self._plan_mode_prior_prompt = prompt
                    self.context_assembler.system_prompt = plan_mode_prompt(prompt)
                else:
                    self.context_assembler.system_prompt = prompt
        for listener in self._skill_catalog_listeners:
            listener(refresh.catalog)
        notice = self._skill_refresh_notice(refresh)
        if notice:
            self._skill_refresh_notices.append(notice)

    async def _flush_skill_refresh_notices(self) -> None:
        """Persist queued notices only at a new-turn transcript boundary."""

        if not self._skill_refresh_notices:
            return
        count = len(self._skill_refresh_notices)
        notice = "\n".join(self._skill_refresh_notices[:count])
        await self._append_turn_message(
            Message(
                MessageRole.USER,
                [TextContent(notice)],
                metadata={"zeta.origin": MessageOrigin.HARNESS_NUDGE.value},
            )
        )
        del self._skill_refresh_notices[:count]

    @staticmethod
    def _skill_refresh_notice(refresh: SkillRefresh) -> str:
        changes = []
        if refresh.added:
            changes.append(f"added {', '.join(refresh.added)}")
        if refresh.removed:
            changes.append(f"removed {', '.join(refresh.removed)}")
        if refresh.updated:
            changes.append(f"updated {', '.join(refresh.updated)}")
        messages = [f"skills changed: {'; '.join(changes)}"] if changes else []
        messages.extend(refresh.notices)
        return "\n".join(messages)


__all__ = ["RuntimePrompt", "SkillRefreshMixin"]
