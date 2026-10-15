"""Live skill discovery at agent turn boundaries."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import replace
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
    ) -> None:
        self._skill_watcher = (
            SkillCatalogWatcher(catalog, home=home, project_dir=project_dir)
            if home is not None and project_dir is not None
            else None
        )
        self._skill_catalog_listeners: list[Callable[[SkillCatalog], None]] = []
        self.tool_registry.set_skill_refresher(self.refresh_skills)

    def add_skill_catalog_listener(
        self, listener: Callable[[SkillCatalog], None]
    ) -> None:
        self._skill_catalog_listeners.append(listener)

    async def refresh_skills(self, force: bool = False) -> None:
        """Refresh changed skills without filesystem work on the event loop."""

        if self._skill_watcher is None:
            return
        refresh = await asyncio.to_thread(self._skill_watcher.refresh, force=force)
        if refresh is None:
            return
        self.tool_registry.skill_catalog = refresh.catalog
        active_prompt = self.context_assembler.system_prompt
        prompt = (
            self._plan_mode_prior_prompt
            if self._plan_mode and self._plan_mode_prior_prompt is not None
            else active_prompt
        )
        if isinstance(prompt, Message):
            prompt_text = "".join(
                block.text
                for block in prompt.content
                if isinstance(block, TextContent)
            )
            metadata = getattr(self, "session_metadata", None)
            components = (
                {key: dict(value) for key, value in metadata.prompt_components.items()}
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
            prompt_text, components = replace_skill_index_metadata(
                prompt_text, refresh.catalog, components
            )
            prompt = replace(prompt, content=[TextContent(prompt_text)])
            if self._plan_mode:
                self._plan_mode_prior_prompt = prompt
                self.context_assembler.system_prompt = plan_mode_prompt(prompt)
            else:
                self.context_assembler.system_prompt = prompt
            manager = getattr(self, "manager", None)
            if metadata is not None and manager is not None:
                await asyncio.to_thread(
                    manager.record_skill_catalog,
                    metadata,
                    catalog=refresh.catalog,
                    system_prompt=prompt_text,
                    prompt_components=components,
                )
        for listener in self._skill_catalog_listeners:
            listener(refresh.catalog)
        notice = self._skill_refresh_notice(refresh)
        if notice:
            await self._append_turn_message(
                Message(
                    MessageRole.USER,
                    [TextContent(notice)],
                    metadata={"zeta.origin": MessageOrigin.HARNESS_NUDGE.value},
                )
            )

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


__all__ = ["SkillRefreshMixin"]
