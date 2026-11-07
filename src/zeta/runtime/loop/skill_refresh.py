"""Live skill discovery at agent turn boundaries."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...agent.plan_mode import plan_mode_prompt
from ...protocol.types import harness_notice
from ...skills import SkillCatalog, SkillCatalogWatcher, SkillRefresh
from ..prompt import PromptFragment, RuntimePrompt, replace_owned_fragment


@dataclass(frozen=True, slots=True)
class _PreparedSkillPrompt:
    base: RuntimePrompt
    displayed: RuntimePrompt


def _prepare_skill_prompt(
    runtime_prompt: RuntimePrompt, catalog: SkillCatalog, *, plan_mode: bool
) -> _PreparedSkillPrompt | None:
    if "skill_index" not in runtime_prompt.prompt_components:
        return None
    updated = replace_owned_fragment(
        runtime_prompt,
        "skill_index",
        PromptFragment(catalog.index(), {}),
    )
    if not plan_mode:
        return _PreparedSkillPrompt(updated, updated)
    return _PreparedSkillPrompt(
        updated,
        RuntimePrompt(
            plan_mode_prompt(updated.system_prompt), updated.prompt_components
        ),
    )


class SkillRefreshMixin:
    """Keep one running loop's catalog, prompt, and listeners synchronized."""

    tool_registry: Any
    context_assembler: Any
    _plan_mode: bool
    _plan_mode_prior_prompt: RuntimePrompt | None

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
        self._prompt_generation = 0
        self._skill_refresh_notices: list[str] = []
        self._skill_catalog_listeners: list[Callable[[SkillCatalog], None]] = []
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
        self._legacy_skill_restart_noticed = False
        self.tool_registry.set_skill_refresher(self.refresh_skills)

    def _set_runtime_prompt(self, runtime_prompt: RuntimePrompt) -> None:
        self._runtime_prompt = runtime_prompt
        self.context_assembler.system_prompt = runtime_prompt.system_prompt
        self._prompt_generation += 1

    def _skill_prompt_base(self) -> RuntimePrompt:
        if self._plan_mode and self._plan_mode_prior_prompt is not None:
            return self._plan_mode_prior_prompt
        return self._runtime_prompt

    def add_skill_catalog_listener(
        self, listener: Callable[[SkillCatalog], None]
    ) -> None:
        self._skill_catalog_listeners.append(listener)

    async def refresh_skills(self, force: bool = False) -> None:
        """Prepare one catalog and prompt off-loop, then commit if still current."""

        watcher = self._skill_watcher
        if watcher is None:
            return
        self._skill_refresh_generation += 1
        generation = self._skill_refresh_generation
        prompt_generation = self._prompt_generation
        prompt_base = self._skill_prompt_base()
        plan_mode = self._plan_mode
        async with self._skill_refresh_lock:
            scan = await asyncio.to_thread(watcher.scan, force=force)
            if scan is None or generation != self._skill_refresh_generation:
                return
            prepared = (
                await asyncio.to_thread(
                    _prepare_skill_prompt,
                    prompt_base,
                    scan.catalog,
                    plan_mode=plan_mode,
                )
                if scan.catalog is not None
                else None
            )
            if (
                generation != self._skill_refresh_generation
                or prompt_generation != self._prompt_generation
                or prompt_base is not self._skill_prompt_base()
            ):
                return
            refresh = watcher.commit(scan)
            if refresh is None:
                return
            self._commit_skill_refresh(refresh, prepared)

    def _commit_skill_refresh(
        self, refresh: SkillRefresh, prepared: _PreparedSkillPrompt | None
    ) -> None:
        """Assign one fully prepared catalog and prompt on the event loop."""

        self.tool_registry.skill_catalog = refresh.catalog
        if prepared is not None:
            if self._plan_mode:
                self._plan_mode_prior_prompt = prepared.base
            self._set_runtime_prompt(prepared.displayed)
        for listener in self._skill_catalog_listeners:
            listener(refresh.catalog)
        notice = self._skill_refresh_notice(refresh)
        if notice:
            self._skill_refresh_notices.append(notice)
        if (
            prepared is None
            and (refresh.added or refresh.removed or refresh.updated)
            and not self._legacy_skill_restart_noticed
        ):
            self._skill_refresh_notices.append(
                "skills changed, but this session's prompt predates owned skill spans; "
                "restart the session to refresh its skill index"
            )
            self._legacy_skill_restart_noticed = True

    async def _flush_skill_refresh_notices(self) -> None:
        """Persist queued notices only at a new-turn transcript boundary."""

        if not self._skill_refresh_notices:
            return
        count = len(self._skill_refresh_notices)
        notice = "\n".join(self._skill_refresh_notices[:count])
        await self._append_turn_message(
            harness_notice("skill refresh", notice, event="skill_refresh")
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


__all__ = ["SkillRefreshMixin"]
