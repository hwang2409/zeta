"""Live skill discovery at agent turn boundaries."""

from __future__ import annotations

import asyncio
import logging
import threading
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

_SKILL_PERSISTENCE_FLUSH_TIMEOUT = 0.5
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _SkillCatalogWrite:
    generation: int
    action: Callable[[], None]


class _SkillCatalogPersistence:
    """Coalesce skill metadata writes onto one ordered daemon worker."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: _SkillCatalogWrite | None = None
        self._thread: threading.Thread | None = None
        self._active_generation = 0
        self._start_scheduled = False
        self._settled_generation = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._changed: asyncio.Event | None = None
        self._reported_failure = False

    def submit(self, generation: int, action: Callable[[], None]) -> None:
        loop = asyncio.get_running_loop()
        with self._lock:
            self._pending = _SkillCatalogWrite(generation, action)
            self._loop = loop
            if self._changed is None:
                self._changed = asyncio.Event()
            self._changed.clear()
            if self._thread is None and not self._start_scheduled:
                self._start_scheduled = True
                loop.call_soon(self._start)

    def _start(self) -> None:
        with self._lock:
            self._start_scheduled = False
            if self._thread is not None or self._pending is None:
                return
            self._thread = threading.Thread(
                target=self._run,
                name="zeta-skill-catalog-persistence",
                daemon=True,
            )
            self._thread.start()

    def _run(self) -> None:
        while True:
            with self._lock:
                write = self._pending
                self._pending = None
                if write is None:
                    self._active_generation = 0
                    self._thread = None
                    return
                self._active_generation = write.generation
            try:
                write.action()
            except Exception as exc:  # noqa: BLE001 - persistence cannot stop turns
                with self._lock:
                    self._settled_generation = max(
                        self._settled_generation, write.generation
                    )
                    has_newer_write = self._pending is not None
                    if not has_newer_write:
                        self._active_generation = 0
                        self._thread = None
                    report = not self._reported_failure
                    self._reported_failure = True
                if report:
                    _LOGGER.warning("failed to persist live skill catalog: %s", exc)
                self._signal_change()
                if not has_newer_write:
                    return
            with self._lock:
                self._settled_generation = max(
                    self._settled_generation, write.generation
                )
            self._signal_change()

    def _signal_change(self) -> None:
        with self._lock:
            loop = self._loop
            changed = self._changed
        if loop is not None and changed is not None:
            try:
                loop.call_soon_threadsafe(changed.set)
            except RuntimeError:
                pass

    async def flush(self, timeout: float | None = None) -> None:
        self._start()
        if timeout is None:
            timeout = _SKILL_PERSISTENCE_FLUSH_TIMEOUT
        with self._lock:
            target = max(
                self._active_generation,
                self._pending.generation if self._pending is not None else 0,
            )
            changed = self._changed
        if changed is None or self._is_settled(target):
            return

        async def wait() -> None:
            while not self._is_settled(target):
                await changed.wait()
                changed.clear()

        try:
            await asyncio.wait_for(wait(), timeout)
        except TimeoutError:
            _LOGGER.warning("timed out flushing live skill catalog persistence")

    def _is_settled(self, generation: int) -> bool:
        with self._lock:
            return self._settled_generation >= generation


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
        self._skill_catalog_persistence = _SkillCatalogPersistence()
        self.tool_registry.set_skill_refresher(self.refresh_skills)

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
            self._commit_skill_refresh(refresh, generation=generation)

    def _commit_skill_refresh(
        self, refresh: SkillRefresh, *, generation: int
    ) -> None:
        """Apply all live catalog state synchronously on the event loop."""

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
                manager.apply_skill_catalog(
                    metadata,
                    catalog=refresh.catalog,
                    system_prompt=prompt_text,
                    prompt_components=components,
                )
                session_id = metadata.session_id
                self._skill_catalog_persistence.submit(
                    generation,
                    lambda: manager.record_skill_catalog(
                        session_id,
                        catalog=refresh.catalog,
                        system_prompt=prompt_text,
                        prompt_components=components,
                    ),
                )
        for listener in self._skill_catalog_listeners:
            listener(refresh.catalog)
        notice = self._skill_refresh_notice(refresh)
        if notice:
            self._skill_refresh_notices.append(notice)

    async def _flush_skill_catalog_persistence(self) -> None:
        await self._skill_catalog_persistence.flush()

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


__all__ = ["SkillRefreshMixin"]
