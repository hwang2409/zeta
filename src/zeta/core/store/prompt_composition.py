"""Persistence for session prompt recipes and composed snapshots."""

from __future__ import annotations

from typing import Any

from ...skills import SkillCatalog
from ...skills.agent_catalog import AgentCatalog


class PromptCompositionMixin:
    """Persist complete prompt compositions without exposing storage details."""

    def persist_context_snapshot(
        self,
        metadata: Any,
        *,
        system_prompt: str,
        context_files: list[str] | tuple[str, ...],
        overwrite: bool = False,
    ) -> Any:
        """Persist a legacy prompt snapshot, first-write-wins by default."""

        def update(item: Any) -> Any:
            if item.system_prompt and not overwrite:
                return item
            item.system_prompt = system_prompt
            item.context_files = list(context_files)
            return self._touch(item) if overwrite else item

        current = self._mutate(metadata.session_id, update)
        self._copy_metadata(metadata, current)
        return current

    def persist_prompt_composition(
        self,
        metadata: Any,
        *,
        system_prompt: str,
        context_files: list[str] | tuple[str, ...],
        skill_catalog: SkillCatalog,
        agent_catalog: AgentCatalog,
        prompt_recipe: str,
        prompt_components: dict[str, dict[str, int | str]],
        project_memory_offset: int | None,
        project_memory_length: int | None,
        project_memory_digest: str | None,
    ) -> Any:
        """Persist one automatic resume recomposition without changing recency."""

        expected_recipe = metadata.prompt_recipe

        def update(item: Any) -> Any:
            # Concurrent first resumes of a legacy session adopt one complete
            # composition. Established recipes may be recomposed on each run.
            if expected_recipe is None and item.prompt_recipe is not None:
                return item
            item.system_prompt = system_prompt
            item.context_files = list(context_files)
            item.skill_catalog = skill_catalog.to_snapshot()
            item.agent_catalog = agent_catalog.to_snapshot()
            item.prompt_recipe = prompt_recipe
            item.prompt_components = {
                key: dict(component) for key, component in prompt_components.items()
            }
            item.project_memory_offset = project_memory_offset
            item.project_memory_length = project_memory_length
            item.project_memory_digest = project_memory_digest
            return item

        current = self._mutate(metadata.session_id, update)
        self._copy_metadata(metadata, current)
        return current
