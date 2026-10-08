"""Persistence for session prompt recipes and composed snapshots."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...skills import SkillCatalog
from ...skills.agent_catalog import AgentCatalog
from ..session_files import SessionError


def parse_prompt_recipe(
    value: Mapping[str, Any],
    *,
    system_prompt: object,
    has_context_snapshot: bool,
    path: Path,
) -> tuple[str | None, dict[str, dict[str, int | str]]]:
    """Validate persisted prompt ownership without marker scanning."""

    if not has_context_snapshot:
        return None, {}
    recipe = value.get("prompt_recipe")
    raw_components = value.get("prompt_components", {})
    if recipe not in (None, "default", "custom") or type(raw_components) is not dict:
        raise SessionError(f"session prompt recipe is invalid: {path}")
    components: dict[str, dict[str, int | str]] = {}
    for key, component in raw_components.items():
        if (
            type(system_prompt) is not str
            or type(key) is not str
            or not key
            or type(component) is not dict
            or set(component) != {"offset", "length", "digest"}
            or type(component.get("offset")) is not int
            or type(component.get("length")) is not int
            or type(component.get("digest")) is not str
        ):
            raise SessionError(f"session prompt components are invalid: {path}")
        start, length = component["offset"], component["length"]
        end = start + length
        digest = component["digest"]
        if (
            start < 0
            or length < 0
            or end > len(system_prompt)
            or hashlib.sha256(system_prompt[start:end].encode()).hexdigest() != digest
        ):
            raise SessionError(f"session prompt components are invalid: {path}")
        components[key] = dict(component)
    if recipe is None and components:
        raise SessionError(f"session prompt components require a recipe: {path}")
    return recipe, components


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
