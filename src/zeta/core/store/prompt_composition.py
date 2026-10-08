"""Persistence for session prompt recipes and composed snapshots."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
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


def clone_prompt_composition(metadata: Any) -> dict[str, Any]:
    """Return the complete prompt-composition arguments for a cloned session."""

    return {
        "system_prompt": metadata.system_prompt,
        "context_files": list(metadata.context_files),
        "skill_catalog": (
            SkillCatalog.from_snapshot(metadata.skill_catalog)
            if metadata.skill_catalog is not None
            else None
        ),
        "agent_catalog": (
            AgentCatalog.from_snapshot(metadata.agent_catalog)
            if metadata.agent_catalog is not None
            else None
        ),
        "project_memory_offset": metadata.project_memory_offset,
        "project_memory_length": metadata.project_memory_length,
        "project_memory_digest": metadata.project_memory_digest,
        "prompt_recipe": metadata.prompt_recipe,
        "prompt_components": {
            key: dict(component)
            for key, component in metadata.prompt_components.items()
        },
    }


@dataclass(frozen=True, slots=True)
class PromptComposition:
    """One complete prompt composition returned by a resume composer."""

    system_prompt: str
    context_files: tuple[str, ...]
    skill_catalog: SkillCatalog
    agent_catalog: AgentCatalog
    prompt_recipe: str | None
    prompt_components: dict[str, dict[str, int | str]]
    project_memory_offset: int | None
    project_memory_length: int | None
    project_memory_digest: str | None
    notices: tuple[str, ...] = ()


def _composition_from_metadata(metadata: Any) -> PromptComposition:
    return PromptComposition(
        system_prompt=metadata.system_prompt,
        context_files=tuple(metadata.context_files),
        skill_catalog=SkillCatalog.from_snapshot(metadata.skill_catalog or []),
        agent_catalog=AgentCatalog.from_snapshot(metadata.agent_catalog or []),
        prompt_recipe=metadata.prompt_recipe,
        prompt_components={
            key: dict(component)
            for key, component in metadata.prompt_components.items()
        },
        project_memory_offset=metadata.project_memory_offset,
        project_memory_length=metadata.project_memory_length,
        project_memory_digest=metadata.project_memory_digest,
    )


def _persisted_fields(composition: PromptComposition) -> tuple[object, ...]:
    return (
        composition.system_prompt,
        composition.context_files,
        composition.skill_catalog,
        composition.agent_catalog,
        composition.prompt_recipe,
        composition.prompt_components,
        composition.project_memory_offset,
        composition.project_memory_length,
        composition.project_memory_digest,
    )


class PromptCompositionMixin:
    """Own prompt composition persistence and concurrent resume coordination."""


    def resume_prompt_composition(
        self,
        metadata: Any,
        store: Any,
        compose: Callable[[Any], PromptComposition],
    ) -> PromptComposition:
        """Atomically compose or adopt a prompt and activate its runtime lease.

        The metadata lock covers the lease probe, persistence, and transition
        to a shared runtime lease. A successful exclusive probe means no other
        runtime has this session open, so this runtime recomposes. Otherwise it
        adopts the composition published by the live runtime.
        """

        if store.session_id != metadata.session_id:
            raise SessionError("prompt runtime lease does not match session metadata")
        with self._metadata_lock(metadata.session_id) as directory_fd:
            current = self._read(metadata.session_id, directory_fd=directory_fd)
            with store.prompt_resume_lease() as should_compose:
                if should_compose:
                    composition = compose(current)
                    if _persisted_fields(composition) != _persisted_fields(
                        _composition_from_metadata(current)
                    ):
                        current.system_prompt = composition.system_prompt
                        current.context_files = list(composition.context_files)
                        current.skill_catalog = composition.skill_catalog.to_snapshot()
                        current.agent_catalog = composition.agent_catalog.to_snapshot()
                        current.prompt_recipe = composition.prompt_recipe
                        current.prompt_components = {
                            key: dict(component)
                            for key, component in composition.prompt_components.items()
                        }
                        current.project_memory_offset = (
                            composition.project_memory_offset
                        )
                        current.project_memory_length = (
                            composition.project_memory_length
                        )
                        current.project_memory_digest = (
                            composition.project_memory_digest
                        )
                        self._write_unlocked(current, directory_fd=directory_fd)
                else:
                    composition = _composition_from_metadata(current)
                self._copy_metadata(metadata, current)
                return composition
