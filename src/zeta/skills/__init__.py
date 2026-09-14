"""Skill discovery and loading."""

from .loader import (
    SkillCatalog,
    SkillMeta,
    discover_packaged_skills,
    discover_session_skills,
    discover_skills,
    is_slash_safe_name,
    load_skill,
    load_skill_prompt,
    replace_skill_index,
)

__all__ = [
    "SkillCatalog",
    "SkillMeta",
    "discover_packaged_skills",
    "discover_session_skills",
    "discover_skills",
    "is_slash_safe_name",
    "load_skill",
    "load_skill_prompt",
    "replace_skill_index",
]
