"""Public compatibility surface for the session skill catalog."""

from .catalog import (
    SKILL_INDEX_BYTE_LIMIT,
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
    "SKILL_INDEX_BYTE_LIMIT",
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
