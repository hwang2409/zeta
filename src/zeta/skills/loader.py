"""Discover and load packaged markdown skills."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from html import escape
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class SkillMeta:
    name: str
    description: str
    keywords: list[str]
    path: Path

@dataclass(frozen=True, slots=True)
class SkillCatalog:
    skills: tuple[SkillMeta, ...]

    def index(self) -> str:
        lines = ["<zeta-skills>", "Available skills:"]
        if not self.skills:
            lines.append("- none")
        for skill in self.skills:
            lines.append(
                f"- {escape(skill.name, quote=True)}: "
                f"{escape(skill.description, quote=True)}"
            )
        lines.append("</zeta-skills>")
        return "\n".join(lines)

    def load(self, name: str) -> str:
        matches = [skill for skill in self.skills if skill.name == name]
        if not matches:
            available = ", ".join(skill.name for skill in self.skills) or "none"
            raise ValueError(
                f"unknown skill {name!r}; available skills: {available}"
            )
        return load_skill(matches[0])


def _skills_dir(home: Path) -> Path:
    return home if home.name == "skills" else home / "skills"


def discover_skills(home: Path) -> list[SkillMeta]:
    """Discover strict markdown skill files below ``home/skills``."""

    skills_dir = _skills_dir(home)
    if not skills_dir.is_dir():
        return []
    discovered: list[SkillMeta] = []
    paths_by_name: dict[str, Path] = {}
    for path in sorted(skills_dir.glob("*.md")):
        metadata, _ = _read_skill(path)
        name = metadata["name"]
        assert isinstance(name, str)
        previous_path = paths_by_name.get(name)
        if previous_path is not None:
            raise ValueError(
                f"duplicate skill name {name!r} in {previous_path} and {path}"
            )
        paths_by_name[name] = path
        discovered.append(
            SkillMeta(
                name=name,
                description=metadata["description"],
                keywords=metadata["keywords"],
                path=path,
            )
        )
    return discovered


@cache
def discover_packaged_skills() -> SkillCatalog:
    """Discover the skills shipped in the installed zeta package once."""

    return SkillCatalog(tuple(discover_skills(Path(__file__).parent)))


def load_skill(meta: SkillMeta) -> str:
    """Load the prompt body for one discovered skill."""

    _, body = _read_skill(meta.path)
    return body


def _read_skill(path: Path) -> tuple[dict[str, str | list[str]], str]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"skill {path} is missing YAML frontmatter")
    try:
        end = next(
            index for index, line in enumerate(lines[1:], 1) if line.strip() == "---"
        )
    except StopIteration as exc:
        raise ValueError(f"skill {path} has unterminated YAML frontmatter") from exc
    metadata = _parse_frontmatter(lines[1:end], path)
    body = "\n".join(lines[end + 1 :]).strip()
    if not body:
        raise ValueError(f"skill {path} has an empty prompt body")
    return metadata, body


def _parse_frontmatter(
    lines: list[str], path: Path
) -> dict[str, str | list[str]]:
    try:
        values = yaml.safe_load("\n".join(lines))
    except yaml.YAMLError as exc:
        raise ValueError(f"skill {path} has invalid YAML frontmatter") from exc
    required = {"name", "description", "keywords"}
    if not isinstance(values, dict):
        raise ValueError(f"skill {path} frontmatter must be a mapping")
    if set(values) != required:
        missing = ", ".join(sorted(required - set(values))) or "none"
        raise ValueError(
            f"skill {path} frontmatter must contain name, description, keywords; "
            f"missing: {missing}"
        )
    name = values["name"]
    if type(name) is not str or not name.strip():
        raise ValueError(f"skill {path} frontmatter name must be a nonempty string")
    description = values["description"]
    if type(description) is not str or not description.strip():
        raise ValueError(
            f"skill {path} frontmatter description must be a nonempty string"
        )
    keywords = values["keywords"]
    if type(keywords) is not list or any(
        type(item) is not str or not item.strip() for item in keywords
    ):
        raise ValueError(
            f"skill {path} frontmatter keywords must be a list of strings"
        )
    return {"name": name, "description": description, "keywords": keywords}


__all__ = [
    "SkillCatalog",
    "SkillMeta",
    "discover_packaged_skills",
    "discover_skills",
    "load_skill",
]
