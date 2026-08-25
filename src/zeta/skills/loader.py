"""Discover and load packaged markdown skills."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SkillMeta:
    name: str
    description: str
    keywords: list[str]
    path: Path

    @property
    def triggers(self) -> list[str]:
        """Compatibility name for the ZETA-21 stub field."""

        return self.keywords


@dataclass(frozen=True, slots=True)
class SkillCatalog:
    skills: tuple[SkillMeta, ...]

    def index(self) -> str:
        lines = ["<zeta-skills>", "Available skills:"]
        if not self.skills:
            lines.append("- none")
        for skill in self.skills:
            keywords = ", ".join(skill.keywords) or "none"
            lines.append(
                f"- {skill.name}: {skill.description} (keywords: {keywords})"
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
    for path in sorted(skills_dir.glob("*.md")):
        metadata, _ = _read_skill(path)
        discovered.append(
            SkillMeta(
                name=metadata["name"],
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
    values: dict[str, str | list[str]] = {}
    current_key: str | None = None
    allowed = {"name", "description", "keywords", "triggers"}
    for line in lines:
        if not line.strip():
            continue
        if line.startswith((" ", "\t")):
            if current_key not in {"keywords", "triggers"} or not line.lstrip().startswith(
                "-"
            ):
                raise ValueError(
                    f"skill {path} has malformed frontmatter line: {line!r}"
                )
            values.setdefault(current_key, [])
            value = line.lstrip()[1:].strip()
            if not value or not isinstance(values[current_key], list):
                raise ValueError(f"skill {path} has malformed keyword list")
            values[current_key].append(_scalar(value, path))
            continue
        if ":" not in line:
            raise ValueError(f"skill {path} has malformed frontmatter line: {line!r}")
        key, raw_value = line.split(":", 1)
        key = key.strip()
        if key not in allowed or key in values:
            raise ValueError(f"skill {path} has invalid frontmatter key: {key!r}")
        current_key = key
        raw_value = raw_value.strip()
        values[key] = (
            []
            if key in {"keywords", "triggers"} and not raw_value
            else _scalar(raw_value, path)
        )

    if "keywords" not in values and "triggers" in values:
        values["keywords"] = values.pop("triggers")
    required = {"name", "description", "keywords"}
    if set(values) != required:
        missing = ", ".join(sorted(required - set(values))) or "none"
        raise ValueError(
            f"skill {path} frontmatter must contain name, description, keywords; "
            f"missing: {missing}"
        )
    if not isinstance(values["name"], str) or not values["name"]:
        raise ValueError(f"skill {path} frontmatter name must be a nonempty string")
    if not isinstance(values["description"], str) or not values["description"]:
        raise ValueError(
            f"skill {path} frontmatter description must be a nonempty string"
        )
    keywords = values["keywords"]
    if not isinstance(keywords, list) or any(
        not isinstance(item, str) or not item for item in keywords
    ):
        raise ValueError(
            f"skill {path} frontmatter keywords must be a list of strings"
        )
    return values


def _scalar(value: str, path: Path) -> str | list[str]:
    if not value:
        return ""
    if value.startswith("[") or value.endswith("]"):
        if not value.startswith("[") or not value.endswith("]"):
            raise ValueError(f"skill {path} has malformed list value")
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError) as exc:
                items = [item.strip() for item in value[1:-1].split(",")]
                if not items or any(not item for item in items):
                    raise ValueError(f"skill {path} has malformed list value") from exc
                parsed = [_scalar(item, path) for item in items]
        if not isinstance(parsed, list):
            raise ValueError(f"skill {path} list value is not a list")
        return [str(item) for item in parsed]
    if value[0] in "\"'" or value[-1] in "\"'":
        if value[0] != value[-1]:
            raise ValueError(f"skill {path} has malformed scalar value: {value!r}")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = value
    return parsed if isinstance(parsed, str) else str(parsed)


__all__ = [
    "SkillCatalog",
    "SkillMeta",
    "discover_packaged_skills",
    "discover_skills",
    "load_skill",
]
