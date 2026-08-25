"""Discover and load packaged markdown skills."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from html import escape
from pathlib import Path
import re


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
    values: dict[str, object] = {}
    current_key: str | None = None
    allowed = {"name", "description", "keywords"}
    for line in lines:
        if not line.strip():
            continue
        if line.startswith((" ", "\t")):
            if current_key != "keywords" or not line.lstrip().startswith("-"):
                raise ValueError(
                    f"skill {path} has malformed frontmatter line: {line!r}"
                )
            values.setdefault(current_key, [])
            value = line.lstrip()[1:].strip()
            keyword_values = values[current_key]
            if not value or not isinstance(keyword_values, list):
                raise ValueError(f"skill {path} has malformed keyword list")
            keyword_values.append(_scalar(value, path))
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
            if key == "keywords" and not raw_value
            else _scalar(raw_value, path)
        )

    required = {"name", "description", "keywords"}
    if set(values) != required:
        missing = ", ".join(sorted(required - set(values))) or "none"
        raise ValueError(
            f"skill {path} frontmatter must contain name, description, keywords; "
            f"missing: {missing}"
        )
    name = values["name"]
    if not isinstance(name, str) or not name:
        raise ValueError(f"skill {path} frontmatter name must be a nonempty string")
    description = values["description"]
    if not isinstance(description, str) or not description:
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
    return {"name": name, "description": description, "keywords": keywords}


def _scalar(value: str, path: Path) -> object:
    if not value:
        return ""
    if value.startswith("[") or value.endswith("]"):
        if not value.startswith("[") or not value.endswith("]"):
            raise ValueError(f"skill {path} has malformed list value")
        contents = value[1:-1].strip()
        return [] if not contents else [
            _scalar(item, path) for item in _split_inline_list(contents, path)
        ]
    if value[0] in "\"'" or value[-1] in "\"'":
        if len(value) < 2 or value[0] != value[-1]:
            raise ValueError(f"skill {path} has malformed scalar value: {value!r}")
        return value[1:-1]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "~"}:
        return None
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+)", value):
        return float(value)
    return value


def _split_inline_list(value: str, path: Path) -> list[str]:
    items: list[str] = []
    start = 0
    quote: str | None = None
    for index, character in enumerate(value):
        if quote is not None:
            if character == quote:
                quote = None
        elif character in "\"'":
            quote = character
        elif character == ",":
            item = value[start:index].strip()
            if not item:
                raise ValueError(f"skill {path} has malformed list value")
            items.append(item)
            start = index + 1
    if quote is not None:
        raise ValueError(f"skill {path} has malformed list value")
    item = value[start:].strip()
    if not item:
        raise ValueError(f"skill {path} has malformed list value")
    items.append(item)
    return items


__all__ = [
    "SkillCatalog",
    "SkillMeta",
    "discover_packaged_skills",
    "discover_skills",
    "load_skill",
]
