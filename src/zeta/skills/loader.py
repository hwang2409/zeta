"""Discover and load markdown skills."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SkillMeta:
    name: str
    description: str
    triggers: list[str]
    path: Path


def discover_skills(home: Path) -> list[SkillMeta]:
    skills_dir = home / "skills"
    if not skills_dir.is_dir():
        return []
    discovered: list[SkillMeta] = []
    for path in sorted(skills_dir.rglob("*.md")):
        metadata, _ = _read_skill(path)
        name = metadata.get("name")
        description = metadata.get("description")
        triggers = metadata.get("triggers")
        discovered.append(
            SkillMeta(
                name=name if isinstance(name, str) and name else path.stem,
                description=description if isinstance(description, str) else "",
                triggers=triggers if isinstance(triggers, list) else [],
                path=path,
            )
        )
    return discovered


def load_skill(meta: SkillMeta) -> str:
    _, body = _read_skill(meta.path)
    return body


def _read_skill(path: Path) -> tuple[dict[str, str | list[str]], str]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text.strip()
    try:
        end = next(index for index, line in enumerate(lines[1:], 1) if line.strip() == "---")
    except StopIteration:
        return {}, text.strip()
    frontmatter = _parse_frontmatter(lines[1:end])
    body = "\n".join(lines[end + 1 :]).strip()
    return frontmatter, body


def _parse_frontmatter(lines: list[str]) -> dict[str, str | list[str]]:
    values: dict[str, str | list[str]] = {}
    current_list: list[str] | None = None
    for line in lines:
        if line.startswith((" ", "\t")) and current_list is not None:
            value = line.strip()
            if value.startswith("-"):
                current_list.append(_scalar(value[1:].strip()))
            continue
        if ":" not in line:
            current_list = None
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if key == "triggers" and not raw_value:
            current_list = []
            values[key] = current_list
        else:
            current_list = None
            values[key] = _scalar(raw_value)
    return values


def _scalar(value: str) -> str | list[str]:
    if not value:
        return ""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            if value.startswith("[") and value.endswith("]"):
                return [item.strip().strip("\"'") for item in value[1:-1].split(",") if item.strip()]
            return value.strip("\"'")
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return str(parsed)


__all__ = ["SkillMeta", "discover_skills", "load_skill"]
