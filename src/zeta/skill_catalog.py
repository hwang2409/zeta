"""Discover and load markdown skills for one session."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from html import escape
from pathlib import Path

import yaml

_logger = logging.getLogger(__name__)
SKILL_INDEX_BYTE_LIMIT = 32 * 1024
_PACKAGED_SKILLS_DIR = Path(__file__).parent / "skills"


@dataclass(frozen=True, slots=True)
class SkillMeta:
    name: str
    description: str
    keywords: list[str]
    path: Path
    source: str = ""


@dataclass(frozen=True, slots=True)
class SkillCatalog:
    skills: tuple[SkillMeta, ...]
    notices: tuple[str, ...] = ()

    def index(self) -> str:
        """Render a bounded, deterministic skill index for the system prompt."""

        prefix = ["<zeta-skills>", "Available skills:"]
        suffix = ["</zeta-skills>"]
        entries = [
            f"- {escape(skill.name, quote=True)}: "
            f"{escape(skill.description, quote=True)}"
            for skill in self.skills
        ]
        if not entries:
            return "\n".join(prefix + ["- none"] + suffix)
        selected = entries[:]
        while True:
            omitted = len(entries) - len(selected)
            lines = prefix + selected
            if omitted:
                lines.append(f"- ...{omitted} more skills omitted")
            lines.extend(suffix)
            rendered = "\n".join(lines)
            if len(rendered.encode("utf-8")) <= SKILL_INDEX_BYTE_LIMIT:
                return rendered
            if not selected:
                return "\n".join(
                    prefix + [f"- ...{len(entries)} more skills omitted"] + suffix
                )
            selected.pop()

    def find(self, name: str) -> SkillMeta:
        for skill in self.skills:
            if skill.name == name:
                return skill
        available = ", ".join(skill.name for skill in self.skills) or "none"
        raise ValueError(f"unknown skill {name!r}; available skills: {available}")

    def load(self, name: str) -> str:
        return load_skill(self.find(name))

    def to_snapshot(self) -> list[dict[str, object]]:
        """Return the discovery result in a session-metadata-safe shape."""

        return [
            {
                "name": skill.name,
                "description": skill.description,
                "keywords": list(skill.keywords),
                "path": str(skill.path),
                "source": skill.source,
            }
            for skill in self.skills
        ]

    @classmethod
    def from_snapshot(cls, value: object) -> SkillCatalog:
        """Restore a catalog snapshot without discovering the filesystem."""

        if type(value) is not list:
            raise ValueError("skill catalog snapshot must be a list")
        skills: list[SkillMeta] = []
        for item in value:
            if type(item) is not dict:
                raise ValueError("skill catalog snapshot entries must be mappings")
            name = item.get("name")
            description = item.get("description")
            keywords = item.get("keywords")
            path = item.get("path")
            source = item.get("source", "")
            if (
                type(name) is not str
                or type(description) is not str
                or type(keywords) is not list
                or any(type(keyword) is not str for keyword in keywords)
                or type(path) is not str
                or type(source) is not str
            ):
                raise ValueError("skill catalog snapshot entry is invalid")
            skills.append(
                SkillMeta(
                    name=name,
                    description=description,
                    keywords=keywords,
                    path=Path(path),
                    source=source,
                )
            )
        return cls(tuple(skills))


def _skills_dir(home: Path) -> Path:
    return home / "skills"


def _document_path(path: Path) -> Path:
    return path / "SKILL.md" if path.is_dir() else path


def _candidate_paths(skills_dir: Path) -> list[Path]:
    flat = list(skills_dir.glob("*.md"))
    directories = [path for path in skills_dir.iterdir() if path.is_dir()]
    return sorted(flat + directories, key=lambda path: path.name)


def _contained_path(path: Path, skills_root: Path) -> Path:
    try:
        resolved = path.resolve()
        resolved.relative_to(skills_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"skill document {path} resolves outside skills root {skills_root}"
        ) from exc
    return resolved


def discover_skills(
    home: Path,
    *,
    source: str = "",
    notices: list[str] | None = None,
    skills_dir: Path | None = None,
) -> list[SkillMeta]:
    """Discover one tier of flat and Claude-Code-style skills."""

    skills_dir = skills_dir or _skills_dir(home)
    if not skills_dir.is_dir():
        return []
    try:
        skills_root = skills_dir.resolve()
    except (OSError, RuntimeError) as exc:
        notice = _warn_skill(skills_dir, exc)
        if notices is not None:
            notices.append(notice)
        return []
    discovered: list[SkillMeta] = []
    paths_by_name: dict[str, Path] = {}
    try:
        paths = _candidate_paths(skills_dir)
    except (OSError, UnicodeError) as exc:
        notice = _warn_skill(skills_dir, exc)
        if notices is not None:
            notices.append(notice)
        return []
    for path in paths:
        document = _document_path(path)
        try:
            resolved_document = _contained_path(document, skills_root)
            if path.is_dir() and not resolved_document.is_file():
                continue
            metadata, _ = _read_skill(resolved_document)
        except (
            OSError,
            UnicodeError,
            ValueError,
            RecursionError,
            yaml.YAMLError,
        ) as exc:
            notice = _warn_skill(document, exc)
            if notices is not None:
                notices.append(notice)
            continue
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
                path=path.resolve(),
                source=source,
            )
        )
    return discovered


def discover_session_skills(
    *, home: str | Path | None = None, project_dir: str | Path | None = None
) -> SkillCatalog:
    """Discover packaged, home, and project skills for one session."""

    notices: list[str] = []
    tiers: list[list[SkillMeta]] = [
        discover_skills(
            _PACKAGED_SKILLS_DIR.parent,
            source="packaged",
            notices=notices,
            skills_dir=_PACKAGED_SKILLS_DIR,
        )
    ]
    if home is not None:
        tiers.append(discover_skills(Path(home), source="home", notices=notices))
    if project_dir is not None:
        tiers.append(
            discover_skills(
                Path(project_dir) / ".zeta", source="project", notices=notices
            )
        )

    selected: dict[str, SkillMeta] = {}
    for tier in tiers:
        for skill in tier:
            selected[skill.name] = skill
    skills = tuple(
        skill for tier in tiers for skill in tier if selected[skill.name] is skill
    )
    return SkillCatalog(skills, tuple(notices))


def discover_packaged_skills() -> SkillCatalog:
    """Discover the skills shipped in the installed zeta package."""

    return SkillCatalog(
        tuple(
            discover_skills(
                _PACKAGED_SKILLS_DIR.parent,
                source="packaged",
                skills_dir=_PACKAGED_SKILLS_DIR,
            )
        )
    )


def load_skill(meta: SkillMeta) -> str:
    """Load the prompt body for one discovered skill."""

    document = _document_path(meta.path)
    resolved_document = _contained_path(document, meta.path.parent.resolve())
    _, body = _read_skill(resolved_document)
    return body


def _warn_skill(path: Path, error: Exception) -> str:
    message = f"ignored skill {path}: {error}"
    _logger.warning(message)
    return message


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


def _parse_frontmatter(lines: list[str], path: Path) -> dict[str, str | list[str]]:
    try:
        values = yaml.safe_load("\n".join(lines))
    except yaml.YAMLError as exc:
        raise ValueError(f"skill {path} has invalid YAML frontmatter") from exc
    if not isinstance(values, dict):
        raise ValueError(  # noqa: TRY004 — malformed metadata is skipped
            f"skill {path} frontmatter must be a mapping"
        )
    missing = [key for key in ("name", "description") if key not in values]
    if missing:
        raise ValueError(f"skill {path} frontmatter is missing: {', '.join(missing)}")
    name = values["name"]
    if type(name) is not str or not name.strip():
        raise ValueError(f"skill {path} frontmatter name must be a nonempty string")
    description = values["description"]
    if type(description) is not str or not description.strip():
        raise ValueError(
            f"skill {path} frontmatter description must be a nonempty string"
        )
    keywords = values.get("keywords", [])
    if type(keywords) is not list or any(
        type(item) is not str or not item.strip() for item in keywords
    ):
        raise ValueError(f"skill {path} frontmatter keywords must be a list of strings")
    return {"name": name, "description": description, "keywords": keywords}


__all__ = [
    "SKILL_INDEX_BYTE_LIMIT",
    "SkillCatalog",
    "SkillMeta",
    "discover_packaged_skills",
    "discover_session_skills",
    "discover_skills",
    "load_skill",
]
