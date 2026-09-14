"""Discover and load markdown skills for one session."""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from pathlib import Path

from .discovery import (
    MarkdownDocument,
    contained_path,
    discover_markdown,
    is_slash_safe_name,
    read_markdown,
    snapshot_metadata,
)

SKILL_INDEX_BYTE_LIMIT = 32 * 1024
_PACKAGED_SKILLS_DIR = Path(__file__).parent


@dataclass(frozen=True, slots=True)
class SkillMeta:
    name: str
    description: str
    keywords: list[str]
    path: Path
    source: str = ""
    # Keep the discovery root independent from later filesystem changes.
    skills_root: Path | None = field(default=None, compare=False)


@dataclass(frozen=True, slots=True)
class SkillCatalog:
    skills: tuple[SkillMeta, ...]
    notices: tuple[str, ...] = ()

    @classmethod
    def empty(cls) -> SkillCatalog:
        """Return an explicit empty catalog for isolated tests."""

        return cls(())

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

        snapshots = []
        for skill in self.skills:
            snapshot = snapshot_metadata(
                name=skill.name,
                description=skill.description,
                source=skill.source,
                path=skill.path,
                root_name="skills_root",
                root=skill.skills_root or skill.path.parent,
            )
            snapshot["keywords"] = list(skill.keywords)
            snapshots.append(snapshot)
        return snapshots

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
            skills_root = item.get("skills_root")
            if skills_root is None:
                # Older snapshots stored resolved skill paths. Their parent
                # remains a stable root even if the original directory moves.
                skills_root = str(Path(path).parent) if type(path) is str else None
            if (
                type(name) is not str
                or type(description) is not str
                or type(keywords) is not list
                or any(type(keyword) is not str for keyword in keywords)
                or type(path) is not str
                or type(source) is not str
                or type(skills_root) is not str
            ):
                raise ValueError("skill catalog snapshot entry is invalid")
            skills.append(
                SkillMeta(
                    name=name,
                    description=description,
                    keywords=keywords,
                    path=Path(path),
                    source=source,
                    skills_root=Path(skills_root),
                )
            )
        return cls(tuple(skills))


def _skills_dir(home: Path) -> Path:
    return home / "skills"


def _document_path(path: Path) -> Path:
    return path / "SKILL.md" if path.is_dir() else path


def discover_skills(
    home: Path,
    *,
    source: str = "",
    notices: list[str] | None = None,
    skills_dir: Path | None = None,
) -> list[SkillMeta]:
    """Discover one tier of flat and Claude-Code-style skills."""

    skills_dir = skills_dir or _skills_dir(home)

    def build(document: MarkdownDocument) -> SkillMeta:
        keywords = document.metadata.get("keywords", [])
        if type(keywords) is not list or any(
            type(item) is not str or not item.strip() for item in keywords
        ):
            raise ValueError(
                f"skill {document.path} frontmatter keywords must be a list of strings"
            )
        name = document.metadata["name"]
        description = document.metadata["description"]
        assert isinstance(name, str)
        assert isinstance(description, str)
        return SkillMeta(
            name=name,
            description=description,
            keywords=keywords,
            path=document.entry_path.absolute(),
            source=source,
            skills_root=document.root,
        )

    return discover_markdown(
        skills_dir,
        source=source,
        kind="skill",
        build=build,
        notices=notices,
        document_name="SKILL.md",
    )


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
    skills_root = meta.skills_root or meta.path.parent
    resolved_document = contained_path(document, skills_root, "skill")
    _, body = read_markdown(resolved_document, "skill")
    return body


def load_skill_prompt(meta: SkillMeta) -> str:
    """Load a skill body and expose resources for directory skills."""

    body = load_skill(meta)
    if meta.path.is_dir():
        body += f"\n\nSkill resources directory: {meta.path}"
    return body


def replace_skill_index(prompt: str, catalog: SkillCatalog) -> str:
    """Replace only the skill index in a saved system prompt."""

    start_marker = "<zeta-skills>"
    end_marker = "</zeta-skills>"
    start = prompt.find(start_marker)
    if start < 0:
        return prompt
    end = prompt.find(end_marker, start)
    if end < 0:
        raise ValueError("saved prompt has an unterminated skill index")
    end += len(end_marker)
    return prompt[:start] + catalog.index() + prompt[end:]


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
