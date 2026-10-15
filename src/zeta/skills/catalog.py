"""Discover and load markdown skills for one session."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from html import escape
from pathlib import Path

from .discovery import (
    MarkdownDocument,
    contained_path,
    discover_markdown,
    discover_session_items,
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


type SkillRootsSignature = tuple[
    tuple[str, tuple[int, int, int, int] | None], ...
]


@dataclass(frozen=True, slots=True)
class SkillRefresh:
    """One changed skill snapshot and its user-visible delta."""

    catalog: SkillCatalog
    added: tuple[str, ...]
    removed: tuple[str, ...]
    updated: tuple[str, ...]
    notices: tuple[str, ...]


class SkillCatalogWatcher:
    """Rediscover session skills only when their filesystem signature changes."""

    def __init__(
        self,
        catalog: SkillCatalog,
        *,
        home: str | Path,
        project_dir: str | Path,
    ) -> None:
        self.catalog = catalog
        self.home = Path(home)
        self.project_dir = Path(project_dir)
        self.signature = skill_roots_signature(
            home=self.home, project_dir=self.project_dir
        )

    def refresh(self, *, force: bool = False) -> SkillRefresh | None:
        signature = skill_roots_signature(
            home=self.home, project_dir=self.project_dir
        )
        if not force and signature == self.signature:
            return None
        discovered = discover_session_skills(
            home=self.home, project_dir=self.project_dir
        )
        current_by_path = {skill.path: skill for skill in self.catalog.skills}
        discovered_paths = {skill.path for skill in discovered.skills}
        retained = tuple(
            skill
            for path, skill in current_by_path.items()
            if path not in discovered_paths and _document_path(path).is_file()
        )
        catalog = SkillCatalog((*discovered.skills, *retained), discovered.notices)
        old_by_name = {skill.name: skill for skill in self.catalog.skills}
        new_by_name = {skill.name: skill for skill in catalog.skills}
        old_names = set(old_by_name)
        new_names = set(new_by_name)
        notices = tuple(
            notice for notice in catalog.notices if notice not in self.catalog.notices
        )
        self.signature = signature
        changed = catalog.skills != self.catalog.skills or bool(notices)
        self.catalog = catalog
        if not changed:
            return None
        return SkillRefresh(
            catalog,
            tuple(sorted(new_names - old_names)),
            tuple(sorted(old_names - new_names)),
            tuple(
                sorted(
                    name
                    for name in old_names & new_names
                    if old_by_name[name] != new_by_name[name]
                )
            ),
            notices,
        )


def _skills_dir(home: Path) -> Path:
    return home / "skills"


def skill_roots_signature(
    *, home: str | Path, project_dir: str | Path
) -> SkillRootsSignature:
    """Return directory entries and document stats for every discovery tier."""

    roots = (
        _PACKAGED_SKILLS_DIR,
        _skills_dir(Path(home)),
        Path(project_dir) / ".zeta" / "skills",
    )
    signature: list[tuple[str, tuple[int, int, int, int] | None]] = []
    for root in roots:
        try:
            entries = sorted(root.iterdir(), key=lambda path: path.name)
            root_stat = root.stat()
            signature.append(
                (
                    str(root),
                    (
                        root_stat.st_mtime_ns,
                        root_stat.st_size,
                        root_stat.st_ino,
                        root_stat.st_mode,
                    ),
                )
            )
        except OSError:
            signature.append((str(root), None))
            continue
        for entry in entries:
            document = entry / "SKILL.md" if entry.is_dir() else entry
            if not (entry.is_dir() or entry.suffix.casefold() == ".md"):
                continue
            try:
                stat = document.stat()
                value = (stat.st_mtime_ns, stat.st_size, stat.st_ino, stat.st_mode)
            except OSError:
                value = None
            signature.append((str(document), value))
    return tuple(signature)


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

    skills, notices = discover_session_items(
        home=home,
        project_dir=project_dir,
        packaged=lambda notices: discover_skills(
            _PACKAGED_SKILLS_DIR.parent,
            source="packaged",
            notices=notices,
            skills_dir=_PACKAGED_SKILLS_DIR,
        ),
        discover=lambda root, source, notices: discover_skills(
            root, source=source, notices=notices
        ),
    )
    return SkillCatalog(skills, notices)


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

    return replace_skill_index_metadata(prompt, catalog, {})[0]


def replace_skill_index_metadata(
    prompt: str,
    catalog: SkillCatalog,
    components: dict[str, dict[str, int | str]],
) -> tuple[str, dict[str, dict[str, int | str]]]:
    """Replace the skill index and move or resize owned prompt components."""

    start_marker = "<zeta-skills>"
    end_marker = "</zeta-skills>"
    start = prompt.find(start_marker)
    if start < 0:
        return prompt, {key: dict(value) for key, value in components.items()}
    end = prompt.find(end_marker, start)
    if end < 0:
        raise ValueError("saved prompt has an unterminated skill index")
    end += len(end_marker)
    replacement = catalog.index()
    updated = prompt[:start] + replacement + prompt[end:]
    delta = len(replacement) - (end - start)
    updated_components: dict[str, dict[str, int | str]] = {}
    for key, component in components.items():
        offset = int(component["offset"])
        length = int(component["length"])
        component_end = offset + length
        if offset >= end:
            offset += delta
        elif offset <= start and component_end >= end:
            length += delta
        elif component_end > start:
            raise ValueError("skill index partially overlaps a prompt component")
        digest = hashlib.sha256(updated[offset : offset + length].encode()).hexdigest()
        updated_components[key] = {
            "offset": offset,
            "length": length,
            "digest": digest,
        }
    return updated, updated_components


__all__ = [
    "SKILL_INDEX_BYTE_LIMIT",
    "SkillCatalog",
    "SkillCatalogWatcher",
    "SkillMeta",
    "SkillRefresh",
    "discover_packaged_skills",
    "discover_session_skills",
    "discover_skills",
    "is_slash_safe_name",
    "load_skill",
    "load_skill_prompt",
    "replace_skill_index",
    "replace_skill_index_metadata",
]
