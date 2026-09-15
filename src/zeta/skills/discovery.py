"""Shared markdown discovery primitives for skills and agents."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import yaml

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MarkdownDocument:
    """A parsed markdown definition and its pinned discovery root."""

    entry_path: Path
    path: Path
    root: Path
    metadata: dict[str, object]
    body: str


class NamedItem(Protocol):
    name: str


def is_slash_safe_name(name: str) -> bool:
    """Return whether a name can be addressed by slash input."""

    return bool(name) and not name.startswith("/") and not any(
        character.isspace() for character in name
    )


def contained_path(path: Path, root: Path, kind: str) -> Path:
    """Resolve a definition path while keeping it inside its discovery root."""

    try:
        resolved = path.resolve()
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"{kind} document {path} resolves outside {kind}s root {root}"
        ) from exc
    return resolved


def warn_discovery(path: Path, error: Exception | str, kind: str) -> str:
    """Log and return one tolerant-discovery warning."""

    message = f"ignored {kind} {path}: {error}"
    _logger.warning(message)
    return message


def parse_frontmatter(
    lines: list[str], path: Path, kind: str
) -> dict[str, object]:
    """Parse common frontmatter while ignoring unknown keys."""

    try:
        values = yaml.safe_load("\n".join(lines))
    except yaml.YAMLError as exc:
        raise ValueError(f"{kind} {path} has invalid YAML frontmatter") from exc
    if not isinstance(values, dict):
        raise ValueError(  # noqa: TRY004 — malformed metadata is skipped
            f"{kind} {path} frontmatter must be a mapping"
        )
    missing = [key for key in ("name", "description") if key not in values]
    if missing:
        raise ValueError(
            f"{kind} {path} frontmatter is missing: {', '.join(missing)}"
        )
    name = values["name"]
    if type(name) is not str or not name.strip():
        raise ValueError(f"{kind} {path} frontmatter name must be a nonempty string")
    if not is_slash_safe_name(name):
        raise ValueError(f"{kind} {path} frontmatter name must be one nonempty word")
    description = values["description"]
    if type(description) is not str or not description.strip():
        raise ValueError(
            f"{kind} {path} frontmatter description must be a nonempty string"
        )
    return values


def read_markdown(path: Path, kind: str) -> tuple[dict[str, object], str]:
    """Read common frontmatter and a non-empty markdown body."""

    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{kind} {path} is missing YAML frontmatter")
    try:
        end = next(
            index for index, line in enumerate(lines[1:], 1) if line.strip() == "---"
        )
    except StopIteration as exc:
        raise ValueError(f"{kind} {path} has unterminated YAML frontmatter") from exc
    metadata = parse_frontmatter(lines[1:end], path, kind)
    body = "\n".join(lines[end + 1 :]).strip()
    if not body:
        raise ValueError(f"{kind} {path} has an empty prompt body")
    return metadata, body


def discover_markdown[Item](
    directory: Path,
    *,
    source: str,
    kind: str,
    build: Callable[[MarkdownDocument], Item],
    notices: list[str] | None = None,
    document_name: str | None = None,
) -> list[Item]:
    """Discover one tier of flat markdown or named directory documents."""

    if not directory.is_dir():
        return []
    try:
        root = directory.resolve()
        flat = list(directory.glob("*.md"))
        directories = (
            [path for path in directory.iterdir() if path.is_dir()]
            if document_name is not None
            else []
        )
        paths = sorted(flat + directories, key=lambda path: path.name)
    except (OSError, RuntimeError, UnicodeError) as exc:
        notice = warn_discovery(directory, exc, kind)
        if notices is not None:
            notices.append(notice)
        return []

    discovered: list[Item] = []
    paths_by_name: dict[str, Path] = {}
    for entry_path in paths:
        document_path = (
            entry_path / document_name
            if document_name is not None and entry_path.is_dir()
            else entry_path
        )
        try:
            resolved_path = contained_path(document_path, root, kind)
            if entry_path.is_dir() and not resolved_path.is_file():
                continue
            metadata, body = read_markdown(resolved_path, kind)
            document = MarkdownDocument(
                entry_path=entry_path,
                path=resolved_path,
                root=root,
                metadata=metadata,
                body=body,
            )
            item = build(document)
        except (
            OSError,
            UnicodeError,
            ValueError,
            RecursionError,
            yaml.YAMLError,
        ) as exc:
            notice = warn_discovery(document_path, exc, kind)
            if notices is not None:
                notices.append(notice)
            continue
        name = metadata["name"]
        assert isinstance(name, str)
        previous_path = paths_by_name.get(name)
        if previous_path is not None:
            raise ValueError(
                f"duplicate {kind} name {name!r} in {previous_path} and {entry_path}"
            )
        paths_by_name[name] = entry_path
        discovered.append(item)
    return discovered


def discover_session_items[Item: NamedItem](
    *,
    home: str | Path | None,
    project_dir: str | Path | None,
    packaged: Callable[[list[str]], list[Item]],
    discover: Callable[[Path, str, list[str]], list[Item]],
    warn_override: Callable[[Item], str] | None = None,
) -> tuple[tuple[Item, ...], tuple[str, ...]]:
    """Discover packaged, home, and project items with last-tier precedence."""

    notices: list[str] = []
    tiers: list[list[Item]] = [packaged(notices)]
    if home is not None:
        tiers.append(discover(Path(home), "home", notices))
    if project_dir is not None:
        tiers.append(discover(Path(project_dir) / ".zeta", "project", notices))

    if warn_override is not None:
        packaged_names = {item.name for item in tiers[0]}
        for tier in tiers[1:]:
            for item in tier:
                if item.name in packaged_names:
                    notices.append(warn_override(item))

    selected: dict[str, Item] = {}
    for tier in tiers:
        for item in tier:
            selected[item.name] = item
    items = tuple(
        item for tier in tiers for item in tier if selected[item.name] is item
    )
    return items, tuple(notices)


def snapshot_metadata(
    *,
    name: str,
    description: str,
    source: str,
    path: Path | None,
    root_name: str,
    root: Path | None,
) -> dict[str, object]:
    """Build the shared metadata-only snapshot shape."""

    return {
        "name": name,
        "description": description,
        "source": source,
        "path": str(path) if path is not None else None,
        root_name: str(root) if root is not None else None,
    }


__all__ = [
    "MarkdownDocument",
    "contained_path",
    "discover_markdown",
    "discover_session_items",
    "is_slash_safe_name",
    "parse_frontmatter",
    "read_markdown",
    "snapshot_metadata",
    "warn_discovery",
]
