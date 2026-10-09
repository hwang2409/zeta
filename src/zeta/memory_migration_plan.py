"""Dependency-free canonical plan for format-1 memory migration."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from typing import NamedTuple

LEGACY_MEMORY_FILES = (
    "brief.md",
    "state.md",
    "backlog.md",
    "changelog.md",
    "decisions.md",
)
_SECTION = re.compile(r"^##[ \t]+(.+?)\s*$")
_TITLE = re.compile(r"^#[ \t]+(.+?)\s*$")
_LIST_ITEM = re.compile(r"^(?:[-+*]|\d+[.)])[ \t]+(.*)$")
_ZETA_SCHEMA = {
    "version": 3,
    "profile": "zeta",
    "kinds": [
        {
            "key": "brief",
            "name": "Brief",
            "description": "Stable project purpose, architecture, and invariants.",
            "prompt_mode": "always",
            "prompt_priority": 100,
            "prompt_max_entries": 100,
            "default_expiry_days": None,
        },
        {
            "key": "decisions",
            "name": "Decisions",
            "description": "Binding user decisions, validated procedures, and failure lessons.",
            "prompt_mode": "always",
            "prompt_priority": 90,
            "prompt_max_entries": 100,
            "default_expiry_days": None,
        },
        {
            "key": "state",
            "name": "Current state",
            "description": "Current project and active work state, not progress narration. Supersede or resolve progress reports when completion evidence arrives.",
            "prompt_mode": "recent",
            "prompt_priority": 80,
            "prompt_max_entries": 100,
            "default_expiry_days": 30,
        },
        {
            "key": "backlog",
            "name": "Backlog",
            "description": "Open follow-up work that remains actionable. Resolve completed items instead of retaining progress narration.",
            "prompt_mode": "recent",
            "prompt_priority": 70,
            "prompt_max_entries": 100,
            "default_expiry_days": 60,
        },
        {
            "key": "changelog",
            "name": "Changelog",
            "description": "Completed durable outcomes.",
            "prompt_mode": "recent",
            "prompt_priority": 40,
            "prompt_max_entries": 100,
            "default_expiry_days": 180,
        },
    ],
}


class CanonicalMigrationPlan(NamedTuple):
    """Complete canonical format-2 state and its human projection."""

    state: dict[str, object]
    canonical_state: bytes
    entries: dict[str, dict[str, object]]
    rendered_mirrors: dict[str, str]


class VersionPruningPlan(NamedTuple):
    """Versions and blobs reachable from the bounded pointer history."""

    versions: set[str]
    blobs: set[str]


def legacy_memory_digest(contents: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for name in LEGACY_MEMORY_FILES:
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(contents[name].encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode()).hexdigest()[:32]
    return f"{prefix}_{digest}"


def _document_lines(text: str) -> list[str]:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return lines


def normalize_migration_text(text: str) -> str:
    return " ".join("\n".join(_document_lines(text)).split())


def _bounded_text(text: str, maximum: int) -> list[str]:
    pieces: list[str] = []
    remaining = text
    while len(remaining.encode()) > maximum:
        encoded = remaining.encode()
        byte_end = maximum
        while byte_end and (encoded[byte_end] & 0xC0) == 0x80:
            byte_end -= 1
        if not byte_end:
            raise ValueError("migration entry limit cannot fit the next character")
        prefix = encoded[:byte_end].decode()
        split = max(prefix.rfind("\n"), prefix.rfind(" "))
        if split <= 0:
            split = len(prefix)
        pieces.append(remaining[:split].rstrip())
        remaining = remaining[split:].lstrip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _fact_chunks(
    text: str, context: str | None, section: str | None, maximum: int
) -> list[str]:
    if context and normalize_migration_text(text).startswith(
        normalize_migration_text(context)
    ):
        context = None
    prefixes = [f"[{section}]" if section else "", context or ""]
    prefix = " ".join(item for item in prefixes if item)
    if not prefix:
        return _bounded_text(text, maximum)
    available = maximum - len((prefix + " ").encode())
    if available > 0 and all(
        len(character.encode()) <= available for character in text
    ):
        return [f"{prefix} {piece}" for piece in _bounded_text(text, available)]
    return [*_bounded_text(prefix, maximum), *_bounded_text(text, maximum)]


def split_memory_document(text: str, maximum: int) -> tuple[str | None, list[str]]:
    """Remove document structure and return independently meaningful facts."""

    lines = _document_lines(text)
    title: str | None = None
    if lines and (match := _TITLE.fullmatch(lines[0])):
        title = match.group(1)
        lines = lines[1:]
    facts: list[str] = []
    section: str | None = None
    lead_in: str | None = None
    index = 0
    while index < len(lines):
        if not lines[index]:
            index += 1
            continue
        if heading := _SECTION.fullmatch(lines[index]):
            section = heading.group(1)
            lead_in = None
            index += 1
            continue
        if item := _LIST_ITEM.fullmatch(lines[index]):
            body = item.group(1)
            index += 1
            continuation: list[str] = []
            while index < len(lines):
                if _SECTION.fullmatch(lines[index]) or _LIST_ITEM.fullmatch(
                    lines[index]
                ):
                    break
                if not lines[index]:
                    following = index + 1
                    while following < len(lines) and not lines[following]:
                        following += 1
                    if following >= len(lines) or _SECTION.fullmatch(lines[following]):
                        index = following
                        break
                    if _LIST_ITEM.fullmatch(lines[following]):
                        index = following
                        break
                    if not lines[following][:1].isspace():
                        index = following
                        break
                continuation.append(lines[index])
                index += 1
            body = "\n".join((body, *continuation)).rstrip()
            facts.extend(_fact_chunks(body, lead_in, section, maximum))
            continue
        start = index
        index += 1
        while (
            index < len(lines)
            and lines[index]
            and not _SECTION.fullmatch(lines[index])
            and not _LIST_ITEM.fullmatch(lines[index])
        ):
            index += 1
        paragraph = "\n".join(lines[start:index]).rstrip()
        following = index
        while following < len(lines) and not lines[following]:
            following += 1
        if (
            paragraph.endswith(":")
            and following < len(lines)
            and _LIST_ITEM.fullmatch(lines[following])
        ):
            lead_in = " ".join(paragraph.split())
            index = following
            continue
        lead_in = None
        facts.extend(_fact_chunks(paragraph, None, section, maximum))
    return title, facts


def migration_operation_id(project_id: str, source_digest: str) -> str:
    return _stable_id("op", project_id, source_digest, "migrate")


def build_migration_plan(
    *,
    project_id: str,
    contents: Mapping[str, str],
    source_digest: str,
    source_version: str | None,
    migrated_at: str,
    automatic_files: frozenset[str],
    max_entry_bytes: int,
) -> CanonicalMigrationPlan:
    """Return the complete canonical format-2 state for one legacy snapshot."""

    if set(contents) != set(LEGACY_MEMORY_FILES):
        raise ValueError("migration requires the complete format-1 snapshot")
    if not automatic_files <= set(LEGACY_MEMORY_FILES):
        raise ValueError("migration has invalid automatic files")
    copied = {name: contents[name] for name in LEGACY_MEMORY_FILES}
    if legacy_memory_digest(copied) != source_digest:
        raise ValueError("migration source digest does not match its snapshot")
    operation_id = migration_operation_id(project_id, source_digest)
    entries: dict[str, dict[str, object]] = {}
    rendered: dict[str, str] = {}
    for name in LEGACY_MEMORY_FILES:
        kind = name.removesuffix(".md")
        title, facts = split_memory_document(copied[name], max_entry_bytes)
        for position, text in enumerate(facts):
            entry_id = _stable_id(
                "m", project_id, source_digest, kind, str(position), text
            )
            entries[entry_id] = {
                "id": entry_id,
                "project_id": project_id,
                "kind": kind,
                "text": text,
                "representation": "entry",
                "status": "active",
                "created_at": migrated_at,
                "updated_at": migrated_at,
                "seen_at": migrated_at,
                "expires_at": None,
                "valid_from": migrated_at,
                "valid_until": None,
                "supersedes": [],
                "superseded_by": [],
                "sources": [],
                "automatic": name in automatic_files,
                "accepted_at": None if name in automatic_files else migrated_at,
                "accepted_by": None if name in automatic_files else "user",
                "last_operation_id": operation_id,
                "section": None,
                "migration_order": position,
                "migration_source": {
                    "source_digest": source_digest,
                    "source_version": source_version,
                },
            }
        blocks = ([f"# {title}"] if title else []) + facts
        rendered[name] = "\n\n".join(blocks) + ("\n" if blocks else "")
    state: dict[str, object] = {
        "format": 2,
        "project_id": project_id,
        "generation": 1,
        "schema": _ZETA_SCHEMA,
        "entries": {entry_id: entries[entry_id] for entry_id in sorted(entries)},
        "compacted_through_version": None,
    }
    canonical = json.dumps(
        state, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return CanonicalMigrationPlan(state, canonical, entries, rendered)


def reachable_version_pruning_plan(
    retained: set[str], read_manifest: Callable[[str], Mapping[str, object]]
) -> VersionPruningPlan:
    """Follow rollback and unfinalized migration references from retained history."""

    manifests = {version: read_manifest(version) for version in retained}
    migration_finalized = any(
        manifest.get("kind") == "migration-finalize" for manifest in manifests.values()
    )
    reachable = set(retained)
    blobs: set[str] = set()
    pending = list(retained)
    while pending:
        version = pending.pop()
        manifest = manifests.get(version)
        if manifest is None:
            manifest = read_manifest(version)
            manifests[version] = manifest
        targets = [manifest.get("target_version")]
        if not migration_finalized:
            targets.append(manifest.get("migration_version"))
        for target in targets:
            if isinstance(target, str) and target not in reachable:
                reachable.add(target)
                pending.append(target)
        for key in ("snapshot", "before_snapshot"):
            value = manifest.get(key)
            if isinstance(value, dict):
                blobs.update(item for item in value.values() if isinstance(item, str))
            elif isinstance(value, str):
                blobs.add(value)
    return VersionPruningPlan(reachable, blobs)
