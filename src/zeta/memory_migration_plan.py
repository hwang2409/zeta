"""Dependency-free deterministic structure for format-1 memory migration."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping

LEGACY_MEMORY_FILES = (
    "brief.md",
    "state.md",
    "backlog.md",
    "changelog.md",
    "decisions.md",
)
_SECTION = re.compile(r"^##[ \t]+(.+?)\s*$")
_LIST_ITEM = re.compile(r"^(?:[-+*][ \t]+|\d+[.)][ \t]+)")


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


def _document_text(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def normalize_migration_text(text: str) -> str:
    return " ".join(_document_text(text).split())


def _bounded_paragraph(paragraph: str, maximum: int) -> list[str]:
    pieces: list[str] = []
    remaining = paragraph
    while len(remaining.encode()) > maximum:
        encoded = remaining.encode()
        byte_end = maximum
        while byte_end and (encoded[byte_end] & 0xC0) == 0x80:
            byte_end -= 1
        prefix = encoded[:byte_end].decode()
        split = max(prefix.rfind("\n"), prefix.rfind(" "))
        if split <= 0:
            split = len(prefix)
        pieces.append(remaining[:split].rstrip())
        remaining = remaining[split:].lstrip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _paragraph_chunks(text: str, maximum: int) -> list[str]:
    paragraphs = [
        piece
        for paragraph in text.split("\n\n")
        for piece in _bounded_paragraph(paragraph, maximum)
    ]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate.encode()) <= maximum:
            current = candidate
        else:
            chunks.append(current)
            current = paragraph
    if current:
        chunks.append(current)
    return chunks


def split_memory_document(text: str, maximum: int) -> list[tuple[str | None, str]]:
    normalized = _document_text(text)
    if not normalized:
        return []
    lines = normalized.splitlines()
    blocks: list[tuple[str | None, str]] = []
    section: str | None = None
    index = 0
    while index < len(lines):
        if not lines[index]:
            index += 1
            continue
        heading = _SECTION.fullmatch(lines[index])
        if heading:
            section = heading.group(1)
            index += 1
            continue
        start = index
        if _LIST_ITEM.match(lines[index]):
            index += 1
            while index < len(lines):
                if _SECTION.fullmatch(lines[index]) or _LIST_ITEM.match(lines[index]):
                    break
                if not lines[index]:
                    following = index + 1
                    while following < len(lines) and not lines[following]:
                        following += 1
                    if (
                        following >= len(lines)
                        or _SECTION.fullmatch(lines[following])
                        or _LIST_ITEM.match(lines[following])
                        or not lines[following][:1].isspace()
                    ):
                        break
                index += 1
        else:
            index += 1
            while (
                index < len(lines)
                and lines[index]
                and not _SECTION.fullmatch(lines[index])
                and not _LIST_ITEM.match(lines[index])
            ):
                index += 1
        block = "\n".join(lines[start:index]).rstrip()
        blocks.extend((section, chunk) for chunk in _paragraph_chunks(block, maximum))
    return blocks


def migration_operation_id(project_id: str, source_digest: str) -> str:
    return _stable_id("op", project_id, source_digest, "migrate")


def build_migration_entries(
    *,
    project_id: str,
    contents: Mapping[str, str],
    source_digest: str,
    source_version: str | None,
    migrated_at: str,
    automatic_files: frozenset[str],
    max_entry_bytes: int,
) -> dict[str, dict[str, object]]:
    operation_id = migration_operation_id(project_id, source_digest)
    entries: dict[str, dict[str, object]] = {}
    for name in LEGACY_MEMORY_FILES:
        kind = name.removesuffix(".md")
        automatic = name in automatic_files
        for position, (section, text) in enumerate(
            split_memory_document(contents[name], max_entry_bytes)
        ):
            entry_id = _stable_id(
                "m", project_id, source_digest, kind, str(position), section or "", text
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
                "automatic": automatic,
                "accepted_at": None if automatic else migrated_at,
                "accepted_by": None if automatic else "user",
                "last_operation_id": operation_id,
                "section": section,
                "migration_order": position,
                "migration_source": {
                    "source_digest": source_digest,
                    "source_version": source_version,
                },
            }
    return entries
