from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from zeta.memory.entry_store import (
    MAX_ENTRY_TEXT_BYTES,
    MemoryEntry,
    canonical_state_bytes,
    state_from_bytes,
)
from zeta.memory.entry_views import active_entries
from zeta.memory.migration import LEGACY_FILES, migrate_format_one

PROJECT_ID = "p_" + "1" * 32
MIGRATED_AT = "2026-10-09T00:00:00Z"
FIXTURES = Path(__file__).parent / "fixtures" / "memory-migration"


def _digest(contents: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name in LEGACY_FILES:
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(contents[name].encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _plan(contents: dict[str, str], *, automatic_files: frozenset[str] = frozenset()):
    return migrate_format_one(
        project_id=PROJECT_ID,
        contents=contents,
        source_digest=_digest(contents),
        source_version="1" * 32,
        migrated_at=MIGRATED_AT,
        automatic_files=automatic_files,
    )


def _contents(**overrides: str) -> dict[str, str]:
    return {name: overrides.get(name, "") for name in LEGACY_FILES}


def _normalized(text: str) -> str:
    return " ".join(text.split())


def test_migration_splits_lists_paragraphs_and_sections() -> None:
    state = """# Current state

Intro paragraph
continues here.

## Active

- First item
  - nested item
  continuation

- Second item

Closing paragraph.
"""
    plan = _plan(_contents(**{"state.md": state}))
    entries = [
        entry
        for entry in plan.state.entries.values()
        if isinstance(entry, MemoryEntry) and entry.kind == "state"
    ]

    assert [(entry.section, entry.text) for entry in entries] == [
        (None, "# Current state"),
        (None, "Intro paragraph\ncontinues here."),
        ("Active", "- First item\n  - nested item\n  continuation"),
        ("Active", "- Second item"),
        ("Active", "Closing paragraph."),
    ]
    assert [entry.migration_order for entry in entries] == list(range(len(entries)))
    assert all(
        entry.representation == "entry"
        and not entry.automatic
        and entry.accepted_at == MIGRATED_AT
        and entry.accepted_by == "user"
        for entry in entries
    )
    assert _normalized(plan.rendered_mirrors["state.md"]) == _normalized(state)


def test_migration_splits_overlong_blocks_at_paragraph_boundaries() -> None:
    paragraph = "x" * 2050
    state = f"# Current state\n\n- {paragraph}\n\n  {paragraph}\n"

    plan = _plan(_contents(**{"state.md": state}))
    entries = [
        entry
        for entry in plan.state.entries.values()
        if isinstance(entry, MemoryEntry) and entry.kind == "state"
    ]

    assert len(entries) == 3
    assert all(len(entry.text.encode()) <= MAX_ENTRY_TEXT_BYTES for entry in entries)
    assert _normalized(plan.rendered_mirrors["state.md"]) == _normalized(state)


@pytest.mark.parametrize(
    ("project", "expected_counts"),
    (
        ("zeta", {"brief": 20, "state": 21, "backlog": 15, "changelog": 12, "decisions": 10}),
        ("phoebe", {"brief": 27, "state": 14, "backlog": 19, "changelog": 15, "decisions": 13}),
        ("research", {"brief": 3, "state": 11, "backlog": 29, "changelog": 14, "decisions": 48}),
    ),
)
def test_real_memory_fixture_round_trips_as_typed_entries(
    project: str, expected_counts: dict[str, int]
) -> None:
    root = FIXTURES / project
    contents = {name: (root / name).read_text(encoding="utf-8") for name in LEGACY_FILES}
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))

    plan = _plan(contents, automatic_files=frozenset(metadata["automatic_files"]))

    assert Counter(entry.kind for entry in plan.state.entries.values()) == expected_counts
    restored = state_from_bytes(canonical_state_bytes(plan.state))
    for kind, count in expected_counts.items():
        assert [
            entry.migration_order for entry in active_entries(restored, kind=kind)
        ] == list(range(count))
    assert {
        name: _normalized(content) for name, content in plan.rendered_mirrors.items()
    } == {name: _normalized(content) for name, content in contents.items()}
    assert all(
        isinstance(entry, MemoryEntry)
        and entry.representation == "entry"
        and entry.automatic
        and entry.accepted_at is None
        and entry.migration_source is not None
        and entry.migration_source.source_digest == plan.source_digest
        and entry.migration_source.source_version == plan.source_version
        for entry in plan.state.entries.values()
    )
