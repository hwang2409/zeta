from __future__ import annotations

import hashlib
import json
import re
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
    entries = active_entries(plan.state, kind="state")

    assert [(entry.section, entry.text) for entry in entries] == [
        (None, "Intro paragraph\ncontinues here."),
        (None, "[Active] First item\n  - nested item\n  continuation"),
        (None, "[Active] Second item"),
        (None, "[Active] Closing paragraph."),
    ]
    assert [entry.migration_order for entry in entries] == list(range(len(entries)))
    assert all(
        entry.representation == "entry"
        and not entry.automatic
        and entry.accepted_at == MIGRATED_AT
        and entry.accepted_by == "user"
        for entry in entries
    )
    assert (
        plan.rendered_mirrors["state.md"]
        == """# Current state

Intro paragraph
continues here.

[Active] First item
  - nested item
  continuation

[Active] Second item

[Active] Closing paragraph.
"""
    )


def test_migration_splits_overlong_blocks_at_paragraph_boundaries() -> None:
    paragraph = "x" * 2050
    state = f"# Current state\n\n- {paragraph}\n\n  {paragraph}\n"

    plan = _plan(_contents(**{"state.md": state}))
    entries = [
        entry
        for entry in plan.state.entries.values()
        if isinstance(entry, MemoryEntry) and entry.kind == "state"
    ]

    assert len(entries) == 2
    assert all(len(entry.text.encode()) <= MAX_ENTRY_TEXT_BYTES for entry in entries)
    assert all(not entry.text.startswith("#") for entry in entries)


@pytest.mark.parametrize(
    ("project", "expected_counts"),
    (
        (
            "zeta",
            {"brief": 19, "state": 19, "backlog": 14, "changelog": 11, "decisions": 9},
        ),
        (
            "phoebe",
            {"brief": 26, "state": 13, "backlog": 18, "changelog": 14, "decisions": 12},
        ),
        (
            "research",
            {"brief": 2, "state": 10, "backlog": 28, "changelog": 13, "decisions": 47},
        ),
    ),
)
def test_real_memory_fixture_produces_standalone_typed_entries(
    project: str, expected_counts: dict[str, int]
) -> None:
    root = FIXTURES / project
    contents = {
        name: (root / name).read_text(encoding="utf-8") for name in LEGACY_FILES
    }
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))

    plan = _plan(contents, automatic_files=frozenset(metadata["automatic_files"]))

    counts = Counter(entry.kind for entry in plan.state.entries.values())
    assert counts == expected_counts
    restored = state_from_bytes(canonical_state_bytes(plan.state))
    for kind, count in counts.items():
        assert [
            entry.migration_order for entry in active_entries(restored, kind=kind)
        ] == list(range(count))
    texts = [entry.text for entry in plan.state.entries.values()]
    assert all(not re.fullmatch(r"#{1,6} .+", text) for text in texts)
    assert all(not text.rstrip().endswith(":") for text in texts)
    for content in contents.values():
        for section in re.findall(r"^## (.+)$", content, flags=re.MULTILINE):
            assert any(text.startswith(f"[{section}] ") for text in texts)
    if project == "zeta":
        state_entries = active_entries(restored, kind="state")
        assert all("As of 2026-10-09" in entry.text for entry in state_entries[:7])
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
