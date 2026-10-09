from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
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
    ("lead_in", "item"),
    (
        ("x" * (MAX_ENTRY_TEXT_BYTES - 4) + ":", "€"),
        ("x" * (MAX_ENTRY_TEXT_BYTES - 4) + ":", "😀"),
        ("x" * (MAX_ENTRY_TEXT_BYTES + 1000) + ":", "item"),
    ),
)
def test_migration_splits_context_that_cannot_share_an_entry(
    lead_in: str, item: str
) -> None:
    source = f"{lead_in}\n\n- {item}"
    code = f"""
from zeta.memory_migration_plan import split_memory_document
split_memory_document({source!r}, {MAX_ENTRY_TEXT_BYTES})
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        timeout=2,
    )
    restored = state_from_bytes(
        canonical_state_bytes(_plan(_contents(**{"state.md": source})).state)
    )
    facts = [entry.text for entry in active_entries(restored, kind="state")]

    assert "".join(facts[:-1]).encode() == lead_in.encode()
    assert facts[-1] == item
    assert all(len(fact.encode()) <= MAX_ENTRY_TEXT_BYTES for fact in facts)


@pytest.mark.parametrize(
    ("fixture", "expected_counts", "dated_context", "plain_paragraph"),
    (
        (
            "alpha",
            {
                "brief": 4,
                "state": 4,
                "backlog": 3,
                "changelog": 3,
                "decisions": 3,
            },
            "As of 2025-01-02:",
            "A neutral collection records observations about colored tiles.",
        ),
        (
            "bravo",
            {
                "brief": 3,
                "state": 3,
                "backlog": 3,
                "changelog": 2,
                "decisions": 2,
            },
            "As of 2025-02-10:",
            "This guide describes a small public garden with numbered plots.",
        ),
        (
            "charlie",
            {
                "brief": 4,
                "state": 2,
                "backlog": 2,
                "changelog": 2,
                "decisions": 3,
            },
            "As of 2025-03-20:",
            "The catalog contains imaginary objects with simple measurements.",
        ),
    ),
)
def test_synthetic_memory_fixture_produces_standalone_typed_entries(
    fixture: str,
    expected_counts: dict[str, int],
    dated_context: str,
    plain_paragraph: str,
) -> None:
    root = FIXTURES / fixture
    contents = {
        name: (root / name).read_text(encoding="utf-8") for name in LEGACY_FILES
    }
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    automatic_files = frozenset(metadata["automatic_files"])

    plan = _plan(contents, automatic_files=automatic_files)

    counts = Counter(entry.kind for entry in plan.state.entries.values())
    assert counts == expected_counts
    assert (
        _plan(contents, automatic_files=automatic_files).canonical_state
        == plan.canonical_state
    )
    restored = state_from_bytes(canonical_state_bytes(plan.state))
    assert canonical_state_bytes(restored) == plan.canonical_state
    for kind, count in counts.items():
        assert [
            entry.migration_order for entry in active_entries(restored, kind=kind)
        ] == list(range(count))
    texts = [entry.text for entry in plan.state.entries.values()]
    assert plain_paragraph in texts
    assert any(
        dated_context in text and not text.rstrip().endswith(":") for text in texts
    )
    if fixture == "alpha":
        assert (
            texts.count(
                "[Current items] As of 2025-01-02: The amber tile is in the first row.\n"
                "  - Its label is visible."
            )
            == 1
        )
    assert any("\n  - " in text or "\n   - " in text for text in texts)
    assert all(len(text.encode()) <= MAX_ENTRY_TEXT_BYTES for text in texts)
    if fixture == "charlie":
        long_source = contents["brief.md"].split("## Long description\n\n", 1)[1]
        assert len(long_source.encode()) > MAX_ENTRY_TEXT_BYTES
        assert sum(text.startswith("[Long description] ") for text in texts) == 2
    assert all(not re.fullmatch(r"#{1,6} .+", text) for text in texts)
    assert all(not text.rstrip().endswith(":") for text in texts)
    for content in contents.values():
        for section in re.findall(r"^## (.+)$", content, flags=re.MULTILINE):
            assert any(text.startswith(f"[{section}] ") for text in texts)
    for entry in plan.state.entries.values():
        assert isinstance(entry, MemoryEntry)
        assert entry.representation == "entry"
        assert entry.automatic is (f"{entry.kind}.md" in automatic_files)
        assert entry.accepted_at == (None if entry.automatic else MIGRATED_AT)
        assert entry.accepted_by == (None if entry.automatic else "user")
        assert entry.migration_source is not None
        assert entry.migration_source.source_digest == plan.source_digest
        assert entry.migration_source.source_version == plan.source_version
