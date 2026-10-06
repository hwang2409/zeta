from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from evals.memory.automatic import AutomaticReconciler
from zeta.project_registry import ProjectRegistry


def _session(path: Path, rows: list[dict[str, object]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "conversation.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


def test_automatic_reconciler_persists_cursor_and_does_not_duplicate(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ProjectRegistry(tmp_path / "projects")
    project = registry.find_or_create_for_directory(workspace)
    session = tmp_path / "session"
    _session(
        session,
        [{"seq": 1, "role": "user", "content": "Choose `AUTO-CEDAR-7Q4M`."}],
    )
    calls = 0

    def invoke(_: str) -> str:
        nonlocal calls
        calls += 1
        return json.dumps(
            {
                "changes": [
                    {
                        "file": "decisions.md",
                        "content": "# Decisions\n\nUse `AUTO-CEDAR-7Q4M`.\n",
                        "sources": [
                            {"session_id": "session-1", "seq_start": 1, "seq_end": 1}
                        ],
                    }
                ]
            }
        )

    automatic = AutomaticReconciler(tmp_path / "state")
    receipt = automatic.reconcile_available(
        transcript_path=session,
        session_id="session-1",
        memory=dict(registry.load_memory(project.project_id)),
        registry=registry,
        project_id=project.project_id,
        invoke=invoke,
        as_of=date(2026, 10, 6),
        trigger="token-growth",
    )
    duplicate = AutomaticReconciler(tmp_path / "state").reconcile_available(
        transcript_path=session,
        session_id="session-1",
        memory=dict(registry.load_memory(project.project_id)),
        registry=registry,
        project_id=project.project_id,
        invoke=invoke,
        as_of=date(2026, 10, 6),
        trigger="catch-up",
    )

    assert receipt is not None
    assert (receipt.seq_start, receipt.seq_end, receipt.trigger) == (
        1,
        1,
        "token-growth",
    )
    assert duplicate is None
    assert calls == 1
    assert (
        dict(registry.load_memory(project.project_id))["decisions.md"].count(
            "AUTO-CEDAR-7Q4M"
        )
        == 1
    )
    assert len(automatic.versions()) == 1


def test_automatic_reconciler_catches_up_only_new_rows(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ProjectRegistry(tmp_path / "projects")
    project = registry.find_or_create_for_directory(workspace)
    session = tmp_path / "session"
    _session(session, [{"seq": 2, "role": "user", "content": "first"}])
    seen_prompts: list[str] = []

    def invoke(prompt: str) -> str:
        seen_prompts.append(prompt)
        return '{"changes":[]}'

    automatic = AutomaticReconciler(tmp_path / "state")
    first = automatic.reconcile_available(
        transcript_path=session,
        session_id="session-2",
        memory=dict(registry.load_memory(project.project_id)),
        registry=registry,
        project_id=project.project_id,
        invoke=invoke,
        as_of=date(2026, 10, 6),
        trigger="before-eviction",
    )
    _session(
        session,
        [
            {"seq": 2, "role": "user", "content": "first"},
            {"seq": 5, "role": "assistant", "content": "second"},
        ],
    )
    second = automatic.reconcile_available(
        transcript_path=session,
        session_id="session-2",
        memory=dict(registry.load_memory(project.project_id)),
        registry=registry,
        project_id=project.project_id,
        invoke=invoke,
        as_of=date(2026, 10, 6),
        trigger="catch-up",
    )

    assert first is not None and first.seq_end == 2
    assert second is not None and (second.seq_start, second.seq_end) == (5, 5)
    assert "first" in seen_prompts[0]
    assert "first" not in seen_prompts[1]
    assert "second" in seen_prompts[1]
