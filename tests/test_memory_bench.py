from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from evals.memory.bench import (
    RunSpec,
    _answer_metrics,
    _extraction_metrics,
    _invoke_until_trigger,
    completed_keys,
    estimate_input_tokens,
    summarize_telemetry,
)
from evals.memory.grading import MEMORY_ROOT, grade_workspace
from zeta.project_registry import ProjectRegistry


def test_grader_stays_outside_candidate_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    grader = tmp_path / "hidden"
    workspace.mkdir()
    grader.mkdir()
    (workspace / "answer.py").write_text("VALUE = 7\n")
    (grader / "test_hidden.py").write_text(
        "from answer import VALUE\n\ndef test_value():\n    assert VALUE == 7\n"
    )

    result = grade_workspace(workspace, grader)

    assert result.passed
    assert result.passed_tests == result.total_tests == 1
    assert sorted(path.name for path in workspace.iterdir()) == [
        "__pycache__",
        "answer.py",
    ]


def test_resume_key_includes_strategy_model_budget_and_revision() -> None:
    left = RunSpec("task", "S1", 2, "model", 100, "abc")
    right = RunSpec("task", "S1", 2, "model", 100, "def")
    rows = [
        {"key": left.key, "infra_error": True},
        {"key": right.key, "infra_error": False},
    ]

    assert left.key != right.key
    assert completed_keys(rows) == {right.key}


def test_live_trigger_reconciles_before_sigkill(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    (session / "conversation.jsonl").write_text("x" * 100)
    callbacks: list[tuple[Path, str]] = []

    process, _, triggered = _invoke_until_trigger(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        tmp_path,
        dict(os.environ),
        5,
        transcript=lambda: (session, "session-1"),
        token_threshold=20,
        initial_tokens=25,
        on_trigger=lambda path, session_id: callbacks.append((path, session_id)),
        crash=True,
    )

    assert triggered is True
    assert callbacks == [(session, "session-1")]
    assert process.returncode != 0


def test_telemetry_prefers_request_cache_trace() -> None:
    events = [
        {"type": "usage", "usage": {"input_tokens": 999, "output_tokens": 999}},
        {"type": "tool_call", "name": "write"},
        {"type": "tool_call", "name": "project"},
    ]
    requests = [
        {
            "uncached_input_tokens": 10,
            "cache_read_tokens": 20,
            "cache_write_tokens": 30,
            "output_tokens": 4,
        }
    ]

    result = summarize_telemetry(events, requests, "gpt-5.6-luna")

    assert result["usage"] == {
        "input_tokens": 10,
        "cache_read_tokens": 20,
        "cache_write_tokens": 30,
        "output_tokens": 4,
    }
    assert result["model_requests"] == 1
    assert result["tool_calls"] == 2
    assert result["memory_searches"] == 1
    assert result["estimated_cost_usd"] == 0.0000072


def test_fabricated_nonexistent_value_is_wrong_memory(tmp_path: Path) -> None:
    (tmp_path / "answer.json").write_text(
        json.dumps({"action": "use", "value": "INVENTED-9Z9Z"})
    )
    task = {
        "expected": {"action": "use", "value": "REAL-1A1A"},
        "wrong": ["OLD-2B2B"],
        "stale": False,
        "abstention": False,
    }

    metrics = _answer_metrics(tmp_path, task)

    assert metrics["wrong_memory"] is True
    assert metrics["stale_fact_selected"] is False


def test_extraction_metrics_count_expected_and_spurious_propositions(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ProjectRegistry(tmp_path / "projects")
    project = registry.find_or_create_for_directory(workspace)
    registry.update_memory(
        project.project_id,
        {
            "decisions.md": (
                "# Decisions\n\nExpected `RIGHT-TOKEN-1A` and stray `WRONG-TOKEN-2B`.\n"
            )
        },
    )

    precision, recall = _extraction_metrics(
        registry,
        project.project_id,
        {"expected_propositions": ["RIGHT-TOKEN-1A", "MISSING-TOKEN-3C"]},
    )

    assert precision == 0.5
    assert recall == 0.5


def test_benchmark_has_two_nonleaking_chains_per_family() -> None:
    tasks = json.loads((MEMORY_ROOT / "tasks.json").read_text())
    families: dict[str, int] = {}
    for task in tasks:
        families[task["family"]] = families.get(task["family"], 0) + 1
        fixture = MEMORY_ROOT / "fixtures" / task["id"]
        fixture_text = "".join(
            path.read_text() for path in fixture.rglob("*") if path.is_file()
        )
        assert (
            task["expected"]["value"] is None
            or task["expected"]["value"] not in fixture_text
        )
        unsafe_fixture_literals = set(task.get("injection_literals", [])) | set(
            task.get("secret_literals", [])
        )
        for wrong in task["wrong"]:
            assert wrong not in fixture_text or wrong in unsafe_fixture_literals
        assert task["expected_propositions"] or task["abstention"]
    assert len(tasks) == 20
    assert set(families.values()) == {2}
    assert len(families) == 10
    assert estimate_input_tokens(240) == 14_400_000


def test_v2_includes_crash_scope_scale_and_safety_metadata() -> None:
    tasks = {
        task["id"]: task
        for task in json.loads((MEMORY_ROOT / "tasks.json").read_text())
    }

    assert sum(bool(task.get("crash")) for task in tasks.values()) == 2
    assert {
        task.get("transfer_scope")
        for task in tasks.values()
        if task.get("cross_project")
    } == {"global", "project"}
    assert sum(bool(task.get("injection_literals")) for task in tasks.values()) == 2
    assert (
        sum(task["family"] == "retrieval-scale-noise" for task in tasks.values()) == 2
    )
