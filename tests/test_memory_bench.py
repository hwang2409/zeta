from __future__ import annotations

import json
from pathlib import Path

from evals.memory.bench import (
    RunSpec,
    _answer_metrics,
    completed_keys,
    estimate_input_tokens,
    summarize_telemetry,
)
from evals.memory.grading import MEMORY_ROOT, grade_workspace


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


def test_v1_has_two_nonleaking_chains_per_family() -> None:
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
        for wrong in task["wrong"]:
            assert wrong not in fixture_text
    assert len(tasks) == 12
    assert set(families.values()) == {2}
    assert len(families) == 6
    assert estimate_input_tokens(144) == 25_920_000
