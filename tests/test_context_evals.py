import argparse
import json
import shutil
from pathlib import Path

import pytest

from evals.context import bench, grading, report
from evals.context.grading import CONTEXT_ROOT, apply_overlay, grade_workspace


@pytest.mark.parametrize("task", json.loads((CONTEXT_ROOT / "tasks.json").read_text()))
def test_context_fixture_fails_and_reference_passes(
    tmp_path: Path, task: dict[str, object]
) -> None:
    workspace = tmp_path / "workspace"
    shutil.copytree(CONTEXT_ROOT / str(task["fixture"]), workspace)
    grader = CONTEXT_ROOT / str(task["grader"])
    expected = int(task["expected_passes"])
    assert grade_workspace(workspace, grader, expected)[0] is False
    apply_overlay(workspace, CONTEXT_ROOT / str(task["reference"]))
    assert grade_workspace(workspace, grader, expected) == (True, None)


def test_context_session_graders_fail_fixture_and_pass_reference(
    tmp_path: Path,
) -> None:
    untouched = tmp_path / "untouched"
    shutil.copytree(CONTEXT_ROOT / "session/fixture", untouched)
    turns = json.loads((CONTEXT_ROOT / "session/turns.json").read_text())
    for index, turn in enumerate(turns, 1):
        grader = CONTEXT_ROOT / f"session/graders/turn{index}"
        assert grade_workspace(untouched, grader, turn["expected_passes"])[0] is False

    solved = tmp_path / "solved"
    shutil.copytree(CONTEXT_ROOT / "session/fixture", solved)
    for index, turn in enumerate(turns, 1):
        apply_overlay(solved, CONTEXT_ROOT / f"session/reference/turn{index}")
        grader = CONTEXT_ROOT / f"session/graders/turn{index}"
        assert grade_workspace(solved, grader, turn["expected_passes"]) == (True, None)


def test_grading_retries_only_transient_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    grader = tmp_path / "grader"
    grader.mkdir()
    (grader / "test_hidden.py").write_text("def test_ok():\n    pass\n")
    failures = iter(["command timed out: python", None])
    monkeypatch.setattr(grading, "_check", lambda *args, **kwargs: next(failures))
    assert grading.grade_workspace(tmp_path, grader, 1) == (True, None)


def test_metrics_parse_cache_telemetry_and_tools() -> None:
    events = [
        {"type": "tool_call", "name": "read"},
        {"type": "tool_call", "name": "read"},
    ]
    requests = [
        {
            "uncached_input_tokens": 100,
            "cache_read_tokens": 50,
            "cache_write_tokens": 10,
            "output_tokens": 20,
        }
    ]
    telemetry = [
        {"event": "compaction_complete", "duration_seconds": 1.25},
        {"event": "recall_call"},
    ]
    result = bench.summarize_run(events, requests, telemetry, "gpt-5.6-luna")
    assert result["usage"] == {
        "input_tokens": 100,
        "output_tokens": 20,
        "cache_read_tokens": 50,
        "cache_write_tokens": 10,
    }
    assert result["tool_calls_by_name"] == {"read": 2}
    assert result["compactions"] == 1
    assert result["compaction_seconds"] == 1.25
    assert result["recall_calls"] == 1
    assert result["estimated_cost_usd"] == pytest.approx(0.000045)

    cache_only = bench.summarize_run([], [{"compacted": True}], [], "unknown")
    assert cache_only["compactions"] == 1


def test_repo_runner_uses_fake_zeta_and_grades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    task = json.loads((CONTEXT_ROOT / "tasks.json").read_text())[0]
    seen: dict[str, object] = {}

    def fake_invoke(command, workspace, env, timeout):
        seen.update(command=command, env=env, timeout=timeout)
        apply_overlay(workspace, CONTEXT_ROOT / str(task["reference"]))
        output = (
            "\n".join(
                [
                    json.dumps({"type": "tool_call", "name": "edit"}),
                    json.dumps(
                        {
                            "type": "usage",
                            "usage": {"input_tokens": 7, "output_tokens": 3},
                        }
                    ),
                    json.dumps({"type": "message", "text": "done"}),
                ]
            )
            + "\n"
        )
        return __import__("subprocess").CompletedProcess(command, 0, output, ""), 0.5

    monkeypatch.setattr(bench, "_invoke", fake_invoke)
    monkeypatch.setattr(bench, "grade_workspace", lambda *args: (True, None))
    args = argparse.Namespace(
        zeta_checkout=checkout,
        model="gpt-5.6-luna",
        token_budget=24_000,
        max_turns=20,
        timeout=10,
    )
    result = bench.run_repo_task(
        task, bench.RunSpec(task["id"], "", 1), args, tmp_path / "run"
    )
    assert result["passed"] is True
    assert result["usage"]["input_tokens"] == 7
    assert result["tool_calls_by_name"] == {"edit": 1}
    assert seen["env"]["ZETA_CONTEXT_STRATEGY"] == ""
    assert "--project" in seen["command"]


def test_session_runner_resumes_one_persisted_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_invoke(command, workspace, env, timeout):
        calls.append((command, env))
        stderr = "resume with: zeta --resume session-123\n" if len(calls) == 1 else ""
        output = json.dumps({"type": "message", "text": "done"}) + "\n"
        return __import__("subprocess").CompletedProcess(
            command, 0, output, stderr
        ), 0.25

    monkeypatch.setattr(bench, "_invoke", fake_invoke)
    monkeypatch.setattr(bench, "grade_workspace", lambda *args: (True, None))
    args = argparse.Namespace(
        zeta_checkout=tmp_path / "checkout",
        model="gpt-5.6-luna",
        token_budget=24_000,
        max_turns=20,
        timeout=10,
    )
    turns = json.loads((CONTEXT_ROOT / "session/turns.json").read_text())
    result = bench.run_session(
        turns, bench.RunSpec("session", "recall", 1), args, tmp_path / "run"
    )

    assert result["passed"] is True
    assert result["turn_grades"] == [True, True, True, True]
    assert "--resume" not in calls[0][0]
    assert all(
        command[command.index("--resume") + 1] == "session-123"
        for command, _env in calls[1:]
    )
    assert {env["ZETA_HOME"] for _command, env in calls} == {str(tmp_path / "run/home")}


def test_report_wilson_and_tables(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    rows = [
        {
            "key": "one||1",
            "task": "one",
            "strategy": "",
            "passed": True,
            "usage": {
                "input_tokens": 100,
                "cache_read_tokens": 100,
                "output_tokens": 10,
            },
            "estimated_cost_usd": 0.01,
            "wall_seconds": 2,
            "compactions": 1,
            "recall_calls": 0,
        },
        {
            "key": "two|recall|1",
            "task": "two",
            "strategy": "recall",
            "passed": False,
            "usage": {"input_tokens": 50, "cache_read_tokens": 0, "output_tokens": 5},
            "estimated_cost_usd": 0.02,
            "wall_seconds": 4,
            "compactions": 2,
            "recall_calls": 3,
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    rendered = report.render(report.load_results(path))
    assert "Pass rate (Wilson 95% CI)" in rendered
    assert "| baseline |" in rendered
    assert "| recall |" in rendered
    assert "50.0%" in rendered
    low, high = report.wilson(1, 1)
    assert 0 < low < high == pytest.approx(1.0)
