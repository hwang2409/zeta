import subprocess
from pathlib import Path

import pytest

from evals import run as eval_run
from evals.run import _check, _file


def test_eval_paths_stay_in_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    (workspace / "leak").symlink_to(outside)

    with pytest.raises(ValueError):
        _file(workspace, "../outside.txt")
    with pytest.raises(ValueError):
        _file(workspace, "leak")


def test_eval_grades_artifacts_not_model_claims(tmp_path: Path) -> None:
    (tmp_path / "result.txt").write_text("correct\n")
    assert _check(tmp_path, {}, {"path": "result.txt", "equals": "correct\n"}) is None
    assert _check(tmp_path, {}, {"path": "result.txt", "equals": "wrong\n"})
    assert _check(tmp_path, {}, {"path": "missing.txt"}) == "missing file: missing.txt"
    assert _check(tmp_path, {}, {"path": "result.txt", "nonempty_lines": ["correct"]}) is None


def test_eval_replays_pinned_zeta_checkout(tmp_path: Path) -> None:
    ref = subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"], text=True
    ).strip()
    result = eval_run.run_task(
        {
            "id": "pinned",
            "git_ref": ref,
            "prompt": "hello",
            "checks": [{"command": ["git", "rev-parse", "HEAD"], "stdout": ref + "\n"}],
        },
        provider="fake", model="fake", timeout=20, keep_workspaces=tmp_path,
    )
    assert result["passed"] is True
    saved = Path(result["saved_workspace"])
    assert saved.is_dir()
    assert subprocess.check_output(["git", "-C", str(saved), "rev-parse", "HEAD"], text=True).strip() == ref

    with pytest.raises(ValueError, match="full lowercase commit SHA"):
        eval_run.run_task(
            {"id": "invalid", "git_ref": "HEAD", "prompt": "hello", "checks": []},
            provider="fake", model="fake", timeout=20,
        )


def test_eval_command_imports_workspace_source(tmp_path: Path) -> None:
    package = tmp_path / "src" / "zeta"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("marker = 'workspace'\n")
    assert _check(
        tmp_path, {},
        {"command": ["python", "-c", "import zeta; assert zeta.marker == 'workspace'"]},
    ) is None


def test_eval_rejects_non_json_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    class Process:
        returncode = 0

        def communicate(self, *, timeout: int) -> tuple[str, str]:
            return '{"type":"message"}\n\x1b[31mbackground done\x1b[0m\n', ""

    monkeypatch.setattr(eval_run.subprocess, "Popen", lambda *args, **kwargs: Process())
    result = eval_run.run_task(
        {"id": "jsonl", "prompt": "check", "checks": []},
        provider="codex", model="gpt-5.6-luna", timeout=1,
    )
    assert result["artifact_passed"] is True
    assert result["completed"] is False
    assert result["run_error"] == "agent emitted invalid JSONL line 2"


@pytest.mark.parametrize(
    ("line", "error"),
    [
        ('{"type":"tool_call"}', "malformed tool_call"),
        ('{"type":"tool_call","name":1}', "malformed tool_call"),
        ('{"type":"tool_call","name":""}', "malformed tool_call"),
        ('{"type":"usage"}', "malformed usage"),
        ('{"type":"usage","usage":[]}', "malformed usage"),
    ],
)
def test_eval_rejects_malformed_event_fields(
    monkeypatch: pytest.MonkeyPatch, line: str, error: str
) -> None:
    class Process:
        returncode = 0

        def communicate(self, *, timeout: int) -> tuple[str, str]:
            return f'{line}\n{{"type":"message","text":"done"}}\n', ""

    monkeypatch.setattr(eval_run.subprocess, "Popen", lambda *args, **kwargs: Process())
    result = eval_run.run_task(
        {"id": "jsonl", "prompt": "check", "checks": []},
        provider="codex", model="gpt-5.6-luna", timeout=1,
    )
    assert result["artifact_passed"] is True
    assert result["completed"] is False
    assert result["passed"] is False
    assert result["run_error"] == f"agent emitted {error} JSONL line 1"
