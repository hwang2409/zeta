import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
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
    assert (
        _check(tmp_path, {}, {"path": "result.txt", "nonempty_lines": ["correct"]})
        is None
    )


def test_eval_grades_final_browser_result(tmp_path: Path) -> None:
    events = [
        {
            "type": "tool_result",
            "name": "browser",
            "is_error": False,
            "content": "Buy groceries",
        },
        {
            "type": "tool_result",
            "name": "browser",
            "is_error": False,
            "content": "Water flowers",
        },
    ]
    assert (
        _check(
            tmp_path,
            {},
            {"last_tool_result": "browser", "contains": "Water flowers"},
            events=events,
        )
        is None
    )
    assert (
        _check(
            tmp_path,
            {},
            {"last_tool_result": "browser", "not_contains": "Buy groceries"},
            events=events,
        )
        is None
    )
    assert (
        _check(
            tmp_path,
            {},
            {"last_tool_result": "browser", "contains": "Buy groceries"},
            events=events,
        )
        == "tool result missing expected text: browser"
    )
    assert (
        _check(tmp_path, {}, {"last_tool_result": "browser"}, events=[])
        == "missing tool result: browser"
    )
    events.append(
        {
            "type": "tool_result",
            "name": "browser",
            "is_error": True,
            "content": "Water flowers",
        }
    )
    assert (
        _check(tmp_path, {}, {"last_tool_result": "browser"}, events=events)
        == "invalid tool result: browser"
    )


@pytest.mark.parametrize("other_tool", [None, "bash", "agent"])
def test_browser_eval_requires_only_browser_calls(
    monkeypatch: pytest.MonkeyPatch, other_tool: str | None
) -> None:
    class Process:
        returncode = 0

        def communicate(self, *, timeout: int) -> tuple[str, str]:
            events = [
                {
                    "type": "tool_call",
                    "id": "browser-1",
                    "name": "browser",
                    "arguments": {},
                }
            ]
            if other_tool is not None:
                events.append(
                    {
                        "type": "tool_call",
                        "id": f"{other_tool}-1",
                        "name": other_tool,
                        "arguments": {},
                    }
                )
            events.extend(
                [
                    {
                        "type": "tool_result",
                        "id": "browser-1",
                        "name": "browser",
                        "is_error": False,
                        "content": "correct cart",
                    },
                    {"type": "message", "text": "done"},
                ]
            )
            return "\n".join(json.dumps(event) for event in events) + "\n", ""

    monkeypatch.setattr(eval_run.subprocess, "Popen", lambda *args, **kwargs: Process())
    result = eval_run.run_task(
        {
            "id": "browser",
            "prompt": "check",
            "checks": [
                {"allowed_tools": ["browser"]},
                {"last_tool_result": "browser", "contains": "correct cart"},
            ],
        },
        provider="codex",
        model="gpt-5.6-luna",
        timeout=1,
    )
    assert result["passed"] is (other_tool is None)
    assert result["failures"] == (
        [] if other_tool is None else [f"disallowed tool: {other_tool}"]
    )


def test_eval_rejects_browser_claim_without_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        returncode = 0

        def communicate(self, *, timeout: int) -> tuple[str, str]:
            return '{"type":"message","text":"I completed the browser task"}\n', ""

    monkeypatch.setattr(eval_run.subprocess, "Popen", lambda *args, **kwargs: Process())
    result = eval_run.run_task(
        {
            "id": "browser",
            "prompt": "check",
            "checks": [{"last_tool_result": "browser", "contains": "expected state"}],
        },
        provider="codex",
        model="gpt-5.6-luna",
        timeout=1,
    )
    assert result["completed"] is True
    assert result["artifact_passed"] is False
    assert result["failures"] == ["missing tool result: browser"]


def test_eval_serves_local_browser_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    class Process:
        returncode = 0

        def communicate(self, *, timeout: int) -> tuple[str, str]:
            with urllib.request.urlopen(prompt_url, timeout=timeout) as response:
                assert b"Copper Glow" in response.read()
            return '{"type":"message","text":"done"}\n', ""

    def start(command: list[str], **_kwargs: object) -> Process:
        nonlocal prompt_url
        prompt_url = command[-1].split("open ", 1)[1]
        return Process()

    prompt_url = ""
    monkeypatch.setattr(eval_run.subprocess, "Popen", start)
    result = eval_run.run_task(
        {
            "id": "local",
            "local_fixture": "catalog_fixture.html",
            "prompt": "open {base_url}/catalog_fixture.html",
            "checks": [],
        },
        provider="codex",
        model="gpt-5.6-luna",
        timeout=1,
    )
    assert result["passed"] is True


def test_eval_replays_pinned_zeta_checkout(tmp_path: Path) -> None:
    ref = subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    ruff = shutil.which("ruff")
    assert ruff is not None
    ruff_version = subprocess.check_output([ruff, "--version"], text=True).split()[1]
    result = eval_run.run_task(
        {
            "id": "pinned",
            "git_ref": ref,
            "toolchain": {"python": platform.python_version(), "ruff": ruff_version},
            "prompt": "hello",
            "checks": [{"command": ["git", "rev-parse", "HEAD"], "stdout": ref + "\n"}],
        },
        provider="fake",
        model="fake",
        timeout=20,
        keep_workspaces=tmp_path,
    )
    assert result["passed"] is True
    saved = Path(result["saved_workspace"])
    assert saved.is_dir()
    assert (
        subprocess.check_output(
            ["git", "-C", str(saved), "rev-parse", "HEAD"], text=True
        ).strip()
        == ref
    )

    with pytest.raises(ValueError, match="full lowercase commit SHA"):
        eval_run.run_task(
            {"id": "invalid", "git_ref": "HEAD", "prompt": "hello", "checks": []},
            provider="fake",
            model="fake",
            timeout=20,
        )


def test_eval_command_imports_workspace_source(tmp_path: Path) -> None:
    package = tmp_path / "src" / "zeta"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("marker = 'workspace'\n")
    assert (
        _check(
            tmp_path,
            {},
            {
                "command": [
                    "python",
                    "-c",
                    "import zeta; assert zeta.marker == 'workspace'",
                ]
            },
        )
        is None
    )


def test_eval_rejects_non_json_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    class Process:
        returncode = 0

        def communicate(self, *, timeout: int) -> tuple[str, str]:
            return '{"type":"message"}\n\x1b[31mbackground done\x1b[0m\n', ""

    monkeypatch.setattr(eval_run.subprocess, "Popen", lambda *args, **kwargs: Process())
    result = eval_run.run_task(
        {"id": "jsonl", "prompt": "check", "checks": []},
        provider="codex",
        model="gpt-5.6-luna",
        timeout=1,
    )
    assert result["artifact_passed"] is True
    assert result["completed"] is False
    assert result["run_error"] == "agent emitted malformed message JSONL line 1"


def test_eval_jsonl_uses_newline_not_unicode_line_separators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        returncode = 0
        output = (
            json.dumps(
                {"type": "message", "text": "first\u2028second"}, ensure_ascii=False
            )
            + "\n"
        )

        def communicate(self, *, timeout: int) -> tuple[str, str]:
            return self.output, ""

    monkeypatch.setattr(eval_run.subprocess, "Popen", lambda *args, **kwargs: Process())
    task = {"id": "jsonl", "prompt": "check", "checks": []}
    assert (
        eval_run.run_task(task, provider="codex", model="gpt-5.6-luna", timeout=1)[
            "passed"
        ]
        is True
    )

    Process.output = '{"type":"message","text":"done"}\n\n'
    result = eval_run.run_task(task, provider="codex", model="gpt-5.6-luna", timeout=1)
    assert result["passed"] is False
    assert result["run_error"] == "agent emitted invalid JSONL line 2"


def test_eval_task_jsonl_uses_newline_not_unicode_line_separators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = {"id": "unicode", "prompt": "first\u2028second", "checks": []}
    path = tmp_path / "tasks.jsonl"
    path.write_text(json.dumps(task, ensure_ascii=False) + "\n")

    def run_task(loaded: dict, **_kwargs: object) -> dict:
        assert loaded == task
        return {"passed": True, "artifact_passed": True, "completed": True}

    monkeypatch.setattr(eval_run, "run_task", run_task)
    monkeypatch.setattr(sys, "argv", ["evals/run.py", "--tasks", str(path)])
    assert eval_run.main() == 0


def test_eval_reports_root_and_child_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    class Process:
        returncode = 0

        def communicate(self, *, timeout: int) -> tuple[str, str]:
            events = [
                {
                    "type": "tool_call",
                    "id": "agent-1",
                    "name": "agent",
                    "arguments": {},
                },
                {
                    "type": "tool_call",
                    "id": "read-1",
                    "name": "read",
                    "arguments": {},
                    "agent_instance_id": "root:1",
                },
                {
                    "type": "tool_result",
                    "id": "agent-1",
                    "name": "agent",
                    "is_error": False,
                    "content": "child done",
                },
                {
                    "type": "tool_result",
                    "id": "read-1",
                    "name": "read",
                    "is_error": False,
                    "content": "file contents",
                },
                {"type": "usage", "usage": {"input_tokens": 5, "total_tokens": 5}},
                {
                    "type": "child_usage",
                    "usage": {"input_tokens": 10},
                    "by_model": {"gpt-5.6-luna": {"input_tokens": 10}},
                },
                {"type": "message", "text": "done"},
            ]
            return "\n".join(json.dumps(event) for event in events) + "\n", ""

    monkeypatch.setattr(eval_run.subprocess, "Popen", lambda *args, **kwargs: Process())
    result = eval_run.run_task(
        {"id": "team", "prompt": "check", "checks": []},
        provider="codex",
        model="gpt-5.6-luna",
        timeout=1,
    )
    assert result["passed"] is True
    assert result["usage"]["input_tokens"] == 5
    assert result["child_usage"]["input_tokens"] == 10
    assert result["child_usage_by_model"]["gpt-5.6-luna"]["input_tokens"] == 10
    assert result["total_usage"]["input_tokens"] == 15
    assert result["total_usage"]["total_tokens"] == 15
    assert result["tool_calls_by_agent"] == {"root": 1, "root:1": 1}


def test_eval_rejects_duplicate_child_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    class Process:
        returncode = 0

        def communicate(self, *, timeout: int) -> tuple[str, str]:
            events = [
                {"type": "child_usage", "usage": {}, "by_model": {}},
                {"type": "child_usage", "usage": {}, "by_model": {}},
                {"type": "message", "text": "done"},
            ]
            return "\n".join(json.dumps(event) for event in events) + "\n", ""

    monkeypatch.setattr(eval_run.subprocess, "Popen", lambda *args, **kwargs: Process())
    result = eval_run.run_task(
        {"id": "team", "prompt": "check", "checks": []},
        provider="codex",
        model="gpt-5.6-luna",
        timeout=1,
    )
    assert result["completed"] is False
    assert result["run_error"] == "agent emitted duplicate child_usage JSONL line 2"


@pytest.mark.parametrize(
    ("line", "error"),
    [
        ('{"type":"tool_call"}', "malformed tool_call"),
        ('{"type":"tool_call","name":1}', "malformed tool_call"),
        ('{"type":"tool_call","name":""}', "malformed tool_call"),
        ('{"type":"usage"}', "malformed usage"),
        ('{"type":"usage","usage":[]}', "malformed usage"),
        ('{"type":"child_usage"}', "malformed usage"),
        ('{"type":"child_usage","usage":{},"by_model":[]}', "malformed child_usage"),
        (
            '{"type":"child_usage","usage":{"input_tokens":2},"by_model":{"luna":{"input_tokens":1}}}',
            "inconsistent child_usage",
        ),
        (
            '{"type":"tool_call","name":"read","agent_instance_id":""}',
            "malformed tool_call",
        ),
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
        provider="codex",
        model="gpt-5.6-luna",
        timeout=1,
    )
    assert result["artifact_passed"] is True
    assert result["completed"] is False
    assert result["passed"] is False
    assert result["run_error"] == f"agent emitted {error} JSONL line 1"


def test_eval_rejects_orphan_tool_result(monkeypatch: pytest.MonkeyPatch) -> None:
    class Process:
        returncode = 0

        def communicate(self, *, timeout: int) -> tuple[str, str]:
            events = [
                {
                    "type": "tool_result",
                    "id": "missing",
                    "name": "browser",
                    "is_error": False,
                    "content": "done",
                },
                {"type": "message", "text": "done"},
            ]
            return "\n".join(json.dumps(event) for event in events) + "\n", ""

    monkeypatch.setattr(eval_run.subprocess, "Popen", lambda *args, **kwargs: Process())
    result = eval_run.run_task(
        {"id": "orphan", "prompt": "check", "checks": []},
        provider="codex",
        model="gpt-5.6-luna",
        timeout=1,
    )
    assert result["completed"] is False
    assert result["passed"] is False
    assert "tool_result" in result["run_error"]


@pytest.mark.parametrize(
    "output",
    [
        "{}\n",
        '{"type":"message"}\n',
    ],
)
def test_eval_rejects_empty_or_incomplete_events(
    monkeypatch: pytest.MonkeyPatch, output: str
) -> None:
    class Process:
        returncode = 0

        def communicate(self, *, timeout: int) -> tuple[str, str]:
            return output, ""

    monkeypatch.setattr(eval_run.subprocess, "Popen", lambda *args, **kwargs: Process())
    result = eval_run.run_task(
        {"id": "malformed", "prompt": "check", "checks": []},
        provider="codex",
        model="gpt-5.6-luna",
        timeout=1,
    )
    assert result["completed"] is False
    assert result["passed"] is False
    assert result["run_error"]


def test_eval_child_environment_is_allowlisted(monkeypatch: pytest.MonkeyPatch) -> None:
    poisoned_home = "/tmp/zeta-poisoned-home"
    monkeypatch.setenv("ZETA_HOME", poisoned_home)
    monkeypatch.setenv("PYTEST_ADDOPTS", "--pdb")
    observed: dict[str, str] = {}

    class Process:
        returncode = 0

        def communicate(self, *, timeout: int) -> tuple[str, str]:
            return '{"type":"message","text":"done"}\n', ""

    def start(_command: list[str], **kwargs: object) -> Process:
        observed.update(kwargs["env"])
        return Process()

    monkeypatch.setattr(eval_run.subprocess, "Popen", start)
    result = eval_run.run_task(
        {"id": "env", "prompt": "check", "checks": []},
        provider="codex",
        model="gpt-5.6-luna",
        timeout=1,
    )
    assert result["passed"] is True
    assert observed["ZETA_HOME"] != poisoned_home
    assert observed["HOME"] != os.environ.get("HOME", "")
    assert not any(name.startswith("PYTEST_") for name in observed)


def test_pinned_grader_ignores_workspace_skip_hook(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ref = subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    ruff = shutil.which("ruff")
    assert ruff is not None
    ruff_version = subprocess.check_output([ruff, "--version"], text=True).split()[1]

    class Process:
        returncode = 0

        def communicate(self, *, timeout: int) -> tuple[str, str]:
            Path(process_cwd, "conftest.py").write_text(
                "import pytest\n\ndef pytest_collection_modifyitems(items):\n"
                "    for item in items:\n        item.add_marker(pytest.mark.skip())\n"
            )
            return '{"type":"message","text":"done"}\n', ""

    process_cwd = ""
    real_popen = eval_run.subprocess.Popen

    def start(_command: list[str], **kwargs: object) -> Process:
        nonlocal process_cwd
        if "--no-session" not in _command:
            return real_popen(_command, **kwargs)
        process_cwd = str(kwargs["cwd"])
        return Process()

    monkeypatch.setattr(eval_run.subprocess, "Popen", start)
    task = {
        "id": "skip-hook",
        "git_ref": ref,
        "prompt": "check",
        "setup": {
            "tests/test_skip_hook_regression.py": """
from pathlib import Path


def test_workspace_hook_is_not_in_the_grader():
    assert Path(\"conftest.py\").exists()
"""
        },
        "checks": [
            {
                "command": [
                    "python",
                    "-m",
                    "pytest",
                    "-q",
                    "tests/test_skip_hook_regression.py",
                ],
                "expected_passes": 1,
                "expected_skips": 0,
            },
        ],
        "toolchain": {"python": platform.python_version(), "ruff": ruff_version},
    }
    result = eval_run.run_task(
        task,
        provider="codex",
        model="gpt-5.6-luna",
        timeout=1,
        keep_failures=tmp_path,
    )
    assert result["passed"] is False
    assert result["failures"]


def test_pinned_grader_tests_candidate_source_and_lints_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref = subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    ruff = shutil.which("ruff")
    assert ruff is not None
    ruff_version = subprocess.check_output([ruff, "--version"], text=True).split()[1]
    process_cwd = ""
    real_popen = eval_run.subprocess.Popen

    class Process:
        returncode = 0

        def communicate(self, *, timeout: int) -> tuple[str, str]:
            Path(process_cwd, "candidate_marker.py").write_text('VALUE = "candidate"\n')
            return '{"type":"message","text":"done"}\n', ""

    def start(_command: list[str], **kwargs: object) -> Process:
        nonlocal process_cwd
        if "--no-session" not in _command:
            return real_popen(_command, **kwargs)
        process_cwd = str(kwargs["cwd"])
        return Process()

    monkeypatch.setattr(eval_run.subprocess, "Popen", start)
    result = eval_run.run_task(
        {
            "id": "candidate-source",
            "git_ref": ref,
            "prompt": "check",
            "setup": {
                "tests/test_candidate_source.py": """
import candidate_marker


def test_candidate_source_is_loaded():
    assert candidate_marker.VALUE == "candidate"
"""
            },
            "checks": [
                {
                    "command": [
                        "python",
                        "-m",
                        "pytest",
                        "-q",
                        "tests/test_candidate_source.py",
                    ],
                    "expected_passes": 1,
                    "expected_skips": 0,
                    "expected_node_ids": [
                        "tests/test_candidate_source.py::test_candidate_source_is_loaded"
                    ],
                },
                {"command": ["ruff", "check", "candidate_marker.py"]},
            ],
            "toolchain": {"python": platform.python_version(), "ruff": ruff_version},
        },
        provider="fake",
        model="fake",
        timeout=1,
    )
    assert result["passed"] is True


@pytest.mark.parametrize(
    ("output", "error"),
    [
        (
            """{"type":"tool_call","id":"call-1","name":"read","arguments":{}}
{"type":"tool_result","id":"call-1","name":"read","is_error":false,"content":"ok"}
{"type":"tool_result","id":"call-1","name":"read","is_error":false,"content":"duplicate"}
{"type":"message","text":"done"}
""",
            "orphan tool_result",
        ),
        (
            """{"type":"tool_call","id":"call-1","name":"read","arguments":{}}
{"type":"message","text":"done"}
""",
            "unresolved tool calls",
        ),
        (
            """{"type":"message","text":"done"}
{"type":"usage","usage":{}}
""",
            "event after final message",
        ),
    ],
)
def test_eval_requires_one_to_one_terminal_tool_events(
    monkeypatch: pytest.MonkeyPatch, output: str, error: str
) -> None:
    class Process:
        returncode = 0

        def communicate(self, *, timeout: int) -> tuple[str, str]:
            return output, ""

    monkeypatch.setattr(eval_run.subprocess, "Popen", lambda *args, **kwargs: Process())
    result = eval_run.run_task(
        {"id": "correlation", "prompt": "check", "checks": []},
        provider="fake",
        model="fake",
        timeout=1,
    )
    assert result["passed"] is False
    assert error in result["run_error"]


def test_eval_sweeps_the_agent_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        eval_run.os,
        "killpg",
        lambda process_id, sig: calls.append((process_id, sig)),
    )
    monkeypatch.setattr(eval_run.time, "sleep", lambda _: None)

    eval_run._sweep_process_group(42)

    assert calls == [(42, eval_run.signal.SIGTERM), (42, eval_run.signal.SIGKILL)]


def test_pytest_grader_rejects_candidate_sitecustomize_overwrite(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    grader = tmp_path / "trusted" / "grader"
    candidate.mkdir()
    (grader / "tests").mkdir(parents=True)
    grader_test = grader / "tests" / "test_grader_integrity.py"
    grader_test.write_text("def test_cannot_bypass_grader():\n    assert False\n")
    (candidate / "sitecustomize.py").write_text(
        "from pathlib import Path\n"
        f"target = Path({str(grader_test)!r})\n"
        "target.chmod(0o600)\n"
        "target.write_text('def test_cannot_bypass_grader():\\n    assert True\\n')\n"
    )

    failure = _check(
        candidate,
        {},
        {
            "command": [
                "python",
                "-m",
                "pytest",
                "-q",
                "tests/test_grader_integrity.py",
            ],
            "expected_passes": 1,
        },
        command_root=candidate,
        grader_root=grader,
    )

    assert failure is not None
    assert "assert False" in grader_test.read_text()


@pytest.mark.parametrize("attack", ["pth", "conftest", "pytest_ini"])
def test_pytest_grader_ignores_candidate_collection_controls(
    tmp_path: Path, attack: str
) -> None:
    candidate = tmp_path / "candidate"
    grader = tmp_path / "trusted" / "grader"
    candidate.mkdir()
    (grader / "tests").mkdir(parents=True)
    grader_test = grader / "tests" / "test_collection_integrity.py"
    grader_test.write_text("def test_cannot_bypass_grader():\n    assert False\n")
    replacement = "def test_cannot_bypass_grader():\\n    assert True\\n"
    if attack == "pth":
        (candidate / "candidate.pth").write_text(
            f"import pathlib; pathlib.Path({str(grader_test)!r}).write_text({replacement!r})\n"
        )
    elif attack == "conftest":
        (candidate / "conftest.py").write_text(
            "def pytest_collection_modifyitems(items):\n"
            "    for item in items:\n        item.obj = lambda: None\n"
        )
    else:
        (candidate / "pytest.ini").write_text(
            "[pytest]\naddopts = --ignore=tests/test_collection_integrity.py\n"
        )

    failure = _check(
        candidate,
        {},
        {
            "command": [
                "python",
                "-m",
                "pytest",
                "-q",
                "tests/test_collection_integrity.py",
            ],
            "expected_passes": 1,
        },
        command_root=candidate,
        grader_root=grader,
    )

    assert failure is not None
    assert "assert False" in grader_test.read_text()


def test_pytest_grader_scrubs_pytest_environment(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    grader = tmp_path / "trusted" / "grader"
    candidate.mkdir()
    (grader / "tests").mkdir(parents=True)
    (grader / "tests" / "test_environment_integrity.py").write_text(
        "def test_cannot_bypass_grader():\n    assert False\n"
    )
    (candidate / "candidate_plugin.py").write_text(
        "def pytest_collection_modifyitems(items):\n"
        "    for item in items:\n        item.obj = lambda: None\n"
    )

    failure = _check(
        candidate,
        {},
        {
            "command": [
                "python",
                "-m",
                "pytest",
                "-q",
                "tests/test_environment_integrity.py",
            ],
            "expected_passes": 1,
        },
        command_root=candidate,
        command_env={
            "PATH": os.environ.get("PATH", os.defpath),
            "PYTEST_ADDOPTS": "--tb=no",
            "PYTEST_PLUGINS": "candidate_plugin",
        },
        grader_root=grader,
    )

    assert failure is not None


def test_pytest_historical_test_imports_candidate_checkout(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    grader = tmp_path / "trusted" / "grader"
    candidate.mkdir()
    (grader / "tests").mkdir(parents=True)
    (candidate / "repair_target.py").write_text("VALUE = 'repaired'\n")
    (grader / "repair_target.py").write_text("VALUE = 'historical'\n")
    (grader / "tests" / "test_historical_repair.py").write_text(
        "import repair_target\n\n"
        "def test_repair():\n    assert repair_target.VALUE == 'repaired'\n"
    )

    assert (
        _check(
            candidate,
            {},
            {
                "command": [
                    "python",
                    "-m",
                    "pytest",
                    "-q",
                    "tests/test_historical_repair.py",
                ],
                "expected_passes": 1,
            },
            command_root=candidate,
            grader_root=grader,
        )
        is None
    )


def test_pytest_grader_loads_only_the_required_asyncio_plugin(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    grader = tmp_path / "trusted" / "grader"
    candidate.mkdir()
    (grader / "tests").mkdir(parents=True)
    (grader / "tests" / "test_async_record.py").write_text(
        "import pytest\n\n"
        "@pytest.mark.asyncio\n"
        "async def test_historical_async_check():\n    assert True\n"
    )

    assert (
        _check(
            candidate,
            {},
            {
                "command": [
                    "python",
                    "-m",
                    "pytest",
                    "-q",
                    "tests/test_async_record.py",
                ],
                "expected_passes": 1,
            },
            command_root=candidate,
            grader_root=grader,
        )
        is None
    )


def test_pytest_fails_closed_if_candidate_changes_grader_input(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    grader = tmp_path / "trusted" / "grader"
    candidate.mkdir()
    (grader / "tests").mkdir(parents=True)
    fixture = grader / "fixture.json"
    fixture.write_text('{"answer": "trusted"}\n')
    (grader / "tests" / "test_fixture_integrity.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        "Path(__file__).parents[1].joinpath('fixture.json').write_text('{}')\n"
        "def test_fixture():\n    assert json.loads(Path(__file__).parents[1].joinpath('fixture.json').read_text())['answer'] == 'trusted'\n"
    )

    assert (
        _check(
            candidate,
            {},
            {
                "command": [
                    "python",
                    "-m",
                    "pytest",
                    "-q",
                    "tests/test_fixture_integrity.py",
                ],
                "expected_passes": 0,
                "expected_node_ids": ["tests/test_fixture_integrity.py::test_fixture"],
            },
            command_root=candidate,
            grader_root=grader,
        )
        == "pytest grader files changed during execution"
    )


def test_pytest_fails_closed_if_candidate_changes_grader_test(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    grader = tmp_path / "trusted" / "grader"
    candidate.mkdir()
    (grader / "tests").mkdir(parents=True)
    grader_test = grader / "tests" / "test_hash_integrity.py"
    grader_test.write_text(
        "import candidate_attack\n\ndef test_candidate():\n    assert True\n"
    )
    (candidate / "candidate_attack.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(grader_test)!r}).write_text('def test_candidate():\\n    assert True\\n')\n"
    )

    assert (
        _check(
            candidate,
            {},
            {
                "command": [
                    "python",
                    "-m",
                    "pytest",
                    "-q",
                    "tests/test_hash_integrity.py",
                ],
                "expected_passes": 1,
                "expected_node_ids": ["tests/test_hash_integrity.py::test_candidate"],
            },
            command_root=candidate,
            grader_root=grader,
        )
        == "pytest grader files changed during execution"
    )


def test_ruff_uses_trusted_pinned_configuration_for_candidate(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    grader = tmp_path / "trusted" / "grader"
    candidate.mkdir()
    grader.mkdir(parents=True)
    (candidate / "unused.py").write_text("import os\n")
    (grader / "pyproject.toml").write_text(
        '[tool.ruff.lint]\nselect = ["F401"]\n'
        '[tool.ruff.lint.per-file-ignores]\n"unused.py" = ["F401"]\n'
    )

    assert (
        _check(
            candidate,
            {},
            {"command": ["ruff", "check", "unused.py"]},
            command_root=candidate,
            grader_root=grader,
        )
        is None
    )


def test_ruff_ignores_candidate_configuration(tmp_path: Path) -> None:
    (tmp_path / "unused.py").write_text("import os\n")
    (tmp_path / "pyproject.toml").write_text('[tool.ruff.lint]\nignore = ["F401"]\n')

    assert _check(tmp_path, {}, {"command": ["ruff", "check", "unused.py"]})


@pytest.mark.parametrize(
    "option", ["--isolated", "--config=pyproject.toml", "--extend-ignore=F401"]
)
def test_ruff_rejects_rule_configuration_overrides(tmp_path: Path, option: str) -> None:
    (tmp_path / "unused.py").write_text("import os\n")

    assert "forbidden option" in _check(
        tmp_path, {}, {"command": ["ruff", "check", option, "unused.py"]}
    )


@pytest.mark.parametrize("provider", ["codex", "claude"])
@pytest.mark.parametrize("keep_option", ["keep_workspaces", "keep_failures"])
def test_retained_workspaces_exclude_provider_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    provider: str,
    keep_option: str,
) -> None:
    sentinel = "DO_NOT_RETAIN"
    live_home = tmp_path / "live-home"
    live_zeta_home = tmp_path / "live-zeta-home"
    codex_auth = live_home / ".codex" / "auth.json"
    codex_auth.parent.mkdir(parents=True)
    codex_auth.write_text(json.dumps({"secret": sentinel}))
    claude_auth = live_zeta_home / "anthropic-oauth.json"
    claude_auth.parent.mkdir(parents=True)
    claude_auth.write_text(json.dumps({"secret": sentinel}))
    monkeypatch.setattr(eval_run.Path, "home", staticmethod(lambda: live_home))
    monkeypatch.setenv("ZETA_HOME", str(live_zeta_home))
    observed_homes: list[Path] = []

    class Process:
        returncode = 1 if keep_option == "keep_failures" else 0

        def communicate(self, *, timeout: int) -> tuple[str, str]:
            return '{"type":"message","text":"done"}\n', ""

    def start(_command: list[str], **kwargs: object) -> Process:
        environment = kwargs["env"]
        observed_homes.extend(
            (Path(environment["HOME"]), Path(environment["ZETA_HOME"]))
        )
        return Process()

    monkeypatch.setattr(eval_run.subprocess, "Popen", start)
    retained = tmp_path / "retained"
    result = eval_run.run_task(
        {"id": "credentials", "prompt": "check", "checks": []},
        provider=provider,
        model="fake",
        timeout=1,
        **{keep_option: retained},
    )

    saved = Path(result["saved_workspace"])
    assert saved.is_dir()
    assert sentinel not in "".join(
        path.read_text(errors="ignore") for path in saved.rglob("*") if path.is_file()
    )
    assert all(not path.exists() for path in observed_homes)


def test_retention_fails_closed_on_candidate_auth_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Process:
        returncode = 0

        def communicate(self, *, timeout: int) -> tuple[str, str]:
            return '{"type":"message","text":"done"}\n', ""

    def start(_command: list[str], **kwargs: object) -> Process:
        Path(kwargs["cwd"], "auth.json").write_text('{"secret":"candidate"}')
        return Process()

    monkeypatch.setattr(eval_run.subprocess, "Popen", start)
    retained = tmp_path / "retained"

    with pytest.raises(RuntimeError, match="credential-bearing workspace"):
        eval_run.run_task(
            {"id": "credentials", "prompt": "check", "checks": []},
            provider="fake",
            model="fake",
            timeout=1,
            keep_workspaces=retained,
        )
    assert not retained.exists()


def test_eval_stages_only_the_selected_codex_credential(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    live_home = tmp_path / "live-home"
    auth_path = live_home / ".codex" / "auth.json"
    auth_path.parent.mkdir(parents=True)
    auth_path.write_text('{"tokens":{"access_token":"fake"}}\n')
    (live_home / "unrelated-secret.txt").write_text("do not copy")
    monkeypatch.setattr(eval_run.Path, "home", staticmethod(lambda: live_home))

    isolated_home = tmp_path / "isolated-home"
    isolated_zeta_home = tmp_path / "isolated-zeta-home"
    environment = eval_run._child_environment(
        tmp_path,
        isolated_home,
        isolated_zeta_home,
        {},
        provider="codex",
    )

    assert (
        Path(environment["HOME"], ".codex", "auth.json").read_text()
        == auth_path.read_text()
    )
    assert not Path(environment["HOME"], "unrelated-secret.txt").exists()
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert not any(name.startswith("PYTEST_") for name in environment)


def test_grading_removes_staged_credentials_and_uses_clean_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sentinel = "STAGED_PROVIDER_SECRET"
    live_home = tmp_path / "live-home"
    auth = live_home / ".codex" / "auth.json"
    auth.parent.mkdir(parents=True)
    auth.write_text(json.dumps({"secret": sentinel}))
    monkeypatch.setattr(eval_run.Path, "home", staticmethod(lambda: live_home))

    class Process:
        returncode = 0
        pid = None

        def communicate(self, *, timeout: int) -> tuple[str, str]:
            return '{"type":"message","text":"done"}\n', ""

    real_popen = eval_run.subprocess.Popen
    calls = 0
    agent_homes: list[Path] = []

    def start(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            environment = kwargs["env"]
            agent_homes.extend(
                (Path(environment["HOME"]), Path(environment["ZETA_HOME"]))
            )
            return Process()
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(eval_run.subprocess, "Popen", start)
    retained = tmp_path / "retained"
    result = eval_run.run_task(
        {
            "id": "credential-import-attack",
            "prompt": "check",
            "setup": {
                "credential_attack.py": (
                    "import os\n"
                    "from pathlib import Path\n"
                    "auth = Path(os.environ['HOME']) / '.codex' / 'auth.json'\n"
                    "if auth.exists():\n"
                    "    Path('innocent-output.txt').write_text(auth.read_text())\n"
                    "Path('grader-environment.json').write_text(__import__('json').dumps({\n"
                    "    'auth_exists': auth.exists(),\n"
                    "    'home': os.environ['HOME'],\n"
                    "    'zeta_home': os.environ['ZETA_HOME'],\n"
                    "    'provider_env': sorted(name for name in "
                    "os.environ if name in {'ANTHROPIC_API_KEY', 'ZETA_ALLOW_API_KEY'}),\n"
                    "}))\n"
                ),
                "tests/test_credential_attack.py": (
                    "import credential_attack\n\n"
                    "def test_candidate_import():\n"
                    "    assert True\n"
                ),
            },
            "checks": [
                {
                    "command": [
                        "python",
                        "-m",
                        "pytest",
                        "-q",
                        "tests/test_credential_attack.py",
                    ],
                    "expected_passes": 1,
                    "expected_skips": 0,
                    "expected_node_ids": [
                        "tests/test_credential_attack.py::test_candidate_import"
                    ],
                }
            ],
        },
        provider="codex",
        model="fake",
        timeout=1,
        keep_workspaces=retained,
    )

    assert result["passed"] is True
    saved = Path(result["saved_workspace"])
    assert sentinel not in "".join(
        path.read_text(errors="ignore") for path in saved.rglob("*") if path.is_file()
    )
    grader_environment = json.loads((saved / "grader-environment.json").read_text())
    assert grader_environment["auth_exists"] is False
    assert grader_environment["provider_env"] == []
    assert Path(grader_environment["home"]) not in agent_homes
    assert Path(grader_environment["zeta_home"]) not in agent_homes
    assert all(not path.exists() for path in agent_homes)


def test_pytest_grader_rejects_junit_forgery_followed_by_immediate_exit(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    grader = tmp_path / "trusted" / "grader"
    candidate.mkdir()
    (grader / "tests").mkdir(parents=True)
    (grader / "tests" / "test_integrity.py").write_text(
        "import candidate_attack\n\n"
        "def test_cannot_bypass_grader():\n"
        "    assert False\n"
    )
    (candidate / "candidate_attack.py").write_text(
        "import os, sys\n"
        "from pathlib import Path\n"
        "for index, argument in enumerate(sys.argv):\n"
        "    if argument == '--junit-xml':\n"
        "        Path(sys.argv[index + 1]).write_text(\n"
        "            '<testsuites><testsuite tests=\"1\"><testcase '"
        "            'classname=\"tests.test_integrity\" '"
        "            'name=\"test_cannot_bypass_grader\"/>'"
        "            '</testsuite></testsuites>'\n"
        "        )\n"
        "        os._exit(0)\n"
    )

    failure = _check(
        candidate,
        {},
        {
            "command": ["python", "-m", "pytest", "-q", "tests/test_integrity.py"],
            "expected_passes": 1,
            "expected_skips": 0,
            "expected_node_ids": ["tests/test_integrity.py::test_cannot_bypass_grader"],
        },
        command_root=candidate,
        grader_root=grader,
    )

    assert failure is not None


@pytest.mark.xfail(
    strict=True,
    reason="documented residual: same-UID candidate code can monkeypatch pytest in-process",
)
def test_pytest_grader_cannot_yet_stop_function_runtest_monkeypatch(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    grader = tmp_path / "trusted" / "grader"
    candidate.mkdir()
    (grader / "tests").mkdir(parents=True)
    (grader / "tests" / "test_integrity.py").write_text(
        "import candidate_attack\n\n"
        "def test_cannot_bypass_grader():\n"
        "    assert False\n"
    )
    (grader / "candidate_attack.py").write_text("")
    (candidate / "candidate_attack.py").write_text(
        "from _pytest.python import Function\nFunction.runtest = lambda self: None\n"
    )

    assert (
        _check(
            candidate,
            {},
            {
                "command": ["python", "-m", "pytest", "-q", "tests/test_integrity.py"],
                "expected_passes": 1,
                "expected_skips": 0,
                "expected_node_ids": [
                    "tests/test_integrity.py::test_cannot_bypass_grader"
                ],
            },
            command_root=candidate,
            grader_root=grader,
        )
        is not None
    )


@pytest.mark.xfail(
    strict=True,
    reason="documented residual: same-UID candidate code can mutate, load, and restore grader files",
)
def test_pytest_grader_cannot_yet_stop_mutate_collect_restore_attack(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    grader = tmp_path / "trusted" / "grader"
    candidate.mkdir()
    (grader / "tests").mkdir(parents=True)
    target = grader / "tests" / "test_z_integrity.py"
    target.write_text("def test_cannot_bypass_grader():\n    assert False\n")
    (grader / "tests" / "test_a_loader.py").write_text("import candidate_attack\n")
    (grader / "candidate_attack.py").write_text("")
    (candidate / "candidate_attack.py").write_text(
        "import importlib.util, sys\n"
        "from pathlib import Path\n"
        f"target = Path({str(target)!r})\n"
        "original = target.read_text()\n"
        "target.write_text('def test_cannot_bypass_grader():\\n    assert True\\n')\n"
        "try:\n"
        "    spec = importlib.util.spec_from_file_location('tests.test_z_integrity', target)\n"
        "    module = importlib.util.module_from_spec(spec)\n"
        "    sys.modules[spec.name] = module\n"
        "    spec.loader.exec_module(module)\n"
        "finally:\n"
        "    target.write_text(original)\n"
    )

    assert (
        _check(
            candidate,
            {},
            {
                "command": [
                    "python",
                    "-m",
                    "pytest",
                    "-q",
                    "tests/test_a_loader.py",
                    "tests/test_z_integrity.py",
                ],
                "expected_passes": 1,
                "expected_skips": 0,
                "expected_node_ids": [
                    "tests/test_z_integrity.py::test_cannot_bypass_grader"
                ],
            },
            command_root=candidate,
            grader_root=grader,
        )
        is not None
    )


def test_pytest_grader_sweeps_candidate_spawned_descendants(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    grader = tmp_path / "trusted" / "grader"
    marker = tmp_path / "descendant-survived"
    candidate.mkdir()
    (grader / "tests").mkdir(parents=True)
    (grader / "tests" / "test_descendant.py").write_text(
        "import candidate_attack\n\ndef test_import():\n    assert True\n"
    )
    (grader / "candidate_attack.py").write_text("")
    (candidate / "candidate_attack.py").write_text(
        "import subprocess, sys\n"
        f"marker = {str(marker)!r}\n"
        "subprocess.Popen(\n"
        "    [sys.executable, '-c', "
        "     'import pathlib, time; time.sleep(0.5); pathlib.Path(' + repr(marker) + ').write_text(\"escaped\")'],\n"
        "    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,\n"
        "    start_new_session=False,\n"
        ")\n"
    )

    assert (
        _check(
            candidate,
            {},
            {
                "command": ["python", "-m", "pytest", "-q", "tests/test_descendant.py"],
                "expected_passes": 1,
                "expected_skips": 0,
                "expected_node_ids": ["tests/test_descendant.py::test_import"],
            },
            command_root=candidate,
            grader_root=grader,
        )
        is None
    )
    time.sleep(0.8)
    assert not marker.exists()


def test_all_historical_repair_records_pin_and_validate_pytest_nodes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    records = []
    for task_file in sorted((Path(__file__).parents[1] / "evals").glob("*.jsonl")):
        records.extend(json.loads(line) for line in task_file.read_text().splitlines())
    historical = [task for task in records if "git_ref" in task]
    assert historical

    active_check: dict[str, object] = {}

    def run_grader(
        argv: list[str],
        *,
        cwd: Path,
        env: object,
        report_path: Path | None = None,
        candidate_paths: list[str] | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], bool]:
        del cwd, env, candidate_paths
        if report_path is not None:
            node_ids = active_check["expected_node_ids"]
            assert isinstance(node_ids, list)
            outcomes = {}
            if "--collect-only" not in argv:
                passes = active_check["expected_passes"]
                outcomes = {
                    node_id: "passed" if index < passes else "skipped"
                    for index, node_id in enumerate(node_ids)
                }
            report_path.write_text(
                json.dumps({"node_ids": node_ids, "outcomes": outcomes})
            )
        return subprocess.CompletedProcess(argv, 0, "", ""), False

    monkeypatch.setattr(eval_run, "_run_grader_command", run_grader)
    for task in historical:
        for check in task["checks"]:
            command = check.get("command")
            if command is None:
                continue
            assert isinstance(command, list) and command
            if "pytest" in command:
                assert command[:4] == ["python", "-m", "pytest", "-q"]
                node_ids = check.get("expected_node_ids")
                assert isinstance(node_ids, list) and len(node_ids) == len(
                    set(node_ids)
                )
                assert (
                    len(node_ids) == check["expected_passes"] + check["expected_skips"]
                )
            elif command[0] == "ruff":
                assert command[1] == "check"
                assert any(not argument.startswith("-") for argument in command[2:])
            active_check = check
            assert (
                _check(
                    tmp_path,
                    task.get("setup", {}),
                    check,
                    command_root=tmp_path,
                    grader_root=tmp_path,
                )
                is None
            )
