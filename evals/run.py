"""Run isolated, artifact- or tool-result-graded tasks through Zeta."""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

TASKS = Path(__file__).with_name("tasks.jsonl")
_EVENT_TYPES = frozenset(
    {
        "turn_start",
        "tool_call",
        "tool_result",
        "usage",
        "child_usage",
        "retry",
        "turn_end",
        "error",
        "message",
    }
)
_PROVIDER_ENV_NAMES = frozenset({"ANTHROPIC_API_KEY", "ZETA_ALLOW_API_KEY"})


def _file(root: Path, name: str) -> Path:
    path = Path(name)
    if not name or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe eval path: {name!r}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"eval path escapes workspace: {name!r}")
    return resolved


@contextlib.contextmanager
def _local_site(root: Path, name: str):
    shutil.copyfile(_file(TASKS.parent, name), _file(root, name))
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _command_environment(
    root: Path, *, base: Mapping[str, str] | None = None
) -> dict[str, str]:
    # The evaluated agent needs provider auth, but its children do not inherit
    # ambient pytest controls from the machine running the harness.
    env = {
        name: value
        for name, value in (base or {}).items()
        if not name.startswith("PYTEST_")
    }
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(root / "src"), str(root)) if part
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _is_pytest_command(argv: list[str]) -> bool:
    return "pytest" in argv or (
        "-m" in argv
        and argv[argv.index("-m") + 1 :]
        and argv[argv.index("-m") + 1] == "pytest"
    )


def _is_ruff_command(argv: list[str]) -> bool:
    return bool(argv) and Path(argv[0]).name == "ruff"


def _candidate_import_paths(root: Path, overlay: Path) -> list[str]:
    overlay.mkdir()
    excluded = {
        "src",
        "tests",
        "conftest.py",
        "sitecustomize.py",
        "usercustomize.py",
        "pytest.ini",
        "pyproject.toml",
    }
    for path in root.iterdir():
        if path.name in excluded or path.name.startswith("."):
            continue
        (overlay / path.name).symlink_to(path, target_is_directory=path.is_dir())
    return [str(root / "src"), str(overlay)]


_GRADER_CACHE_DIRS = frozenset(
    {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".hypothesis",
        ".cache",
        ".nox",
        ".tox",
    }
)


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and not any(part in _GRADER_CACHE_DIRS for part in path.relative_to(root).parts)
    }


def _pytest_report(path: Path) -> tuple[list[str], dict[str, str]] | None:
    try:
        report = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if (
        type(report) is not dict
        or type(report.get("node_ids")) is not list
        or any(type(node_id) is not str for node_id in report["node_ids"])
        or len(set(report["node_ids"])) != len(report["node_ids"])
        or type(report.get("outcomes")) is not dict
        or any(
            type(node_id) is not str or outcome not in {"passed", "failed", "skipped"}
            for node_id, outcome in report["outcomes"].items()
        )
    ):
        return None
    return report["node_ids"], report["outcomes"]


_PYTEST_BOOTSTRAP = """
import json
import os
import sys

import pytest
import pytest_asyncio.plugin

report_fd = REPORT_FD
report_path = os.read(report_fd, 65536).decode()
os.close(report_fd)

class TrustedReport:
    def __init__(self):
        self.node_ids = []
        self.canonical_ids = {}
        self.outcomes = {}

    def pytest_collection_finish(self, session):
        for item in session.items:
            try:
                relative_path = (
                    item.path.resolve()
                    .relative_to(session.config.rootpath.resolve())
                    .as_posix()
                )
            except ValueError:
                canonical_id = item.nodeid
            else:
                separator = item.nodeid.find("::")
                suffix = item.nodeid[separator:] if separator >= 0 else ""
                canonical_id = relative_path + suffix
            self.canonical_ids[item.nodeid] = canonical_id
            self.node_ids.append(canonical_id)

    def pytest_runtest_logreport(self, report):
        node_id = self.canonical_ids.get(report.nodeid, report.nodeid)
        if report.when == "setup" and report.outcome != "passed":
            self.outcomes[node_id] = report.outcome
        elif report.when == "call":
            self.outcomes[node_id] = report.outcome
        elif report.when == "teardown" and report.outcome == "failed":
            self.outcomes[node_id] = "failed"

    def pytest_sessionfinish(self, session, exitstatus):
        with open(report_path, "x", encoding="utf-8") as destination:
            json.dump({"node_ids": self.node_ids, "outcomes": self.outcomes}, destination)

plugin = TrustedReport()
sys.dont_write_bytecode = True
sys.path[:0] = CANDIDATE_PATHS
raise SystemExit(pytest.main(sys.argv[1:], plugins=[plugin]))
"""


def _run_grader_command(
    argv: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    report_path: Path | None = None,
    candidate_paths: list[str] | None = None,
    timeout: float = 30,
) -> tuple[subprocess.CompletedProcess[str] | None, bool]:
    read_fd = write_fd = None
    pass_fds: tuple[int, ...] = ()
    if report_path is not None:
        read_fd, write_fd = os.pipe()
        os.write(write_fd, os.fsencode(report_path))
        os.close(write_fd)
        write_fd = None
        pass_fds = (read_fd,)
        bootstrap = _PYTEST_BOOTSTRAP.replace("REPORT_FD", str(read_fd)).replace(
            "CANDIDATE_PATHS", repr(candidate_paths or [])
        )
        argv = [sys.executable, "-I", "-c", bootstrap, *argv]
    process = None
    timed_out = False
    try:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            pass_fds=pass_fds,
        )
        if read_fd is not None:
            os.close(read_fd)
            read_fd = None
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        result = subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
    except OSError:
        result = None
    finally:
        if read_fd is not None:
            os.close(read_fd)
        if write_fd is not None:
            os.close(write_fd)
        _sweep_process_group(getattr(process, "pid", None))
    return result, timed_out


def _check(
    root: Path,
    setup: dict[str, str],
    check: dict[str, Any],
    *,
    events: list[dict[str, Any]] | None = None,
    command_root: Path | None = None,
    command_env: Mapping[str, str] | None = None,
    grader_root: Path | None = None,
    command_timeout: float = 30,
) -> str | None:
    if "allowed_tools" in check:
        allowed = check["allowed_tools"]
        if type(allowed) is not list or any(
            type(name) is not str or not name for name in allowed
        ):
            raise ValueError("allowed_tools must be a list of nonempty tool names")
        return next(
            (
                f"disallowed tool: {event['name']}"
                for event in events or []
                if event.get("type") == "tool_call" and event["name"] not in allowed
            ),
            None,
        )

    if "command" in check:
        command = check["command"]
        if (
            not isinstance(command, list)
            or not command
            or any(type(part) is not str for part in command)
        ):
            raise ValueError("check command must be a nonempty argv list")
        command_root = command_root or root
        argv = [sys.executable if command[0] == "python" else command[0], *command[1:]]
        execution_root = (
            grader_root
            if grader_root is not None and _is_pytest_command(argv)
            else command_root
        )
        env = _command_environment(
            command_root,
            base=command_env or {"PATH": os.environ.get("PATH", os.defpath)},
        )
        if _is_ruff_command(argv) and "check" in argv:
            check_index = argv.index("check")
            ruff_args = argv[check_index + 1 :]
            forbidden = next(
                (
                    part
                    for part in ruff_args
                    if part in {"--config", "--isolated"}
                    or part.startswith(("--config=", "--extend-"))
                ),
                None,
            )
            if forbidden is not None:
                return f"ruff check contains forbidden option: {forbidden}"
            if not any(not part.startswith("-") for part in ruff_args):
                return "ruff check requires an explicit target"
            if grader_root is not None:
                trusted_config = grader_root / "pyproject.toml"
                if trusted_config.is_file():
                    argv[check_index + 1 : check_index + 1] = [
                        "--config",
                        str(trusted_config),
                    ]
                else:
                    argv.insert(check_index + 1, "--isolated")
            else:
                argv.insert(check_index + 1, "--isolated")
        report_path = execution_root.parent / f"zeta-pytest-{uuid.uuid4().hex}.json"
        collect_path = execution_root.parent / f"zeta-collect-{uuid.uuid4().hex}.json"
        config_path = execution_root.parent / f"zeta-pytest-{uuid.uuid4().hex}.ini"
        grader_hashes = None
        if _is_pytest_command(argv):
            env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
            expected_passes = check.get("expected_passes")
            expected_skips = check.get("expected_skips", 0)
            if type(expected_passes) is not int or expected_passes < 0:
                return "pytest check must declare a nonnegative expected_passes"
            if type(expected_skips) is not int or expected_skips < 0:
                return "pytest check must declare a nonnegative expected_skips"
            config_path.write_text("[pytest]\naddopts =\nasyncio_mode = auto\n")
            pytest_index = argv.index("pytest")
            pytest_args = [
                *argv[pytest_index + 1 :],
                "-s",
                "-p",
                "no:cacheprovider",
                "-p",
                "pytest_asyncio.plugin",
                "-c",
                str(config_path),
                "--rootdir",
                str(execution_root),
                "--confcutdir",
                str(execution_root),
                "--import-mode=importlib",
            ]
            # Historical task records pin grader-owned node IDs. Ad hoc checks may
            # instead derive them in a clean collection process before candidate paths
            # are importable. The actual run must report each exact node and outcome.
            expected_node_ids = check.get("expected_node_ids")
            if expected_node_ids is not None:
                if (
                    type(expected_node_ids) is not list
                    or any(type(node_id) is not str for node_id in expected_node_ids)
                    or len(set(expected_node_ids)) != len(expected_node_ids)
                ):
                    config_path.unlink(missing_ok=True)
                    return "pytest expected_node_ids must be a unique string list"
            else:
                pristine_paths = [str(execution_root / "src"), str(execution_root)]
                collect_result, collect_timed_out = _run_grader_command(
                    [*pytest_args, "--collect-only"],
                    cwd=execution_root,
                    env=env,
                    report_path=collect_path,
                    candidate_paths=pristine_paths,
                    timeout=command_timeout,
                )
                collected = _pytest_report(collect_path)
                collect_path.unlink(missing_ok=True)
                if collect_timed_out:
                    config_path.unlink(missing_ok=True)
                    return f"pytest collection timed out: {command[0]}"
                if (
                    collect_result is None
                    or collect_result.returncode != 0
                    or collected is None
                ):
                    config_path.unlink(missing_ok=True)
                    return (
                        f"pytest could not collect pristine grader nodes: {command[0]}"
                    )
                expected_node_ids, _ = collected
            import_overlay = execution_root.parent / f"zeta-imports-{uuid.uuid4().hex}"
            candidate_paths = _candidate_import_paths(command_root, import_overlay)
            # This detects persistent grader-file tampering only. Same-UID candidate
            # code can still mutate, load, and restore files between these snapshots.
            if grader_root is not None:
                grader_hashes = _file_hashes(execution_root)
            result, timed_out = _run_grader_command(
                pytest_args,
                cwd=execution_root,
                env=env,
                report_path=report_path,
                candidate_paths=candidate_paths,
                timeout=command_timeout,
            )
            shutil.rmtree(import_overlay, ignore_errors=True)
        else:
            result, timed_out = _run_grader_command(
                argv, cwd=execution_root, env=env, timeout=command_timeout
            )
        config_path.unlink(missing_ok=True)
        if timed_out:
            return f"command timed out: {command[0]}"
        if result is None:
            return f"command failed to start: {command[0]}"
        if grader_hashes is not None:
            if _file_hashes(execution_root) != grader_hashes:
                report_path.unlink(missing_ok=True)
                return "pytest grader files changed during execution"
            report = _pytest_report(report_path)
            report_path.unlink(missing_ok=True)
            if report is None:
                return f"pytest produced no valid report: {command[0]}"
            node_ids, outcomes = report
            if node_ids != expected_node_ids or set(outcomes) != set(expected_node_ids):
                return (
                    f"pytest node IDs or outcomes differed: {command[0]} "
                    f"nodes={node_ids!r} outcomes={sorted(outcomes)!r} "
                    f"expected={expected_node_ids!r}"
                )
            passed = sum(outcome == "passed" for outcome in outcomes.values())
            skipped = sum(outcome == "skipped" for outcome in outcomes.values())
            if (passed, skipped) != (expected_passes, expected_skips):
                return (
                    f"pytest counts differed: {command[0]} "
                    f"passed={passed} skipped={skipped}; "
                    f"expected passed={expected_passes} skipped={expected_skips}"
                )
        if result.returncode != check.get("exit_code", 0):
            return f"command exited {result.returncode}: {command[0]}"
        if "stdout" in check and result.stdout != check["stdout"]:
            return f"command stdout differed: {command[0]}"
        return None

    if "last_tool_result" in check:
        name = check["last_tool_result"]
        if type(name) is not str or not name:
            raise ValueError("last_tool_result must name a tool")
        result = next(
            (
                event
                for event in reversed(events or [])
                if event.get("type") == "tool_result" and event.get("name") == name
            ),
            None,
        )
        if result is None:
            return f"missing tool result: {name}"
        content = result.get("content")
        if result.get("is_error") is not False or type(content) is not str:
            return f"invalid tool result: {name}"
        if "contains" in check and check["contains"] not in content:
            return f"tool result missing expected text: {name}"
        if "not_contains" in check and check["not_contains"] in content:
            return f"tool result contains forbidden text: {name}"
        return None

    name = check["path"]
    path = _file(root, name)
    if not path.is_file():
        return f"missing file: {name}"
    content = path.read_text()
    if "equals" in check and content != check["equals"]:
        return f"file differed: {name}"
    if "contains" in check and check["contains"] not in content:
        return f"file missing expected text: {name}"
    if (
        "nonempty_lines" in check
        and [line.strip() for line in content.splitlines() if line.strip()]
        != check["nonempty_lines"]
    ):
        return f"file lines differed: {name}"
    if check.get("unchanged") and content != setup[name]:
        return f"setup file changed: {name}"
    return None


def _string(event: dict[str, Any], name: str, *, nonempty: bool = False) -> bool:
    value = event.get(name)
    return type(value) is str and (not nonempty or bool(value))


def _nonnegative_int(value: object) -> bool:
    return type(value) is int and value >= 0


def _valid_usage(value: object) -> bool:
    return type(value) is dict and all(
        _nonnegative_int(item) for item in value.values()
    )


def _validate_event(
    event: object, line_number: int, tool_calls: dict[str, str]
) -> str | None:
    if type(event) is not dict:
        return f"agent emitted non-object JSONL line {line_number}"
    event_type = event.get("type")
    if type(event_type) is not str or event_type not in _EVENT_TYPES:
        return f"agent emitted unknown event type JSONL line {line_number}"
    if event_type == "turn_start":
        if not _string(event, "prompt"):
            return f"agent emitted malformed turn_start JSONL line {line_number}"
    elif event_type == "tool_call":
        if not (
            _string(event, "id", nonempty=True)
            and _string(event, "name", nonempty=True)
        ):
            return f"agent emitted malformed tool_call JSONL line {line_number}"
        if not isinstance(event.get("arguments"), (dict, str)):
            return f"agent emitted malformed tool_call JSONL line {line_number}"
        if "agent_instance_id" in event and not _string(
            event, "agent_instance_id", nonempty=True
        ):
            return f"agent emitted malformed tool_call JSONL line {line_number}"
        if event["id"] in tool_calls:
            return f"agent emitted duplicate tool_call JSONL line {line_number}"
        tool_calls[event["id"]] = event["name"]
    elif event_type == "tool_result":
        if not (
            _string(event, "id", nonempty=True)
            and _string(event, "name", nonempty=True)
            and type(event.get("is_error")) is bool
            and type(event.get("content")) is str
        ):
            return f"agent emitted malformed tool_result JSONL line {line_number}"
        expected_name = tool_calls.pop(event["id"], None)
        if expected_name is None or expected_name != event["name"]:
            return f"agent emitted orphan tool_result JSONL line {line_number}"
    elif event_type == "usage":
        if not _valid_usage(event.get("usage")):
            return f"agent emitted malformed usage JSONL line {line_number}"
    elif event_type == "child_usage":
        by_model = event.get("by_model")
        if not _valid_usage(event.get("usage")):
            return f"agent emitted malformed usage JSONL line {line_number}"
        if type(by_model) is not dict:
            return f"agent emitted malformed child_usage JSONL line {line_number}"
        if any(
            type(model) is not str or not model or not _valid_usage(counts)
            for model, counts in by_model.items()
        ):
            return f"agent emitted malformed child_usage JSONL line {line_number}"
        if any(
            sum(counts.get(name, 0) for counts in by_model.values())
            != event["usage"].get(name, 0)
            for name in (
                "input_tokens",
                "output_tokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
            )
        ):
            return f"agent emitted inconsistent child_usage JSONL line {line_number}"
    elif event_type == "retry":
        if not (
            _string(event, "text")
            and _nonnegative_int(event.get("retry"))
            and type(event.get("delay")) in {int, float}
            and event["delay"] >= 0
        ):
            return f"agent emitted malformed retry JSONL line {line_number}"
        if "is_stall" in event and type(event["is_stall"]) is not bool:
            return f"agent emitted malformed retry JSONL line {line_number}"
    elif event_type == "turn_end":
        if not _nonnegative_int(event.get("tool_calls")):
            return f"agent emitted malformed turn_end JSONL line {line_number}"
    elif event_type == "error":
        if not (_string(event, "code", nonempty=True) and _string(event, "message")):
            return f"agent emitted malformed error JSONL line {line_number}"
    elif event_type == "message":
        if type(event.get("text")) is not str:
            return f"agent emitted malformed message JSONL line {line_number}"
        if "role" in event and event["role"] != "assistant":
            return f"agent emitted malformed message JSONL line {line_number}"
    return None


def _toolchain_error(task: dict[str, Any]) -> str | None:
    if "git_ref" not in task:
        return None
    toolchain = task.get("toolchain")
    if type(toolchain) is not dict:
        return "pinned task must declare a toolchain"
    expected_python = toolchain.get("python")
    expected_ruff = toolchain.get("ruff")
    if type(expected_python) is not str or not expected_python:
        return "pinned task must declare a Python version"
    if type(expected_ruff) is not str or not expected_ruff:
        return "pinned task must declare a Ruff version"
    actual_python = platform.python_version()
    if actual_python != expected_python:
        return (
            f"pinned Python mismatch: expected {expected_python}, got {actual_python}"
        )
    ruff = shutil.which("ruff")
    if ruff is None:
        return "pinned Ruff mismatch: ruff is not installed"
    try:
        result = subprocess.run(
            [ruff, "--version"], capture_output=True, text=True, check=False, timeout=10
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"pinned Ruff version check failed: {exc}"
    match = re.search(r"ruff (\S+)", result.stdout)
    actual_ruff = match.group(1) if match else "unknown"
    if result.returncode != 0 or actual_ruff != expected_ruff:
        return f"pinned Ruff mismatch: expected {expected_ruff}, got {actual_ruff}"
    return None


def _child_environment(
    root: Path,
    home: Path,
    zeta_home: Path,
    provider_env: Mapping[str, str] | None,
    *,
    provider: str | None = None,
) -> dict[str, str]:
    home.mkdir(mode=0o700)
    zeta_home.mkdir(mode=0o700)
    if provider is not None:
        _stage_provider_credential(provider, home, zeta_home)
    env = {
        "PATH": os.environ.get("PATH", os.defpath),
        "HOME": str(home),
        "ZETA_HOME": str(zeta_home),
    }
    env.update(_command_environment(root))
    for name, value in (provider_env or {}).items():
        if (
            provider != "claude"
            or name not in _PROVIDER_ENV_NAMES
            or type(value) is not str
        ):
            raise ValueError(f"unsupported provider environment variable: {name}")
        env[name] = value
    return env


def _stage_provider_credential(provider: str, home: Path, zeta_home: Path) -> None:
    if provider == "codex":
        source = Path.home() / ".codex" / "auth.json"
        destination = home / ".codex" / "auth.json"
    elif provider == "claude":
        live_zeta_home = Path(os.environ.get("ZETA_HOME", Path.home() / ".zeta"))
        source = live_zeta_home / "anthropic-oauth.json"
        destination = zeta_home / "anthropic-oauth.json"
    else:
        return
    if not source.is_file():
        return
    destination.parent.mkdir(mode=0o700, exist_ok=True)
    shutil.copyfile(source, destination)
    os.chmod(destination, 0o600)


def _retention_credential_paths(root: Path) -> list[Path]:
    credential_names = {
        ".claude.json",
        ".credentials.json",
        "auth.json",
        "credentials.json",
    }
    return [
        path
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.name.lower() in credential_names
            or ("oauth" in path.name.lower() and path.suffix.lower() == ".json")
        )
    ]


def _assert_retention_safe(root: Path) -> None:
    credentials = _retention_credential_paths(root)
    if credentials:
        names = ", ".join(str(path.relative_to(root)) for path in credentials)
        raise RuntimeError(f"refusing to retain credential-bearing workspace: {names}")


def _provider_environment(provider: str) -> dict[str, str]:
    if provider != "claude":
        return {}
    return {
        name: os.environ[name] for name in _PROVIDER_ENV_NAMES if name in os.environ
    }


def _immutable_grader(
    task: dict[str, Any], setup: dict[str, str], temporary: Path
) -> Path | None:
    if "git_ref" not in task:
        return None
    grader_root = temporary / "grader"
    source = str(Path(__file__).resolve().parents[1])
    subprocess.run(
        ["git", "clone", "--quiet", "--shared", source, str(grader_root)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(grader_root),
            "checkout",
            "--quiet",
            "--detach",
            task["git_ref"],
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    for name, content in setup.items():
        path = _file(grader_root, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    for path in (grader_root, *grader_root.rglob("*")):
        path.chmod(path.stat().st_mode & ~0o222)
    return grader_root


def _sweep_process_group(process_id: int | None) -> None:
    if process_id is None:
        return
    try:
        os.killpg(process_id, signal.SIGTERM)
    except ProcessLookupError:
        return
    time.sleep(0.05)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process_id, signal.SIGKILL)


def run_task(
    task: dict[str, Any],
    *,
    provider: str,
    model: str,
    timeout: int,
    instruction: str | None = None,
    keep_failures: Path | None = None,
    keep_workspaces: Path | None = None,
    provider_env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    with (
        tempfile.TemporaryDirectory(prefix="zeta-workflow-eval-") as temporary,
        tempfile.TemporaryDirectory(prefix="zeta-eval-grader-") as grader_temporary,
        tempfile.TemporaryDirectory(prefix="zeta-eval-auth-") as auth_temporary,
    ):
        root = Path(temporary)
        if "git_ref" in task:
            ref = task["git_ref"]
            if type(ref) is not str or re.fullmatch(r"[0-9a-f]{40}", ref) is None:
                raise ValueError("git_ref must be a full lowercase commit SHA")
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--quiet",
                    "--shared",
                    str(Path(__file__).resolve().parents[1]),
                    str(root),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            subprocess.run(
                ["git", "-C", str(root), "checkout", "--quiet", "--detach", ref],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            toolchain_error = _toolchain_error(task)
            if toolchain_error is not None:
                raise ValueError(toolchain_error)
        setup = task.get("setup", {})
        for name, content in setup.items():
            path = _file(root, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

        # Provider credentials must be available to the agent process itself.
        # Keeping its homes outside the candidate root prevents accidental retention,
        # not hostile access: the same-UID agent can still copy or exfiltrate them.
        auth_root = Path(auth_temporary)
        child_env = _child_environment(
            root,
            auth_root / "home",
            auth_root / "zeta-home",
            provider_env,
            provider=provider,
        )

        command = [
            str(Path(sys.executable).with_name("zeta")),
            "--no-session",
            "--provider",
            provider,
            "--model",
            model,
            "--yolo",
            "--max-turns",
            str(task.get("max_turns", 12)),
        ]
        if instruction:
            command.extend(("--append-system-prompt", instruction))
        site = (
            _local_site(root, task["local_fixture"])
            if "local_fixture" in task
            else contextlib.nullcontext("")
        )
        with site as base_url:
            command.extend(
                (
                    "--format",
                    "json",
                    "--print",
                    task["prompt"].replace("{base_url}", base_url),
                )
            )
            started = time.monotonic()
            process = subprocess.Popen(
                command,
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                env=child_env,
            )
            timed_out = False
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
            finally:
                _sweep_process_group(getattr(process, "pid", None))

        # Credentials are required only by the agent. Remove their staged files
        # before any candidate code can be imported into a grader process, then
        # grade with fresh empty homes and no provider variables.
        shutil.rmtree(auth_root)
        grader_environment = _child_environment(
            root,
            Path(grader_temporary) / "home",
            Path(grader_temporary) / "zeta-home",
            {},
        )

        events: list[dict[str, Any]] = []
        parse_error = None
        tool_calls: dict[str, str] = {}
        saw_child_usage = False
        lines = stdout.split("\n")
        if lines[-1] == "":
            lines.pop()
        for line_number, line in enumerate(lines, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                parse_error = f"agent emitted invalid JSONL line {line_number}"
                break
            if events and events[-1].get("type") == "message":
                parse_error = (
                    f"agent emitted event after final message JSONL line {line_number}"
                )
                break
            parse_error = _validate_event(event, line_number, tool_calls)
            if parse_error is not None:
                break
            if event["type"] == "child_usage":
                if saw_child_usage:
                    parse_error = (
                        f"agent emitted duplicate child_usage JSONL line {line_number}"
                    )
                    break
                saw_child_usage = True
            events.append(event)
        if parse_error is None and tool_calls:
            pending = ", ".join(sorted(tool_calls))
            parse_error = f"agent ended with unresolved tool calls: {pending}"
        usage: dict[str, int] = {}
        child_usage: dict[str, int] = {}
        child_usage_by_model: dict[str, dict[str, int]] = {}
        for event in events:
            if event.get("type") in ("usage", "child_usage"):
                target = child_usage if event["type"] == "child_usage" else usage
                for name, value in event["usage"].items():
                    if type(value) is int and value >= 0:
                        target[name] = target.get(name, 0) + value
                if event["type"] == "child_usage":
                    child_usage_by_model = event.get("by_model", {})
        total_usage = usage.copy()
        for name, value in child_usage.items():
            total_usage[name] = total_usage.get(name, 0) + value
        if "total_tokens" in total_usage:
            total_usage["total_tokens"] += sum(
                child_usage.get(name, 0)
                for name in (
                    "input_tokens",
                    "cache_read_input_tokens",
                    "cache_creation_input_tokens",
                    "output_tokens",
                )
            )
        tool_calls_by_agent: dict[str, int] = {}
        for event in events:
            if event.get("type") == "tool_call":
                agent = event.get("agent_instance_id", "root")
                tool_calls_by_agent[agent] = tool_calls_by_agent.get(agent, 0) + 1
        grader_root = _immutable_grader(task, setup, Path(grader_temporary))
        failures = []
        for check in task["checks"]:
            failure = _check(
                root,
                setup,
                check,
                events=events,
                command_root=root,
                command_env=grader_environment,
                grader_root=grader_root,
            )
            if failure is not None:
                failures.append(failure)
        if timed_out:
            run_error = "agent timed out"
        elif parse_error is not None:
            run_error = parse_error
        elif process.returncode != 0:
            run_error = f"agent exited {process.returncode}"
        elif not any(event.get("type") == "message" for event in events):
            run_error = "agent produced no final message"
        else:
            run_error = None
        saved_workspace = None
        destination = keep_workspaces or (
            keep_failures if failures or run_error else None
        )
        if destination is not None:
            _assert_retention_safe(root)
            destination.mkdir(parents=True, exist_ok=True)
            saved_workspace = destination / uuid.uuid4().hex
            shutil.copytree(root, saved_workspace)
            try:
                _assert_retention_safe(saved_workspace)
            except RuntimeError:
                shutil.rmtree(saved_workspace)
                raise
        return {
            "task": task["id"],
            "passed": not failures and run_error is None,
            "artifact_passed": not failures,
            "completed": run_error is None,
            "failures": failures,
            "run_error": run_error,
            "seconds": round(time.monotonic() - started, 2),
            "tool_calls": sum(event.get("type") == "tool_call" for event in events),
            "tool_names": [
                event["name"] for event in events if event.get("type") == "tool_call"
            ],
            "tool_calls_by_agent": tool_calls_by_agent,
            "usage": usage,
            "child_usage": child_usage,
            "child_usage_by_model": child_usage_by_model,
            "total_usage": total_usage,
            "error": stderr.strip()[-400:] if failures or run_error else None,
            "saved_workspace": str(saved_workspace) if saved_workspace else None,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=TASKS)
    parser.add_argument("--task", action="append", help="run only this task ID")
    parser.add_argument("--provider", choices=("codex", "claude"), default="codex")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--instruction", help="append an experimental system rule")
    parser.add_argument(
        "--keep-failures", type=Path, help="copy failed workspaces here"
    )
    parser.add_argument("--keep-workspaces", type=Path, help="copy all workspaces here")
    args = parser.parse_args()
    if args.timeout < 1 or args.repeat < 1:
        parser.error("timeout and repeat must be positive")
    tasks = [
        json.loads(line) for line in args.tasks.read_text().split("\n") if line.strip()
    ]
    if args.task:
        tasks = [task for task in tasks if task["id"] in args.task]
    if not tasks:
        parser.error("no matching tasks")
    provider_env = _provider_environment(args.provider)
    results = []
    for task in tasks:
        for _ in range(args.repeat):
            result = run_task(
                task,
                provider=args.provider,
                model=args.model,
                timeout=args.timeout,
                instruction=args.instruction,
                keep_failures=args.keep_failures,
                keep_workspaces=args.keep_workspaces,
                provider_env=provider_env,
            )
            print(json.dumps(result, sort_keys=True), flush=True)
            results.append(result)
    print(
        f"passed {sum(result['passed'] for result in results)}/{len(results)}; "
        f"artifacts {sum(result['artifact_passed'] for result in results)}/{len(results)}; "
        f"completed {sum(result['completed'] for result in results)}/{len(results)}"
    )
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
