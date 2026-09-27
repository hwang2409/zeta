"""Run isolated, artifact-graded tasks through the real Zeta agent loop."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

TASKS = Path(__file__).with_name("tasks.jsonl")


def _file(root: Path, name: str) -> Path:
    path = Path(name)
    if not name or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe eval path: {name!r}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"eval path escapes workspace: {name!r}")
    return resolved


def _check(root: Path, setup: dict[str, str], check: dict[str, Any]) -> str | None:
    if "command" in check:
        command = check["command"]
        if not isinstance(command, list) or not command or any(
            type(part) is not str for part in command
        ):
            raise ValueError("check command must be a nonempty argv list")
        argv = [sys.executable if command[0] == "python" else command[0], *command[1:]]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(root / "src"), str(root), env.get("PYTHONPATH")) if part
        )
        try:
            result = subprocess.run(
                argv, cwd=root, env=env, capture_output=True, text=True, timeout=30,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return f"command timed out: {command[0]}"
        if result.returncode != check.get("exit_code", 0):
            return f"command exited {result.returncode}: {command[0]}"
        if "stdout" in check and result.stdout != check["stdout"]:
            return f"command stdout differed: {command[0]}"
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
    if "nonempty_lines" in check and [line.strip() for line in content.splitlines() if line.strip()] != check["nonempty_lines"]:
        return f"file lines differed: {name}"
    if check.get("unchanged") and content != setup[name]:
        return f"setup file changed: {name}"
    return None


def run_task(
    task: dict[str, Any], *, provider: str, model: str, timeout: int,
    instruction: str | None = None, keep_failures: Path | None = None,
    keep_workspaces: Path | None = None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="zeta-workflow-eval-") as temporary:
        root = Path(temporary)
        if "git_ref" in task:
            ref = task["git_ref"]
            if type(ref) is not str or re.fullmatch(r"[0-9a-f]{40}", ref) is None:
                raise ValueError("git_ref must be a full lowercase commit SHA")
            subprocess.run(
                ["git", "clone", "--quiet", "--shared", str(Path(__file__).resolve().parents[1]), str(root)],
                check=True, capture_output=True, text=True, timeout=30,
            )
            subprocess.run(
                ["git", "-C", str(root), "checkout", "--quiet", "--detach", ref],
                check=True, capture_output=True, text=True, timeout=30,
            )
        setup = task.get("setup", {})
        for name, content in setup.items():
            path = _file(root, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

        command = [
            str(Path(sys.executable).with_name("zeta")),
            "--no-session", "--provider", provider, "--model", model,
            "--yolo", "--max-turns", str(task.get("max_turns", 12)),
        ]
        if instruction:
            command.extend(("--append-system-prompt", instruction))
        command.extend(("--format", "json", "--print", task["prompt"]))
        started = time.monotonic()
        process = subprocess.Popen(
            command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True,
        )
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()

        events = []
        parse_error = None
        for line_number, line in enumerate(stdout.splitlines(), start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                parse_error = f"agent emitted invalid JSONL line {line_number}"
                break
            if not isinstance(event, dict):
                parse_error = f"agent emitted non-object JSONL line {line_number}"
                break
            if event.get("type") == "tool_call" and (
                type(event.get("name")) is not str or not event["name"]
            ):
                parse_error = f"agent emitted malformed tool_call JSONL line {line_number}"
                break
            if event.get("type") == "usage" and type(event.get("usage")) is not dict:
                parse_error = f"agent emitted malformed usage JSONL line {line_number}"
                break
            events.append(event)
        usage: dict[str, int] = {}
        for event in events:
            if event.get("type") == "usage":
                for name, value in event["usage"].items():
                    if type(value) is int and value >= 0:
                        usage[name] = usage.get(name, 0) + value
        failures = [
            failure
            for check in task["checks"]
            if (failure := _check(root, setup, check)) is not None
        ]
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
        destination = keep_workspaces or (keep_failures if failures or run_error else None)
        if destination is not None:
            destination.mkdir(parents=True, exist_ok=True)
            saved_workspace = destination / uuid.uuid4().hex
            shutil.copytree(root, saved_workspace)
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
            "usage": usage,
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
    parser.add_argument("--keep-failures", type=Path, help="copy failed workspaces here")
    parser.add_argument("--keep-workspaces", type=Path, help="copy all workspaces here")
    args = parser.parse_args()
    if args.timeout < 1 or args.repeat < 1:
        parser.error("timeout and repeat must be positive")
    tasks = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    if args.task:
        tasks = [task for task in tasks if task["id"] in args.task]
    if not tasks:
        parser.error("no matching tasks")
    results = []
    for task in tasks:
        for _ in range(args.repeat):
            result = run_task(
                task, provider=args.provider, model=args.model,
                timeout=args.timeout, instruction=args.instruction,
                keep_failures=args.keep_failures,
                keep_workspaces=args.keep_workspaces,
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
