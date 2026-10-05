"""Run the Zeta context-management benchmark.

Each run gets an isolated fixture, ZETA_HOME, telemetry file, and cache trace.
Hidden tests remain outside the agent workspace until trusted grading starts.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.context.grading import CONTEXT_ROOT, grade_workspace

PRICES_PER_MILLION = {
    "gpt-5.6-luna": {"input": 0.20, "cache_read": 0.02, "output": 1.20},
}
_EVENT_USAGE_KEYS = {
    "input_tokens": "input_tokens",
    "output_tokens": "output_tokens",
    "cache_read_input_tokens": "cache_read_tokens",
    "cache_creation_input_tokens": "cache_write_tokens",
}
_CACHE_USAGE_KEYS = {
    "uncached_input_tokens": "input_tokens",
    "output_tokens": "output_tokens",
    "cache_read_tokens": "cache_read_tokens",
    "cache_write_tokens": "cache_write_tokens",
}


@dataclass(frozen=True)
class RunSpec:
    task_id: str
    token_budget: int
    rep: int

    @property
    def key(self) -> str:
        return f"{self.task_id}|cap={self.token_budget}|{self.rep}"


def _run_key(spec: RunSpec, args: argparse.Namespace) -> str:
    return f"{spec.key}|{args.model}|turns={args.max_turns}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def parse_events(output: str) -> tuple[list[dict[str, Any]], list[str]]:
    events, errors = [], []
    for number, line in enumerate(output.splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"invalid JSONL line {number}")
            continue
        if isinstance(value, dict):
            events.append(value)
        else:
            errors.append(f"non-object JSONL line {number}")
    return events, errors


def _usage(
    events: list[dict[str, Any]], requests: list[dict[str, Any]]
) -> dict[str, int]:
    totals = {value: 0 for value in _EVENT_USAGE_KEYS.values()}
    # Cache traces are request-level and include all root and child requests.
    for row in requests:
        for source, target in _CACHE_USAGE_KEYS.items():
            value = row.get(source)
            if type(value) is int:
                totals[target] += value
    if requests:
        return totals
    # Fake/offline runners may only produce the public usage event.
    for event in events:
        if event.get("type") not in {"usage", "child_usage"}:
            continue
        usage = event.get("usage", {})
        if not isinstance(usage, dict):
            continue
        for source, target in _EVENT_USAGE_KEYS.items():
            value = usage.get(source)
            if type(value) is int:
                totals[target] += value
    return totals


def _telemetry(rows: list[dict[str, Any]]) -> tuple[int, float, int]:
    compactions = recall_calls = 0
    compaction_seconds = 0.0
    for row in rows:
        label = " ".join(
            str(row.get(key, "")).lower() for key in ("type", "event", "name", "kind")
        )
        if any(word in label for word in ("compact", "evict")) and not any(
            word in label for word in ("start", "begin")
        ):
            compactions += 1
            for key in ("duration_seconds", "duration", "elapsed_seconds"):
                value = row.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    compaction_seconds += float(value)
                    break
        if "recall" in label and not any(word in label for word in ("result", "end")):
            recall_calls += 1
    return compactions, compaction_seconds, recall_calls


def summarize_run(
    events: list[dict[str, Any]],
    cache_rows: list[dict[str, Any]],
    telemetry_rows: list[dict[str, Any]],
    model: str,
) -> dict[str, Any]:
    usage = _usage(events, cache_rows)
    compactions, duration, recalls = _telemetry(telemetry_rows)
    if compactions == 0:
        # Baseline checkouts do not emit strategy telemetry. Cache traces still
        # identify the request made immediately after each compaction.
        compactions = sum(row.get("compacted") is True for row in cache_rows)
    tools = Counter(
        str(event.get("name"))
        for event in events
        if event.get("type") == "tool_call" and event.get("name")
    )
    recalls = max(recalls, tools["recall_history"])
    prices = PRICES_PER_MILLION.get(model)
    cost = None
    if prices:
        cost = (
            usage["input_tokens"] * prices["input"]
            + usage["cache_read_tokens"] * prices["cache_read"]
            + usage["output_tokens"] * prices["output"]
        ) / 1_000_000
    return {
        "usage": usage,
        "request_usage": cache_rows,
        "tool_calls": sum(tools.values()),
        # One cache-trace row per agent-loop model request; usage events are a
        # cross-check when cache tracing is unavailable.
        "model_requests": len(cache_rows),
        "usage_events": sum(event.get("type") == "usage" for event in events),
        "tool_calls_by_name": dict(sorted(tools.items())),
        "compactions": compactions,
        "compaction_seconds": duration,
        "recall_calls": recalls,
        "estimated_cost_usd": cost,
    }


def _command(
    checkout: Path,
    model: str,
    budget: int,
    max_turns: int,
    prompt: str,
    resume: str | None = None,
) -> list[str]:
    command = [
        "uv",
        "run",
        "--project",
        str(checkout),
        "zeta",
        "--provider",
        "codex",
        "--model",
        model,
        "--yolo",
        "--token-budget",
        str(budget),
        "--max-turns",
        str(max_turns),
        "--format",
        "json",
    ]
    if resume:
        command.extend(["--resume", resume])
    command.extend(["--print", prompt])
    return command


def _invoke(
    command: list[str], workspace: Path, env: dict[str, str], timeout: int
) -> tuple[subprocess.CompletedProcess[str], float]:
    started = time.monotonic()
    try:
        process = subprocess.run(
            command,
            cwd=workspace,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        process = subprocess.CompletedProcess(
            command, 124, exc.stdout or "", (exc.stderr or "") + "\nbenchmark timeout"
        )
    return process, time.monotonic() - started


def _resume_id(stderr: str) -> str | None:
    matches = re.findall(r"resume with: zeta --resume ([^\s]+)", stderr)
    return matches[-1] if matches else None


def _stage_codex_auth(home: Path) -> Path:
    """Stage ambient Codex auth in the isolated run home, never the workspace."""
    codex_home = home / "provider" / "codex"
    codex_home.mkdir(parents=True, mode=0o700, exist_ok=True)
    source = Path.home() / ".codex" / "auth.json"
    if source.is_file():
        destination = codex_home / "auth.json"
        shutil.copyfile(source, destination)
        destination.chmod(0o600)
    return codex_home


def _initialize_workspace(workspace: Path) -> None:
    """Create the baseline commit that evaluated agents use for project discovery."""
    commands = [
        ["git", "init", "--quiet"],
        ["git", "add", "--all"],
        [
            "git",
            "-c",
            "user.name=Zeta Context Benchmark",
            "-c",
            "user.email=context-benchmark@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "Initial fixture",
        ],
    ]
    for command in commands:
        subprocess.run(command, cwd=workspace, check=True, capture_output=True, text=True)


def _environment(home: Path, telemetry: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["ZETA_HOME"] = str(home)
    env["CODEX_HOME"] = str(_stage_codex_auth(home))
    env["ZETA_CACHE_TRACE"] = "1"
    env["ZETA_CONTEXT_TELEMETRY"] = str(telemetry)
    env.pop("ZETA_ANTHROPIC_OAUTH_COMPAT", None)
    return env


def run_repo_task(
    task: dict[str, Any], spec: RunSpec, args: argparse.Namespace, root: Path
) -> dict[str, Any]:
    workspace = root / "workspace"
    home = root / "home"
    telemetry = root / "telemetry.jsonl"
    shutil.copytree(CONTEXT_ROOT / task["fixture"], workspace)
    _initialize_workspace(workspace)
    home.mkdir(mode=0o700)
    process, wall = _invoke(
        _command(
            args.zeta_checkout,
            args.model,
            spec.token_budget,
            args.max_turns,
            task["prompt"],
        ),
        workspace,
        _environment(home, telemetry),
        args.timeout,
    )
    events, parse_errors = parse_events(process.stdout)
    grading_attempts: list[dict[str, Any]] = []
    passed, grade_error = grade_workspace(
        workspace,
        CONTEXT_ROOT / task["grader"],
        task["expected_passes"],
        grading_attempts,
    )
    errors = list(parse_errors)
    if process.returncode:
        errors.append(f"zeta exited {process.returncode}")
    if grade_error:
        errors.append(grade_error)
    metrics = summarize_run(
        events,
        read_jsonl(home / "logs/cache-trace.jsonl"),
        read_jsonl(telemetry),
        args.model,
    )
    return {
        "key": _run_key(spec, args),
        "task": spec.task_id,
        "cap": spec.token_budget,
        "rep": spec.rep,
        "passed": passed and process.returncode == 0 and not parse_errors,
        "grader_passed": passed,
        "wall_seconds": wall,
        "model": args.model,
        "token_budget": spec.token_budget,
        "max_turns": args.max_turns,
        "errors": errors,
        "stderr": process.stderr[-4000:],
        "grader_attempts": grading_attempts,
        "grader_infra_timeout": any(
            attempt["timed_out"] for attempt in grading_attempts
        ),
        **metrics,
    }


def run_session(
    turns: list[dict[str, Any]], spec: RunSpec, args: argparse.Namespace, root: Path
) -> dict[str, Any]:
    workspace, home = root / "workspace", root / "home"
    telemetry = root / "telemetry.jsonl"
    shutil.copytree(CONTEXT_ROOT / "session/fixture", workspace)
    _initialize_workspace(workspace)
    home.mkdir(mode=0o700)
    all_events: list[dict[str, Any]] = []
    errors: list[str] = []
    grades: list[bool] = []
    grading_attempts: list[dict[str, Any]] = []
    resume = None
    wall = 0.0
    for index, turn in enumerate(turns, 1):
        process, elapsed = _invoke(
            _command(
                args.zeta_checkout,
                args.model,
                spec.token_budget,
                args.max_turns,
                turn["prompt"],
                resume,
            ),
            workspace,
            _environment(home, telemetry),
            args.timeout,
        )
        wall += elapsed
        events, parse_errors = parse_events(process.stdout)
        all_events.extend(events)
        errors.extend(f"turn {index}: {error}" for error in parse_errors)
        if process.returncode:
            errors.append(f"turn {index}: zeta exited {process.returncode}")
        if index == 1:
            resume = _resume_id(process.stderr)
            if not resume:
                # Headless --print does not always print a resume hint; a fresh
                # run home holds exactly the one persisted root session.
                sessions = sorted(p.name for p in (home / "sessions").glob("*") if p.is_dir())
                resume = sessions[0] if len(sessions) == 1 else None
            if not resume:
                errors.append("turn 1: missing persisted session id")
        turn_attempts: list[dict[str, Any]] = []
        passed, failure = grade_workspace(
            workspace,
            CONTEXT_ROOT / f"session/graders/turn{index}",
            turn["expected_passes"],
            turn_attempts,
        )
        grading_attempts.extend({"turn": index, **attempt} for attempt in turn_attempts)
        grades.append(passed)
        if failure:
            errors.append(f"turn {index}: {failure}")
        if process.returncode or parse_errors or (index == 1 and not resume):
            break
    metrics = summarize_run(
        all_events,
        read_jsonl(home / "logs/cache-trace.jsonl"),
        read_jsonl(telemetry),
        args.model,
    )
    return {
        "key": _run_key(spec, args),
        "task": "session",
        "cap": spec.token_budget,
        "rep": spec.rep,
        "passed": len(grades) == len(turns) and all(grades) and not errors,
        "grader_passed": len(grades) == len(turns) and all(grades),
        "turn_grades": grades,
        "wall_seconds": wall,
        "model": args.model,
        "token_budget": spec.token_budget,
        "max_turns": args.max_turns,
        "errors": errors,
        "grader_attempts": grading_attempts,
        "grader_infra_timeout": any(
            attempt["timed_out"] for attempt in grading_attempts
        ),
        **metrics,
    }


def _worker(
    spec: RunSpec, args: argparse.Namespace, tasks: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix=f"zeta-context-{spec.task_id}-"
    ) as temporary:
        if spec.task_id == "session":
            turns = json.loads((CONTEXT_ROOT / "session/turns.json").read_text())
            result = run_session(turns, spec, args, Path(temporary))
        else:
            result = run_repo_task(tasks[spec.task_id], spec, args, Path(temporary))
        # Network/provider outages are infrastructure failures, not results.
        stderr = result.get("stderr") or ""
        result["infra_error"] = any(
            marker in stderr for marker in ("http_error", "server_is_overloaded")
        )
        result["hit_max_turns"] = "maximum turns reached" in (result.get("stderr") or "")
        result["hit_wall_timeout"] = "benchmark timeout" in (result.get("stderr") or "")
        if not result.get("passed") and args.keep_failed is not None:
            name = f"{spec.task_id}-cap-{spec.token_budget}-{spec.rep}"
            target = args.keep_failed / name.replace("/", "_")
            shutil.rmtree(target, ignore_errors=True)
            shutil.copytree(
                temporary,
                target,
                symlinks=True,
                ignore_dangling_symlinks=True,
                # Never retain staged credentials with kept diagnostics.
                ignore=shutil.ignore_patterns("*oauth*", "auth.json", "provider"),
            )
            result["kept_workspace"] = str(target)
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zeta-checkout", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=CONTEXT_ROOT / "results.jsonl")
    parser.add_argument("--tasks", help="comma-separated IDs; session is also valid")
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument(
        "--token-budgets",
        default="100000,200000,400000,1050000",
        help="comma-separated eviction caps; 1050000 is the luna full window",
    )
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--keep-failed", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.zeta_checkout = args.zeta_checkout.resolve()
    tasks_list = json.loads((CONTEXT_ROOT / "tasks.json").read_text())
    tasks = {task["id"]: task for task in tasks_list}
    selected = args.tasks.split(",") if args.tasks else [*tasks, "session"]
    unknown = set(selected) - {*tasks, "session"}
    if unknown:
        raise SystemExit(f"unknown tasks: {', '.join(sorted(unknown))}")
    if args.reps < 1 or args.concurrency < 1:
        raise SystemExit("reps and concurrency must be positive")
    budgets = tuple(dict.fromkeys(int(value) for value in args.token_budgets.split(",")))
    if not budgets or any(value < 1 for value in budgets):
        raise SystemExit("token budgets must be positive")
    completed = {
        row.get("key")
        for row in read_jsonl(args.results)
        if row.get("model") == args.model
        and row.get("token_budget") in budgets
        and row.get("max_turns") == args.max_turns
    }
    specs = [
        RunSpec(task, budget, rep)
        for budget in budgets
        for task in selected
        for rep in range(1, args.reps + 1)
        if _run_key(RunSpec(task, budget, rep), args) not in completed
    ]
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.results.open("a") as output,
        concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool,
    ):
        futures = {pool.submit(_worker, spec, args, tasks): spec for spec in specs}
        for future in concurrent.futures.as_completed(futures):
            spec = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - persist worker failures for resume
                result = {
                    "key": _run_key(spec, args),
                    "task": spec.task_id,
                    "cap": spec.token_budget,
                    "rep": spec.rep,
                    "passed": False,
                    "errors": [f"harness: {type(exc).__name__}: {exc}"],
                }
            output.write(json.dumps(result, sort_keys=True) + "\n")
            output.flush()
            print(f"{spec.key}: {'PASS' if result['passed'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
