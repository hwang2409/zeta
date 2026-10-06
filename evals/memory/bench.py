"""Run Zeta's cross-session persistent-memory benchmark.

Every cell uses an isolated fixture repository and private ZETA_HOME. A chain's
phases are separate `zeta -p` sessions and never use --resume. The only
cross-session agent state is the project registry and its memory files.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.memory.grading import MEMORY_ROOT, grade_workspace
from zeta.project_registry import ProjectRegistry

STRATEGIES = ("S0", "S1", "oracle-snippet", "oracle-history")
PRICES_PER_MILLION = {
    "gpt-5.6-luna": {"input": 0.20, "cache_read": 0.02, "output": 1.20},
}
_NETWORK_MARKERS = (
    "http_error",
    "server_is_overloaded",
    "request failed",
    "rate limit",
    "rate_limit",
    "too many requests",
    "connection reset",
    "connection error",
)
_USAGE_KEYS = {
    "uncached_input_tokens": "input_tokens",
    "cache_read_tokens": "cache_read_tokens",
    "cache_write_tokens": "cache_write_tokens",
    "output_tokens": "output_tokens",
}


@dataclass(frozen=True)
class RunSpec:
    task: str
    strategy: str
    rep: int
    model: str
    budget: int
    revision: str

    @property
    def key(self) -> str:
        return "|".join(
            (
                self.task,
                self.strategy,
                str(self.rep),
                self.model,
                str(self.budget),
                self.revision,
            )
        )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read valid object rows; tolerate an interrupted final append."""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def completed_keys(rows: list[dict[str, Any]]) -> set[str]:
    """Return keys with a terminal result; infrastructure failures are resumable."""
    return {
        str(row["key"])
        for row in rows
        if isinstance(row.get("key"), str) and not row.get("infra_error", False)
    }


def parse_events(output: str) -> tuple[list[dict[str, Any]], list[str]]:
    events: list[dict[str, Any]] = []
    errors: list[str] = []
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


def summarize_telemetry(
    events: list[dict[str, Any]], cache_rows: list[dict[str, Any]], model: str
) -> dict[str, Any]:
    """Summarize content-free public events and request cache traces."""
    usage = {target: 0 for target in _USAGE_KEYS.values()}
    for row in cache_rows:
        for source, target in _USAGE_KEYS.items():
            value = row.get(source)
            if type(value) is int:
                usage[target] += value
    if not cache_rows:
        event_keys = {
            "input_tokens": "input_tokens",
            "cache_read_input_tokens": "cache_read_tokens",
            "cache_creation_input_tokens": "cache_write_tokens",
            "output_tokens": "output_tokens",
        }
        for event in events:
            if event.get("type") not in {"usage", "child_usage"}:
                continue
            values = event.get("usage")
            if not isinstance(values, dict):
                continue
            for source, target in event_keys.items():
                value = values.get(source)
                if type(value) is int:
                    usage[target] += value
    calls = Counter(
        str(event.get("name"))
        for event in events
        if event.get("type") == "tool_call" and event.get("name")
    )
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
        "model_requests": len(cache_rows),
        "tool_calls": sum(calls.values()),
        "tool_calls_by_name": dict(sorted(calls.items())),
        "memory_searches": calls["project"],
        "estimated_cost_usd": cost,
    }


def _initialize_workspace(workspace: Path) -> None:
    for command in (
        ["git", "init", "--quiet"],
        ["git", "add", "--all"],
        [
            "git",
            "-c",
            "user.name=Zeta Memory Benchmark",
            "-c",
            "user.email=memory-benchmark@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "Initial fixture",
        ],
    ):
        subprocess.run(
            command, cwd=workspace, check=True, capture_output=True, text=True
        )


def _stage_codex_auth(home: Path) -> Path:
    """Copy ambient Codex auth privately into the temporary home."""
    codex_home = home / "provider" / "codex"
    codex_home.mkdir(parents=True, mode=0o700)
    source = Path.home() / ".codex" / "auth.json"
    if source.is_file():
        destination = codex_home / "auth.json"
        shutil.copyfile(source, destination)
        destination.chmod(0o600)
    return codex_home


def _environment(home: Path, codex_home: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["ZETA_HOME"] = str(home)
    env["CODEX_HOME"] = str(codex_home)
    env["ZETA_CACHE_TRACE"] = "1"
    env.pop("ZETA_ANTHROPIC_OAUTH_COMPAT", None)
    return env


def _command(args: argparse.Namespace, prompt: str) -> list[str]:
    # The allowlist is the security control. --yolo only auto-approves `read` and
    # `write`, the sole advertised tools, so headless phases cannot block on input.
    return [
        "uv",
        "run",
        "--project",
        str(args.zeta_checkout),
        "zeta",
        "--provider",
        "codex",
        "--model",
        args.model,
        "--tools",
        "read,write",
        "--require-tools",
        "--yolo",
        "--token-budget",
        str(args.budget),
        "--max-turns",
        str(args.max_turns),
        "--format",
        "json",
        "-p",
        prompt,
    ]


def _invoke(
    command: list[str], workspace: Path, env: dict[str, str], timeout: int
) -> tuple[subprocess.CompletedProcess[str], float]:
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            cwd=workspace,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        result = subprocess.CompletedProcess(
            command, 124, exc.stdout or "", (exc.stderr or "") + "\nbenchmark timeout"
        )
    return result, time.monotonic() - started


def _assistant_text(events: list[dict[str, Any]]) -> str:
    messages = [
        str(event.get("text", ""))
        for event in events
        if event.get("type") == "message" and event.get("role") == "assistant"
    ]
    return messages[-1] if messages else ""


def _memory_bytes(registry: ProjectRegistry, project_id: str) -> int:
    return sum(len(content.encode()) for _, content in registry.load_memory(project_id))


def _memory_records(registry: ProjectRegistry, project_id: str) -> int:
    return sum(
        bool(
            [line for line in content.splitlines() if line and not line.startswith("#")]
        )
        for _, content in registry.load_memory(project_id)
    )


def _seed_project(
    home: Path, workspace: Path, task: dict[str, Any], strategy: str
) -> tuple[ProjectRegistry, str]:
    registry = ProjectRegistry(home / "projects")
    project = registry.find_or_create_for_directory(
        workspace, name=f"memory-bench-{task['id']}"
    )
    if strategy == "S1":
        registry.update_memory(
            project.project_id, {task["memory_file"]: task["memory"]}
        )
    return registry, project.project_id


def _final_prompt(
    task: dict[str, Any], strategy: str, history: list[tuple[str, str]]
) -> tuple[str, int]:
    supplement = ""
    if strategy == "oracle-snippet":
        supplement = "\n\nTrusted source snippet from prior work:\n" + task["memory"]
    elif strategy == "oracle-history":
        rendered = []
        for prompt, response in history:
            rendered.append(f"Earlier user: {prompt}\nEarlier assistant: {response}")
        supplement = "\n\nFull prior session history:\n" + "\n\n".join(rendered)
    prompt = (
        task["final"]
        + supplement
        + "\n\nRead README.md for the output contract. Write answer.json and do not change any other file."
    )
    return prompt, len(supplement.encode())


def _answer_metrics(workspace: Path, task: dict[str, Any]) -> dict[str, bool]:
    try:
        answer = json.loads((workspace / "answer.json").read_text())
    except (OSError, json.JSONDecodeError):
        answer = {}
    encoded = json.dumps(answer, sort_keys=True)
    wrong = any(value in encoded for value in task["wrong"])
    if task["abstention"] and answer.get("action") != "abstain":
        wrong = True
    stale = bool(task["stale"] and wrong)
    abstained = answer.get("action") == "abstain" and answer.get("value") is None
    return {
        "wrong_memory": wrong,
        "stale_fact_selected": stale,
        "abstained": abstained,
        "correct_abstention": bool(task["abstention"] and abstained),
    }


def _network_failure(stderr: str, errors: list[str]) -> bool:
    text = " ".join((stderr, *errors)).lower()
    return any(marker in text for marker in _NETWORK_MARKERS)


def _run_attempt(
    task: dict[str, Any], spec: RunSpec, args: argparse.Namespace, root: Path
) -> dict[str, Any]:
    workspace, home = root / "workspace", root / "home"
    shutil.copytree(MEMORY_ROOT / "fixtures" / task["id"], workspace)
    _initialize_workspace(workspace)
    home.mkdir(mode=0o700)
    codex_home = _stage_codex_auth(home)
    env = _environment(home, codex_home)
    registry, project_id = _seed_project(home, workspace, task, spec.strategy)
    initial_memory_bytes = _memory_bytes(registry, project_id)
    memory_records = _memory_records(registry, project_id)
    initial_memory_records = _memory_records(registry, project_id)
    events: list[dict[str, Any]] = []
    history: list[tuple[str, str]] = []
    grades: list[dict[str, Any]] = []
    errors: list[str] = []
    stderr_parts: list[str] = []
    wall_seconds = 0.0

    phases = [
        (f"phase{index}", prompt) for index, prompt in enumerate(task["turns"], 1)
    ]
    for phase, prompt in phases:
        process, wall = _invoke(_command(args, prompt), workspace, env, args.timeout)
        wall_seconds += wall
        phase_events, parse_errors = parse_events(process.stdout)
        events.extend(phase_events)
        history.append((prompt, _assistant_text(phase_events)))
        stderr_parts.append(process.stderr[-2000:])
        errors.extend(f"{phase}: {error}" for error in parse_errors)
        if process.returncode:
            errors.append(f"{phase}: zeta exited {process.returncode}")
        grade = grade_workspace(workspace, MEMORY_ROOT / "graders" / task["id"] / phase)
        grades.append({"phase": phase, **grade.__dict__})
        if grade.error:
            errors.append(f"{phase}: {grade.error}")
        if process.returncode or parse_errors or not grade.passed:
            break

    if len(grades) == len(phases) and all(grade["passed"] for grade in grades):
        prompt, supplement_bytes = _final_prompt(task, spec.strategy, history)
        process, wall = _invoke(_command(args, prompt), workspace, env, args.timeout)
        wall_seconds += wall
        phase_events, parse_errors = parse_events(process.stdout)
        events.extend(phase_events)
        stderr_parts.append(process.stderr[-2000:])
        errors.extend(f"final: {error}" for error in parse_errors)
        if process.returncode:
            errors.append(f"final: zeta exited {process.returncode}")
        final_grade = grade_workspace(
            workspace, MEMORY_ROOT / "graders" / task["id"] / "final"
        )
        grades.append({"phase": "final", **final_grade.__dict__})
        if final_grade.error:
            errors.append(f"final: {final_grade.error}")
    else:
        supplement_bytes = 0
        final_grade = None

    cache_rows = read_jsonl(home / "logs" / "cache-trace.jsonl")
    metrics = summarize_telemetry(events, cache_rows, spec.model)
    final_passed = bool(final_grade and final_grade.passed)
    memory_bytes = _memory_bytes(registry, project_id)
    memory_records = _memory_records(registry, project_id)
    session_count = len(grades)
    result = {
        "key": spec.key,
        "task": spec.task,
        "family": task["family"],
        "strategy": spec.strategy,
        "rep": spec.rep,
        "model": spec.model,
        "budget": spec.budget,
        "revision": spec.revision,
        "passed": final_passed and not errors,
        "partial_passes": final_grade.passed_tests if final_grade else 0,
        "partial_total": final_grade.total_tests
        if final_grade
        else task["expected_passes"],
        "phase_grades": grades,
        "errors": errors,
        "stderr": "\n".join(stderr_parts)[-4000:],
        "wall_seconds": wall_seconds,
        "memory_bytes_before": initial_memory_bytes,
        "memory_bytes_after": memory_bytes,
        "memory_growth_bytes": memory_bytes - initial_memory_bytes,
        "memory_records_before": initial_memory_records,
        "memory_records_after": memory_records,
        "memory_growth_records": memory_records - initial_memory_records,
        "retrieved_bytes": (
            initial_memory_bytes * session_count
            if spec.strategy == "S1"
            else supplement_bytes
        ),
        "retrieved_tokens_estimate": (
            (
                initial_memory_bytes * session_count
                if spec.strategy == "S1"
                else supplement_bytes
            )
            + 3
        )
        // 4,
        # Reserved S2/S3 write-path telemetry. Baselines have no proposals.
        "proposals": 0,
        "approval_dialogs": 0,
        "memory_files_changed": 0,
        "accepted_items": 0,
        "rejected_items": 0,
        "proposed_characters": 0,
        "source_provenance_accurate": None,
        "extraction_precision": None,
        "extraction_recall": None,
        **_answer_metrics(workspace, task),
        **metrics,
    }
    result["cache_read_tokens_before_memory_update"] = metrics["usage"][
        "cache_read_tokens"
    ]
    result["cache_read_tokens_after_memory_update"] = None
    result["infra_error"] = _network_failure(result["stderr"], errors)
    result["hit_wall_timeout"] = "benchmark timeout" in result["stderr"]
    return result


def _combine_attempts(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    result = dict(attempts[-1])
    usage = {key: 0 for key in _USAGE_KEYS.values()}
    for attempt in attempts:
        for key in usage:
            usage[key] += int(attempt.get("usage", {}).get(key, 0))
    result["usage"] = usage
    result["wall_seconds"] = sum(
        float(item.get("wall_seconds", 0)) for item in attempts
    )
    result["model_requests"] = sum(
        int(item.get("model_requests", 0)) for item in attempts
    )
    result["tool_calls"] = sum(int(item.get("tool_calls", 0)) for item in attempts)
    result["network_drops"] = sum(bool(item.get("infra_error")) for item in attempts)
    result["attempt_count"] = len(attempts)
    prices = PRICES_PER_MILLION.get(str(result.get("model")))
    if prices:
        result["estimated_cost_usd"] = (
            usage["input_tokens"] * prices["input"]
            + usage["cache_read_tokens"] * prices["cache_read"]
            + usage["output_tokens"] * prices["output"]
        ) / 1_000_000
    result["attempts"] = [
        {
            "infra_error": item.get("infra_error"),
            "passed": item.get("passed"),
            "errors": item.get("errors"),
        }
        for item in attempts
    ]
    return result


def _worker(
    task: dict[str, Any], spec: RunSpec, args: argparse.Namespace
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    for attempt_number in (1, 2):
        with tempfile.TemporaryDirectory(
            prefix=f"zeta-memory-{spec.task}-{spec.strategy}-{attempt_number}-"
        ) as temporary:
            result = _run_attempt(task, spec, args, Path(temporary))
            attempts.append(result)
            if not result["infra_error"]:
                break
            if args.keep_failed:
                target = (
                    args.keep_failed
                    / f"{spec.task}-{spec.strategy}-{spec.rep}-attempt-{attempt_number}"
                )
                shutil.rmtree(target, ignore_errors=True)
                shutil.copytree(
                    temporary,
                    target,
                    ignore=shutil.ignore_patterns("auth.json", "provider", "*oauth*"),
                )
    return _combine_attempts(attempts)


def estimate_input_tokens(cells: int) -> int:
    """Conservative section 6.6 estimate: 60k uncached + 120k cached per cell."""
    return cells * 180_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zeta-checkout", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=MEMORY_ROOT / "results.jsonl")
    parser.add_argument("--tasks", help="comma-separated task IDs")
    parser.add_argument("--strategies", default=",".join(STRATEGIES))
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--budget", type=int, default=100_000)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-projected-input", type=int, default=40_000_000)
    parser.add_argument("--keep-failed", type=Path)
    parser.add_argument("--estimate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.zeta_checkout = args.zeta_checkout.resolve()
    tasks_list = json.loads((MEMORY_ROOT / "tasks.json").read_text())
    tasks = {task["id"]: task for task in tasks_list}
    selected_tasks = args.tasks.split(",") if args.tasks else list(tasks)
    selected_strategies = args.strategies.split(",")
    unknown_tasks = set(selected_tasks) - set(tasks)
    unknown_strategies = set(selected_strategies) - set(STRATEGIES)
    if unknown_tasks or unknown_strategies:
        raise SystemExit(
            f"unknown tasks={sorted(unknown_tasks)} strategies={sorted(unknown_strategies)}"
        )
    if args.reps < 1 or args.concurrency < 1 or args.budget < 1:
        raise SystemExit("reps, concurrency, and budget must be positive")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=args.zeta_checkout,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    specs = [
        RunSpec(task, strategy, rep, args.model, args.budget, revision)
        for task in selected_tasks
        for strategy in selected_strategies
        for rep in range(1, args.reps + 1)
    ]
    projected = estimate_input_tokens(len(specs))
    print(
        f"matrix: {len(specs)} cells; projected input tokens: {projected:,} "
        f"(guard: {args.max_projected_input:,})"
    )
    if projected > args.max_projected_input:
        raise SystemExit(
            "projected input token budget exceeds guard; cut cells explicitly"
        )
    if args.estimate_only:
        return 0
    done = completed_keys(read_jsonl(args.results))
    pending = [spec for spec in specs if spec.key not in done]
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.results.open("a") as output,
        concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool,
    ):
        futures = {
            pool.submit(_worker, tasks[spec.task], spec, args): spec for spec in pending
        }
        for future in concurrent.futures.as_completed(futures):
            spec = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - persist cells for diagnosis/resume
                result = {
                    **spec.__dict__,
                    "key": spec.key,
                    "passed": False,
                    "infra_error": False,
                    "errors": [f"harness: {type(exc).__name__}: {exc}"],
                }
            output.write(json.dumps(result, sort_keys=True) + "\n")
            output.flush()
            print(
                f"{spec.task}|{spec.strategy}|rep={spec.rep}: "
                f"{'PASS' if result['passed'] else 'FAIL'}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
