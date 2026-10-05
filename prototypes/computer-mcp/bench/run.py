#!/usr/bin/env python3
"""Run task/repetition/model matrices for the pixel desktop benchmark."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent
sys.path.insert(0, str(BENCH))

from tasks import ALL_TASKS, TASK_BY_ID, Task

DEFAULT_OUTPUT = Path("/tmp/computer-bench")
DEFAULT_DOCKER_HOST = f"unix://{Path.home()}/.lima/zeta-sandbox/sock/docker.sock"
MODEL_IMAGE_PATCHES = (1024 // 32) * (640 // 32)
# Change this one line to ["--tools", "computer__*"] when the allowlist lands.
HEADLESS_TOOL_ARGS: list[str] = []
HOST_TOOLS = (
    "agent",
    "agent_cancel",
    "agent_output",
    "agent_send",
    "agent_status",
    "automation",
    "bash",
    "edit",
    "fetch",
    "mcp_discover",
    "project",
    "project_update",
    "read",
    "run_background",
    "skill",
    "task_input",
    "task_kill",
    "task_output",
    "todo",
    "websearch",
    "write",
)
_PRINT_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class Model:
    provider: str
    name: str

    @property
    def id(self) -> str:
        return f"{self.provider}-{self.name}".replace("/", "-")


def command(
    args: list[str], *, env: dict[str, str], timeout: int = 1200
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args, env=env, text=True, capture_output=True, timeout=timeout, check=False
    )


def events(output: str) -> list[dict[str, object]]:
    parsed = []
    for line in output.splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if type(value) is dict:
            parsed.append(value)
    return parsed


def total_usage(records: list[dict[str, object]]) -> dict[str, int]:
    total: dict[str, int] = {}
    for event in records:
        usage = event.get("usage")
        if event.get("type") != "usage" or type(usage) is not dict:
            continue
        for key, value in usage.items():
            if type(value) is int:
                total[key] = total.get(key, 0) + value
    return total


def docker_env(home: Path, docker_host: str) -> dict[str, str]:
    config = home / "docker-cli"
    config.mkdir()
    (config / "config.json").write_text("{}\n")
    env = os.environ.copy()
    env.update(
        {
            "DOCKER_CONFIG": str(config),
            "DOCKER_HOST": docker_host,
            "ZETA_COMPUTER_DOCKER_CONFIG": str(config),
            "ZETA_COMPUTER_DOCKER_HOST": docker_host,
        }
    )
    env.pop("DOCKER_CONTEXT", None)
    return env


def cleanup_containers(env: dict[str, str], run_id: str) -> tuple[bool, list[str]]:
    query = ["docker", "ps", "-aq", "--filter", f"label=zeta.computer-mcp.run={run_id}"]
    found = command(query, env=env, timeout=30).stdout.split()
    for container in found:
        command(["docker", "stop", "--timeout", "1", container], env=env, timeout=30)
    remaining = command(query, env=env, timeout=30)
    return remaining.returncode == 0 and not remaining.stdout.strip(), found


def run_trial(
    task: Task,
    repetition: int,
    model: Model,
    suite: Path,
    docker_host: str,
    auth_source: Path,
    features: str,
) -> dict[str, object]:
    run_id = f"{task.id}-{model.id}-r{repetition}"
    output = suite / run_id
    output.mkdir(parents=True, exist_ok=False)
    artifacts = output / "artifacts"
    artifacts.mkdir()
    metrics = output / "metrics.jsonl"
    started = time.monotonic()
    error = ""
    zeta_exit = -1
    stdout = ""
    stderr = ""
    cleanup_success = False

    with tempfile.TemporaryDirectory(prefix=f"zeta-bench-{task.id}-") as temporary:
        home = Path(temporary)
        env = docker_env(home, docker_host)
        shutil.copyfile(auth_source, home / "codex-oauth.json")
        denied = ", ".join(json.dumps(name) for name in HOST_TOOLS)
        (home / "settings.toml").write_text(f"[approval]\ndeny = [{denied}]\n")
        server_env = {
            "ZETA_COMPUTER_ARTIFACT_DIR": str(artifacts),
            "ZETA_COMPUTER_DOCKER_CONFIG": env["DOCKER_CONFIG"],
            "ZETA_COMPUTER_DOCKER_HOST": docker_host,
            "ZETA_COMPUTER_METRICS": str(metrics),
            "ZETA_COMPUTER_RUN_ID": run_id,
            "ZETA_COMPUTER_TASK": task.id,
            "ZETA_COMPUTER_FEATURES": features,
        }
        env.update(server_env)
        env["ZETA_HOME"] = str(home)
        add = ["zeta", "mcp", "add", "--scope", "user"]
        for key, value in server_env.items():
            add += ["--env", f"{key}={value}"]
        add += ["computer", sys.executable, str(BENCH / "serve_bench.py")]
        try:
            added = command(add, env=env, timeout=60)
            if added.returncode:
                raise RuntimeError(f"mcp add failed: {added.stderr.strip()}")
            invoke = [
                "zeta",
                "--provider",
                model.provider,
                "--model",
                model.name,
                "--yolo",
                "--max-turns",
                "80",
                "--format",
                "json",
                "--print",
                *HEADLESS_TOOL_ARGS,
                task.prompt,
            ]
            result = command(invoke, env=env)
            zeta_exit, stdout, stderr = result.returncode, result.stdout, result.stderr
            (output / "zeta.jsonl").write_text(stdout)
            (output / "zeta.stderr").write_text(stderr)
            for _ in range(100):
                if (artifacts / "grade.json").exists():
                    break
                time.sleep(0.1)
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            error = f"{type(exc).__name__}: {exc}"
            (output / "runner-error.txt").write_text(error + "\n")
        finally:
            cleanup_success, cleaned = cleanup_containers(env, run_id)

    parsed = events(stdout)
    calls = [event for event in parsed if event.get("type") == "tool_call"]
    names = [str(event.get("name", "")) for event in calls]
    non_computer = [name for name in names if not name.startswith("computer__")]
    shortcut_calls = []
    for event in calls:
        if event.get("name") != "computer__computer_key":
            continue
        arguments = event.get("arguments")
        keys = str(arguments.get("keys", "")).lower() if type(arguments) is dict else ""
        if any(modifier in keys for modifier in ("ctrl", "alt", "super", "meta")) or keys == "f2":
            shortcut_calls.append(event)
    try:
        grade = json.loads((artifacts / "grade.json").read_text())
    except (OSError, ValueError):
        grade = {"pass": False, "checks": [], "error": "missing or invalid grade"}
    metric_records = []
    if metrics.exists():
        for line in metrics.read_text().splitlines():
            try:
                metric_records.append(json.loads(line))
            except ValueError:
                pass
    screenshots = [item for item in metric_records if "screenshot_bytes" in item]
    screenshot_bytes = sum(int(item["screenshot_bytes"]) for item in screenshots)
    policy_pass = not non_computer and not (task.forbid_shortcuts and shortcut_calls)
    passed = (
        bool(grade.get("pass")) and zeta_exit == 0 and policy_pass and cleanup_success
    )
    summary = {
        "run": run_id,
        "task": task.id,
        "description": task.description,
        "repetition": repetition,
        "provider": model.provider,
        "model": model.name,
        "features": features,
        "pass": passed,
        "guest_grade": grade,
        "policy_pass": policy_pass,
        "zeta_exit_code": zeta_exit,
        "steps": sum(1 for event in parsed if event.get("type") == "usage"),
        "tool_calls": names,
        "tool_call_count": len(calls),
        "non_computer_tool_calls": non_computer,
        "shortcut_violation": bool(task.forbid_shortcuts and shortcut_calls),
        "screenshot_count": len(screenshots),
        "screenshot_bytes": screenshot_bytes,
        "screenshot_patch_units": len(screenshots) * MODEL_IMAGE_PATCHES,
        "tokens": total_usage(parsed),
        "wall_seconds": time.monotonic() - started,
        "cleanup_success": cleanup_success,
        "cleaned_containers": cleaned,
        "error": error,
        "artifacts": str(artifacts),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with _PRINT_LOCK:
        print(
            f"{run_id}: {'PASS' if passed else 'FAIL'} ({summary['wall_seconds']:.1f}s)",
            flush=True,
        )
    return summary


def wilson(successes: int, total: int) -> tuple[float, float]:
    if not total:
        return 0.0, 0.0
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    )
    return center - radius, center + radius


def mean(results: list[dict[str, object]], key: str) -> float:
    return sum(float(item[key]) for item in results) / len(results)


def mean_tokens(results: list[dict[str, object]]) -> float:
    return sum(int(item["tokens"].get("total_tokens", 0)) for item in results) / len(
        results
    )


def markdown(results: list[dict[str, object]]) -> str:
    lines = [
        "# Computer-use benchmark results",
        "",
        "Pass rates use 95% Wilson score intervals.",
        "",
        "| Model | Task | Pass | Rate (95% CI) | Avg steps | Avg tools | Avg screenshots | Avg tokens | Avg wall |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    keys = sorted(
        {
            (str(item["provider"]), str(item["model"]), str(item["task"]))
            for item in results
        }
    )
    for provider, model, task in keys:
        group = [
            item
            for item in results
            if (item["provider"], item["model"], item["task"])
            == (provider, model, task)
        ]
        passed = sum(bool(item["pass"]) for item in group)
        low, high = wilson(passed, len(group))
        lines.append(
            f"| {provider}/{model} | {task} | {passed}/{len(group)} | "
            f"{passed / len(group):.0%} ({low:.0%}–{high:.0%}) | "
            f"{mean(group, 'steps'):.1f} | {mean(group, 'tool_call_count'):.1f} | "
            f"{mean(group, 'screenshot_count'):.1f} | "
            f"{mean_tokens(group):.0f} | {mean(group, 'wall_seconds'):.1f}s |"
        )
    for provider, model in sorted(
        {(str(item["provider"]), str(item["model"])) for item in results}
    ):
        group = [
            item
            for item in results
            if (item["provider"], item["model"]) == (provider, model)
        ]
        passed = sum(bool(item["pass"]) for item in group)
        low, high = wilson(passed, len(group))
        overall = (
            f"**Overall {provider}/{model}: {passed}/{len(group)} "
            f"({passed / len(group):.1%}, 95% CI {low:.1%}–{high:.1%}).**"
        )
        lines += ["", overall]
    return "\n".join(lines) + "\n"


def parse_model(value: str) -> Model:
    provider, separator, name = value.partition(":")
    if not separator or not provider or not name:
        raise argparse.ArgumentTypeError("model must be PROVIDER:NAME")
    return Model(provider, name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", action="append", choices=sorted(TASK_BY_ID))
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--model", action="append", type=parse_model)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--suite", default=time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument(
        "--features",
        default="",
        help="comma-separated ZETA_COMPUTER_FEATURES passed to the MCP server",
    )
    args = parser.parse_args()
    if args.reps < 1 or not 1 <= args.concurrency <= 3:
        parser.error("reps must be positive and concurrency must be 1..3")
    tasks = [TASK_BY_ID[item] for item in args.task] if args.task else list(ALL_TASKS)
    models = args.model or [Model("codex", "gpt-5.6-luna")]
    suite = args.output / args.suite
    suite.mkdir(parents=True, exist_ok=False)
    docker_host = os.environ.get("ZETA_COMPUTER_DOCKER_HOST", DEFAULT_DOCKER_HOST)
    auth_source = Path(
        os.environ.get(
            "ZETA_COMPUTER_CODEX_AUTH", Path.home() / ".zeta/codex-oauth.json"
        )
    )
    if not auth_source.is_file():
        parser.error(f"Codex OAuth source does not exist: {auth_source}")
    jobs = [
        (task, repetition, model)
        for model in models
        for task in tasks
        for repetition in range(1, args.reps + 1)
    ]
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                run_trial,
                task,
                repetition,
                model,
                suite,
                docker_host,
                auth_source,
                args.features,
            )
            for task, repetition, model in jobs
        ]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(
        key=lambda item: (
            str(item["model"]),
            str(item["task"]),
            int(item["repetition"]),
        )
    )
    (suite / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    report = markdown(results)
    (suite / "report.md").write_text(report)
    print(f"\n{report}\nArtifacts: {suite}")
    return 0 if all(bool(item["pass"]) for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
