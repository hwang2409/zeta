#!/usr/bin/env python3
"""Run and grade one Zeta computer-use trial with temporary configuration."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = Path("/tmp/computer-demo")
DEFAULT_DOCKER_HOST = f"unix://{Path.home()}/.lima/zeta-sandbox/sock/docker.sock"
MODEL_IMAGE_PATCHES = (1024 // 32) * (640 // 32)
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


def run(
    command: list[str], *, env: dict[str, str], timeout: int = 1200
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _events(output: str) -> list[dict[str, object]]:
    events = []
    for line in output.splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if type(value) is dict:
            events.append(value)
    return events


def _total_usage(events: list[dict[str, object]]) -> dict[str, int]:
    total: dict[str, int] = {}
    for event in events:
        if event.get("type") != "usage" or type(event.get("usage")) is not dict:
            continue
        for key, value in event["usage"].items():
            if type(value) is int:
                total[key] = total.get(key, 0) + value
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = args.output / args.run
    output.mkdir(parents=True, exist_ok=True)
    artifact_dir = output / "artifacts"
    metrics = output / "metrics.jsonl"
    today = datetime.now().astimezone().date().isoformat()
    expected = f"Zeta VM demo\n{today}\n"
    docker_host = os.environ.get("ZETA_COMPUTER_DOCKER_HOST", DEFAULT_DOCKER_HOST)
    started = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="zeta-computer-home-") as temporary:
        home = Path(temporary)
        env = os.environ.copy()
        docker_config = home / "docker-cli"
        docker_config.mkdir()
        (docker_config / "config.json").write_text("{}\n")
        auth_source = Path(
            os.environ.get(
                "ZETA_COMPUTER_CODEX_AUTH",
                Path.home() / ".zeta" / "codex-oauth.json",
            )
        )
        if not auth_source.is_file():
            raise SystemExit(f"Codex OAuth source does not exist: {auth_source}")
        shutil.copyfile(auth_source, home / "codex-oauth.json")
        denied = ", ".join(json.dumps(name) for name in HOST_TOOLS)
        (home / "settings.toml").write_text(f"[approval]\ndeny = [{denied}]\n")
        env.update(
            {
                "DOCKER_CONFIG": str(docker_config),
                "DOCKER_HOST": docker_host,
                "ZETA_HOME": str(home),
                "ZETA_COMPUTER_ARTIFACT_DIR": str(artifact_dir),
                "ZETA_COMPUTER_DOCKER_HOST": docker_host,
                "ZETA_COMPUTER_METRICS": str(metrics),
                "ZETA_COMPUTER_RUN_ID": args.run,
            }
        )
        env.pop("DOCKER_CONTEXT", None)
        server_env = {
            "ZETA_COMPUTER_ARTIFACT_DIR": str(artifact_dir),
            "ZETA_COMPUTER_DOCKER_CONFIG": str(docker_config),
            "ZETA_COMPUTER_DOCKER_HOST": docker_host,
            "ZETA_COMPUTER_METRICS": str(metrics),
            "ZETA_COMPUTER_RUN_ID": args.run,
        }
        add_command = ["zeta", "mcp", "add", "--scope", "user"]
        for key, value in server_env.items():
            add_command += ["--env", f"{key}={value}"]
        add_command += ["computer", sys.executable, str(ROOT / "server.py")]
        add = run(add_command, env=env)
        if add.returncode:
            raise SystemExit(f"mcp add failed: {add.stderr}")
        task = (
            "Use the computer tools to open the text editor in the sandbox desktop. "
            'Write a note whose first line is "Zeta VM demo" and second line is '
            f"today's date, {today}. Save it as ~/notes/demo.txt. "
            "Confirm success from the final visible desktop state."
        )
        command = [
            "zeta",
            "--provider",
            "codex",
            "--model",
            "gpt-5.6-luna",
            "--yolo",
            "--max-turns",
            "60",
            "--format",
            "json",
            "--print",
            task,
        ]
        result = run(command, env=env)
        (output / "zeta.jsonl").write_text(result.stdout)
        (output / "zeta.stderr").write_text(result.stderr)

        container_query = [
            "docker",
            "ps",
            "-aq",
            "--filter",
            f"label=zeta.computer-mcp.run={args.run}",
        ]
        containers = run(container_query, env=env).stdout.split()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        for container in containers:
            note = subprocess.run(
                ["docker", "exec", container, "cat", "/home/zeta/notes/demo.txt"],
                env=env,
                capture_output=True,
                timeout=30,
                check=False,
            )
            (artifact_dir / "demo.txt").write_bytes(note.stdout)
            screenshot = subprocess.run(
                [
                    "docker",
                    "exec",
                    "-e",
                    "DISPLAY=:99",
                    container,
                    "import",
                    "-window",
                    "root",
                    "-resize",
                    "1024x640!",
                    "-quality",
                    "75",
                    "jpeg:-",
                ],
                env=env,
                capture_output=True,
                timeout=30,
                check=False,
            )
            if screenshot.returncode == 0:
                (artifact_dir / "final.jpg").write_bytes(screenshot.stdout)
            run(["docker", "stop", "--timeout", "1", container], env=env, timeout=30)
        cleanup = run(container_query, env=env)

    records = (
        [json.loads(line) for line in metrics.read_text().splitlines()]
        if metrics.exists()
        else []
    )
    note_path = artifact_dir / "demo.txt"
    actual = note_path.read_text() if note_path.exists() else ""
    screenshot_records = [item for item in records if "screenshot_bytes" in item]
    events = _events(result.stdout)
    tool_calls = [event for event in events if event.get("type") == "tool_call"]
    usage = _total_usage(events)
    only_sandbox_tools = all(
        str(event.get("name", "")).startswith("computer__") for event in tool_calls
    )
    screenshot_bytes = sum(item["screenshot_bytes"] for item in screenshot_records)
    summary = {
        "run": args.run,
        "command": command,
        "task": task,
        "expected": expected,
        "actual": actual,
        "pass": (
            result.returncode == 0
            and actual.rstrip("\n") == expected.rstrip("\n")
            and only_sandbox_tools
        ),
        "zeta_exit_code": result.returncode,
        "wall_seconds": time.monotonic() - started,
        "steps": sum(1 for event in events if event.get("type") == "usage"),
        "tool_calls": [event.get("name") for event in tool_calls],
        "tool_call_count": len(tool_calls),
        "only_sandbox_tool_calls": only_sandbox_tools,
        "screenshot_count": len(screenshot_records),
        "screenshot_bytes": screenshot_bytes,
        "screenshot_base64_bytes": 4 * ((screenshot_bytes + 2) // 3),
        "screenshot_patch_units": len(screenshot_records) * MODEL_IMAGE_PATCHES,
        "screenshot_token_note": (
            "640 32px patches per 1024x640 image; the private Codex model's "
            "billing multiplier is not exposed"
        ),
        "latency_by_action_seconds": records,
        "tokens": usage,
        "cleanup_success": cleanup.returncode == 0 and not cleanup.stdout.strip(),
        "final_screenshot": str(artifact_dir / "final.jpg"),
        "docker_host": docker_host,
        "temporary_zeta_home": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["pass"] and summary["cleanup_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
