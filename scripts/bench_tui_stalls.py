"""Reproduce interactive TUI stalls in an isolated tmux PTY.

The harness creates a temporary ZETA_HOME, resumes a large synthetic session,
streams scripted fake-provider markdown and tool calls, completes background
children, runs periodic-output shell tasks, and types at a fixed rate. It
prints key-to-frame latency and watchdog stall records as JSON. No real session
content is read or modified.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from zeta.core.session import SessionManager
from zeta.protocol.types import Message, MessageRole, TextContent, ToolUseContent


def _fake_script(path: Path, *, agents: int, tasks: int) -> None:
    parent_steps: list[dict[str, object]] = []
    for index in range(agents):
        parent_steps.append(
            {
                "type": "tool_call",
                "name": "agent",
                "arguments": {
                    "prompt": f"child work {index}",
                    "description": f"load child {index}",
                    "background": True,
                },
            }
        )
    command = "for i in $(seq 1 80); do echo tick; sleep 0.1; done"
    for _ in range(tasks):
        parent_steps.append(
            {
                "type": "tool_call",
                "name": "run_background",
                "arguments": {"command": command},
            }
        )
    markdown = "\n\n".join(
        f"## Section {index}\n\n" + ("text with **markdown** and `code` " * 20)
        for index in range(80)
    )
    script = {
        "version": 1,
        "rules": [
            {
                "match": {"contains": "child work"},
                "responses": [
                    {
                        "steps": [
                            {
                                "type": "text",
                                "text": "child result " * 80,
                                "chunk_size": 40,
                                "delay": 0.01,
                            }
                        ]
                    }
                ],
            },
            {
                "match": {"contains": "start load"},
                "responses": [
                    {"steps": parent_steps},
                    {
                        "steps": [
                            {
                                "type": "text",
                                "text": markdown,
                                "chunk_size": 80,
                                "delay": 0.01,
                            }
                        ]
                    },
                ],
            },
        ],
    }
    path.write_text(json.dumps(script), encoding="utf-8")


def _synthetic_home(home: Path, *, messages: int, children: int) -> str:
    opened = SessionManager(home).create(
        provider="fake", model="offline", cwd=Path.cwd()
    )
    store = opened.store
    body = "A prior transcript paragraph with markdown-shaped text. " * 8
    for index in range(messages):
        role = MessageRole.USER if index % 2 == 0 else MessageRole.ASSISTANT
        store.append_message(Message(role, [TextContent(f"entry {index}: {body}")]))
    agents = store.session_dir / "agents"
    agents.mkdir(exist_ok=True)
    for index in range(children):
        child = agents / f"synthetic-{index:03d}"
        child.mkdir()
        (child / "agent_lifecycle.json").write_text(
            json.dumps(
                {
                    "description": f"synthetic child {index}",
                    "state": "completed",
                    "agent_instance_id": f"synthetic-{index:03d}",
                }
            ),
            encoding="utf-8",
        )
    session_id = store.session_id
    store.close()
    return session_id


def _tmux(socket: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["tmux", "-S", str(socket), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _pane(socket: Path, session: str) -> str:
    return _tmux(socket, "capture-pane", "-p", "-t", session).stdout


def _wait_for(socket: Path, session: str, text: str, timeout: float) -> float:
    started = time.perf_counter()
    deadline = started + timeout
    while time.perf_counter() < deadline:
        if text in _pane(socket, session):
            return time.perf_counter() - started
        time.sleep(0.005)
    raise TimeoutError(f"TUI did not render marker within {timeout}s")


def _submit_when_ready(
    socket: Path, session: str, marker: str, timeout: float
) -> None:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        _tmux(socket, "send-keys", "-t", session, "Enter")
        try:
            _wait_for(
                socket,
                session,
                marker,
                max(0.001, min(0.25, deadline - time.perf_counter())),
            )
        except TimeoutError:
            continue
        return
    raise TimeoutError(f"TUI did not start scripted load within {timeout}s")


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[max(0, int(len(ordered) * fraction + 0.999999) - 1)]


def run(args: argparse.Namespace) -> dict[str, object]:
    if shutil.which("tmux") is None:
        raise RuntimeError("tmux is required")
    with tempfile.TemporaryDirectory(prefix="zeta-stall-harness-") as temporary:
        root = Path(temporary)
        home = root / "home"
        home.mkdir()
        session_id = _synthetic_home(
            home, messages=args.messages, children=args.existing_children
        )
        script = root / "fake-script.json"
        _fake_script(script, agents=args.agents, tasks=args.tasks)
        socket = root / "tmux.sock"
        session = f"zeta-stalls-{uuid.uuid4().hex[:8]}"
        zeta = Path(sys.executable).with_name("zeta")
        environment = os.environ.copy()
        environment.update(
            {
                "ZETA_HOME": str(home),
                "ZETA_FAKE_SCRIPT": str(script),
                "ZETA_STALL_TRACE": "1",
                "ZETA_STALL_THRESHOLD_MS": str(args.stall_threshold_ms),
                "TERM": "xterm-256color",
                "COLORTERM": "truecolor",
            }
        )
        command = (
            f"exec {zeta} --resume {session_id} --provider fake --force-provider "
            "--model offline --yolo"
        )
        launched = time.perf_counter()
        try:
            subprocess.run(
                [
                    "tmux",
                    "-S",
                    str(socket),
                    "new-session",
                    "-d",
                    "-s",
                    session,
                    "-x",
                    str(args.columns),
                    "-y",
                    str(args.rows),
                    command,
                ],
                check=True,
                env=environment,
                cwd=Path.cwd(),
            )
            startup = _wait_for(socket, session, "type a message", args.timeout)
            time.sleep(args.startup_settle)
            _tmux(socket, "send-keys", "-t", session, "start load")
            _submit_when_ready(socket, session, "load child", args.timeout)
            latencies: list[float] = []
            typed = ""
            alphabet = "abcdefghijklmnopqrstuvwxyz"
            for index in range(args.keys):
                character = alphabet[index % len(alphabet)]
                typed += character
                sent = time.perf_counter()
                _tmux(socket, "send-keys", "-t", session, character)
                _wait_for(socket, session, typed[-min(20, len(typed)) :], args.timeout)
                latencies.append(time.perf_counter() - sent)
                target = sent + args.key_interval
                time.sleep(max(0.0, target - time.perf_counter()))
            time.sleep(0.5)
            _tmux(socket, "send-keys", "-t", session, "C-c", "C-d", check=False)
        finally:
            _tmux(socket, "kill-server", check=False)
        reopened = SessionManager(home).open(session_id)
        try:
            tool_names = [
                block.tool_call.name
                for message in reopened.store.messages()
                for block in message.content
                if isinstance(block, ToolUseContent)
            ]
            agent_directory = reopened.store.session_dir / "agents"
            activity = {
                "tool_calls": {
                    name: tool_names.count(name) for name in sorted(set(tool_names))
                },
                "child_directories": sum(path.is_dir() for path in agent_directory.iterdir()),
                "notifications": len(
                    reopened.store.agent_notifications(pending_only=False)
                ),
                "conversation_log_bytes": reopened.store.path.stat().st_size,
            }
        finally:
            reopened.store.close()
        records: list[dict[str, object]] = []
        for log in (home / "logs" / "stalls.jsonl.1", home / "logs" / "stalls.jsonl"):
            if not log.is_file():
                continue
            for line in log.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
        stalls = [
            float(record["duration_ms"])
            for record in records
            if record.get("kind") == "loop_stall"
            and isinstance(record.get("duration_ms"), (int, float))
        ]
        return {
            "messages": args.messages,
            "existing_children": args.existing_children,
            "background_agents": args.agents,
            "background_tasks": args.tasks,
            "observed_activity": activity,
            "keys": args.keys,
            "startup_seconds": round(startup, 3),
            "elapsed_seconds": round(time.perf_counter() - launched, 3),
            "key_to_frame_ms": {
                "p50": round(statistics.median(latencies) * 1000, 3),
                "p95": round(_percentile(latencies, 0.95) * 1000, 3),
                "max": round(max(latencies) * 1000, 3),
            },
            "loop_stalls": len(stalls),
            "stall_ms": {
                "p50": round(statistics.median(stalls), 3) if stalls else 0,
                "p95": round(_percentile(stalls, 0.95), 3) if stalls else 0,
                "max": round(max(stalls), 3) if stalls else 0,
            },
            "stall_records": records,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--messages", type=int, default=2_000)
    parser.add_argument("--existing-children", type=int, default=32)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--tasks", type=int, default=4)
    parser.add_argument("--keys", type=int, default=80)
    parser.add_argument("--key-interval", type=float, default=0.1)
    parser.add_argument("--stall-threshold-ms", type=float, default=20)
    parser.add_argument("--columns", type=int, default=100)
    parser.add_argument("--rows", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--startup-settle", type=float, default=1.0)
    args = parser.parse_args()
    if min(args.messages, args.existing_children, args.agents, args.tasks, args.keys) < 0:
        parser.error("load counts must be non-negative")
    if (
        args.keys == 0
        or args.key_interval <= 0
        or args.stall_threshold_ms <= 0
        or args.startup_settle < 0
    ):
        parser.error("keys, intervals, and stall threshold must be positive")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
