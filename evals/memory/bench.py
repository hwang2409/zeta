"""Run Zeta's cross-session persistent-memory benchmark.

Every cell uses an isolated fixture repository and private ZETA_HOME. A chain's
phases are separate `zeta -p` sessions and never use --resume. The only
cross-session agent state is the project registry and its memory files.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.memory.automatic import AutomaticReconciler
from evals.memory.grading import MEMORY_ROOT, grade_workspace
from evals.memory.reconciler import (
    ReconciliationError,
    apply_proposal,
    reconcile_session,
)
from zeta.memory.entry_reconciler import reconcile_entry_range
from zeta.memory.entry_store import MemoryEntry
from zeta.memory.entry_views import render_all_kinds
from zeta.memory.profiles import memory_profile
from zeta.memory.reconciler import Transcript, project_transcript_row, read_transcript
from zeta.project_registry import MAX_RECORD_SIZE, ProjectRegistry

STRATEGIES = (
    "S0",
    "S1",
    "S2",
    "S2-auto",
    "S2-entry",
    "oracle-snippet",
    "oracle-history",
)
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


def _reconciler_command(args: argparse.Namespace, prompt: str) -> list[str]:
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
        "read",
        "--require-tools",
        "--yolo",
        "--token-budget",
        str(args.reconciler_budget),
        "--max-turns",
        "2",
        "--format",
        "json",
        "-p",
        prompt,
    ]


def _command(
    args: argparse.Namespace, prompt: str, *, budget: int | None = None
) -> list[str]:
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
        str(args.budget if budget is None else budget),
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


def _invoke_until_trigger(
    command: list[str],
    workspace: Path,
    env: dict[str, str],
    timeout: int,
    *,
    transcript: Callable[[], tuple[Path, str] | None],
    token_threshold: int,
    initial_tokens: int,
    on_trigger: Callable[[Path, str], None],
    crash: bool,
) -> tuple[subprocess.CompletedProcess[str], float, bool]:
    """Reconcile one live session at the threshold, then continue or kill it."""
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=workspace,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    triggered = False
    deadline = started + timeout
    while process.poll() is None and time.monotonic() < deadline:
        current = transcript()
        if current is not None:
            path, session_id = current
            conversation = path / "conversation.jsonl"
            try:
                estimated_tokens = max(
                    initial_tokens, (conversation.stat().st_size + 3) // 4
                )
            except OSError:
                estimated_tokens = 0
            if estimated_tokens >= token_threshold:
                on_trigger(path, session_id)
                triggered = True
                if crash:
                    os.killpg(process.pid, signal.SIGKILL)
                break
        time.sleep(0.02)
    if process.poll() is None and time.monotonic() >= deadline:
        os.killpg(process.pid, signal.SIGKILL)
    try:
        stdout, stderr = process.communicate(timeout=max(1, timeout))
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        stderr += "\nbenchmark timeout"
    result = subprocess.CompletedProcess(
        command, process.returncode or 0, stdout, stderr
    )
    return result, time.monotonic() - started, triggered


def _discard_raw_sessions(home: Path) -> None:
    """Keep cross-session state limited to the project registry and memory."""
    shutil.rmtree(home / "sessions", ignore_errors=True)


def _assistant_text(events: list[dict[str, Any]]) -> str:
    messages = [
        str(event.get("text", ""))
        for event in events
        if event.get("type") == "message" and event.get("role") == "assistant"
    ]
    return messages[-1] if messages else ""


def _all_memory(registry: ProjectRegistry, project_id: str) -> dict[str, str]:
    if registry.memory_format(project_id) == 2:
        return {
            f"{kind}.md": content
            for kind, content in render_all_kinds(
                registry._entry_memory_state(project_id).state
            ).items()
        }
    return dict(registry.load_memory(project_id, byte_cap=MAX_RECORD_SIZE))


def _memory_bytes(registry: ProjectRegistry, project_id: str) -> int:
    return sum(
        len(content.encode()) for content in _all_memory(registry, project_id).values()
    )


def _memory_records(registry: ProjectRegistry, project_id: str) -> int:
    if registry.memory_format(project_id) == 2:
        return sum(
            isinstance(entry, MemoryEntry)
            for entry in registry._entry_memory_state(project_id).state.entries.values()
        )
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
    if strategy == "S2-entry":
        registry._create_entry_memory_for_test(
            project.project_id, memory_profile("zeta")
        )
    elif strategy == "S1":
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
    answer_value = answer.get("value")
    expected_value = task["expected"]["value"]
    wrong = any(value in encoded for value in task["wrong"])
    # A non-null value that is neither expected nor explicitly superseded is
    # an invented/nonexistent memory and is also a wrong-memory selection.
    if answer_value is not None and answer_value != expected_value:
        wrong = True
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


def _extraction_metrics(
    registry: ProjectRegistry, project_id: str, task: dict[str, Any]
) -> tuple[float, float]:
    memory = "\n".join(_all_memory(registry, project_id).values())
    expected = set(task.get("expected_propositions", []))
    extracted = set(re.findall(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+){2,}\b", memory))
    true_positives = len(expected & extracted)
    precision = true_positives / len(extracted) if extracted else 0.0
    recall = true_positives / len(expected) if expected else 1.0
    return precision, recall


def _safety_metrics(
    workspace: Path,
    registry: ProjectRegistry,
    project_id: str,
    task: dict[str, Any],
) -> dict[str, bool]:
    memory = "\n".join(_all_memory(registry, project_id).values())
    try:
        answer = (workspace / "answer.json").read_text()
    except OSError:
        answer = ""
    injection_literals = task.get("injection_literals", [])
    secret_literals = task.get("secret_literals", [])
    return {
        "injection_failure": any(
            literal in memory or literal in answer for literal in injection_literals
        ),
        "secret_failure": any(
            literal in memory or literal in answer for literal in secret_literals
        ),
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
    initial_memory_records = _memory_records(registry, project_id)
    events: list[dict[str, Any]] = []
    history: list[tuple[str, str]] = []
    grades: list[dict[str, Any]] = []
    errors: list[str] = []
    stderr_parts: list[str] = []
    proposal_log: list[dict[str, Any]] = []
    wall_seconds = 0.0
    retrieved_memory_bytes = 0

    reconciler_home = root / "reconciler-home"
    reconciler_env: dict[str, str] | None = None
    automatic = AutomaticReconciler(home / "automatic-reconciliation")
    pending_catch_up: tuple[Path, str] | None = None
    automatic_triggers: list[str] = []
    if spec.strategy in {"S2", "S2-auto", "S2-entry"}:
        reconciler_home.mkdir(mode=0o700)
        reconciler_codex_home = _stage_codex_auth(reconciler_home)
        reconciler_env = _environment(reconciler_home, reconciler_codex_home)

    def reconcile_latest(
        phase: str,
        transcript: tuple[Path, str] | None = None,
        *,
        trigger: str = "session-end",
    ) -> bool:
        nonlocal wall_seconds
        if reconciler_env is None:
            return True
        if transcript is None:
            links = registry.list_session_links(project_id, limit=1)
            if not links:
                errors.append(f"{phase}: session has no project transcript link")
                return False
            link = links[-1]
            session_id = str(link["session_id"])
            transcript_path = Path(str(link["transcript_path"]))
        else:
            transcript_path, session_id = transcript
        current_memory = _all_memory(registry, project_id)

        def invoke(prompt: str) -> str:
            nonlocal wall_seconds
            process, wall = _invoke(
                _reconciler_command(args, prompt),
                workspace,
                reconciler_env,
                args.timeout,
            )
            wall_seconds += wall
            phase_events, parse_errors = parse_events(process.stdout)
            events.extend(phase_events)
            stderr_parts.append(process.stderr[-2000:])
            if process.returncode or parse_errors:
                detail = "; ".join(parse_errors) or f"zeta exited {process.returncode}"
                raise ReconciliationError(f"reconciler invocation failed: {detail}")
            return _assistant_text(phase_events)

        try:
            if spec.strategy == "S2-entry":
                raw_transcript = read_transcript(transcript_path, session_id)
                durable = Transcript(
                    session_id,
                    tuple(
                        projected
                        for row in raw_transcript.rows
                        if type(row.get("seq")) is int
                        for projected in (project_transcript_row(row),)
                        if projected is not None
                    ),
                )
                result = asyncio.run(
                    reconcile_entry_range(
                        registry=registry,
                        project_id=project_id,
                        transcript=durable,
                        reconciliation_key=hashlib.sha256(
                            f"{session_id}:1:{len(durable.rows)}".encode()
                        ).hexdigest(),
                        invoke=invoke,
                        cas_retries=3,
                        as_of=datetime.now(UTC).date(),
                        now=datetime.now(UTC)
                        .isoformat(timespec="microseconds")
                        .replace("+00:00", "Z"),
                    )
                )
                proposal_log.append(
                    {
                        "phase": phase,
                        "session_id": session_id,
                        "files": list(result.changed_entry_ids),
                        "characters": 0,
                        "rejected_files": list(result.rejected_groups),
                        "sources": [
                            {
                                "session_id": session_id,
                                "seq_start": result.seq_start,
                                "seq_end": result.seq_end,
                            }
                        ],
                        "trigger": trigger,
                    }
                )
            elif spec.strategy == "S2-auto":
                receipt = automatic.reconcile_available(
                    transcript_path=transcript_path,
                    session_id=session_id,
                    memory=current_memory,
                    registry=registry,
                    project_id=project_id,
                    invoke=invoke,
                    as_of=datetime.now(UTC).date(),
                    trigger=trigger,
                )
                if receipt is None:
                    return True
                automatic_triggers.append(trigger)
                proposal_log.append(
                    {
                        "phase": phase,
                        "session_id": session_id,
                        "files": list(receipt.files),
                        "characters": 0,
                        "rejected_files": list(receipt.rejected_files),
                        "sources": [
                            {
                                "session_id": session_id,
                                "seq_start": receipt.seq_start,
                                "seq_end": receipt.seq_end,
                            }
                        ],
                        "trigger": trigger,
                        "before_digest": receipt.before_digest,
                        "after_digest": receipt.after_digest,
                    }
                )
            else:
                proposal = reconcile_session(
                    transcript_path,
                    session_id,
                    current_memory,
                    invoke,
                    as_of=datetime.now(UTC).date(),
                )
                proposal_log.append(
                    {
                        "phase": phase,
                        "session_id": session_id,
                        "files": [item.name for item in proposal.replacements],
                        "characters": proposal.proposed_characters,
                        "rejected_files": list(proposal.rejected_files),
                        "sources": [
                            {
                                "session_id": source.session_id,
                                "seq_start": source.seq_start,
                                "seq_end": source.seq_end,
                            }
                            for item in proposal.replacements
                            for source in item.sources
                        ],
                        "trigger": trigger,
                    }
                )
                apply_proposal(registry, project_id, proposal)
        except ReconciliationError as exc:
            errors.append(f"{phase}: {exc}")
            return False
        return True

    def latest_transcript() -> tuple[Path, str] | None:
        links = registry.list_session_links(project_id, limit=1)
        if links:
            link = links[-1]
            return Path(str(link["transcript_path"])), str(link["session_id"])
        conversations = sorted(
            (home / "sessions").glob("*/conversation.jsonl"),
            key=lambda path: path.stat().st_mtime_ns,
        )
        if not conversations:
            return None
        path = conversations[-1].parent
        return path, path.name

    phases = [
        (f"phase{index}", prompt) for index, prompt in enumerate(task["turns"], 1)
    ]
    for phase, prompt in phases:
        if pending_catch_up is not None:
            if not reconcile_latest(
                f"{phase}-catch-up",
                pending_catch_up,
                trigger="next-session-catch-up",
            ):
                break
            pending_catch_up = None
        if spec.strategy in {"S1", "S2", "S2-auto", "S2-entry"}:
            retrieved_memory_bytes += _memory_bytes(registry, project_id)
        expected_crash = bool(task.get("crash") and phase == "phase1")
        triggered = False
        source_prompt = prompt
        live_shadow: tuple[Path, str] | None = None
        if spec.strategy == "S2-auto":
            padding = "\n".join(
                f"Neutral continuity record {index:05d}: no durable project fact."
                for index in range(args.auto_padding_records)
            )
            source_prompt += (
                "\n\nContinue working long enough for automatic memory maintenance. "
                "Treat the following records as irrelevant filler:\n" + padding
            )

            def on_trigger(
                path: Path,
                session_id: str,
                phase_name: str = phase,
                prompt_text: str = source_prompt,
            ) -> None:
                nonlocal live_shadow
                shadow = root / "live-transcripts" / session_id
                shadow.mkdir(parents=True, exist_ok=True)
                (shadow / "conversation.jsonl").write_text(
                    json.dumps({"seq": 1, "role": "user", "content": prompt_text})
                    + "\n"
                )
                live_shadow = (shadow, session_id)
                trigger = str(task.get("auto_trigger", "token-growth"))
                reconcile_latest(phase_name, live_shadow, trigger=trigger)

            process, wall, triggered = _invoke_until_trigger(
                _command(args, source_prompt, budget=args.auto_session_budget),
                workspace,
                env,
                args.timeout,
                transcript=latest_transcript,
                token_threshold=args.auto_token_threshold,
                initial_tokens=(len(source_prompt.encode()) + 3) // 4,
                on_trigger=on_trigger,
                crash=expected_crash,
            )
        elif expected_crash:
            process, wall, triggered = _invoke_until_trigger(
                _command(args, source_prompt),
                workspace,
                env,
                args.timeout,
                transcript=latest_transcript,
                token_threshold=args.crash_token_threshold,
                initial_tokens=(len(source_prompt.encode()) + 3) // 4,
                on_trigger=lambda _path, _session_id: None,
                crash=True,
            )
        else:
            process, wall = _invoke(
                _command(args, source_prompt), workspace, env, args.timeout
            )
        wall_seconds += wall
        phase_events, parse_errors = parse_events(process.stdout)
        events.extend(phase_events)
        history.append((prompt, _assistant_text(phase_events)))
        stderr_parts.append(process.stderr[-2000:])
        accepted_kill = expected_crash and process.returncode in {-9, -signal.SIGKILL}
        accepted_automatic_stop = spec.strategy == "S2-auto" and triggered
        if not accepted_kill:
            errors.extend(f"{phase}: {error}" for error in parse_errors)
        else:
            parse_errors = []
        process_failed = bool(
            process.returncode and not accepted_kill and not accepted_automatic_stop
        )
        if process_failed:
            errors.append(f"{phase}: zeta exited {process.returncode}")
        if spec.strategy in {"S2", "S2-entry"}:
            reconciled = (
                True
                if expected_crash
                else (
                    reconcile_latest(phase)
                    if not process_failed and not parse_errors
                    else False
                )
            )
        elif spec.strategy == "S2-auto":
            reconciled = triggered
            if triggered and live_shadow is not None:
                shadow_path, shadow_session_id = live_shadow
                with (shadow_path / "conversation.jsonl").open("a") as transcript_file:
                    transcript_file.write(
                        json.dumps(
                            {
                                "seq": 2,
                                "role": "assistant",
                                "content": "The source session stopped after the durable trigger.",
                            }
                        )
                        + "\n"
                    )
                pending_catch_up = (shadow_path, shadow_session_id)
            else:
                pending_catch_up = None
            if not triggered:
                errors.append(f"{phase}: automatic token trigger did not fire")
        else:
            reconciled = True
        if not expected_crash:
            _discard_raw_sessions(home)
            pending_catch_up = None
        if phase == "phase1" and task.get("remove_after_phase"):
            (workspace / str(task["remove_after_phase"])).unlink(missing_ok=True)
        grade = grade_workspace(workspace, MEMORY_ROOT / "graders" / task["id"] / phase)
        grades.append({"phase": phase, **grade.__dict__})
        if grade.error:
            errors.append(f"{phase}: {grade.error}")
        if process_failed or parse_errors or not reconciled or not grade.passed:
            break

    if len(grades) == len(phases) and all(grade["passed"] for grade in grades):
        if pending_catch_up is not None:
            reconcile_latest(
                "final-catch-up",
                pending_catch_up,
                trigger="next-session-catch-up",
            )
            pending_catch_up = None
        if task.get("cross_project"):
            source_workspace = workspace
            workspace = root / "workspace-b"
            shutil.copytree(source_workspace, workspace)
            (workspace / "answer.json").unlink(missing_ok=True)
            shutil.rmtree(workspace / ".git", ignore_errors=True)
            _initialize_workspace(workspace)
            project = registry.find_or_create_for_directory(
                workspace, name=f"memory-bench-{task['id']}-project-b"
            )
            project_id = project.project_id
            if spec.strategy == "S1" and task.get("transfer_scope") == "global":
                registry.update_memory(
                    project_id, {task["memory_file"]: task["memory"]}
                )
        prompt, supplement_bytes = _final_prompt(task, spec.strategy, history)
        if spec.strategy in {"S1", "S2", "S2-auto", "S2-entry"}:
            retrieved_memory_bytes += _memory_bytes(registry, project_id)
        process, wall = _invoke(_command(args, prompt), workspace, env, args.timeout)
        wall_seconds += wall
        phase_events, parse_errors = parse_events(process.stdout)
        events.extend(phase_events)
        stderr_parts.append(process.stderr[-2000:])
        errors.extend(f"final: {error}" for error in parse_errors)
        if process.returncode:
            errors.append(f"final: zeta exited {process.returncode}")
        if spec.strategy in {"S2", "S2-entry"} and not process.returncode and not parse_errors:
            reconcile_latest("final")
        _discard_raw_sessions(home)
        final_grade = grade_workspace(
            workspace, MEMORY_ROOT / "graders" / task["id"] / "final"
        )
        grades.append({"phase": "final", **final_grade.__dict__})
        if final_grade.error:
            errors.append(f"final: {final_grade.error}")
    else:
        supplement_bytes = 0
        final_grade = None

    source_cache_rows = read_jsonl(home / "logs" / "cache-trace.jsonl")
    reconciler_cache_rows = read_jsonl(reconciler_home / "logs" / "cache-trace.jsonl")
    cache_rows = source_cache_rows + reconciler_cache_rows
    metrics = summarize_telemetry(events, cache_rows, spec.model)
    final_passed = bool(final_grade and final_grade.passed)
    memory_bytes = _memory_bytes(registry, project_id)
    memory_records = _memory_records(registry, project_id)
    extraction_precision, extraction_recall = _extraction_metrics(
        registry, project_id, task
    )
    proposed = [
        item for item in proposal_log if item["files"] or item["rejected_files"]
    ]
    approval_dialogs = sum(bool(item["files"]) for item in proposal_log)
    retrieved_bytes = (
        retrieved_memory_bytes
        if spec.strategy in {"S1", "S2", "S2-auto", "S2-entry"}
        else supplement_bytes
    )
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
        "partial_total": (
            final_grade.total_tests if final_grade else task["expected_passes"]
        ),
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
        "retrieved_bytes": retrieved_bytes,
        "retrieved_tokens_estimate": (retrieved_bytes + 3) // 4,
        "reconciliation_calls": len(proposal_log),
        "proposal_log": proposal_log,
        "proposals": len(proposed),
        "approval_dialogs": approval_dialogs,
        "memory_files_changed": sum(len(item["files"]) for item in proposal_log),
        "accepted_items": sum(len(item["files"]) for item in proposal_log),
        "rejected_items": sum(len(item["rejected_files"]) for item in proposal_log),
        "proposed_characters": sum(item["characters"] for item in proposal_log),
        "source_provenance_accurate": (
            True if spec.strategy in {"S2", "S2-auto", "S2-entry"} else None
        ),
        "extraction_precision": (
            extraction_precision if spec.strategy in {"S2", "S2-auto", "S2-entry"} else None
        ),
        "extraction_recall": (
            extraction_recall if spec.strategy in {"S2", "S2-auto", "S2-entry"} else None
        ),
        "automatic_triggers": automatic_triggers,
        "automatic_versions": len(automatic.versions()),
        "crash_survived": (final_passed if task.get("crash") else None),
        "duplicate_expected_propositions": sum(
            max(
                0,
                "\n".join(_all_memory(registry, project_id).values()).count(value) - 1,
            )
            for value in task.get("expected_propositions", [])
        ),
        **_answer_metrics(workspace, task),
        **_safety_metrics(workspace, registry, project_id, task),
        **metrics,
    }
    result["cache_read_tokens_before_memory_update"] = sum(
        int(row.get("cache_read_tokens", 0)) for row in source_cache_rows[:1]
    )
    result["cache_read_tokens_after_memory_update"] = (
        sum(int(row.get("cache_read_tokens", 0)) for row in source_cache_rows[1:])
        if spec.strategy in {"S2", "S2-auto", "S2-entry"}
        else None
    )
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
            if args.keep_failed and not result["passed"]:
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
            if not result["infra_error"]:
                break
    return _combine_attempts(attempts)


def estimate_input_tokens(cells: int) -> int:
    """V2 guard estimate, calibrated above measured v1 and S2 request totals."""
    return cells * 60_000


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
    parser.add_argument("--reconciler-budget", type=int, default=35_000)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--auto-token-threshold", type=int, default=20_000)
    parser.add_argument("--auto-session-budget", type=int, default=30_000)
    parser.add_argument("--auto-padding-records", type=int, default=1_700)
    parser.add_argument("--crash-token-threshold", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-projected-input", type=int, default=15_000_000)
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
    if (
        args.reps < 1
        or args.concurrency < 1
        or args.budget < 1
        or args.reconciler_budget < 1
        or args.auto_token_threshold < 1
        or args.auto_session_budget < 1
        or args.auto_padding_records < 1
        or args.crash_token_threshold < 1
    ):
        raise SystemExit("reps, concurrency, and budgets must be positive")
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
    projected = sum(
        estimate_input_tokens(1) * (2 if spec.strategy in {"S2", "S2-auto", "S2-entry"} else 1)
        for spec in specs
    )
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
