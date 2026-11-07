#!/usr/bin/env python3
"""Replay deterministic eviction through production code without provider calls."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Self


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value)).hexdigest()


def _pointer_diff(left: Any, right: Any, pointer: str = "") -> str | None:
    if type(left) is not type(right):
        return pointer or "/"
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                return f"{pointer}/{key}"
            found = _pointer_diff(left[key], right[key], f"{pointer}/{key}")
            if found:
                return found
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{pointer}/length"
        for index, (a, b) in enumerate(zip(left, right)):
            found = _pointer_diff(a, b, f"{pointer}/{index}")
            if found:
                return found
        return None
    return None if left == right else (pointer or "/")


def _normalize_entries(
    entries: list[dict[str, Any]], ids: dict[str, str], next_id: list[int]
) -> list[dict[str, Any]]:
    for entry in entries:
        if entry["id"] not in ids:
            next_id[0] += 1
            ids[entry["id"]] = f"generated-{next_id[0]:04d}"
    def replace(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: replace(ids.get(item, item) if key in {"id", "parent_id"} else item)
                    for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item) for item in value]
        return value
    result = replace(entries)
    for entry in result:
        data = entry.get("data")
        if isinstance(data, dict) and isinstance(data.get("replaces"), list):
            data["replaces"] = [ids.get(item, item) for item in data["replaces"]]
    return result


def _copy_session(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in ("conversation.jsonl", "meta.json", "session_state.json"):
        source_file = source / name
        if source_file.exists():
            shutil.copy2(source_file, target / name)
    (target / ".lock").touch()


def _fixture_store(store_dir: Path) -> None:
    (store_dir.parent / "storage-format.json").write_text(json.dumps({"writer_epoch": 2}))
    from zeta.core.store import ConversationStore
    from zeta.protocol.types import (
        Message,
        MessageOrigin,
        MessageRole,
        TextContent,
        ToolCall,
        ToolResult,
        ToolUseContent,
        with_message_origin,
    )
    fixture_path = Path(__file__).parents[1] / "tests" / "fixtures" / "eviction_replay_fixture.json"
    names = json.loads(fixture_path.read_text())["tool_names"]
    original_uuid4 = uuid.uuid4
    # The fixture is synthetic, so deterministic ids make its committed golden set stable.
    counter = 0
    def next_uuid() -> uuid.UUID:
        nonlocal counter
        counter += 1
        return uuid.UUID(int=counter)
    uuid.uuid4 = next_uuid  # type: ignore[assignment]
    store = ConversationStore(store_dir, session_id="fixture")
    try:
        for index in range(48):
            name = names[index % len(names)]
            call_id = f"fixture-call-{index:03d}"
            user = with_message_origin(
                Message(
                    MessageRole.USER,
                    [TextContent(
                        f"fixture request {index} PR #{index + 1} verdict=pass "
                        f"child fixture-{index} status=completed"
                    )],
                ),
                MessageOrigin.USER,
            )
            call = with_message_origin(Message(MessageRole.ASSISTANT, [ToolUseContent(ToolCall(call_id, name, {"index": index}))]), MessageOrigin.AGENT_PROMPT)
            result = with_message_origin(Message(MessageRole.TOOL_RESULT, [], tool_result=ToolResult(call_id, "fixture result " + ("x" * 500), is_error=index % 11 == 0)), MessageOrigin.AUTOMATION_RECEIPT)
            store.append_message(user)
            store.append_message(call)
            store.append_message(result)
        store.append_message(with_message_origin(Message(MessageRole.USER, [TextContent("fixture final request")]), MessageOrigin.USER))
    finally:
        store.close()
        uuid.uuid4 = original_uuid4  # type: ignore[assignment]


def _wire(messages: list[Any], provider: str) -> dict[str, Any]:
    if provider == "codex":
        from zeta.providers.codex_payload import build_responses_payload
        return build_responses_payload(messages, [], model="fixture",)
    from zeta.providers.anthropic_payload import build_messages_payload
    return build_messages_payload(messages, [], model="fixture", max_tokens=16_384)


def _production_limits() -> dict[str, int]:
    from zeta.context_eviction import _fact_floor

    return {
        "oldest": _fact_floor._OLDEST_LIMIT,
        "newest": _fact_floor._NEWEST_LIMIT,
        "chars": _fact_floor.FACT_FLOOR_CHAR_CAP,
        "tokens": _fact_floor.FACT_FLOOR_TOKEN_CAP,
    }


def _measure_floor(row: dict[str, Any]) -> dict[str, int] | None:
    from zeta.context_accounting import message_token_count
    from zeta.context_eviction._fact_floor import render_fact_floor, validate_fact_floor

    floor = row.get("data", {}).get("fact_floor")
    if floor is None:
        return None
    if not validate_fact_floor(floor):
        raise ValueError(f"seq={row.get('seq')} invalid")
    assert isinstance(floor, dict)
    rendered = render_fact_floor(floor)
    return {
        "oldest": len(floor["oldest"]),
        "newest": len(floor["newest"]),
        "chars": len(rendered.content[0].text) if rendered else 0,
        "tokens": message_token_count(rendered) if rendered else 0,
    }


def _floor_metrics(rows: list[dict[str, Any]], limits: dict[str, int], source: str, violations: list[str] | None = None) -> list[dict[str, int]]:
    metrics: list[dict[str, int]] = []
    for row in rows:
        try:
            metric = _measure_floor(row)
        except ValueError as error:
            detail = str(error)
            if violations is not None:
                violations.append(detail)
                continue
            raise RuntimeError(f"retained-floor bound check failed ({source}): {detail}") from error
        if metric is None:
            continue
        if any(metric[key] > limits[key] for key in metric):
            detail = (f"seq={row.get('seq')} oldest={metric['oldest']} newest={metric['newest']} "
                      f"chars={metric['chars']} tokens={metric['tokens']}")
            if violations is not None:
                violations.append(detail)
            else:
                raise RuntimeError(f"retained-floor bound check failed ({source}): {detail}")
        metrics.append(metric)
    return metrics


def _assembler(store: Any, *, budget: int, tail: int, system_prompt: str = "") -> Any:
    from zeta.core.context import ContextAssembler

    assembler = ContextAssembler(store, token_budget=budget, retained_tail=tail,
                                 system_prompt=system_prompt, backend=None)
    assert assembler.backend is None
    _install_summary_guard(assembler)
    return assembler


def _install_summary_guard(assembler: Any) -> None:
    async def forbidden(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError("eviction replay reached model-summary fallback")
    assembler.compaction_policy.summarize_chunked = forbidden  # type: ignore[method-assign]


def _collect_artifacts(context: Any, readback: Any, normalized: list[dict[str, Any]], trace: dict[str, Any], provider: str) -> dict[str, Any]:
    return {"context.json": [message.to_dict() for message in context.messages], "wire.json": _wire(list(context.messages), provider), "appends.json": normalized, "readback.json": [message.to_dict() for message in readback.messages], "scalar.json": trace}


async def _run_one(session_dir: Path, artifact_dir: Path, budget: int, tail: int, provider: str, limits: dict[str, int] | None = None) -> list[dict[str, Any]]:
    from zeta.core.store import ConversationStore
    store = await asyncio.to_thread(ConversationStore, session_dir.parent, session_id=session_dir.name, _collect_persisted_appends=True)
    try:
        meta = json.loads((session_dir / "meta.json").read_text()) if (session_dir / "meta.json").exists() else {}
        assembler = _assembler(store, budget=budget, tail=tail, system_prompt=meta.get("system_prompt", ""))
        limits = limits or _production_limits()
        records: list[dict[str, Any]] = []
        previous_count = len(store.replay())
        stable = 0
        previous_context_digest: str | None = None
        normalized_ids: dict[str, str] = {}
        next_normalized_id = [0]
        for invocation in range(1, 9):
            before = {entry.id for entry in store.replay()}
            store.enable_persisted_append_tracking()
            context = await assembler.assemble_context(backend=None, force=True, bypass_eviction_hysteresis=True)
            assert assembler.backend is None
            after_entries = store.replay()
            appended = [entry.to_dict() for entry in after_entries if entry.id not in before]
            store.take_persisted_appends()
            normalized = _normalize_entries(appended, normalized_ids, next_normalized_id)
            readback_store = await asyncio.to_thread(ConversationStore, session_dir.parent, session_id=store.session_id, _read_only=True, _must_exist=True)
            try:
                readback_assembler = _assembler(readback_store, budget=budget, tail=tail, system_prompt=meta.get("system_prompt", ""))
                readback = await readback_assembler.assemble_context(backend=None)
            finally:
                readback_store.close()
            stored_floor_violations: list[str] = []
            persisted_rows = [entry.to_dict() for entry in after_entries]
            floor_metrics = _floor_metrics(persisted_rows, limits, "persisted", stored_floor_violations)
            appended_floor_metrics = _floor_metrics(normalized, limits, "appended")
            trace = {"invocation": invocation, "appended": len(normalized), "token_count": context.token_count, "compacted": context.compacted, "marker_count": store.compaction_marker_count(), "retained_floor": (assembler.last_compaction_telemetry or {}).get("retained_floor"), "floor_metrics": floor_metrics, "appended_floor_metrics": appended_floor_metrics, "stored_floor_violations": stored_floor_violations}
            artifacts = _collect_artifacts(context, readback, normalized, trace, provider)
            invocation_dir = artifact_dir / str(invocation)
            invocation_dir.mkdir(parents=True)
            for name, value in artifacts.items():
                (invocation_dir / name).write_bytes(_json(value))
            records.append({"invocation": invocation, "appended": len(normalized), "digests": {name: _digest(value) for name, value in artifacts.items()}, "trace": trace})
            context_digest = _digest(artifacts["context.json"])
            if not appended and len(after_entries) == previous_count and context_digest == previous_context_digest:
                stable += 1
            else:
                stable = 0
            previous_context_digest = context_digest
            previous_count = len(after_entries)
            if stable >= 2:
                break
        if stable < 2:
            raise RuntimeError("eviction replay did not stabilize within 8 invocations")
        return records
    finally:
        store.close()


class _Scratch:
    def __init__(self, path: Path | None) -> None:
        self._owned = tempfile.TemporaryDirectory(prefix="eviction-gate-") if path is None else None
        self.root = path or Path(self._owned.name)  # type: ignore[union-attr]

    def __enter__(self) -> Self:
        self.root.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, *args: object) -> None:
        if self._owned is not None:
            self._owned.cleanup()


def _run_in_scratch(args: argparse.Namespace, scratch: Path) -> int:
    root = Path(args.session).expanduser() if args.session else None
    session_root = scratch / "sessions"
    session_dir = session_root / ("fixture" if args.fixture else Path(args.session).name)
    if args.fixture:
        _fixture_store(session_root)
        budget, tail, provider = 4_000, 3, "anthropic"
    else:
        if root is None:
            raise SystemExit("--session is required")
        _copy_session(root, session_dir)
        shutil.copy2(Path.home() / ".zeta" / "storage-format.json", scratch / "storage-format.json")
        meta = json.loads((session_dir / "meta.json").read_text())
        budget, tail, provider = int(meta.get("compaction_budget", 400_000)), int(meta.get("retained_tail", 8)), str(meta.get("provider", "anthropic"))
    if args.keep_private_artifacts:
        artifact_dir = Path(args.keep_private_artifacts).expanduser()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        print("warning: keeping private replay artifacts", file=sys.stderr)
    else:
        artifact_dir = scratch / "artifacts"
        artifact_dir.mkdir()
    started = time.monotonic()
    records = asyncio.run(asyncio.wait_for(_run_one(session_dir, artifact_dir, budget, tail, provider, json.loads(args.limits_json) if args.limits_json else None), 600))
    import zeta
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(f"identity: head={head} zeta.__file__={Path(zeta.__file__).resolve()}", file=sys.stderr)
    record = {"git_head": head, "limits": _production_limits(), "records": records, "elapsed_seconds": round(time.monotonic() - started, 3)}
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return 0


def run(args: argparse.Namespace) -> int:
    with _Scratch(Path(args.scratch_root).expanduser() if args.scratch_root else None) as scratch:
        return _run_in_scratch(args, scratch.root)


def _run_side(session: str, label: str, worktree: str, fixture: bool, scratch: Path, limits: dict[str, int] | None = None) -> dict[str, Any] | None:
    runner = Path(worktree) / "scripts/eviction_replay_gate.py"
    if not runner.exists():
        runner = Path(__file__).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    tmpdir = scratch / "tmp"
    tmpdir.mkdir()
    command = [sys.executable, str(runner), "run", "--scratch-root", str(scratch)]
    environment = {**os.environ, "TMPDIR": str(tmpdir), "PYTHONPATH": str(Path(worktree) / "src"), "UV_CACHE_DIR": str(Path(worktree) / ".uv-cache")}
    if limits is not None:
        command.extend(["--limits-json", json.dumps(limits, sort_keys=True)])
    if fixture:
        command.append("--fixture")
    else:
        command.extend(["--session", str(Path.home() / ".zeta" / "sessions" / session)])
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=620)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        print(f"FAIL {session} {label}: child terminated", file=sys.stderr)
        return None
    result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    if result.returncode:
        detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "runner failed"
        print(f"FAIL {session} {label}: {detail}", file=sys.stderr)
        return None
    record = json.loads(result.stdout.strip().splitlines()[-1])
    identity = next((line for line in result.stderr.splitlines() if line.startswith("identity: ")), "")
    expected = str(Path(worktree).resolve())
    if expected not in identity:
        print(f"FAIL {session} {label}: identity outside expected worktree", file=sys.stderr)
        return None
    return {"label": label, "record": record, "artifacts": scratch / "artifacts"}


def _compare_artifacts(session: str, base: dict[str, Any], head: dict[str, Any]) -> str | None:
    base_record, head_record = base["record"], head["record"]
    if base_record["limits"] != head_record["limits"]:
        return f"limit change: base={base_record['limits']} head={head_record['limits']}"
    if len(base_record["records"]) != len(head_record["records"]):
        return "scalar trace invocation count differs"
    for index, (left, right) in enumerate(zip(base_record["records"], head_record["records"]), 1):
        for metric in right["trace"]["floor_metrics"]:
            if any(metric[key] > base_record["limits"][key] for key in metric):
                return f"retained-floor bound check failed: {metric} limits={base_record['limits']}"
        for name in left["digests"]:
            if left["digests"][name] != right["digests"][name]:
                a = json.loads((base["artifacts"] / str(index) / name).read_text())
                b = json.loads((head["artifacts"] / str(index) / name).read_text())
                return f"invocation {index}: {name} at {_pointer_diff(a, b)}"
    return None


def compare(args: argparse.Namespace) -> int:
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"phase": "compare", "sessions": []}
    sessions = ["fixture"] if args.fixture else args.session
    for session in sessions:
        with tempfile.TemporaryDirectory(prefix="eviction-compare-") as scratch:
            scratch_path = Path(scratch)
            base = _run_side(session, "base", args.base, args.fixture, scratch_path / "base")
            head = _run_side(session, "head", args.head, args.fixture, scratch_path / "head", base["record"]["limits"] if base else None)
            if base is None or head is None:
                return 1
            failure = _compare_artifacts(session, base, head)
            if failure:
                print(f"FAIL {session}: {failure}", file=sys.stderr)
                return 1
            manifest["sessions"].append({"session": session, "base": base["record"], "head": head["record"], "status": "PASS"})
            print(f"PASS {session} {base['record']['elapsed_seconds']}s/{head['record']['elapsed_seconds']}s")
    out.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    print(f"PASS manifest={out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--session")
    run_parser.add_argument("--fixture", action="store_true")
    run_parser.add_argument("--keep-private-artifacts")
    run_parser.add_argument("--limits-json")
    run_parser.add_argument("--scratch-root")
    run_parser.set_defaults(func=run)
    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("--base", required=True)
    compare_parser.add_argument("--head", required=True)
    compare_parser.add_argument("--session", action="append")
    compare_parser.add_argument("--out", required=True)
    compare_parser.add_argument("--fixture", action="store_true")
    compare_parser.set_defaults(func=compare)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
