#!/usr/bin/env python3
"""Replay deterministic eviction through production code without provider calls."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any


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


def _normalize_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ids: dict[str, str] = {}
    for index, entry in enumerate(entries, 1):
        ids[entry["id"]] = f"generated-{index:04d}"
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
    target.mkdir(parents=True)
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
            user = with_message_origin(Message(MessageRole.USER, [TextContent(f"fixture request {index}")]), MessageOrigin.USER)
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


async def _run_one(session_dir: Path, artifact_dir: Path, budget: int, tail: int, provider: str) -> list[dict[str, Any]]:
    from zeta.core.context import ContextAssembler
    from zeta.core.store import ConversationStore
    store = await asyncio.to_thread(ConversationStore, session_dir.parent, session_id=session_dir.name, _collect_persisted_appends=True)
    try:
        meta = json.loads((session_dir / "meta.json").read_text()) if (session_dir / "meta.json").exists() else {}
        assembler = ContextAssembler(store, token_budget=budget, retained_tail=tail, system_prompt=meta.get("system_prompt", ""), backend=None)
        assert assembler.backend is None
        # A deterministic gate must fail loudly if production reaches model summary.
        async def forbidden(*args: Any, **kwargs: Any) -> str:
            raise RuntimeError("eviction replay reached model-summary fallback")
        assembler.compaction_policy.summarize_chunked = forbidden  # type: ignore[method-assign]
        records: list[dict[str, Any]] = []
        previous_count = len(store.replay())
        stable = 0
        for invocation in range(1, 9):
            before = {entry.id for entry in store.replay()}
            store.enable_persisted_append_tracking()
            context = await assembler.assemble_context(backend=None, force=True, bypass_eviction_hysteresis=True)
            assert assembler.backend is None
            after_entries = store.replay()
            appended = [entry.to_dict() for entry in after_entries if entry.id not in before]
            store.take_persisted_appends()
            normalized = _normalize_entries(appended)
            readback_store = await asyncio.to_thread(ConversationStore, session_dir.parent, session_id=store.session_id, _read_only=True, _must_exist=True)
            try:
                readback_assembler = ContextAssembler(readback_store, token_budget=budget, retained_tail=tail, system_prompt=meta.get("system_prompt", ""), backend=None)
                readback = await readback_assembler.assemble_context(backend=None)
            finally:
                readback_store.close()
            context_data = [message.to_dict() for message in context.messages]
            readback_data = [message.to_dict() for message in readback.messages]
            trace = {"invocation": invocation, "appended": len(normalized), "token_count": context.token_count, "compacted": context.compacted, "marker_count": store.compaction_marker_count(), "retained_floor": (assembler.last_compaction_telemetry or {}).get("retained_floor")}
            artifacts = {"context.json": context_data, "wire.json": _wire(list(context.messages), provider), "appends.json": normalized, "readback.json": readback_data, "scalar.json": trace}
            invocation_dir = artifact_dir / str(invocation)
            invocation_dir.mkdir(parents=True)
            for name, value in artifacts.items():
                (invocation_dir / name).write_bytes(_json(value))
            records.append({"invocation": invocation, "appended": len(normalized), "digests": {name: _digest(value) for name, value in artifacts.items()}, "trace": trace})
            if not appended and len(after_entries) == previous_count:
                stable += 1
            else:
                stable = 0
            previous_count = len(after_entries)
            if stable >= 2:
                break
        return records
    finally:
        store.close()


def run(args: argparse.Namespace) -> int:
    root = Path(args.session).expanduser() if args.session else None
    temporary = tempfile.TemporaryDirectory(prefix="eviction-gate-")
    session_root = Path(temporary.name) / "sessions"
    session_dir = session_root / ("fixture" if args.fixture else Path(args.session).name)
    if args.fixture:
        _fixture_store(session_root)
        budget, tail, provider = 4_000, 3, "anthropic"
    else:
        if root is None:
            raise SystemExit("--session is required")
        _copy_session(root, session_dir)
        manifest = session_root / "storage-format.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path.home() / ".zeta" / "storage-format.json", manifest)
        meta = json.loads((session_dir / "meta.json").read_text())
        budget, tail, provider = int(meta.get("compaction_budget", 400_000)), int(meta.get("retained_tail", 8)), str(meta.get("provider", "anthropic"))
    artifact_dir = Path(args.artifact_dir or tempfile.mkdtemp(prefix="eviction-artifacts-"))
    started = time.monotonic()
    try:
        records = asyncio.run(asyncio.wait_for(_run_one(session_dir, artifact_dir, budget, tail, provider), 600))
        import zeta
        record = {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "zeta_file": str(Path(zeta.__file__).resolve()), "session": args.session or "fixture", "records": records, "elapsed_seconds": round(time.monotonic() - started, 3), "artifact_dir": str(artifact_dir)}
        print(json.dumps(record, sort_keys=True, separators=(",", ":")))
        return 0
    finally:
        temporary.cleanup()


def compare(args: argparse.Namespace) -> int:
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"phase": "compare", "sessions": []}
    sessions = ["fixture"] if args.fixture else args.session
    for session in sessions:
        rows: list[dict[str, Any]] = []
        for label, worktree in (("base", args.base), ("head", args.head)):
            artifact_dir = Path(tempfile.mkdtemp(prefix=f"eviction-{label}-"))
            command = ["uv", "run", "--directory", worktree, "python", str(Path(worktree) / "scripts/eviction_replay_gate.py"), "run", "--artifact-dir", str(artifact_dir)]
            if session == "fixture":
                command.append("--fixture")
            else:
                command.extend(["--session", str(Path.home() / ".zeta" / "sessions" / session)])
            result = subprocess.run(command, capture_output=True, text=True, timeout=620, check=False)
            if result.returncode:
                print(f"FAIL {session} {label}: runner failed", file=sys.stderr)
                return 1
            record = json.loads(result.stdout.strip().splitlines()[-1])
            expected = str(Path(worktree).resolve())
            if not Path(record["zeta_file"]).resolve().is_relative_to(Path(expected)):
                print(f"FAIL {session} {label}: zeta.__file__ outside expected worktree", file=sys.stderr)
                return 1
            rows.append({"label": label, "record": record, "artifacts": artifact_dir})
        base, head = rows
        if len(base["record"]["records"]) != len(head["record"]["records"]):
            print(f"FAIL {session}: scalar trace invocation count differs", file=sys.stderr)
            return 1
        for index, (left, right) in enumerate(zip(base["record"]["records"], head["record"]["records"]), 1):
            for name in left["digests"]:
                if left["digests"][name] != right["digests"][name]:
                    a = json.loads((base["artifacts"] / str(index) / name).read_text())
                    b = json.loads((head["artifacts"] / str(index) / name).read_text())
                    print(f"FAIL {session} invocation {index}: {name} at {_pointer_diff(a, b)}", file=sys.stderr)
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
    run_parser.add_argument("--artifact-dir")
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
