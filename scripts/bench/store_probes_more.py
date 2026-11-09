"""Probes 3-7 for store_probes.py: eviction, turn-start context, child RPCs,
zeta serve, and transcript search / memory prep. See store_probes.py."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import store_probes
from sandbox import STUB_MODEL, STUB_PROVIDER, load1
from store_probes import (
    OPS,
    RESULTS,
    WORKTREE,
    _meta,
    count_ops,
    fractions_for,
    load,
    make_home,
    profile,
    reset_home,
    sample,
    scale,
    summary,
    user_message,
)

from zeta.protocol.types import Message, MessageRole, TextContent


def _assistant(text: str) -> Message:
    return Message(MessageRole.ASSISTANT, [TextContent(text)])


def _assembler(opened: Any) -> Any:
    from zeta.core.context import ContextAssembler

    metadata = opened.metadata
    return ContextAssembler(
        opened.store,
        token_budget=metadata.compaction_budget,
        retained_tail=metadata.retained_tail,
        system_prompt=metadata.system_prompt,
    )


# --------------------------------------------------------- 3. eviction events


async def _evict_event(store: Any, assembler: Any, event: int) -> dict[str, Any]:
    """One counted eviction event and what it wrote."""
    seq_before = store.active_branch_head_sequence()
    size_before = store.path.stat().st_size
    markers_before = store.compaction_marker_count()
    with OPS.measure() as ops:
        started = time.perf_counter_ns()
        context = await assembler.assemble_context(
            force=True, bypass_eviction_hysteresis=True
        )
        elapsed = (time.perf_counter_ns() - started) / 1e6
    marker = next((e for e in reversed(store._entries) if e.type == "compaction"), None)
    data = marker.data if marker is not None else {}
    floor, view = data.get("fact_floor"), data.get("view")
    return {
        "event": event + 1,
        "ms_while_counting": round(elapsed, 3),
        "load": load(),
        "rows_written": store.active_branch_head_sequence() - seq_before,
        "bytes_written": store.path.stat().st_size - size_before,
        "new_marker": store.compaction_marker_count() > markers_before,
        "marker_bytes": len(json.dumps(marker.to_dict())) if marker is not None else 0,
        "fact_floor_bytes": len(json.dumps(floor)) if floor is not None else 0,
        "view_items": len(view) if isinstance(view, list) else None,
        "context_messages": len(context.messages),
        "context_tokens": context.token_count,
        "log_rows": len(store._entries),
        "ops": ops,
    }


def probe_evict(args: argparse.Namespace) -> dict[str, Any]:
    """ContextAssembler.assemble_context(force=True, bypass_eviction_hysteresis=True).

    This is the manual /compact deterministic eviction path. No backend is
    given, so no model is called. Before each event the probe appends 20
    user/assistant pairs of about 2 KB, so every event has new material.
    """
    from zeta.core.session import SessionManager

    home, sid = make_home(args.session, 1.0)
    opened = SessionManager(home).open(sid)
    store = opened.store
    events: list[dict[str, Any]] = []
    filler = "word " * 400
    try:
        assembler = _assembler(opened)

        async def run() -> None:
            store.bind_owner_thread()
            for event in range(args.events):
                for index in range(20):
                    store.append_message(
                        user_message(f"evict probe {event}.{index} {filler}")
                    )
                    store.append_message(_assistant(f"reply {event}.{index} {filler}"))
                events.append(await _evict_event(store, assembler, event))
            times = []
            for event in range(args.n):
                for index in range(4):
                    store.append_message(
                        user_message(f"timed {event}.{index} {filler}")
                    )
                    store.append_message(
                        _assistant(f"timed reply {event}.{index} {filler}")
                    )
                started = time.perf_counter_ns()
                await assembler.assemble_context(
                    force=True, bypass_eviction_hysteresis=True
                )
                times.append(((time.perf_counter_ns() - started) / 1e6, load1()))
            events.append({"timed_events_uninstrumented": summary(times)})

        asyncio.run(run())
        store._owner_thread_id = None
        store.append_message(user_message("profile " + filler))
        events.append(
            {
                "profile": profile(
                    f"evict-{args.session}",
                    lambda: asyncio.run(
                        assembler.assemble_context(
                            force=True, bypass_eviction_hysteresis=True
                        )
                    ),
                )
            }
        )
    finally:
        store.close()
        reset_home(args.session)
    return {
        "function": "ContextAssembler.assemble_context(force=True, bypass_eviction_hysteresis=True)",
        "events": events,
    }


# ------------------------------------------------------ 4. turn-start context


def probe_context(args: argparse.Namespace) -> dict[str, Any]:
    """Per-turn prompt build: needs_compaction() then assemble()
    (runtime/loop/agent.py:845-866), plus the resume-time memory block refresh
    zeta.runtime.prompt_resume._conservative_context -> refresh_project_memory."""

    out: dict[str, Any] = {
        "functions": [
            "ContextAssembler.needs_compaction",
            "ContextAssembler.assemble",
            "zeta.runtime.prompt_resume._conservative_context",
        ]
    }
    for fraction in fractions_for(args):
        out[str(fraction)] = _context_row(args, fraction)
    return out


def _context_row(args: argparse.Namespace, fraction: float) -> dict[str, Any]:
    from zeta.core.session import SessionManager
    from zeta.runtime.prompt_resume import _conservative_context

    home, sid = make_home(args.session, fraction)
    opened = SessionManager(home).open(sid)
    store = opened.store
    row: dict[str, Any] = {"scale": scale(home, sid)}
    try:
        assembler = _assembler(opened)
        counter = iter(range(10**9))

        def needs() -> Any:
            return assembler.needs_compaction()

        def assemble() -> Any:
            return asyncio.run(assembler.assemble())

        def turn_after_append() -> Any:
            store.append_message(user_message(f"turn {next(counter)}"))
            assembler.needs_compaction()
            return asyncio.run(assembler.assemble())

        def memory() -> Any:
            return _conservative_context(
                opened.metadata, home=home, project_features_enabled=True
            )

        row["needs_compaction"], _ = sample(needs, args.n)
        row["needs_compaction_ops"] = count_ops(needs)
        row["assemble"], messages = sample(assemble, args.n)
        row["assemble_messages"] = len(messages)
        row["assemble_ops"] = count_ops(assemble)
        row["turn_after_append"], _ = sample(turn_after_append, args.n)
        row["turn_after_append_ops"] = count_ops(turn_after_append)
        row["memory_refresh"], _ = sample(memory, args.n)
        row["memory_refresh_ops"] = count_ops(memory)
        row["system_prompt_chars"] = len(opened.metadata.system_prompt)
        if fraction == 1.0:
            row["turn_profile"] = profile(
                f"context-turn-{args.session}", turn_after_append
            )
    finally:
        store.close()
        reset_home(args.session, fraction)
    return row


# ---------------------------------------------------------- 5. child RPCs


def probe_children(args: argparse.Namespace) -> dict[str, Any]:
    """zeta.tools.agent._agent_status / _agent_output (the agent_status and
    agent_output tool handlers) and the parent-side spawn writes
    ConversationStore.allocate_agent_index + register_agent_child
    (called by agent/runner.py:run_agent_tool)."""
    from types import SimpleNamespace

    from zeta.core.session import SessionManager
    from zeta.protocol.types import ToolCall
    from zeta.tools.agent import _active_agent_receipts, _agent_output, _agent_status

    home, sid = make_home(args.session, 1.0)
    opened = SessionManager(home).open(sid)
    store = opened.store
    out: dict[str, Any] = {"scale": scale(home, sid)}
    try:
        registry = SimpleNamespace(session_store=store, max_output_chars=10_000)
        receipts = _active_agent_receipts(store)
        out["active_receipts"] = len(receipts)
        sizes = {
            handle: (path / "conversation.jsonl").stat().st_size
            for handle, path in receipts.items()
            if (path / "conversation.jsonl").exists()
        }
        big = max(sizes, key=sizes.get)
        out["largest_child"] = {"handle": big, "conversation_bytes": sizes[big]}

        def call(handler: Any, arguments: dict[str, Any]) -> Any:
            return asyncio.run(handler(registry, arguments, None))

        first = call(_agent_output, {"handle": big})
        total = first["structuredContent"]["total"]
        out["agent_output_total_chars"] = total
        cases = {
            "agent_status_list": (_agent_status, {}),
            "agent_status_handle": (_agent_status, {"handle": big}),
            "agent_output_first": (_agent_output, {"handle": big}),
            "agent_output_last": (
                _agent_output,
                {"handle": big, "offset": max(0, total - 5000)},
            ),
        }
        for name, (handler, arguments) in cases.items():
            out[name], result = sample(
                lambda h=handler, a=arguments: call(h, a), args.n
            )
            out[name + "_ops"] = count_ops(lambda h=handler, a=arguments: call(h, a))
            out[name + "_response_bytes"] = len(json.dumps(result))
        out["agent_status_list_profile"] = profile(
            f"agent-status-{args.session}", lambda: call(_agent_status, {})
        )
        out["agent_output_profile"] = profile(
            f"agent-output-{args.session}", lambda: call(_agent_output, {"handle": big})
        )
        state = store.session_dir / "session_state.json"
        out["session_state_bytes"] = state.stat().st_size

        def spawn_parent_side() -> None:
            index = store.allocate_agent_index()
            store.register_agent_child(
                ToolCall(
                    f"probe-{index}", "agent", {"prompt": "x", "description": "probe"}
                ),
                child_session_path=str(store.session_dir / "agents" / str(index)),
                description="probe",
                child_instance_id=f"{sid}:{index}",
            )

        out["spawn_parent_side"], _ = sample(spawn_parent_side, args.n)
        out["spawn_parent_side_ops"] = count_ops(spawn_parent_side)
        out["session_state_bytes_after"] = state.stat().st_size
    finally:
        store.close()
        reset_home(args.session)
    return out


# ------------------------------------------------------------- 6. zeta serve


FRAME_TIMEOUT = 120  # seconds to wait for one serve frame


class Client:
    def __init__(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self.reader, self.writer, self.ident = reader, writer, 0
        self.retries = 0

    async def request(
        self, method: str, params: dict[str, Any]
    ) -> tuple[dict[str, Any], float, int]:
        self.ident += 1
        ident = self.ident
        started = time.perf_counter_ns()
        frame_out = {"jsonrpc": "2.0", "id": ident, "method": method, "params": params}
        self.writer.write((json.dumps(frame_out) + "\n").encode())
        await self.writer.drain()
        while True:
            # A stalled serve fails the probe instead of hanging it.
            line = await asyncio.wait_for(self.reader.readline(), FRAME_TIMEOUT)
            if not line:
                raise RuntimeError("serve closed the connection")
            frame = json.loads(line)
            if frame.get("id") == ident:
                elapsed = (time.perf_counter_ns() - started) / 1e6
                if "error" in frame:
                    raise RuntimeError(f"{method}: {frame['error']}")
                return frame["result"], elapsed, len(line)


FEATURES = [
    "session_history_fidelity",
    "session_history_backward",
    "approval_reconnect",
    "served_turn_end",
]


async def connect(socket: Path) -> Client:
    """Connect and say hello. A controller that just disconnected can still be
    registered for a moment; retry that refusal (counted in Client.retries)."""
    retries = 0
    while True:
        reader, writer = await asyncio.open_unix_connection(
            str(socket), limit=8 * 1024 * 1024
        )
        client = Client(reader, writer)
        try:
            await client.request(
                "hello", {"protocol_version": "1.2", "features": FEATURES}
            )
        except RuntimeError as exc:
            writer.close()
            if "already connected" not in str(exc) or retries >= 500:
                raise
            retries += 1
            await asyncio.sleep(0.01)
            continue
        client.retries = retries
        return client


async def serve_session(socket: Path, sid: str, n: int) -> dict[str, Any]:
    out: dict[str, Any] = {}

    def record(name: str, elapsed: float, size: int, **extra: Any) -> None:
        item = out.setdefault(name, {"times": [], "frames": []})
        item["times"].append((elapsed, load1()))
        item["frames"].append(size)
        item.update(extra)

    client = await connect(socket)
    _, elapsed, size = await client.request("resume", {"session_id": sid})
    record("resume", elapsed, size)
    for _ in range(n):
        _, elapsed, size = await client.request("status", {})
        record("status", elapsed, size)
    pages: list[tuple[int, int]] = []
    offset: int | None = 0
    walk_started = time.perf_counter_ns()
    while offset is not None:
        result, _, _ = await client.request(
            "session_history", {"session_id": sid, "offset": offset}
        )
        pages.append((offset, len(result["messages"])))
        offset = result.get("next_offset")
    out["forward_walk"] = {
        "pages": len(pages),
        "ms_total": round((time.perf_counter_ns() - walk_started) / 1e6, 3),
        "messages_per_page_head": [count for _, count in pages[:4]],
        "messages_per_page_tail": [count for _, count in pages[-3:]],
    }
    picks = {
        "first": pages[0][0],
        "middle": pages[len(pages) // 2][0],
        "last": pages[-1][0],
    }
    for name, start in picks.items():
        for _ in range(n):
            result, elapsed, size = await client.request(
                "session_history", {"session_id": sid, "offset": start}
            )
            record(
                f"history_forward_{name}",
                elapsed,
                size,
                offset=start,
                messages=len(result["messages"]),
            )
    for _ in range(n):
        result, elapsed, size = await client.request(
            "session_history", {"session_id": sid, "before": None}
        )
        record(
            "history_backward_first", elapsed, size, messages=len(result["messages"])
        )
    validator = result.get("history_validator")
    before = result.get("next_before")
    for _ in range(n):
        if before is None:
            break
        page, elapsed, size = await client.request(
            "session_history",
            {"session_id": sid, "before": before, "history_validator": validator},
        )
        record("history_backward_next", elapsed, size, messages=len(page["messages"]))
        before = page.get("next_before")
    client.writer.close()
    await client.writer.wait_closed()
    cursor = picks["last"]
    for _ in range(n):
        started = time.perf_counter_ns()
        again = await connect(socket)
        await again.request("status", {})
        _, _, size = await again.request(
            "session_history", {"session_id": sid, "offset": cursor}
        )
        record(
            "reconnect_hello_status_history",
            (time.perf_counter_ns() - started) / 1e6,
            size,
            cursor=cursor,
            hello_retries_last=again.retries,
        )
        again.writer.close()
        await again.writer.wait_closed()
    for item in out.values():
        if "times" in item:
            item.update(summary(item.pop("times")))
            item["frame_bytes_p50"] = int(statistics.median(item.pop("frames")))
    return out


def probe_serve(args: argparse.Namespace) -> dict[str, Any]:
    """Real `python -m zeta serve` subprocess for timings; an in-process
    ZetaServer over the same socket protocol for op counts per request."""
    home, sid = make_home(args.session, 1.0)
    out: dict[str, Any] = {"scale": scale(home, sid)}
    box = store_probes.BOX
    assert box is not None and box.stub is not None
    socket = box.root / "serve.sock"
    env = box.env(home, PYTHONPATH=f"{box.site}{os.pathsep}{WORKTREE / 'src'}")
    seen = len(box.stub.requests)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / f"serve-stderr-{args.session}.log", "w") as stderr:
        process = box.spawn(
            [
                sys.executable,
                "-c",
                # The `zeta` console script entry point (zeta.cli.main:main).
                "import sys; from zeta.cli.main import main; sys.exit(main())",
                "serve",
                "--socket",
                str(socket),
                "--provider",
                STUB_PROVIDER,
                "--model",
                STUB_MODEL,
                "--no-auto-memory",
                "--approval-reconnect-grace",
                "0",
            ],
            env=env,
            cwd=WORKTREE,
            stdout=subprocess.DEVNULL,
            stderr=stderr,
        )
        try:
            deadline = time.monotonic() + 90
            while not socket.exists():
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("serve did not start; see serve-stderr log")
                time.sleep(0.05)
            out["subprocess"] = asyncio.run(serve_session(socket, sid, args.n))
            out["subprocess_env"] = "PYTHONPATH=" + env["PYTHONPATH"]
        finally:
            try:
                box.stop(process)
            finally:
                socket.unlink(missing_ok=True)
    # Requests a resume-started wake turn sent to the stub (never a real provider).
    out["subprocess_stub_requests"] = len(box.stub.requests) - seen

    from zeta.server import ZetaServer
    from zeta.server.handlers import history_projection

    async def in_process() -> dict[str, Any]:
        server = await asyncio.to_thread(
            lambda: ZetaServer(
                home=home,
                socket_path=socket,
                provider=STUB_PROVIDER,
                model=STUB_MODEL,
                auto_memory=False,
                approval_reconnect_grace=0,
            )
        )
        result: dict[str, Any] = {}
        try:
            await server.start()
            client = await connect(socket)
            with OPS.measure() as ops:
                await client.request("resume", {"session_id": sid})
            result["resume_ops"] = ops
            store = server.runtime.opened.store
            for name, params in (
                ("history_forward_first_cold", {"session_id": sid, "offset": 0}),
                ("history_forward_first_cached", {"session_id": sid, "offset": 0}),
                ("history_backward_first", {"session_id": sid, "before": None}),
            ):
                with OPS.measure() as ops:
                    await client.request("session_history", params)
                result[name + "_ops"] = ops
            store.append_message(user_message("serve probe append"))
            with OPS.measure() as ops:
                started = time.perf_counter_ns()
                await client.request(
                    "session_history", {"session_id": sid, "before": None}
                )
                ops["ms_while_counting"] = round(
                    (time.perf_counter_ns() - started) / 1e6, 3
                )
            result["history_after_one_append_ops"] = ops
            with OPS.measure() as ops:
                await client.request("status", {})
            result["status_ops"] = ops
            client.writer.close()
            store.append_message(user_message("profile append"))
            result["history_projection_profile"] = profile(
                f"serve-history-projection-{args.session}",
                lambda: history_projection.history_projection(
                    store, fidelity=True, served_turn_end=True
                ),
            )
        finally:
            try:
                await server.close()
            finally:
                socket.unlink(missing_ok=True)
        return result

    out["in_process_ops"] = asyncio.run(in_process())
    reset_home(args.session)
    return out


# ---------------------------------------------- 7. search and memory prep


def probe_search(args: argparse.Namespace) -> dict[str, Any]:
    """TranscriptIndex.search / .append (transcript_search/index.py) and the
    memory reconcile prep: AutoMemoryReconciler._transcript_state,
    _transcript_chunk, ProjectRegistry._entry_memory_state, and
    entry_reconciler._prepare_request. The invoke (provider) is never called."""
    from datetime import UTC, datetime

    from zeta.memory.auto import AutoMemoryReconciler
    from zeta.memory.entry_reconciler import _prepare_request
    from zeta.memory.reconciler import Transcript
    from zeta.project_registry import ProjectRegistry
    from zeta.transcript_search.index import TranscriptIndex, TranscriptSource

    home, sid = make_home(args.session, 1.0)
    session_dir = home / "sessions" / sid
    project_id = _meta(session_dir)["project_id"]
    project_dir = home / "projects" / project_id
    out: dict[str, Any] = {"scale": scale(home, sid), "project_id": project_id}
    index = TranscriptIndex(project_dir, project_id)
    out["index_bytes"] = index.path.stat().st_size
    for query in (
        "eviction marker",
        "merge lock",
        "serve resume history",
        "flaky test macos",
    ):
        out[f"search[{query}]"], hits = sample(
            lambda q=query: index.search(q, limit=10), args.n
        )
        out[f"search[{query}]"]["hits"] = len(hits)
    out["search_ops"] = count_ops(lambda: index.search("eviction marker", limit=10))
    source = TranscriptSource(sid, session_dir, project_id)
    started = time.perf_counter_ns()
    with OPS.measure() as ops:
        status = index.append(source)
    out["index_append_first"] = {
        "ms_while_counting": round((time.perf_counter_ns() - started) / 1e6, 3),
        "ops": ops,
        "status": str(status)[:300],
    }
    out["index_append_steady"], _ = sample(lambda: index.append(source), args.n)
    out["index_append_steady_ops"] = count_ops(lambda: index.append(source))

    registry = ProjectRegistry(home / "projects")

    async def stub_invoke(
        prompt: str,
    ) -> str:  # the provider boundary; never awaited here
        raise RuntimeError("provider stub")

    reconciler = AutoMemoryReconciler(
        registry=registry,
        project_id=project_id,
        session_id=sid,
        session_dir=session_dir,
        invoke=stub_invoke,
    )
    out["last_reconciled_seq"] = reconciler.last_reconciled_seq

    def cold_state() -> Any:
        reconciler._transcript_cursor = None
        return reconciler._transcript_state()

    out["transcript_state_cold"], state = sample(cold_state, args.n)
    out["transcript_state_cold_ops"] = count_ops(cold_state)
    out["transcript_state_warm"], _ = sample(reconciler._transcript_state, args.n)
    out["transcript_state_warm_ops"] = count_ops(reconciler._transcript_state)
    latest = state[0]
    start = max(1, min(reconciler.last_reconciled_seq + 1, latest))
    out["chunk_range"] = [start, latest]
    out["transcript_chunk"], chunk = sample(
        lambda: reconciler._transcript_chunk(start, latest), args.n
    )
    out["transcript_chunk_ops"] = count_ops(
        lambda: reconciler._transcript_chunk(start, latest)
    )
    out["chunk_rows"] = len(chunk[0])
    reconciler._transcript_cursor = None
    out["transcript_chunk_no_cursor_ops"] = count_ops(
        lambda: reconciler._transcript_chunk(start, latest)
    )
    try:
        out["memory_state"], snapshot = sample(
            lambda: registry._entry_memory_state(project_id), args.n
        )
        out["memory_state_ops"] = count_ops(
            lambda: registry._entry_memory_state(project_id)
        )
        rows = chunk[0] or [{"seq": latest, "type": "message"}]
        transcript = Transcript(sid, tuple(rows))
        today = datetime.now(UTC).date()
        out["prepare_request"], request = sample(
            lambda: _prepare_request(transcript, snapshot.state, as_of=today), args.n
        )
        out["prompt_bytes"] = len(request.prompt.encode())
        out["prepare_request_profile"] = profile(
            f"memory-prepare-request-{args.session}",
            lambda: _prepare_request(transcript, snapshot.state, as_of=today),
        )
        out["transcript_state_cold_profile"] = profile(
            f"memory-transcript-state-cold-{args.session}", cold_state
        )
    except Exception as exc:  # noqa: BLE001 - report the failure instead of hiding it
        out["memory_prep_error"] = repr(exc)[:400]
    return out


MORE_PROBES: dict[str, Callable[[argparse.Namespace], dict[str, Any]]] = {
    "evict": probe_evict,
    "context": probe_context,
    "children": probe_children,
    "serve": probe_serve,
    "search": probe_search,
}
