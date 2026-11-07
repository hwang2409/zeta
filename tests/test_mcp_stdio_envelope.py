from __future__ import annotations

import asyncio
import gc
import inspect
import io
import json
import sys
import threading
from pathlib import Path

import pytest

import zeta.mcp.server_actor as server_actor_module
import zeta.mcp.stdio as stdio_module
from zeta.core.abort import AbortSignal
from zeta.mcp import MCPConfig, MCPServerConfig, StdioMCPClient
from zeta.mcp.client import canceled_result, make_error_result
from zeta.mcp.connection import build_mcp_client
from zeta.mcp.envelope import MCPCallEnvelope, MCPCallLimits
from zeta.mcp.mount import mount_mcp_servers
from zeta.skills import SkillCatalog
from zeta.tools._spill import SpillStore
from zeta.tools.registry import ToolRegistry

LIMIT = 512
LIMITS = MCPCallLimits(max_wire_bytes=LIMIT, spill_after_bytes=64)


def _client(
    name: str, source: str, *, spill_store: SpillStore | None = None
) -> StdioMCPClient:
    config = MCPServerConfig(name, "stdio", sys.executable, ("-u", "-c", source))
    kwargs: dict[str, object] = {"spill_store": spill_store}
    if "limits" in inspect.signature(StdioMCPClient).parameters:
        kwargs["limits"] = LIMITS
    client = StdioMCPClient(config, **kwargs)
    if not hasattr(client, "_limits"):
        client._limits = LIMITS
    return client


def _server_source(body: str) -> str:
    return f"""
import json
import sys
calls = []
first_batch = True
for line in sys.stdin:
    request = json.loads(line)
    method = request.get("method")
    if method == "notifications/initialized":
        continue
    if method == "initialize":
        result = {{"protocolVersion": "2025-06-18", "capabilities": {{"tools": {{}}}}}}
    elif method == "tools/list":
        result = {{"tools": [{{"name": "echo", "description": "", "inputSchema": {{"type": "object"}}}}]}}
    elif method == "tools/call":
        calls.append(request)
{body}
        continue
    else:
        result = {{}}
    print(json.dumps({{"jsonrpc": "2.0", "id": request["id"], "result": result}}, separators=(",", ":")), flush=True)
"""


@pytest.mark.asyncio
async def test_constructed_stdio_client_uses_resolved_wire_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WIKI_AGENT_RUNTIME_DIR", str(tmp_path))
    one_mib = 1024 * 1024
    limits = MCPCallLimits(max_wire_bytes=one_mib, spill_after_bytes=64 * 1024)
    registry = ToolRegistry(
        tmp_path,
        register_builtin=False,
        skill_catalog=SkillCatalog.empty(),
        mcp_limits=limits,
    )
    source = _server_source(
        f"""        value = {{"jsonrpc": "2.0", "id": request["id"], "result": {{"content": [{{"type": "text", "text": "x" * {one_mib}}}]}}}}
        print(json.dumps(value, separators=(",", ":")), flush=True)"""
    )
    client = build_mcp_client(
        MCPServerConfig("configured", "stdio", sys.executable, ("-u", "-c", source)),
        registry=registry,
    )
    try:
        await client.connect()
        result = await client.call_tool("echo", {}, AbortSignal())
        assert result == make_error_result(
            "MCP tool configured__echo exceeded the 1 MiB response limit"
        )
    finally:
        await client.close()
        await registry.close()


def _sized_frame_server_source(frame_size: int, *, terminated: bool) -> str:
    return _server_source(
        f"""        value = {{"jsonrpc": "2.0", "id": request["id"], "result": {{"content": [{{"type": "text", "text": ""}}]}}}}
        encoded = json.dumps(value, separators=(",", ":")).encode()
        padding = {frame_size} - len(encoded) - {int(terminated)}
        assert padding >= 0
        value["result"]["content"][0]["text"] = "x" * padding
        encoded = json.dumps(value, separators=(",", ":")).encode()
        assert len(encoded) + {int(terminated)} == {frame_size}
        sys.stdout.buffer.write(encoded + (b"\\n" if {terminated!r} else b""))
        sys.stdout.buffer.flush()"""
    )


@pytest.mark.asyncio
async def test_concurrent_stdio_deadlines_share_invalidation_and_reconnect_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WIKI_AGENT_RUNTIME_DIR", str(tmp_path))
    launches = tmp_path / "launches"
    source = f"""
import json
import pathlib
import sys
path = pathlib.Path({str(launches)!r})
launch = int(path.read_text()) + 1 if path.exists() else 1
path.write_text(str(launch))
for line in sys.stdin:
    request = json.loads(line)
    method = request.get("method")
    if method == "notifications/initialized":
        continue
    if method == "initialize":
        result = {{"protocolVersion": "2025-06-18", "capabilities": {{"tools": {{}}}}}}
    elif method == "tools/list":
        result = {{"tools": [{{"name": "echo", "description": "", "inputSchema": {{"type": "object"}}}}]}}
    elif method == "tools/call" and launch == 1:
        note = {{"jsonrpc": "2.0", "method": "notifications/progress", "params": {{}}}}
        while True:
            print(json.dumps(note, separators=(",", ":")), flush=True)
    elif method == "tools/call":
        result = {{"content": [{{"type": "text", "text": "ok"}}]}}
    else:
        result = {{}}
    print(json.dumps({{"jsonrpc": "2.0", "id": request["id"], "result": result}}, separators=(",", ":")), flush=True)
"""
    limits = MCPCallLimits(
        deadline_seconds=300.0,
        max_wire_bytes=64 * 1024 * 1024,
        spill_after_bytes=64 * 1024,
    )
    registry = ToolRegistry(
        tmp_path,
        register_builtin=False,
        skill_catalog=SkillCatalog.empty(),
        tool_allow=("deadline__echo",),
        mcp_limits=limits,
    )
    config = MCPServerConfig("deadline", "stdio", sys.executable, ("-u", "-c", source))
    kill_calls = 0
    real_kill = stdio_module._kill_and_reap

    async def counted_kill(process, tasks) -> None:
        nonlocal kill_calls
        kill_calls += 1
        await real_kill(process, tasks)

    monkeypatch.setattr(stdio_module, "_kill_and_reap", counted_kill)
    notification_seen = asyncio.Event()
    abort_on_base: asyncio.Task[None] | None = None
    signals = (AbortSignal(), AbortSignal())
    deadline_type = getattr(server_actor_module, "MCPCallDeadline", None)
    clock_offset = 0.0
    loop = asyncio.get_running_loop()

    def clock() -> float:
        return loop.time() + clock_offset

    if deadline_type is not None:
        real_wait = deadline_type.wait
        deadline_waits = 0

        async def wait_for_notification_deadline(deadline) -> None:
            nonlocal deadline_waits, clock_offset
            deadline_waits += 1
            if deadline_waits <= 2:
                await notification_seen.wait()
                clock_offset = 301.0
                return
            await real_wait(deadline)

        monkeypatch.setattr(deadline_type, "wait", wait_for_notification_deadline)
    else:

        async def abort_calls_on_base() -> None:
            await notification_seen.wait()
            for signal in signals:
                signal.abort()

        abort_on_base = asyncio.create_task(abort_calls_on_base())
    mount = await mount_mcp_servers(
        registry,
        MCPConfig(tmp_path / "mcp.json", {"deadline": config}),
    )
    expected = make_error_result("MCP tool deadline__echo exceeded the 300s deadline")
    if deadline_type is not None:
        mount._actors["deadline"]._deadline_clock = clock
    try:
        mount._clients["deadline"].set_notification_sink(
            lambda _method: notification_seen.set()
        )
        results = await asyncio.gather(
            mount.call_tool("deadline", "echo", {"call": 1}, signals[0]),
            mount.call_tool("deadline", "echo", {"call": 2}, signals[1]),
        )
        assert results == [expected, expected]
        client = mount._clients["deadline"]
        assert client._invalidation is not None
        await asyncio.shield(client._invalidation.task)
        clock_offset = 0.0
        follow_up = await mount.call_tool("deadline", "echo", {})
        assert follow_up["content"][0]["text"] == "ok"
        assert mount.statuses["deadline"].state == "mounted"
        assert launches.read_text() == "2"
        assert kill_calls == 1
    finally:
        if abort_on_base is not None:
            await abort_on_base
        await mount.close()
        await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("frame_size", "terminated", "over_limit"),
    [
        pytest.param(LIMIT, True, False, id="exact-limit"),
        pytest.param(LIMIT + 1, True, True, id="terminated-over-limit"),
        pytest.param(LIMIT + 1, False, True, id="unterminated-over-limit"),
    ],
)
async def test_stdio_prebuffer_frame_cap_is_hard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frame_size: int,
    terminated: bool,
    over_limit: bool,
) -> None:
    monkeypatch.setenv("WIKI_AGENT_RUNTIME_DIR", str(tmp_path))
    spill = SpillStore()
    client = _client(
        "frame-cap",
        _sized_frame_server_source(frame_size, terminated=terminated),
        spill_store=spill,
    )
    retained: list[tuple[int, int]] = []
    real_append = MCPCallEnvelope.append

    async def observe_append(envelope: MCPCallEnvelope, chunk: bytes) -> None:
        try:
            await real_append(envelope, chunk)
        finally:
            retained.append((envelope.total, spill._transient_bytes))

    monkeypatch.setattr(MCPCallEnvelope, "append", observe_append)
    try:
        await client.connect()
        result = await client.call_tool("echo", {}, AbortSignal())
        if over_limit:
            assert result == make_error_result(
                f"MCP tool frame-cap__echo exceeded the {LIMIT} bytes response limit"
            )
        else:
            assert result["isError"] is False
            assert result["content"][0]["text"]
        assert retained
        assert max(total for total, _ in retained) <= LIMIT
        assert max(transient for _, transient in retained) <= LIMIT
    finally:
        await client.close()
        spill.close()


@pytest.mark.asyncio
async def test_stdio_valid_frames_charge_only_the_matching_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WIKI_AGENT_RUNTIME_DIR", str(tmp_path))
    source = _server_source(
        """        if len(calls) < 2:
            continue
        note = {"jsonrpc": "2.0", "method": "notifications/progress", "params": {"padding": "x" * 100}}
        print(json.dumps(note, separators=(",", ":")), flush=True)
        first, second = calls
        value = {"jsonrpc": "2.0", "id": first["id"], "result": {"content": [{"type": "text", "text": "x" * 300}]}}
        print(json.dumps(value, separators=(",", ":")), flush=True)
        value = {"jsonrpc": "2.0", "id": second["id"], "result": {"content": [{"type": "text", "text": "ok"}]}}
        print(json.dumps(value, separators=(",", ":")), flush=True)"""
    )
    client = _client("owned", source)
    try:
        await client.connect()
        results = await asyncio.gather(
            client.call_tool("echo", {"n": 1}, AbortSignal()),
            client.call_tool("echo", {"n": 2}, AbortSignal()),
        )
        assert results == [
            make_error_result(
                f"MCP tool owned__echo exceeded the {LIMIT} bytes response limit"
            ),
            make_error_result(
                "MCP stdio server owned restarted after a call limit; retry the tool"
            ),
        ]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_stdio_unowned_notification_flood_charges_every_pending_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WIKI_AGENT_RUNTIME_DIR", str(tmp_path))
    source = _server_source(
        """        if len(calls) < 2:
            continue
        note = {"jsonrpc": "2.0", "method": "notifications/progress", "params": {"padding": "x" * 120}}
        for _ in range(4):
            print(json.dumps(note, separators=(",", ":")), flush=True)
        for saved in calls:
            value = {"jsonrpc": "2.0", "id": saved["id"], "result": {"content": [{"type": "text", "text": "unbounded"}]}}
            print(json.dumps(value, separators=(",", ":")), flush=True)"""
    )
    client = _client("notifications", source)
    try:
        await client.connect()
        results = await asyncio.gather(
            client.call_tool("echo", {"n": 1}, AbortSignal()),
            client.call_tool("echo", {"n": 2}, AbortSignal()),
        )
        expected = make_error_result(
            f"MCP tool notifications__echo exceeded the {LIMIT} bytes response limit"
        )
        assert results == [expected, expected]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_stdio_abort_records_initiating_and_displaced_results_synchronously(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WIKI_AGENT_RUNTIME_DIR", str(tmp_path))
    source = _server_source(
        """        if len(calls) < 3:
            continue"""
    ).replace(
        '    if method == "notifications/initialized":\n        continue',
        '    if method == "notifications/initialized":\n        continue\n'
        '    if method == "notifications/cancelled":\n        sys.exit(0)',
    )
    client = _client("abort", source)
    writes_complete = asyncio.Event()
    write_count = 0
    write_request = client._write_request

    async def observe_write(*args) -> None:
        nonlocal write_count
        await write_request(*args)
        request = args[1]
        if request.get("method") == "tools/call":
            write_count += 1
            if write_count == 3:
                writes_complete.set()

    monkeypatch.setattr(client, "_write_request", observe_write)
    first_signal = AbortSignal()
    second_signal = AbortSignal()
    try:
        await client.connect()
        first = asyncio.create_task(client.call_tool("echo", {"n": 1}, first_signal))
        second = asyncio.create_task(client.call_tool("echo", {"n": 2}, second_signal))
        displaced = asyncio.create_task(
            client.call_tool("echo", {"n": 3}, AbortSignal())
        )
        await writes_complete.wait()
        first_signal.abort()
        second_signal.abort()
        results = await asyncio.gather(first, second, displaced)
        assert results[:2] == [canceled_result(), canceled_result()]
        assert results[2] == make_error_result(
            "MCP stdio server abort restarted after a call limit; retry the tool"
        )
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_simultaneous_stdio_overflows_invalidate_once_and_detach_before_await(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WIKI_AGENT_RUNTIME_DIR", str(tmp_path))
    source = _server_source(
        """        if len(calls) < 2:
            continue
        note = {"jsonrpc": "2.0", "method": "notifications/progress", "params": {"padding": "x" * 120}}
        for _ in range(4):
            print(json.dumps(note, separators=(",", ":")), flush=True)
        for saved in calls:
            value = {"jsonrpc": "2.0", "id": saved["id"], "result": {"content": [{"type": "text", "text": "unbounded"}]}}
            print(json.dumps(value, separators=(",", ":")), flush=True)"""
    )
    spill = SpillStore()
    client = _client("single", source, spill_store=spill)
    failures: list[str] = []
    client.set_failure_sink(failures.append)
    real_kill = stdio_module._kill_and_reap
    kill_calls = 0
    detached_before_cleanup: list[bool] = []

    async def checked_kill(process, tasks) -> None:
        nonlocal kill_calls
        kill_calls += 1
        detached_before_cleanup.append(
            client._process is None
            and client._reader_task is None
            and client._stderr_task is None
            and getattr(client, "_active_accumulator", None) is None
        )
        await real_kill(process, tasks)

    monkeypatch.setattr(stdio_module, "_kill_and_reap", checked_kill)
    try:
        await client.connect()
        results = await asyncio.gather(
            client.call_tool("echo", {"n": 1}, AbortSignal()),
            client.call_tool("echo", {"n": 2}, AbortSignal()),
        )
        expected = make_error_result(
            f"MCP tool single__echo exceeded the {LIMIT} bytes response limit"
        )
        assert results == [expected, expected]
        assert kill_calls == 1
        assert detached_before_cleanup == [True]
        assert len(failures) == 1
        assert client._pending == {}
        assert spill._transient_bytes == 0
    finally:
        await client.close()
        spill.close()


@pytest.mark.asyncio
async def test_valid_large_stdio_frame_parse_allows_event_loop_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spill = SpillStore()
    client = _client("parse", "unused", spill_store=spill)
    stream = asyncio.StreamReader()
    frame = (
        json.dumps(
            {"jsonrpc": "2.0", "id": 1, "result": {"padding": "x" * 200}}
        ).encode()
        + b"\n"
    )
    stream.feed_data(frame)
    stream.feed_eof()
    loop = asyncio.get_running_loop()
    heartbeat = threading.Event()
    parse_threads: list[int] = []
    main_thread = threading.get_ident()
    real_load = json.load

    def observed_load(handle):
        parse_threads.append(threading.get_ident())
        if threading.get_ident() != main_thread:
            loop.call_soon_threadsafe(heartbeat.set)
            heartbeat.wait()
        return real_load(handle)

    monkeypatch.setattr(stdio_module.json, "load", observed_load)
    try:
        frames = [item async for item in client._read_messages(stream)]
        assert len(frames) == 1
        assert heartbeat.is_set()
        assert parse_threads != [main_thread]
    finally:
        await client.close()
        spill.close()


class _OneByteReader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._offset = 0

    async def read(self, _size: int) -> bytes:
        if self._offset == len(self._data):
            return b""
        chunk = self._data[self._offset : self._offset + 1]
        self._offset += 1
        return chunk


class _CountingHandle(io.BytesIO):
    def __init__(self) -> None:
        super().__init__()
        self.write_count = 0

    def write(self, value: bytes) -> int:
        self.write_count += 1
        return super().write(value)


class _CountingTemporary:
    def __init__(self) -> None:
        self.handle = _CountingHandle()

    def __enter__(self) -> _CountingHandle:
        return self.handle

    def __exit__(self, *_args) -> None:
        pass


@pytest.mark.asyncio
async def test_tiny_stdio_reads_keep_spill_writes_and_frame_accounting_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spill = SpillStore()
    temporary = _CountingTemporary()
    monkeypatch.setattr(spill, "temporary_file", lambda: temporary)
    client = _client("tiny-reads", "unused", spill_store=spill)
    data = (
        json.dumps(
            {"jsonrpc": "2.0", "method": "notifications/progress", "padding": "x" * 200}
        ).encode()
        + b"\n"
    )

    try:
        frames = [frame async for frame in client._read_messages(_OneByteReader(data))]
        assert temporary.handle.write_count == 1
        assert len(frames) == 1
        assert frames[0].total_bytes == len(data)
        assert frames[0].bytes_by_pending_id == {}
    finally:
        await client.close()
        spill.close()


@pytest.mark.asyncio
async def test_cancel_during_stdio_invalidation_observes_pending_future_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WIKI_AGENT_RUNTIME_DIR", str(tmp_path))
    loop = asyncio.get_running_loop()
    unhandled: list[dict[str, object]] = []
    old_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))

    async def cancel_during_cleanup() -> None:
        client = _client(
            "cancel-cleanup",
            _server_source("""        continue"""),
        )
        request_written = asyncio.Event()
        cleanup_entered = asyncio.Event()
        allow_cleanup = asyncio.Event()
        write_request = client._write_request
        real_kill = stdio_module._kill_and_reap

        async def observe_write(*args) -> None:
            await write_request(*args)
            if args[1].get("method") == "tools/call":
                request_written.set()

        async def blocked_cleanup(process, tasks) -> None:
            cleanup_entered.set()
            await allow_cleanup.wait()
            await real_kill(process, tasks)

        client._write_request = observe_write
        stdio_module._kill_and_reap = blocked_cleanup
        try:
            await client.connect()
            signal = AbortSignal()
            call = asyncio.create_task(client.call_tool("echo", {}, signal))
            await request_written.wait()
            signal.abort()
            await cleanup_entered.wait()
            call.cancel()
            allow_cleanup.set()
            with pytest.raises(asyncio.CancelledError):
                await call
        finally:
            allow_cleanup.set()
            await client.close()
            stdio_module._kill_and_reap = real_kill
            del client._write_request

    try:
        await cancel_during_cleanup()
        callbacks_drained = loop.create_future()
        loop.call_soon(callbacks_drained.set_result, None)
        await callbacks_drained
        gc.collect()
        assert unhandled == []
    finally:
        loop.set_exception_handler(old_handler)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["kill", "reap"])
async def test_stdio_cleanup_cancellation_releases_transient_quota(
    stage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    spill = SpillStore()
    client = _client("cancel-cleanup", "unused", spill_store=spill)
    envelope = MCPCallEnvelope(LIMITS, spill)
    await envelope.append(b"x" * 100)
    client._active_accumulator = envelope
    client._process = object()
    kill_entered = asyncio.Event()
    allow_kill = asyncio.Event()
    reap_entered = asyncio.Event()
    allow_reap = asyncio.Event()

    async def blocked_cleanup(_process, _tasks) -> None:
        kill_entered.set()
        await allow_kill.wait()
        reap_entered.set()
        await allow_reap.wait()

    monkeypatch.setattr(stdio_module, "_kill_and_reap", blocked_cleanup)
    close_task = asyncio.create_task(client.close())
    await kill_entered.wait()
    if stage == "reap":
        allow_kill.set()
        await reap_entered.wait()
    invalidation = getattr(client, "_invalidation", None)
    cleanup_task = invalidation.task if invalidation is not None else None

    close_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await close_task
    allow_kill.set()
    allow_reap.set()
    if cleanup_task is not None:
        await cleanup_task
    assert spill._transient_bytes == 0
    spill.close()


@pytest.mark.asyncio
async def test_late_old_generation_frame_cannot_resolve_new_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Stdin:
        def __init__(self) -> None:
            self.drained = asyncio.Event()

        def write(self, _value: bytes) -> None:
            pass

        async def drain(self) -> None:
            self.drained.set()

    class Process:
        def __init__(self, stdout: asyncio.StreamReader) -> None:
            self.stdin = Stdin()
            self.stdout = stdout
            self.returncode = None

        async def wait(self) -> None:
            pass

    async def cleanup(_process, _tasks) -> None:
        pass

    monkeypatch.setattr(stdio_module, "_kill_and_reap", cleanup)
    client = _client("generation", "unused")
    old_stream = asyncio.StreamReader()
    client._process = Process(old_stream)
    reader_started = asyncio.Event()
    read_messages = client._read_messages

    async def observe_reader(*args):
        reader_started.set()
        async for frame in read_messages(*args):
            yield frame

    monkeypatch.setattr(client, "_read_messages", observe_reader)
    read_stdout = client._read_stdout
    if "generation" in inspect.signature(read_stdout).parameters:
        old_reader = asyncio.create_task(read_stdout(client._generation))
        await reader_started.wait()
        client._generation += 1
    else:
        old_reader = asyncio.create_task(read_stdout())
        await reader_started.wait()

    new_stream = asyncio.StreamReader()
    new_process = Process(new_stream)
    client._process = new_process
    request = asyncio.create_task(client._request("tools/call", {"name": "echo"}))
    await new_process.stdin.drained.wait()
    old_stream.feed_data(
        json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}).encode() + b"\n"
    )
    old_stream.feed_eof()
    await old_reader
    assert not request.done()

    await client.close()
    with pytest.raises(stdio_module.MCPTransportError):
        await request
