from __future__ import annotations

import asyncio
import inspect
import io
import json
import sys
import threading
from pathlib import Path

import pytest

import zeta.mcp.stdio as stdio_module
from zeta.core.abort import AbortSignal
from zeta.mcp import MCPServerConfig, StdioMCPClient
from zeta.mcp.actors.prompt import CallFinished, CallRequest
from zeta.mcp.client import canceled_result, make_error_result
from zeta.mcp.envelope import MCPCallEnvelope, MCPCallLimits
from zeta.mcp.server_actor import (
    MCPServerActor,
    MCPServerStatus,
    _TransportFailure,
)
from zeta.tools._spill import SpillStore

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
async def test_stdio_prebuffer_frame_cap_is_hard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WIKI_AGENT_RUNTIME_DIR", str(tmp_path))
    body = '''        value = {"jsonrpc": "2.0", "id": request["id"], "result": {"content": [{"type": "text", "text": "x" * FRAME_LIMIT}]}}
        print(json.dumps(value, separators=(",", ":")), flush=True)'''
    source = _server_source(body.replace("FRAME_LIMIT", str(LIMIT)))
    client = _client("frame-cap", source)
    try:
        await client.connect()
        result = await client.call_tool("echo", {}, AbortSignal())
        assert result == make_error_result(
            f"MCP tool frame-cap__echo exceeded the {LIMIT} response limit"
        )
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_stdio_valid_frames_charge_only_the_matching_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WIKI_AGENT_RUNTIME_DIR", str(tmp_path))
    source = _server_source(
        '''        if first_batch:
            if len(calls) < 3:
                continue
            batch = calls[:2]
            calls[:] = calls[2:]
            first_batch = False
        else:
            if len(calls) < 2:
                continue
            batch = list(calls)
            calls.clear()
        for saved in batch:
            value = {"jsonrpc": "2.0", "id": saved["id"], "result": {"content": [{"type": "text", "text": "x" * 300}]}}
            print(json.dumps(value, separators=(",", ":")), flush=True)'''
    )
    client = _client("owned", source)
    try:
        await client.connect()
        first, second, innocent = (
            asyncio.create_task(client.call_tool("echo", {"n": index}, AbortSignal()))
            for index in range(3)
        )
        first_results = await asyncio.gather(first, second)
        assert [result["content"][0]["text"] for result in first_results] == [
            "x" * 300,
            "x" * 300,
        ]
        pending = next(iter(client._pending.values()))
        assert getattr(pending, "wire_bytes", -1) == 0

        fourth = asyncio.create_task(
            client.call_tool("echo", {"n": 4}, AbortSignal())
        )
        later_results = await asyncio.gather(innocent, fourth)
        assert [result["content"][0]["text"] for result in later_results] == [
            "x" * 300,
            "x" * 300,
        ]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_stdio_unowned_notification_flood_charges_every_pending_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WIKI_AGENT_RUNTIME_DIR", str(tmp_path))
    source = _server_source(
        '''        if len(calls) < 2:
            continue
        note = {"jsonrpc": "2.0", "method": "notifications/progress", "params": {"padding": "x" * 120}}
        for _ in range(4):
            print(json.dumps(note, separators=(",", ":")), flush=True)
        for saved in calls:
            value = {"jsonrpc": "2.0", "id": saved["id"], "result": {"content": [{"type": "text", "text": "unbounded"}]}}
            print(json.dumps(value, separators=(",", ":")), flush=True)'''
    )
    client = _client("notifications", source)
    try:
        await client.connect()
        results = await asyncio.gather(
            client.call_tool("echo", {"n": 1}, AbortSignal()),
            client.call_tool("echo", {"n": 2}, AbortSignal()),
        )
        expected = make_error_result(
            f"MCP tool notifications__echo exceeded the {LIMIT} response limit"
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
        '''        if len(calls) < 3:
            continue'''
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
        second = asyncio.create_task(
            client.call_tool("echo", {"n": 2}, second_signal)
        )
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
        '''        if len(calls) < 2:
            continue
        note = {"jsonrpc": "2.0", "method": "notifications/progress", "params": {"padding": "x" * 120}}
        for _ in range(4):
            print(json.dumps(note, separators=(",", ":")), flush=True)
        for saved in calls:
            value = {"jsonrpc": "2.0", "id": saved["id"], "result": {"content": [{"type": "text", "text": "unbounded"}]}}
            print(json.dumps(value, separators=(",", ":")), flush=True)'''
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
            and client._stderr is None
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
            f"MCP tool single__echo exceeded the {LIMIT} response limit"
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
    frame = json.dumps(
        {"jsonrpc": "2.0", "id": 1, "result": {"padding": "x" * 200}}
    ).encode() + b"\n"
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


class _BlockingHandle(io.BytesIO):
    def __init__(
        self, stage: str, entered: threading.Event, release: threading.Event
    ) -> None:
        super().__init__()
        self.stage = stage
        self.entered = entered
        self.release = release

    def write(self, value: bytes) -> int:
        if self.stage == "write":
            self.entered.set()
            self.release.wait()
        return super().write(value)


class _BlockingTemporary:
    def __init__(
        self,
        stage: str,
        entered: threading.Event,
        release: threading.Event,
    ) -> None:
        self.stage = stage
        self.entered = entered
        self.release = release
        self.handle = _BlockingHandle(stage, entered, release)

    def __enter__(self):
        if self.stage == "spill":
            self.entered.set()
            self.release.wait()
        return self.handle

    def __exit__(self, *_args) -> None:
        self.handle.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["spill", "write", "parse"])
async def test_stdio_accumulator_cancellation_releases_transient_quota(
    stage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(stdio_module, "MCP_STDIO_LINE_LIMIT", 64)
    spill = SpillStore()
    client = _client("cancel-accumulator", "unused", spill_store=spill)
    entered = threading.Event()
    release = threading.Event()
    temporary = _BlockingTemporary(stage, entered, release)
    monkeypatch.setattr(spill, "temporary_file", lambda: temporary)
    real_load = json.load

    def blocking_load(handle):
        if stage == "parse":
            entered.set()
            release.wait()
        return real_load(handle)

    monkeypatch.setattr(stdio_module.json, "load", blocking_load)
    stream = asyncio.StreamReader()
    stream.feed_data(
        json.dumps({"jsonrpc": "2.0", "id": 1, "padding": "x" * 200}).encode()
        + b"\n"
    )
    stream.feed_eof()

    async def consume() -> None:
        async for _frame in client._read_messages(stream):
            pass

    task = asyncio.create_task(consume())
    await asyncio.to_thread(entered.wait)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert spill._transient_bytes == 0
    await client.close()
    assert getattr(client, "_generation", -1) == 1
    spill.close()


@pytest.mark.asyncio
async def test_stdio_write_cancellation_releases_transient_quota(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Stdin:
        def __init__(self) -> None:
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        def write(self, _value: bytes) -> None:
            pass

        async def drain(self) -> None:
            self.entered.set()
            await self.release.wait()

    class Process:
        def __init__(self) -> None:
            self.stdin = Stdin()

    spill = SpillStore()
    client = _client("cancel-write", "unused", spill_store=spill)
    envelope = MCPCallEnvelope(LIMITS, spill)
    await envelope.append(b"x" * 100)
    client._active_accumulator = envelope
    process = Process()
    client._process = process

    async def cleanup(_process, _tasks) -> None:
        pass

    monkeypatch.setattr(stdio_module, "_kill_and_reap", cleanup)
    request = asyncio.create_task(client._request("tools/call", {"name": "echo"}))
    await process.stdin.entered.wait()
    request.cancel()
    process.stdin.release.set()
    with pytest.raises(asyncio.CancelledError):
        await request
    invalidation = getattr(client, "_invalidation", None)
    assert invalidation is not None
    assert invalidation.task is not None
    await invalidation.task
    assert spill._transient_bytes == 0
    await client.close()
    spill.close()


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
    invalidation = client._start_invalidation(
        stdio_module.MCPTransportError("test cleanup"), report_failure=False
    )
    assert invalidation.task is not None
    await kill_entered.wait()
    if stage == "reap":
        allow_kill.set()
        await reap_entered.wait()

    async def join_cleanup() -> None:
        assert invalidation.task is not None
        await asyncio.shield(invalidation.task)

    joiner = asyncio.create_task(join_cleanup())
    joiner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await joiner
    allow_kill.set()
    allow_reap.set()
    await invalidation.task
    assert spill._transient_bytes == 0
    await client.close()
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


@pytest.mark.asyncio
async def test_transport_failure_before_call_finished_preserves_typed_invalidation_outcomes(
    tmp_path: Path,
) -> None:
    class Client:
        def __init__(self) -> None:
            self.result = make_error_result(
                f"MCP tool ordered__echo exceeded the {LIMIT} response limit"
            )

        def take_invalidation_result(self, _task):
            result, self.result = self.result, None
            return result

        def set_failure_sink(self, _sink) -> None:
            pass

        async def close(self) -> None:
            pass

    client = Client()

    async def connect_and_list(_client):
        return []

    actor = MCPServerActor(
        MCPServerConfig("ordered", "stdio", "unused"),
        source=tmp_path / "mcp.json",
        registry=None,
        publish=lambda *_args: None,
        build_client=lambda _config: client,
        connect_and_list=connect_and_list,
    )
    actor._client = client
    actor._status = MCPServerStatus("ordered", "stdio", "mounted")
    result = asyncio.get_running_loop().create_future()
    request = CallRequest(1, "echo", {}, AbortSignal(), 0, result, client=client)
    async def completed_call():
        return client.result

    task = asyncio.create_task(completed_call())
    await task
    request.task = task
    actor._requests[1] = request
    actor._children.add(task)

    actor._handle_transport_failure(_TransportFailure(None, client, "bounded failure"))
    actor._handle_call_finished(CallFinished(request, task, client.result, None))

    assert await result == make_error_result(
        f"MCP tool ordered__echo exceeded the {LIMIT} response limit"
    )
    await asyncio.gather(*actor._children, return_exceptions=True)
