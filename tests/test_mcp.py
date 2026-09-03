import asyncio
import importlib
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from zeta.core.abort import AbortSignal
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.store import ConversationStore
from zeta.loop import AgentLoop
from zeta.mcp import (
    MCPConfig,
    MCPConfigError,
    MCPMount,
    MCPServerConfig,
    MCPTool,
    StdioMCPClient,
    StreamableHTTPMCPClient,
    home_config_path,
    load_mcp_config,
    load_mcp_config_overlay,
    mount_mcp_servers,
    project_config_path,
    server_to_json,
    write_mcp_config,
)
from zeta.mcp.client import (
    MCPProtocolError,
    parse_rpc_response,
    translate_call_result,
)
from zeta.tools import ToolRegistry
from zeta.types import TextContent, ToolCall

mount_module = importlib.import_module("zeta.mcp.mount")


def test_mcp_non_text_blocks_keep_the_standard_content_shape() -> None:
    result = translate_call_result(
        {
            "content": [
                {
                    "type": "image",
                    "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNg+M8AAAAEAAEBouDEsAAAAABJRU5ErkJggg==",
                    "mimeType": "image/png",
                },
                {
                    "type": "resource",
                    "resource": {
                        "uri": "file:///tmp/note.txt",
                        "blob": "bm90ZQ==",
                    },
                },
            ],
            "isError": False,
        }
    )

    assert result["content"] == [
        {
            "type": "image",
            "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNg+M8AAAAEAAEBouDEsAAAAABJRU5ErkJggg==",
            "mimeType": "image/png",
        },
        {
            "type": "resource",
            "resource": {
                "uri": "file:///tmp/note.txt",
                "blob": "bm90ZQ==",
            },
        },
    ]


def test_mcp_json_rpc_error_is_a_tool_error() -> None:
    result = translate_call_result(
        parse_rpc_response(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "error": {"code": -1, "message": "tool failed"},
            },
            1,
        )
    )

    assert result["isError"] is True
    assert result["content"][0]["text"] == "tool failed"


def test_mcp_malformed_tool_result_is_a_protocol_error() -> None:
    with pytest.raises(MCPProtocolError, match="content must be an array"):
        translate_call_result({"content": "invalid"})


def _stdio_source() -> str:
    return """
import json
import sys
for line in sys.stdin:
    request = json.loads(line)
    method = request.get("method")
    if method == "notifications/initialized" or method == "notifications/cancelled":
        continue
    if method == "initialize":
        result = {"protocolVersion": "2025-06-18", "capabilities": {}, "serverInfo": {"name": "fake", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "echo", "description": "echo text", "inputSchema": {"type": "object", "title": "EchoInput", "$defs": {"value": {"type": "string"}}, "properties": {"value": {"type": "string", "default": "hello"}}, "required": ["value"]}}]}
    elif method == "tools/call":
        result = {"content": [{"type": "text", "text": request["params"]["arguments"]["value"]}], "isError": False, "structuredContent": {"ok": True, "latency": 1.5}}
    else:
        result = {"content": [], "isError": False}
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
"""


def _stdio_config(name: str = "fake") -> MCPServerConfig:
    return MCPServerConfig(name, "stdio", sys.executable, ("-u", "-c", _stdio_source()))


def _failing_stdio_config(name: str = "fail") -> MCPServerConfig:
    source = _stdio_source().replace(
        '    elif method == "tools/call":\n        result = {"content": [{"type": "text", "text": request["params"]["arguments"]["value"]}], "isError": False, "structuredContent": {"ok": True, "latency": 1.5}}\n',
        '    elif method == "tools/call":\n        print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "error": {"code": -1, "message": "tool failed"}}), flush=True)\n        continue\n',
    )
    return MCPServerConfig(name, "stdio", sys.executable, ("-u", "-c", source))


def _descendant_stdio_config(marker: Path) -> MCPServerConfig:
    child_source = f"import time; time.sleep(0.6); open({str(marker)!r}, 'w').write('alive')"
    source = _stdio_source().replace(
        '        result = {"content": [{"type": "text", "text": request["params"]["arguments"]["value"]}], "isError": False, "structuredContent": {"ok": True, "latency": 1.5}}',
        f'        import subprocess; subprocess.Popen([sys.executable, "-c", {child_source!r}]); sys.exit(0)',
    )
    return MCPServerConfig("descendant", "stdio", sys.executable, ("-u", "-c", source))


class _FakeClient:
    def __init__(self, config: MCPServerConfig, *, fail_connect: bool = False, fail_close: bool = False) -> None:
        self.config = config
        self.protocol_version = "2025-06-18"
        self.fail_connect = fail_connect
        self.fail_close = fail_close

    async def connect(self) -> None:
        if self.fail_connect:
            raise RuntimeError("connect failed")

    async def list_tools(self) -> list[MCPTool]:
        return [MCPTool("echo", "", {"type": "object"})]

    async def call_tool(self, name: str, arguments: dict[str, object], abort_signal: AbortSignal):
        del name, arguments, abort_signal
        return {"content": [], "isError": False, "structuredContent": None}

    async def close(self) -> None:
        if self.fail_close:
            raise RuntimeError("close failed")


class _LifecycleClient(_FakeClient):
    def __init__(self, config: MCPServerConfig) -> None:
        super().__init__(config)
        self._failure_sink: Callable[[str], None] | None = None
        self.closed = False

    def set_failure_sink(self, sink: Callable[[str], None] | None) -> None:
        self._failure_sink = sink

    def fail_transport(self, reason: str) -> None:
        assert self._failure_sink is not None
        self._failure_sink(reason)

    async def close(self) -> None:
        self.closed = True


class _ApplicationErrorClient(_LifecycleClient):
    async def call_tool(
        self, name: str, arguments: dict[str, object], abort_signal: AbortSignal
    ):
        del name, arguments, abort_signal
        return {
            "content": [{"type": "text", "text": "application error"}],
            "isError": True,
            "structuredContent": None,
        }


def test_config_interpolates_and_skips_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"servers": {
        "present": {"transport": "stdio", "command": "${MCP_COMMAND}", "env": {"TOKEN": "${MCP_TOKEN}"}},
        "missing": {"transport": "stdio", "command": "${MCP_MISSING}"},
    }}))
    monkeypatch.setenv("MCP_COMMAND", "server")
    monkeypatch.setenv("MCP_TOKEN", "secret")

    config = load_mcp_config(path)

    assert config.servers["present"].command == "server"
    assert config.servers["present"].env == {"TOKEN": "secret"}
    assert "missing" not in config.servers
    assert config.skipped_servers["missing"].missing_env == ("MCP_MISSING",)
    assert set(config.configured_servers) == {"present", "missing"}


def test_config_records_missing_transport_before_strict_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"servers": {
        "missing": {"transport": "${MCP_TRANSPORT}", "command": "server"},
        "valid": {"transport": "stdio", "command": "server"},
    }}))

    config = load_mcp_config(path)

    assert set(config.servers) == {"valid"}
    assert config.skipped_servers["missing"].missing_env == ("MCP_TRANSPORT",)


def test_config_override_and_malformed_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    override = tmp_path / "override.json"
    override.write_text('{"servers": {}}')
    monkeypatch.setenv("ZETA_MCP_CONFIG", str(override))
    assert load_mcp_config().path == override
    override.write_text("{")
    with pytest.raises(MCPConfigError, match=str(override)):
        load_mcp_config()


@pytest.mark.asyncio
async def test_missing_config_mount_is_a_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZETA_MCP_CONFIG", str(tmp_path / "missing.json"))
    registry = ToolRegistry(tmp_path, register_builtin=False)
    mount = await mount_mcp_servers(registry)
    assert registry.schemas == []
    await mount.close()


@pytest.mark.asyncio
async def test_agent_loop_bootstrap_checks_missing_mcp_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZETA_MCP_CONFIG", str(tmp_path / "missing.json"))
    calls = 0
    original_mount = mount_module.mount_mcp_servers

    async def observe_mount(registry: ToolRegistry, config=None, *, notice_sink=None):
        nonlocal calls
        calls += 1
        return await original_mount(registry, config, notice_sink=notice_sink)

    monkeypatch.setattr("zeta.loop.mount_mcp_servers", observe_mount)
    backend = FakeBackend([ScriptedTurn([TextContent("booted")])])
    loop = AgentLoop(backend, ConversationStore(tmp_path))
    events = [event async for event in loop.run_turn("hello")]
    await loop.close()

    assert calls == 1
    assert events[-1].type.value == "agent_end"


@pytest.mark.asyncio
async def test_stdio_handshake_list_call_and_close(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WIKI_AGENT_RUNTIME_DIR", str(tmp_path))
    client = StdioMCPClient(_stdio_config())
    await client.connect()
    tools = await client.list_tools()
    result = await client.call_tool("echo", {"value": "hello"}, AbortSignal())

    assert [tool.name for tool in tools] == ["echo"]
    assert result["content"][0]["text"] == "hello"
    assert result["structuredContent"] == {"ok": True, "latency": 1.5}
    await client.close()


@pytest.mark.asyncio
async def test_stdio_abort_returns_canceled_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _stdio_source().replace(
        'result = {"content": [{"type": "text", "text": request["params"]["arguments"]["value"]}], "isError": False, "structuredContent": {"ok": True, "latency": 1.5}}',
        'import time; time.sleep(5); result = {"content": [], "isError": False}',
    )
    monkeypatch.setenv("WIKI_AGENT_RUNTIME_DIR", str(tmp_path))
    config = MCPServerConfig("abort", "stdio", sys.executable, ("-u", "-c", source))
    client = StdioMCPClient(config)
    await client.connect()
    signal = AbortSignal()
    task = asyncio.create_task(client.call_tool("echo", {"value": "wait"}, signal))
    await asyncio.sleep(0.05)
    signal.abort()
    result = await task
    assert result["content"][0]["text"] == "tool execution canceled"
    await client.close()


@pytest.mark.asyncio
async def test_stdio_abort_marks_mount_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _stdio_source().replace(
        'result = {"content": [{"type": "text", "text": request["params"]["arguments"]["value"]}], "isError": False, "structuredContent": {"ok": True, "latency": 1.5}}',
        'import time; time.sleep(5); result = {"content": [], "isError": False}',
    )
    monkeypatch.setenv("WIKI_AGENT_RUNTIME_DIR", str(tmp_path))
    config = MCPServerConfig("abort", "stdio", sys.executable, ("-u", "-c", source))
    registry = ToolRegistry(tmp_path, register_builtin=False)
    mount = await mount_mcp_servers(
        registry,
        MCPConfig(tmp_path / "mcp.json", {"abort": config}),
    )
    signal = AbortSignal()
    task = asyncio.create_task(
        registry.execute(
            ToolCall("call", "abort:echo", {"value": "wait"}),
            abort_signal=signal,
        )
    )
    await asyncio.sleep(0.05)
    signal.abort()
    result = await task

    assert result["isError"] is True
    for _ in range(100):
        if mount.statuses["abort"].state == "failed":
            break
        await asyncio.sleep(0.01)
    assert mount.statuses["abort"].state == "failed"
    assert mount.statuses["abort"].tool_count == 0
    assert mount.clients == ()
    assert registry.schemas == []
    assert "abort: failed" in mount.render()
    await mount.close()


@pytest.mark.asyncio
async def test_stdio_abort_kills_descendant_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "descendant-alive"
    monkeypatch.setenv("WIKI_AGENT_RUNTIME_DIR", str(tmp_path))
    client = StdioMCPClient(_descendant_stdio_config(marker))
    await client.connect()
    signal = AbortSignal()
    task = asyncio.create_task(client.call_tool("echo", {"value": "wait"}, signal))
    await asyncio.sleep(0.1)
    signal.abort()
    result = await task
    await asyncio.sleep(0.9)

    assert result["content"][0]["text"] == "tool execution canceled"
    assert not marker.exists()
    await client.close()


@pytest.mark.asyncio
async def test_stdio_crash_returns_error_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _stdio_source().replace(
        '    elif method == "tools/call":\n',
        '    elif method == "tools/call":\n        sys.exit(3)\n',
    )
    monkeypatch.setenv("WIKI_AGENT_RUNTIME_DIR", str(tmp_path))
    config = MCPServerConfig("crash", "stdio", sys.executable, ("-u", "-c", source))
    client = StdioMCPClient(config)
    await client.connect()
    result = await client.call_tool("echo", {"value": "crash"}, AbortSignal())
    assert result["isError"] is True
    assert "exited" in result["content"][0]["text"] or "reader" in result["content"][0]["text"]
    await client.close()


@pytest.mark.asyncio
async def test_http_auth_list_call_and_401() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = json.loads(request.content)
        if "id" not in body:
            return httpx.Response(202, request=request)
        if body["method"] == "initialize":
            result = {"protocolVersion": "2025-06-18", "capabilities": {}}
        elif body["method"] == "tools/list":
            result = {"tools": []}
        elif body["method"] == "tools/call":
            return httpx.Response(401, text="expired", request=request)
        else:
            result = {"content": [{"type": "text", "text": "unexpected"}], "isError": True}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result}, request=request)

    config = MCPServerConfig("http", "streamable-http", url="https://mcp.test", auth_type="bearer", auth_token="token")
    client = StreamableHTTPMCPClient(config, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    await client.connect()
    assert client.protocol_version == "2025-06-18"
    await client.list_tools()
    result = await client.call_tool("echo", {}, AbortSignal())
    assert result["isError"] is True
    assert "MCP HTTP 401" in result["content"][0]["text"]
    assert requests[0].headers["authorization"] == "Bearer token"
    assert all(
        request.headers["mcp-protocol-version"] == "2025-06-18"
        for request in requests[1:]
    )
    await client.close()


@pytest.mark.asyncio
async def test_http_sse_call() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "id" not in body:
            return httpx.Response(202, request=request)
        if body["method"] == "initialize":
            result = {"protocolVersion": "2025-06-18", "capabilities": {}}
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result}, request=request)
        if body["method"] == "tools/list":
            result = {"tools": []}
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result}, request=request)
        sse = "event: message\ndata: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\",\"params\":{}}\n\n" \
            + f"data: {{\"jsonrpc\":\"2.0\",\"id\":{body['id']},\"result\":{{\"content\":[{{\"type\":\"text\",\"text\":\"streamed\"}}],\"isError\":false}}}}\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=sse, request=request)

    config = MCPServerConfig("http", "streamable-http", url="https://mcp.test")
    client = StreamableHTTPMCPClient(config, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    await client.connect()
    result = await client.call_tool("echo", {}, AbortSignal())
    assert result["content"][0]["text"] == "streamed"
    await client.close()


@pytest.mark.asyncio
async def test_http_abort_closes_request() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "id" not in body:
            return httpx.Response(202, request=request)
        if body.get("method") == "initialize":
            result = {"protocolVersion": "2025-06-18", "capabilities": {}}
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": body["id"], "result": result},
                request=request,
            )
        if body.get("method") == "tools/call":
            await asyncio.sleep(5)
        result = {"content": [], "isError": False}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result}, request=request)

    config = MCPServerConfig("http", "streamable-http", url="https://mcp.test")
    client = StreamableHTTPMCPClient(config, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    await client.connect()
    signal = AbortSignal()
    task = asyncio.create_task(client.call_tool("echo", {}, signal))
    await asyncio.sleep(0.05)
    signal.abort()
    result = await task
    assert result["content"][0]["text"] == "tool execution canceled"
    await client.close()


@pytest.mark.asyncio
async def test_mount_registers_prefixed_tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WIKI_AGENT_RUNTIME_DIR", str(tmp_path))
    registry = ToolRegistry(tmp_path, register_builtin=False)
    mount = await mount_mcp_servers(registry, MCPConfig(tmp_path / "mcp.json", {"fake": _stdio_config()}))
    result = await registry.execute(ToolCall("call", "fake:echo", {"value": "mounted"}))
    assert result["content"][0]["text"] == "mounted"
    assert registry.schemas[0]["parameters"]["$defs"] == {"value": {"type": "string"}}
    await mount.close()


@pytest.mark.asyncio
async def test_mount_reports_states_and_notices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WIKI_AGENT_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("MCP_PRESENT", "yes")
    config_path = tmp_path / "mcp.json"
    config_path.write_text(json.dumps({"servers": {
        "mounted": {"transport": "stdio", "command": sys.executable, "args": ["-u", "-c", _stdio_source()]},
        "failed": {"transport": "stdio", "command": "failed"},
        "missing": {"transport": "stdio", "command": "${MCP_MISSING}"},
        "timed": {"transport": "stdio", "command": "timed"},
    }}))
    monkeypatch.setattr(mount_module, "SERVER_SETUP_TIMEOUT_SECONDS", 0.01)
    notices: list[str] = []

    def build_client(server_config: MCPServerConfig):
        return _FakeClient(server_config, fail_connect=server_config.name == "failed")

    original_connect_and_list = mount_module._connect_and_list

    async def connect_and_list(client):
        if client.config.name == "timed":
            await asyncio.sleep(1)
        return await original_connect_and_list(client)

    monkeypatch.setattr(mount_module, "_build_client", build_client)
    monkeypatch.setattr(mount_module, "_connect_and_list", connect_and_list)
    registry = ToolRegistry(tmp_path, register_builtin=False)
    config = load_mcp_config(config_path)
    mount = await mount_mcp_servers(registry, config, notice_sink=notices.append)

    assert mount.statuses["mounted"].state == "mounted"
    assert mount.statuses["failed"].state == "failed"
    assert mount.statuses["missing"].state == "skipped-missing-env"
    assert mount.statuses["timed"].state == "timed-out"
    assert "missing environment variable(s): MCP_MISSING" in mount.render()
    assert "stderr:" in mount.render()
    assert len(notices) == 4
    await mount.close()


@pytest.mark.asyncio
async def test_mount_reconnects_one_server_and_rejects_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = 0

    def build_client(config: MCPServerConfig):
        nonlocal attempts
        attempts += 1
        return _FakeClient(config, fail_connect=attempts == 1)

    monkeypatch.setattr(mount_module, "_build_client", build_client)
    configs = {
        "recover": MCPServerConfig("recover", "stdio", sys.executable),
        "healthy": MCPServerConfig("healthy", "stdio", sys.executable),
    }
    registry = ToolRegistry(tmp_path, register_builtin=False)
    mount = await mount_mcp_servers(registry, MCPConfig(tmp_path / "mcp.json", configs))

    assert mount.statuses["recover"].state == "failed"
    await mount.reconnect("recover")
    assert mount.statuses["recover"].state == "mounted"
    assert "recover:echo" in {schema["name"] for schema in registry.schemas}
    with pytest.raises(ValueError, match="unknown MCP server: absent"):
        await mount.reconnect("absent")
    await mount.close()


@pytest.mark.asyncio
async def test_canceled_reconnect_keeps_mount_state_consistent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = asyncio.Event()
    never = asyncio.Event()
    initial = _LifecycleClient(MCPServerConfig("server", "stdio", "unused"))
    replacement = _LifecycleClient(initial.config)
    clients = iter((initial, replacement))

    def build_client(config: MCPServerConfig) -> _LifecycleClient:
        del config
        return next(clients)

    async def connect_and_list(client: _LifecycleClient) -> list[MCPTool]:
        if client is replacement:
            started.set()
            await never.wait()
        return await client.list_tools()

    monkeypatch.setattr(mount_module, "_build_client", build_client)
    monkeypatch.setattr(mount_module, "_connect_and_list", connect_and_list)
    registry = ToolRegistry(tmp_path, register_builtin=False)
    mount = await mount_mcp_servers(
        registry,
        MCPConfig(tmp_path / "mcp.json", {"server": initial.config}),
    )

    reconnect = asyncio.create_task(mount.reconnect("server"))
    await started.wait()
    reconnect.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reconnect

    status = mount.statuses["server"]
    assert status.state == "failed"
    assert status.tool_count == 0
    assert mount.clients == ()
    assert registry.schemas == []
    assert replacement.closed
    await mount.close()


@pytest.mark.asyncio
async def test_reconnect_queued_behind_remove_does_not_mount_ghost_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remove_started = asyncio.Event()
    release_remove = asyncio.Event()
    client = _LifecycleClient(MCPServerConfig("server", "stdio", "unused"))

    monkeypatch.setattr(mount_module, "_build_client", lambda config: client)
    registry = ToolRegistry(tmp_path, register_builtin=False)
    mount = await mount_mcp_servers(
        registry,
        MCPConfig(tmp_path / "mcp.json", {"server": client.config}),
    )
    original_remove = mount._remove_server_locked

    async def blocked_remove(_mount: MCPMount, name: str) -> None:
        remove_started.set()
        await release_remove.wait()
        await original_remove(name)

    monkeypatch.setattr(MCPMount, "_remove_server_locked", blocked_remove)
    remove = asyncio.create_task(mount.remove_server("server"))
    await remove_started.wait()
    reconnect = asyncio.create_task(mount.reconnect("server"))
    release_remove.set()

    await remove
    with pytest.raises(ValueError, match="unknown MCP server: server"):
        await reconnect
    assert mount.clients == ()
    assert registry.schemas == []
    await mount.close()


@pytest.mark.asyncio
async def test_stdio_exit_updates_mount_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WIKI_AGENT_RUNTIME_DIR", str(tmp_path))
    registry = ToolRegistry(tmp_path, register_builtin=False)
    mount = await mount_mcp_servers(
        registry,
        MCPConfig(tmp_path / "mcp.json", {"dead": _stdio_config("dead")}),
    )
    client = mount._clients["dead"]
    process = client._process
    assert process is not None
    process.terminate()
    await process.wait()
    for _ in range(100):
        if mount.statuses["dead"].state == "failed":
            break
        await asyncio.sleep(0.01)

    assert mount.statuses["dead"].state == "failed"
    assert "dead:echo" not in {schema["name"] for schema in registry.schemas}
    await mount.close()


@pytest.mark.asyncio
async def test_mount_arms_failure_handler_during_sibling_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sibling_started = asyncio.Event()
    release_sibling = asyncio.Event()
    clients: dict[str, _LifecycleClient] = {}

    def build_client(config: MCPServerConfig) -> _LifecycleClient:
        client = _LifecycleClient(config)
        clients[config.name] = client
        return client

    async def connect_and_list(client: _LifecycleClient) -> list[MCPTool]:
        if client.config.name == "sibling":
            sibling_started.set()
            await release_sibling.wait()
        else:
            await sibling_started.wait()
            client.fail_transport("early exit")
        return [MCPTool("echo", "", {"type": "object"})]

    monkeypatch.setattr(mount_module, "_build_client", build_client)
    monkeypatch.setattr(mount_module, "_connect_and_list", connect_and_list)
    configs = {
        "early": MCPServerConfig("early", "stdio", "unused"),
        "sibling": MCPServerConfig("sibling", "stdio", "unused"),
    }
    registry = ToolRegistry(tmp_path, register_builtin=False)
    task = asyncio.create_task(
        mount_mcp_servers(registry, MCPConfig(tmp_path / "mcp.json", configs))
    )
    await sibling_started.wait()
    release_sibling.set()
    mount = await task

    assert mount.statuses["early"].state == "failed"
    assert "early:echo" not in {schema["name"] for schema in registry.schemas}
    assert "sibling:echo" in {schema["name"] for schema in registry.schemas}
    await mount.close()


@pytest.mark.asyncio
async def test_mount_cancellation_closes_clients_created_during_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = asyncio.Event()
    never = asyncio.Event()
    clients: list[_LifecycleClient] = []

    def build_client(config: MCPServerConfig) -> _LifecycleClient:
        client = _LifecycleClient(config)
        clients.append(client)
        return client

    async def connect_and_list(client: _LifecycleClient) -> list[MCPTool]:
        started.set()
        await never.wait()
        return [MCPTool("echo", "", {"type": "object"})]

    monkeypatch.setattr(mount_module, "_build_client", build_client)
    monkeypatch.setattr(mount_module, "_connect_and_list", connect_and_list)
    config = MCPConfig(
        tmp_path / "mcp.json",
        {"blocked": MCPServerConfig("blocked", "stdio", "unused")},
    )
    registry = ToolRegistry(tmp_path, register_builtin=False)
    task = asyncio.create_task(mount_mcp_servers(registry, config))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert clients and all(client.closed for client in clients)


@pytest.mark.asyncio
async def test_mcp_application_error_keeps_server_mounted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = ToolRegistry(tmp_path, register_builtin=False)
    client = _ApplicationErrorClient(
        MCPServerConfig("app", "stdio", "unused")
    )
    monkeypatch.setattr(mount_module, "_build_client", lambda config: client)
    mount = await mount_mcp_servers(
        registry,
        MCPConfig(tmp_path / "mcp.json", {"app": client.config}),
    )
    result = await registry.execute(ToolCall("call", "app:echo", {}))

    assert result["isError"] is True
    assert mount.statuses["app"].state == "mounted"
    assert "app:echo" in {schema["name"] for schema in registry.schemas}
    await mount.close()


@pytest.mark.asyncio
async def test_mcp_failure_refreshes_agent_loop_tool_schemas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _LifecycleClient(MCPServerConfig("dead", "stdio", "unused"))
    monkeypatch.setattr(mount_module, "_build_client", lambda config: client)
    config_path = tmp_path / "mcp.json"
    config_path.write_text(json.dumps({"servers": {"dead": {"transport": "stdio", "command": "unused"}}}))
    monkeypatch.setenv("ZETA_MCP_CONFIG", str(config_path))
    loop = AgentLoop(
        FakeBackend([]),
        ConversationStore(tmp_path),
        tool_schemas=[{"name": "dead:echo", "description": "", "parameters": {}}],
    )

    await loop.ensure_mcp_servers()
    assert "dead:echo" in {schema["name"] for schema in loop.tool_schemas}
    client.fail_transport("transport dropped")

    assert "dead:echo" not in {schema["name"] for schema in loop.tool_schemas}
    await loop.close()


@pytest.mark.asyncio
async def test_mount_continues_when_failed_server_cleanup_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = {
        "bad": MCPServerConfig("bad", "stdio", sys.executable),
        "good": MCPServerConfig("good", "stdio", sys.executable),
    }

    def build_client(config: MCPServerConfig):
        return _FakeClient(config, fail_connect=config.name == "bad", fail_close=config.name == "bad")

    monkeypatch.setattr(mount_module, "_build_client", build_client)
    registry = ToolRegistry(tmp_path, register_builtin=False)
    mount = await mount_mcp_servers(registry, MCPConfig(tmp_path / "mcp.json", configs))

    assert "good:echo" in {schema["name"] for schema in registry.schemas}
    await mount.close()


@pytest.mark.asyncio
async def test_mount_isolates_bad_server_from_good_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WIKI_AGENT_RUNTIME_DIR", str(tmp_path))
    bad = MCPServerConfig("bad", "stdio", str(tmp_path / "does-not-exist"))
    config = MCPConfig(tmp_path / "mcp.json", {"bad": bad, "good": _stdio_config("good")})
    registry = ToolRegistry(tmp_path, register_builtin=False)
    mount = await mount_mcp_servers(registry, config)
    assert "good:echo" in {schema["name"] for schema in registry.schemas}
    await mount.close()


@pytest.mark.asyncio
async def test_mcp_application_error_does_not_affect_other_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WIKI_AGENT_RUNTIME_DIR", str(tmp_path))
    config = MCPConfig(
        tmp_path / "mcp.json",
        {"fail": _failing_stdio_config(), "good": _stdio_config("good")},
    )
    registry = ToolRegistry(tmp_path, register_builtin=False)
    mount = await mount_mcp_servers(registry, config)
    failed = await registry.execute(ToolCall("failed", "fail:echo", {"value": "x"}))
    healthy = await registry.execute(ToolCall("healthy", "good:echo", {"value": "ok"}))
    assert failed["isError"] is True
    assert mount.statuses["fail"].state == "mounted"
    assert "fail:echo" in {schema["name"] for schema in registry.schemas}
    assert healthy["isError"] is False
    assert healthy["content"][0]["text"] == "ok"
    await mount.close()


@pytest.mark.asyncio
async def test_mcp_status_waits_for_one_shared_initial_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def delayed_mount(
        registry: ToolRegistry, config=None, *, notice_sink=None
    ) -> MCPMount:
        nonlocal calls
        del notice_sink, config
        calls += 1
        started.set()
        await release.wait()
        return MCPMount(registry, {}, {})

    monkeypatch.setattr("zeta.loop.mount_mcp_servers", delayed_mount)
    loop = AgentLoop(FakeBackend([]), ConversationStore(tmp_path))
    ensure_task = asyncio.create_task(loop.ensure_mcp_servers())
    await started.wait()
    status_task = asyncio.create_task(loop.slash_mcp(""))
    await asyncio.sleep(0)
    assert not status_task.done()

    release.set()
    await ensure_task
    assert await status_task == "mcp: 0 mounted, 0 failed"
    assert calls == 1
    await loop.close()


def _write_json(path: Path, servers: dict[str, dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"servers": servers}))


def test_overlay_project_shadows_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    project = tmp_path / "proj"
    monkeypatch.delenv("ZETA_MCP_CONFIG", raising=False)
    _write_json(
        home_config_path(home),
        {
            "shared": {"transport": "stdio", "command": "home-cmd"},
            "home_only": {"transport": "stdio", "command": "home-only"},
        },
    )
    _write_json(
        project_config_path(project),
        {
            "shared": {"transport": "stdio", "command": "project-cmd"},
            "project_only": {"transport": "stdio", "command": "project-only"},
        },
    )

    config = load_mcp_config_overlay(home=home, project_dir=project)

    assert config.servers["shared"].command == "project-cmd"
    assert config.servers["home_only"].command == "home-only"
    assert config.servers["project_only"].command == "project-only"
    assert config.sources["shared"] == project_config_path(project)
    assert config.sources["home_only"] == home_config_path(home)


def test_overlay_records_malformed_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ZETA_MCP_CONFIG", raising=False)
    home = tmp_path / "home"
    _write_json(
        home_config_path(home),
        {
            "good": {"transport": "stdio", "command": "ok"},
            "bad": {"transport": "sockets", "command": "nope"},
        },
    )

    config = load_mcp_config_overlay(home=home, project_dir=None)

    assert set(config.servers) == {"good"}
    assert "bad" in config.malformed_servers
    assert config.malformed_servers["bad"].malformed_reason is not None
    assert "transport" in config.malformed_servers["bad"].malformed_reason


def test_write_mcp_config_is_atomic(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "mcp.json"
    write_mcp_config(target, {"a": {"transport": "stdio", "command": "x"}})

    assert json.loads(target.read_text())["servers"]["a"]["command"] == "x"
    assert list(target.parent.glob("*.json.tmp")) == []
    assert list(target.parent.glob(".mcp.*")) == []


def test_write_mcp_config_leaves_no_partial_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "mcp.json"
    target.write_text('{"servers": {"keep": {"transport": "stdio", "command": "orig"}}}')
    original_replace = os.replace

    def boom(src, dst):  # noqa: ANN001
        raise RuntimeError("simulated rename failure")

    monkeypatch.setattr("zeta.mcp.config.os.replace", boom)
    with pytest.raises(RuntimeError, match="simulated"):
        write_mcp_config(target, {"new": {"transport": "stdio", "command": "x"}})

    assert json.loads(target.read_text())["servers"] == {
        "keep": {"transport": "stdio", "command": "orig"}
    }
    assert list(tmp_path.glob(".mcp.*")) == []
    del original_replace


def test_write_mcp_config_preserves_extra_root_fields(tmp_path: Path) -> None:
    target = tmp_path / "mcp.json"
    target.write_text(
        json.dumps(
            {
                "version": 2,
                "metadata": {"owner": "test"},
                "servers": {"old": {"transport": "stdio", "command": "old"}},
            }
        )
    )

    write_mcp_config(
        target,
        {"new": {"transport": "stdio", "command": "new"}},
    )

    assert json.loads(target.read_text()) == {
        "version": 2,
        "metadata": {"owner": "test"},
        "servers": {"new": {"transport": "stdio", "command": "new"}},
    }


def test_server_to_json_round_trips_stdio_and_http() -> None:
    stdio = MCPServerConfig(
        name="s", transport="stdio", command="cmd", args=("a", "b"), env={"K": "V"}
    )
    http = MCPServerConfig(
        name="h",
        transport="streamable-http",
        url="https://example",
        auth_type="bearer",
        auth_token="secret",
    )
    assert server_to_json(stdio) == {
        "transport": "stdio",
        "command": "cmd",
        "args": ["a", "b"],
        "env": {"K": "V"},
    }
    assert server_to_json(http) == {
        "transport": "streamable-http",
        "url": "https://example",
        "auth": {"type": "bearer", "token": "secret"},
    }


@pytest.mark.asyncio
async def test_slash_mcp_add_stdio_writes_project_file_and_mounts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.delenv("ZETA_MCP_CONFIG", raising=False)
    monkeypatch.setenv("WIKI_AGENT_RUNTIME_DIR", str(tmp_path))

    def build_client(config: MCPServerConfig) -> _FakeClient:
        return _FakeClient(config)

    monkeypatch.setattr(mount_module, "_build_client", build_client)
    loop = AgentLoop(FakeBackend([]), ConversationStore(project))
    loop.set_mcp_scope(home=home, project_dir=project)

    output = await loop.slash_mcp("add live --stdio server-cmd arg1 arg2")

    assert "live: mounted" in output
    project_file = project_config_path(project)
    entry = json.loads(project_file.read_text())["servers"]["live"]
    assert entry == {
        "transport": "stdio",
        "command": "server-cmd",
        "args": ["arg1", "arg2"],
    }
    assert loop._mcp_mount is not None
    assert loop._mcp_mount.statuses["live"].state == "mounted"
    assert "live:echo" in {schema["name"] for schema in loop.tool_registry.schemas}
    await loop.close()


@pytest.mark.asyncio
async def test_slash_mcp_add_http_writes_and_registers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.delenv("ZETA_MCP_CONFIG", raising=False)
    monkeypatch.setenv("WIKI_AGENT_RUNTIME_DIR", str(tmp_path))

    async def fake_connect_and_list(client):
        return [MCPTool("ping", "", {"type": "object"})]

    monkeypatch.setattr(mount_module, "_connect_and_list", fake_connect_and_list)
    loop = AgentLoop(FakeBackend([]), ConversationStore(project))
    loop.set_mcp_scope(home=home, project_dir=project)

    output = await loop.slash_mcp("add remote --http https://mcp.example")
    assert "remote: mounted" in output
    entry = json.loads(project_config_path(project).read_text())["servers"]["remote"]
    assert entry == {"transport": "streamable-http", "url": "https://mcp.example"}
    await loop.close()


@pytest.mark.asyncio
async def test_slash_mcp_add_rejects_duplicate_and_bad_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.delenv("ZETA_MCP_CONFIG", raising=False)
    monkeypatch.setenv("WIKI_AGENT_RUNTIME_DIR", str(tmp_path))

    async def fake_connect_and_list(client):
        return []

    monkeypatch.setattr(mount_module, "_connect_and_list", fake_connect_and_list)
    loop = AgentLoop(FakeBackend([]), ConversationStore(project))
    loop.set_mcp_scope(home=home, project_dir=project)

    await loop.slash_mcp("add remote --http https://mcp.example")
    duplicate = await loop.slash_mcp("add remote --http https://mcp.example")
    assert "already configured" in duplicate
    bad_flag = await loop.slash_mcp("add other --socket cmd")
    assert "unknown add flag" in bad_flag
    bad_http = await loop.slash_mcp("add three --http a b c")
    assert "--http takes exactly one" in bad_http
    await loop.close()


@pytest.mark.asyncio
async def test_slash_mcp_remove_deletes_entry_and_unmounts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.delenv("ZETA_MCP_CONFIG", raising=False)
    monkeypatch.setenv("WIKI_AGENT_RUNTIME_DIR", str(tmp_path))

    async def fake_connect_and_list(client):
        return [MCPTool("ping", "", {"type": "object"})]

    monkeypatch.setattr(mount_module, "_connect_and_list", fake_connect_and_list)
    loop = AgentLoop(FakeBackend([]), ConversationStore(project))
    loop.set_mcp_scope(home=home, project_dir=project)

    await loop.slash_mcp("add live --http https://mcp.example")
    assert "live:ping" in {schema["name"] for schema in loop.tool_registry.schemas}
    output = await loop.slash_mcp("remove live")

    assert "live" not in output
    assert loop._mcp_mount is not None
    assert "live" not in loop._mcp_mount.configs
    assert "live:ping" not in {schema["name"] for schema in loop.tool_registry.schemas}
    data = json.loads(project_config_path(project).read_text())["servers"]
    assert "live" not in data
    await loop.close()


@pytest.mark.asyncio
async def test_slash_mcp_remove_project_entry_unshadows_home_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.delenv("ZETA_MCP_CONFIG", raising=False)
    monkeypatch.setenv("WIKI_AGENT_RUNTIME_DIR", str(tmp_path))
    _write_json(
        home_config_path(home),
        {"shared": {"transport": "streamable-http", "url": "https://home"}},
    )
    _write_json(
        project_config_path(project),
        {"shared": {"transport": "streamable-http", "url": "https://project"}},
    )

    seen: list[str] = []

    async def fake_connect_and_list(client):
        seen.append(client.config.url or "?")
        return [MCPTool("ping", "", {"type": "object"})]

    monkeypatch.setattr(mount_module, "_connect_and_list", fake_connect_and_list)
    loop = AgentLoop(FakeBackend([]), ConversationStore(project))
    loop.set_mcp_scope(home=home, project_dir=project)

    await loop.ensure_mcp_servers()
    assert loop._mcp_mount is not None
    assert loop._mcp_mount.configs["shared"].url == "https://project"
    await loop.slash_mcp("remove shared")

    assert loop._mcp_mount.configs["shared"].url == "https://home"
    assert loop._mcp_mount.statuses["shared"].state == "mounted"
    assert loop._mcp_mount.sources["shared"] == home_config_path(home)
    project_data = json.loads(project_config_path(project).read_text())["servers"]
    assert "shared" not in project_data
    home_data = json.loads(home_config_path(home).read_text())["servers"]
    assert "shared" in home_data
    assert seen == ["https://project", "https://home"]
    await loop.close()


@pytest.mark.asyncio
async def test_slash_mcp_lists_malformed_with_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.delenv("ZETA_MCP_CONFIG", raising=False)
    monkeypatch.setenv("WIKI_AGENT_RUNTIME_DIR", str(tmp_path))
    _write_json(
        home_config_path(home),
        {"broken": {"transport": "carrier-pigeon", "command": "nope"}},
    )
    loop = AgentLoop(FakeBackend([]), ConversationStore(project))
    loop.set_mcp_scope(home=home, project_dir=project)

    output = await loop.slash_mcp("")
    assert "broken: malformed" in output
    assert "transport" in output
    await loop.close()


@pytest.mark.asyncio
async def test_slash_mcp_rejects_unmatched_quotes_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    loop = AgentLoop(FakeBackend([]), ConversationStore(project))
    loop.set_mcp_scope(home=tmp_path / "home", project_dir=project)

    output = await loop.slash_mcp('add bad --stdio command "unterminated')

    assert "No closing quotation" in output
    assert not project_config_path(project).exists()
    await loop.close()


@pytest.mark.asyncio
async def test_malformed_project_config_does_not_fall_back_to_ambient_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ambient = tmp_path / "ambient"
    project = tmp_path / "proj"
    project.mkdir()
    _write_json(
        home_config_path(ambient),
        {"sentinel": {"transport": "stdio", "command": "must-not-run"}},
    )
    project_config_path(project).parent.mkdir(parents=True, exist_ok=True)
    project_config_path(project).write_text("{")
    monkeypatch.setenv("ZETA_HOME", str(ambient))
    monkeypatch.delenv("ZETA_MCP_CONFIG", raising=False)

    loop = AgentLoop(FakeBackend([]), ConversationStore(project))
    loop.set_mcp_scope(home=ambient, project_dir=project)

    output = await loop.slash_mcp("")

    assert "could not read MCP config" in output
    assert str(project_config_path(project)) in output
    assert loop._mcp_mount is not None
    assert "sentinel" not in loop._mcp_mount.configs
    assert loop._mcp_mount.clients == ()
    await loop.close()


@pytest.mark.asyncio
async def test_slash_mcp_add_interpolates_only_the_live_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.setenv("MCP_TEST_COMMAND", "resolved-command")
    monkeypatch.setenv("MCP_TEST_ARG", "hello world")
    seen: list[MCPServerConfig] = []

    def build_client(config: MCPServerConfig) -> _FakeClient:
        seen.append(config)
        return _FakeClient(config)

    monkeypatch.setattr(mount_module, "_build_client", build_client)
    loop = AgentLoop(FakeBackend([]), ConversationStore(project))
    loop.set_mcp_scope(home=tmp_path / "home", project_dir=project)

    await loop.slash_mcp(
        'add live --stdio "${MCP_TEST_COMMAND}" "${MCP_TEST_ARG}"'
    )

    assert seen[0].command == "resolved-command"
    assert seen[0].args == ("hello world",)
    raw = json.loads(project_config_path(project).read_text())["servers"]["live"]
    assert raw["command"] == "${MCP_TEST_COMMAND}"
    assert raw["args"] == ["${MCP_TEST_ARG}"]
    await loop.close()


@pytest.mark.asyncio
async def test_slash_mcp_add_skips_missing_live_env_without_spawning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.delenv("MCP_MISSING_ADD", raising=False)
    spawned: list[MCPServerConfig] = []

    def build_client(config: MCPServerConfig) -> _FakeClient:
        spawned.append(config)
        return _FakeClient(config)

    monkeypatch.setattr(mount_module, "_build_client", build_client)
    loop = AgentLoop(FakeBackend([]), ConversationStore(project))
    loop.set_mcp_scope(home=tmp_path / "home", project_dir=project)

    output = await loop.slash_mcp(
        "add missing --stdio '${MCP_MISSING_ADD}'"
    )

    assert "missing: skipped-missing-env" in output
    assert not spawned
    assert json.loads(project_config_path(project).read_text())["servers"][
        "missing"
    ]["command"] == "${MCP_MISSING_ADD}"
    await loop.close()


@pytest.mark.asyncio
async def test_canceled_mcp_add_leaves_no_ghost_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_connect_and_list(client: _FakeClient) -> list[MCPTool]:
        del client
        started.set()
        await release.wait()
        return []

    monkeypatch.setattr(mount_module, "_build_client", _FakeClient)
    monkeypatch.setattr(mount_module, "_connect_and_list", slow_connect_and_list)
    loop = AgentLoop(FakeBackend([]), ConversationStore(project))
    loop.set_mcp_scope(home=tmp_path / "home", project_dir=project)
    task = asyncio.create_task(loop.slash_mcp("add ghost --stdio command"))
    await started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert loop._mcp_mount is not None
    assert "ghost" not in loop._mcp_mount.configs
    assert "ghost" not in loop._mcp_mount.statuses
    assert "ghost" not in loop._mcp_mount.sources
    assert "ghost" not in json.loads(
        project_config_path(project).read_text()
    ).get("servers", {})
    await loop.close()


@pytest.mark.asyncio
async def test_canceled_mcp_unshadow_keeps_failed_home_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    project = tmp_path / "proj"
    project.mkdir()
    _write_json(
        home_config_path(home),
        {"shared": {"transport": "streamable-http", "url": "https://home"}},
    )
    _write_json(
        project_config_path(project),
        {"shared": {"transport": "streamable-http", "url": "https://project"}},
    )
    started = asyncio.Event()
    never = asyncio.Event()
    initial = _LifecycleClient(
        MCPServerConfig("shared", "streamable-http", url="https://project")
    )
    fallback = _LifecycleClient(
        MCPServerConfig("shared", "streamable-http", url="https://home")
    )
    clients = iter((initial, fallback))

    def build_client(config: MCPServerConfig) -> _LifecycleClient:
        del config
        return next(clients)

    async def connect_and_list(client: _LifecycleClient) -> list[MCPTool]:
        if client is fallback:
            started.set()
            await never.wait()
        return await client.list_tools()

    monkeypatch.setattr(mount_module, "_build_client", build_client)
    monkeypatch.setattr(mount_module, "_connect_and_list", connect_and_list)
    loop = AgentLoop(FakeBackend([]), ConversationStore(project))
    loop.set_mcp_scope(home=home, project_dir=project)
    await loop.ensure_mcp_servers()

    remove = asyncio.create_task(loop.slash_mcp("remove shared"))
    await started.wait()
    remove.cancel()
    with pytest.raises(asyncio.CancelledError):
        await remove

    assert loop._mcp_mount is not None
    assert loop._mcp_mount.configs["shared"].url == "https://home"
    assert loop._mcp_mount.sources["shared"] == home_config_path(home)
    assert loop._mcp_mount.statuses["shared"].state == "failed"
    assert "shared" not in json.loads(
        project_config_path(project).read_text()
    )["servers"]
    assert "shared" in json.loads(home_config_path(home).read_text())["servers"]
    await loop.close()


@pytest.mark.asyncio
async def test_concurrent_mcp_removes_do_not_remount_removed_home_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    project = tmp_path / "proj"
    project.mkdir()
    _write_json(
        home_config_path(home),
        {"shared": {"transport": "streamable-http", "url": "https://home"}},
    )
    _write_json(
        project_config_path(project),
        {"shared": {"transport": "streamable-http", "url": "https://project"}},
    )
    fallback_started = asyncio.Event()
    release_fallback = asyncio.Event()
    initial = _LifecycleClient(
        MCPServerConfig("shared", "streamable-http", url="https://project")
    )
    fallback = _LifecycleClient(
        MCPServerConfig("shared", "streamable-http", url="https://home")
    )
    second_fallback = _LifecycleClient(fallback.config)
    clients = iter((initial, fallback, second_fallback))

    def build_client(config: MCPServerConfig) -> _LifecycleClient:
        del config
        return next(clients)

    async def connect_and_list(client: _LifecycleClient) -> list[MCPTool]:
        if client is fallback:
            fallback_started.set()
            await release_fallback.wait()
        return await client.list_tools()

    monkeypatch.setattr(mount_module, "_build_client", build_client)
    monkeypatch.setattr(mount_module, "_connect_and_list", connect_and_list)
    loop = AgentLoop(FakeBackend([]), ConversationStore(project))
    loop.set_mcp_scope(home=home, project_dir=project)
    await loop.ensure_mcp_servers()

    first = asyncio.create_task(loop.slash_mcp("remove shared"))
    await fallback_started.wait()
    second = asyncio.create_task(loop.slash_mcp("remove shared"))
    await asyncio.sleep(0)
    release_fallback.set()
    await asyncio.gather(first, second)

    assert loop._mcp_mount is not None
    assert "shared" not in loop._mcp_mount.configs
    assert "shared" not in json.loads(home_config_path(home).read_text())["servers"]
    assert "shared" not in json.loads(
        project_config_path(project).read_text()
    )["servers"]
    await loop.close()


@pytest.mark.asyncio
async def test_concurrent_mcp_adds_keep_both_disk_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "proj"
    project.mkdir()

    async def yield_then_list(client: _FakeClient) -> list[MCPTool]:
        del client
        await asyncio.sleep(0)
        return []

    monkeypatch.setattr(mount_module, "_build_client", _FakeClient)
    monkeypatch.setattr(mount_module, "_connect_and_list", yield_then_list)
    loop = AgentLoop(FakeBackend([]), ConversationStore(project))
    loop.set_mcp_scope(home=tmp_path / "home", project_dir=project)

    await asyncio.gather(
        loop.slash_mcp("add first --stdio command-one"),
        loop.slash_mcp("add second --stdio command-two"),
    )

    servers = json.loads(project_config_path(project).read_text())["servers"]
    assert set(servers) == {"first", "second"}
    await loop.close()


def test_concurrent_process_config_edits_keep_both_entries(tmp_path: Path) -> None:
    target = tmp_path / "mcp.json"
    script = """
import sys
import time
from pathlib import Path
from zeta.mcp.commands import rewrite_mcp_file

target = Path(sys.argv[1])
name = sys.argv[2]

def edit(servers):
    time.sleep(0.3)
    servers[name] = {"transport": "stdio", "command": name}
    return servers

rewrite_mcp_file(target, edit)
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(target), name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for name in ("first", "second")
    ]
    results = [process.communicate(timeout=5) for process in processes]

    assert all(process.returncode == 0 for process in processes), results
    assert set(json.loads(target.read_text())["servers"]) == {"first", "second"}


@pytest.mark.asyncio
async def test_mcp_remove_clears_provider_schema_after_unmount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "proj"
    project.mkdir()

    async def fake_connect_and_list(client: _FakeClient) -> list[MCPTool]:
        del client
        return [MCPTool("echo", "", {"type": "object"})]

    monkeypatch.setattr(mount_module, "_connect_and_list", fake_connect_and_list)
    loop = AgentLoop(
        FakeBackend([]),
        ConversationStore(project),
        tool_schemas=[{"name": "dead:echo", "description": "", "parameters": {}}],
    )
    loop.set_mcp_scope(home=tmp_path / "home", project_dir=project)

    await loop.slash_mcp("add dead --http https://mcp.example")
    assert "dead:echo" in {schema["name"] for schema in loop.tool_schemas}
    await loop.slash_mcp("remove dead")

    assert "dead:echo" not in {schema["name"] for schema in loop.tool_schemas}
    await loop.close()


@pytest.mark.asyncio
async def test_mcp_schema_refresh_does_not_duplicate_current_provider_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_connect_and_list(client: _FakeClient) -> list[MCPTool]:
        del client
        return [MCPTool("echo", "", {"type": "object"})]

    monkeypatch.setattr(mount_module, "_connect_and_list", fake_connect_and_list)
    loop = AgentLoop(
        FakeBackend([]),
        ConversationStore(tmp_path),
        tool_schemas=[{"name": "live:echo", "description": "", "parameters": {}}],
    )
    loop.set_mcp_scope(home=tmp_path / "home", project_dir=tmp_path / "project")

    await loop.slash_mcp("add live --http https://mcp.example")
    loop._refresh_mcp_tool_schemas()

    assert [schema["name"] for schema in loop.tool_schemas].count("live:echo") == 1
    await loop.close()


@pytest.mark.asyncio
async def test_late_failure_from_removed_client_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _LifecycleClient(MCPServerConfig("dead", "stdio", "unused"))
    monkeypatch.setattr(mount_module, "_build_client", lambda config: client)
    registry = ToolRegistry(tmp_path, register_builtin=False)
    mount = await mount_mcp_servers(
        registry,
        MCPConfig(tmp_path / "mcp.json", {"dead": client.config}),
    )

    await mount.remove_server("dead")
    client.fail_transport("late failure")

    assert "dead" not in mount.statuses
    assert mount.clients == ()
    assert registry.schemas == []
    await mount.close()
