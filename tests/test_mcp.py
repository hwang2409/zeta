import asyncio
import importlib
import json
import sys
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
    load_mcp_config,
    mount_mcp_servers,
)
from zeta.mcp.client import translate_call_result
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

    async def observe_mount(registry: ToolRegistry):
        nonlocal calls
        calls += 1
        return await original_mount(registry)

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
async def test_call_failure_does_not_affect_other_server(
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
    assert mount.statuses["fail"].state == "failed"
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
        registry: ToolRegistry, *, notice_sink=None
    ) -> MCPMount:
        nonlocal calls
        del notice_sink
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
