import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

from zeta.core.abort import AbortSignal
from zeta.mcp import (
    MCPConfig,
    MCPConfigError,
    MCPServerConfig,
    StdioMCPClient,
    StreamableHTTPMCPClient,
    load_mcp_config,
    mount_mcp_servers,
)
from zeta.tools import ToolRegistry
from zeta.types import ToolCall


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
        result = {"tools": [{"name": "echo", "description": "echo text", "inputSchema": {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]}}]}
    elif method == "tools/call":
        result = {"content": [{"type": "text", "text": request["params"]["arguments"]["value"]}], "isError": False, "structuredContent": {"ok": True}}
    else:
        result = {"content": [], "isError": False}
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
"""


def _stdio_config(name: str = "fake") -> MCPServerConfig:
    return MCPServerConfig(name, "stdio", sys.executable, ("-u", "-c", _stdio_source()))


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


def test_config_override_and_malformed_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    override = tmp_path / "override.json"
    override.write_text('{"servers": {}}')
    monkeypatch.setenv("ZETA_MCP_CONFIG", str(override))
    assert load_mcp_config().path == override
    override.write_text("{")
    with pytest.raises(MCPConfigError, match=str(override)):
        load_mcp_config()


@pytest.mark.asyncio
async def test_stdio_handshake_list_call_and_close(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WIKI_AGENT_RUNTIME_DIR", str(tmp_path))
    client = StdioMCPClient(_stdio_config())
    await client.connect()
    tools = await client.list_tools()
    result = await client.call_tool("echo", {"value": "hello"}, AbortSignal())

    assert [tool.name for tool in tools] == ["echo"]
    assert result["content"][0]["text"] == "hello"
    assert result["structuredContent"] == {"ok": True}
    await client.close()


@pytest.mark.asyncio
async def test_stdio_abort_returns_canceled_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _stdio_source().replace(
        'result = {"content": [{"type": "text", "text": request["params"]["arguments"]["value"]}], "isError": False, "structuredContent": {"ok": True}}',
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
        else:
            result = {"content": [{"type": "text", "text": "ok"}], "isError": True}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result}, request=request)

    config = MCPServerConfig("http", "streamable-http", url="https://mcp.test", auth_type="bearer", auth_token="token")
    client = StreamableHTTPMCPClient(config, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    await client.connect()
    await client.list_tools()
    result = await client.call_tool("echo", {}, AbortSignal())
    assert result["isError"] is True
    assert requests[0].headers["authorization"] == "Bearer token"
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
    await mount.close()
