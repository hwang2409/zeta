"""MCP caller context across the actor queue, through real stdio and HTTP transports."""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import sys
from collections.abc import AsyncIterator, Iterator, Sequence

import httpx
import pytest

from tests.support.owned import await_or_fail
from zeta.core.store import ConversationStore
from zeta.mcp import MCPConfig, mount_mcp_servers
from zeta.mcp.client import CALLER_TRACE, CallerTrace
from zeta.mcp.config import MCPServerConfig
from zeta.mcp.http import StreamableHTTPMCPClient
from zeta.mcp.stdio import StdioMCPClient
from zeta.protocol.types import (
    Message,
    MessageOrigin,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolSchema,
    ToolUseContent,
)
from zeta.providers.completion import owned_completion
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.telemetry import Telemetry
from zeta.tools import ToolRegistry

pytest.importorskip("opentelemetry.sdk")

mount_module = importlib.import_module("zeta.mcp.mount")

SECRET = "secret-mcp-text"
SESSION = "0123456789abcdef0123456789abcdef"
REMOTE = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"

# A real stdio MCP server. Every caller request is answered with the exact line
# it read, so a caller sees the bytes the server received. With argv "pair" it
# holds the first caller request until a second one arrives, then answers the
# second first: both requests are in flight together.
_SERVER = r"""
import json
import sys

pair = sys.argv[1:] == ["pair"]
held = []


def reply(request, raw):
    method = request["method"]
    if method == "tools/call":
        result = {"content": [{"type": "text", "text": raw}], "isError": False}
    elif method == "prompts/get":
        result = {"messages": [{"role": "user", "content": {"type": "text", "text": raw}}]}
    elif method == "resources/list":
        result = {"resources": [{"uri": "mem://one", "name": "one", "description": raw}]}
    else:
        result = {"contents": [{"uri": "mem://one", "text": raw}]}
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)


for line in sys.stdin:
    raw = line.rstrip("\n")
    request = json.loads(raw)
    if "id" not in request:
        continue
    method = request["method"]
    if method == "initialize":
        capabilities = {"tools": {}, "prompts": {}, "resources": {}}
        result = {"protocolVersion": "2025-06-18", "capabilities": capabilities, "serverInfo": {"name": "fake", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "echo", "description": "echo", "inputSchema": {"type": "object"}}]}
    elif method == "prompts/list":
        result = {"prompts": [{"name": "greet", "description": "greet"}]}
    else:
        held.append((request, raw))
        if pair and len(held) < 2:
            continue
        for request, raw in reversed(held):
            reply(request, raw)
        held.clear()
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
"""


def _stdio(*args: str) -> MCPServerConfig:
    return MCPServerConfig("fake", "stdio", sys.executable, ("-u", "-c", _SERVER, *args))


def _http_handler(seen: list[httpx.Request]):
    """The same replies as ``_SERVER`` over streamable HTTP, one request at a time."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "id" not in body:
            return httpx.Response(202, request=request)
        method = body["method"]
        raw = request.content.decode()
        if method == "initialize":
            result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}, "prompts": {}, "resources": {}}}
        elif method == "tools/list":
            result = {"tools": [{"name": "echo", "description": "echo", "inputSchema": {"type": "object"}}]}
        elif method == "prompts/list":
            result = {"prompts": [{"name": "greet", "description": "greet"}]}
        else:
            seen.append(request)
            result = {
                "tools/call": {"content": [{"type": "text", "text": raw}], "isError": False},
                "prompts/get": {"messages": [{"role": "user", "content": {"type": "text", "text": raw}}]},
                "resources/list": {"resources": [{"uri": "mem://one", "name": "one", "description": raw}]},
                "resources/read": {"contents": [{"uri": "mem://one", "text": raw}]},
            }[method]
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result}, request=request)

    return handler


@pytest.fixture
def traced() -> Iterator[tuple[Telemetry, object]]:
    from opentelemetry.exporter.otlp.proto.http import trace_exporter
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    class DropExporter:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def export(self, spans: object) -> SpanExportResult:
            return SpanExportResult.SUCCESS

        def shutdown(self) -> None:
            pass

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(trace_exporter, "OTLPSpanExporter", DropExporter)
        telemetry = Telemetry.configure("http://127.0.0.1:4318/v1/traces")
    memory = InMemorySpanExporter()
    telemetry._implementation._provider.add_span_processor(SimpleSpanProcessor(memory))
    try:
        yield telemetry, memory
    finally:
        telemetry.shutdown()


def _traceparent(span: object) -> str:
    return f"00-{span.context.trace_id:032x}-{span.context.span_id:016x}-{span.context.trace_flags:02x}"


def _meta(raw: str) -> dict[str, str] | None:
    return json.loads(raw)["params"].get("_meta")


async def _call(mount: object, method: str) -> str:
    """One caller request through the mount; returns the line the server read."""

    if method == "tools/call":
        return (await mount.call_tool("fake", "echo", {"value": SECRET}))["content"][0]["text"]
    if method == "prompts/get":
        return await mount.get_prompt("fake:greet", {"topic": SECRET})
    if method == "resources/list":
        return (await mount.list_resources("fake"))[0].description
    return (await mount.read_resource("fake", f"mem://{SECRET}")).text


METHODS = ["tools/call", "prompts/get", "resources/list", "resources/read"]


@pytest.mark.parametrize("method", METHODS)
def test_concurrent_callers_each_carry_their_own_parent_across_the_actor_queue(tmp_path, traced, caplog, method) -> None:
    """Tools enrich the caller's execute_tool span; prompts and resources get one CLIENT span under the caller."""

    from opentelemetry.trace import SpanKind

    telemetry, memory = traced

    async def scenario() -> tuple[dict[str, str], dict[str, str]]:
        registry = ToolRegistry(tmp_path, register_builtin=False, skill_catalog=SkillCatalog.empty())
        mount = await mount_mcp_servers(registry, MCPConfig(tmp_path / "mcp.json", {"fake": _stdio("pair")}), telemetry=telemetry)
        try:
            async def caller(label: str, number: int) -> str:
                parent = (
                    telemetry.tool_span("fake__echo", tool_type="mcp", call_id=f"toolu_{label}", agent_name="zeta", conversation_id=SESSION)
                    if method == "tools/call"
                    else telemetry.turn_span(conversation_id=SESSION, turn_number=number, notification=False)
                )
                with parent:
                    return await _call(mount, method)

            first, second = await await_or_fail(asyncio.gather(caller("a", 1), caller("b", 2)), what="both callers")
            # the actor task detached each caller's context right after creating its task
            actor = mount._actors["fake"]
            assert actor._task.get_context().run(telemetry.current_trace_context) == {}
            assert actor._task.get_context().run(CALLER_TRACE.get) is None
            return {"a": first, "b": second}
        finally:
            await mount.close()
            await registry.close()

    with caplog.at_level(logging.WARNING, logger="opentelemetry"):
        lines = asyncio.run(scenario())
    assert [record.getMessage() for record in caplog.records] == []
    spans = memory.get_finished_spans()
    by_id = {span.context.span_id: span for span in spans}
    mcp = [span for span in spans if "mcp.method.name" in span.attributes]
    assert len(mcp) == 2
    carried = {}
    for span in mcp:
        if method == "tools/call":
            assert span.name == "execute_tool fake__echo" and span.kind is SpanKind.INTERNAL
            caller = span.attributes["zeta.tool.call.id"].removeprefix("toolu_")
        else:
            caller_span = by_id[span.parent.span_id]
            assert span.kind is SpanKind.CLIENT and span.parent.span_id == caller_span.context.span_id
            caller = "ab"[caller_span.attributes["zeta.turn.number"] - 1]
        carried[caller] = _traceparent(span)
        assert {key: value for key, value in span.attributes.items() if key.startswith(("mcp.", "zeta.mcp.", "gen_ai.prompt", "jsonrpc"))} == {
            "mcp.method.name": method,
            "zeta.mcp.server.name": "fake",
            "mcp.protocol.version": "2025-06-18",
            "jsonrpc.request.id": str(json.loads(lines[caller])["id"]),
            **({"gen_ai.prompt.name": "greet"} if method == "prompts/get" else {}),
        }
    expected_name = {"tools/call": "execute_tool fake__echo", "prompts/get": "prompts/get greet"}.get(method, method)
    assert sorted(span.name for span in mcp) == [expected_name, expected_name]
    # each request carried its own caller's span, and the two callers are separate traces
    assert {caller: _meta(line) for caller, line in lines.items()} == {caller: {"traceparent": value} for caller, value in carried.items()}
    assert carried["a"][3:35] != carried["b"][3:35]
    for span in spans:
        assert SECRET not in str(span.attributes) and SECRET not in span.name
        assert "mem://" not in str(span.attributes)


@pytest.mark.parametrize("transport", ["stdio", "http"])
def test_real_write_keeps_meta_adds_tracestate_and_never_baggage(tmp_path, traced, monkeypatch, transport) -> None:
    from opentelemetry import baggage, context

    telemetry, memory = traced
    seen: list[httpx.Request] = []

    async def scenario() -> str:
        if transport == "stdio":
            client = StdioMCPClient(_stdio())
        else:
            config = MCPServerConfig("fake", "streamable-http", url="https://mcp.test")
            client = StreamableHTTPMCPClient(config, client=httpx.AsyncClient(transport=httpx.MockTransport(_http_handler(seen))))
        await client.connect()
        try:
            with telemetry.use_trace_context({"traceparent": REMOTE, "tracestate": "zeta=t1"}):
                token = context.attach(baggage.set_baggage("user", SECRET))
                try:
                    with telemetry.mcp_span("tools/call", server="fake", target="echo") as span:
                        caller = CALLER_TRACE.set(CallerTrace(telemetry, span))
                        try:
                            params = {"name": "echo", "arguments": {}, "_meta": {"progressToken": "p1"}}
                            result = await client._request("tools/call", params)
                        finally:
                            CALLER_TRACE.reset(caller)
                finally:
                    context.detach(token)
            return result["content"][0]["text"]
        finally:
            await client.close()

    raw = asyncio.run(scenario())
    (span,) = memory.get_finished_spans()
    assert f"{span.parent.trace_id:032x}" == REMOTE[3:35]
    assert _meta(raw) == {"progressToken": "p1", "traceparent": _traceparent(span), "tracestate": "zeta=t1"}
    assert "baggage" not in raw and SECRET not in raw
    if transport == "http":
        (request,) = seen
        assert {key: request.headers.get(key) for key in ("traceparent", "tracestate", "baggage")} == {
            "traceparent": _traceparent(span), "tracestate": "zeta=t1", "baggage": None,
        }


@pytest.mark.parametrize("transport", ["stdio", "http"])
@pytest.mark.parametrize("method", METHODS)
def test_telemetry_off_leaves_mcp_wire_bytes_unchanged(tmp_path, monkeypatch, transport, method) -> None:
    seen: list[httpx.Request] = []
    if transport == "http":
        client = StreamableHTTPMCPClient(
            MCPServerConfig("fake", "streamable-http", url="https://mcp.test"),
            client=httpx.AsyncClient(transport=httpx.MockTransport(_http_handler(seen))),
        )
        monkeypatch.setattr(mount_module, "_build_client", lambda _config: client)
    params = {
        "tools/call": {"name": "echo", "arguments": {"value": SECRET}},
        "prompts/get": {"name": "greet", "arguments": {"topic": SECRET}},
        "resources/list": {},
        "resources/read": {"uri": f"mem://{SECRET}"},
    }[method]

    async def scenario() -> str:
        registry = ToolRegistry(tmp_path, register_builtin=False, skill_catalog=SkillCatalog.empty())
        config = _stdio() if transport == "stdio" else MCPServerConfig("fake", "streamable-http", url="https://mcp.test")
        mount = await mount_mcp_servers(registry, MCPConfig(tmp_path / "mcp.json", {"fake": config}))
        try:
            return await _call(mount, method)
        finally:
            await mount.close()
            await registry.close()

    raw = asyncio.run(scenario())
    request_id = json.loads(raw)["id"]
    if transport == "stdio":
        expected = json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}, separators=(",", ":"))
    else:
        expected = httpx.Request("POST", "https://mcp.test", json={"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}).content.decode()
        (request,) = seen
        assert {"traceparent", "tracestate", "baggage"}.isdisjoint(request.headers)
    assert raw == expected


class _Backend:
    """The first call asks for one mounted MCP tool; every later call replies with text."""

    provider = "claude"
    model = "claude-opus-5-5"

    def __init__(self) -> None:
        self.results: list[str] = []

    async def _completion_wait(self, delay: float) -> None:
        del delay

    @owned_completion
    async def complete(self, messages: Sequence[Message], tool_schemas: Sequence[ToolSchema]) -> AsyncIterator[StreamEvent]:
        del tool_schemas
        self.results = [message.tool_result.content for message in messages if message.tool_result is not None]
        answered = bool(self.results)
        blocks = [TextContent("done")] if answered else [ToolUseContent(ToolCall("toolu_mcp", "fake__echo", {"value": SECRET}))]
        yield StreamEvent(StreamEventType.MESSAGE_START)
        yield StreamEvent(StreamEventType.MESSAGE_END, message=Message(MessageRole.ASSISTANT, blocks), data={"stop_reason": "end_turn"})


def test_model_issued_mcp_call_has_one_enriched_tool_span(tmp_path, traced) -> None:
    telemetry, memory = traced

    async def scenario() -> str:
        store = ConversationStore(tmp_path / "sessions")
        backend = _Backend()
        loop = AgentLoop(backend, store, max_turns=3, skill_catalog=SkillCatalog.empty(), telemetry=telemetry)
        try:
            mount = await mount_mcp_servers(loop.tool_registry, MCPConfig(tmp_path / "mcp.json", {"fake": _stdio()}), telemetry=telemetry)
            loop.attach_mcp_mount(mount)
            async for _event in loop.run_turn("use the tool", origin=MessageOrigin.USER):
                pass
            (result,) = backend.results
            return result
        finally:
            await loop.close()
            loop.store.close()

    raw = asyncio.run(scenario())
    spans = memory.get_finished_spans()
    (tool,) = [span for span in spans if span.name.startswith("execute_tool")]
    assert tool.name == "execute_tool fake__echo"
    assert [span for span in spans if "mcp.method.name" in span.attributes] == [tool]
    assert not any(span.name.startswith("tools/call") for span in spans)
    assert tool.attributes["mcp.method.name"] == "tools/call"
    assert tool.attributes["gen_ai.tool.name"] == "echo"
    assert tool.attributes["zeta.mcp.server.name"] == "fake"
    assert _meta(raw) == {"traceparent": _traceparent(tool)}


def test_request_metadata_written_after_the_caller_span_ended_is_dropped(traced, caplog) -> None:
    """The operation task can outlive a caller that stopped waiting; its late write must not reach the span."""

    telemetry, memory = traced
    with telemetry.mcp_span("resources/list", server="fake") as span:
        pass
    with caplog.at_level(logging.WARNING, logger="opentelemetry"):
        span.set_mcp_request(request_id=9, protocol_version="2025-06-18")
    assert [record.getMessage() for record in caplog.records] == []
    (ended,) = memory.get_finished_spans()
    assert "jsonrpc.request.id" not in ended.attributes
