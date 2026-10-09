import asyncio
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from zeta.core.abort import AbortSignal
from zeta.mcp import (
    MCPConfig,
    MCPServerConfig,
    MCPTool,
    load_mcp_config,
    mount_mcp_servers,
    server_to_json,
)
from zeta.protocol.types import ToolCall
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry


class FilterClient:
    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self.protocol_version = "2025-06-18"
        self.capabilities = {"tools": {"listChanged": True}}
        self.tools = [
            MCPTool("read_file", "", {"type": "object"}),
            MCPTool("write_file", "", {"type": "object"}),
            MCPTool("delete_file", "", {"type": "object"}),
        ]
        self.calls: list[str] = []
        self.list_calls = 0
        self.notification_sink: Callable[[str], None] | None = None

    def set_failure_sink(self, _sink) -> None:
        pass

    def set_notification_sink(self, sink: Callable[[str], None] | None) -> None:
        self.notification_sink = sink

    async def connect(self) -> None:
        pass

    async def list_tools(self) -> list[MCPTool]:
        self.list_calls += 1
        return list(self.tools)

    async def call_tool(self, name: str, arguments, abort_signal: AbortSignal):
        del arguments, abort_signal
        self.calls.append(name)
        return {"content": [], "isError": False, "structuredContent": None}

    async def close(self) -> None:
        pass


@pytest.mark.parametrize(
    ("field", "value"),
    [("allowed_tools", "read_file"), ("disallowed_tools", ["", "write_file"])],
)
def test_mcp_tool_filter_config_rejects_invalid_lists(
    tmp_path: Path, field: str, value: object
) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"servers": {"files": {
        "transport": "stdio", "command": "files", field: value,
    }}}))

    config = load_mcp_config(path)

    assert "files" in config.malformed_servers
    assert field in (config.malformed_servers["files"].malformed_reason or "")


def test_mcp_tool_filter_config_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"servers": {"files": {
        "transport": "stdio",
        "command": "files",
        "allowed_tools": ["read_*", "search"],
        "disallowed_tools": ["read_secret"],
    }}}))

    server = load_mcp_config(path).servers["files"]

    assert server.allowed_tools == ("read_*", "search")
    assert server.disallowed_tools == ("read_secret",)
    assert server_to_json(server)["allowed_tools"] == ["read_*", "search"]
    assert server_to_json(server)["disallowed_tools"] == ["read_secret"]


async def _mount(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config: MCPServerConfig,
    *,
    tool_allow: tuple[str, ...] | None = None,
    tool_deny: tuple[str, ...] = (),
    notices: list[str] | None = None,
):
    registry = ToolRegistry(
        tmp_path,
        register_builtin=False,
        skill_catalog=SkillCatalog.empty(),
        tool_allow=tool_allow,
        tool_deny=tool_deny,
    )
    client = FilterClient(config)
    monkeypatch.setattr("zeta.mcp.mount._build_client", lambda _config: client)
    mount = await mount_mcp_servers(
        registry,
        MCPConfig(tmp_path / "mcp.json", {config.name: config}),
        notice_sink=None if notices is None else notices.append,
    )
    return registry, mount, client


@pytest.mark.asyncio
async def test_mcp_allowed_tools_filter_before_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = MCPServerConfig(
        "files", "stdio", "unused", allowed_tools=("read_*",)
    )
    registry, mount, _client = await _mount(monkeypatch, tmp_path, config)
    try:
        assert registry.registered_names == {"files__read_file"}
        assert [tool.name for tool in mount._actors["files"]._tools] == ["read_file"]
    finally:
        await mount.close()


@pytest.mark.asyncio
async def test_mcp_disallowed_tools_filter_and_global_policy_combine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = MCPServerConfig(
        "files", "stdio", "unused", disallowed_tools=("write_*",)
    )
    registry, mount, _client = await _mount(
        monkeypatch,
        tmp_path,
        config,
        tool_allow=("files__read_*", "files__write_*"),
        tool_deny=("files__delete_*",),
    )
    try:
        assert [tool.name for tool in mount._actors["files"]._tools] == [
            "read_file",
            "delete_file",
        ]
        assert registry.registered_names == {"files__read_file"}
    finally:
        await mount.close()


@pytest.mark.asyncio
async def test_mcp_unknown_filter_names_warn_and_zero_tools_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    notices: list[str] = []
    config = MCPServerConfig(
        "files",
        "stdio",
        "unused",
        allowed_tools=("missing",),
        disallowed_tools=("also_missing",),
    )
    registry, mount, _client = await _mount(
        monkeypatch, tmp_path, config, notices=notices
    )
    try:
        assert registry.registered_names == frozenset()
        assert any(
            "unknown allowed_tools: missing; unknown disallowed_tools: also_missing"
            in notice
            for notice in notices
        )
        assert any("mounted with zero tools" in notice for notice in notices)
    finally:
        await mount.close()


@pytest.mark.asyncio
async def test_mcp_filtered_tool_call_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = MCPServerConfig(
        "files", "stdio", "unused", allowed_tools=("read_file",)
    )
    registry, mount, client = await _mount(monkeypatch, tmp_path, config)
    try:
        actor = mount._actors["files"]
        result = await actor.call_tool(
            "write_file", {}, AbortSignal(), generation=actor.generation
        )
        assert result["isError"] is True
        assert "filtered" in result["content"][0]["text"]
        assert client.calls == []
        unknown = await registry.execute(ToolCall("old", "files__write_file", {}))
        assert unknown["isError"] is True
    finally:
        await mount.close()


@pytest.mark.asyncio
async def test_mcp_tools_list_changed_reapplies_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = MCPServerConfig(
        "files", "stdio", "unused", allowed_tools=("read_*",)
    )
    registry, mount, client = await _mount(monkeypatch, tmp_path, config)
    try:
        client.tools = [
            MCPTool("read_next", "", {"type": "object"}),
            MCPTool("write_next", "", {"type": "object"}),
        ]
        assert client.notification_sink is not None
        client.notification_sink("notifications/tools/list_changed")
        client.notification_sink("notifications/tools/list_changed")
        for _ in range(20):
            if "files__read_next" in registry.registered_names and client.list_calls == 3:
                break
            await asyncio.sleep(0)
        assert registry.registered_names == {"files__read_next"}
        assert client.list_calls == 3
    finally:
        await mount.close()
