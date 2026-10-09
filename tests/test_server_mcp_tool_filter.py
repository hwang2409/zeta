import json
from pathlib import Path

import pytest

from tests.support.fake_backend import FakeBackend
from tests.test_mcp_tool_filter import FilterClient
from zeta.mcp import MCPServerConfig
from zeta.server.runtime import ServerRuntime


@pytest.mark.asyncio
async def test_serve_mounts_mcp_server_tool_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / "mcp.json").write_text(json.dumps({"servers": {"google": {
        "transport": "stdio",
        "command": "unused",
        "allowed_tools": ["read_*"],
    }}}))
    clients: list[FilterClient] = []

    def build(config: MCPServerConfig) -> FilterClient:
        client = FilterClient(config)
        clients.append(client)
        return client

    monkeypatch.setattr("zeta.mcp.mount._build_client", build)
    runtime = ServerRuntime(
        home,
        cwd=tmp_path,
        provider="codex",
        backend_factory=lambda *_args, **_kwargs: (FakeBackend([]), "test-model"),
    )
    try:
        await runtime.create_session()
        await runtime.loop.ensure_mcp_servers()
        assert clients
        assert runtime.loop.tool_registry.registered_names >= {"google__read_file"}
        assert "google__write_file" not in runtime.loop.tool_registry.registered_names
    finally:
        await runtime.close()
