from pathlib import Path

import pytest

from tests.test_mcp import _ListedClient
from zeta.mcp import MCPMount, MCPServerConfig, MCPTool
from zeta.mcp import mount as mount_module
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry


@pytest.mark.asyncio
async def test_remove_and_readd_server_updates_activation_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clients = iter(
        (
            _ListedClient(
                MCPServerConfig("server", "stdio", "unused"),
                [MCPTool("echo", "", {"type": "object"})],
            ),
            _ListedClient(
                MCPServerConfig("server", "stdio", "unused"),
                [MCPTool("echo", "", {"type": "object"})],
            ),
        )
    )
    # test-contract: seam _build_client is the existing MCP fixture boundary.
    monkeypatch.setattr(mount_module, "_build_client", lambda _config: next(clients))
    registry = ToolRegistry(
        tmp_path, register_builtin=False, skill_catalog=SkillCatalog.empty()
    )
    mount = MCPMount(registry, {}, {})
    try:
        config = MCPServerConfig("server", "stdio", "unused")
        await mount.add_server(config, source=tmp_path / "mcp.json")
        old_actor = mount._actors["server"]
        activated, rejected = mount.activate_tools(registry, ["server__echo"])
        assert activated == ["server__echo"]
        assert rejected == []

        await mount.remove_server("server")
        activated, rejected = mount.activate_tools(registry, ["server__echo"])
        assert activated == []
        assert rejected == ["server__echo: unavailable"]

        await mount.add_server(config, source=tmp_path / "mcp.json")
        new_actor = mount._actors["server"]
        assert new_actor is not old_actor
        activated, rejected = mount.activate_tools(registry, ["server__echo"])
        assert activated == ["server__echo"]
        assert rejected == []
        assert registry.is_mcp_owned("server__echo", new_actor, new_actor.generation)
    finally:
        await mount.close()
        await registry.close()
