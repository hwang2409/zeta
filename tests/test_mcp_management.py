import asyncio
import json
import sys

import httpx
import pytest

from zeta.cli.main import build_parser, main
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.store import ConversationStore
from zeta.mcp.client import MCPHTTPError
from zeta.mcp.config import load_mcp_config_overlay
from zeta.mcp.http import StreamableHTTPMCPClient
from zeta.mcp.management import MCPManagementError, MCPManagementService
from zeta.mcp.mount import MCPMount, mount_mcp_servers
from zeta.protocol.types import TextContent
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry


def service(tmp_path, mount=None):
    return MCPManagementService(
        home=tmp_path / "home", project_dir=tmp_path / "repo", mount=mount
    )


def test_add_scopes_precedence_redaction_and_env_reference(tmp_path):
    manager = service(tmp_path)
    manager.add("shared", scope="user", url="https://example.test", env={"TOKEN": "${TOKEN}"})
    manager.add("shared", scope="project", command=sys.executable, args=("-c", "pass"))
    assert manager.show("shared").scope == "project"
    assert manager.show("shared", scope="user").config["env"]["TOKEN"] == "${TOKEN}"
    manager.remove("shared", scope="project")
    assert manager.show("shared").scope == "user"


def test_enable_disable_and_remove(tmp_path):
    manager = service(tmp_path)
    manager.add("x", scope="user", command=sys.executable)
    manager.set_enabled("x", scope="user", enabled=False)
    assert manager.show("x", scope="user").enabled is False
    manager.set_enabled("x", scope="user", enabled=True)
    manager.remove("x", scope="user")
    assert manager.list(scope="user") == []


def test_project_trust_is_invalidated_by_definition_change(tmp_path):
    manager = service(tmp_path)
    manager.add("x", scope="project", command=sys.executable, args=("-c", "pass"))
    assert manager.show("x", scope="project").trusted is False
    manager.trust("x")
    assert manager.show("x", scope="project").trusted is True
    manager.add("x", scope="project", command=sys.executable, args=("-c", "changed"))
    assert manager.show("x", scope="project").trusted is False


def test_literal_credentials_are_redacted(tmp_path):
    manager = service(tmp_path)
    manager.add("x", scope="user", url="https://example.test", oauth=True)
    path = manager.path("user")
    data = json.loads(path.read_text())
    data["servers"]["x"]["auth"]["token"] = "secret"
    path.write_text(json.dumps(data))
    assert "secret" not in json.dumps(manager.show("x", scope="user").as_json())


def test_trust_fingerprint_covers_every_security_field_and_untrusts(tmp_path):
    fields = {
        "command": "other",
        "args": ["other"],
        "env": {"TOKEN": "${OTHER}"},
        "url": "https://other.test",
        "headers": {"X-Key": "${OTHER}"},
        "auth": {"type": "oauth", "client_id": "other"},
        "client": {"setting": "other"},
    }
    baseline = {
        "transport": "stdio",
        "command": "cmd",
        "args": ["arg"],
        "env": {"TOKEN": "${TOKEN}"},
        "url": None,
        "headers": {},
        "auth": {},
        "client": {},
    }
    first = MCPManagementService.trust_fingerprint(baseline)
    for key, value in fields.items():
        changed = dict(baseline)
        changed[key] = value
        assert MCPManagementService.trust_fingerprint(changed) != first, key

    manager = service(tmp_path)
    manager.add("x", scope="project", command=sys.executable)
    manager.trust("x")
    manager.untrust("x")
    assert not manager.show("x", scope="project").trusted


def test_project_http_does_not_require_trust_but_stdio_does(tmp_path):
    manager = service(tmp_path)
    manager.add("http", scope="project", url="https://example.test")
    manager.add("local", scope="project", command=sys.executable)
    assert manager.show("http", scope="project").trusted
    assert not manager.show("local", scope="project").trusted


def test_clone_path_has_independent_trust_and_pending_stdio_is_filtered(tmp_path):
    first = MCPManagementService(home=tmp_path / "home", project_dir=tmp_path / "clone-a")
    second = MCPManagementService(home=tmp_path / "home", project_dir=tmp_path / "clone-b")
    first.add("local", scope="project", command=sys.executable)
    second.path("project").parent.mkdir(parents=True)
    second.path("project").write_text(first.path("project").read_text())
    first.trust("local")

    assert first.show("local").trusted
    assert not second.show("local").trusted
    filtered = second.runtime_config()
    assert "local" not in filtered.configured_servers


@pytest.mark.asyncio
async def test_untrusted_cloned_project_never_creates_subprocess(tmp_path, monkeypatch):
    manager = service(tmp_path)
    manager.add("local", scope="project", command=sys.executable, args=("server.py",))
    spawned = []

    async def spy(*args, **kwargs):
        spawned.append((args, kwargs))
        raise AssertionError("untrusted server spawned")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spy)
    (tmp_path / "registry").mkdir()
    registry = ToolRegistry(
        tmp_path / "registry", register_builtin=False, skill_catalog=SkillCatalog.empty()
    )
    mount = await mount_mcp_servers(registry, manager.runtime_config())
    assert spawned == []
    assert mount.configs == {}
    await mount.close()


def test_concurrent_mutations_do_not_lose_writers(tmp_path):
    manager = service(tmp_path)

    async def add(index):
        await asyncio.to_thread(
            manager.add,
            f"server-{index}",
            scope="user",
            command=sys.executable,
        )

    async def run_all():
        await asyncio.gather(*(add(index) for index in range(12)))

    asyncio.run(run_all())
    assert {item.name for item in manager.list(scope="user")} == {
        f"server-{index}" for index in range(12)
    }


def test_old_config_defaults_enabled_and_headers_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("API_KEY", "resolved")
    manager = service(tmp_path)
    manager.path("user").parent.mkdir(parents=True)
    manager.path("user").write_text(json.dumps({"servers": {"old": {
        "transport": "streamable-http", "url": "https://example.test",
        "headers": {"X-Key": "${API_KEY}"},
    }}}))
    config = load_mcp_config_overlay(home=tmp_path / "home")
    assert config.configured_servers["old"].enabled
    raw = json.loads(manager.path("user").read_text())
    assert raw["servers"]["old"]["headers"] == {"X-Key": "${API_KEY}"}


@pytest.mark.asyncio
async def test_missing_connect_time_header_is_named_without_leaking_values(
    tmp_path, monkeypatch
):
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"servers": {"http": {
        "transport": "streamable-http", "url": "https://example.test",
        "headers": {
            "Authorization": "Bearer ${PRESENT_SECRET}",
            "X-Missing": "${MISSING_SECRET}",
        },
    }}}))
    monkeypatch.setenv("PRESENT_SECRET", "must-not-leak")
    monkeypatch.delenv("MISSING_SECRET", raising=False)
    config = load_mcp_config_overlay(home=tmp_path, project_dir=None).servers["http"]
    client = StreamableHTTPMCPClient(config)

    with pytest.raises(MCPHTTPError) as caught:
        await client.connect()
    await client.close()

    message = str(caught.value)
    assert "MISSING_SECRET" in message
    assert "PRESENT_SECRET" not in message
    assert "must-not-leak" not in message


@pytest.mark.asyncio
async def test_env_header_resolved_at_connect_time(tmp_path, monkeypatch):
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"servers": {"http": {
        "transport": "streamable-http", "url": "https://example.test",
        "headers": {"X-Key": "${API_KEY}"},
    }}}))
    monkeypatch.setenv("API_KEY", "before")
    config = load_mcp_config_overlay(home=tmp_path, project_dir=None).servers["http"]
    assert config.headers == {"X-Key": "${API_KEY}"}
    monkeypatch.setenv("API_KEY", "at-connect")
    requests = []

    async def handler(request):
        requests.append(request)
        body = json.loads(request.content)
        if "id" not in body:
            return httpx.Response(202, request=request)
        result = (
            {"protocolVersion": "2025-06-18", "capabilities": {}}
            if body["method"] == "initialize"
            else {"tools": []}
        )
        return httpx.Response(200, request=request, json={
            "jsonrpc": "2.0", "id": body["id"], "result": result,
        })

    client = StreamableHTTPMCPClient(
        config, client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    await client.connect()
    assert requests[0].headers["X-Key"] == "at-connect"
    monkeypatch.setenv("API_KEY", "rotated")
    await client.list_tools()
    assert requests[-1].headers["X-Key"] == "rotated"
    assert config.headers == {"X-Key": "${API_KEY}"}
    assert json.loads(path.read_text())["servers"]["http"]["headers"] == {
        "X-Key": "${API_KEY}"
    }
    await client.close()


def test_service_honors_zeta_mcp_config_with_explicit_home(tmp_path, monkeypatch):
    override = tmp_path / "override.json"
    monkeypatch.setenv("ZETA_MCP_CONFIG", str(override))
    manager = MCPManagementService(home=tmp_path / "explicit-home")

    manager.add("override", scope="user", command=sys.executable)

    assert manager.path("user") == override
    assert manager.show("override", scope="user").name == "override"
    assert not (tmp_path / "explicit-home" / "mcp.json").exists()


def test_malformed_url_rejected_and_redaction_total(tmp_path):
    manager = service(tmp_path)
    malformed = "https://user:pass@example.test:bad/?token=secret"

    with pytest.raises(MCPManagementError, match="invalid MCP server URL"):
        manager.add("bad", scope="user", url=malformed)

    assert manager.list(scope="user") == []
    redacted = manager.redact({"url": malformed})["url"]
    assert redacted == "<redacted-url>"
    assert manager.redact({"url": "user:pass?token=secret"})["url"] == (
        "<redacted-url>"
    )
    assert "user" not in redacted and "secret" not in redacted


def test_redacts_env_headers_auth_and_url_credentials(tmp_path):
    manager = service(tmp_path)
    manager.add(
        "x", scope="user", url="https://user:pass@example.test/path?token=secret",
        env={"TOKEN": "literal", "REF": "${SAFE_REF}"},
        headers={"Authorization": "Bearer literal"}, oauth=True,
    )
    text = json.dumps(manager.show("x", scope="user").as_json())
    assert "literal" not in text
    assert "pass" not in text
    assert "secret" not in text
    assert "${SAFE_REF}" in text


def test_cli_parser_accepts_stdio_separator_and_all_management_commands():
    parser = build_parser()
    parsed = parser.parse_args(
        ["mcp", "add", "local", "--scope", "project", "--", "cmd", "arg"]
    )
    assert parsed.server_command == ["cmd", "arg"]
    option_like = parser.parse_args(
        [
            "mcp",
            "add",
            "local",
            "--scope",
            "project",
            "--",
            "npx",
            "-y",
            "@x/server",
            "--flag",
        ]
    )
    assert option_like.server_command == ["npx", "-y", "@x/server", "--flag"]
    for action in ("list", "show", "remove", "enable", "disable", "test", "login", "logout", "trust", "untrust"):
        args = ["mcp", action]
        if action != "list":
            args.append("server")
        if action in {"remove", "enable", "disable"}:
            args.extend(("--scope", "user"))
        assert parser.parse_args(args).mcp_action == action


@pytest.mark.parametrize(
    "raw",
    ["must-not-leak", ["must-not-leak"], None],
    ids=["string", "list", "null"],
)
def test_malformed_non_object_server_is_listed_safely_and_excluded(tmp_path, raw):
    manager = service(tmp_path)
    manager.path("user").parent.mkdir(parents=True)
    manager.path("user").write_text(json.dumps({"servers": {"broken": raw}}))

    listed = manager.list(scope="user")
    assert len(listed) == 1
    assert listed[0].name == "broken"
    assert listed[0].status == "malformed"
    assert "must-not-leak" not in json.dumps(listed[0].as_json())

    loaded = load_mcp_config_overlay(home=tmp_path / "home")
    assert "broken" in loaded.malformed_servers

    runtime = manager.runtime_config()
    assert "broken" not in runtime.configured_servers


@pytest.mark.asyncio
async def test_agent_turn_mounts_valid_server_alongside_malformed_definition(
    tmp_path, monkeypatch
):
    source = """import json,sys
for line in sys.stdin:
 r=json.loads(line); m=r.get('method')
 if m=='initialize': out={'protocolVersion':'2025-06-18','capabilities':{'tools':{}},'serverInfo':{}}
 elif m=='tools/list': out={'tools':[{'name':'echo','description':'echo','inputSchema':{'type':'object'}}]}
 else: continue
 print(json.dumps({'jsonrpc':'2.0','id':r.get('id'),'result':out}),flush=True)
"""
    home = tmp_path / "home"
    home.mkdir()
    (home / "mcp.json").write_text(json.dumps({"servers": {
        "broken": "must-not-leak",
        "working": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-c", source],
        },
    }}))
    monkeypatch.setenv("WIKI_AGENT_RUNTIME_DIR", str(tmp_path))
    loop = AgentLoop(
        FakeBackend([ScriptedTurn([TextContent("done")])]),
        ConversationStore(tmp_path / "sessions"),
        skill_catalog=SkillCatalog.empty(),
    )
    loop.set_mcp_scope(home=home)

    events = [event async for event in loop.run_turn("hello")]

    assert events[-1].type.value == "agent_end"
    assert loop._mcp_mount is not None
    assert set(loop._mcp_mount.configs) == {"working"}
    assert "working__echo" in loop.tool_registry.registered_names
    await loop.close()


def test_cli_json_shapes_redaction_and_error_exit(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ZETA_MCP_CONFIG", str(tmp_path / "user.json"))
    assert main(["mcp", "add", "remote", "--url", "https://u:p@example.test?q=secret", "--header", "Authorization=secret"]) == 0
    capsys.readouterr()
    assert main(["mcp", "list", "--json"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert isinstance(listing, list) and listing[0]["name"] == "remote"
    assert "secret" not in json.dumps(listing)
    assert main(["mcp", "show", "remote", "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["config"]["headers"]["Authorization"] == "<redacted>"
    assert main(["mcp", "show", "missing"]) == 2
    assert "unknown MCP server" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_live_sync_enable_disable_remove_and_failed_activation(tmp_path):
    source = """import json,sys
for line in sys.stdin:
 r=json.loads(line); m=r.get('method')
 if m=='initialize': out={'protocolVersion':'2025-06-18','capabilities':{'tools':{}},'serverInfo':{}}
 elif m=='tools/list': out={'tools':[{'name':'echo','description':'echo','inputSchema':{'type':'object'}}]}
 else: continue
 print(json.dumps({'jsonrpc':'2.0','id':r.get('id'),'result':out}),flush=True)
"""
    (tmp_path / "registry").mkdir()
    registry = ToolRegistry(
        tmp_path / "registry", register_builtin=False, skill_catalog=SkillCatalog.empty()
    )
    mount = MCPMount(registry, {}, {}, home=str(tmp_path / "home"))
    manager = service(tmp_path, mount)
    manager.add("live", scope="user", command=sys.executable, args=("-c", source))
    await manager.sync_runtime()
    assert "live__echo" in registry.registered_names

    manager.set_enabled("live", scope="user", enabled=False)
    await manager.sync_runtime()
    assert "live__echo" not in registry.registered_names
    assert manager.show("live", scope="user").status == "disabled"

    manager.set_enabled("live", scope="user", enabled=True)
    await manager.sync_runtime()
    manager.remove("live", scope="user")
    await manager.sync_runtime()
    assert "live" not in mount.configs

    manager.add("broken", scope="user", command=str(tmp_path / "missing"))
    await manager.sync_runtime()
    assert manager.show("broken", scope="user").status in {"degraded", "failed"}
    assert "broken" in json.loads(manager.path("user").read_text())["servers"]
    await mount.close()


@pytest.mark.asyncio
async def test_activate_starts_slow_mcp_mount_without_blocking_first_render(tmp_path, monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_mount(registry, config=None, *, notice_sink=None, home=None):
        del config, notice_sink
        started.set()
        await release.wait()
        return MCPMount(registry, {}, {}, home=home)

    monkeypatch.setattr("zeta.runtime.loop.mcp_session.mount_mcp_servers", slow_mount)
    loop = AgentLoop(
        FakeBackend([]), ConversationStore(tmp_path / "sessions"),
        skill_catalog=SkillCatalog.empty(),
    )
    await loop.activate()
    await started.wait()
    assert loop._mcp_mount_task is not None and not loop._mcp_mount_task.done()
    release.set()
    await loop.ensure_mcp_servers()
    assert await loop.slash_mcp("status") == await loop.slash_mcp("")
    await loop.close()


@pytest.mark.asyncio
async def test_login_uses_the_explicit_scope_when_project_shadows_user(
    tmp_path, monkeypatch
):
    manager = service(tmp_path)
    manager.add(
        "oauth", scope="user", url="https://user.example.test", oauth=True
    )
    manager.add(
        "oauth", scope="project", url="https://project.example.test", oauth=True
    )
    calls = []

    async def fake_authorize(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("zeta.mcp.oauth.authorize", fake_authorize)
    await manager.login("oauth", scope="user")

    assert calls[0]["server_name"] == "oauth"
    assert calls[0]["server_url"] == "https://user.example.test"


def test_logout_validates_explicit_scope_without_removing_definition(
    tmp_path, monkeypatch
):
    manager = service(tmp_path)
    manager.add("oauth", scope="project", url="https://project.example.test", oauth=True)
    deleted = []
    monkeypatch.setattr(
        "zeta.mcp.oauth_store.delete_token",
        lambda name, *, home: deleted.append((name, home)),
    )

    with pytest.raises(MCPManagementError, match="unknown MCP server"):
        manager.logout("oauth", scope="user")
    assert deleted == []

    manager.logout("oauth", scope="project")
    assert deleted == [("oauth", manager.home)]
    assert manager.show("oauth", scope="project").name == "oauth"


@pytest.mark.parametrize(
    ("scope", "enabled"), (("user", False), ("project", True))
)
def test_stdio_test_only_lists_tools_without_changing_definition_state(
    tmp_path, scope, enabled
):
    source = "import json,sys\nfor line in sys.stdin:\n r=json.loads(line); m=r.get('method')\n if m=='tools/call': raise RuntimeError('test must not call tools')\n result={'protocolVersion':'2025-06-18','capabilities':{'tools':{}},'serverInfo':{}} if m=='initialize' else {'tools':[{'name':'z-last','description':'z description','inputSchema':{}},{'name':'a-first','description':'" + ("x" * 200) + "','inputSchema':{}}]}\n print(json.dumps({'jsonrpc':'2.0','id':r.get('id'),'result':result}),flush=True)"
    manager = service(tmp_path)
    manager.add(
        "x", scope=scope, command=sys.executable, args=("-c", source),
        enabled=enabled,
    )
    path = manager.path(scope)
    before = path.read_bytes()
    before_item = manager.show("x", scope=scope)

    result = asyncio.run(manager.test("x", scope=scope))

    assert result == {
        "name": "x",
        "tools": [
            {"name": "a-first", "description": "x" * 117 + "..."},
            {"name": "z-last", "description": "z description"},
        ],
        "status": "ok",
    }
    assert path.read_bytes() == before
    after_item = manager.show("x", scope=scope)
    assert after_item.enabled is before_item.enabled
    assert after_item.trusted is before_item.trusted
    if scope == "project":
        assert after_item.trusted is False
