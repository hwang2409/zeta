import asyncio
import fcntl
import inspect
import json
import sys
import threading
from pathlib import Path

import httpx
import pytest

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from zeta.cli.main import build_parser, main
from zeta.core.store import ConversationStore
from zeta.mcp.client import MCPHTTPError
from zeta.mcp.config import load_mcp_config_overlay
from zeta.mcp.http import StreamableHTTPMCPClient
from zeta.mcp.management import MCPManagementError, MCPManagementService
from zeta.mcp.mount import MCPMount, mount_mcp_servers
from zeta.protocol.types import MessageOrigin, TextContent
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry


def service(tmp_path, mount=None):
    return MCPManagementService(
        home=tmp_path / "home", project_dir=tmp_path / "repo", mount=mount
    )


@pytest.mark.asyncio
async def test_add_scopes_precedence_redaction_and_env_reference(tmp_path):
    manager = service(tmp_path)
    await manager.add(
        "shared", scope="user", url="https://example.test", env={"TOKEN": "${TOKEN}"}
    )
    await manager.add(
        "shared", scope="project", command=sys.executable, args=("-c", "pass")
    )
    assert manager.show("shared").scope == "project"
    assert manager.show("shared", scope="user").config["env"]["TOKEN"] == "${TOKEN}"
    await manager.remove("shared", scope="project")
    assert manager.show("shared").scope == "user"


@pytest.mark.asyncio
async def test_enable_disable_and_remove(tmp_path):
    manager = service(tmp_path)
    await manager.add("x", scope="user", command=sys.executable)
    await manager.set_enabled("x", scope="user", enabled=False)
    assert manager.show("x", scope="user").enabled is False
    await manager.set_enabled("x", scope="user", enabled=True)
    await manager.remove("x", scope="user")
    assert [item for item in manager.snapshot.entries if item.scope == "user"] == []


@pytest.mark.asyncio
async def test_project_trust_is_invalidated_by_definition_change(tmp_path):
    manager = service(tmp_path)
    await manager.add("x", scope="project", command=sys.executable, args=("-c", "pass"))
    assert manager.show("x", scope="project").trusted is False
    await manager.set_trusted("x", trusted=True)
    assert manager.show("x", scope="project").trusted is True
    path = manager.path("project")
    document = json.loads(path.read_text())
    document["servers"]["x"]["args"] = ["-c", "changed"]
    path.write_text(json.dumps(document))
    await manager.refresh()
    assert manager.show("x", scope="project").trusted is False


@pytest.mark.asyncio
async def test_literal_credentials_are_redacted(tmp_path):
    manager = service(tmp_path)
    await manager.add("x", scope="user", url="https://example.test", oauth=True)
    path = manager.path("user")
    data = json.loads(path.read_text())
    data["servers"]["x"]["auth"]["token"] = "secret"
    path.write_text(json.dumps(data))
    await manager.refresh()
    assert "secret" not in json.dumps(manager.show("x", scope="user").as_json())


@pytest.mark.asyncio
async def test_trust_fingerprint_covers_every_security_field_and_untrusts(tmp_path):
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
    await manager.add("x", scope="project", command=sys.executable)
    await manager.set_trusted("x", trusted=True)
    await manager.set_trusted("x", trusted=False)
    assert not manager.show("x", scope="project").trusted


@pytest.mark.asyncio
async def test_project_http_does_not_require_trust_but_stdio_does(tmp_path):
    manager = service(tmp_path)
    await manager.add("http", scope="project", url="https://example.test")
    await manager.add("local", scope="project", command=sys.executable)
    assert manager.show("http", scope="project").trusted
    assert not manager.show("local", scope="project").trusted


@pytest.mark.asyncio
async def test_clone_path_has_independent_trust_and_pending_stdio_is_filtered(tmp_path):
    first = MCPManagementService(
        home=tmp_path / "home", project_dir=tmp_path / "clone-a"
    )
    second = MCPManagementService(
        home=tmp_path / "home", project_dir=tmp_path / "clone-b"
    )
    await first.add("local", scope="project", command=sys.executable)
    second.path("project").parent.mkdir(parents=True)
    second.path("project").write_text(first.path("project").read_text())
    await first.set_trusted("local", trusted=True)

    assert first.show("local").trusted
    await second.refresh()
    assert not second.show("local").trusted
    filtered = await second.runtime_config()
    assert "local" not in filtered.configured_servers


@pytest.mark.asyncio
async def test_untrusted_cloned_project_never_creates_subprocess(tmp_path, monkeypatch):
    manager = service(tmp_path)
    await manager.add(
        "local", scope="project", command=sys.executable, args=("server.py",)
    )
    spawned = []

    async def spy(*args, **kwargs):
        spawned.append((args, kwargs))
        raise AssertionError("untrusted server spawned")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spy)
    (tmp_path / "registry").mkdir()
    registry = ToolRegistry(
        tmp_path / "registry",
        register_builtin=False,
        skill_catalog=SkillCatalog.empty(),
    )
    mount = await mount_mcp_servers(registry, await manager.runtime_config())
    assert spawned == []
    assert mount.configs == {}
    await mount.close()


@pytest.mark.parametrize(
    ("route", "gated_read", "expected_reconciles"),
    (
        ("sync_runtime", 1, 2),
        ("refresh", 1, 2),
        ("refresh_credentials", 2, 3),
    ),
)
@pytest.mark.asyncio
async def test_management_routes_cannot_publish_stale_candidates(
    tmp_path, monkeypatch, route, gated_read, expected_reconciles
):
    manager = service(tmp_path)
    stale_read = threading.Event()
    release_stale = threading.Event()
    original_transaction = manager._transaction
    read_count = 0
    read_count_lock = threading.Lock()

    def gated_transaction(mutation=None):
        nonlocal read_count
        candidate = original_transaction(mutation)
        if mutation is None:
            with read_count_lock:
                read_count += 1
                should_gate = read_count == gated_read
            if should_gate:
                stale_read.set()
                assert release_stale.wait(10), "timed out releasing stale transaction"
        return candidate

    reconciled = []

    async def record_reconcile(desired):
        reconciled.append(set(desired.configured_servers))

    # test-contract: seam force the transaction-to-publication interleaving
    monkeypatch.setattr(manager, "_transaction", gated_transaction)
    # test-contract: seam observe runtime reconciliation order without a real server
    monkeypatch.setattr(manager, "_reconcile", record_reconcile)
    route_task = asyncio.create_task(getattr(manager, route)())
    add_task = None
    try:
        assert await asyncio.to_thread(stale_read.wait, 10), "transaction did not start"
        add_task = asyncio.create_task(
            manager.add("new", scope="user", command=sys.executable)
        )
        if not manager._transaction_gate.locked():
            await asyncio.wait_for(asyncio.shield(add_task), 10)
        release_stale.set()
        await asyncio.wait_for(route_task, 10)
        await asyncio.wait_for(add_task, 10)
    finally:
        release_stale.set()
        for task in (route_task, add_task):
            if task is not None and not task.done():
                task.cancel()
                await asyncio.wait_for(
                    asyncio.gather(task, return_exceptions=True), 10
                )

    durable = set(json.loads(manager.path("user").read_text())["servers"])
    assert {item.name for item in manager.snapshot.entries} == durable
    assert reconciled == [set()] * (expected_reconciles - 1) + [durable]


@pytest.mark.parametrize("cancel_at", ("gate", "transaction", "reconcile"))
@pytest.mark.asyncio
async def test_cancelled_management_owner_keeps_old_snapshot(
    tmp_path, monkeypatch, cancel_at
):
    manager = service(tmp_path)
    await manager.refresh()
    before = manager.snapshot
    release_gate = False
    release_thread = threading.Event()
    thread_finished = threading.Event()
    release_reconcile = asyncio.Event()

    if cancel_at == "gate":
        await manager._transaction_gate.acquire()
        release_gate = True
    elif cancel_at == "transaction":
        started = threading.Event()
        original_transaction = manager._transaction

        def blocked_transaction(mutation=None):
            try:
                candidate = original_transaction(mutation)
                started.set()
                assert release_thread.wait(10), "timed out releasing transaction"
                return candidate
            finally:
                thread_finished.set()

        # test-contract: seam pause the worker transaction await before publication
        monkeypatch.setattr(manager, "_transaction", blocked_transaction)
    else:
        started_async = asyncio.Event()

        async def blocked_reconcile(_desired):
            started_async.set()
            await asyncio.wait_for(release_reconcile.wait(), 10)

        # test-contract: seam pause runtime reconciliation before publication
        monkeypatch.setattr(manager, "_reconcile", blocked_reconcile)

    entered = asyncio.Event()

    async def refresh():
        entered.set()
        return await manager.refresh()

    task = asyncio.create_task(refresh())
    try:
        await asyncio.wait_for(entered.wait(), 10)
        if cancel_at == "transaction":
            assert await asyncio.to_thread(started.wait, 10), "transaction did not start"
        elif cancel_at == "reconcile":
            await asyncio.wait_for(started_async.wait(), 10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 10)
    finally:
        if release_gate:
            manager._transaction_gate.release()
        release_thread.set()
        release_reconcile.set()
        if cancel_at == "transaction":
            assert await asyncio.wait_for(
                asyncio.to_thread(thread_finished.wait, 10), 10
            )
        if not task.done():
            task.cancel()
            await asyncio.wait_for(
                asyncio.gather(task, return_exceptions=True), 10
            )

    assert manager.snapshot is before


@pytest.mark.asyncio
async def test_reconcile_failure_does_not_publish_candidate(tmp_path, monkeypatch):
    manager = service(tmp_path)
    await manager.refresh()
    before = manager.snapshot

    async def fail_reconcile(_desired):
        raise RuntimeError("reconcile failed")

    # test-contract: seam inject a runtime reconciliation failure
    monkeypatch.setattr(manager, "_reconcile", fail_reconcile)
    with pytest.raises(RuntimeError, match="reconcile failed"):
        await manager.add("new", scope="user", command=sys.executable)

    assert manager.snapshot is before


@pytest.mark.asyncio
async def test_concurrent_mutations_do_not_lose_writers(tmp_path):
    manager = service(tmp_path)

    async def add(index):
        await manager.add(f"server-{index}", scope="user", command=sys.executable)

    async def run_all():
        await asyncio.wait_for(
            asyncio.gather(*(add(index) for index in range(12))), 10
        )

    await run_all()
    assert {
        item.name
        for item in [item for item in manager.snapshot.entries if item.scope == "user"]
    } == {f"server-{index}" for index in range(12)}


def test_old_config_defaults_enabled_and_headers_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("API_KEY", "resolved")
    manager = service(tmp_path)
    manager.path("user").parent.mkdir(parents=True)
    manager.path("user").write_text(
        json.dumps(
            {
                "servers": {
                    "old": {
                        "transport": "streamable-http",
                        "url": "https://example.test",
                        "headers": {"X-Key": "${API_KEY}"},
                    }
                }
            }
        )
    )
    config = load_mcp_config_overlay(home=tmp_path / "home")
    assert config.configured_servers["old"].enabled
    raw = json.loads(manager.path("user").read_text())
    assert raw["servers"]["old"]["headers"] == {"X-Key": "${API_KEY}"}


@pytest.mark.asyncio
async def test_missing_connect_time_header_is_named_without_leaking_values(
    tmp_path, monkeypatch
):
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "servers": {
                    "http": {
                        "transport": "streamable-http",
                        "url": "https://example.test",
                        "headers": {
                            "Authorization": "Bearer ${PRESENT_SECRET}",
                            "X-Missing": "${MISSING_SECRET}",
                        },
                    }
                }
            }
        )
    )
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
    path.write_text(
        json.dumps(
            {
                "servers": {
                    "http": {
                        "transport": "streamable-http",
                        "url": "https://example.test",
                        "headers": {"X-Key": "${API_KEY}"},
                    }
                }
            }
        )
    )
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
        return httpx.Response(
            200,
            request=request,
            json={
                "jsonrpc": "2.0",
                "id": body["id"],
                "result": result,
            },
        )

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


@pytest.mark.asyncio
async def test_service_honors_zeta_mcp_config_with_explicit_home(tmp_path, monkeypatch):
    override = tmp_path / "override.json"
    monkeypatch.setenv("ZETA_MCP_CONFIG", str(override))
    manager = MCPManagementService(home=tmp_path / "explicit-home")

    await manager.add("override", scope="user", command=sys.executable)

    assert manager.path("user") == override
    assert manager.show("override", scope="user").name == "override"
    assert not (tmp_path / "explicit-home" / "mcp.json").exists()


@pytest.mark.asyncio
async def test_malformed_url_rejected_and_redaction_total(tmp_path):
    manager = service(tmp_path)
    malformed = "https://user:pass@example.test:bad/?token=secret"

    with pytest.raises(MCPManagementError, match="invalid MCP server URL"):
        await manager.add("bad", scope="user", url=malformed)

    assert [item for item in manager.snapshot.entries if item.scope == "user"] == []
    redacted = manager.redact({"url": malformed})["url"]
    assert redacted == "<redacted-url>"
    assert manager.redact({"url": "user:pass?token=secret"})["url"] == (
        "<redacted-url>"
    )
    assert "user" not in redacted and "secret" not in redacted


@pytest.mark.asyncio
async def test_redacts_env_headers_auth_and_url_credentials(tmp_path):
    manager = service(tmp_path)
    await manager.add(
        "x",
        scope="user",
        url="https://user:pass@example.test/path?token=secret",
        env={"TOKEN": "literal", "REF": "${SAFE_REF}"},
        headers={"Authorization": "Bearer literal"},
        oauth=True,
    )
    text = json.dumps(manager.show("x", scope="user").as_json())
    assert "literal" not in text
    assert "pass" not in text
    assert "secret" not in text
    assert "${SAFE_REF}" in text


def test_non_mcp_subcommand_double_dash_is_not_swallowed():
    parser = build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["session", "rename", "mcp", "add", "--", "x"])

    assert exc_info.value.code == 2


def test_top_level_double_dash_prompt_unchanged():
    parser = build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--", "prompt"])

    assert exc_info.value.code == 2


def test_mcp_add_after_global_options_parses_command_tail():
    parser = build_parser()

    parsed = parser.parse_args(
        [
            "--model",
            "m",
            "mcp",
            "add",
            "n",
            "--scope",
            "user",
            "--",
            "npx",
            "-y",
            "pkg",
            "--flag",
        ]
    )

    assert parsed.server_command == ["npx", "-y", "pkg", "--flag"]


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
    for action in (
        "list",
        "show",
        "remove",
        "enable",
        "disable",
        "test",
        "login",
        "logout",
        "trust",
        "untrust",
    ):
        args = ["mcp", action]
        if action != "list":
            args.append("server")
        if action in {"remove", "enable", "disable"}:
            args.extend(("--scope", "user"))
        assert parser.parse_args(args).mcp_action == action


def test_mcp_add_without_command_or_url_errors(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ZETA_MCP_CONFIG", str(tmp_path / "mcp.json"))

    assert main(["mcp", "add", "name"]) == 2
    assert "specify exactly one of command or url" in capsys.readouterr().err


@pytest.mark.parametrize(
    "raw",
    ["must-not-leak", ["must-not-leak"], None],
    ids=["string", "list", "null"],
)
@pytest.mark.asyncio
async def test_malformed_non_object_server_is_listed_safely_and_excluded(tmp_path, raw):
    manager = service(tmp_path)
    manager.path("user").parent.mkdir(parents=True)
    manager.path("user").write_text(json.dumps({"servers": {"broken": raw}}))

    await manager.refresh()
    listed = [item for item in manager.snapshot.entries if item.scope == "user"]
    assert len(listed) == 1
    assert listed[0].name == "broken"
    assert listed[0].status == "malformed"
    assert listed[0].trusted is False
    assert "must-not-leak" not in json.dumps(listed[0].as_json())

    loaded = load_mcp_config_overlay(home=tmp_path / "home")
    assert "broken" in loaded.malformed_servers

    runtime = await manager.runtime_config()
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
    (home / "mcp.json").write_text(
        json.dumps(
            {
                "servers": {
                    "broken": "must-not-leak",
                    "working": {
                        "transport": "stdio",
                        "command": sys.executable,
                        "args": ["-c", source],
                    },
                }
            }
        )
    )
    monkeypatch.setenv("WIKI_AGENT_RUNTIME_DIR", str(tmp_path))
    loop = AgentLoop(
        FakeBackend([ScriptedTurn([TextContent("done")])]),
        ConversationStore(tmp_path / "sessions"),
        skill_catalog=SkillCatalog.empty(),
    )
    loop.set_mcp_scope(home=home)

    events = [
        event async for event in loop.run_turn("hello", origin=MessageOrigin.USER)
    ]

    assert events[-1].type.value == "agent_end"
    assert loop._mcp_mount is not None
    assert set(loop._mcp_mount.configs) == {"working"}
    assert "working__echo" in loop.tool_registry.registered_names
    await loop.close()


def test_cli_json_shapes_redaction_and_error_exit(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ZETA_MCP_CONFIG", str(tmp_path / "user.json"))
    assert (
        main(
            [
                "mcp",
                "add",
                "remote",
                "--url",
                "https://u:p@example.test?q=secret",
                "--header",
                "Authorization=secret",
            ]
        )
        == 0
    )
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
        tmp_path / "registry",
        register_builtin=False,
        skill_catalog=SkillCatalog.empty(),
    )
    mount = MCPMount(registry, {}, {}, home=str(tmp_path / "home"))
    manager = service(tmp_path, mount)
    await manager.add("live", scope="user", command=sys.executable, args=("-c", source))
    await manager.sync_runtime()
    assert "live__echo" in registry.registered_names

    await manager.set_enabled("live", scope="user", enabled=False)
    await manager.sync_runtime()
    assert "live__echo" not in registry.registered_names
    assert manager.show("live", scope="user").status == "disabled"

    await manager.set_enabled("live", scope="user", enabled=True)
    await manager.sync_runtime()
    await manager.remove("live", scope="user")
    await manager.sync_runtime()
    assert "live" not in mount.configs

    await manager.add("broken", scope="user", command=str(tmp_path / "missing"))
    await manager.sync_runtime()
    assert manager.show("broken", scope="user").status in {"degraded", "failed"}
    assert "broken" in json.loads(manager.path("user").read_text())["servers"]
    await mount.close()


@pytest.mark.asyncio
async def test_activate_starts_slow_mcp_mount_without_blocking_first_render(
    tmp_path, monkeypatch
):
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_mount(registry, config=None, *, notice_sink=None, home=None):
        del config, notice_sink
        started.set()
        await asyncio.wait_for(release.wait(), 10)
        return MCPMount(registry, {}, {}, home=home)

    monkeypatch.setattr("zeta.runtime.loop.mcp_session.mount_mcp_servers", slow_mount)
    loop = AgentLoop(
        FakeBackend([]),
        ConversationStore(tmp_path / "sessions"),
        skill_catalog=SkillCatalog.empty(),
    )
    await loop.activate()
    try:
        await asyncio.wait_for(started.wait(), 10)
        assert loop._mcp_mount_task is not None and not loop._mcp_mount_task.done()
        release.set()
        await loop.ensure_mcp_servers()
        assert await loop.slash_mcp("status") == await loop.slash_mcp("")
    finally:
        release.set()
        if loop._mcp_mount_task is not None:
            loop._mcp_mount_task.cancel()
            await asyncio.wait_for(
                asyncio.gather(loop._mcp_mount_task, return_exceptions=True), 10
            )
        await loop.close()


@pytest.mark.asyncio
async def test_login_uses_the_explicit_scope_when_project_shadows_user(
    tmp_path, monkeypatch
):
    manager = service(tmp_path)
    await manager.add(
        "oauth", scope="user", url="https://user.example.test", oauth=True
    )
    await manager.add(
        "oauth", scope="project", url="https://project.example.test", oauth=True
    )
    calls = []

    async def fake_authorize(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("zeta.mcp.oauth.authorize", fake_authorize)
    await manager.login("oauth", scope="user")

    assert calls[0]["server_name"] == "oauth"
    assert calls[0]["server_url"] == "https://user.example.test"


@pytest.mark.asyncio
async def test_login_skips_oauth_flow_blocked_by_session_policy(tmp_path, monkeypatch):
    registry = ToolRegistry(
        tmp_path,
        register_builtin=False,
        skill_catalog=SkillCatalog.empty(),
        tool_allow=("remote__*",),
    )
    mount = MCPMount(registry, {})
    manager = service(tmp_path, mount=mount)
    await manager.add("oauth", scope="user", url="https://example.test", oauth=True)

    async def forbidden_authorize(**_kwargs):
        raise AssertionError("blocked OAuth flow started")

    monkeypatch.setattr("zeta.mcp.oauth.authorize", forbidden_authorize)
    try:
        with pytest.raises(MCPManagementError, match="skipped by tool policy"):
            await manager.login("oauth", scope="user")
    finally:
        await mount.close()
        await registry.close()


@pytest.mark.asyncio
async def test_logout_validates_explicit_scope_without_removing_definition(tmp_path):
    manager = service(tmp_path)
    await manager.add(
        "oauth", scope="project", url="https://project.example.test", oauth=True
    )

    with pytest.raises(MCPManagementError, match="unknown MCP server"):
        await manager.logout("oauth", scope="user")

    await manager.logout("oauth", scope="project")
    token_file = manager.home / "mcp-tokens" / "oauth.json"
    assert json.loads(token_file.read_text()) == {"generation": 1, "token": None}
    assert manager.show("oauth", scope="project").name == "oauth"


@pytest.mark.asyncio
async def test_management_test_skips_server_blocked_by_session_policy(tmp_path):
    marker = tmp_path / "spawned"
    source = (
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).touch()\n"
        "import time\ntime.sleep(30)\n"
    )
    registry = ToolRegistry(
        tmp_path,
        register_builtin=False,
        skill_catalog=SkillCatalog.empty(),
        tool_allow=("remote__*",),
    )
    mount = MCPMount(registry, {})
    manager = service(tmp_path, mount=mount)
    await manager.add("x", scope="user", command=sys.executable, args=("-c", source))
    try:
        result = await manager.test("x", scope="user")

        assert result == {
            "name": "x",
            "tools": [],
            "status": "skipped-policy",
            "detail": "skipped by tool policy",
        }
        assert not marker.exists()
    finally:
        await mount.close()
        await registry.close()


@pytest.mark.parametrize(("scope", "enabled"), (("user", False), ("project", True)))
@pytest.mark.asyncio
async def test_stdio_test_only_lists_tools_without_changing_definition_state(
    tmp_path, scope, enabled
):
    source = (
        "import json,sys\nfor line in sys.stdin:\n r=json.loads(line); m=r.get('method')\n if m=='tools/call': raise RuntimeError('test must not call tools')\n result={'protocolVersion':'2025-06-18','capabilities':{'tools':{}},'serverInfo':{}} if m=='initialize' else {'tools':[{'name':'z-last','description':'z description','inputSchema':{}},{'name':'a-first','description':'"
        + ("x" * 200)
        + "','inputSchema':{}}]}\n print(json.dumps({'jsonrpc':'2.0','id':r.get('id'),'result':result}),flush=True)"
    )
    manager = service(tmp_path)
    await manager.add(
        "x",
        scope=scope,
        command=sys.executable,
        args=("-c", source),
        enabled=enabled,
    )
    path = manager.path(scope)
    before = path.read_bytes()
    before_item = manager.show("x", scope=scope)

    if scope == "project":
        await manager.set_trusted("x", trusted=True)
        before = path.read_bytes()
        before_item = manager.show("x", scope=scope)

    result = await manager.test("x", scope=scope)

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
        assert after_item.trusted is True


@pytest.mark.asyncio
async def test_untrusted_project_stdio_server_is_not_spawned_by_test(tmp_path):
    marker = tmp_path / "SPAWNED"
    source = f"open({str(marker)!r}, 'w').close()"
    manager = service(tmp_path)
    await manager.add(
        "untrusted",
        scope="project",
        command=sys.executable,
        args=("-c", source),
        enabled=True,
    )
    assert manager.show("untrusted", scope="project").trusted is False

    result = await manager.test("untrusted", scope="project")

    assert result["status"] == "untrusted"
    assert result["tools"] == []
    assert not marker.exists()


@pytest.mark.asyncio
async def test_constructor_and_scope_setup_do_not_resolve_paths_on_loop_thread(
    tmp_path, monkeypatch
):
    loop_thread = threading.get_ident()
    loop = AgentLoop(
        FakeBackend([]),
        ConversationStore(tmp_path / "sessions"),
        skill_catalog=SkillCatalog.empty(),
    )
    calls = []
    original_resolve = Path.resolve
    original_expanduser = Path.expanduser

    def record_resolve(path, *args, **kwargs):
        calls.append(("resolve", threading.get_ident()))
        return original_resolve(path, *args, **kwargs)

    def record_expanduser(path, *args, **kwargs):
        calls.append(("expanduser", threading.get_ident()))
        return original_expanduser(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", record_resolve)
    monkeypatch.setattr(Path, "expanduser", record_expanduser)
    manager = MCPManagementService(
        home=tmp_path / "home", project_dir=tmp_path / "repo"
    )
    assert calls == []
    loop.set_mcp_scope(home=tmp_path / "home", project_dir=tmp_path / "repo")

    assert calls == []
    await manager.refresh()
    assert {operation for operation, _thread in calls} == {"expanduser", "resolve"}
    assert all(thread != loop_thread for _operation, thread in calls)
    await loop.close()


@pytest.mark.asyncio
async def test_contended_config_lock_fails_off_loop_with_nonblocking_flock(
    tmp_path, monkeypatch
):
    manager = MCPManagementService(
        home=tmp_path / "home", project_dir=tmp_path / "repo"
    )
    manager._lock_timeout = 0
    path = (tmp_path / "home" / "mcp.json").resolve()
    path.parent.mkdir(parents=True)
    locks = [
        path.with_name(f".{path.name}.lock").open("a+"),
        path.with_name(f"{path.name}.lock").open("a+"),
    ]
    for lock in locks:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    loop_thread = threading.get_ident()
    calls = []
    original = fcntl.flock

    def record_flock(fd, operation):
        calls.append((threading.get_ident(), operation))
        if operation & fcntl.LOCK_EX and not operation & fcntl.LOCK_NB:
            return None
        return original(fd, operation)

    monkeypatch.setattr(fcntl, "flock", record_flock)
    try:
        result = manager.add("busy", scope="user", command=sys.executable)
        if not inspect.isawaitable(result):
            assert calls and all(thread != loop_thread for thread, _operation in calls)
        else:
            with pytest.raises(MCPManagementError, match="mcp busy"):
                await asyncio.wait_for(result, 10)
    finally:
        for lock in locks:
            original(lock.fileno(), fcntl.LOCK_UN)
            lock.close()

    attempted = [call for call in calls if call[1] & fcntl.LOCK_EX]
    assert attempted
    assert all(thread != loop_thread for thread, _operation in attempted)
    assert all(operation & fcntl.LOCK_NB for _thread, operation in attempted)


@pytest.mark.parametrize("failure_at", ("transaction", "reconcile"))
@pytest.mark.asyncio
async def test_duplicate_add_repairs_a_committed_unpublished_mutation(
    tmp_path, monkeypatch, failure_at
):
    class OneFailureMount:
        pinned = False

        def __init__(self):
            self.configs = {}
            self.statuses = {}
            self._actors = {}
            self.fail = failure_at == "reconcile"

        async def add_server(self, config, **_kwargs):
            if self.fail:
                self.fail = False
                raise RuntimeError("mount failed")
            self.configs[config.name] = config

        async def remove_server(self, name):
            self.configs.pop(name)

    mount = OneFailureMount()
    manager = service(tmp_path, mount)
    assert inspect.iscoroutinefunction(manager.add)
    before = manager.snapshot
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    original_transaction = manager._transaction

    def gated_transaction(mutation=None):
        candidate = original_transaction(mutation)
        if mutation is not None and not started.is_set():
            started.set()
            try:
                assert release.wait(10), "timed out releasing committed transaction"
            finally:
                finished.set()
        return candidate

    if failure_at == "transaction":
        # test-contract: durability cancel the public call after its commit
        monkeypatch.setattr(manager, "_transaction", gated_transaction)

    task = asyncio.create_task(
        manager.add("durable", scope="user", url="https://example.test")
    )
    try:
        if failure_at == "transaction":
            assert await asyncio.to_thread(started.wait, 10), "transaction did not commit"
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 10)
        else:
            with pytest.raises(RuntimeError, match="mount failed"):
                await asyncio.wait_for(task, 10)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 10)
        if failure_at == "transaction":
            assert await asyncio.to_thread(finished.wait, 10), "transaction did not finish"

    assert manager.snapshot is before
    assert "durable" in json.loads(manager.path("user").read_text())["servers"]

    snapshot = await manager.add(
        "durable", scope="user", url="https://example.test"
    )
    assert [item.name for item in snapshot.entries] == ["durable"]
    assert set(mount.configs) == {"durable"}
    with pytest.raises(MCPManagementError, match="already configured"):
        await manager.add("durable", scope="user", url="https://different.test")


@pytest.mark.asyncio
async def test_published_snapshot_is_recursively_immutable(tmp_path):
    manager = service(tmp_path)
    assert inspect.iscoroutinefunction(manager.add)
    snapshot = await manager.add(
        "immutable",
        scope="user",
        url="https://example.test",
        headers={"Authorization": "${TOKEN}"},
    )
    entry = snapshot.entries[0]

    with pytest.raises(TypeError):
        entry.config["enabled"] = False
    with pytest.raises(TypeError):
        entry.config["headers"]["Authorization"] = "changed"
    assert manager.snapshot is snapshot
