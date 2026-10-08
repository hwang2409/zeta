"""Session wiring for ``zeta --computer`` and ``/computer``."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from zeta.cli.main import build_parser, main
from zeta.computer import session as computer_session
from zeta.computer.backend import BACKENDS, BackendKind
from zeta.computer.session import TOOLS_ARGUMENT, ComputerSession, restart_args
from zeta.computer.tools import QUALIFIED_TOOL_NAMES
from zeta.config.settings import load_settings
from zeta.core.approval import ApprovalDecision
from zeta.core.session import SessionError
from zeta.mcp.connection import mcp_server_allowed
from zeta.mcp.management import MCPManagementService
from zeta.protocol.types import ToolCall
from zeta.runtime.headless import run_headless
from zeta.tools._action_metadata import ApprovalBinding, ResolvedCapability
from zeta.tui.app import create_app
from zeta.tui.slash_handlers import SlashHandlerMixin


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("ZETA_HOME", str(home))
    workspace = tmp_path / "work"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    return home


@pytest.fixture
def removed_sessions(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace host-side desktop removal; no test here reaches a VM."""

    removed: list[str] = []
    kind = BACKENDS["local"]
    monkeypatch.setitem(
        BACKENDS, "local", BackendKind(create=kind.create, remove_session=removed.append)
    )
    return removed


def _spy_mcp_server(home: Path, marker: Path) -> None:
    (home / "mcp.json").write_text(
        json.dumps(
            {
                "servers": {
                    "other": {
                        "transport": "stdio",
                        "command": sys.executable,
                        "args": ["-c", f"open({str(marker)!r}, 'w').close()"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def _args(*extra: str) -> object:
    return build_parser().parse_args(["--provider", "codex", "--computer", *extra])


@pytest.mark.asyncio
async def test_computer_session_advertises_only_computer_tools(
    home: Path, tmp_path: Path, removed_sessions: list[str]
) -> None:
    marker = tmp_path / "other-server-started"
    _spy_mcp_server(home, marker)
    (home / "hooks.toml").write_text('[[hooks]]\nevent = "session_start"\ncommand = "true"\n')
    app = create_app(_args("-p", "go"))
    try:
        loop = app.loop
        registry = loop.tool_registry
        await loop.activate()
        await loop.ensure_mcp_servers()
        names = [schema["name"] for schema in loop.tool_schemas]
        assert names == list(QUALIFIED_TOOL_NAMES)
        assert sorted(registry.registered_names) == sorted(QUALIFIED_TOOL_NAMES)
        assert [schema["name"] for schema in registry.schemas] == list(QUALIFIED_TOOL_NAMES)
        for host_tool in ("bash", "read", "write", "edit", "agent", "mcp_discover"):
            assert host_tool not in registry.definitions_by_name
            result = await registry.execute(ToolCall(f"call-{host_tool}", host_tool, {}))
            assert result["isError"] is True
        assert loop.require_allowed_tools() is None
        assert loop.hooks is None
        assert not marker.exists()
        assert not mcp_server_allowed(registry, "other")
        assert mcp_server_allowed(registry, "computer")
        policy = app.approval_policy
        assert policy.decide(
            registry.resolve_call("computer__click", {"x": 1, "y": 1})
        ) is ApprovalDecision.ALLOW
        assert policy.decide(
            ResolvedCapability(
                "bash", None, True, "command", "ls", ApprovalBinding.CWD, None
            )
        ) is not ApprovalDecision.ALLOW
        metadata = loop.session_metadata
        assert metadata.tool_allow == QUALIFIED_TOOL_NAMES
    finally:
        await app.close()
    assert removed_sessions == [app.loop.store.session_id]


@pytest.mark.asyncio
async def test_computer_mount_is_pinned_against_management_sync(
    home: Path, tmp_path: Path, removed_sessions: list[str]
) -> None:
    marker = tmp_path / "other-server-started"
    app = create_app(_args("-p", "go"))
    try:
        await app.loop.activate()
        await app.loop.ensure_mcp_servers()
        _spy_mcp_server(home, marker)
        mount = app.loop._mcp_mount
        assert mount.pinned is True
        await MCPManagementService(home=home, mount=mount).sync_runtime()
        assert list(mount.configs) == ["computer"]
        assert not marker.exists()
        with pytest.raises(ValueError, match="already mounted"):
            app.loop.select_mcp_config(mount)
    finally:
        await app.close()


@pytest.mark.asyncio
async def test_session_end_removes_desktops_and_finishes_recording(
    home: Path, removed_sessions: list[str]
) -> None:
    app = create_app(_args("-p", "go"))
    computer = app.computer_session
    assert isinstance(computer, ComputerSession)
    recording = app.loop.store.session_dir / "computer"
    assert computer.recording == recording
    assert computer.spectator is not None and computer.spectator.url.startswith("http://127.0.0.1:")
    (recording / "metadata.json").write_text(json.dumps({"active": True}))
    notices = "\n".join(computer.notices)
    assert "spectator http://127.0.0.1:" in notices
    assert f"zeta computer watch --live {app.loop.store.session_id}" in notices
    await app.close()
    assert removed_sessions == [app.loop.store.session_id]
    assert computer.spectator is None
    assert json.loads((recording / "metadata.json").read_text())["active"] is False


def test_headless_abort_still_removes_desktops(
    home: Path, removed_sessions: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    async def aborted_turn(*args: object, **kwargs: object) -> int:
        raise asyncio.CancelledError

    monkeypatch.setattr("zeta.runtime.headless.drive_turn", aborted_turn)
    args = _args("-p", "write a note")
    with pytest.raises(asyncio.CancelledError):
        run_headless(args, args.prompt)
    assert len(removed_sessions) == 1


def test_headless_fails_when_the_computer_server_cannot_start(
    home: Path,
    removed_sessions: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original = computer_session.server_config

    def broken(**kwargs: object):
        config = original(**kwargs)
        return type(config)(name=config.name, transport="stdio", command="definitely-not-a-server")

    monkeypatch.setattr(computer_session, "server_config", broken)
    args = _args("-p", "go")
    assert run_headless(args, args.prompt) == 1
    error = capsys.readouterr().err
    assert "required tools are unavailable" in error
    assert "computer__screenshot" in error
    assert len(removed_sessions) == 1


def test_computer_flag_conflicts_are_rejected(home: Path) -> None:
    with pytest.raises(SessionError, match="--tools"):
        create_app(_args("--tools", "read", "-p", "go"))
    with pytest.raises(SystemExit):
        main(["--computer-backend", "local", "-p", "go"])
    with pytest.raises(SystemExit):
        main(["--computer", "session", "list"])


@pytest.mark.asyncio
async def test_disallowed_tools_still_narrow_a_computer_session(
    home: Path, removed_sessions: list[str]
) -> None:
    app = create_app(_args("--disallowed-tools", "computer__drag", "-p", "go"))
    try:
        await app.loop.activate()
        await app.loop.ensure_mcp_servers()
        names = {schema["name"] for schema in app.loop.tool_schemas}
        assert names == set(QUALIFIED_TOOL_NAMES) - {"computer__drag"}
    finally:
        await app.close()


@pytest.mark.asyncio
async def test_computer_restart_narrows_an_existing_session(
    home: Path, removed_sessions: list[str]
) -> None:
    args = build_parser().parse_args(["--provider", "codex", "-p", "go"])
    plain = create_app(args)
    session_id = plain.loop.store.session_id
    assert "bash" in plain.loop.tool_registry.registered_names
    assert plain.computer_session is None
    await plain.close()

    restart_args(args, session_id=session_id, provider="codex", model=plain.model)
    assert args.tools == TOOLS_ARGUMENT and args.require_tools is True
    reopened = create_app(args)
    try:
        assert reopened.loop.store.session_id == session_id
        assert reopened.computer_session is not None
        assert "bash" not in reopened.loop.tool_registry.registered_names
        assert reopened.loop.session_metadata.tool_allow == QUALIFIED_TOOL_NAMES
    finally:
        await reopened.close()


class _SlashStub(SlashHandlerMixin):
    def __init__(self, *, ephemeral: bool = False, active: bool = False) -> None:
        self.ephemeral_root = Path("/tmp") if ephemeral else None
        self.active = active
        self.pending_approvals = ()
        self.exits = 0

    def request_exit(self) -> None:
        self.exits += 1


def test_slash_computer_requests_a_computer_restart() -> None:
    stub = _SlashStub()
    assert "reopening" in stub.slash_computer("")
    assert stub.computer_requested is True and stub.exits == 1
    assert "ephemeral" in _SlashStub(ephemeral=True).slash_computer("")
    busy = _SlashStub(active=True)
    assert "unchanged" in busy.slash_computer("")
    assert busy.computer_requested is False
    assert "does not accept" in _SlashStub().slash_computer("now")


def test_computer_settings_table_is_a_known_global_key(home: Path) -> None:
    (home / "settings.toml").write_text("[computer]\ncpus = 2\n")
    loaded = load_settings(home=home)
    assert not any("computer" in notice for notice in loaded.notices)
