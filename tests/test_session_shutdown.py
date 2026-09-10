"""Every frontend must release storage without assistance from cyclic GC."""

from __future__ import annotations

import asyncio
import gc
from contextlib import nullcontext
from pathlib import Path

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.input import DummyInput
from prompt_toolkit.output import DummyOutput

from zeta.cli import build_parser, main
from zeta.core.checkpoints.workspace import WorkspaceSnapshotStore
from zeta.core.session import SessionInUseError, SessionManager
from zeta.core.store import ConversationStore
from zeta.headless import run_headless
from zeta.loop import AgentLoop
from zeta.server.runtime import ServerRuntime
from zeta.tui.app import TUIApp, create_app


@pytest.fixture
def no_gc():
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


@pytest.mark.usefixtures("no_gc")
@pytest.mark.parametrize("entry", ["server", "tui", "tui-new", "headless"])
@pytest.mark.parametrize("failure", [None, "activate", "startup", "close"])
def test_entrypoint_shutdown_releases_every_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str, failure: str | None
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    manager = SessionManager(home)
    # Keep all acquired stores/loops/apps alive, including failed startup frames.
    stores = []
    apps = []
    snapshots = []
    original_init = ConversationStore.__init__
    original_activate = AgentLoop.activate
    original_close = AgentLoop.close
    original_create = create_app

    def record_store(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        stores.append(self)

    def build_app(args):
        app = original_create(args)
        apps.append(app)
        snapshots.append(app._snapshots())
        return app

    async def activate(loop):
        await original_activate(loop)
        if failure == "activate" and (entry != "tui-new" or len(apps) == 2):
            raise RuntimeError("injected activate")

    def startup(*args):
        if failure == "startup" and (entry != "tui-new" or len(apps) == 2):
            raise RuntimeError("injected startup")

    async def close(loop):
        if entry == "tui-new" and len(apps) == 1:
            await original_close(loop)
            return
        raise RuntimeError("injected close")

    async def prompt(app, session):
        if entry == "tui-new" and len(apps) == 1:
            app._slash_commands.dispatch(app, "/new")
        elif entry == "tui-new":
            # The replacement is running. Delete its predecessor immediately.
            old = apps[0].loop.store
            manager.delete(old.session_id)
            assert not old.session_dir.exists()

    async def turn(*args, **kwargs):
        startup()
        return 0

    monkeypatch.setattr(ConversationStore, "__init__", record_store)
    monkeypatch.setattr(AgentLoop, "activate", activate)
    if failure == "close":
        monkeypatch.setattr(AgentLoop, "close", close)
    monkeypatch.setattr("zeta.cli.create_app", build_app)
    monkeypatch.setattr("zeta.tui.app.create_app", build_app)
    monkeypatch.setattr(TUIApp, "_make_session", lambda self: PromptSession(
        input=DummyInput(), output=DummyOutput(),
    ))
    if entry == "tui-new":
        monkeypatch.setattr(TUIApp, "_read_prompt", prompt)
    monkeypatch.setattr(TUIApp, "_rebuild_transcript", startup)
    monkeypatch.setattr("zeta.headless.drive_turn", turn)
    args = build_parser().parse_args(["--provider", "fake"])

    async def server():
        runtime = ServerRuntime(home, cwd=tmp_path, provider="fake")
        monkeypatch.setattr(runtime, "_bind_background_event_sink", startup)
        try:
            await runtime.create_session()
        finally:
            await runtime.close()

    expected = pytest.raises(RuntimeError, match="injected") if failure else nullcontext()
    with expected:
        if entry == "server":
            asyncio.run(server())
        elif entry == "headless":
            assert run_headless(args, "hello") == 0
        elif entry == "tui-new":
            assert main(["--provider", "fake"]) == 0
        else:
            asyncio.run(build_app(args).run())

    assert stores
    assert all(not store._release_lease.alive for store in stores)
    for metadata in manager.list_sessions():
        manager.delete(metadata.session_id)
    assert list(manager.sessions_dir.iterdir()) == []
    for snapshot in snapshots:
        assert not snapshot._release_lease.alive
    for app in apps:
        assert app.loop._background_event_sink is None
        assert app.loop._mcp_notice_sink is None
        assert app.loop._mcp_prompt_refresh is None


@pytest.mark.usefixtures("no_gc")
@pytest.mark.parametrize("entry", ["server", "tui", "headless"])
@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("failure", ["loop", "composition", "frontend"])
def test_construction_failure_releases_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    entry: str, resume: bool, failure: str,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    manager = SessionManager(home)
    session_id = None
    if resume:
        opened = manager.create(provider="fake", model="offline", cwd=tmp_path)
        session_id = opened.metadata.session_id
        opened.store.close()
    retained = []

    def fail(*args, **kwargs):
        retained.extend(args)
        raise RuntimeError("injected construction")

    if failure == "loop":
        monkeypatch.setattr("zeta.loop.ContextAssembler", fail)
    elif failure == "composition":
        monkeypatch.setattr("zeta.runtime.composition.apply_external_tools", fail)
    elif entry == "server":
        monkeypatch.setattr(ServerRuntime, "_bind_background_event_sink", fail)
    else:
        monkeypatch.setattr("zeta.tui.bootstrap._validate_keybindings", fail)
    argv = ["--provider", "fake"] + (["--resume", session_id] if resume else [])

    async def server():
        runtime = ServerRuntime(home, cwd=tmp_path, provider="fake")
        if session_id:
            await runtime.resume_session(session_id)
        else:
            await runtime.create_session()

    with pytest.raises(RuntimeError, match="injected construction"):
        if entry == "server":
            asyncio.run(server())
        elif entry == "headless":
            run_headless(build_parser().parse_args(argv), "hello")
        else:
            create_app(build_parser().parse_args(argv))
    sessions = manager.list_sessions()
    assert len(sessions) == 1
    manager.delete(sessions[0].session_id)


@pytest.mark.usefixtures("no_gc")
def test_snapshot_close_releases_its_independent_lease(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path / "home")
    opened = manager.create(provider="fake", model="offline", cwd=tmp_path)
    snapshots = WorkspaceSnapshotStore(opened.store.session_dir, opened.store.session_id)
    opened.store.close()
    with pytest.raises(SessionInUseError):
        manager.delete(opened.metadata.session_id)
    snapshots.close()
    manager.delete(opened.metadata.session_id)


@pytest.mark.usefixtures("no_gc")
@pytest.mark.parametrize("component", ["submissions", "draft", "registry", "snapshots"])
async def test_tui_cleanup_continues_after_close_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, component: str,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    app = create_app(build_parser().parse_args(["--provider", "fake"]))
    snapshots = app._snapshots()
    registry = app.loop.tool_registry.background_tasks
    original_registry_close = registry.close
    original_snapshot_close = snapshots.close

    async def fail_async():
        if component == "registry":
            await original_registry_close()
        raise RuntimeError("injected cleanup")

    def fail_sync():
        if component == "snapshots":
            original_snapshot_close()
        raise RuntimeError("injected cleanup")

    if component == "submissions":
        monkeypatch.setattr(app._submissions, "close", fail_async)
    elif component == "draft":
        monkeypatch.setattr(app._draft, "flush", fail_sync)
    elif component == "registry":
        monkeypatch.setattr(registry, "close", fail_async)
    else:
        monkeypatch.setattr(snapshots, "close", fail_sync)
    with pytest.raises(RuntimeError, match="injected cleanup"):
        await app.close()
    SessionManager(home).delete(app.loop.store.session_id)


@pytest.mark.usefixtures("no_gc")
@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("startup_failure", [False, True])
async def test_shutdown_releases_child_stores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, background: bool,
    startup_failure: bool,
) -> None:
    from zeta.core.abort import AbortSignal
    from zeta.core.fake import FakeBackend, ScriptedTurn
    from zeta.types import TextContent, ToolCall

    home = tmp_path / "home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    runtime = ServerRuntime(home, cwd=tmp_path, provider="fake")
    await runtime.create_session()
    loop = runtime.loop
    assert loop is not None
    loop.backend = FakeBackend([ScriptedTurn(content=[TextContent("child done")])])
    call = ToolCall("child", "agent", {
        "prompt": "hello", "description": "child", "background": background,
    })
    if startup_failure:
        def fail(*args, **kwargs):
            raise RuntimeError("injected child setup")
        monkeypatch.setattr("zeta.loop.ContextAssembler", fail)
    result = await loop._run_agent_tool(call, call.arguments, AbortSignal(), None)
    assert result["isError"] is startup_failure
    children = list(loop._agent_child_stores.values())
    assert children
    session_id = runtime.session_id
    await runtime.close()
    assert all(not child._release_lease.alive for child in children)
    SessionManager(home).delete(session_id)


@pytest.mark.usefixtures("no_gc")
def test_recovery_and_send_release_borrowed_child_stores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from zeta.agent_background import recover_agent_children
    from zeta.core.fake import FakeBackend
    from zeta.tools.agent_send import send_to_run
    from zeta.types import ToolCall

    manager = SessionManager(tmp_path / "home")
    opened = manager.create(provider="fake", model="offline", cwd=tmp_path)
    loop = AgentLoop(FakeBackend([]), opened.store)
    call = ToolCall("child", "agent", {"prompt": "hello", "description": "child"})
    with ConversationStore(opened.store.session_dir / "agents", session_id="1") as child:
        with ConversationStore(child.session_dir / "agents", session_id="1") as nested:
            nested.mark_agent_parent("nested")
            child.register_agent_child(
                ToolCall("nested", "agent", {}),
                child_session_path=str(nested.session_dir), description="nested",
            )
        opened.store.register_agent_child(
            call, child_session_path=str(child.session_dir), description="child",
            agent_type="run", child_instance_id="child",
        )
    retained = []
    original_init = ConversationStore.__init__

    def record(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        retained.append(self)

    monkeypatch.setattr(ConversationStore, "__init__", record)
    assert send_to_run(opened.store, "child", "follow-up") is None
    recover_agent_children(loop)
    assert len(retained) == 3
    assert all(not store._release_lease.alive for store in retained)
    asyncio.run(loop.close())
    opened.store.close()
    manager.delete(opened.metadata.session_id)


@pytest.mark.usefixtures("no_gc")
@pytest.mark.parametrize("run", ["close", "full-screen"])
async def test_closed_tui_drops_callbacks_without_gc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run: str,
) -> None:
    import weakref

    monkeypatch.setenv("ZETA_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    app = create_app(build_parser().parse_args(["--provider", "fake"]))
    app._snapshots()
    loop = app.loop
    reference = weakref.ref(app)
    if run == "full-screen":
        from prompt_toolkit.application import create_app_session
        with create_app_session(input=DummyInput(), output=DummyOutput()):
            app._make_session()
    await app.close()
    del app
    assert reference() is None
    SessionManager(tmp_path / "home").delete(loop.store.session_id)
