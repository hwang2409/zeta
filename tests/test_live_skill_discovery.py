from __future__ import annotations

import asyncio
import fcntl
import inspect
import json
import os
import threading
import time
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from tests.test_server import (
    _close,
    _ready_extensions,
    _request,
    _seed_slash_fixtures,
)
from zeta.core.loop import AgentLoop
from zeta.core.session import SessionManager
from zeta.core.store import ConversationStore
from zeta.protocol.types import MessageOrigin, MessageRole, TextContent
from zeta.server.server import ZetaServer
from zeta.skills import SkillCatalog, SkillMeta, SkillRefresh, discover_session_skills
from zeta.tui.app import TUIApp


def _live_skill_kwargs(home: Path, project: Path) -> dict[str, Path]:
    """Let this behavioral test run against main before the interface existed."""

    if "skill_home" not in inspect.signature(AgentLoop).parameters:
        return {}
    return {"skill_home": home, "skill_project_dir": project}


def _session_loop(
    tmp_path: Path, backend: FakeBackend | None = None
) -> tuple[AgentLoop, SessionManager, Path, Path]:
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    catalog = discover_session_skills(home=home, project_dir=project)
    manager = SessionManager(home)
    opened = manager.create(
        provider="codex",
        model="offline",
        cwd=project,
        system_prompt=f"identity\n\n{catalog.index()}",
        skill_catalog=catalog,
        auto_project=False,
    )
    loop = AgentLoop(
        backend or FakeBackend([]),
        opened.store,
        skill_catalog=catalog,
        system_prompt=opened.metadata.system_prompt,
        skill_home=home,
        skill_project_dir=project,
    )
    loop.manager = manager
    loop.session_metadata = opened.metadata
    return loop, manager, home, project


def _hold_metadata_lock(manager: SessionManager, session_id: str, seconds: float = 2.0):
    ready = threading.Event()
    release = threading.Event()

    def hold() -> None:
        path = manager.sessions_dir / session_id / ".meta.lock"
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            ready.set()
            release.wait(seconds)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    thread = threading.Thread(target=hold, daemon=True)
    thread.start()
    assert ready.wait(1)
    return release, thread


def _catalog(name: str, root: Path) -> SkillCatalog:
    return SkillCatalog((SkillMeta(name, f"{name} description", [], root / name),))


def _refresh(catalog: SkillCatalog) -> SkillRefresh:
    return SkillRefresh(catalog, (catalog.skills[0].name,), (), (), ())


def _install_live_skill(home: Path) -> None:
    skill = home / "skills" / "live.md"
    skill.parent.mkdir(parents=True, exist_ok=True)
    skill.write_text(
        "---\nname: live\ndescription: live description\n---\n\nlive body\n",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_new_skill_reaches_provider_prompt_on_actual_next_turn(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    catalog = discover_session_skills(home=home, project_dir=project)
    backend = FakeBackend([ScriptedTurn([TextContent("done")])])
    store = ConversationStore(tmp_path / "sessions", cwd=project)
    loop = AgentLoop(
        backend,
        store,
        skill_catalog=catalog,
        system_prompt=f"identity\n\n{catalog.index()}",
        **_live_skill_kwargs(home, project),
    )
    _install_live_skill(home)

    _ = [event async for event in loop.run_turn("use live", origin=MessageOrigin.USER)]

    system_text = "\n".join(
        block.text
        for message in backend.calls[0][0]
        if message.role is MessageRole.SYSTEM
        for block in message.content
        if isinstance(block, TextContent)
    )
    assert "live description" in system_text
    loop.store.close()


@pytest.mark.asyncio
async def test_tui_submit_discovers_new_skill_slash_before_parse(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    catalog = discover_session_skills(home=home, project_dir=project)
    backend = FakeBackend([ScriptedTurn([TextContent("done")])])
    loop = AgentLoop(
        backend,
        ConversationStore(tmp_path / "sessions", cwd=project),
        skill_catalog=catalog,
        **_live_skill_kwargs(home, project),
    )
    app = TUIApp(
        loop,
        provider="codex",
        model="offline",
        zeta_home=home,
        project_dir=project,
        console=Console(file=StringIO(), force_terminal=False),
    )
    _install_live_skill(home)

    await app._handle_prompt_value("/live request")
    if app._active_task is not None:
        await app._active_task

    user_text = "\n".join(
        block.text
        for message in backend.calls[0][0]
        if message.role is MessageRole.USER
        for block in message.content
        if isinstance(block, TextContent)
    )
    assert "live body" in user_text
    assert "request" in user_text
    await app._submissions.close()
    await loop.close()


@pytest.mark.asyncio
async def test_serve_discovers_new_skill_for_slash_list_then_run(tmp_path: Path) -> None:
    project = _seed_slash_fixtures(tmp_path)
    home = tmp_path / "home"
    server = ZetaServer(home=home, cwd=project, port=0, provider="codex")
    reader, writer, sid = await _ready_extensions(server)
    try:
        _install_live_skill(home)
        listed = (
            await _request(
                reader, writer, "live-list", "slash_list", {"session_id": sid}
            )
        )[-1]["result"]
        assert any(
            command["name"] == "live" and command["kind"] == "skill"
            for command in listed["commands"]
        )

        result = (
            await _request(
                reader,
                writer,
                "live-run",
                "slash_run",
                {"session_id": sid, "text": "/live request"},
            )
        )[-1]["result"]
        assert result["kind"] == "model_input"
        assert "live body" in result["text"]
        assert "request" in result["text"]
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_locked_metadata_does_not_block_skill_commit_or_next_turn(
    tmp_path: Path,
) -> None:
    backend = FakeBackend([ScriptedTurn([TextContent("done")])])
    loop, manager, home, _ = _session_loop(tmp_path, backend)
    release, holder = _hold_metadata_lock(manager, loop.session_metadata.session_id)
    _install_live_skill(home)
    started = time.monotonic()
    try:
        await loop.refresh_skills()
        _ = [
            event
            async for event in loop.run_turn("next", origin=MessageOrigin.USER)
        ]
        assert time.monotonic() - started < 0.5
        assert loop.tool_registry.skill_catalog.find("live").description == "live description"
    finally:
        release.set()
        holder.join(1)
        await loop.close()
        loop.store.close()


@pytest.mark.asyncio
async def test_skill_catalog_persistence_runs_off_loop_after_lock_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop, manager, home, _ = _session_loop(tmp_path)
    release, holder = _hold_metadata_lock(manager, loop.session_metadata.session_id)
    event_thread = threading.get_ident()
    called: list[int] = []
    original = manager.record_skill_catalog

    def record(*args: object, **kwargs: object) -> None:
        called.append(threading.get_ident())
        original(*args, **kwargs)

    monkeypatch.setattr(manager, "record_skill_catalog", record)
    _install_live_skill(home)
    await loop.refresh_skills()
    for _ in range(10):
        if called:
            break
        await asyncio.sleep(0)
    assert called and called[0] != event_thread

    release.set()
    holder.join(1)
    await loop._flush_skill_catalog_persistence()
    saved = manager.read_metadata(loop.session_metadata.session_id)
    assert saved.skill_catalog[0]["name"] == "live"
    await loop.close()
    loop.store.close()


@pytest.mark.asyncio
async def test_two_quick_skill_commits_persist_only_newest_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop, manager, _, project = _session_loop(tmp_path)
    writes: list[list[str]] = []
    original = manager.record_skill_catalog

    def record(*args: object, **kwargs: object) -> None:
        catalog = kwargs["catalog"]
        assert isinstance(catalog, SkillCatalog)
        writes.append([skill.name for skill in catalog.skills])
        original(*args, **kwargs)

    monkeypatch.setattr(manager, "record_skill_catalog", record)
    loop._commit_skill_refresh(_refresh(_catalog("first", project)), generation=1)
    loop._commit_skill_refresh(_refresh(_catalog("second", project)), generation=2)

    await loop._flush_skill_catalog_persistence()
    assert writes == [["second"]]
    saved = manager.read_metadata(loop.session_metadata.session_id)
    assert saved.skill_catalog[0]["name"] == "second"
    await loop.close()
    loop.store.close()


@pytest.mark.asyncio
async def test_close_does_not_wait_indefinitely_for_skill_metadata_lock(
    tmp_path: Path,
) -> None:
    loop, manager, home, _ = _session_loop(tmp_path)
    release, holder = _hold_metadata_lock(manager, loop.session_metadata.session_id)
    _install_live_skill(home)
    await loop.refresh_skills()

    started = time.monotonic()
    await loop.close()
    elapsed = time.monotonic() - started

    release.set()
    holder.join(1)
    loop.store.close()
    assert elapsed < 0.8


@pytest.mark.asyncio
async def test_skill_catalog_persistence_and_touch_preserve_each_other(
    tmp_path: Path,
) -> None:
    loop, manager, _, project = _session_loop(tmp_path)
    original_updated_at = loop.session_metadata.updated_at
    loop._commit_skill_refresh(_refresh(_catalog("latest", project)), generation=1)
    manager.touch(loop.session_metadata)
    assert loop.session_metadata.skill_catalog[0]["name"] == "latest"
    await loop._flush_skill_catalog_persistence()

    saved = manager.read_metadata(loop.session_metadata.session_id)
    assert saved.skill_catalog[0]["name"] == "latest"
    assert saved.updated_at != original_updated_at

    manager.touch(loop.session_metadata)
    touched_at = loop.session_metadata.updated_at
    loop._commit_skill_refresh(_refresh(_catalog("newest", project)), generation=2)
    await loop._flush_skill_catalog_persistence()
    saved = manager.read_metadata(loop.session_metadata.session_id)
    assert saved.skill_catalog[0]["name"] == "newest"
    assert saved.updated_at == touched_at
    assert json.loads(
        (manager.sessions_dir / saved.session_id / "meta.json").read_text()
    )["skill_catalog"][0]["name"] == "newest"
    await loop.close()
    loop.store.close()
