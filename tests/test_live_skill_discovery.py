from __future__ import annotations

import fcntl
import hashlib
import inspect
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
from zeta.runtime.prompt_resume import resume_prompt
from zeta.server.server import ZetaServer
from zeta.skills import SkillCatalog, discover_session_skills
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
async def test_touch_adopts_metadata_without_reverting_live_prompt(
    tmp_path: Path,
) -> None:
    backend = FakeBackend([ScriptedTurn([TextContent("done")])])
    loop, manager, home, _ = _session_loop(tmp_path, backend)
    original_prompt = loop.session_metadata.system_prompt
    original_catalog = loop.session_metadata.skill_catalog
    _install_live_skill(home)
    await loop.refresh_skills()

    other = SessionManager(home).open(loop.session_metadata.session_id)
    try:
        SessionManager(home).record_override(
            other.metadata, provider=None, model="new-model"
        )
    finally:
        other.store.close()
    manager.touch(loop.session_metadata)

    assert loop.session_metadata.model == "new-model"
    assert loop.session_metadata.system_prompt == original_prompt
    _ = [event async for event in loop.run_turn("next", origin=MessageOrigin.USER)]
    system_text = "\n".join(
        block.text
        for message in backend.calls[0][0]
        if message.role is MessageRole.SYSTEM
        for block in message.content
        if isinstance(block, TextContent)
    )
    assert "live description" in system_text
    saved = manager.read_metadata(loop.session_metadata.session_id)
    assert saved.system_prompt == original_prompt
    assert saved.skill_catalog == original_catalog
    await loop.close()
    loop.store.close()


@pytest.mark.asyncio
async def test_resume_replaces_live_skill_index_and_refreshes_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    catalog = SkillCatalog.empty()
    memory = "old memory"
    prompt = f"identity\n\n{catalog.index()}\n\n{memory}\n"
    memory_offset = prompt.index(memory)
    memory_digest = hashlib.sha256(memory.encode()).hexdigest()
    manager = SessionManager(home)
    opened = manager.create(
        provider="codex",
        model="offline",
        cwd=project,
        system_prompt=prompt,
        skill_catalog=catalog,
        project_memory_offset=memory_offset,
        project_memory_length=len(memory),
        project_memory_digest=memory_digest,
        prompt_recipe="custom",
        prompt_components={
            "project_memory": {
                "offset": memory_offset,
                "length": len(memory),
                "digest": memory_digest,
            }
        },
        auto_project=False,
    )
    loop = AgentLoop(
        FakeBackend([]),
        opened.store,
        skill_catalog=catalog,
        system_prompt=prompt,
        skill_home=home,
        skill_project_dir=project,
    )
    loop.manager = manager
    loop.session_metadata = opened.metadata
    _install_live_skill(home)
    await loop.refresh_skills()
    assert "live description" in loop.context_assembler.system_prompt.content[0].text
    assert manager.read_metadata(opened.metadata.session_id).system_prompt == prompt
    await loop.close()
    opened.store.close()

    refreshed_memory = "current project memory"
    monkeypatch.setattr(
        "zeta.runtime.prompt_resume.refresh_project_memory",
        lambda value, **kwargs: value.replace(memory, refreshed_memory),
    )
    resumed = manager.open(opened.metadata.session_id)
    try:
        result = resume_prompt(
            resumed.metadata,
            manager=manager,
            store=resumed.store,
            home=home,
            repo_root=project,
            inbox_enabled=False,
            context_loader=lambda **kwargs: (_ for _ in ()).throw(
                AssertionError(f"unexpected default rebuild: {kwargs}")
            ),
        )
        component = result.context.prompt_components["project_memory"]
        offset = int(component["offset"])
        length = int(component["length"])
        assert "live description" in result.context.system_prompt
        assert result.context.system_prompt[offset : offset + length] == refreshed_memory
        assert component["digest"] == hashlib.sha256(refreshed_memory.encode()).hexdigest()
        saved = manager.read_metadata(opened.metadata.session_id)
        assert saved.system_prompt == result.context.system_prompt
        assert saved.project_memory_offset == offset
        assert saved.project_memory_length == length
        assert saved.project_memory_digest == component["digest"]
        assert saved.skill_catalog[0]["name"] == "live"
    finally:
        resumed.store.close()
