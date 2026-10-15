from __future__ import annotations

import inspect
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
from zeta.core.store import ConversationStore
from zeta.protocol.types import MessageOrigin, MessageRole, TextContent
from zeta.server.server import ZetaServer
from zeta.skills import discover_session_skills
from zeta.tui.app import TUIApp


def _live_skill_kwargs(home: Path, project: Path) -> dict[str, Path]:
    """Let this behavioral test run against main before the interface existed."""

    if "skill_home" not in inspect.signature(AgentLoop).parameters:
        return {}
    return {"skill_home": home, "skill_project_dir": project}


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
