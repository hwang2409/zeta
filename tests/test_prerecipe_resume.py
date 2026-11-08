"""A pre-recipe session must stay readable when resume cannot refresh its memory."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from tests.support.serve_close import close_serve
from tests.support.server_client import request
from tests.support.sockets import short_socket_path
from zeta.cli.main import build_parser, main
from zeta.core.session import SessionError, SessionManager, SessionMetadata
from zeta.core.store.prompt_composition import PromptComposition
from zeta.project_registry import ProjectRegistry
from zeta.server.server import ZetaServer
from zeta.skills import SkillCatalog
from zeta.skills.agent_catalog import AgentCatalog
from zeta.tui.app import create_app

_TIMEOUT = 10
_MEMORY = "MEMORY-SPAN"
_PROMPT = f"before\n{_MEMORY}\nafter"


_TRIGGERS = ("project_missing", "registry_unavailable", "malformed_memory")


def _pre_recipe_session(
    home: Path, repository: Path, trigger: str = "project_missing"
) -> tuple[str, Path]:
    """Build the shape of the six at-risk real sessions: recipe null, legacy span."""

    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    project = ProjectRegistry(home / "projects").create_project(
        "linked", "git", repository
    )
    opened = SessionManager(home).create(
        provider="codex",
        model="gpt-5.6-luna",
        cwd=repository,
        project_id=project.project_id,
        auto_project=False,
        system_prompt=_PROMPT,
        project_memory_offset=len("before\n"),
        project_memory_length=len(_MEMORY),
        project_memory_digest=hashlib.sha256(_MEMORY.encode()).hexdigest(),
        prompt_recipe=None,
    )
    session_id = opened.metadata.session_id
    opened.store.close()
    meta = home / "sessions" / session_id / "meta.json"
    stored = json.loads(meta.read_text())
    assert stored["prompt_recipe"] is None
    assert stored["prompt_components"] == {}
    assert stored["project_memory_length"] == len(_MEMORY)
    # Each trigger makes resume drop the session's memory span.
    projects = home / "projects"
    if trigger == "project_missing":
        shutil.rmtree(projects / project.project_id)
    elif trigger == "registry_unavailable":
        shutil.rmtree(projects)
        projects.write_text("not a directory")
    else:
        (projects / project.project_id / "memory-current.json").write_text(
            '{"current": 1}'
        )
    return session_id, meta


async def _serve_resume(home: Path, tmp_path: Path, session_id: str) -> dict:
    server = ZetaServer(
        home=home, socket_path=short_socket_path(tmp_path), provider="codex"
    )
    await asyncio.wait_for(server.start(), _TIMEOUT)
    reader, writer = await asyncio.wait_for(
        asyncio.open_unix_connection(str(server.socket_path)), _TIMEOUT
    )
    try:
        await request(reader, writer, 1, "hello", {"protocol_version": "1.0"})
        return (await request(reader, writer, 2, "resume", {"session_id": session_id}))[
            -1
        ]
    finally:
        await close_serve(server, writer)


@pytest.mark.parametrize("trigger", _TRIGGERS)
@pytest.mark.parametrize("surface", ["tui", "serve"])
def test_pre_recipe_resume_without_project_memory_keeps_session_readable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    surface: str,
    trigger: str,
) -> None:
    home = tmp_path / "home"
    repository = tmp_path / "repository"
    session_id, meta = _pre_recipe_session(home, repository, trigger)
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(repository)

    if surface == "tui":
        app = create_app(
            build_parser().parse_args(["--resume", session_id, "--provider", "codex"])
        )
        try:
            prompt = app.loop.context_assembler.system_prompt.content[0].text
        finally:
            asyncio.run(app.close())
    else:
        frame = asyncio.run(_serve_resume(home, tmp_path, session_id))
        assert "error" not in frame, frame
        assert frame["result"]["session"]["session_id"] == session_id
        prompt = json.loads(meta.read_text())["system_prompt"]

    assert _MEMORY not in prompt
    stored = json.loads(meta.read_text())
    assert stored["prompt_recipe"] is None
    assert stored["prompt_components"] == {}
    # The legacy fields keep the dropped span: same offset, empty text.
    assert stored["system_prompt"] == "before\n\nafter"
    assert stored["project_memory_offset"] == len("before\n")
    assert stored["project_memory_length"] == 0
    assert stored["project_memory_digest"] == hashlib.sha256(b"").hexdigest()
    capsys.readouterr()
    assert main(["session", "export", session_id]) == 0
    exported = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert exported[0]["metadata"]["session_id"] == session_id


@pytest.mark.parametrize("writer", ["resume", "mutate"])
def test_metadata_writer_refuses_unreadable_shape_without_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    writer: str,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    session_id, meta = _pre_recipe_session(home, tmp_path / "repository")
    before = meta.read_bytes()
    manager = SessionManager(home)
    opened = manager.open(session_id)
    span = {
        "offset": len("before\n"),
        "length": len(_MEMORY),
        "digest": hashlib.sha256(_MEMORY.encode()).hexdigest(),
    }

    def compose(_current: object) -> PromptComposition:
        # The reader rejects prompt components without a recipe.
        return PromptComposition(
            system_prompt=_PROMPT + "\nchanged",
            context_files=(),
            skill_catalog=SkillCatalog.empty(),
            agent_catalog=AgentCatalog.empty(),
            prompt_recipe=None,
            prompt_components={"project_memory": span},
            project_memory_offset=span["offset"],
            project_memory_length=span["length"],
            project_memory_digest=span["digest"],
        )

    def bad_components(current: SessionMetadata) -> SessionMetadata:
        return replace(current, prompt_components={"project_memory": span})

    try:
        with pytest.raises(SessionError, match="require a recipe"):
            if writer == "resume":
                manager.resume_prompt_composition(
                    opened.metadata, opened.store, compose
                )
            else:
                manager._mutate(
                    session_id,
                    bad_components,
                    execution_lease=opened.metadata._execution_lease,
                )
    finally:
        opened.store.close()
    assert meta.read_bytes() == before
    capsys.readouterr()
    assert main(["session", "export", session_id]) == 0
    exported = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert exported[0]["metadata"]["prompt_components"] == {}
