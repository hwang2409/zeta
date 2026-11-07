from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import shutil
import stat
import subprocess
import threading
from collections.abc import Callable
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console
from rich.text import Text

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from zeta.cli.main import build_parser
from zeta.core.session import SessionManager
from zeta.project_errors import ProjectRegistryBusyError
from zeta.project_registry import ProjectRegistry
from zeta.protocol.types import MessageOrigin, TextContent
from zeta.runtime.project_registry_state import (
    ProjectRegistryUnavailable,
    load_project_registry,
)
from zeta.server.server import ZetaServer
from zeta.tui.app import create_app

_REGISTRY_WARNING = (
    "warning: project registry is temporarily unavailable; try again later"
)


def _invalid_json(_home: Path, _linked: Path, record: Path) -> None:
    record.write_text("{", encoding="utf-8")


def _wrong_schema(_home: Path, _linked: Path, record: Path) -> None:
    value = json.loads(record.read_text(encoding="utf-8"))
    value["schema_version"] = 999
    record.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _unreadable(_home: Path, _linked: Path, record: Path) -> None:
    record.chmod(0)


def _dangling_symlink(home: Path, _linked: Path, _record: Path) -> None:
    projects = home / "projects"
    shutil.rmtree(projects)
    projects.symlink_to(home / "missing-projects")


_CORRUPTIONS: tuple[tuple[str, Callable[[Path, Path, Path], None]], ...] = (
    ("invalid-json", _invalid_json),
    ("wrong-schema-version", _wrong_schema),
    ("unreadable-permissions", _unreadable),
    ("dangling-symlink", _dangling_symlink),
)
_CASE_IDS = [case for case, _corrupt in _CORRUPTIONS]


def test_registry_lock_wait_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "projects"
    registry = ProjectRegistry(root)
    registry.create_project("one", "repo")
    lock = root / ".lock"
    fd = os.open(lock, os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    monkeypatch.setattr("zeta.project_registry._PROJECT_REGISTRY_LOCK_TIMEOUT_SECONDS", 0.05)
    finished = threading.Event()
    outcome: list[BaseException] = []

    def list_projects() -> None:
        try:
            registry.list_projects()
        except BaseException as exc:  # noqa: BLE001
            outcome.append(exc)
        finally:
            finished.set()

    worker = threading.Thread(target=list_projects)
    worker.start()
    try:
        assert finished.wait(10), "registry lock wait exceeded the watchdog"
        assert len(outcome) == 1
        assert isinstance(outcome[0], ProjectRegistryBusyError)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        worker.join(2)


def _make_registry(home: Path) -> tuple[Path, Path]:
    linked = home / "linked-project"
    linked.mkdir(parents=True)
    project = ProjectRegistry(home / "projects").create_project(
        "linked", "git", linked
    )
    return linked, home / "projects" / project.project_id / "project.json"


def _corrupt_registry(
    home: Path, corrupt: Callable[[Path, Path, Path], None]
) -> Path:
    linked, record = _make_registry(home)
    corrupt(home, linked, record)
    return record


def _registry_snapshot(path: Path) -> tuple[object, ...]:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode):
        return (info.st_mode, os.readlink(path))
    rows: list[tuple[str, int, bytes | None]] = []
    for child in sorted((path, *path.rglob("*"))):
        child_info = child.lstat()
        try:
            content = child.read_bytes() if stat.S_ISREG(child_info.st_mode) else None
        except PermissionError:
            content = None
        rows.append((str(child.relative_to(path.parent)), child_info.st_mode, content))
    return tuple(rows)


def _render_startup(app: object) -> str:
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=500)
    app.console = console
    app._presenter.console = console
    app._present_startup_output()
    return Text.from_ansi(output.getvalue()).plain


async def _serve_hello(home: Path) -> dict[str, object]:
    server = ZetaServer(home=home, port=0, provider="codex")
    await server.start()
    assert server.port is not None
    reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
    try:
        writer.write(
            (
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "hello",
                        "params": {"protocol_version": "1.1", "features": ["projects"]},
                    }
                )
                + "\n"
            ).encode()
        )
        await writer.drain()
        response = json.loads(await reader.readline())
        return response["result"]
    finally:
        writer.close()
        await writer.wait_closed()
        await server.close()


@pytest.mark.parametrize(("_case", "corrupt"), _CORRUPTIONS, ids=_CASE_IDS)
def test_broken_registry_tui_start_and_resume_without_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _case: str,
    corrupt: Callable[[Path, Path, Path], None],
) -> None:
    home = tmp_path / "home"
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    linked, record = _make_registry(home)
    memory = "MEMORY-SECRET"
    prompt = f"before\n{memory}\nafter"
    existing = SessionManager(home).create(
        provider="codex",
        model="gpt-5.6-luna",
        cwd=repository,
        project_id=record.parent.name,
        auto_project=False,
        system_prompt=prompt,
        project_memory_offset=len("before\n"),
        project_memory_length=len(memory),
        project_memory_digest=hashlib.sha256(memory.encode()).hexdigest(),
        prompt_recipe="custom",
    )
    session_id = existing.metadata.session_id
    existing.store.close()
    corrupt(home, linked, record)
    projects = home / "projects"
    before = _registry_snapshot(projects)
    state = load_project_registry(projects)
    assert state.registry is not None
    assert isinstance(state.unavailable, ProjectRegistryUnavailable)
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(repository)

    apps = []
    try:
        for arguments in (
            ["--provider", "codex"],
            ["--resume", session_id, "--provider", "codex"],
        ):
            app = create_app(build_parser().parse_args(arguments))
            apps.append(app)
            _render_startup(app)
            assert (_REGISTRY_WARNING in app._startup_warnings) == (
                _case == "dangling-symlink"
            )
            assert "project" in app.loop.tool_registry.registered_names
            assert ("inbox" in app.loop.tool_registry.registered_names) == (
                "--resume" in arguments
            )
            assert app.loop.project_registry is state.registry or app.loop.project_registry is not None
            assert app.loop.project_context_health is not None
            assert memory not in app.loop.context_assembler.system_prompt.content[0].text
        assert _registry_snapshot(projects) == before
    finally:
        if record.exists():
            record.chmod(0o600)
        for app in apps:
            asyncio.run(app.close())


@pytest.mark.asyncio
@pytest.mark.parametrize(("_case", "corrupt"), _CORRUPTIONS, ids=_CASE_IDS)
async def test_broken_registry_serve_hello_without_repair(
    tmp_path: Path,
    _case: str,
    corrupt: Callable[[Path, Path, Path], None],
) -> None:
    home = tmp_path / "home"
    record = _corrupt_registry(home, corrupt)
    projects = home / "projects"
    before = _registry_snapshot(projects)
    try:
        hello = await _serve_hello(home)
        notices = hello["notices"]
        assert notices == [_REGISTRY_WARNING]
        assert "projects" in hello["capabilities"].get("features", ())
        assert _registry_snapshot(projects) == before
    finally:
        if record.exists():
            record.chmod(0o600)


@pytest.mark.asyncio
async def test_custom_resume_recovers_through_real_turns_without_losing_skill_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.skills import SkillCatalog

    home = tmp_path / "home"
    repository = tmp_path / "repository"
    repository.mkdir()
    await asyncio.to_thread(
        subprocess.run, ["git", "init", "-q", str(repository)], check=True
    )
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("healthy", "git", repository)
    memory_dir = registry.root / project.project_id / "memory"
    (memory_dir / "state.md").write_bytes(b"\xff")
    skill_index = SkillCatalog.empty().index()
    old_memory = "OLD-MEMORY"
    prompt = f"CUSTOM-BEFORE\n{skill_index}\n{old_memory}\nCUSTOM-AFTER"
    skill_offset = prompt.index(skill_index)
    memory_offset = prompt.index(old_memory)
    manager = SessionManager(home)
    opened = await asyncio.to_thread(
        manager.create,
        provider="codex",
        model="gpt-5.6-luna",
        cwd=repository,
        project_id=project.project_id,
        auto_project=False,
        system_prompt=prompt,
        project_memory_offset=memory_offset,
        project_memory_length=len(old_memory),
        project_memory_digest=hashlib.sha256(old_memory.encode()).hexdigest(),
        prompt_recipe="custom",
        prompt_components={
            "skill_index": {
                "offset": skill_offset,
                "length": len(skill_index),
                "digest": hashlib.sha256(skill_index.encode()).hexdigest(),
            },
            "project_memory": {
                "offset": memory_offset,
                "length": len(old_memory),
                "digest": hashlib.sha256(old_memory.encode()).hexdigest(),
            },
        },
    )
    session_id = opened.store.session_id
    opened.store.close()
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(repository)
    args = build_parser().parse_args(
        ["--resume", session_id, "--provider", "codex", "--no-auto-memory"]
    )
    app = await asyncio.to_thread(create_app, args)
    backend = FakeBackend(
        [ScriptedTurn(content=[TextContent("ok")]) for _ in range(15)]
    )
    app.loop.backend = backend
    projection_calls = 0
    original_projection = app.loop.project_registry._read_context_projection

    def count_projection(operation):
        nonlocal projection_calls
        projection_calls += 1
        return original_projection(operation)

    # test-contract: seam Count projection reads without changing recovery flow.
    monkeypatch.setattr(
        app.loop.project_registry, "_read_context_projection", count_projection
    )
    prepare_entered = threading.Event()
    prepare_release = threading.Event()
    prepare_done = [threading.Event(), threading.Event()]
    prepare_calls = 0
    from zeta.runtime import composition as composition_module

    original_load = composition_module.load_project_context

    def gated_load(**kwargs):
        nonlocal prepare_calls
        index = prepare_calls
        prepare_calls += 1
        if index == 0:
            prepare_entered.set()
            if not prepare_release.wait(30):
                raise TimeoutError("project-context prepare was not released")
        try:
            return original_load(**kwargs)
        finally:
            if index < 2:
                prepare_done[index].set()

    monkeypatch.setattr(composition_module, "load_project_context", gated_load)

    async def turn(text: str) -> None:
        await _collect_turn(app.loop, text)

    closing_app = None
    close_task = None
    close_read_release = threading.Event()
    try:
        initial = app.loop.context_assembler.system_prompt.content[0].text
        assert "CUSTOM-BEFORE" in initial and "CUSTOM-AFTER" in initial
        assert old_memory not in initial
        assert app.loop.project_context_health is not None

        (memory_dir / "state.md").write_text("RECOVERED-MEMORY", encoding="utf-8")
        for index in range(4):
            await turn(f"advance-{index}")
            if prepare_entered.is_set():
                break
        assert prepare_entered.is_set()
        skill = home / "skills" / "recovered" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(
            "---\nname: recovered\ndescription: RECOVERED-SKILL\n---\nbody\n",
            encoding="utf-8",
        )
        await turn("fourth")
        fourth_prompt = backend.calls[-1][0][0].content[0].text
        assert "RECOVERED-SKILL" in fourth_prompt
        assert "RECOVERED-MEMORY" not in fourth_prompt
        prepare_release.set()
        assert await asyncio.wait_for(
            asyncio.to_thread(prepare_done[0].wait), timeout=30
        )

        await turn("fifth")
        for index in range(3):
            if prepare_done[1].is_set():
                break
            await turn(f"retry-{index}")
        assert prepare_done[1].is_set()
        assert app.loop.project_context_health is not None

        await turn("sixth")
        recovered = backend.calls[-1][0][0].content[0].text
        assert "CUSTOM-BEFORE" in recovered and "CUSTOM-AFTER" in recovered
        assert "RECOVERED-SKILL" in recovered
        assert "RECOVERED-MEMORY" in recovered
        assert app.loop.project_context_health is None
        calls_after_recovery = projection_calls

        await turn("healthy")
        assert projection_calls == calls_after_recovery

        (memory_dir / "state.md").write_bytes(b"\xff")
        closing_app = await asyncio.to_thread(create_app, args)
        closing_app.loop.backend = FakeBackend(
            [ScriptedTurn(content=[TextContent("ok")])]
        )
        close_read_entered = threading.Event()
        original_close_projection = (
            closing_app.loop.project_registry._read_context_projection
        )

        def held_close_projection(operation):
            close_read_entered.set()
            if not close_read_release.wait(30):
                raise TimeoutError("project-context close read was not released")
            return original_close_projection(operation)

        # test-contract: seam Hold the production projection read during close.
        monkeypatch.setattr(
            closing_app.loop.project_registry,
            "_read_context_projection",
            held_close_projection,
        )
        await _collect_turn(closing_app.loop, "close")
        close_task = asyncio.create_task(closing_app.close())
        assert await asyncio.wait_for(
            asyncio.to_thread(close_read_entered.wait), timeout=30
        )
        # test-contract: seam Prove close joins the held projection read.
        assert not close_task.done()
        close_read_release.set()
        await close_task
    finally:
        prepare_release.set()
        close_read_release.set()
        if close_task is not None:
            await close_task
        elif closing_app is not None:
            await closing_app.close()
        await app.close()


async def _collect_turn(loop: Any, text: str) -> None:
    async for _event in loop.run_turn(text, origin=MessageOrigin.USER):
        pass


@pytest.mark.parametrize("damage", ["pointer", "manifest"])
def test_tui_omits_malformed_version_authority_but_keeps_inbox_guidance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    home = tmp_path / "home"
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("healthy", "git", repository)
    project_root = registry.root / project.project_id
    pointer_path = project_root / "memory-current.json"
    if damage == "manifest":
        registry.activate_entry_memory(project.project_id)
        registry._replace_entry_kind(project.project_id, "state", "current")
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        (
            project_root
            / "memory-versions"
            / "versions"
            / f"{pointer['current']}.json"
        ).write_text(
            "{", encoding="utf-8"
        )
    else:
        pointer_path.write_text("{", encoding="utf-8")
    legacy_path = project_root / "memory" / "brief.md"
    if legacy_path.exists():
        legacy_path.chmod(0o600)
    legacy_path.write_text("LEGACY-MUST-NOT-LEAK", encoding="utf-8")
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(repository)

    app = create_app(build_parser().parse_args(["--provider", "codex"]))
    try:
        prompt = app.loop.context_assembler.system_prompt.content[0].text
        assert "LEGACY-MUST-NOT-LEAK" not in prompt
        assert "Check your inbox with inbox action list" in prompt
        assert app.loop.project_context_health is not None
        expected_source = (
            "memory-current.json"
            if damage == "pointer"
            else f"memory-versions/versions/{pointer['current']}.json"
        )
        assert app.loop.project_context_health.source == expected_source
    finally:
        asyncio.run(app.close())


def test_missing_registry_remains_available(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path / "home")

    assert manager.project_registry is not None
    assert manager.project_registry_unavailable is None


def test_stale_project_roots_do_not_disable_registry(tmp_path: Path) -> None:
    home = tmp_path / "home"
    registry = ProjectRegistry(home / "projects")
    roots = [tmp_path / f"repository-{index}" for index in range(11)]
    projects = []
    for index, root in enumerate(roots):
        root.mkdir()
        projects.append(registry.create_project(f"project-{index}", "git", root))
    roots[2].rmdir()
    roots[7].rmdir()

    state = load_project_registry(home / "projects")

    assert state.unavailable is None
    assert state.registry is not None
    assert len(state.registry.list_projects()) == 11
    assert state.registry.find_for_directory(roots[5]) == projects[5]


def test_session_manager_construction_does_not_open_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_root = tmp_path / "home" / "projects"
    registry_root.mkdir(parents=True)
    opened: list[object] = []
    original_open = os.open

    def count_open(path: object, *args: object, **kwargs: object) -> int:
        if path == registry_root or kwargs.get("dir_fd") is not None:
            opened.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", count_open)

    SessionManager(tmp_path / "home")

    assert opened == []


@pytest.mark.asyncio
async def test_resolve_attention_does_not_open_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from zeta.runtime.project_registry_state import available_project_registry
    from zeta.tools import resolve_attention

    home = tmp_path / "home"
    session_dir = home / "sessions" / ("a" * 32)
    session_dir.mkdir(parents=True)
    state = available_project_registry(home / "projects")
    opened: list[object] = []
    original_open = os.open

    def count_open(path: object, *args: object, **kwargs: object) -> int:
        if path == home / "projects":
            opened.append(path)
        return original_open(path, *args, **kwargs)

    record = SimpleNamespace(decision="yes")
    monkeypatch.setattr(os, "open", count_open)
    monkeypatch.setattr(
        resolve_attention,
        "validate_attention_fork",
        lambda **_kwargs: SimpleNamespace(record=record),
    )
    monkeypatch.setattr(
        resolve_attention,
        "deliver_attention_decision",
        lambda *_args, **_kwargs: ("message-id", record, False),
    )
    registry = SimpleNamespace(
        session_store=SimpleNamespace(
            session_dir=session_dir,
            session_id=session_dir.name,
            directory_fd=-1,
        ),
        project_registry=state.registry,
        project_registry_state=state,
        project_id="p_" + "b" * 32,
    )

    await resolve_attention._resolve_attention(registry, {"decision": "yes"})

    assert opened == []


def test_malformed_registry_decisions_use_startup_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    _corrupt_registry(home, _invalid_json)
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(repository)

    app = create_app(build_parser().parse_args(["--provider", "codex"]))
    try:
        assert app.loop.manager.project_registry_unavailable is None
        assert app.loop.project_context_health is not None
        assert app.loop.project_context_health.code == "malformed_project"
        asyncio.run(app._decisions_tick())
        assert asyncio.run(app.slash_decisions("")) == "no open decisions"
    finally:
        asyncio.run(app.close())


def test_project_instructions_refuses_unavailable_registry(tmp_path: Path) -> None:
    from zeta.core.project_context import project_instructions_path

    home = tmp_path / "home"
    repository = tmp_path / "repository"
    repository.mkdir()
    record = _corrupt_registry(home, _invalid_json)
    state = load_project_registry(home / "projects")
    assert state.unavailable is not None

    result = project_instructions_path("", repository, home, registry_state=state)

    assert result == (
        "project instructions unavailable: project registry is temporarily "
        "unavailable; try again later"
    )
    assert not (record.parent / "AGENTS.md").exists()


def test_only_registry_state_constructs_project_registry() -> None:
    source_root = Path(__file__).parents[1] / "src" / "zeta"
    offenders = []
    for path in source_root.rglob("*.py"):
        if path.name in {"project_registry.py", "project_registry_state.py"}:
            continue
        if "ProjectRegistry(" in path.read_text(encoding="utf-8"):
            offenders.append(path.relative_to(source_root))

    assert offenders == []
