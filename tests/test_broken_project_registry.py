from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import stat
import subprocess
from collections.abc import Callable
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console
from rich.text import Text

from zeta.cli.main import build_parser
from zeta.core.session import SessionManager
from zeta.project_registry import ProjectRegistry
from zeta.runtime.project_registry_state import (
    ProjectRegistryUnavailable,
    load_project_registry,
)
from zeta.server.server import ZetaServer
from zeta.tui.app import create_app

_WARNING_PREFIX = "warning: project registry unavailable at "


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
    assert state.registry is None
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
            rendered = _render_startup(app)
            warning = next(
                item for item in app._startup_warnings if item.startswith(_WARNING_PREFIX)
            )
            assert warning.startswith(f"{_WARNING_PREFIX}{projects}: ")
            assert warning.endswith("; project features are disabled")
            assert rendered.count(warning) == 1
            assert "project" not in app.loop.tool_registry.registered_names
            assert "inbox" not in app.loop.tool_registry.registered_names
            assert app.loop.project_registry is None
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
        assert len(notices) == 1
        assert notices[0].startswith(f"{_WARNING_PREFIX}{projects}: ")
        assert notices[0].endswith("; project features are disabled")
        assert "projects" not in hello["capabilities"].get("features", ())
        assert _registry_snapshot(projects) == before
    finally:
        if record.exists():
            record.chmod(0o600)


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
        warning = app.loop.manager.project_registry_unavailable
        assert warning is not None
        assert app._startup_warnings.count(warning.warning) == 1
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

    assert result == f"project instructions unavailable: {state.unavailable.reason}"
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
