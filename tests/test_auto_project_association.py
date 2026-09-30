from __future__ import annotations

import asyncio
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from zeta.config.settings import load_settings
from zeta.core.session import SessionManager
from zeta.project_registry import ProjectRegistry
from zeta.skills import SkillCatalog
from zeta.tools.registry import ToolRegistry


def git(path: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=path, check=True, capture_output=True)


def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "Test")
    (root / "README").write_text("x")
    git(root, "add", ".")
    git(root, "commit", "-m", "initial")
    return root


def manager(tmp_path: Path) -> SessionManager:
    return SessionManager(tmp_path / "zeta")


def test_new_session_auto_creates_and_links_project(tmp_path: Path) -> None:
    root = repo(tmp_path)
    opened = manager(tmp_path).create(provider="fake", model="test", cwd=root)
    project = ProjectRegistry(tmp_path / "zeta" / "projects").show_project(opened.metadata.project_id)
    assert opened.metadata.project_id == project.project_id
    assert project.name == root.name
    assert project.canonical_integration_root == str(root.resolve())


def test_subdirectory_and_inside_outside_worktrees_share_one_project(tmp_path: Path) -> None:
    root = repo(tmp_path)
    inside = root / ".worktrees" / "x"
    inside.parent.mkdir()
    git(root, "worktree", "add", str(inside), "-b", "inside")
    outside = tmp_path / "outside-worktree"
    git(root, "worktree", "add", str(outside), "-b", "outside")
    (root / "sub").mkdir()
    m = manager(tmp_path)
    ids = [m.create(provider="fake", model="test", cwd=where).metadata.project_id
           for where in (root / "sub", inside, outside)]
    assert len(set(ids)) == 1
    assert len(ProjectRegistry(tmp_path / "zeta" / "projects").list_projects()) == 1


@pytest.mark.parametrize("location", ["plain", "home", "root"])
def test_non_git_home_and_filesystem_root_are_unassociated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, location: str) -> None:
    if location == "plain":
        cwd = tmp_path / "plain"; cwd.mkdir()
    elif location == "home":
        cwd = tmp_path / "home"; cwd.mkdir(); monkeypatch.setenv("HOME", str(cwd))
    else:
        cwd = Path("/")
    opened = manager(tmp_path).create(provider="fake", model="test", cwd=cwd)
    assert opened.metadata.project_id is None
    assert not list((tmp_path / "zeta" / "projects").glob("p_*/project.json"))


def test_auto_project_can_be_disabled(tmp_path: Path) -> None:
    root = repo(tmp_path)
    opened = manager(tmp_path).create(provider="fake", model="test", cwd=root, auto_project=False)
    assert opened.metadata.project_id is None


def test_concurrent_session_creation_is_unique(tmp_path: Path) -> None:
    root = repo(tmp_path)
    m = manager(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        sessions = list(pool.map(lambda _: m.create(provider="fake", model="test", cwd=root), range(8)))
    assert len({s.metadata.project_id for s in sessions}) == 1
    assert len(m.project_registry.list_projects()) == 1


def test_registry_failure_is_nonfatal_and_warns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    root = repo(tmp_path)
    m = manager(tmp_path)
    def fail(*args: object, **kwargs: object) -> object:
        raise OSError("registry down")
    monkeypatch.setattr(m.project_registry, "find_for_directory", fail)
    with caplog.at_level(logging.WARNING):
        opened = m.create(provider="fake", model="test", cwd=root)
    assert opened.metadata.project_id is None
    assert "continuing without project" in caplog.text


def test_resumed_session_keeps_association_and_child_inherits(tmp_path: Path) -> None:
    root = repo(tmp_path); m = manager(tmp_path)
    parent = m.create(provider="fake", model="test", cwd=root)
    resumed = m.open(parent.metadata.session_id)
    child = m.create(provider="fake", model="test", cwd=tmp_path, project_id=parent.metadata.project_id, parent_session_id=parent.metadata.session_id, project_role="worker")
    assert resumed.metadata.project_id == parent.metadata.project_id
    assert child.metadata.project_id == parent.metadata.project_id
    assert len(m.project_registry.list_projects()) == 1


def test_no_session_auto_project_flag_is_disabled(tmp_path: Path) -> None:
    root = repo(tmp_path)
    opened = manager(tmp_path).create(provider="fake", model="test", cwd=root, auto_project=False)
    assert opened.metadata.project_id is None


def test_project_tool_inspects_auto_associated_memory(tmp_path: Path) -> None:
    root = repo(tmp_path); m = manager(tmp_path)
    opened = m.create(provider="fake", model="test", cwd=root)
    tools = ToolRegistry(cwd=str(root), project_id=opened.metadata.project_id, project_registry=m.project_registry, skill_catalog=SkillCatalog.empty())
    # Registration is the public capability boundary; invoke the async handler directly.
    from zeta.tools.project import _inspect_project
    result = asyncio.run(_inspect_project(tools, {"action": "inspect"}))
    assert result["isError"] is False
    assert result["structuredContent"]["project"]["project_id"] == opened.metadata.project_id


def test_project_slash_show_and_init(tmp_path: Path) -> None:
    root = repo(tmp_path); m = manager(tmp_path)
    opened = m.create(provider="fake", model="test", cwd=root)
    class Loop:
        project_registry = m.project_registry
        manager = m
        session_metadata = opened.metadata
        class Store: cwd = str(root)
        store = Store()
    class Session:
        loop = Loop()
        def slash_project(self, args: str) -> str: return ""
    # Exercise the production slash handler through the session mixin implementation.
    from zeta.tui.slash_handlers import SlashHandlerMixin
    handler = object.__new__(SlashHandlerMixin); handler.loop = Loop()
    output = handler.slash_project("")
    assert "project: repo" in output
    plain = tmp_path / "plain"; plain.mkdir()
    opened2 = m.create(provider="fake", model="test", cwd=plain, auto_project=False)
    handler.loop.session_metadata = opened2.metadata; handler.loop.store.cwd = str(plain)
    assert "unassociated" in handler.slash_project("")
    assert "project: plain" in handler.slash_project("init")


def test_settings_auto_project_false(tmp_path: Path) -> None:
    settings = tmp_path / "settings.toml"
    settings.write_text("auto_project = false\n")
    loaded = load_settings(home=tmp_path)
    assert loaded.settings.auto_project is False
