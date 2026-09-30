from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from zeta.cli.main import build_parser
from zeta.config.settings import load_settings
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.session import SessionManager
from zeta.project_registry import ProjectRegistry
from zeta.protocol.types import TextContent, ToolCall
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


def _create_session_process(home: str, root: str, queue: multiprocessing.Queue) -> None:
    """Spawn-safe worker for the cross-process registry race regression."""
    opened = SessionManager(home).create(provider="fake", model="test", cwd=root)
    queue.put(opened.metadata.project_id)
    opened.store.close()


def test_git_repo_at_home_does_not_create_project_for_descendant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_home = tmp_path / "home"
    user_home.mkdir()
    git(user_home, "init")
    nested = user_home / "src"
    nested.mkdir()
    monkeypatch.setenv("HOME", str(user_home))

    opened = manager(tmp_path).create(provider="fake", model="test", cwd=nested)

    assert opened.metadata.project_id is None
    assert ProjectRegistry(tmp_path / "zeta" / "projects").list_projects() == []
    opened.store.close()


def test_same_basename_repositories_get_distinct_stable_projects(tmp_path: Path) -> None:
    roots = []
    for parent_name in ("alpha", "beta"):
        root = tmp_path / parent_name / "repo"
        root.mkdir(parents=True)
        git(root, "init")
        roots.append(root)

    m = manager(tmp_path)
    first = [m.create(provider="fake", model="test", cwd=root) for root in roots]
    reopened = [m.create(provider="fake", model="test", cwd=root) for root in roots]

    first_ids = [opened.metadata.project_id for opened in first]
    assert len(set(first_ids)) == 2
    assert [opened.metadata.project_id for opened in reopened] == first_ids
    projects = {project.project_id: project for project in m.project_registry.list_projects()}
    assert [projects[project_id].name for project_id in first_ids] == ["repo", "repo-beta"]
    assert [projects[project_id].canonical_integration_root for project_id in first_ids] == [
        str(root.resolve()) for root in roots
    ]
    for opened in [*first, *reopened]:
        opened.store.close()


def test_ambient_git_repository_selection_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shadow = repo(tmp_path)
    plain = tmp_path / "plain"
    plain.mkdir()
    monkeypatch.setenv("GIT_DIR", str(shadow / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(shadow))
    monkeypatch.setenv("GIT_INDEX_FILE", str(shadow / ".git" / "index"))

    opened = manager(tmp_path).create(provider="fake", model="test", cwd=plain)

    assert opened.metadata.project_id is None
    assert ProjectRegistry(tmp_path / "zeta" / "projects").list_projects() == []
    opened.store.close()


def test_git_discovery_timeout_is_bounded_and_warns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text("#!/bin/sh\nsleep 6\n", encoding="utf-8")
    fake_git.chmod(0o755)
    plain = tmp_path / "plain"
    plain.mkdir()
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")

    started = time.monotonic()
    with caplog.at_level(logging.WARNING):
        opened = manager(tmp_path).create(provider="fake", model="test", cwd=plain)
    elapsed = time.monotonic() - started

    assert elapsed < 5
    assert opened.metadata.project_id is None
    assert "git project discovery timed out" in caplog.text
    opened.store.close()


def test_git_discovery_does_not_invoke_fsmonitor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.core import project_context

    root = repo(tmp_path)
    marker = tmp_path / "fsmonitor-ran"
    monitor = tmp_path / "fsmonitor.sh"
    monitor.write_text(f"#!/bin/sh\ntouch {marker}\nprintf '2\\n'\n", encoding="utf-8")
    monitor.chmod(0o755)
    git(root, "config", "core.fsmonitor", str(monitor))
    real_run = subprocess.run
    discovery_commands: list[list[str]] = []

    def record_run(command: list[str], *args: object, **kwargs: object):
        discovery_commands.append(command)
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(project_context.subprocess, "run", record_run)
    opened = manager(tmp_path).create(provider="fake", model="test", cwd=root)

    assert opened.metadata.project_id is not None
    assert discovery_commands
    assert all("core.fsmonitor=false" in command for command in discovery_commands)
    assert not marker.exists()
    opened.store.close()


def test_separate_git_dir_and_linked_worktree_share_primary_project(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    git_dir = tmp_path / "git-data"
    primary.mkdir()
    subprocess.run(
        ["git", "init", "--separate-git-dir", str(git_dir), str(primary)],
        check=True,
        capture_output=True,
    )
    git(primary, "config", "user.email", "test@example.com")
    git(primary, "config", "user.name", "Test")
    (primary / "README").write_text("x", encoding="utf-8")
    git(primary, "add", ".")
    git(primary, "commit", "-m", "initial")
    linked = tmp_path / "linked"
    git(primary, "worktree", "add", "-b", "linked", str(linked))

    m = manager(tmp_path)
    primary_session = m.create(provider="fake", model="test", cwd=primary)
    linked_session = m.create(provider="fake", model="test", cwd=linked)

    assert linked_session.metadata.project_id == primary_session.metadata.project_id
    projects = m.project_registry.list_projects()
    assert len(projects) == 1
    assert projects[0].canonical_integration_root == str(primary.resolve())
    primary_session.store.close()
    linked_session.store.close()


def test_first_auto_created_session_refreshes_project_memory_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.tui.app import create_app

    root = repo(tmp_path)
    home = tmp_path / "zeta-home"
    user_home = tmp_path / "user-home"
    user_home.mkdir()
    monkeypatch.setenv("HOME", str(user_home))
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(root)

    app = create_app(build_parser().parse_args(["--provider", "fake"]))
    session_id = app.loop.store.session_id
    project_id = app.loop.session_metadata.project_id
    assert project_id is not None
    assert "<zeta-project-memory>" in app.loop.session_metadata.system_prompt
    ProjectRegistry(home / "projects").update_memory(
        project_id, {"state.md": "fresh memory after first open"}
    )
    asyncio.run(app.close())

    resumed = create_app(
        build_parser().parse_args(["--provider", "fake", "--resume", session_id])
    )
    try:
        prompt = resumed.loop.context_assembler.system_prompt.content[0].text
        assert "fresh memory after first open" in prompt
    finally:
        asyncio.run(resumed.close())


@pytest.mark.asyncio
async def test_project_init_rebinds_runtime_child_and_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.tui.app import create_app

    plain = tmp_path / "plain"
    plain.mkdir()
    home = tmp_path / "zeta-home"
    user_home = tmp_path / "user-home"
    user_home.mkdir()
    monkeypatch.setenv("HOME", str(user_home))
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(plain)
    app = create_app(build_parser().parse_args(["--provider", "fake"]))
    try:
        assert app.loop.root_project_id is None
        assert "project: plain" in app.slash_project("init")
        project_id = app.loop.session_metadata.project_id
        assert project_id is not None
        assert app.loop.root_project_id == project_id
        assert app.loop.tool_registry.project_id == project_id

        approval = app.loop.tool_registry.approval_display(
            ToolCall(
                "update",
                "project_update",
                {"name": "state.md", "content": "new state"},
            )
        )
        assert approval.project_id == project_id
        assert approval.filename == "state.md"

        app.loop.backend = FakeBackend(
            [
                ScriptedTurn(
                    tool_calls=[
                        ToolCall(
                            "child",
                            "agent",
                            {"prompt": "work", "description": "child work"},
                        )
                    ]
                ),
                ScriptedTurn([TextContent("child done")]),
                ScriptedTurn([TextContent("root done")]),
            ]
        )
        async for _ in app.loop.run_turn("spawn child"):
            pass
        links = app.loop.project_registry.list_session_links(project_id)
        child_links = [item for item in links if item["role"] == "worker"]
        assert len(child_links) == 1
        assert child_links[0]["parent_session_id"] == app.loop.store.session_id
    finally:
        await app.close()


def test_project_init_uses_durable_pending_link_on_registry_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    root = tmp_path / "plain"
    root.mkdir()
    m = manager(tmp_path)
    opened = m.create(provider="fake", model="test", cwd=root, auto_project=False)
    project = m.project_registry.find_or_create_for_directory(root)

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("injected registry failure")

    monkeypatch.setattr(m.project_registry, "record_session", fail)
    with caplog.at_level(logging.WARNING):
        current = m.associate_project(opened.metadata, project.project_id)

    pending = m.sessions_dir / current.session_id / "project_link_pending.json"
    assert pending.exists()
    assert "linkage remains pending" in caplog.text
    assert m.read_metadata(current.session_id).project_id == project.project_id
    opened.store.close()


def test_concurrent_session_creation_across_processes_is_unique(tmp_path: Path) -> None:
    root = repo(tmp_path)
    home = tmp_path / "zeta"
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(target=_create_session_process, args=(str(home), str(root), queue))
        for _ in range(6)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0
    project_ids = [queue.get(timeout=2) for _ in processes]

    assert len(set(project_ids)) == 1
    assert len(ProjectRegistry(home / "projects").list_projects()) == 1


def test_real_cli_no_session_does_not_create_project(tmp_path: Path) -> None:
    root = repo(tmp_path)
    home = tmp_path / "zeta-home"
    user_home = tmp_path / "user-home"
    user_home.mkdir()
    env = {
        **os.environ,
        "HOME": str(user_home),
        "ZETA_HOME": str(home),
        "ZETA_TESTING": "1",
    }

    result = subprocess.run(
        ["zeta", "--provider", "fake", "--no-session", "--print", "hello"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert ProjectRegistry(home / "projects").list_projects() == []
