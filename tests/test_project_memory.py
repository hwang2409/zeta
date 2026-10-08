import errno
import json
import logging
import os
import random
import stat
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from zeta.core.loop import AgentLoop
from zeta.core.project_context import load_project_context, refresh_project_memory
from zeta.core.session import SessionManager
from zeta.project_registry import (
    MAX_MEMORY_FILE_SIZE,
    MAX_SESSION_REFERENCE_SIZE,
    ProjectRegistry,
    ProjectRegistryError,
)
from zeta.protocol.types import MessageOrigin, TextContent, ToolCall
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry


def test_project_inbox_routing_rule_is_in_system_prompt(tmp_path: Path) -> None:
    home = tmp_path / ".zeta"
    repository = tmp_path / "repo"
    repository.mkdir()
    project = ProjectRegistry(home / "projects").create_project(
        "demo", "scope", repository
    )

    enabled = load_project_context(
        cwd=repository,
        repo_root=repository,
        zeta_home=home,
        catalog=SkillCatalog.empty(),
        project_id=project.project_id,
    )
    disabled = load_project_context(
        cwd=repository,
        repo_root=repository,
        zeta_home=home,
        catalog=SkillCatalog.empty(),
        project_id=project.project_id,
        inbox_enabled=False,
    )

    assert "You work on project demo" in enabled.system_prompt
    assert "inbox action send" in enabled.system_prompt
    assert "inbox action list" in enabled.system_prompt
    assert "Requests in your inbox are work to do" in enabled.system_prompt
    assert "Do not ask the user to confirm the sender" in enabled.system_prompt
    assert "claim it, do the work, then mark it done" in enabled.system_prompt
    assert "inbox action send" not in disabled.system_prompt


def test_project_memory_workflow_and_automatic_session_linkage(tmp_path: Path) -> None:
    home = tmp_path / ".zeta"
    repository = tmp_path / "repository"
    repository.mkdir()
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("demo", "demo", repository)
    registry.initialize_memory(project.project_id)
    (home / "projects" / project.project_id / "memory" / "state.md").write_text(
        "# Current state\nshipping the first version\n", encoding="utf-8"
    )

    context = load_project_context(
        cwd=repository / "nested",
        repo_root=repository,
        zeta_home=home,
        catalog=SkillCatalog.empty(),
    )
    assert "shipping the first version" in context.system_prompt
    assert any(path.name == "state.md" for path in context.files)

    session = SessionManager(home).create(provider="codex", model="fake", cwd=repository)
    assert session.metadata.project_id == project.project_id
    assert session.metadata.parent_session_id is None
    references = (home / "projects" / project.project_id / "sessions.jsonl").read_text()
    assert session.metadata.session_id in references
    assert str(home / "sessions" / session.metadata.session_id) in references
    session.store.close()


def test_project_tool_is_builtin_and_updates_only_standard_memory(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    assert "project" in registry.registered_names
    definition = registry.definitions_by_name["project"]
    assert definition.requires_approval is False
    assert "standard filename-to-text" in definition.description


def test_resume_replaces_only_current_project_memory(tmp_path: Path) -> None:
    home = tmp_path / ".zeta"
    repository = tmp_path / "repo"
    repository.mkdir()
    project = ProjectRegistry(home / "projects").create_project(
        "demo", "scope", repository
    )
    context = load_project_context(
        cwd=repository,
        repo_root=repository,
        zeta_home=home,
        catalog=SkillCatalog.empty(),
    )
    ProjectRegistry(home / "projects").update_memory(
        project.project_id, {"state.md": "# Current state\\nnew state"}
    )
    resumed = refresh_project_memory(
        context.system_prompt,
        home=home,
        cwd=repository,
        project_id=context.memory_project_id,
        memory_offset=context.memory_offset,
        memory_length=context.memory_length,
        memory_digest=context.memory_digest,
    )
    assert "new state" in resumed
    assert "# Current state\\n" not in resumed or "new state" in resumed


def test_duplicate_roots_and_update_tool_boundary(tmp_path: Path) -> None:
    root = tmp_path / "projects"
    repository = tmp_path / "repo"
    repository.mkdir()
    registry = ProjectRegistry(root)
    registry.create_project("one", "scope", repository)
    with pytest.raises(ProjectRegistryError, match="canonical integration root"):
        registry.create_project("two", "scope", repository)
    tools = ToolRegistry(repository, skill_catalog=SkillCatalog.empty())
    assert tools.definitions_by_name["project_update"].requires_approval
    assert (
        tools.definitions_by_name["project_update"].parameters["additionalProperties"]
        is False
    )
    assert tools.definitions_by_name["project_update"].parameters["properties"]["name"][
        "enum"
    ] == ["brief.md", "state.md", "backlog.md", "changelog.md", "decisions.md"]


def test_project_memory_is_bounded_and_rejects_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "projects"
    repository = tmp_path / "repo"
    repository.mkdir()
    registry = ProjectRegistry(root)
    project = registry.create_project("demo", "demo", repository)
    registry.initialize_memory(project.project_id)
    memory = root / project.project_id / "memory"
    (memory / "brief.md").write_text("x" * 100, encoding="utf-8")
    assert all(
        name != "brief.md"
        for name, _ in registry.load_memory(project.project_id, byte_cap=10)
    )
    (memory / "state.md").unlink()
    (memory / "state.md").symlink_to(memory / "brief.md")
    with pytest.raises(ProjectRegistryError):
        registry.load_memory(project.project_id)


def test_project_registry_serializes_concurrent_creates_and_rejects_hardlinks(
    tmp_path: Path,
) -> None:
    root = tmp_path / "projects"
    registry = ProjectRegistry(root)
    with ThreadPoolExecutor(max_workers=4) as pool:
        projects = list(
            pool.map(
                lambda index: registry.create_project(f"project-{index}", "scope"),
                range(12),
            )
        )
    assert len({project.project_id for project in projects}) == 12
    project = projects[0]
    memory = root / project.project_id / "memory"
    external = tmp_path / "external.md"
    external.write_text("secret", encoding="utf-8")
    (memory / "brief.md").unlink()
    (memory / "brief.md").hardlink_to(external)
    with pytest.raises(ProjectRegistryError):
        registry.load_memory(project.project_id)


def test_root_link_pending_intent_dir_fsync_precedes_record_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / ".zeta"
    repository = tmp_path / "repo"
    repository.mkdir()
    ProjectRegistry(home / "projects").create_project("demo", "scope", repository)

    events: list[str] = []
    real_fsync = os.fsync

    def spy_fsync(fd: int) -> None:
        try:
            is_dir = stat.S_ISDIR(os.fstat(fd).st_mode)
        except OSError:
            is_dir = False
        events.append("fsync_dir" if is_dir else "fsync_file")
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", spy_fsync)

    manager = SessionManager(home)
    real_record = manager.project_registry.record_session

    def spy_record(*args: object, **kwargs: object) -> None:
        events.append("record_session")
        return real_record(*args, **kwargs)

    monkeypatch.setattr(manager.project_registry, "record_session", spy_record)

    session = manager.create(provider="codex", model="fake", cwd=repository)
    session.store.close()

    assert "record_session" in events
    first_record = events.index("record_session")
    # The durable intent's directory entry is fsynced immediately before the
    # registry is published, so a crash between the two boundaries is
    # recoverable rather than losing the rename.
    assert events[first_record - 1] == "fsync_dir"


@pytest.mark.parametrize(
    "case",
    ["absent", "malformed", "non_dict", "empty", "wrong_types"],
)
def test_root_link_reconstructs_from_metadata(tmp_path: Path, case: str) -> None:
    home = tmp_path / ".zeta"
    repository = tmp_path / "repo"
    repository.mkdir()
    project = ProjectRegistry(home / "projects").create_project(
        "demo", "scope", repository
    )

    manager = SessionManager(home)
    session = manager.create(provider="codex", model="fake", cwd=repository)
    session_id = session.metadata.session_id
    session.store.close()

    # Simulate a crash where the registry publication never landed: drop the
    # recorded link and leave the pending-intent file absent or corrupt.
    links_path = home / "projects" / project.project_id / "sessions.jsonl"
    links_path.unlink()
    session_dir = home / "sessions" / session_id
    pending = session_dir / "project_link_pending.json"
    if case == "absent":
        pending.unlink(missing_ok=True)
    elif case == "malformed":
        pending.write_text("{not valid json", encoding="utf-8")
    elif case == "non_dict":
        pending.write_text("[1, 2, 3]", encoding="utf-8")
    elif case == "empty":
        pending.write_text("{}", encoding="utf-8")
    elif case == "wrong_types":
        pending.write_text(
            json.dumps(
                {
                    "project_id": 123,
                    "role": "session",
                    "parent_session_id": None,
                    "transcript_path": str(session_dir),
                }
            ),
            encoding="utf-8",
        )

    # Re-opening the session must reconstruct the link from immutable metadata.
    reopened = SessionManager(home)
    opened = reopened.open(session_id)
    opened.store.close()

    records = reopened.project_registry.list_session_links(project.project_id)
    matched = [record for record in records if record["session_id"] == session_id]
    assert len(matched) == 1
    assert matched[0]["role"] == "session"
    assert matched[0]["parent_session_id"] is None
    assert matched[0]["transcript_path"] == str(session_dir)


def _agent_call(call_id: str, prompt: str) -> ToolCall:
    return ToolCall(call_id, "agent", {"prompt": prompt, "description": "task"})


def _nested_spawn_turns() -> list[ScriptedTurn]:
    # One root turn spawns a child; the child spawns a grandchild, then finishes.
    return [
        ScriptedTurn(tool_calls=[_agent_call("child-call", "child work")]),
        ScriptedTurn(tool_calls=[_agent_call("grand-call", "grand work")]),
        ScriptedTurn([TextContent("leaf done")]),
        ScriptedTurn([TextContent("child done")]),
    ]


async def _run_root_with_child_and_grandchild(manager: SessionManager, repository: Path):
    opened = manager.create(provider="codex", model="fake", cwd=repository)
    loop = AgentLoop(
        FakeBackend(_nested_spawn_turns()),
        opened.store,
        max_turns=1,
        skill_catalog=SkillCatalog.empty(),
        root_project_id=opened.metadata.project_id,
        project_registry=manager.project_registry,
    )
    try:
        async for _ in loop.run_turn("start", origin=MessageOrigin.USER):
            pass
    finally:
        await loop.close()
    return opened


@pytest.mark.asyncio
async def test_child_lineage_two_roots_each_with_child_and_grandchild(
    tmp_path: Path,
) -> None:
    home = tmp_path / ".zeta"
    repository = tmp_path / "repo"
    repository.mkdir()
    project = ProjectRegistry(home / "projects").create_project(
        "demo", "scope", repository
    )

    manager = SessionManager(home)
    root_a = await _run_root_with_child_and_grandchild(manager, repository)
    root_b = await _run_root_with_child_and_grandchild(manager, repository)

    records = manager.project_registry.list_session_links(project.project_id)
    by_id = {record["session_id"]: record for record in records}
    assert len(by_id) == 6

    for root in (root_a, root_b):
        root_id = root.metadata.session_id
        child_id = f"{root_id}:1"
        grand_id = f"{root_id}:1:1"
        assert by_id[root_id]["role"] == "session"
        assert by_id[root_id]["parent_session_id"] is None
        assert by_id[child_id]["role"] == "worker"
        assert by_id[child_id]["parent_session_id"] == root_id
        assert by_id[grand_id]["role"] == "worker"
        assert by_id[grand_id]["parent_session_id"] == child_id


@pytest.mark.asyncio
async def test_child_lineage_reconciled_after_registry_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / ".zeta"
    repository = tmp_path / "repo"
    repository.mkdir()
    project = ProjectRegistry(home / "projects").create_project(
        "demo", "scope", repository
    )

    manager = SessionManager(home)
    opened = manager.create(provider="codex", model="fake", cwd=repository)
    root_id = opened.metadata.session_id

    # Inject a registry-append failure for the child lineage records: the
    # durable intent is written and fsynced before this raises, so it must be
    # recoverable on the next open.
    real_record = manager.project_registry.record_session

    def failing_record(*args: object, **kwargs: object) -> None:
        if ":" in str(kwargs.get("session_id", "")):
            raise ProjectRegistryError("injected append failure")
        return real_record(*args, **kwargs)

    monkeypatch.setattr(manager.project_registry, "record_session", failing_record)

    loop = AgentLoop(
        FakeBackend(_nested_spawn_turns()),
        opened.store,
        max_turns=1,
        skill_catalog=SkillCatalog.empty(),
        root_project_id=opened.metadata.project_id,
        project_registry=manager.project_registry,
    )
    try:
        async for _ in loop.run_turn("start", origin=MessageOrigin.USER):
            pass
    finally:
        await loop.close()
    opened.store.close()

    # Only the root link was published; the child/grandchild appends failed.
    before = manager.project_registry.list_session_links(project.project_id)
    assert [record["session_id"] for record in before] == [root_id]
    monkeypatch.undo()

    def _link_ids(mgr: SessionManager) -> list[str]:
        opened_root = mgr.open(root_id)
        opened_root.store.close()
        return [
            record["session_id"]
            for record in mgr.project_registry.list_session_links(project.project_id)
        ]

    child_id = f"{root_id}:1"
    grand_id = f"{root_id}:1:1"

    first = sorted(_link_ids(SessionManager(home)))
    assert first == sorted([root_id, child_id, grand_id])
    # A second open must not double-publish any link.
    second = sorted(_link_ids(SessionManager(home)))
    assert second == first


@pytest.mark.asyncio
async def test_project_tools_use_session_manager_home_not_ambient_zeta_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.core.approval import ApprovalDecision, ApprovalPolicy

    home_a = tmp_path / "home_a"
    home_b = tmp_path / "home_b"
    repository = tmp_path / "repo"
    repository.mkdir()
    registry_a = ProjectRegistry(home_a / "projects")
    project = registry_a.create_project("demo", "scope", repository)
    registry_a.initialize_memory(project.project_id)

    manager = SessionManager(home_a)
    opened = manager.create(provider="codex", model="fake", cwd=repository)

    # The ambient environment points at a different home; the tools must ignore
    # it and use the registry bound to the SessionManager's home instead.
    monkeypatch.setenv("ZETA_HOME", str(home_b))

    turns = [
        ScriptedTurn(
            tool_calls=[
                ToolCall("root-inspect", "project", {"action": "inspect"}),
                ToolCall(
                    "root-update",
                    "project_update",
                    {"name": "state.md", "content": "ROOT-EDIT"},
                ),
            ]
        ),
        ScriptedTurn(tool_calls=[_agent_call("child-call", "child work")]),
        ScriptedTurn(
            tool_calls=[
                ToolCall("child-inspect", "project", {"action": "inspect"}),
                ToolCall(
                    "child-update",
                    "project_update",
                    {"name": "backlog.md", "content": "CHILD-EDIT"},
                ),
            ]
        ),
        ScriptedTurn([TextContent("child done")]),
    ]
    policy = ApprovalPolicy(store=opened.store, default=ApprovalDecision.ALLOW)
    loop = AgentLoop(
        FakeBackend(turns),
        opened.store,
        approval_policy=policy,
        max_turns=2,
        skill_catalog=SkillCatalog.empty(),
        skip_mcp_mount=True,
        root_project_id=opened.metadata.project_id,
        project_registry=manager.project_registry,
    )
    try:
        async for _ in loop.run_turn("start", origin=MessageOrigin.USER):
            pass
    finally:
        await loop.close()
    opened.store.close()

    memory = dict(registry_a.load_memory(project.project_id))
    assert memory["state.md"] == "ROOT-EDIT"
    assert memory["backlog.md"] == "CHILD-EDIT"
    # The ambient ZETA_HOME must never be created or written by the tools.
    assert not (home_b / "projects").exists()


_VALID_LINK = (
    '{"parent_session_id":null,"recorded_at":"2024-01-01T00:00:00.000000Z",'
    '"role":"session","session_id":"%s","transcript_path":"/x"}'
)


def _write_sessions_file(path: Path, data: bytes) -> None:
    path.write_bytes(data)
    os.chmod(path, 0o600)


@pytest.mark.parametrize(
    "case",
    ["torn_final", "malformed_middle", "oversized_final", "fifo", "symlink", "hardlink"],
)
def test_list_session_links_file_safety(tmp_path: Path, case: str) -> None:
    root = tmp_path / "projects"
    registry = ProjectRegistry(root)
    project = registry.create_project("demo", "scope")
    project_dir = root / project.project_id
    sessions = project_dir / "sessions.jsonl"

    first = (_VALID_LINK % ("a" * 32)).encode("utf-8")
    if case == "torn_final":
        # A crash can leave one unterminated final record; the reader tolerates
        # that bounded tail and returns only the completed records.
        _write_sessions_file(sessions, first + b"\n" + (_VALID_LINK % ("b" * 32)).encode())
        links = registry.list_session_links(project.project_id)
        assert [link["session_id"] for link in links] == ["a" * 32]
        return
    if case == "malformed_middle":
        _write_sessions_file(
            sessions, first + b"\n" + b"{not valid json}\n" + first + b"\n"
        )
    elif case == "oversized_final":
        _write_sessions_file(
            sessions, first + b"\n" + b"x" * (MAX_SESSION_REFERENCE_SIZE + 8)
        )
    elif case == "fifo":
        os.mkfifo(sessions, 0o600)
    elif case == "symlink":
        target = tmp_path / "outside.jsonl"
        _write_sessions_file(target, first + b"\n")
        sessions.symlink_to(target)
    elif case == "hardlink":
        target = tmp_path / "outside.jsonl"
        _write_sessions_file(target, first + b"\n")
        os.link(target, sessions)

    with pytest.raises(ProjectRegistryError):
        registry.list_session_links(project.project_id)


def test_memory_read_handles_short_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "projects"
    registry = ProjectRegistry(root)
    project = registry.create_project("demo", "scope")
    registry.initialize_memory(project.project_id)
    memory_dir = root / project.project_id / "memory"
    for other in ("brief.md", "backlog.md", "changelog.md", "decisions.md"):
        (memory_dir / other).unlink(missing_ok=True)
    payload = "HELLO-" * 2000
    (memory_dir / "state.md").write_text(payload, encoding="utf-8")

    real_read = os.read

    def dribbling_read(fd: int, count: int) -> bytes:
        # A regular file can legally satisfy a read with fewer bytes than asked;
        # the reader must loop until EOF rather than trusting one read.
        return real_read(fd, min(count, 5))

    monkeypatch.setattr(os, "read", dribbling_read)
    loaded = dict(registry.load_memory(project.project_id))
    assert loaded == {"state.md": payload}


def test_memory_read_rejects_growth_beyond_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "projects"
    registry = ProjectRegistry(root)
    project = registry.create_project("demo", "scope")
    registry.initialize_memory(project.project_id)
    memory_dir = root / project.project_id / "memory"
    for other in ("brief.md", "backlog.md", "changelog.md", "decisions.md"):
        (memory_dir / other).unlink(missing_ok=True)
    (memory_dir / "state.md").write_text("small", encoding="utf-8")

    def growing_read(fd: int, count: int) -> bytes:
        # Simulate a file that keeps yielding bytes past the cap after the
        # initial stat; the bounded reader must refuse it, never allocate past
        # the cap while holding the registry lock.
        return b"a" * count

    monkeypatch.setattr(os, "read", growing_read)
    with pytest.raises(ProjectRegistryError, match="too large"):
        registry.load_memory(project.project_id)
    assert MAX_MEMORY_FILE_SIZE > 0


def _reconcile_fixture(tmp_path: Path):
    home = tmp_path / ".zeta"
    repository = tmp_path / "repo"
    repository.mkdir()
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("demo", "scope", repository)
    manager = SessionManager(home)
    opened = manager.create(provider="codex", model="fake", cwd=repository)
    root_id = opened.metadata.session_id
    opened.store.close()
    return manager, project, root_id


def _write_child_intent(
    sessions_dir: Path,
    root_id: str,
    project_id: str,
    session_id: str,
    parent: str | None,
) -> None:
    """Seed one pending child-lineage intent in the root's durable index."""
    from zeta.core.session_links import persist_pending_child_link

    link = {
        "project_id": project_id,
        "session_id": session_id,
        "role": "worker",
        "parent_session_id": parent,
        "transcript_path": f"/sessions/{session_id}",
    }
    persist_pending_child_link(sessions_dir / root_id, link)


def _child_link_ids(registry: ProjectRegistry, project_id: str) -> list[str]:
    return sorted(
        record["session_id"]
        for record in registry.list_session_links(project_id)
        if ":" in str(record["session_id"])
    )


def _quarantined_path(quarantine_dir: Path, original_name: str) -> Path:
    matches = list(quarantine_dir.glob(f"{original_name}.*"))
    assert len(matches) == 1
    return matches[0]


@pytest.fixture(params=("reversed", "shuffled"))
def reordered_pending_scandir(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
):
    """Install a deterministic non-native order for one pending index."""
    from zeta.core import session_links

    real_scandir = os.scandir

    class OrderedScandir:
        def __init__(self, names: list[str]) -> None:
            self._entries = [type("Entry", (), {"name": name})() for name in names]

        def __iter__(self):
            return iter(self._entries)

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    def install(pending_dir: Path) -> None:
        pending_stat = pending_dir.stat()

        def reordered(path):
            if isinstance(path, int) and os.path.samestat(os.fstat(path), pending_stat):
                with real_scandir(path) as entries:
                    names = sorted(entry.name for entry in entries)
                if request.param == "reversed":
                    names.reverse()
                else:
                    random.Random(3).shuffle(names)
                return OrderedScandir(names)
            return real_scandir(path)

        monkeypatch.setattr(session_links.os, "scandir", reordered)

    return install


def test_child_reconciliation_makes_progress_across_budgeted_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.core import session_links
    from zeta.core.session_links import reconcile_child_links

    manager, project, root_id = _reconcile_fixture(tmp_path)
    total = 6
    ids = [f"{root_id}:{i}" for i in range(1, total + 1)]
    for sid in ids:
        _write_child_intent(
            manager.sessions_dir, root_id, project.project_id, sid, root_id
        )
    # A tight per-pass bound truncates each pass, but because published entries
    # leave the index every pass makes forward progress.
    monkeypatch.setattr(session_links, "_CHILD_LINK_MAX_ENTRIES_PER_PASS", 2)
    registry = manager.project_registry

    seen: list[int] = []
    for _ in range(total):  # generous upper bound on passes
        reconcile_child_links(registry, manager.sessions_dir, root_id)
        seen.append(len(_child_link_ids(registry, project.project_id)))
        if seen[-1] == total:
            break
    # Every link was eventually published, and it took more than one pass.
    assert _child_link_ids(registry, project.project_id) == sorted(ids)
    assert len(seen) > 1
    # No duplicate session links despite repeated bounded passes.
    records = registry.list_session_links(project.project_id)
    published_ids = [r["session_id"] for r in records if ":" in str(r["session_id"])]
    assert sorted(published_ids) == sorted(ids)
    assert len(published_ids) == len(set(published_ids))


def test_child_reconciliation_does_not_walk_agents_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.core.session_files import session_directory
    from zeta.core.session_links import (
        PENDING_CHILD_LINKS_DIRNAME,
        reconcile_child_links,
    )

    manager, project, root_id = _reconcile_fixture(tmp_path)
    ids = [f"{root_id}:{i}" for i in range(1, 4)]
    for sid in ids:
        _write_child_intent(
            manager.sessions_dir, root_id, project.project_id, sid, root_id
        )
    # Plant an agents subtree that would trip any traversal: if reconciliation
    # ever opened it the spy below would record the visit.
    with session_directory(manager.sessions_dir, root_id) as (_, root_fd):
        os.mkdir("agents", 0o700, dir_fd=root_fd)
        agents_fd = os.open("agents", os.O_RDONLY | os.O_DIRECTORY, dir_fd=root_fd)
        try:
            os.mkdir("1", 0o700, dir_fd=agents_fd)
        finally:
            os.close(agents_fd)

    real_open = os.open
    walked_agents = {"hit": False}

    def spy_open(path, *args, **kwargs):
        if path == "agents" or (
            isinstance(path, (str, bytes)) and os.fspath(path) in ("agents", b"agents")
        ):
            walked_agents["hit"] = True
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", spy_open)
    reconcile_child_links(manager.project_registry, manager.sessions_dir, root_id)
    monkeypatch.undo()
    assert walked_agents["hit"] is False
    # The flat index was still fully reconciled.
    assert _child_link_ids(manager.project_registry, project.project_id) == sorted(ids)
    assert PENDING_CHILD_LINKS_DIRNAME  # index name is a public constant


def test_child_reconciliation_resumes_on_next_open_without_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.core.session_links import reconcile_child_links

    manager, project, root_id = _reconcile_fixture(tmp_path)
    ids = [f"{root_id}:{i}" for i in range(1, 4)]
    for sid in ids:
        _write_child_intent(
            manager.sessions_dir, root_id, project.project_id, sid, root_id
        )
    registry = manager.project_registry

    def boom(*args: object, **kwargs: object) -> None:
        raise ProjectRegistryError("injected batch failure")

    monkeypatch.setattr(registry, "record_sessions", boom, raising=False)
    reconcile_child_links(registry, manager.sessions_dir, root_id)
    # The batch publish failed, so the durable intents remain unpublished.
    assert _child_link_ids(registry, project.project_id) == []
    monkeypatch.undo()

    reconcile_child_links(registry, manager.sessions_dir, root_id)
    assert _child_link_ids(registry, project.project_id) == sorted(ids)
    # A later open must not double-publish any link.
    reconcile_child_links(registry, manager.sessions_dir, root_id)
    assert _child_link_ids(registry, project.project_id) == sorted(ids)


def test_child_reconciliation_reads_registry_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.core.session_links import reconcile_child_links

    manager, project, root_id = _reconcile_fixture(tmp_path)
    total = 5
    for i in range(1, total + 1):
        _write_child_intent(
            manager.sessions_dir, root_id, project.project_id, f"{root_id}:{i}", root_id
        )
    registry = manager.project_registry
    original = registry._read_session_records
    reads = {"count": 0}

    def counting(*args: object, **kwargs: object):
        reads["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(registry, "_read_session_records", counting)
    reconcile_child_links(registry, manager.sessions_dir, root_id)
    # The whole reconciliation reads the JSONL registry exactly once.
    assert reads["count"] == 1
    monkeypatch.undo()
    assert _child_link_ids(registry, project.project_id) == sorted(
        f"{root_id}:{i}" for i in range(1, total + 1)
    )


def test_pending_index_fsyncs_root_dir_before_record_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.core.session_links import persist_pending_child_link

    manager, project, root_id = _reconcile_fixture(tmp_path)
    # The pending index directory's own entry lives in the root session dir; a
    # crash after persist returns but before the registry append must not lose
    # it, so root_fd itself has to be fsynced -- not just the index dir and the
    # intent file inside it.
    root_stat = os.stat(manager.sessions_dir / root_id)

    events: list[str] = []
    real_fsync = os.fsync

    def spy_fsync(fd: int) -> None:
        try:
            if os.path.samestat(os.fstat(fd), root_stat):
                events.append("fsync_root_dir")
        except OSError:
            pass
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", spy_fsync)

    registry = manager.project_registry
    real_record = registry.record_session

    def spy_record(*args: object, **kwargs: object):
        events.append("record_session")
        return real_record(*args, **kwargs)

    monkeypatch.setattr(registry, "record_session", spy_record)

    child_id = f"{root_id}:1"
    link = {
        "project_id": project.project_id,
        "session_id": child_id,
        "role": "worker",
        "parent_session_id": root_id,
        "transcript_path": f"/sessions/{child_id}",
    }
    # Mirror the runner's ordering: persist the durable intent, then append.
    persist_pending_child_link(manager.sessions_dir / root_id, link)
    registry.record_session(
        project.project_id,
        session_id=child_id,
        transcript_path=f"/sessions/{child_id}",
        role="worker",
        parent_session_id=root_id,
    )

    assert "record_session" in events
    first_record = events.index("record_session")
    assert "fsync_root_dir" in events[:first_record]


def test_invalid_entries_are_quarantined_not_deleted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reordered_pending_scandir,
) -> None:
    from zeta.core import session_links
    from zeta.core.session_links import (
        PENDING_CHILD_LINKS_DIRNAME,
        reconcile_child_links,
    )

    manager, project, root_id = _reconcile_fixture(tmp_path)
    valid_ids = [f"{root_id}:{i}" for i in range(1, 4)]
    # Seed one valid intent first so the durable index directory exists.
    _write_child_intent(
        manager.sessions_dir, root_id, project.project_id, valid_ids[0], root_id
    )
    pending_dir = manager.sessions_dir / root_id / PENDING_CHILD_LINKS_DIRNAME

    # Permanently unpublishable entries planted ahead of the valid intents:
    # malformed JSON, schema-invalid JSON, an oversized blob, an empty object,
    # and a symlink.  None can ever be published; all must be unlinked so they
    # never occupy the first page forever.
    invalid_names = ["malformed", "wrong_schema", "oversized", "empty_json", "symlinked"]
    (pending_dir / "malformed").write_text("{ not json", encoding="utf-8")
    (pending_dir / "wrong_schema").write_text(
        json.dumps({"foo": "bar"}), encoding="utf-8"
    )
    (pending_dir / "oversized").write_text("x" * 9000, encoding="utf-8")
    (pending_dir / "empty_json").write_text("{}", encoding="utf-8")
    (pending_dir / "symlinked").symlink_to(tmp_path / "does-not-exist")

    # Seed the remaining valid intents behind the invalid ones.
    for sid in valid_ids[1:]:
        _write_child_intent(
            manager.sessions_dir, root_id, project.project_id, sid, root_id
        )

    reordered_pending_scandir(pending_dir)
    monkeypatch.setattr(session_links, "_CHILD_LINK_MAX_ENTRIES_PER_PASS", 2)
    registry = manager.project_registry

    for _ in range(30):  # generous upper bound on passes
        reconcile_child_links(registry, manager.sessions_dir, root_id)
        all_published = _child_link_ids(registry, project.project_id) == sorted(valid_ids)
        all_invalid_quarantined = all(
            not os.path.lexists(pending_dir / name) for name in invalid_names
        )
        if all_published and all_invalid_quarantined:
            break

    # Every valid intent was eventually published despite the invalid entries.
    assert _child_link_ids(registry, project.project_id) == sorted(valid_ids)
    # And every permanently invalid entry moved intact out of the index.
    quarantine_dir = pending_dir.parent / "pending_child_links.invalid"
    for name in invalid_names:
        assert not os.path.lexists(pending_dir / name)
        assert os.path.lexists(_quarantined_path(quarantine_dir, name))
    assert _quarantined_path(quarantine_dir, "malformed").read_text(
        encoding="utf-8"
    ) == "{ not json"
    assert _quarantined_path(quarantine_dir, "oversized").read_text(
        encoding="utf-8"
    ) == "x" * 9000
    remaining = [n for n in os.listdir(pending_dir) if not n.startswith(".")]
    assert remaining == []


def test_quarantine_handles_max_length_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.core import session_links
    from zeta.core.session_links import (
        PENDING_CHILD_LINKS_DIRNAME,
        reconcile_child_links,
    )

    manager, project, root_id = _reconcile_fixture(tmp_path)
    valid_id = f"{root_id}:later-valid"
    _write_child_intent(
        manager.sessions_dir, root_id, project.project_id, valid_id, root_id
    )
    pending_dir = manager.sessions_dir / root_id / PENDING_CHILD_LINKS_DIRNAME
    valid_name = session_links._encode_link_name(valid_id)
    invalid_name = "x" * 255
    (pending_dir / invalid_name).write_text("not json", encoding="utf-8")

    monkeypatch.setattr(
        session_links,
        "_scan_pending_entries",
        lambda _fd: ([invalid_name, valid_name], [], False),
    )
    reconcile_child_links(manager.project_registry, manager.sessions_dir, root_id)

    assert _child_link_ids(manager.project_registry, project.project_id) == [valid_id]
    assert not (pending_dir / invalid_name).exists()
    quarantine_dir = pending_dir.parent / "pending_child_links.invalid"
    quarantined = list(quarantine_dir.iterdir())
    assert len(quarantined) == 1
    assert quarantined[0].read_text(encoding="utf-8") == "not json"
    assert len(os.fsencode(quarantined[0].name)) <= 255


@pytest.mark.parametrize(
    ("failure_point", "error_number"),
    [("open", errno.EMFILE), ("read", errno.EIO)],
)
def test_transient_read_error_keeps_intent_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
    error_number: int,
) -> None:
    from zeta.core import session_links
    from zeta.core.session_links import (
        PENDING_CHILD_LINKS_DIRNAME,
        reconcile_child_links,
    )

    manager, project, root_id = _reconcile_fixture(tmp_path)
    blocked_id = f"{root_id}:blocked"
    later_id = f"{root_id}:later"
    for child_id in (blocked_id, later_id):
        _write_child_intent(
            manager.sessions_dir, root_id, project.project_id, child_id, root_id
        )
    pending_dir = manager.sessions_dir / root_id / PENDING_CHILD_LINKS_DIRNAME
    blocked_name = session_links._encode_link_name(blocked_id)
    later_name = session_links._encode_link_name(later_id)
    blocked_stat = (pending_dir / blocked_name).stat()

    # Fix the scan order so the test proves that the transient failure stops the
    # pass before a later, otherwise valid intent can be published.
    monkeypatch.setattr(
        session_links,
        "_scan_pending_entries",
        lambda _fd: ([blocked_name, later_name], [], False),
    )
    if failure_point == "open":
        real_open = session_links.open_session_file

        def failing_open(directory_fd: int, name: str, flags: int) -> int:
            if name == blocked_name:
                raise OSError(error_number, os.strerror(error_number))
            return real_open(directory_fd, name, flags)

        monkeypatch.setattr(session_links, "open_session_file", failing_open)
    else:
        real_read = os.read

        def failing_read(fd: int, count: int) -> bytes:
            if os.path.samestat(os.fstat(fd), blocked_stat):
                raise OSError(error_number, os.strerror(error_number))
            return real_read(fd, count)

        monkeypatch.setattr(session_links.os, "read", failing_read)

    reconcile_child_links(manager.project_registry, manager.sessions_dir, root_id)

    assert _child_link_ids(manager.project_registry, project.project_id) == []
    assert (pending_dir / blocked_name).exists()
    assert (pending_dir / later_name).exists()
    quarantine_dir = pending_dir.parent / "pending_child_links.invalid"
    assert not quarantine_dir.exists() or list(quarantine_dir.iterdir()) == []

    monkeypatch.undo()
    reconcile_child_links(manager.project_registry, manager.sessions_dir, root_id)
    assert _child_link_ids(manager.project_registry, project.project_id) == sorted(
        [blocked_id, later_id]
    )
    assert list(pending_dir.iterdir()) == []


def test_directory_entries_do_not_stall_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reordered_pending_scandir,
) -> None:
    from zeta.core import session_links
    from zeta.core.session_links import (
        PENDING_CHILD_LINKS_DIRNAME,
        reconcile_child_links,
    )

    manager, project, root_id = _reconcile_fixture(tmp_path)
    valid_ids = [f"{root_id}:valid-{i}" for i in range(2)]
    for child_id in valid_ids:
        _write_child_intent(
            manager.sessions_dir, root_id, project.project_id, child_id, root_id
        )
    pending_dir = manager.sessions_dir / root_id / PENDING_CHILD_LINKS_DIRNAME
    directory_names = [f"invalid-directory-{i}" for i in range(5)]
    for name in directory_names:
        entry = pending_dir / name
        entry.mkdir()
        (entry / "marker").write_text(name, encoding="utf-8")

    reordered_pending_scandir(pending_dir)
    monkeypatch.setattr(session_links, "_CHILD_LINK_MAX_ENTRIES_PER_PASS", 2)

    for _ in range(10):
        reconcile_child_links(manager.project_registry, manager.sessions_dir, root_id)
        all_published = _child_link_ids(
            manager.project_registry, project.project_id
        ) == sorted(valid_ids)
        all_directories_quarantined = all(
            not (pending_dir / name).exists() for name in directory_names
        )
        if all_published and all_directories_quarantined:
            break

    assert _child_link_ids(manager.project_registry, project.project_id) == sorted(valid_ids)
    quarantine_dir = pending_dir.parent / "pending_child_links.invalid"
    for name in directory_names:
        assert not (pending_dir / name).exists()
        assert (_quarantined_path(quarantine_dir, name) / "marker").read_text(
            encoding="utf-8"
        ) == name


def test_hardlinked_entry_is_quarantined(tmp_path: Path) -> None:
    from zeta.core import session_links
    from zeta.core.session_links import (
        PENDING_CHILD_LINKS_DIRNAME,
        reconcile_child_links,
    )

    manager, project, root_id = _reconcile_fixture(tmp_path)
    child_id = f"{root_id}:hardlinked"
    _write_child_intent(
        manager.sessions_dir, root_id, project.project_id, child_id, root_id
    )
    pending_dir = manager.sessions_dir / root_id / PENDING_CHILD_LINKS_DIRNAME
    name = session_links._encode_link_name(child_id)
    original_content = (pending_dir / name).read_bytes()
    other_link = tmp_path / "other-hardlink"
    os.link(pending_dir / name, other_link)

    reconcile_child_links(manager.project_registry, manager.sessions_dir, root_id)

    quarantine = _quarantined_path(
        pending_dir.parent / "pending_child_links.invalid", name
    )
    assert _child_link_ids(manager.project_registry, project.project_id) == []
    assert not (pending_dir / name).exists()
    assert quarantine.read_bytes() == original_content
    assert other_link.read_bytes() == original_content
    assert os.path.samestat(quarantine.stat(), other_link.stat())


def test_temp_files_count_toward_pass_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from zeta.core import session_links
    from zeta.core.session_links import (
        PENDING_CHILD_LINKS_DIRNAME,
        reconcile_child_links,
    )

    manager, project, root_id = _reconcile_fixture(tmp_path)
    # Create the durable index directory, then leave only in-flight temp files.
    _write_child_intent(
        manager.sessions_dir, root_id, project.project_id, f"{root_id}:seed", root_id
    )
    pending_dir = manager.sessions_dir / root_id / PENDING_CHILD_LINKS_DIRNAME
    for name in list(os.listdir(pending_dir)):
        os.unlink(pending_dir / name)
    for i in range(5):
        (pending_dir / f".seed{i}.{i:032x}.tmp").write_text("in-flight", encoding="utf-8")

    monkeypatch.setattr(session_links, "_CHILD_LINK_MAX_ENTRIES_PER_PASS", 2)
    with caplog.at_level(logging.WARNING, logger="zeta.core.session_links"):
        reconcile_child_links(manager.project_registry, manager.sessions_dir, root_id)

    # Dot-prefixed temp files count toward the per-pass budget, so a bound of 2
    # is exhausted by the temp files alone and the pass-limit warning fires.
    assert "pass limit" in caplog.text


def test_only_writer_temp_pattern_is_treated_as_temp(tmp_path: Path) -> None:
    from zeta.core.session_links import (
        PENDING_CHILD_LINKS_DIRNAME,
        reconcile_child_links,
    )

    manager, project, root_id = _reconcile_fixture(tmp_path)
    _write_child_intent(
        manager.sessions_dir, root_id, project.project_id, f"{root_id}:1", root_id
    )
    pending_dir = manager.sessions_dir / root_id / PENDING_CHILD_LINKS_DIRNAME

    stale = pending_dir / (".stale." + "a" * 32 + ".tmp")
    stale.write_text("half-written", encoding="utf-8")
    fresh = pending_dir / (".fresh." + "b" * 32 + ".tmp")
    fresh.write_text("in-flight", encoding="utf-8")
    nonmatching = pending_dir / ".ordinary.tmp"
    nonmatching.write_text("not json", encoding="utf-8")
    old = time.time() - 7200  # ~2 hours ago
    os.utime(stale, (old, old))
    os.utime(nonmatching, (old, old))

    reconcile_child_links(manager.project_registry, manager.sessions_dir, root_id)

    # Only the writer's exact stale-temp pattern is reclaimed. A fresh writer
    # temp stays put, while another dotfile is processed and quarantined intact.
    assert not stale.exists()
    assert fresh.read_text(encoding="utf-8") == "in-flight"
    quarantine = pending_dir.parent / "pending_child_links.invalid"
    assert not nonmatching.exists()
    assert _quarantined_path(quarantine, nonmatching.name).read_text(
        encoding="utf-8"
    ) == "not json"
    # The valid intent was still published.
    assert _child_link_ids(manager.project_registry, project.project_id) == [
        f"{root_id}:1"
    ]


def test_quarantine_never_clobbers_existing_entry(tmp_path: Path) -> None:
    from zeta.core.session_links import (
        PENDING_CHILD_LINKS_DIRNAME,
        reconcile_child_links,
    )

    manager, project, root_id = _reconcile_fixture(tmp_path)
    _write_child_intent(
        manager.sessions_dir, root_id, project.project_id, f"{root_id}:seed", root_id
    )
    pending_dir = manager.sessions_dir / root_id / PENDING_CHILD_LINKS_DIRNAME
    for entry in pending_dir.iterdir():
        entry.unlink()
    invalid_name = "same-name"
    (pending_dir / invalid_name).write_text("new invalid payload", encoding="utf-8")
    quarantine_dir = pending_dir.parent / "pending_child_links.invalid"
    quarantine_dir.mkdir()
    (quarantine_dir / invalid_name).write_text("existing payload", encoding="utf-8")

    reconcile_child_links(manager.project_registry, manager.sessions_dir, root_id)

    assert (quarantine_dir / invalid_name).read_text(encoding="utf-8") == "existing payload"
    assert _quarantined_path(quarantine_dir, invalid_name).read_text(
        encoding="utf-8"
    ) == "new invalid payload"


def test_quarantine_is_unbounded_and_never_deletes(tmp_path: Path) -> None:
    from zeta.core.session_links import (
        PENDING_CHILD_LINKS_DIRNAME,
        reconcile_child_links,
    )

    manager, project, root_id = _reconcile_fixture(tmp_path)
    _write_child_intent(
        manager.sessions_dir, root_id, project.project_id, f"{root_id}:seed", root_id
    )
    pending_dir = manager.sessions_dir / root_id / PENDING_CHILD_LINKS_DIRNAME
    for entry in pending_dir.iterdir():
        entry.unlink()
    invalid_name = "invalid-after-large-quarantine"
    (pending_dir / invalid_name).write_text("not json", encoding="utf-8")
    quarantine_dir = pending_dir.parent / "pending_child_links.invalid"
    quarantine_dir.mkdir()
    for i in range(256):
        (quarantine_dir / f"existing-{i:03d}").write_text(str(i), encoding="utf-8")

    reconcile_child_links(manager.project_registry, manager.sessions_dir, root_id)

    assert not (pending_dir / invalid_name).exists()
    assert _quarantined_path(quarantine_dir, invalid_name).read_text(
        encoding="utf-8"
    ) == "not json"
    assert len(list(quarantine_dir.iterdir())) == 257
    for i in range(256):
        assert (quarantine_dir / f"existing-{i:03d}").read_text(encoding="utf-8") == str(i)


def test_invalid_entries_never_starve_valid_intents_across_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.core import session_links
    from zeta.core.session_links import (
        PENDING_CHILD_LINKS_DIRNAME,
        reconcile_child_links,
    )

    manager, project, root_id = _reconcile_fixture(tmp_path)
    valid_id = f"{root_id}:valid"
    _write_child_intent(
        manager.sessions_dir, root_id, project.project_id, valid_id, root_id
    )
    pending_dir = manager.sessions_dir / root_id / PENDING_CHILD_LINKS_DIRNAME
    valid_name = session_links._encode_link_name(valid_id)
    for i in range(2 * session_links._CHILD_LINK_MAX_ENTRIES_PER_PASS):
        (pending_dir / f"invalid-{i:03d}").write_text("not json", encoding="utf-8")

    def invalids_first(pending_fd: int) -> tuple[list[str], list[str], bool]:
        names = sorted(os.listdir(pending_fd), key=lambda name: (name == valid_name, name))
        return names[: session_links._CHILD_LINK_MAX_ENTRIES_PER_PASS], [], False

    monkeypatch.setattr(session_links, "_scan_pending_entries", invalids_first)

    for _ in range(3):
        reconcile_child_links(manager.project_registry, manager.sessions_dir, root_id)

    assert _child_link_ids(manager.project_registry, project.project_id) == [valid_id]
    assert not (pending_dir / valid_name).exists()
    assert list(pending_dir.iterdir()) == []


def test_reconciliation_reads_pending_intent_until_eof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.core import session_links
    from zeta.core.session_links import (
        PENDING_CHILD_LINKS_DIRNAME,
        reconcile_child_links,
    )

    manager, project, root_id = _reconcile_fixture(tmp_path)
    child_id = f"{root_id}:short-read"
    _write_child_intent(manager.sessions_dir, root_id, project.project_id, child_id, root_id)
    pending_path = (
        manager.sessions_dir / root_id / PENDING_CHILD_LINKS_DIRNAME
    )
    intent_path = next(pending_path.iterdir())
    intent_stat = intent_path.stat()
    real_read = os.read

    def dribbling_read(fd: int, count: int) -> bytes:
        try:
            if os.fstat(fd).st_ino == intent_stat.st_ino:
                return real_read(fd, min(count, 2))
        except OSError:
            pass
        return real_read(fd, count)

    monkeypatch.setattr(session_links.os, "read", dribbling_read)
    reconcile_child_links(manager.project_registry, manager.sessions_dir, root_id)

    assert _child_link_ids(manager.project_registry, project.project_id) == [child_id]


def test_registry_reads_legacy_record_with_lanes_field(tmp_path: Path) -> None:
    registry = ProjectRegistry(tmp_path / "projects")
    project = registry.create_project("demo", "scope")
    # Earlier builds persisted a ``lanes`` field in project.json.  New code must
    # still read those records, ignoring the lane data entirely.
    path = registry.root / project.project_id / "project.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["lanes"] = [
        {
            "lane_id": "l_" + "0" * 32,
            "project_id": project.project_id,
            "name": "legacy",
            "scope": "legacy scope",
            "created_at": data["created_at"],
            "updated_at": data["updated_at"],
        }
    ]
    path.write_text(json.dumps(data), encoding="utf-8")
    os.chmod(path, 0o600)

    reread = registry.show_project(project.project_id)
    assert reread.project_id == project.project_id
    assert reread.name == "demo"
    # The lane machinery is gone: the field is ignored, not surfaced.
    assert not hasattr(reread, "lanes")


@pytest.mark.asyncio
async def test_child_with_explicit_cwd_records_lineage_to_root_project(
    tmp_path: Path,
) -> None:
    home = tmp_path / ".zeta"
    root_repository = tmp_path / "root-repo"
    child_repository = tmp_path / "child-worktree"
    root_repository.mkdir()
    child_repository.mkdir()
    registry = ProjectRegistry(home / "projects")
    root_project = registry.create_project("root", "root scope", root_repository)
    child_project = registry.create_project("child", "child scope", child_repository)

    manager = SessionManager(home)
    opened = manager.create(provider="codex", model="fake", cwd=root_repository)
    root_id = opened.metadata.session_id
    turns = [
        ScriptedTurn(
            tool_calls=[
                ToolCall(
                    "child-call",
                    "agent",
                    {
                        "prompt": "child work",
                        "description": "work in another project",
                        "cwd": str(child_repository),
                    },
                )
            ]
        ),
        ScriptedTurn([TextContent("child done")]),
    ]
    loop = AgentLoop(
        FakeBackend(turns),
        opened.store,
        max_turns=1,
        skill_catalog=SkillCatalog.empty(),
        root_project_id=opened.metadata.project_id,
        project_registry=manager.project_registry,
    )
    try:
        async for _ in loop.run_turn("start", origin=MessageOrigin.USER):
            pass
    finally:
        await loop.close()
    opened.store.close()

    root_links = manager.project_registry.list_session_links(root_project.project_id)
    assert {record["session_id"] for record in root_links} == {
        root_id,
        f"{root_id}:1",
    }
    assert root_links[1]["parent_session_id"] == root_id
    assert manager.project_registry.list_session_links(child_project.project_id) == []


def test_memory_state_retries_if_first_writer_creates_registry_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "projects"
    reader = ProjectRegistry(root)
    project = reader.create_project("demo", "scope")
    reader.update_memory(project.project_id, {"brief.md": "old\n"})
    old = reader.memory_state(project.project_id)
    (root / ".lock").unlink()

    writer = ProjectRegistry(root)
    original_snapshot = reader._snapshot_locked
    wrote = False

    def snapshot_then_write(directory_fd: int):
        nonlocal wrote
        snapshot = original_snapshot(directory_fd)
        if not wrote:
            wrote = True
            writer.compare_and_swap_memory(
                project.project_id,
                expected_digest=old.digest,
                updates={"brief.md": "new\n"},
                provenance={
                    "session_id": "a" * 32,
                    "seq_start": 1,
                    "seq_end": 2,
                },
            )
        return snapshot

    monkeypatch.setattr(reader, "_snapshot_locked", snapshot_then_write)

    state = reader.memory_state(project.project_id)

    assert state.contents["brief.md"] == "new\n"
    assert state.digest == reader.memory_digest(project.project_id)
    assert state.version is not None and state.version != old.version
    assert state.automatic_files == ("brief.md",)
