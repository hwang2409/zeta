from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from zeta.cli import project as project_cli
from zeta.core.project_context import refresh_project_memory
from zeta.memory.auto import AutoMemoryConfig, AutoMemoryReconciler
from zeta.memory.entry_store import MemoryKind, MemorySchema
from zeta.memory.user_authorization import MemoryMutationAuthorization
from zeta.memory.version_store import UNSUPPORTED_FORMAT_2
from zeta.project_errors import UnsupportedMemoryFormatError
from zeta.project_memory_commands import run_memory_command
from zeta.project_memory_history import MemoryExport
from zeta.project_registry import ProjectRegistry
from zeta.remote_sync import LocalTransport, push_project_memory
from zeta.server.project_requests import ProjectRequests, project_request_error
from zeta.server.protocol import FrameCodec

SESSION_ID = "a" * 32


def _schema() -> MemorySchema:
    return MemorySchema(
        version=1,
        profile="test",
        kinds=(MemoryKind("state", "State", "State.", "always", 1, 1),),
    )


def _format_two_project(tmp_path: Path) -> tuple[Path, ProjectRegistry, str, Path]:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("test", "test", workspace)
    registry.initialize_memory(project.project_id)
    registry._create_entry_memory_for_test(project.project_id, _schema())
    return home, registry, project.project_id, workspace


def _snapshot(root: Path) -> dict[str, tuple[int, bytes]]:
    return {
        str(path.relative_to(root)): (
            path.stat(follow_symlinks=False).st_mode,
            path.read_bytes() if path.is_file() else b"",
        )
        for path in sorted(root.rglob("*"))
    }


def test_all_format_one_registry_paths_fail_closed_without_writes(tmp_path: Path) -> None:
    home, registry, project_id, _ = _format_two_project(tmp_path)
    project_root = home / "projects" / project_id
    exported_contents = {
        name: ""
        for name in ("brief.md", "state.md", "backlog.md", "changelog.md", "decisions.md")
    }
    exported = MemoryExport(
        contents=exported_contents,
        digest=registry._memory_digest_value(exported_contents),
        versions=(),
        automatic_files=(),
    )

    def refresh_mirror() -> None:
        with registry._locked(write=True) as root_fd:
            directory_fd = registry._project_dir(root_fd, project_id)
            try:
                registry._refresh_memory_mirror(
                    directory_fd, {}, mirror_path=project_root / "memory"
                )
            finally:
                import os

                os.close(directory_fd)

    calls = {
        "snapshot": lambda: registry.memory_snapshot(project_id),
        "digest": lambda: registry.memory_digest(project_id),
        "load": lambda: registry.load_memory(project_id),
        "context": lambda: registry.load_memory_for_context(project_id),
        "state": lambda: registry.memory_state(project_id),
        "version file": lambda: registry.memory_version_file(
            project_id, "0" * 32, "brief.md"
        ),
        "history": lambda: registry.memory_log(project_id),
        "undo": lambda: registry.undo_memory(project_id),
        "accept": lambda: registry.accept_memory(project_id, "brief.md"),
        "update": lambda: registry.update_memory(project_id, {"brief.md": "changed"}),
        "compare-and-swap": lambda: registry.compare_and_swap_memory(
            project_id, expected_digest="0" * 64, updates={"brief.md": "changed"}
        ),
        "export": lambda: registry.export_memory(project_id),
        "import": lambda: registry.import_memory(
            project_id, exported, expected_digest="0" * 64
        ),
        "initialize": lambda: registry.initialize_memory(project_id),
        "mirror": refresh_mirror,
    }
    before = _snapshot(project_root)
    for name, call in calls.items():
        with pytest.raises(UnsupportedMemoryFormatError, match=UNSUPPORTED_FORMAT_2):
            call()
        assert _snapshot(project_root) == before, name


def _cli_args(verb: str, project_id: str, workspace: Path) -> argparse.Namespace:
    common = {
        "project_verb": verb,
        "project": project_id,
        "directory": str(workspace),
        "name": None,
        "scope": "",
        "remote": None,
        "action": None,
        "sync_project": None,
        "remote_home": None,
        "accept": None,
        "set": [],
        "from_file": [],
    }
    return argparse.Namespace(**common)


def test_format_two_views_are_available_after_per_project_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home, registry, project_id, workspace = _format_two_project(tmp_path)
    monkeypatch.setenv("ZETA_HOME", str(home))
    project_root = home / "projects" / project_id
    before = _snapshot(project_root)

    memory_out = io.StringIO()
    assert project_cli.run(
        _cli_args("memory", project_id, workspace),
        stdout=memory_out,
        stderr=io.StringIO(),
    ) == 0
    assert '"state.md"' in memory_out.getvalue()

    for verb in ("show", "init"):
        assert project_cli.run(
            _cli_args(verb, project_id, workspace),
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        ) == 0

    index_args = _cli_args("index", project_id, workspace)
    index_args.index_action = "status"
    assert project_cli.run(
        index_args, stdout=io.StringIO(), stderr=io.StringIO()
    ) == 0

    reconciler = SimpleNamespace(terminal_receipts=lambda: (), retry_terminal=lambda key: False)
    authorization = MemoryMutationAuthorization.direct_slash()
    assert run_memory_command(registry, project_id, "log", reconciler) == "memory log: empty"
    assert "no retained memory" in run_memory_command(
        registry, project_id, "undo", reconciler, authorization
    )
    assert "missing entry" in run_memory_command(
        registry, project_id, "accept m_" + "1" * 32, reconciler, authorization
    )
    assert "no terminal receipts" in run_memory_command(
        registry, project_id, "retry", reconciler
    )

    runtime = SimpleNamespace(
        manager=SimpleNamespace(list_sessions_read_only=list)
    )
    requests = ProjectRequests(home=home, runtime=runtime, codec=FrameCodec())
    for method in ("project_show", "project_memory_log"):
        with pytest.raises(UnsupportedMemoryFormatError, match=UNSUPPORTED_FORMAT_2):
            requests.dispatch(1, method, {"project_id": project_id})
        result = requests.dispatch(
            2,
            method,
            {"project_id": project_id},
            features=frozenset({"projects-memory-v2"}),
        )
        assert "memory" in result if method == "project_show" else "versions" in result
    code, message, _ = project_request_error(
        UnsupportedMemoryFormatError(UNSUPPORTED_FORMAT_2)
    )
    assert (code, message) == (-32000, UNSUPPORTED_FORMAT_2)

    remote = tmp_path / "remote"
    remote.mkdir()
    synced = push_project_memory(home, LocalTransport(remote), project_id=project_id)
    assert synced.conflicts == ()
    before = _snapshot(project_root)

    prompt = "prefix<zeta-project-memory></zeta-project-memory>suffix"
    block = "<zeta-project-memory></zeta-project-memory>"
    refreshed = refresh_project_memory(
        prompt,
        home=home,
        project_id=project_id,
        memory_offset=len("prefix"),
        memory_length=len(block),
        memory_digest=__import__("hashlib").sha256(block.encode()).hexdigest(),
    )
    assert refreshed.startswith("prefix<zeta-project-memory>")
    assert refreshed.endswith("</zeta-project-memory>suffix")

    assert _snapshot(project_root) == before


@pytest.mark.asyncio
async def test_private_format_two_fixture_reaches_dormant_updater(
    tmp_path: Path,
) -> None:
    home, registry, project_id, _ = _format_two_project(tmp_path)
    session_dir = home / "sessions" / SESSION_ID
    session_dir.mkdir(parents=True)
    (session_dir / "conversation.jsonl").write_text(
        json.dumps(
            {
                "seq": 1,
                "id": "m1",
                "parent_id": None,
                "type": "message",
                "data": {
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": "remember this"}],
                    }
                },
            }
        )
        + "\n"
    )
    invoked = False

    async def invoke(_prompt: str) -> str:
        nonlocal invoked
        invoked = True
        return '{"operations":[]}'

    notices: list[str] = []
    runner = AutoMemoryReconciler(
        registry=registry,
        project_id=project_id,
        session_id=SESSION_ID,
        session_dir=session_dir,
        invoke=invoke,
        config=AutoMemoryConfig(minimum_interval=0),
        notice=notices.append,
    )
    runner.before_eviction(1, 1)
    await runner.drain()
    await runner.close()
    assert invoked is True
    assert runner.last_failure is None
    assert runner.last_reconciled_seq == 1
    assert notices == []


def test_profile_cli_activates_only_the_created_project(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    project_cli.add_subcommand(commands)
    args = parser.parse_args(
        [
            "project",
            "create",
            "demo",
            "--scope",
            "scope",
            "--memory-profile",
            "messaging",
        ]
    )
    home = tmp_path / "home"
    with patch.dict("os.environ", {"ZETA_HOME": str(home)}):
        assert project_cli.run(args, stdout=io.StringIO(), stderr=io.StringIO()) == 0
    registry = ProjectRegistry(home / "projects")
    project = registry.show_project(name="demo")
    assert registry.memory_format(project.project_id) == 2
    assert registry._entry_memory_state(project.project_id).state.schema.profile == "messaging"


def test_format_one_manifest_and_pointer_remain_byte_identical(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ProjectRegistry(tmp_path / "projects")
    project = registry.create_project("test", "test", workspace)
    registry.initialize_memory(project.project_id)
    fixed_uuid = SimpleNamespace(hex="1" * 32)
    with (
        patch("zeta.memory.version_store.uuid.uuid4", return_value=fixed_uuid),
        patch(
            "zeta.project_memory_history._now",
            return_value="2026-01-01T00:00:00.000000Z",
        ),
    ):
        registry.update_memory(
            project.project_id, {"state.md": "# Current state\nchanged\n"}
        )

    project_root = registry.root / project.project_id
    assert (project_root / "memory-current.json").read_bytes() == (
        b'{"current": "11111111111111111111111111111111", '
        b'"history": ["11111111111111111111111111111111"]}'
    )
    manifest = project_root / "memory-versions" / "versions" / f"{'1' * 32}.json"
    assert manifest.read_text() == (
        '{"automatic_files": [], "before_automatic_files": [], "before_snapshot": '
        '{"backlog.md": "ed769bd406173a15c796f7a4a236b521ad24c64c3b9330039b3e4c13dca02112", '
        '"brief.md": "1a2036951819553b36c38faad3aa3eb4aa9421072cb9ea61ea67ab7ada105f10", '
        '"changelog.md": "3e79c4cafb504a21f8913e4e0e66f2ff7b1192a127c6f564aab379c8b5fa9bdd", '
        '"decisions.md": "2cce984feac8953790a34cb9a7176e7bb9b7dc08d81140ee7a42766571afb28d", '
        '"state.md": "042d53aa4099112008c7f9cfda005ca21ecad53ef217bdae498d39a912176d6a"}, '
        '"created_at": "2026-01-01T00:00:00.000000Z", "files": ["state.md"], '
        '"kind": "manual", "snapshot": {"backlog.md": '
        '"ed769bd406173a15c796f7a4a236b521ad24c64c3b9330039b3e4c13dca02112", '
        '"brief.md": "1a2036951819553b36c38faad3aa3eb4aa9421072cb9ea61ea67ab7ada105f10", '
        '"changelog.md": "3e79c4cafb504a21f8913e4e0e66f2ff7b1192a127c6f564aab379c8b5fa9bdd", '
        '"decisions.md": "2cce984feac8953790a34cb9a7176e7bb9b7dc08d81140ee7a42766571afb28d", '
        '"state.md": "12c2bd9d512d95436c8fea8880a8648febeb2f2f998972479ad766750190a4c5"}, '
        '"version": "11111111111111111111111111111111"}'
    )
