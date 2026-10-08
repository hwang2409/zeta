from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from zeta.cli.user_action import confirm_memory_accept
from zeta.memory.entry_store import AddOperation, MemoryKind, MemorySchema, MemorySource
from zeta.project_errors import ProjectRegistryError
from zeta.project_memory_commands import run_memory_command
from zeta.project_memory_history import _MEMORY_MIRROR_HEADER
from zeta.project_registry import ProjectRegistry
from zeta.server.project_requests import ProjectRequests
from zeta.server.protocol import FrameCodec
from zeta.tools.project import _inspect_project


def _schema() -> MemorySchema:
    return MemorySchema(
        version=3,
        profile="zeta",
        kinds=(
            MemoryKind("state", "State", "Current project state.", "always", 1, 20),
            MemoryKind("decisions", "Decisions", "Accepted decisions.", "always", 2, 20),
        ),
    )


def _source(seq: int = 1) -> tuple[MemorySource, ...]:
    return (
        MemorySource(
            "a" * 32,
            seq,
            seq,
            ("user",),
            f"2026-10-0{seq}T00:00:00.000000Z",
            6,
        ),
    )


def _fixture(tmp_path: Path) -> tuple[Path, ProjectRegistry, str, str, str]:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("demo", "scope", workspace)
    registry.initialize_memory(project.project_id)
    initial = registry._create_entry_memory_for_test(project.project_id, _schema())
    added = registry._compare_and_swap_entries(
        project.project_id,
        expected_digest=initial.digest,
        operations=(
            AddOperation("state", "Automatic state", _source()),
            AddOperation("decisions", "Accepted decision", _source(2)),
        ),
        reconciliation_key="b" * 64,
    )
    state_id, decision_id = added.state.entries
    registry._accept_memory_entry(project.project_id, decision_id)
    return home, registry, project.project_id, state_id, decision_id


class _SizedCodec(FrameCodec):
    def __init__(self, maximum: int) -> None:
        self.maximum = maximum

    def response_fits(self, request_id: str | int | None, result: object) -> bool:
        return len(
            json.dumps(
                {"jsonrpc": "2.0", "id": request_id, "result": result},
                separators=(",", ":"),
            ).encode()
        ) <= self.maximum


def _requests(home: Path, *, frame_size: int = 16 * 1024 * 1024) -> ProjectRequests:
    runtime = SimpleNamespace(manager=SimpleNamespace(list_sessions_read_only=list))
    return ProjectRequests(home=home, runtime=runtime, codec=_SizedCodec(frame_size))


def test_project_show_v2_returns_kind_views_and_entry_counts(tmp_path: Path) -> None:
    home, _, project_id, _, _ = _fixture(tmp_path)
    result = _requests(home).dispatch(
        1,
        "project_show",
        {"project_id": project_id},
        features=frozenset({"projects-memory-v2"}),
    )
    memory = result["memory"]
    assert memory["schema_version"] == 3
    assert memory["profile"] == "zeta"
    assert [(item["key"], item["active_count"]) for item in memory["kinds"]] == [
        ("state", 1),
        ("decisions", 1),
    ]
    assert memory["kinds"][0]["automatic_count"] == 1
    assert memory["kinds"][1]["accepted_count"] == 1
    assert "Automatic state" in memory["kinds"][0]["rendered"]


@pytest.mark.asyncio
async def test_project_inspect_reads_fixture_format_two_state(tmp_path: Path) -> None:
    _, registry, project_id, state_id, _ = _fixture(tmp_path)
    project = registry.show_project(project_id)
    tool_registry = SimpleNamespace(
        project_registry=registry,
        project_id=project_id,
        cwd=Path(project.canonical_integration_root),
    )
    result = await _inspect_project(tool_registry, {"action": "inspect"})
    assert result["isError"] is False
    memory = result["structuredContent"]["memory"]
    assert memory["format"] == 2
    assert memory["entries"][0]["id"] == state_id
    assert memory["entries"][0]["text"] == "Automatic state"


def test_entry_commands_parse_targets_and_render_receipts(tmp_path: Path) -> None:
    _, registry, project_id, state_id, _ = _fixture(tmp_path)
    log = run_memory_command(registry, project_id, f"log {state_id}")
    assert state_id in log
    assert "add" in log
    assert run_memory_command(registry, project_id, f"accept {state_id}") == (
        f"memory accepted: {state_id}"
    )
    assert registry._entry_memory_state(project_id).state.entries[state_id].accepted_by == "user"  # type: ignore[union-attr]
    assert run_memory_command(registry, project_id, f"undo {state_id}") == (
        f"memory undo complete: {state_id}"
    )


class _TTY(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_entry_accept_confirmation_targets_entry_id(tmp_path: Path) -> None:
    _, registry, project_id, state_id, _ = _fixture(tmp_path)
    output = _TTY()
    confirm_memory_accept(
        registry,
        project_id,
        state_id,
        stdin=_TTY(state_id + "\n"),
        stdout=output,
    )
    assert f"Automatic memory entry: {state_id}" in output.getvalue()
    assert "Automatic state" in output.getvalue()


def test_entry_accept_confirmation_cancellation_does_not_mutate(tmp_path: Path) -> None:
    _, registry, project_id, state_id, _ = _fixture(tmp_path)
    before = registry._entry_memory_state(project_id).digest
    with pytest.raises(ProjectRegistryError, match="entry was not changed"):
        confirm_memory_accept(
            registry,
            project_id,
            state_id,
            stdin=_TTY("no\n"),
            stdout=_TTY(),
        )
    assert registry._entry_memory_state(project_id).digest == before


def test_entry_commands_reject_missing_or_multiple_targets(tmp_path: Path) -> None:
    _, registry, project_id, state_id, _ = _fixture(tmp_path)
    assert "entry-id" in run_memory_command(registry, project_id, "accept")
    assert "entry-id" in run_memory_command(
        registry, project_id, f"accept {state_id} extra"
    )
    assert "entry-id|version-id" in run_memory_command(
        registry, project_id, f"undo {state_id} extra"
    )


def test_entry_mirrors_are_repaired_and_edits_never_change_state(tmp_path: Path) -> None:
    home, registry, project_id, state_id, _ = _fixture(tmp_path)
    mirror = home / "projects" / project_id / "memory" / "state.md"
    assert mirror.read_text().startswith(_MEMORY_MIRROR_HEADER + "# State\n")
    assert f"`{state_id}`" in mirror.read_text()
    mirror.chmod(0o600)
    mirror.write_text(_MEMORY_MIRROR_HEADER + "user edit\n")
    before = registry._entry_memory_state(project_id)
    registry.repair_entry_memory_mirror(project_id)
    after = registry._entry_memory_state(project_id)
    assert "user edit" not in mirror.read_text()
    assert after.digest == before.digest


def test_project_show_v2_truncates_rendered_kind_to_frame(tmp_path: Path) -> None:
    home, registry, project_id, _, _ = _fixture(tmp_path)
    current = registry._entry_memory_state(project_id)
    registry._compare_and_swap_entries(
        project_id,
        expected_digest=current.digest,
        operations=(AddOperation("state", "x" * 3500, _source(3)),),
        reconciliation_key="c" * 64,
    )
    result = _requests(home, frame_size=2500).dispatch(
        "frame",
        "project_show",
        {"project_id": project_id},
        features=frozenset({"projects-memory-v2"}),
    )
    assert any(item["rendered_truncated"] for item in result["memory"]["kinds"])


def test_project_memory_log_v2_is_bounded_and_filters_entry(tmp_path: Path) -> None:
    home, _, project_id, state_id, decision_id = _fixture(tmp_path)
    result = _requests(home, frame_size=3500).dispatch(
        1,
        "project_memory_log",
        {"project_id": project_id, "entry_id": state_id},
        features=frozenset({"projects-memory-v2"}),
    )
    assert result["versions"]
    encoded = json.dumps(result).encode()
    assert len(encoded) < 3500
    assert state_id in json.dumps(result)
    assert decision_id not in json.dumps(result)


def test_format_one_command_mirror_and_serve_views_remain_golden(tmp_path: Path) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("demo", "scope", workspace)
    registry.initialize_memory(project.project_id)
    assert run_memory_command(registry, project.project_id, "log") == "memory log: empty"
    assert (
        home / "projects" / project.project_id / "memory" / "brief.md"
    ).read_text() == "# Brief\n"
    result = _requests(home).dispatch(
        1, "project_show", {"project_id": project.project_id}, features=frozenset({"projects"})
    )
    assert result["memory"] == {
        "version_id": None,
        "digest": registry.memory_digest(project.project_id),
        "files": [
            {
                "name": name,
                "content": content,
                "automatic": False,
                "content_truncated": False,
            }
            for name, content in registry.load_memory(project.project_id)
        ],
    }


def test_project_response_versions_are_explicitly_negotiated(tmp_path: Path) -> None:
    home, _, project_id, _, _ = _fixture(tmp_path)
    requests = _requests(home)
    with pytest.raises(ProjectRegistryError, match="format 2"):
        requests.dispatch(
            1,
            "project_show",
            {"project_id": project_id},
            features=frozenset({"projects"}),
        )
    result = requests.dispatch(
        2,
        "project_show",
        {"project_id": project_id},
        features=frozenset({"projects-memory-v2"}),
    )
    assert "kinds" in result["memory"]
    assert "files" not in result["memory"]
