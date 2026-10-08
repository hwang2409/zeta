from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from zeta.cli import project as project_cli
from zeta.memory.entry_store import (
    AddOperation,
    EntryCASResult,
    MemoryKind,
    MemorySchema,
    MemorySource,
    SupersedeOperation,
    UpdateOperation,
)
from zeta.memory.user_authorization import MemoryMutationAuthorization
from zeta.project_memory_commands import run_memory_command
from zeta.project_memory_history import _MEMORY_MIRROR_HEADER
from zeta.project_registry import ProjectRegistry
from zeta.server.project_requests import ProjectRequests
from zeta.server.protocol import FrameCodec


class _TTY(io.StringIO):
    def isatty(self) -> bool:
        return True


class _SizedCodec(FrameCodec):
    def __init__(self, maximum: int) -> None:
        self.maximum = maximum

    def response_fits(self, request_id: str | int | None, result: object) -> bool:
        envelope = {"jsonrpc": "2.0", "id": request_id, "result": result}
        return len(json.dumps(envelope, separators=(",", ":")).encode()) <= self.maximum


def _schema() -> MemorySchema:
    return MemorySchema(
        version=3,
        profile="zeta",
        kinds=(MemoryKind("state", "State", "Current state.", "always", 1, 1000),),
    )


def _source(index: int = 1) -> tuple[MemorySource, ...]:
    return (
        MemorySource(
            f"session-{index}",
            index,
            index,
            ("user",),
            "2026-10-08T00:00:00.000000Z",
            6,
        ),
    )


def _fixture(tmp_path: Path) -> tuple[Path, ProjectRegistry, str]:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("demo", "scope", workspace)
    registry.initialize_memory(project.project_id)
    registry._create_entry_memory_for_test(project.project_id, _schema())
    return home, registry, project.project_id


def _cli_args(project_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        project_verb="memory",
        project=project_id,
        remote=None,
        action=None,
        sync_project=None,
        remote_home=None,
        accept=None,
        set=[],
        from_file=[],
        json=False,
    )


@pytest.mark.parametrize("mode", ["set", "from_file"])
@pytest.mark.parametrize("blocked_by", ["tool", "non_tty"])
def test_cli_replacement_requires_direct_user_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    blocked_by: str,
) -> None:
    home, registry, project_id = _fixture(tmp_path)
    monkeypatch.setenv("ZETA_HOME", str(home))
    if blocked_by == "tool":
        monkeypatch.setenv("ZETA_TOOL_SUBPROCESS", "1")
    path = tmp_path / "replacement.md"
    path.write_text("replacement")
    args = _cli_args(project_id)
    setattr(args, mode, [("state", "replacement" if mode == "set" else str(path))])
    before = registry._entry_memory_state(project_id).digest

    monkeypatch.setattr(
        project_cli.sys,
        "stdin",
        _TTY("state\n") if blocked_by == "tool" else io.StringIO("state\n"),
    )
    result = project_cli.run(args, stdout=_TTY(), stderr=io.StringIO())

    assert result == 1
    assert registry._entry_memory_state(project_id).digest == before


@pytest.mark.parametrize("mode", ["set", "from_file"])
@pytest.mark.parametrize("blocked_by", ["tool", "non_tty"])
def test_cli_format_one_replacement_requires_direct_user_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    blocked_by: str,
) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("demo", "scope", workspace)
    registry.initialize_memory(project.project_id)
    monkeypatch.setenv("ZETA_HOME", str(home))
    if blocked_by == "tool":
        monkeypatch.setenv("ZETA_TOOL_SUBPROCESS", "1")
    path = tmp_path / "replacement.md"
    path.write_text("replacement")
    args = _cli_args(project.project_id)
    setattr(
        args,
        mode,
        [("state.md", "replacement" if mode == "set" else str(path))],
    )
    before = registry.memory_snapshot(project.project_id).digest
    monkeypatch.setattr(
        project_cli.sys,
        "stdin",
        _TTY("state.md\n") if blocked_by == "tool" else io.StringIO("state.md\n"),
    )

    result = project_cli.run(args, stdout=_TTY(), stderr=io.StringIO())

    assert result == 1
    assert registry.memory_snapshot(project.project_id).digest == before


def test_slash_mutations_require_authorized_direct_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, registry, project_id = _fixture(tmp_path)
    current = registry._entry_memory_state(project_id)
    added = registry._compare_and_swap_entries(
        project_id,
        expected_digest=current.digest,
        operations=(AddOperation("state", "automatic", _source()),),
        reconciliation_key="a" * 64,
    )
    entry_id = next(iter(added.state.entries))
    monkeypatch.setenv("ZETA_TOOL_SUBPROCESS", "1")
    before = added.digest

    authorization = MemoryMutationAuthorization.direct_slash()
    assert "unavailable" in run_memory_command(
        registry, project_id, f"accept {entry_id}", authorization=authorization
    )
    assert "unavailable" in run_memory_command(
        registry, project_id, f"undo {entry_id}", authorization=authorization
    )
    assert registry._entry_memory_state(project_id).digest == before


def test_slash_mutations_reject_missing_direct_user_authorization(tmp_path: Path) -> None:
    _, registry, project_id = _fixture(tmp_path)
    current = registry._entry_memory_state(project_id)
    added = registry._compare_and_swap_entries(
        project_id,
        expected_digest=current.digest,
        operations=(AddOperation("state", "automatic", _source()),),
        reconciliation_key="b" * 64,
    )
    entry_id = next(iter(added.state.entries))

    assert "direct user authorization" in run_memory_command(
        registry, project_id, f"accept {entry_id}"
    )
    assert "direct user authorization" in run_memory_command(
        registry, project_id, f"undo {entry_id}"
    )
    assert registry._entry_memory_state(project_id).digest == added.digest


def test_slash_retry_requires_direct_user_authorization(tmp_path: Path) -> None:
    _, registry, project_id = _fixture(tmp_path)

    class Reconciler:
        queued = False

        def terminal_receipts(self) -> tuple[SimpleNamespace, ...]:
            return (SimpleNamespace(key="a" * 64),)

        def retry_terminal(self, _key: str) -> bool:
            self.queued = True
            return True

    reconciler = Reconciler()
    result = run_memory_command(registry, project_id, "retry a", reconciler)

    assert "direct user authorization" in result
    assert reconciler.queued is False


def test_undo_older_transaction_preserves_independent_later_change(tmp_path: Path) -> None:
    _, registry, project_id = _fixture(tmp_path)
    initial = registry._entry_memory_state(project_id)
    first = registry._compare_and_swap_entries(
        project_id,
        expected_digest=initial.digest,
        operations=(AddOperation("state", "first", _source(1)),),
        reconciliation_key="1" * 64,
    )
    first_id = next(iter(first.state.entries))
    later = registry._compare_and_swap_entries(
        project_id,
        expected_digest=first.digest,
        operations=(AddOperation("state", "later", _source(2)),),
        reconciliation_key="2" * 64,
    )
    later_id = next(entry_id for entry_id in later.state.entries if entry_id != first_id)

    undone = registry._undo_entry_transaction(project_id, first_id)

    assert first_id not in undone.state.entries
    assert later_id in undone.state.entries


def test_undo_older_transaction_rejects_dependent_later_change(tmp_path: Path) -> None:
    _, registry, project_id = _fixture(tmp_path)
    initial = registry._entry_memory_state(project_id)
    first = registry._compare_and_swap_entries(
        project_id,
        expected_digest=initial.digest,
        operations=(AddOperation("state", "first", _source(1)),),
        reconciliation_key="1" * 64,
    )
    first_id = next(iter(first.state.entries))
    later = registry._compare_and_swap_entries(
        project_id,
        expected_digest=first.digest,
        operations=(UpdateOperation(first_id, _source(2), text="changed"),),
        reconciliation_key="2" * 64,
    )

    with pytest.raises(Exception, match="dependent.*update"):
        registry._undo_entry_transaction(project_id, first.version)

    assert registry._entry_memory_state(project_id).digest == later.digest


def _superseded_entry(
    registry: ProjectRegistry, project_id: str
) -> tuple[EntryCASResult, EntryCASResult, str, str]:
    initial = registry._entry_memory_state(project_id)
    added = registry._compare_and_swap_entries(
        project_id,
        expected_digest=initial.digest,
        operations=(AddOperation("state", "old", _source(1)),),
        reconciliation_key="3" * 64,
    )
    old_id = next(iter(added.state.entries))
    superseded = registry._compare_and_swap_entries(
        project_id,
        expected_digest=added.digest,
        operations=(
            SupersedeOperation((old_id,), "state", "replacement", _source(2)),
        ),
        reconciliation_key="4" * 64,
    )
    replacement_id = next(
        entry_id for entry_id in superseded.state.entries if entry_id != old_id
    )
    return added, superseded, old_id, replacement_id


def test_undo_supersede_ignores_accept_that_was_undone(tmp_path: Path) -> None:
    _, registry, project_id = _fixture(tmp_path)
    added, superseded, _, replacement_id = _superseded_entry(registry, project_id)
    accepted = registry._accept_memory_entry(project_id, replacement_id)

    registry._undo_entry_transaction(project_id, accepted.version)
    restored = registry._undo_entry_transaction(project_id, superseded.version)

    assert restored.state.entries == added.state.entries


def test_undo_supersede_rejects_active_accept_dependency(tmp_path: Path) -> None:
    _, registry, project_id = _fixture(tmp_path)
    _, superseded, _, replacement_id = _superseded_entry(registry, project_id)
    accepted = registry._accept_memory_entry(project_id, replacement_id)

    with pytest.raises(Exception, match="dependent.*accept"):
        registry._undo_entry_transaction(project_id, superseded.version)

    assert registry._entry_memory_state(project_id).digest == accepted.digest


def test_undo_of_undo_chain_tracks_effective_dependency(tmp_path: Path) -> None:
    _, registry, project_id = _fixture(tmp_path)
    added, superseded, _, replacement_id = _superseded_entry(registry, project_id)
    accepted = registry._accept_memory_entry(project_id, replacement_id)
    accept_undo = registry._undo_entry_transaction(project_id, accepted.version)

    accept_restored = registry._undo_entry_transaction(project_id, accept_undo.version)
    with pytest.raises(Exception, match="dependent"):
        registry._undo_entry_transaction(project_id, superseded.version)
    registry._undo_entry_transaction(project_id, accept_restored.version)
    restored = registry._undo_entry_transaction(project_id, superseded.version)

    assert restored.state.entries == added.state.entries


def test_undo_latest_transaction_still_restores_before_snapshot(tmp_path: Path) -> None:
    _, registry, project_id = _fixture(tmp_path)
    initial = registry._entry_memory_state(project_id)
    added = registry._compare_and_swap_entries(
        project_id,
        expected_digest=initial.digest,
        operations=(AddOperation("state", "latest", _source()),),
        reconciliation_key="3" * 64,
    )

    undone = registry._undo_entry_transaction(project_id)

    assert undone.state.entries == initial.state.entries
    assert undone.state.generation == added.state.generation + 1


def test_large_version_detail_is_bounded_and_reports_paging(tmp_path: Path) -> None:
    home, registry, project_id = _fixture(tmp_path)
    initial = registry._entry_memory_state(project_id)
    sources = tuple(_source(index)[0] for index in range(1, 65))
    large = registry._compare_and_swap_entries(
        project_id,
        expected_digest=initial.digest,
        operations=tuple(AddOperation("state", f"entry {index}", sources) for index in range(128)),
        reconciliation_key="4" * 64,
    )
    runtime = SimpleNamespace(manager=SimpleNamespace(list_sessions_read_only=list))
    requests = ProjectRequests(
        home=home,
        runtime=runtime,
        codec=_SizedCodec(1024 * 1024),
    )

    result = requests.dispatch(
        1,
        "project_memory_log",
        {"project_id": project_id, "version_id": large.version},
        features=frozenset({"projects-memory-v2"}),
    )

    assert requests.codec.response_fits(1, result)
    detail = result["version"]
    assert detail["after_entries_count"] == 128
    assert detail["after_entries_next_offset"] is not None
    assert detail["after_entries"][0]["sources_count"] == 64
    assert detail["after_entries"][0]["sources_next_offset"] is not None


def test_format_two_mirror_is_repaired_on_production_render_path(tmp_path: Path) -> None:
    _, registry, project_id = _fixture(tmp_path)
    path = registry.root / project_id / "memory" / "state.md"
    path.chmod(0o600)
    path.write_text(_MEMORY_MIRROR_HEADER + "edited")

    rendered = registry._entry_memory_mirrors(project_id)

    assert rendered["state"] in path.read_text()
    assert "edited" not in path.read_text()
