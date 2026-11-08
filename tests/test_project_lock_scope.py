from __future__ import annotations

import fcntl
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from zeta.project_registry import ProjectRegistry
from zeta.storage_format import WRITER_EPOCH, advance


def _send(channel: socket.socket, value: str) -> None:
    channel.sendall(value.encode("ascii") + b"\n")


def _receive(channel: socket.socket) -> str:
    channel.settimeout(10)
    return channel.recv(64).decode("ascii").strip()


def _hold_memory_publication(home: str, project_id: str, channel: socket.socket) -> None:
    registry = ProjectRegistry(Path(home) / "projects")
    original = registry._acquire_lock

    def gated(fd: int, mode: int) -> None:
        original(fd, mode)
        if mode == fcntl.LOCK_EX:
            _send(channel, "acquired")
            if _receive(channel) != "release":
                raise AssertionError("release signal was not sent")

    registry._acquire_lock = gated  # type: ignore[method-assign]
    state = registry.memory_state(project_id)
    registry.compare_and_swap_memory(
        project_id,
        expected_digest=state.digest,
        updates={"brief.md": "project a\n"},
        provenance={"session_id": "a" * 32, "seq_start": 1, "seq_end": 1},
    )
    _send(channel, "a-done")


def _publish_other_project(home: str, project_id: str, channel: socket.socket) -> None:
    registry = ProjectRegistry(Path(home) / "projects")
    state = registry.memory_state(project_id)
    registry.compare_and_swap_memory(
        project_id,
        expected_digest=state.digest,
        updates={"brief.md": "project b\n"},
        provenance={"session_id": "b" * 32, "seq_start": 1, "seq_end": 1},
    )
    _send(channel, "b-done")


def _record_session(home: str, project_id: str, channel: socket.socket) -> None:
    ProjectRegistry(Path(home) / "projects").record_session(
        project_id, session_id="session", transcript_path="session"
    )
    _send(channel, "session-done")


def _hold_identity_publication(home: str, _project_id: str, channel: socket.socket) -> None:
    registry = ProjectRegistry(Path(home) / "projects")
    original = registry._acquire_lock

    def gated(fd: int, mode: int) -> None:
        original(fd, mode)
        if mode == fcntl.LOCK_EX:
            _send(channel, "acquired")
            if _receive(channel) != "release":
                raise AssertionError("release signal was not sent")

    registry._acquire_lock = gated  # type: ignore[method-assign]
    registry.create_project("new", "new")
    _send(channel, "identity-done")


_WORKERS = {
    "hold-memory": _hold_memory_publication,
    "hold-identity": _hold_identity_publication,
    "publish-memory": _publish_other_project,
    "record-session": _record_session,
}


def _start_worker(kind: str, home: Path, project_id: str) -> tuple[subprocess.Popen[bytes], socket.socket]:
    parent, child = socket.socketpair()
    process = subprocess.Popen(
        [sys.executable, __file__, kind, str(home), project_id, str(child.fileno())],
        pass_fds=(child.fileno(),),
    )
    child.close()
    return process, parent


def _run_while_lock_is_held(
    home: Path,
    *,
    holder_kind: str,
    holder_project: str,
    other_kind: str,
    other_project: str,
    expected: str,
) -> None:
    holder, holder_channel = _start_worker(holder_kind, home, holder_project)
    other: subprocess.Popen[bytes] | None = None
    other_channel: socket.socket | None = None
    try:
        assert _receive(holder_channel) == "acquired", (
            "held mutation did not acquire its lock"
        )
        other, other_channel = _start_worker(other_kind, home, other_project)
        assert _receive(other_channel) == expected, (
            "unrelated public mutation remained serialized"
        )
        _send(holder_channel, "release")
        assert _receive(holder_channel).endswith("done")
        assert holder.wait(timeout=10) == 0
        assert other.wait(timeout=10) == 0
    finally:
        for process in (holder, other):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=10)
        holder_channel.close()
        if other_channel is not None:
            other_channel.close()


if __name__ == "__main__":
    worker = _WORKERS[sys.argv[1]]
    channel = socket.socket(fileno=int(sys.argv[4]))
    try:
        worker(sys.argv[2], sys.argv[3], channel)
    finally:
        channel.close()

def test_p06_1_unrelated_projects_publish_without_root_lock_serialization(
    tmp_path: Path,
) -> None:
    home = tmp_path / ".zeta"
    advance(home, expected_epoch=0)
    registry = ProjectRegistry(home / "projects")
    project_a = registry.create_project("a", "a")
    project_b = registry.create_project("b", "b")
    registry.initialize_memory(project_a.project_id)
    registry.initialize_memory(project_b.project_id)

    _run_while_lock_is_held(
        home,
        holder_kind="hold-memory",
        holder_project=project_a.project_id,
        other_kind="publish-memory",
        other_project=project_b.project_id,
        expected="b-done",
    )


def test_p06_2_memory_and_sessions_domains_do_not_serialize(tmp_path: Path) -> None:
    home = tmp_path / ".zeta"
    advance(home, expected_epoch=0)
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("demo", "demo")
    registry.initialize_memory(project.project_id)

    _run_while_lock_is_held(
        home,
        holder_kind="hold-memory",
        holder_project=project.project_id,
        other_kind="record-session",
        other_project=project.project_id,
        expected="session-done",
    )


def test_p06_3_identity_and_domain_publications_do_not_serialize(
    tmp_path: Path,
) -> None:
    home = tmp_path / ".zeta"
    advance(home, expected_epoch=0)
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("demo", "demo")
    registry.initialize_memory(project.project_id)

    _run_while_lock_is_held(
        home,
        holder_kind="hold-identity",
        holder_project=project.project_id,
        other_kind="publish-memory",
        other_project=project.project_id,
        expected="b-done",
    )

def test_p06_4_epoch_selects_legacy_then_named_authority(tmp_path: Path) -> None:
    home = tmp_path / ".zeta"
    for epoch in range(WRITER_EPOCH - 1):
        advance(home, expected_epoch=epoch, target_epoch=epoch + 1)
    legacy = ProjectRegistry(home / "projects")
    old_project = legacy.create_project("old", "old")
    legacy.initialize_memory(old_project.project_id)
    assert (home / "projects" / ".lock").is_file()
    assert not (home / "projects" / old_project.project_id / ".memory.lock").exists()

    advance(home, expected_epoch=WRITER_EPOCH - 1, target_epoch=WRITER_EPOCH)
    current = ProjectRegistry(home / "projects")
    current.load_memory(old_project.project_id)
    assert (home / "projects" / old_project.project_id / ".memory.lock").is_file()


def test_p06_5_session_reference_is_reassociated_by_exact_upsert(
    tmp_path: Path,
) -> None:
    home = tmp_path / ".zeta"
    advance(home, expected_epoch=0)
    registry = ProjectRegistry(home / "projects")
    old_project = registry.create_project("old", "old")
    project = registry.create_project("demo", "demo")
    registry.record_session(
        old_project.project_id, session_id="session", transcript_path="old"
    )
    registry.record_session(
        project.project_id,
        session_id="session",
        transcript_path="new",
        role="worker",
        parent_session_id="parent",
    )

    links = registry.list_session_links(project.project_id)
    assert registry.list_session_links(old_project.project_id) == [], (
        "reassociation left the session in its previous project"
    )
    assert len(links) == 1
    links_path = home / "projects" / project.project_id / "sessions.jsonl"
    before = links_path.read_bytes(), links_path.stat().st_mtime_ns
    registry.record_session(
        project.project_id,
        session_id="session",
        transcript_path="new",
        role="worker",
        parent_session_id="parent",
    )
    assert (links_path.read_bytes(), links_path.stat().st_mtime_ns) == before
    assert {
        key: value for key, value in links[0].items() if key != "recorded_at"
    } == {
        "session_id": "session",
        "transcript_path": "new",
        "role": "worker",
        "parent_session_id": "parent",
    }


def test_p06_8_rank_guard_rejects_inversions_and_unsorted_domains() -> None:
    try:
        from zeta.locking import (
            domain_lock,
            execution_lock,
            format_lock,
            identity_lock,
            shape_lock,
        )
    except ModuleNotFoundError:
        pytest.skip("rank guard does not exist before p06")

    with (
        shape_lock(),
        pytest.raises(RuntimeError, match="order inversion"),
        identity_lock(),
    ):
        pass
    with execution_lock(), domain_lock("p_a", "memory"):
        pass
    with (
        shape_lock(),
        pytest.raises(RuntimeError, match="order inversion"),
        domain_lock("p_a", "memory"),
    ):
        pass
    with (
        domain_lock("p_a", "memory"),
        pytest.raises(RuntimeError, match="order inversion"),
        format_lock(),
    ):
        pass
    with (
        domain_lock("p_a", "sessions"),
        pytest.raises(RuntimeError, match="sorted order"),
        domain_lock("p_a", "memory"),
    ):
        pass
