from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pty
import secrets
import selectors
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.support.processes import read_pty_until, reap_process
from tests.support.storage_homes import (
    STORAGE_HOME_SHAPES,
    StorageHomeShape,
    prepared_storage_home,
)
from zeta.cli.main import main
from zeta.core.checkpoints import ConversationEntry, ConversationIntegrityError
from zeta.core.checkpoints.workspace import WorkspaceSnapshotStore
from zeta.core.session import SessionManager
from zeta.core.store import ConversationStore
from zeta.protocol.types import ToolCall
from zeta.runtime.loop.persistence import DraftPersistence
from zeta.storage_format import WRITER_EPOCH, StorageFormatError
from zeta.tools._shared.process import BackgroundTaskRegistry


def _inject(path: Path) -> None:
    value = json.loads(path.read_text())
    value["future"] = {"nested": {"value": 7}}
    path.write_text(json.dumps(value))


def _assert_future(path: Path) -> None:
    assert json.loads(path.read_text())["future"] == {"nested": {"value": 7}}


def _run_cli(home: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["ZETA_HOME"] = str(home)
    return subprocess.run(
        [str(Path(sys.executable).with_name("zeta")), *arguments],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _tree_digest(home: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(home.rglob("*")):
        digest.update(path.relative_to(home).as_posix().encode())
        digest.update(b"d" if path.is_dir() else b"f")
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _console_environment(home: Path) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("ZETA_TEST_")
    }
    environment.update(
        {
            "ZETA_HOME": str(home),
            "TERM": "xterm-256color",
            "COLUMNS": "100",
            "LINES": "30",
        }
    )
    assert not any(key.startswith("ZETA_TEST_") for key in environment)
    return environment


def _storage_error(home: Path, shape: StorageHomeShape) -> str | None:
    if shape == "empty":
        return None
    if shape == "too-new":
        return (
            f"stored format {WRITER_EPOCH + 1} is newer than writer capability "
            f"{WRITER_EPOCH}"
        )
    if shape == "invalid":
        return f"stored-format manifest is invalid at {home}"
    return (
        f"stored-format manifest is missing at {home}; run "
        "zeta storage-format advance --expected-epoch 0 "
        f"--target-epoch {WRITER_EPOCH}"
    )


def _read_stream_until(
    stream, marker: bytes, *, process: subprocess.Popen[bytes], timeout: float
) -> bytes:
    output = bytearray()
    deadline = time.monotonic() + timeout
    with selectors.DefaultSelector() as selector:
        selector.register(stream, selectors.EVENT_READ)
        while marker not in output:
            remaining = deadline - time.monotonic()
            assert remaining > 0 and selector.select(remaining), (
                f"timed out waiting for {marker!r}; return code: {process.poll()!r}; "
                f"captured: {bytes(output)!r}"
            )
            chunk = os.read(stream.fileno(), 4096)
            assert chunk, (
                f"process exited before {marker!r}; return code: {process.poll()!r}; "
                f"captured: {bytes(output)!r}"
            )
            output.extend(chunk)
    return bytes(output)


def _serve_request(file, request_id: str, method: str) -> dict[str, object]:
    file.write(
        (
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": {"protocol_version": "1.2"} if method == "hello" else {},
                }
            )
            + "\n"
        ).encode()
    )
    file.flush()
    while True:
        line = file.readline()
        assert line, f"serve closed before replying to {method}"
        frame = json.loads(line)
        if frame.get("id") == request_id:
            return frame


def _exercise_serve(socket_path: Path) -> None:
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(60)
        client.connect(str(socket_path))
        with client.makefile("rwb") as file:
            hello = _serve_request(file, "hello", "hello")
            created = _serve_request(file, "new", "new_session")
    assert hello["result"]["server"] == "zeta"
    assert created["result"]["session"]["session_id"]


def _run_console_surface(
    home: Path, surface: str, *, expect_start: bool
) -> tuple[bool, str, int]:
    console = Path(sys.executable).with_name("zeta")
    assert console.is_file(), f"installed zeta console entry is missing: {console}"
    socket_path = Path("/tmp") / (
        f"zeta-storage-{os.getpid()}-{secrets.token_hex(4)}.sock"
    )
    commands = {
        "headless": [
            str(console),
            "--provider",
            "codex",
            "--model",
            "offline",
            "-p",
            "hello",
        ],
        "tui": [str(console), "--provider", "codex", "--model", "offline"],
        "serve": [
            str(console),
            "serve",
            "--ws",
            "0",
            "--socket",
            str(socket_path),
            "--provider",
            "codex",
            "--model",
            "offline",
            "--no-auto-memory",
        ],
    }
    environment = _console_environment(home)
    if surface == "headless":
        result = subprocess.run(
            commands[surface],
            env=environment,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        return result.returncode != 2, result.stderr, result.returncode

    master_fd, slave_fd = pty.openpty()
    process = subprocess.Popen(
        commands[surface],
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=subprocess.PIPE,
        env=environment,
        close_fds=True,
    )
    os.close(slave_fd)
    if not expect_start:
        try:
            _stdout, stderr = process.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=60)
            raise
        finally:
            os.close(master_fd)
            socket_path.unlink(missing_ok=True)
            Path(f"{socket_path}.lock").unlink(missing_ok=True)
        return False, stderr.decode(errors="replace"), process.returncode

    stderr_prefix = b""
    try:
        if surface == "tui":
            output = bytearray()
            read_pty_until(
                master_fd,
                b" \xe2\x80\xba ",
                process=process,
                output=output,
                timeout=60,
            )
        else:
            assert process.stderr is not None
            stderr_prefix = _read_stream_until(
                process.stderr,
                b"zeta serve: websocket ready:",
                process=process,
                timeout=60,
            )
            _exercise_serve(socket_path)
    finally:
        if process.poll() is None:
            process.terminate()
        return_code = reap_process(process, timeout=60)
        _stdout, stderr_tail = process.communicate(timeout=60)
        os.close(master_fd)
        socket_path.unlink(missing_ok=True)
        Path(f"{socket_path}.lock").unlink(missing_ok=True)
    return True, (stderr_prefix + stderr_tail).decode(errors="replace"), return_code


@pytest.mark.parametrize("shape", STORAGE_HOME_SHAPES)
@pytest.mark.parametrize("surface", ("headless", "tui", "serve"))
def test_real_console_start_has_one_storage_format_owner(
    tmp_path: Path, shape: StorageHomeShape, surface: str
) -> None:
    home = tmp_path / f"{surface}-{shape}"
    with prepared_storage_home(home, shape):
        before = _tree_digest(home)
        expected_error = _storage_error(home, shape)
        started, stderr, return_code = _run_console_surface(
            home, surface, expect_start=expected_error is None
        )

        assert "Traceback" not in stderr
        if expected_error is None:
            assert started, stderr
            assert json.loads((home / "storage-format.json").read_text()) == {
                "writer_epoch": WRITER_EPOCH
            }
        else:
            assert not started
            assert return_code == 2
            assert stderr.splitlines() == [f"zeta: {expected_error}"]
            assert _tree_digest(home) == before


def test_storage_format_advance_defaults_to_writer_epoch(tmp_path: Path) -> None:
    result = _run_cli(tmp_path, "storage-format", "advance", "--expected-epoch", "0")

    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "storage-format.json").read_text()) == {
        "writer_epoch": WRITER_EPOCH
    }

    following_start = _run_cli(
        tmp_path, "--provider", "codex", "--model", "offline", "-p", "hello"
    )
    assert "stored-format manifest is missing" not in following_start.stderr


def test_every_session_mapping_writer_preserves_unknown_nested_keys(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    opened = manager.create(provider="codex", model="offline", cwd=tmp_path)
    store = opened.store
    session_dir = store.session_dir

    metadata_path = session_dir / "meta.json"
    state_path = session_dir / "session_state.json"
    _inject(metadata_path)
    _inject(state_path)
    manager.touch(opened.metadata)
    store.set_bash_cwd(tmp_path / "next")
    _assert_future(metadata_path)
    _assert_future(state_path)

    store.start_agent_lifecycle(
        handle="agent-1",
        started_at="2026-01-01T00:00:00+00:00",
        depth=1,
        agent_type="worker",
        description="test",
    )
    lifecycle_path = session_dir / "agent_lifecycle.json"
    _inject(lifecycle_path)
    store.finish_agent_lifecycle("completed", final_result="done")
    _assert_future(lifecycle_path)
    store.close()


def test_keyed_agent_child_delete_survives_reload_and_keeps_unknown_data(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    opened = manager.create(provider="codex", model="offline", cwd=tmp_path)
    store = opened.store
    for child_id in ("child-1", "child-2"):
        store.register_agent_child(
            ToolCall(child_id, "agent", {"description": child_id}),
            child_session_path=str(store.session_dir / "agents" / child_id),
            description=child_id,
            child_instance_id=child_id,
        )

    state_path = store.state_path
    state = json.loads(state_path.read_text())
    state["agent_children"]["child-2"]["future"] = {"nested": True}
    future_child = dict(state["agent_children"]["child-2"])
    future_child["future_only"] = True
    state["agent_children"]["future-child"] = future_child
    state_path.write_text(json.dumps(state))

    store.finish_agent_child("child-1")
    store.close()
    reopened = ConversationStore(
        manager.sessions_dir, session_id=opened.metadata.session_id
    )
    persisted = json.loads(state_path.read_text())["agent_children"]
    assert set(reopened.agent_children()) == {"child-2", "future-child"}
    assert "child-1" not in persisted
    assert persisted["child-2"]["future"] == {"nested": True}
    assert persisted["future-child"]["future_only"] is True
    reopened.close()


@pytest.mark.asyncio
async def test_remaining_session_publishers_preserve_unknown_nested_keys(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    opened = await asyncio.to_thread(
        manager.create, provider="codex", model="offline", cwd=tmp_path
    )
    store = opened.store

    workspace = WorkspaceSnapshotStore(
        store.session_dir,
        store.session_id,
        storage_lease=store._storage_lease,
    )
    workspace.state_path.write_text(json.dumps({}))
    _inject(workspace.state_path)
    workspace._persist()
    _assert_future(workspace.state_path)
    workspace.close()

    background_path = store.session_dir / "background_tasks.json"
    background_path.write_text(
        json.dumps(
            [
                {
                    "task_id": "task-1",
                    "command": "true",
                    "pid": 1,
                    "running": False,
                    "future": {"nested": True},
                },
                {
                    "task_id": "future-task",
                    "future_shape": {"version": 2},
                },
            ]
        )
    )
    background = BackgroundTaskRegistry(
        session_dir=store.session_dir,
        directory_fd=store.directory_fd,
        storage_lease=store._storage_lease,
    )
    background._persist()
    background_rows = {
        row["task_id"]: row for row in json.loads(background_path.read_text())
    }
    assert background_rows["task-1"]["future"] == {"nested": True}
    assert "future-task" in background_rows
    assert background_rows["future-task"] == {
        "task_id": "future-task",
        "future_shape": {"version": 2},
    }
    await background.close()

    from zeta.core.session_links import persist_pending_root_link

    persist_pending_root_link(
        manager.sessions_dir,
        store.session_id,
        "project-1",
        "session",
        None,
        store._storage_lease,
    )
    link_path = store.session_dir / "project_link_pending.json"
    _inject(link_path)
    persist_pending_root_link(
        manager.sessions_dir,
        store.session_id,
        "project-1",
        "session",
        None,
        store._storage_lease,
    )
    _assert_future(link_path)

    draft_path = store.session_dir / "draft"
    draft = DraftPersistence(
        draft_path,
        delay=0,
        directory_fd=store.directory_fd,
        storage_lease=store._storage_lease,
    )
    draft.schedule("first")
    await draft.flush_pending()
    _inject(draft_path)
    draft.schedule("second")
    await draft.flush_pending()
    _assert_future(draft_path)
    store.close()


def test_session_manager_refuses_too_new_home_without_changing_metadata(
    tmp_path: Path,
) -> None:
    (tmp_path / "storage-format.json").write_bytes(b'{"writer_epoch":999}\n')
    metadata = tmp_path / "sessions" / "session-1" / "meta.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_bytes(b'{"future":"metadata"}\n')

    with pytest.raises(StorageFormatError, match="newer than writer capability"):
        SessionManager(tmp_path)

    assert metadata.read_bytes() == b'{"future":"metadata"}\n'


def test_conversation_store_refuses_too_new_home_without_changing_session_state(
    tmp_path: Path,
) -> None:
    (tmp_path / "storage-format.json").write_bytes(b'{"writer_epoch":999}\n')
    state = tmp_path / "sessions" / "session-1" / "session_state.json"
    state.parent.mkdir(parents=True)
    state.write_bytes(b'{"bash_cwd":"/future","future":"session state"}\n')

    with pytest.raises(StorageFormatError, match="newer than writer capability"):
        ConversationStore(tmp_path / "sessions", session_id="session-1")

    assert state.read_bytes() == b'{"bash_cwd":"/future","future":"session state"}\n'


def test_workspace_store_refuses_too_new_home_without_changing_snapshots(
    tmp_path: Path,
) -> None:
    (tmp_path / "storage-format.json").write_bytes(b'{"writer_epoch":999}\n')
    session_dir = tmp_path / "sessions" / "session-1"
    snapshots = session_dir / "workspace_snapshots.json"
    snapshots.parent.mkdir(parents=True)
    snapshots.write_bytes(b'{"future":"snapshots"}\n')

    with pytest.raises(StorageFormatError, match="newer than writer capability"):
        WorkspaceSnapshotStore(session_dir, "session-1")

    assert snapshots.read_bytes() == b'{"future":"snapshots"}\n'


def test_too_new_refusal_is_visible_in_tui_and_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "storage-format.json").write_text('{"writer_epoch":999}\n')
    monkeypatch.setenv("ZETA_HOME", str(tmp_path))
    assert main(["--provider", "codex", "--model", "offline"]) == 2
    assert "newer than writer capability" in capsys.readouterr().err
    assert main(["serve", "--provider", "codex", "--model", "offline"]) == 2
    assert "newer than writer capability" in capsys.readouterr().err


def test_unknown_conversation_variant_fails_closed() -> None:
    with pytest.raises(
        ConversationIntegrityError, match="unsupported conversation entry type"
    ):
        ConversationEntry.from_dict(
            {
                "seq": 1,
                "id": "entry-1",
                "parent_id": None,
                "lane": "main",
                "type": "future_variant",
                "data": {},
            }
        )
