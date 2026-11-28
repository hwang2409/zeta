from __future__ import annotations

import asyncio
import fcntl
import json
import multiprocessing
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from zeta import storage_format
from zeta.core.checkpoints import ConversationIntegrityError
from zeta.core.checkpoints.workspace import WorkspaceSnapshotStore
from zeta.core.store import ConversationStore
from zeta.storage_format import (
    GATED_HOME_PATHS,
    DeleteValue,
    ReplaceSubtree,
    SetValue,
    StorageFormatError,
    advance,
    apply_mutations,
    open_writer,
    verify,
)
from zeta.tools._shared.process import BackgroundTaskRegistry


def _first_open(home: str, connection) -> None:
    lease = None
    try:
        if not connection.poll(10):
            return
        connection.recv()
        try:
            lease = open_writer(home)
            connection.send("opened")
            connection.poll(10) and connection.recv()
        except StorageFormatError as exc:
            connection.send(str(exc))
    finally:
        if lease is not None:
            lease.close()
        connection.close()


def _try_exclusive_lock(home: str, result: multiprocessing.Queue[str]) -> None:
    fd = os.open(Path(home) / ".storage-format.lock", os.O_RDWR)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            result.put("active writers")
        else:
            result.put("exclusive lock acquired")
    finally:
        os.close(fd)


def _closed_owner_allows_exclusive_lock(home: str, owner: str, result) -> None:
    path = Path(home)
    advance(path, expected_epoch=0)
    root_lease = open_writer(path)
    if owner == "background":
        with ConversationStore(
            path / "sessions", session_id="owner", _storage_lease=root_lease
        ) as store:
            registry = BackgroundTaskRegistry(
                session_dir=store.session_dir,
                directory_fd=store.directory_fd,
                storage_lease=root_lease,
            )
            asyncio.run(registry.close())
    elif owner == "workspace":
        with ConversationStore(
            path / "sessions", session_id="owner", _storage_lease=root_lease
        ) as store:
            try:
                WorkspaceSnapshotStore(
                    store.session_dir,
                    store.session_id,
                    cap=0,
                    storage_lease=root_lease,
                )
            except ValueError:
                pass
    else:
        try:
            ConversationStore(
                path / "sessions", session_id="../unsafe", _storage_lease=root_lease
            )
        except ConversationIntegrityError:
            pass
    root_lease.close()
    _try_exclusive_lock(home, result)


def _assert_closed_owner_allows_exclusive_lock(tmp_path: Path, owner: str) -> None:
    context = multiprocessing.get_context("spawn")
    result = context.Queue()
    process = context.Process(
        target=_closed_owner_allows_exclusive_lock,
        args=(str(tmp_path), owner, result),
    )
    process.start()
    process.join(10)
    assert process.exitcode == 0
    assert result.get(timeout=1) == "exclusive lock acquired"
    result.close()
    result.join_thread()
    process.close()


def test_writer_bootstraps_empty_home_and_refuses_too_new_manifest(
    tmp_path: Path,
) -> None:
    lease = open_writer(tmp_path)
    lease.close()
    assert verify(tmp_path).writer_epoch == storage_format.WRITER_EPOCH

    (tmp_path / "storage-format.json").write_text(
        json.dumps({"writer_epoch": storage_format.WRITER_EPOCH + 1}) + "\n"
    )
    with pytest.raises(StorageFormatError, match="newer than writer capability"):
        open_writer(tmp_path)


def test_predecessor_writer_refuses_advanced_home_without_writes(
    tmp_path: Path,
) -> None:
    advance(tmp_path, expected_epoch=0)
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    assert verify(tmp_path).writer_epoch == storage_format.WRITER_EPOCH
    with pytest.raises(StorageFormatError, match="newer than writer capability"):
        open_writer(tmp_path, capability=2)

    after = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_writer_bootstraps_home_with_empty_sessions_directory(tmp_path: Path) -> None:
    (tmp_path / "sessions").mkdir()

    with open_writer(tmp_path):
        pass

    assert verify(tmp_path).writer_epoch == storage_format.WRITER_EPOCH


@pytest.mark.parametrize(("pattern", "owner"), GATED_HOME_PATHS)
def test_writer_refuses_each_gated_home_shape_without_a_manifest(
    tmp_path: Path, pattern: str, owner: str
) -> None:
    del owner
    path = tmp_path / pattern.replace("*", "stored")
    if path.suffix == ".json":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    else:
        path.mkdir(parents=True)

    with pytest.raises(StorageFormatError) as raised:
        open_writer(tmp_path)

    assert (
        "zeta storage-format advance --expected-epoch 0 "
        f"--target-epoch {storage_format.WRITER_EPOCH}"
    ) in str(raised.value)
    assert not (tmp_path / "storage-format.json").exists()


def _hold_exclusive_lock(home: Path) -> int:
    home.mkdir(parents=True, exist_ok=True)
    fd = os.open(home / ".storage-format.lock", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def test_first_start_waits_for_a_concurrent_bootstrap_then_opens(
    tmp_path: Path,
) -> None:
    holder = _hold_exclusive_lock(tmp_path)
    try:
        command = (
            "import sys; from zeta.cli.main import main; "
            "sys.exit(main(['session', 'list']))"
        )
        started = subprocess.Popen(
            [sys.executable, "-c", command],
            env={**os.environ, "ZETA_HOME": str(tmp_path)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            try:
                started.wait(2)  # the other first start is still publishing
            except subprocess.TimeoutExpired:
                pass
            (tmp_path / "storage-format.json").write_text(
                json.dumps({"writer_epoch": storage_format.WRITER_EPOCH}) + "\n"
            )
        finally:
            fcntl.flock(holder, fcntl.LOCK_UN)
        stdout, stderr = started.communicate(timeout=30)
    finally:
        os.close(holder)
    assert "manifest is missing" not in stderr
    assert started.returncode == 0, stderr
    assert "no stored zeta sessions" in stdout
    assert verify(tmp_path).writer_epoch == storage_format.WRITER_EPOCH


@pytest.mark.parametrize(("held", "notices"), [(2.0, 1), (0.05, 0)])
def test_first_start_says_once_that_it_waits_for_a_slow_bootstrap(
    tmp_path: Path, held: float, notices: int
) -> None:
    holder = _hold_exclusive_lock(tmp_path)
    os.pwrite(holder, f"{os.getpid()}\n".encode(), 0)
    try:
        started = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import sys; from zeta.storage_format import open_writer; "
                    "open_writer(sys.argv[1]).close()"
                ),
                str(tmp_path),
            ],
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            time.sleep(held)
            (tmp_path / "storage-format.json").write_text(
                json.dumps({"writer_epoch": storage_format.WRITER_EPOCH}) + "\n"
            )
        finally:
            fcntl.flock(holder, fcntl.LOCK_UN)
        _stdout, stderr = started.communicate(timeout=30)
    finally:
        os.close(holder)
    assert started.returncode == 0, stderr
    expected = (
        f"zeta: waiting for another zeta process (pid {os.getpid()}) "
        f"to finish creating {tmp_path.resolve()}"
    )
    assert stderr.splitlines().count(expected) == notices, stderr
    assert stderr.count("waiting for another zeta process") == notices, stderr


def test_bootstrap_names_its_pid_in_the_lock_only_while_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    publish = storage_format._publish_manifest
    seen: list[str] = []

    def observed(home: Path, epoch: int) -> None:
        seen.append((home / ".storage-format.lock").read_text())
        publish(home, epoch)

    # test-contract: seam observe the lock payload a concurrent first start would read
    monkeypatch.setattr(storage_format, "_publish_manifest", observed)
    open_writer(tmp_path).close()
    assert seen == [f"{os.getpid()}\n"]
    assert (tmp_path / ".storage-format.lock").read_bytes() == b""


def test_first_start_wait_is_bounded_then_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # test-contract: seam shorten the 10 s bootstrap wait so a stuck holder is cheap to model
    monkeypatch.setattr(storage_format, "_BOOTSTRAP_WAIT_SECONDS", 0.2, raising=False)
    holder = _hold_exclusive_lock(tmp_path)
    # the holder lets go after 5 s, so an unbounded wait fails the elapsed check, not hangs
    release = threading.Timer(5, fcntl.flock, (holder, fcntl.LOCK_UN))
    release.start()
    try:
        started = time.monotonic()
        with pytest.raises(storage_format.MissingManifest, match="manifest is missing"):
            open_writer(tmp_path)
        elapsed = time.monotonic() - started
    finally:
        release.cancel()
        release.join(10)
        os.close(holder)
    # test-contract: latency the refusal must come at its bound, before the 5 s release
    assert elapsed < 4, f"bootstrap wait took {elapsed:.2f} s; it must stop at its bound"
    assert not (tmp_path / "storage-format.json").exists()


def test_data_bearing_home_still_refuses_after_waiting(tmp_path: Path) -> None:
    (tmp_path / "accounts.json").write_text("{}\n")
    holder = _hold_exclusive_lock(tmp_path)
    release = threading.Timer(0.3, fcntl.flock, (holder, fcntl.LOCK_UN))
    release.start()
    try:
        with pytest.raises(storage_format.MissingManifest, match="--expected-epoch 0"):
            open_writer(tmp_path)
    finally:
        release.join(10)
        os.close(holder)
    assert not (tmp_path / "storage-format.json").exists()


def test_writer_creates_and_bootstraps_a_missing_home(tmp_path: Path) -> None:
    home = tmp_path / "not" / "yet"

    with open_writer(home):
        pass

    assert verify(home).writer_epoch == storage_format.WRITER_EPOCH


def test_concurrent_first_starts_all_open_one_empty_home(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    connections = [context.Pipe() for _ in range(6)]
    processes = [
        context.Process(target=_first_open, args=(str(tmp_path), child))
        for (_parent, child) in connections
    ]
    outcomes: list[str] = []
    exitcodes: list[int | None] = []
    try:
        for process in processes:
            process.start()
        for _parent, child in connections:
            child.close()
        for parent, _child in connections:
            parent.send("start")
        for parent, _child in connections:
            assert parent.poll(15), "first-open process did not report an outcome"
            outcomes.append(parent.recv())
    finally:
        for parent, _child in connections:
            try:
                parent.send("release")
            except (BrokenPipeError, OSError):
                pass
        for process in processes:
            process.join(15)
            if process.is_alive():
                process.terminate()
                process.join(15)
        for parent, _child in connections:
            parent.close()
        exitcodes = [process.exitcode for process in processes]
        for process in processes:
            process.close()

    assert all(exitcode == 0 for exitcode in exitcodes)
    assert outcomes == ["opened"] * len(processes)
    assert verify(tmp_path).writer_epoch == storage_format.WRITER_EPOCH
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        ".storage-format.lock",
        "storage-format.json",
    ]


def test_advance_is_exclusive_across_processes(tmp_path: Path) -> None:
    advance(tmp_path, expected_epoch=0)
    lease = open_writer(tmp_path)
    context = multiprocessing.get_context("spawn")
    result = context.Queue()
    process = context.Process(target=_try_exclusive_lock, args=(str(tmp_path), result))
    process.start()
    process.join(10)
    assert process.exitcode == 0
    assert result.get(timeout=1) == "active writers"
    result.close()
    result.join_thread()
    process.close()
    lease.close()
    assert verify(tmp_path).writer_epoch == storage_format.WRITER_EPOCH


def test_advanced_home_refuses_predecessor_before_transcript_parsing(
    tmp_path: Path,
) -> None:
    advance(tmp_path, expected_epoch=0)
    session = tmp_path / "sessions" / "malformed"
    session.mkdir(parents=True)
    (session / "conversation.jsonl").write_bytes(b"not-json\n")
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    assert verify(tmp_path).writer_epoch == storage_format.WRITER_EPOCH, (
        "candidate advanced the home"
    )
    with pytest.raises(StorageFormatError, match="newer than writer capability"):
        open_writer(tmp_path, capability=2)

    after = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_writer_lock_descriptor_is_not_inherited(tmp_path: Path) -> None:
    advance(tmp_path, expected_epoch=0)
    lease = open_writer(tmp_path)
    child_fds = subprocess.check_output(
        [sys.executable, "-c", "import os; print(sorted(os.listdir('/dev/fd')))"],
        text=True,
    )
    assert str(lease._entry.fd) not in child_fds
    lease.close()


def test_background_registry_releases_its_only_writer_lease(tmp_path: Path) -> None:
    _assert_closed_owner_allows_exclusive_lock(tmp_path, "background")


def test_failed_store_construction_releases_writer_lease(tmp_path: Path) -> None:
    _assert_closed_owner_allows_exclusive_lock(tmp_path, "failed-store")


def test_failed_workspace_construction_releases_writer_lease(tmp_path: Path) -> None:
    _assert_closed_owner_allows_exclusive_lock(tmp_path, "workspace")


def test_mutations_preserve_unknown_nested_keys_and_declare_deletes() -> None:
    raw = {
        "known": {"value": 1, "future": {"nested": True}},
        "records": {"a": {"known": 1, "future": "kept"}},
        "remove": True,
    }
    output = apply_mutations(
        raw,
        (
            SetValue(("known",), {"value": 2}),
            SetValue(("records", "a"), {"known": 2}),
            DeleteValue(("remove",)),
            ReplaceSubtree(("owned",), [{"replacement": True}]),
        ),
    )
    assert output == {
        "known": {"value": 2, "future": {"nested": True}},
        "records": {"a": {"known": 2, "future": "kept"}},
        "owned": [{"replacement": True}],
    }


def test_publish_reads_locked_tree_before_writing(tmp_path: Path) -> None:
    advance(tmp_path, expected_epoch=0)
    lease = open_writer(tmp_path)
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    (tmp_path / "state.json").write_text(
        json.dumps({"known": {"value": 1, "future": "kept"}})
    )

    def publish(fd: int, name: str, data: bytes) -> None:
        with open(
            name, "wb", opener=lambda path, flags: os.open(path, flags, dir_fd=fd)
        ) as handle:
            handle.write(data)

    try:
        lease.publish_json(
            directory_fd,
            "state.json",
            (SetValue(("known",), {"value": 2}),),
            publish,
        )
    finally:
        os.close(directory_fd)
        lease.close()
    assert json.loads((tmp_path / "state.json").read_text()) == {
        "known": {"value": 2, "future": "kept"}
    }


def test_registry_matches_queued_steer_validator_shape() -> None:
    from zeta.core.store._client_delivery import validate_client_delivery_data
    from zeta.protocol.types import (
        Message,
        MessageOrigin,
        MessageRole,
        TextContent,
        with_message_origin,
    )

    accepted = with_message_origin(
        Message(MessageRole.USER, [TextContent("resume")]), MessageOrigin.USER
    ).to_dict()
    payload = {
        "delivery_id": "d" * 8,
        "method": "steer",
        "status": "queued",
        "outcome": {"accepted": True},
        "accepted_message": accepted,
    }
    validate_client_delivery_data(payload, "session")
    assert set(payload) == set(storage_format.ACCEPTED_KEY_SETS["client_delivery_queued_steer_acceptance_message"])


def test_registry_dump_is_deterministic_and_detects_epoch_neutral_changes() -> None:
    from zeta.storage_format import compare_registry_dumps, registry_dump

    baseline = registry_dump()
    changed = json.loads(json.dumps(baseline))
    changed["accepted_key_sets"]["todo_item"].append("future")
    with pytest.raises(ValueError, match="shape todo_item.*accepted key.*greater.*bump the epoch"):
        compare_registry_dumps(baseline, changed)

    changed["writer_epoch"] = baseline["writer_epoch"] + 1
    compare_registry_dumps(baseline, changed)

    for registry_name, shape, value in (
        ("closed_variants", "account.provider", "future"),
        ("accepted_key_sets", "todo_item", "removed"),
    ):
        mutated = json.loads(json.dumps(baseline))
        if registry_name == "closed_variants":
            mutated[registry_name][shape].append(value)
        else:
            mutated[registry_name][shape].remove("status")
        with pytest.raises(ValueError, match="WRITER_EPOCH"):
            compare_registry_dumps(baseline, mutated)
        mutated["writer_epoch"] += 1
        compare_registry_dumps(baseline, mutated)
