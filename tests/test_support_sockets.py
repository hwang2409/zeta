from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.support.sockets import cleanup_short_socket_paths, short_socket_path

pytest_plugins = ("tests.support.socket_fixtures",)


def test_short_socket_path_separates_same_named_tests_and_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_root = tmp_path / "worker-a" / "same-test-name0"
    second_root = tmp_path / "worker-b" / "same-test-name0"

    first = short_socket_path(first_root)
    second = short_socket_path(second_root)
    monkeypatch.setattr(os, "getpid", lambda: 999_999)
    other_process = short_socket_path(first_root)

    assert len({first, second, other_process}) == 3
    assert max(len(os.fsencode(path)) for path in (first, second, other_process)) < 104


@pytest.mark.parametrize(
    ("candidate", "removed"),
    [
        pytest.param("owned-socket", True, id="owned-socket"),
        pytest.param("owned-lock", True, id="owned-lock"),
        pytest.param("other-process-lock", False, id="other-process-lock-survives"),
    ],
)
def test_socket_cleanup_removes_only_recorded_paths(
    tmp_path: Path, candidate: str, removed: bool
) -> None:
    socket_path = short_socket_path(tmp_path)
    lock_path = Path(f"{socket_path}.lock")
    other_process_lock = socket_path.with_name(
        f"{socket_path.stem}-other-process.sock.lock"
    )
    paths = {
        "owned-socket": socket_path,
        "owned-lock": lock_path,
        "other-process-lock": other_process_lock,
    }
    path = paths[candidate]
    path.touch()
    try:
        cleanup_short_socket_paths()
        assert path.exists() == (not removed)
    finally:
        path.unlink(missing_ok=True)
