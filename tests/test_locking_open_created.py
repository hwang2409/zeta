from __future__ import annotations

import errno
import os
from collections.abc import Callable
from pathlib import Path

import pytest

from zeta.core.session_files import SessionError, open_session_file
from zeta.locking import _CREATE_RACE_RETRIES, open_created

CREATE = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK


def _winner_creates(directory: Path, name: str) -> None:
    (directory / name).touch(mode=0o600)


def _winner_publishes_symlink(directory: Path, name: str) -> None:
    (directory / "elsewhere").touch()
    os.symlink(directory / "elsewhere", directory / name)


def _winner_publishes_hard_link(directory: Path, name: str) -> None:
    (directory / "elsewhere").touch()
    os.link(directory / "elsewhere", directory / name)


# (case, open through, flags, setup before the first open, darwin race on the first
#  open: what "another process" leaves at the name, expected opens, expected result)
CASES = [
    ("loser of a concurrent create gets the winner's file", "helper", CREATE, None,
     _winner_creates, 2, "same inode"),
    ("existing file opens on the first try", "helper", CREATE, "existing",
     None, 1, "same inode"),
    ("no O_CREAT: ENOENT raises at once", "helper", os.O_RDWR, None,
     None, 1, FileNotFoundError),
    ("deleted directory: ENOENT raises after the bound", "helper", CREATE, "rmdir",
     None, _CREATE_RACE_RETRIES, FileNotFoundError),
    ("session file: retry finds a symlink", "session", os.O_RDWR | os.O_CREAT, None,
     _winner_publishes_symlink, 2, errno.ELOOP),
    ("session file: retry finds a hard-linked file", "session", os.O_RDWR | os.O_CREAT,
     None, _winner_publishes_hard_link, 2, SessionError),
]


@pytest.mark.parametrize(
    ("case", "through", "flags", "setup", "race", "opens", "expected"),
    CASES,
    ids=[case[0] for case in CASES],
)
def test_open_created_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    through: str,
    flags: int,
    setup: str | None,
    race: Callable[[Path, str], None] | None,
    opens: int,
    expected: object,
) -> None:
    directory = tmp_path / "locks"
    directory.mkdir()
    name = "x.lock"
    if setup == "existing":
        (directory / name).touch(mode=0o600)
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    if setup == "rmdir":
        directory.rmdir()
    real_open = os.open
    calls: list[str] = []

    def counting_open(path, open_flags, mode=0o777, *, dir_fd=None):
        if dir_fd != directory_fd:
            return real_open(path, open_flags, mode, dir_fd=dir_fd)
        calls.append(path)
        if race is not None and len(calls) == 1:
            # Darwin reports ENOENT to the loser even though the winner's file exists
            race(directory, path)
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT))
        return real_open(path, open_flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", counting_open)
    try:
        if through == "helper":
            def opened() -> int:
                return open_created(name, flags, 0o600, dir_fd=directory_fd)
        else:
            def opened() -> int:
                return open_session_file(directory_fd, name, flags)
        if expected == "same inode":
            fd = opened()
            try:
                assert os.fstat(fd).st_ino == os.stat(directory / name).st_ino
            finally:
                os.close(fd)
        elif isinstance(expected, int):
            with pytest.raises(OSError) as raised:
                opened()
            assert raised.value.errno == expected
        else:
            assert isinstance(expected, type)
            with pytest.raises(expected):
                opened()
        assert len(calls) == opens, case
    finally:
        os.close(directory_fd)
