from __future__ import annotations

import fcntl
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from zeta.storage_format import WRITER_EPOCH

StorageHomeShape = Literal[
    "empty",
    "gated",
    "too-new",
    "invalid",
    "locked",
    "realistic-old",
]
STORAGE_HOME_SHAPES: tuple[StorageHomeShape, ...] = (
    "empty",
    "gated",
    "too-new",
    "invalid",
    "locked",
    "realistic-old",
)


@contextmanager
def prepared_storage_home(home: Path, shape: StorageHomeShape) -> Iterator[None]:
    home.mkdir(parents=True)
    lock_path = home / ".storage-format.lock"
    lock_path.touch()
    lock_fd: int | None = None

    if shape == "gated":
        (home / "sessions" / "existing").mkdir(parents=True)
    elif shape == "too-new":
        (home / "storage-format.json").write_text(
            f'{{"writer_epoch":{WRITER_EPOCH + 1}}}\n'
        )
    elif shape == "invalid":
        (home / "storage-format.json").write_text("not json\n")
    elif shape == "locked":
        lock_fd = os.open(lock_path, os.O_RDWR)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    elif shape == "realistic-old":
        session = home / "sessions" / "legacy-session"
        session.mkdir(parents=True)
        shutil.copy2(
            Path(__file__).parents[1]
            / "fixtures"
            / "legacy-summary-conversation.jsonl",
            session / "conversation.jsonl",
        )

    try:
        yield
    finally:
        if lock_fd is not None:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
