from __future__ import annotations

import asyncio
import fcntl
import os
import shutil
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from zeta.storage_format import WRITER_EPOCH, StorageFormatError, open_writer

StorageHomeShape = Literal[
    "empty",
    "gated",
    "legacy-oauth",
    "too-new",
    "invalid",
    "locked",
    "realistic-old",
]
STORAGE_HOME_SHAPES: tuple[StorageHomeShape, ...] = (
    "empty",
    "gated",
    "legacy-oauth",
    "too-new",
    "invalid",
    "locked",
    "realistic-old",
)


def initialize_storage_home(home: Path) -> Path:
    def initialize() -> None:
        with open_writer(home):
            pass

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        initialize()
    else:
        error: list[OSError | StorageFormatError] = []

        def initialize_off_loop() -> None:
            try:
                initialize()
            except (OSError, StorageFormatError) as exc:
                error.append(exc)

        thread = threading.Thread(target=initialize_off_loop)
        thread.start()
        thread.join()
        if error:
            raise error[0]
    return home


@contextmanager
def prepared_storage_home(home: Path, shape: StorageHomeShape) -> Iterator[None]:
    home.mkdir(parents=True)
    lock_path = home / ".storage-format.lock"
    lock_fd: int | None = None

    if shape == "realistic-old":
        shutil.copytree(
            Path(__file__).parents[1] / "fixtures" / "fresh_old_home",
            home,
            dirs_exist_ok=True,
        )
    else:
        lock_path.touch()

    if shape == "gated":
        (home / "sessions" / "existing").mkdir(parents=True)
    elif shape == "legacy-oauth":
        (home / "codex-oauth.json").write_text(
            '{"version":1,"access_token":"legacy-access",'
            '"refresh_token":"legacy-refresh","expires_at":1}\n'
        )
    elif shape == "too-new":
        (home / "storage-format.json").write_text(
            f'{{"writer_epoch":{WRITER_EPOCH + 1}}}\n'
        )
    elif shape == "invalid":
        (home / "storage-format.json").write_text("not json\n")
    elif shape == "locked":
        lock_fd = os.open(lock_path, os.O_RDWR)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        yield
    finally:
        if lock_fd is not None:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
