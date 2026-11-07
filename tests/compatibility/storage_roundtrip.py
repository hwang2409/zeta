"""Run one small stored-format writer round trip in a supplied home."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from zeta.core.fsio import write_all
from zeta.storage_format import WRITER_EPOCH, mapping_mutations, open_writer


def _publish(directory_fd: int, name: str, payload: bytes) -> None:
    temporary = f".{name}.compatibility.tmp"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600, dir_fd=directory_fd)
    try:
        write_all(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
    os.fsync(directory_fd)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--mode", choices=("write", "reopen"), required=True)
    args = parser.parse_args()
    home = args.home
    if args.mode == "reopen":
        with open_writer(home, capability=WRITER_EPOCH):
            value = json.loads((home / "meta.json").read_text())
        if value.get("owned") != "after":
            raise RuntimeError("candidate did not reopen the old-writer mutation")
        return 0
    with open_writer(home, capability=WRITER_EPOCH) as lease:
        directory_fd = os.open(home, os.O_RDONLY | os.O_DIRECTORY)
        try:
            lease.publish_json(
                directory_fd,
                "meta.json",
                mapping_mutations({"owned": "after"}),
                _publish,
            )
        finally:
            os.close(directory_fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
