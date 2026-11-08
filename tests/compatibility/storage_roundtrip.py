"""Run one small stored-format writer round trip in a supplied home."""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

from zeta.core.fsio import write_all
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    Message,
    MessageOrigin,
    MessageRole,
    TextContent,
    with_message_origin,
)
from zeta.storage_format import WRITER_EPOCH, mapping_mutations, open_writer

_SESSION_ID = "format-roundtrip"


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


def _write_transcript_recovery(home: Path) -> None:
    sessions = home / "sessions"
    store = ConversationStore(sessions, session_id=_SESSION_ID)
    try:
        store.append_message(
            with_message_origin(
                Message(MessageRole.USER, [TextContent("kept")]), MessageOrigin.USER
            )
        )
        path = store.path
    finally:
        store.close()
    with path.open("ab") as handle:
        handle.write(b'{"seq": 3, "id": "torn"')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        repaired = ConversationStore(sessions, session_id=_SESSION_ID)
    try:
        warning = repaired.entries[-1]
        if warning.type != "warning" or warning.data.get("kind") != "transcript_recovery":
            raise RuntimeError("old writer did not record transcript recovery")
        quarantine_id = warning.data["quarantine_id"]
        expected = (
            repaired.session_dir / f"conversation.quarantine.{quarantine_id}.json",
            repaired.session_dir / f"conversation.quarantine.{quarantine_id}.jsonl",
            repaired.session_dir / f"conversation.recovery.{quarantine_id}.json",
        )
        if not all(path.is_file() for path in expected):
            raise RuntimeError("old writer did not publish transcript recovery records")
    finally:
        repaired.close()


def _reopen(home: Path) -> None:
    with open_writer(home, capability=WRITER_EPOCH):
        value = json.loads((home / "meta.json").read_text())
    if value.get("owned") != "after" or value.get("future") != {"keep": True}:
        raise RuntimeError("candidate did not reopen the old-writer mutation")
    store = ConversationStore(home / "sessions", session_id=_SESSION_ID)
    try:
        warning = store.entries[-1]
        if warning.type != "warning" or warning.data.get("kind") != "transcript_recovery":
            raise RuntimeError("candidate did not reopen the old-writer recovery")
    finally:
        store.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--mode", choices=("write", "reopen"), required=True)
    args = parser.parse_args()
    home = args.home
    if args.mode == "reopen":
        _reopen(home)
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
    _write_transcript_recovery(home)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
