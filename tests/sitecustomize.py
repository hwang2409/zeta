"""Record descendant writes for the session-scoped live-home test guard."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _write_event(event: str, args: tuple[object, ...]) -> None:
    live_home_value = os.environ.get("ZETA_TEST_LIVE_HOME")
    ledger_value = os.environ.get("ZETA_TEST_AUDIT_LEDGER")
    if not live_home_value or not ledger_value:
        return

    if event == "open":
        if not args or not isinstance(args[0], (str, bytes)):
            return
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        if (isinstance(mode, str) and "a" in mode) or (
            isinstance(flags, int) and flags & os.O_APPEND
        ):
            kind = "append"
        elif (
            isinstance(mode, str)
            and any(marker in mode for marker in "wax+")
        ) or (
            isinstance(flags, int)
            and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        ):
            kind = "write"
        else:
            return
        candidates = (args[0],)
    elif event in {"os.mkdir", "os.remove", "os.rename", "os.replace", "os.rmdir"}:
        kind = event.removeprefix("os.")
        candidates = args[:2]
    else:
        return

    live_home = Path(live_home_value).expanduser().resolve()
    for candidate in candidates:
        if not isinstance(candidate, (str, bytes)):
            continue
        try:
            path = Path(candidate).expanduser().resolve()
            path.relative_to(live_home)
        except (OSError, RuntimeError, ValueError):
            continue
        try:
            record = {
                "path": str(path),
                "kind": kind,
                "ownership_token": os.environ.get("ZETA_TEST_OWNERSHIP_TOKEN"),
            }
            with Path(ledger_value).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
        except (OSError, ValueError):
            return


def _audit(event: str, args: tuple[object, ...]) -> None:
    try:
        _write_event(event, args)
    except (OSError, RuntimeError, TypeError, ValueError):
        return


sys.addaudithook(_audit)


def _install_storage_format_bootstrap() -> None:
    parent_pid = os.environ.get("ZETA_TEST_PARENT_PID")
    if parent_pid is None or parent_pid == str(os.getpid()):
        return
    from zeta import storage_format

    real_open_writer = storage_format.open_writer

    def open_test_writer(home, *, capability=storage_format.WRITER_EPOCH):
        path = Path(home)
        if not (path / "storage-format.json").exists():
            storage_format.advance(path, expected_epoch=0)
        return real_open_writer(path, capability=capability)

    storage_format.open_writer = open_test_writer
    for module in tuple(sys.modules.values()):
        if module is not None and getattr(module, "open_writer", None) is real_open_writer:
            module.open_writer = open_test_writer


_install_storage_format_bootstrap()
