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
        if not (
            isinstance(mode, str) and any(marker in mode for marker in "wax+")
        ) and not (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT)
        ):
            return
        candidates = (args[0],)
    elif event in {"os.mkdir", "os.remove", "os.rename", "os.replace", "os.rmdir"}:
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
            session_id = os.getsid(0)
            record = {
                "path": str(path),
                "pid": os.getpid(),
                "parent_pid": os.getppid(),
                "session_id": session_id,
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
