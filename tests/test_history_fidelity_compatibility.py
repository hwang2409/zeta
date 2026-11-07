from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from zeta.core.store import ConversationIntegrityError, ConversationStore
from zeta.protocol.types import (
    Message,
    MessageOrigin,
    MessageRole,
    TextContent,
    with_message_origin,
)


def test_predecessor_store_rejects_new_history_shapes_without_mutation(
    tmp_path: Path,
) -> None:
    sessions = tmp_path / "candidate" / "sessions"
    store = ConversationStore(sessions, session_id="history-fidelity")
    accepted = False
    try:
        accepted_message = with_message_origin(
            Message(MessageRole.USER, [TextContent("keep this steer")]),
            MessageOrigin.USER,
        )
        store._append_row(
            "client_delivery",
            {
                "delivery_id": "new-steer",
                "method": "steer",
                "status": "queued",
                "outcome": {"accepted": True},
                "accepted_message": accepted_message.to_dict(),
            },
        )
        store._append_row("turn_end", {"turn_id": "0" * 32, "reason": "disconnect"})
        store.close()
        reopened = ConversationStore(
            sessions, session_id="history-fidelity", _must_exist=True
        )
        try:
            rows = reopened.replay()
            accepted = {
                row.type for row in rows if row.type in {"client_delivery", "turn_end"}
            } == {"client_delivery", "turn_end"}
        finally:
            reopened.close()
    except (ConversationIntegrityError, ValueError):
        store.close()
    assert accepted, "candidate accepted and reopened both history shapes"

    predecessor_src = os.environ.get("PREDECESSOR_ZETA_SRC")
    if predecessor_src is None:
        pytest.skip("set PREDECESSOR_ZETA_SRC to the #685 worktree src directory")
    predecessor_src_path = Path(predecessor_src).resolve()
    assert (predecessor_src_path / "zeta" / "__init__.py").is_file()

    predecessor_home = tmp_path / "predecessor"
    shutil.copytree(tmp_path / "candidate", predecessor_home)
    (predecessor_home / "storage-format.json").write_text('{"writer_epoch":1}\n')
    before = {
        path.relative_to(predecessor_home): path.read_bytes()
        for path in predecessor_home.rglob("*")
        if path.is_file()
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, zeta; from pathlib import Path; "
                "from zeta.core.store import ConversationStore; "
                "print(zeta.__file__); "
                "root, session_id = Path(sys.argv[1]), sys.argv[2]; store = None\n"
                "try:\n"
                " store = ConversationStore(root, session_id=session_id, _must_exist=True)\n"
                " store.replay()\n"
                "except Exception as exc:\n"
                " print(type(exc).__name__, str(exc), file=sys.stderr)\n"
                " sys.exit(17)\n"
                "finally:\n"
                " store.close() if store is not None else None\n"
            ),
            str(predecessor_home / "sessions"),
            "history-fidelity",
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": str(predecessor_src_path),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 17, result.stdout + result.stderr
    assert str(predecessor_src_path / "zeta" / "__init__.py") in result.stdout
    assert "ConversationIntegrityError" in result.stderr
    after = {
        path.relative_to(predecessor_home): path.read_bytes()
        for path in predecessor_home.rglob("*")
        if path.is_file()
    }
    assert after == before
