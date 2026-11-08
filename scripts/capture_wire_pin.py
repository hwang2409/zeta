"""Capture serve wire-pin rows from the checkout that runs this script."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path.cwd()))

from tests.support.serve_contract import ServeContract
from zeta.core import session as session_module
from zeta.storage_format import WRITER_EPOCH, advance

ERGONOMICS_ROWS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("rename_session", {"session_id": "missing", "name": "renamed"}),
    ("delete_session", {"session_id": "missing", "name": "unused"}),
    ("session_tree", {"session_id": "missing"}),
    ("session_history", {"session_id": "missing"}),
    ("content_read", {"handle": "missing"}),
    ("switch_branch", {"session_id": "missing", "head_id": "missing"}),
    ("fork_message", {"session_id": "missing", "message_id": "missing"}),
    ("model_catalog", {}),
    ("session_settings", {"session_id": "missing"}),
    (
        "set_settings",
        {"session_id": "missing", "model": "gpt-5.6-luna", "approval_mode": "ask"},
    ),
    ("set_compaction", {"session_id": "missing", "mode": "eviction"}),
    ("send_images", {"session_id": "missing", "text": "x", "images": []}),
    ("slash_list", {"session_id": "missing"}),
    ("slash_run", {"session_id": "missing", "text": "/status"}),
    ("account_list", {"session_id": "missing"}),
    (
        "account_logout",
        {"session_id": "missing", "account": "codex:missing"},
    ),
    ("account_refresh", {"session_id": "missing"}),
    (
        "account_switch",
        {"session_id": "missing", "account": "codex:missing"},
    ),
)

ACTIVE_ROWS: tuple[tuple[str, dict[str, Any]], ...] = tuple(
    (method, {**params, "session_id": "$ACTIVE_SESSION"})
    for method, params in ERGONOMICS_ROWS
    if method
    in {
        "switch_branch",
        "fork_message",
        "session_settings",
        "set_settings",
        "set_compaction",
        "send_images",
        "slash_list",
        "slash_run",
        "account_list",
        "account_logout",
        "account_refresh",
        "account_switch",
    }
)


def _normalize_response(raw: str, active_session_id: str | None = None) -> str:
    """sort providers and replace the generated active-session id for stable pins."""
    if active_session_id is not None:
        raw = raw.replace(active_session_id, "$ACTIVE_SESSION")
    message = json.loads(raw)
    result = message.get("result")
    if isinstance(result, dict) and isinstance(result.get("providers"), dict):
        result["providers"] = dict(sorted(result["providers"].items()))
    return json.dumps(message, separators=(",", ":")) + "\n"


def _row(
    method: str,
    request_id: str,
    params: dict[str, Any],
    response: str,
    active_session_id: str | None = None,
) -> dict[str, Any]:
    return {
        "kind": "ergonomics_valid",
        "role": "controller",
        "protocol_version": "1.2",
        "features": [],
        "method": method,
        "request_id": request_id,
        "params": params,
        "response": _normalize_response(response, active_session_id),
    }


async def _capture(fixture: Path) -> list[dict[str, Any]]:
    session_module.uuid.uuid4 = lambda: session_module.uuid.UUID(
        "00000000-0000-0000-0000-000000000002"
    )
    root = Path("/tmp/serve-wire-pin-capture")
    shutil.rmtree(root, ignore_errors=True)
    account_home = root / "account-home"
    account_home.mkdir(parents=True)
    os.environ["ZETA_ACCOUNT_HOME"] = str(account_home)
    advance(account_home, expected_epoch=0, target_epoch=WRITER_EPOCH)
    (account_home / "accounts.json").write_text(
        json.dumps(
            {
                "version": 1,
                "accounts": {"codex:pin": {"provider": "codex", "name": "pin"}},
                "defaults": {"codex": "codex:pin"},
                "labels": {},
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    contract = await ServeContract.start(root, "unix")
    try:
        client = await contract.connect("controller")
        await client.request_raw(
            "hello", "hello", {"protocol_version": "1.2", "features": []}
        )
        rows: list[dict[str, Any]] = []
        for index, (method, params) in enumerate(ERGONOMICS_ROWS):
            rows.append(
                _row(
                    method,
                    str(index),
                    params,
                    await client.request_raw(str(index), method, params),
                )
            )
        created = json.loads(
            await client.request_raw(
                "active-session", "new_session", {"provider": "codex"}
            )
        )
        session_id = created["result"]["session"]["session_id"]
        for index, (method, params) in enumerate(
            ACTIVE_ROWS, start=len(ERGONOMICS_ROWS)
        ):
            request_params = {
                key: session_id if value == "$ACTIVE_SESSION" else value
                for key, value in params.items()
            }
            rows.append(
                _row(
                    method,
                    f"active-{index}",
                    params,
                    await client.request_raw(f"active-{index}", method, request_params),
                    session_id,
                )
            )
        return rows
    finally:
        await contract.close()


def main() -> None:
    fixture = Path(sys.argv[1])
    output = Path(sys.argv[2])
    existing = [json.loads(line) for line in fixture.read_text().splitlines()[1:]]
    rows = [row for row in existing if row.get("kind") != "ergonomics_valid"]
    rows.extend(asyncio.run(_capture(fixture)))
    header = {
        "captured_at": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "command": "uv run python scripts/capture_wire_pin.py tests/fixtures/serve_wire_pin.jsonl /tmp/serve-wire-pin.jsonl",
        "normalization": "model_catalog providers are sorted by key in capture and comparison; provider order is not a wire contract",
    }
    output.write_text(
        json.dumps(header, separators=(",", ":"))
        + "\n"
        + "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
    )


if __name__ == "__main__":
    main()
