"""``home_settings_v1`` through a real ``zeta serve`` websocket: negotiation, roles, and the update result."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.support.serve_contract import ServeContract
from zeta.config import home_settings
from zeta.server.handlers import settings as settings_handlers

FEATURES = ["home_settings_v1"]
BUILTIN = "http://127.0.0.1:4323/v1/traces"
ON = [{"key": "telemetry.enabled", "value": True}]
INVALID = {"code": -32602, "message": "invalid params", "data": {"code": "invalid_params"}}


def _revision(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


async def _client(contract: ServeContract, role: str = "controller", features: list[str] = FEATURES):
    client = await contract.connect(role)  # type: ignore[arg-type]
    hello = await client.request("hello", "hello", {
        "protocol_version": "1.2", "features": features, **({"role": "observer"} if role == "observer" else {}),
    })
    return client, hello["result"]


@pytest.mark.asyncio
async def test_observers_read_but_never_update_home_settings(tmp_path: Path) -> None:
    settings = tmp_path / "home" / "settings.toml"
    settings.parent.mkdir(parents=True)
    settings.write_bytes(b'model = "gpt-5.6-luna"\n')
    contract = await ServeContract.start(tmp_path, "websocket")
    try:
        plain, plain_hello = await _client(contract, features=[])
        observer, observer_hello = await _client(contract, "observer", ["serve_observers", *FEATURES])

        unnegotiated = await plain.request("get", "home_settings_get")
        got = await observer.request("get", "home_settings_get")
        refused = await observer.request(
            "update", "home_settings_update", {"expected_revision": got["result"]["revision"], "changes": ON}
        )
    finally:
        await contract.close()

    assert "home_settings_get" not in plain_hello["capabilities"]["requests"]
    assert unnegotiated["error"]["code"] == -32601
    assert "home_settings_get" in observer_hello["capabilities"]["requests"]
    assert "home_settings_update" not in observer_hello["capabilities"]["requests"]
    assert got["result"] == {
        "revision": _revision(b'model = "gpt-5.6-luna"\n'),
        "settings": {"telemetry": {"enabled": False, "source": "off", "editable": True}},
        "settings_malformed": False,
    }
    assert refused["error"] == {
        "code": -32008,
        "message": "method is unavailable to read-only observers",
        "data": {"code": "observer_read_only", "method": "home_settings_update"},
    }
    assert settings.read_bytes() == b'model = "gpt-5.6-luna"\n'


@pytest.mark.asyncio
async def test_controller_update_checks_revision_and_lists_restarts(tmp_path: Path) -> None:
    settings = tmp_path / "home" / "settings.toml"
    contract = await ServeContract.start(tmp_path, "websocket")
    try:
        controller, hello = await _client(contract)
        created = await controller.request("new", "new_session", {"provider": "codex"})
        session_id = created["result"]["session"]["session_id"]
        got = await controller.request("get", "home_settings_get")
        invalid = [
            await controller.request("get-params", "home_settings_get", {"key": "otel.endpoint"}),
            await controller.request("extra", "home_settings_update", {
                "expected_revision": got["result"]["revision"], "changes": ON, "endpoint": BUILTIN,
            }),
            await controller.request("unknown", "home_settings_update", {
                "expected_revision": got["result"]["revision"],
                "changes": [{"key": "yolo", "value": True}],
            }),
        ]
        stale = await controller.request(
            "stale", "home_settings_update", {"expected_revision": _revision(b"other"), "changes": ON}
        )
        unchanged = settings.exists()
        updated = await controller.request(
            "update", "home_settings_update", {"expected_revision": got["result"]["revision"], "changes": ON}
        )
    finally:
        await contract.close()

    assert {"home_settings_get", "home_settings_update"} <= set(hello["capabilities"]["requests"])
    assert [reply["error"] for reply in invalid] == [INVALID, INVALID, INVALID]
    assert stale["error"] == {
        "code": -32009,
        "message": "settings.toml changed since the given revision",
        "data": {"code": "stale_revision", "current_revision": got["result"]["revision"]},
    }
    assert unchanged is False
    written = settings.read_bytes()
    assert written == f'[otel]\nendpoint = "{BUILTIN}"\n'.encode()
    assert updated["result"] == {
        "revision": _revision(written),
        "settings": {"telemetry": {"enabled": True, "source": "builtin", "editable": True}},
        "restart": {"serve": True, "sessions": [session_id], "complete": True},
    }


@pytest.mark.asyncio
async def test_update_handler_writes_only_through_home_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple] = []

    def update(home: Path, expected_revision: object, changes: object) -> dict:
        calls.append((home, expected_revision, changes))
        return {"written": "by home_settings"}

    monkeypatch.setattr(home_settings, "update", update)
    client = SimpleNamespace(server=SimpleNamespace(home=tmp_path))

    result = await settings_handlers.home_settings_update(
        client, 1, {"expected_revision": _revision(b""), "changes": ON}
    )

    assert result == {"written": "by home_settings"}
    assert calls == [(tmp_path, _revision(b""), ON)]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_a_filesystem_failure_names_no_path(tmp_path: Path) -> None:
    home = tmp_path / "home"
    contract = await ServeContract.start(tmp_path, "websocket")
    try:
        controller, _ = await _client(contract)
        got = await controller.request("get", "home_settings_get")
        home.rename(tmp_path / "moved")
        try:
            replies = [
                await controller.request("get", "home_settings_get"),
                await controller.request(
                    "update", "home_settings_update", {"expected_revision": got["result"]["revision"], "changes": ON}
                ),
            ]
        finally:
            (tmp_path / "moved").rename(home)
    finally:
        await contract.close()

    for reply in replies:
        assert tmp_path.name not in json.dumps(reply)
        assert reply["error"] == {"code": -32000, "message": "unexpected server or storage failure"}
