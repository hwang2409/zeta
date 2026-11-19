"""MCP write-only fields must not escape even when serve telemetry is on."""

import hashlib
import json

import pytest

from tests.support.serve_close import close_serve
from tests.telemetry.test_serve_propagation import (
    Wire,
    _serve_spans,
    _server,
)
from tests.telemetry.test_serve_propagation import (
    traced as traced,  # noqa: PLC0414 - pytest fixture
)


@pytest.mark.asyncio
async def test_mcp_secret_canaries_are_absent_from_wire_logs_and_spans(tmp_path, traced, monkeypatch, caplog):
    telemetry, memory = traced
    server = _server(tmp_path, "unix", telemetry, monkeypatch)
    wire = Wire(server, "unix")
    secrets = [f"CANARY_mcp_{key}_fa2a" for key in ("env", "header", "user", "pass", "path", "query", "client")]
    env, header, user, password, path, query, client_secret = secrets
    await wire.open()
    try:
        await wire.rpc("hello", "hello", {"protocol_version": "1.2", "features": ["otel", "mcp_management_v1"]})
        before = json.loads(await wire.rpc("before", "mcp_server_list"))["result"]
        mutation = {
            "scope": "user", "name": "example", "url": f"https://{user}:{password}@example.test/{path}?key={query}",
            "env": {"transport": env}, "headers": {"Authorization": header},
            "oauth": True, "client_id": "example", "client_secret": client_secret,
        }
        updated = json.loads(await wire.rpc("add", "mcp_server_add", {"revision": before["revision"], "mutation": mutation}))
        assert updated["result"]["data"]["items"][0]["config"] == {
            "transport": "streamable-http", "url": "<redacted-url>", "env": {"transport": "<redacted>"},
            "headers": {"Authorization": "<redacted>"}, "auth": {"type": "oauth", "client_secret": "<redacted>"},
        }
        revision = updated["result"]["revision"]
        listed = json.loads(await wire.rpc("after", "mcp_server_list"))["result"]
        assert listed["revision"] == revision
        bad_url = json.loads(await wire.rpc("invalid", "mcp_server_add", {
            "revision": revision, "mutation": {**mutation, "url": f"https://[{password}"},
        }))
        assert bad_url["error"]["code"] == -32602
    finally:
        await close_serve(server, wire.writer)
    assert len(_serve_spans(memory, "mcp_server_add")) == 2
    surfaces = b"".join(wire.received).decode() + caplog.text + "".join(
        span.to_json() for span in memory.get_finished_spans()
    )
    for secret in secrets:
        assert secret not in surfaces
        assert hashlib.sha256(secret.encode()).hexdigest() not in surfaces
