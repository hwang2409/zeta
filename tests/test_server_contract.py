from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from tests.support.serve_contract import ServeClient, ServeContract


@pytest.fixture(params=("unix", "websocket"))
async def serve_contract(
    request: pytest.FixtureRequest, tmp_path: Path
) -> AsyncIterator[ServeContract]:
    async with asyncio.timeout(60):
        contract = await ServeContract.start(tmp_path, request.param)
        try:
            yield contract
        finally:
            await contract.close()


async def _hello(client: ServeClient, request_id: str = "hello") -> dict[str, object]:
    reply = await client.request(
        request_id,
        "hello",
        {
            "protocol_version": "1.2",
            "features": [
                "serve_observers",
                "session_grants",
                "not_a_server_feature",
            ],
        },
    )
    return reply["result"]


@pytest.mark.asyncio
async def test_hello_negotiates_the_same_features(
    serve_contract: ServeContract,
) -> None:
    client = await serve_contract.connect()

    hello = await _hello(client)

    assert hello["server"] == "zeta"
    assert hello["protocol_version"] == "1.2"
    assert hello["capabilities"]["features"] == [
        "serve_observers",
        "session_grants",
    ]
    assert {
        "new_session",
        "send",
        "session_grants",
        "revoke_session_grants",
    } <= set(hello["capabilities"]["requests"])


@pytest.mark.asyncio
async def test_new_session_returns_public_metadata(
    serve_contract: ServeContract,
) -> None:
    client = await serve_contract.connect()
    await _hello(client)

    reply = await client.request("new", "new_session")

    session = reply["result"]["session"]
    assert session["session_id"]
    assert session["provider"] == "codex"
    assert session["cwd"]


@pytest.mark.asyncio
async def test_send_streams_the_scripted_reply(
    serve_contract: ServeContract,
) -> None:
    client = await serve_contract.connect()
    await _hello(client)
    created = await client.request("new", "new_session")
    session_id = created["result"]["session"]["session_id"]

    sent = await client.request("send", "send", {"text": "contract prompt"})
    assistant = await client.event("assistant_message")
    ended = await client.event("agent_end")

    assert sent["result"] == {"accepted": True, "session_id": session_id}
    assert assistant["session_id"] == session_id
    assert assistant["message"]["content"] == [
        {"type": "text", "text": "you said: contract prompt"}
    ]
    assert ended["session_id"] == session_id


@pytest.mark.asyncio
async def test_public_error_codes_match(
    serve_contract: ServeContract,
) -> None:
    client = await serve_contract.connect()
    await _hello(client)

    unknown_method = await client.request("unknown", "not_a_method")
    bad_params = await client.request("bad-params", "new_session", {"provider": 1})
    unknown_session = await client.request(
        "unknown-session", "resume", {"session_id": "missing-session"}
    )

    assert unknown_method["error"]["code"] == -32601
    assert bad_params["error"]["code"] == -32602
    assert unknown_session["error"]["code"] == -32602
    assert unknown_session["error"]["data"] == {
        "code": "session_not_found",
        "session_id": "missing-session",
    }


@pytest.mark.asyncio
async def test_controller_and_observer_roles_match(
    serve_contract: ServeContract,
) -> None:
    controller = await serve_contract.connect("controller")
    controller_hello = await _hello(controller, "controller-hello")
    observer = await serve_contract.connect("observer")
    observer_hello = await observer.request(
        "observer-hello",
        "hello",
        {
            "protocol_version": "1.2",
            "role": "observer",
            "features": ["serve_observers", "session_grants"],
        },
    )

    observed = await observer.request("observer-list", "list_sessions")
    rejected = await observer.request("observer-new", "new_session")
    created = await controller.request("controller-new", "new_session")

    assert "new_session" in controller_hello["capabilities"]["requests"]
    observer_requests = observer_hello["result"]["capabilities"]["requests"]
    assert "new_session" not in observer_requests
    assert "session_grants" in observer_requests
    assert "revoke_session_grants" not in observer_requests
    assert observed["result"] == {"sessions": []}
    assert rejected["error"]["code"] == -32008
    assert rejected["error"]["data"] == {
        "code": "observer_read_only",
        "method": "new_session",
    }
    assert created["result"]["session"]["session_id"]
