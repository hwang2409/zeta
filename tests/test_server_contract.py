from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

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


def _install_wrong_readiness_process(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[asyncio.subprocess.Process, Path]]:
    spawned: list[tuple[asyncio.subprocess.Process, Path]] = []
    create_process = asyncio.create_subprocess_exec
    child = (
        "import os,signal,socket,sys; "
        "path=sys.argv[1]; "
        "server=socket.socket(socket.AF_UNIX); "
        "server.bind(path); "
        "sys.stderr.write('wrong readiness marker\\n'); "
        "sys.stderr.flush(); "
        "os.close(2); "
        "signal.pause()"
    )

    async def create_wrong_readiness_process(
        *command: object, **kwargs: Any
    ) -> asyncio.subprocess.Process:
        socket_argument = command.index("--socket") + 1
        socket_path = Path(str(command[socket_argument]))
        process = await create_process(
            sys.executable,
            "-c",
            child,
            str(socket_path),
            stdout=kwargs["stdout"],
            stderr=kwargs["stderr"],
        )
        spawned.append((process, socket_path))
        return process

    # test-contract: seam ServeContract has no public process factory
    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", create_wrong_readiness_process
    )
    return spawned


async def _remove_spawned_process(
    process: asyncio.subprocess.Process,
    socket_path: Path,
    unlink: Any = Path.unlink,
) -> None:
    if process.returncode is None:
        process.kill()
    await process.communicate()
    unlink(socket_path, missing_ok=True)


@pytest.mark.asyncio
async def test_failed_start_reaps_process_and_removes_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawned = _install_wrong_readiness_process(monkeypatch)

    try:
        with pytest.raises(AssertionError, match="wrong readiness marker"):
            await ServeContract.start(tmp_path, "unix")

        process, socket_path = spawned[0]
        assert process.returncode is not None
        with pytest.raises(ProcessLookupError):
            os.kill(process.pid, 0)
        assert not socket_path.exists()
    finally:
        if spawned:
            await _remove_spawned_process(*spawned[0])


@pytest.mark.asyncio
async def test_failed_start_preserves_error_when_socket_removal_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawned = _install_wrong_readiness_process(monkeypatch)
    real_unlink = Path.unlink

    def fail_socket_removal(path: Path, *, missing_ok: bool = False) -> None:
        if path.name.startswith("zeta-contract-"):
            raise OSError("forced socket removal failure")
        real_unlink(path, missing_ok=missing_ok)

    # test-contract: seam ServeContract has no public socket-removal dependency
    monkeypatch.setattr(Path, "unlink", fail_socket_removal)
    try:
        with pytest.raises(AssertionError, match="wrong readiness marker") as raised:
            await ServeContract.start(tmp_path, "unix")

        process, _socket_path = spawned[0]
        assert process.returncode is not None
        with pytest.raises(ProcessLookupError):
            os.kill(process.pid, 0)
        assert any(
            "forced socket removal failure" in note for note in raised.value.__notes__
        )
    finally:
        if spawned:
            await _remove_spawned_process(*spawned[0], unlink=real_unlink)


async def _hello(client: ServeClient, request_id: str = "hello") -> dict[str, object]:
    reply = await client.request(
        request_id,
        "hello",
        {
            "protocol_version": "1.2",
            "features": ["serve_observers", "not_a_server_feature"],
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
    assert hello["capabilities"]["features"] == ["serve_observers"]
    assert {"new_session", "send"} <= set(hello["capabilities"]["requests"])


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
            "features": ["serve_observers"],
        },
    )

    observed = await observer.request("observer-list", "list_sessions")
    rejected = await observer.request("observer-new", "new_session")
    created = await controller.request("controller-new", "new_session")

    assert "new_session" in controller_hello["capabilities"]["requests"]
    assert "new_session" not in observer_hello["result"]["capabilities"]["requests"]
    assert observed["result"] == {"sessions": []}
    assert rejected["error"]["code"] == -32008
    assert rejected["error"]["data"] == {
        "code": "observer_read_only",
        "method": "new_session",
    }
    assert created["result"]["session"]["session_id"]
