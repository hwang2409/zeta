"""``project_writes_v1`` through a real ``zeta serve`` websocket: roles, memory CAS, inbox sends."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from threading import Event
from unittest import mock

import pytest

from tests.support.owned import await_or_fail
from tests.support.replay_transport import (
    Transport,
    connect_replay,
    negotiate,
    replay_server,
)
from tests.support.serve_close import close_serve
from tests.support.serve_contract import ServeClient, ServeContract
from tests.test_server import _connect, _request
from tests.test_server_projects import _server
from zeta import storage_format
from zeta.project_errors import ProjectRegistryBusyError
from zeta.project_inbox import ProjectInbox
from zeta.project_registry import ProjectRegistry

FEATURES = ["projects", "project_writes_v1"]
STALE = {
    "code": -32009,
    "message": "resource revision is stale",
    "data": {"code": "stale_project_memory_revision", "reason": "stale_project_memory_revision"},
}


async def _off_loop(call, *args, **kwargs):
    return await await_or_fail(asyncio.to_thread(call, *args, **kwargs), what="registry setup")


def _projects(root: Path, *, entry_memory: bool = False, writer_epoch: int | None = None) -> tuple[str, str]:
    """Alpha owns ``root/alpha``; beta has no root. Release the test's writer lease after setup.
    ``writer_epoch`` creates the home at that lower storage epoch."""
    with mock.patch.object(storage_format, "WRITER_EPOCH", writer_epoch or storage_format.WRITER_EPOCH):
        registry = ProjectRegistry(root / "home" / "projects")
    (root / "alpha").mkdir()
    try:
        alpha = registry.create_project("alpha", "repo", (root / "alpha").resolve())
        beta = registry.create_project("beta", "service")
        if entry_memory:
            registry.activate_entry_memory(beta.project_id)
        return alpha.project_id, beta.project_id
    finally:
        registry._storage_lease.close()


def _registry_write(root: Path, call) -> None:
    registry = ProjectRegistry(root / "home" / "projects")
    try:
        call(registry)
    finally:
        registry._storage_lease.close()


async def _client(contract: ServeContract, features: list[str], role: str = "controller") -> ServeClient:
    client = await contract.connect(role)  # type: ignore[arg-type]
    hello = await client.request("hello", "hello", {
        "protocol_version": "1.2", "features": features, **({"role": "observer"} if role == "observer" else {}),
    })
    assert "result" in hello, hello
    return client


async def _memory(client: ServeClient, project_id: str) -> dict:
    return (await client.request("show", "project_show", {"project_id": project_id}))["result"]["memory"]


@pytest.mark.asyncio
async def test_observer_is_refused_before_params_are_read(tmp_path: Path) -> None:
    await _off_loop(_projects, tmp_path)
    contract = await ServeContract.start(tmp_path, "websocket")
    try:
        observer = await _client(contract, ["serve_observers", *FEATURES], "observer")
        replies = {
            method: await observer.request(method, method, {"origin": "forged"})
            for method in ("project_memory_update", "project_inbox_send")
        }
    finally:
        await contract.close()
    for method, reply in replies.items():
        assert reply["error"]["code"] == -32008, reply
        assert reply["error"]["data"] == {"code": "observer_read_only", "reason": "observer_read_only", "method": method}


@pytest.mark.asyncio
async def test_format_one_memory_compare_and_set(tmp_path: Path) -> None:
    alpha, _beta = await _off_loop(_projects, tmp_path)
    contract = await ServeContract.start(tmp_path, "websocket")
    try:
        controller = await _client(contract, FEATURES)
        first = (await _memory(controller, alpha))["digest"]
        set_brief = {"format": 1, "op": "set", "file": "brief.md", "content": "serve brief\n"}
        written = await controller.request("set", "project_memory_update", {
            "project_id": alpha, "revision": first, "mutation": set_brief,
        })
        after = await _memory(controller, alpha)
        assert written["result"] == {"revision": after["digest"], "data": {"format": 1}}
        assert {item["name"]: item["content"] for item in after["files"]}["brief.md"] == "serve brief\n"

        # A second writer that read the same revision loses.
        replay = await controller.request("replay", "project_memory_update", {
            "project_id": alpha, "revision": first,
            "mutation": {**set_brief, "content": "lost update\n"},
        })
        assert replay["error"] == STALE

        # Forced order: another process writes between this client's read and its write.
        current = after["digest"]
        await _off_loop(_registry_write, tmp_path, lambda registry: registry.update_memory(alpha, {"state.md": "tui write\n"}))
        raced = await controller.request("raced", "project_memory_update", {
            "project_id": alpha, "revision": current, "mutation": set_brief | {"content": "late\n"},
        })
        assert raced["error"] == STALE
        final = await _memory(controller, alpha)
        assert {item["name"]: item["content"] for item in final["files"]}["brief.md"] == "serve brief\n"

        other_format = await controller.request("format", "project_memory_update", {
            "project_id": alpha, "revision": final["digest"],
            "mutation": {"format": 2, "op": "set", "kind": "brief", "content": "x"},
        })
        assert other_format["error"] == STALE
        missing = await controller.request("missing", "project_memory_update", {
            "project_id": "p_" + "0" * 32, "revision": final["digest"], "mutation": set_brief,
        })
        unknown_project = await controller.request("missing-read", "project_show", {"project_id": "p_" + "0" * 32})
        assert missing["error"]["code"] == unknown_project["error"]["code"] == -32602
        assert missing["error"]["data"] == unknown_project["error"]["data"]
        bad = await controller.request("bad", "project_memory_update", {
            "project_id": alpha, "revision": final["digest"], "mutation": {**set_brief, "file": "../brief.md"},
        })
        assert bad["error"]["data"] == {"code": "invalid_params"}
    finally:
        await contract.close()


@pytest.mark.asyncio
async def test_format_two_memory_compare_and_set(tmp_path: Path) -> None:
    _alpha, beta = await _off_loop(_projects, tmp_path, entry_memory=True)
    contract = await ServeContract.start(tmp_path, "websocket")
    try:
        controller = await _client(contract, [*FEATURES, "projects-memory-v2"])
        first = (await _memory(controller, beta))["digest"]
        set_brief = {"format": 2, "op": "set", "kind": "brief", "content": "serve brief"}
        written = (await controller.request("set", "project_memory_update", {
            "project_id": beta, "revision": first, "mutation": set_brief,
        }))["result"]
        after = await _memory(controller, beta)
        assert written == {"revision": after["digest"], "data": {"format": 2, "version_id": after["version_id"]}}
        assert "serve brief" in {kind["key"]: kind["rendered"] for kind in after["kinds"]}["brief"]

        replay = await controller.request("replay", "project_memory_update", {
            "project_id": beta, "revision": first, "mutation": set_brief | {"content": "lost"},
        })
        assert replay["error"] == STALE

        undone = (await controller.request("undo", "project_memory_update", {
            "project_id": beta, "revision": written["revision"], "mutation": {"format": 2, "op": "undo"},
        }))["result"]
        restored = await _memory(controller, beta)
        assert undone["revision"] == restored["digest"] != written["revision"]
        assert "serve brief" not in {kind["key"]: kind["rendered"] for kind in restored["kinds"]}["brief"]
        stale_undo = await controller.request("stale-undo", "project_memory_update", {
            "project_id": beta, "revision": written["revision"], "mutation": {"format": 2, "op": "undo"},
        })
        assert stale_undo["error"] == STALE
    finally:
        await contract.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("memory_format", [1, 2])
async def test_memory_set_with_a_secret_is_refused_and_writes_nothing(tmp_path: Path, memory_format: int) -> None:
    alpha, beta = await _off_loop(_projects, tmp_path, entry_memory=True)
    key = "sk-proj-" + "A1b2C3d4E5f6G7h8I9j0K1l2"
    project_id, mutation, features = (
        (alpha, {"format": 1, "op": "set", "file": "brief.md", "content": f"key {key}\n"}, FEATURES)
        if memory_format == 1
        else (beta, {"format": 2, "op": "set", "kind": "brief", "content": f"key {key}"}, [*FEATURES, "projects-memory-v2"])
    )
    contract = await ServeContract.start(tmp_path, "websocket")
    try:
        controller = await _client(contract, features)
        before = await _memory(controller, project_id)
        refused = await controller.request("secret", "project_memory_update", {
            "project_id": project_id, "revision": before["digest"], "mutation": mutation,
        })
        assert refused["error"]["code"] == -32602, refused
        assert refused["error"]["data"] == {"code": "invalid_params", "reason": "secret_refused"}
        assert key not in json.dumps(refused)
        shown = await controller.request("show", "project_show", {"project_id": project_id})
        assert shown["result"]["memory"]["digest"] == before["digest"]
        assert key not in json.dumps(shown)
    finally:
        await contract.close()


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.asyncio
async def test_inbox_send_is_the_user_send_and_idempotent(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch,
) -> None:
    alpha, beta = await _off_loop(_projects, tmp_path)
    monkeypatch.setenv("ZETA_TEST_SCRIPTED_PROVIDER", "1")
    server = replay_server(tmp_path / "home", transport, None)
    clients = []
    committed = asyncio.Event()
    unblock = Event()
    event_loop = asyncio.get_running_loop()
    sent_ids = []
    send = ProjectInbox.send_user_with_outcome

    def stop_after_commit(inbox, **kwargs):
        result = send(inbox, **kwargs)
        sent_ids.append(result[0])
        event_loop.call_soon_threadsafe(committed.set)
        assert unblock.wait(10), "timed out waiting for sender disconnect"
        return result

    async def connect():
        client = await await_or_fail(connect_replay(server, transport), what="inbox connection")
        clients.append(client)
        hello = await negotiate(client, replay=False, extra=(*FEATURES, "session_cwd"), protocol_version="1.2")
        assert "result" in hello, hello
        return client

    try:
        controller = await connect()
        message = {"kind": "question", "title": "From the web", "body": "hello", "delivery_id": "web-1"}
        detached = await controller.request(2, "project_inbox_send", {"project_id": alpha, "mutation": message})
        assert detached["error"]["code"] == -32003
        session_id = (await controller.request(3, "new_session", {
            "provider": "codex", "cwd": str((tmp_path / "alpha").resolve()),
        }))["result"]["session"]["session_id"]

        for forged in ({"mutation": {**message, "origin": "agent"}}, {"mutation": message, "from_session": "x"}):
            refused = await controller.request(4, "project_inbox_send", {"project_id": beta, **forged})
            assert refused["error"]["data"] == {"code": "invalid_params"}

        params = {"project_id": alpha, "mutation": message}
        sent = (await controller.request(5, "project_inbox_send", params))["result"]
        message_id = sent["data"]["message_id"]
        notice = await controller.event("project_inbox_notification")
        assert notice["data"]["message_ids"] == [message_id]
        assert (await controller.request(6, "project_inbox_send", params))["result"] == sent
        changed = await controller.request(7, "project_inbox_send", {
            "project_id": alpha, "mutation": {**message, "body": "different"},
        })
        assert changed["error"]["data"] == {"code": "invalid_params", "reason": "delivery_id_reused"}

        second = (await controller.request(8, "project_inbox_send", {
            "project_id": alpha, "mutation": {**message, "delivery_id": "web-2"},
        }))["result"]["data"]["message_id"]
        notice = await controller.event("project_inbox_notification")
        # The poller names every unclaimed message; the retry added no second copy of the first.
        assert sorted(notice["data"]["message_ids"]) == sorted([message_id, second])

        inbox = (await controller.request(9, "project_inbox", {"project_id": alpha}))["result"]
        rows = {row["id"]: row for row in inbox["messages"]}
        assert set(rows) == {message_id, second}
        assert rows[message_id]["from"] == {"project": alpha, "session": session_id}
        assert rows[message_id]["origin"] == "user" and rows[message_id]["body"] == "hello"
        # Same trust as `/inbox send` from the TUI: a user-origin message is an assignment.
        assert rows[message_id]["authority"] == "trusted_assignment"

        params = {"project_id": beta, "mutation": {**message, "delivery_id": "lost-response"}}
        with monkeypatch.context() as patch:
            # test-contract: seam pause a committed inbox send before its response
            patch.setattr(ProjectInbox, "send_user_with_outcome", stop_after_commit)
            await controller.send(10, "project_inbox_send", params)
            await await_or_fail(committed.wait(), what="durable inbox commit")
            await await_or_fail(controller.close(), what="drop inbox sender")
            unblock.set()
        committed_id, = sent_ids

        controller = await connect()
        retry = (await controller.request(11, "project_inbox_send", params))["result"]
        assert retry["data"] == {"message_id": committed_id, "pending": False}
        inbox = (await controller.request(12, "project_inbox", {"project_id": beta}))["result"]
        assert [row["id"] for row in inbox["messages"]] == [committed_id]
    finally:
        unblock.set()
        for client in clients:
            await await_or_fail(client.close(), what="close inbox client")
        await await_or_fail(server.close(), what="close inbox serve")


async def _in_process(server, features: list[str]):
    reader, writer = await _connect(server)
    hello = await _request(reader, writer, "hello", "hello", {"protocol_version": "1.2", "features": features})
    assert "result" in hello[-1], hello
    return reader, writer


@pytest.mark.asyncio
async def test_busy_registry_and_mid_call_migration_map_to_retryable_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    alpha, _beta = await _off_loop(_projects, tmp_path)
    server = _server(tmp_path)
    reader, writer = await _in_process(server, FEATURES)
    try:
        show = await _request(reader, writer, "show", "project_show", {"project_id": alpha})
        revision = show[-1]["result"]["memory"]["digest"]
        set_brief = {"format": 1, "op": "set", "file": "brief.md", "content": "serve brief\n"}
        params = {"project_id": alpha, "revision": revision, "mutation": set_brief}
        real_show, real_cas = ProjectRegistry.show_project, ProjectRegistry.compare_and_swap_memory

        def busy(*_args: object, **_kwargs: object):
            raise ProjectRegistryBusyError()

        # test-contract: seam another process holds the registry lock past its deadline
        monkeypatch.setattr(ProjectRegistry, "show_project", busy)
        blocked = await _request(reader, writer, "busy", "project_memory_update", params)
        assert blocked[-1]["error"]["code"] == -32004, blocked
        assert blocked[-1]["error"]["data"]["code"] == "operation_busy"
        monkeypatch.setattr(ProjectRegistry, "show_project", real_show)

        def migrate_then_write(self, project_id, **kwargs):
            self.activate_entry_memory(project_id)  # a 1 -> 2 migration lands after the format check
            return real_cas(self, project_id, **kwargs)

        # test-contract: seam force a migration between the format check and the owner's locked write
        monkeypatch.setattr(ProjectRegistry, "compare_and_swap_memory", migrate_then_write)
        raced = await _request(reader, writer, "raced", "project_memory_update", params)
        assert raced[-1]["error"] == STALE, raced
    finally:
        await close_serve(server, writer)


@pytest.mark.asyncio
async def test_inbox_send_saved_for_retry_tells_the_client_not_to_resend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _alpha, beta = await _off_loop(_projects, tmp_path)
    server = _server(tmp_path)
    reader, writer = await _in_process(server, [*FEATURES, "session_cwd"])
    try:
        created = await _request(reader, writer, "new", "new_session", {
            "provider": "codex", "cwd": str((tmp_path / "alpha").resolve()),
        })
        assert "result" in created[-1], created
        message = {"kind": "question", "title": "t", "body": "late", "delivery_id": "web-pending"}
        params = {"project_id": beta, "mutation": message}

        def unreachable(*_args: object, **_kwargs: object):
            raise OSError("recipient inbox is temporarily unavailable")

        with monkeypatch.context() as m:
            # test-contract: seam the recipient publish fails after the durable intent is saved
            m.setattr(ProjectInbox, "_publish_outgoing", unreachable)
            saved = await _request(reader, writer, "send", "project_inbox_send", params)
        assert saved[-1]["result"]["data"]["pending"] is True, saved
        message_id = saved[-1]["result"]["data"]["message_id"]

        retry = await _request(reader, writer, "retry", "project_inbox_send", params)
        assert retry[-1]["result"]["data"] == {"message_id": message_id, "pending": False}
        inbox = await _request(reader, writer, "inbox", "project_inbox", {"project_id": beta})
        assert [row["id"] for row in inbox[-1]["result"]["messages"]] == [message_id]
    finally:
        await close_serve(server, writer)


@pytest.mark.asyncio
async def test_inbox_resend_below_delegation_epoch_is_idempotent(tmp_path: Path) -> None:
    below = storage_format.FEATURE_EPOCHS[storage_format.StorageFeature.INBOX_DELEGATION] - 1
    _alpha, beta = await _off_loop(_projects, tmp_path, writer_epoch=below)
    server = _server(tmp_path)
    reader, writer = await _in_process(server, [*FEATURES, "session_cwd"])
    try:
        created = await _request(reader, writer, "new", "new_session", {
            "provider": "codex", "cwd": str((tmp_path / "alpha").resolve()),
        })
        assert "result" in created[-1], created
        params = {"project_id": beta, "mutation": {"kind": "question", "title": "t", "body": "b", "delivery_id": "web-old"}}

        sent = await _request(reader, writer, "send", "project_inbox_send", params)
        assert "result" in sent[-1], sent
        retry = await _request(reader, writer, "retry", "project_inbox_send", params)
        assert "result" in retry[-1], retry
        assert retry[-1]["result"] == sent[-1]["result"]
        inbox = await _request(reader, writer, "inbox", "project_inbox", {"project_id": beta})
        assert [row["id"] for row in inbox[-1]["result"]["messages"]] == [sent[-1]["result"]["data"]["message_id"]]
    finally:
        await close_serve(server, writer)
