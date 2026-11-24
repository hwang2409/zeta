from __future__ import annotations

import base64
import json
from contextlib import AsyncExitStack
from pathlib import Path

import pytest

from tests.support.owned import await_or_fail
from tests.support.replay_transport import (
    ReplayClient,
    Transport,
    connect_replay,
    negotiate,
    replay_server,
)
from tests.support.server_backend import ServerFakeBackend
from tests.test_server_content_read import _PNG

_READS = (
    "status", "content_read", "model_catalog", "session_settings", "delivery_status", "session_grants",
    "skill_catalog", "skill_read", "agent_profile_catalog", "agent_profile_read", "mcp_server_list",
    "session_history_read",
)
_FEATURES = (
    "content_read", "session_history_fidelity", "session_cwd", "delivery_id", "session_grants",
    "library_read_v1", "mcp_management_v1", "session_history_read_v1", "run_work", "child_run_control_v1",
)
_READ_ONLY = (
    "send", "send_images", "steer", "approve", "deny", "abort", "set_settings", "set_compaction",
    "new_session", "resume", "rename_session", "delete_session", "switch_branch", "fork_message",
    "background_task_cancel", "child_run_cancel", "revoke_session_grants", "mcp_auth_cancel",
)
_NO_VIEW = {"code": -32003, "message": "session is not active"}


def _project(tmp_path: Path, name: str) -> Path:
    root = tmp_path / name
    for folder in ("skills", "agents"):
        catalog = root / ".zeta" / folder
        catalog.mkdir(parents=True)
        (catalog / "view.md").write_text(f"---\nname: view\ndescription: {name} {folder}\n---\nBody from {name}.\n")
    (root / ".zeta" / "mcp.json").write_text(json.dumps({"servers": {
        name: {"transport": "streamable-http", "url": "http://127.0.0.1:1", "enabled": False,
               "headers": {"Authorization": "synthetic-disk-value"}},
    }}))
    return root


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.parametrize("mode", ("multi", "legacy"))
@pytest.mark.parametrize("method", (*_READS, "write_refusals"))
@pytest.mark.asyncio
async def test_observer_view_table(tmp_path: Path, transport: Transport, mode: str, method: str) -> None:
    project_a, project_b = (_project(tmp_path, name) for name in ("alpha", "beta"))
    server = replay_server(tmp_path / "home", transport, lambda _, model, __: (ServerFakeBackend(delay=0), model or "offline"))
    async with AsyncExitStack() as cleanup:
        async def close_server() -> None:
            await await_or_fail(server.close(), what="close view server")

        cleanup.push_async_callback(close_server)
        await await_or_fail(server.start(), what="start view server")

        async def connect(*, observer: bool = False) -> ReplayClient:
            client = await await_or_fail(connect_replay(server, transport), what="connect view client")

            async def close_client() -> None:
                await await_or_fail(client.close(), what="close view socket")

            cleanup.push_async_callback(close_client)
            features = (*_FEATURES, "multi_controller") if mode == "multi" and not observer else _FEATURES
            hello = await negotiate(client, observer=observer, extra=features, protocol_version="1.2")
            assert "result" in hello, hello
            return client

        async def new_session(client: ReplayClient, request_id: int, project: Path, model: str) -> str:
            reply = await client.request(request_id, "new_session", {"cwd": str(project), "model": model})
            assert "result" in reply, reply
            return reply["result"]["session"]["session_id"]

        a = await connect()
        a_model, b_model = "gpt-5.4", "gpt-5.6-luna"
        archived = await new_session(a, 2, project_a, a_model) if method == "session_history_read" else None
        if archived is not None:
            assert "result" in await a.request(3, "send", {"text": "archived alpha prompt"})
            await a.event("agent_end")
        sid = await new_session(a, 4, project_a, a_model)
        if mode == "multi":
            b = await connect()
            other = await new_session(b, 2, project_b, b_model)
            assert other != sid

        params: dict[str, object] = {}
        if method == "content_read":
            image = {"name": "view.png", "mime_type": "image/png", "data": base64.b64encode(_PNG).decode()}
            sent = await a.request(5, "send_images", {"session_id": sid, "text": "image from alpha", "images": [image]})
            assert sent.get("result", {}).get("accepted"), sent
            await a.event("agent_end")
        elif method == "delivery_status":
            params = {"delivery_id": "controller-alpha-delivery"}
            assert "result" in await a.request(5, "send", {"text": "alpha delivery", **params})
            await a.event("agent_end")
        elif method in {"skill_catalog", "agent_profile_catalog", "mcp_server_list"}:
            params = {"scope": "project"}
        elif method in {"skill_read", "agent_profile_read"}:
            params = {"scope": "project", "name": "view"}
        elif method == "session_history_read":
            params = {"session_id": archived}

        observer = await connect(observer=True)
        joined = await observer.request(2, "session_history_and_subscribe", {"session_id": sid})
        assert "result" in joined, joined
        if method == "write_refusals":
            for request_id, denied in enumerate(_READ_ONLY, start=3):
                reply = await observer.request(request_id, denied, {"session_id": sid})
                data = {"code": "observer_read_only", "method": denied}
                if denied in {"child_run_cancel", "mcp_auth_cancel"}:
                    data["reason"] = "observer_read_only"
                assert reply.get("error") == {
                    "code": -32008, "message": "method is unavailable to read-only observers", "data": data,
                }, reply
            return

        if method == "content_read":
            handle, = [part["handle"] for row in joined["result"]["messages"]
                       for part in row.get("content", ()) if part["type"] == "attachment"]
            params = {"handle": handle}
        reply = await observer.request(3, method, params)
        assert "result" in reply, reply
        result = reply["result"]
        if method == "status":
            assert (result["session"] or {}).get("session_id") == sid, reply
        elif method == "session_settings":
            assert result == {"model": a_model, "approval_mode": "ask"}, reply
        elif method == "content_read":
            assert result["data"] == base64.b64encode(_PNG).decode(), reply
        elif method == "delivery_status":
            owned = (await a.request(6, method, params))["result"]
            assert owned["status"] != "unknown", owned
            assert result == ({**params, "status": "unknown"} if mode == "multi" else owned), reply
        elif method == "session_grants":
            assert result["session_id"] == sid, reply
        elif method == "model_catalog":
            # The current catalog is static; an absent view below still distinguishes resolution.
            assert result == (await a.request(6, method, params))["result"], reply
        elif method in {"skill_catalog", "agent_profile_catalog"}:
            item, = result["data"]["items"]
            assert item["description"] == ("alpha skills" if method == "skill_catalog" else "alpha agents"), reply
        elif method in {"skill_read", "agent_profile_read"}:
            assert result["data"]["body"].strip() == "Body from alpha.", reply
        elif method == "mcp_server_list":
            item, = result["data"]["items"]
            assert item["name"] == "alpha", reply
            assert item["config"]["headers"] == {"Authorization": "<redacted>"}, reply
            assert "synthetic-disk-value" not in json.dumps(result), reply
        elif method == "session_history_read":
            assert "archived alpha prompt" in json.dumps(result), reply

        if mode == "multi":
            unjoined = await connect(observer=True)
            denied = await unjoined.request(2, method, params)
            assert denied.get("error") == _NO_VIEW, denied
