"""A serve join cannot observe a session before restart recovery finishes.

With the session-swap join guard dropped, the only protection is ordering:
serve installs the resumed runtime (``self._state`` in server/runtime.py
``_replace``) only after ``_run_setup`` -> ``compose_runtime`` -> recovery
returns. This test holds recovery in its worker thread during a real serve
resume and proves a concurrent join cannot read the mid-recovery store.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from tests.support.fake_backend import FakeBackend
from tests.support.serve_close import close_serve
from tests.support.server_client import request
from tests.support.sockets import short_socket_path as _socket_path
from zeta.core.session import SessionManager
from zeta.core.store import ConversationStore
from zeta.protocol.types import Message, MessageRole, ToolCall, ToolUseContent
from zeta.server import ZetaServer

pytest_plugins = ("tests.support.socket_fixtures",)


def _seed_delegated(home: Path, cwd: Path) -> str:
    opened = SessionManager(home).create(provider="codex", model="offline", cwd=cwd)
    root = opened.store
    call = ToolCall("call-c", "agent", {"prompt": "work", "description": "child"})
    root.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(call)]))
    child_instance_id = f"{root.session_id}:0"
    root.register_agent_child(
        call,
        child_session_path=str(root.session_dir / "agents" / "0"),
        description="child",
        child_instance_id=child_instance_id,
    )
    root.append_approval_history(
        {
            "event": "approval_request",
            "request_id": "wire-c",
            "tool_call": call.to_dict(),
            "delegated": True,
            "agent_instance_id": child_instance_id,
        }
    )
    child = ConversationStore(
        root.session_dir / "agents",
        session_id="0",
        cwd=cwd,
        _storage_lease=root._storage_lease,
    )
    child.activate_runtime_lease()
    child.append_approval_request("call-c", ToolCall("call-c", "bash", {"command": "true"}))
    child.close()
    session_id = root.session_id
    root.close()
    return session_id


@pytest.mark.asyncio
async def test_join_during_restart_recovery_cannot_see_pre_recovery_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    session_id = await asyncio.to_thread(_seed_delegated, home, tmp_path)

    server = ZetaServer(
        provider="codex",
        home=home,
        socket_path=_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (FakeBackend([]), model or "offline"),
    )
    await server.start()

    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()
    import zeta.runtime.approval_recovery as recovery_module

    original = recovery_module.repair_delegated_child

    def gated_repair(*args: object, **kwargs: object) -> object:
        # Runs in the compose worker thread; hold recovery mid-repair.
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(10), "recovery gate was never released"
        return original(*args, **kwargs)

    monkeypatch.setattr(recovery_module, "repair_delegated_child", gated_repair)

    controller = observer = None
    try:
        reader, writer = await asyncio.open_unix_connection(str(server.socket_path))
        controller = writer
        await request(reader, writer, 1, "hello", {"protocol_version": "1.2"})
        resume = asyncio.create_task(
            request(reader, writer, 2, "resume", {"session_id": session_id})
        )

        await asyncio.wait_for(entered.wait(), 10)

        # A concurrent observer join while recovery is held: the runtime is not
        # installed yet, so the session is not active and the join is refused.
        o_reader, o_writer = await asyncio.open_unix_connection(str(server.socket_path))
        observer = o_writer
        await request(
            o_reader,
            o_writer,
            1,
            "hello",
            {
                "protocol_version": "1.2",
                "features": ["serve_observers", "session_replay_seq", "approval_restart_reason"],
                "role": "observer",
            },
        )
        mid = (
            await request(
                o_reader,
                o_writer,
                2,
                "session_history_and_subscribe",
                {"session_id": session_id},
            )
        )[-1]
        assert "result" not in mid, f"join saw the store mid-recovery: {mid!r}"
        assert mid["error"]["message"] == "server has no active session"

        release.set()
        resumed = (await asyncio.wait_for(resume, 10))[-1]
        assert resumed["result"]["session"]["session_id"] == session_id

        # After recovery the join succeeds and the delegated request is a durable
        # abort end (server_restarted) with no open request left on the wire.
        joined = (
            await request(
                o_reader,
                o_writer,
                3,
                "session_history_and_subscribe",
                {"session_id": session_id},
            )
        )[-1]
        rows = joined["result"]["replay"]["messages"]
        ends = [r for r in rows if r.get("event") == "approval_end" and r.get("request_id") == "wire-c"]
        assert len(ends) == 1
        assert ends[0]["data"] == {"decision": "abort", "reason": "server_restarted"}
        assert not [
            r
            for r in rows
            if r.get("event") == "approval_request" and r.get("request_id") == "wire-c"
        ]
    finally:
        if observer is not None:
            await close_serve(server, observer)
        if controller is not None:
            await close_serve(server, controller)
        await server.close()
