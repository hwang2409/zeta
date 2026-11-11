"""Restart recovery of a delegated child approval and its parent end."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.support.fake_backend import FakeBackend
from zeta.core.session import SessionManager
from zeta.core.store import ConversationStore
from zeta.protocol.types import Message, MessageRole, ToolCall, ToolUseContent
from zeta.server.runtime import ServerRuntime


def _runtime(home: Path, cwd: Path) -> ServerRuntime:
    return ServerRuntime(
        home,
        cwd=cwd,
        provider="codex",
        backend_factory=lambda provider, model, home: (FakeBackend([]), model or "offline"),
    )


def _seed_delegated(home: Path, cwd: Path) -> str:
    opened = SessionManager(home).create(provider="codex", model="offline", cwd=cwd)
    root = opened.store
    call = ToolCall("call-c", "agent", {"prompt": "work", "description": "child"})
    root.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(call)]))
    child_instance_id = f"{root.session_id}:0"
    child_path = root.session_dir / "agents" / "0"
    root.register_agent_child(
        call,
        child_session_path=str(child_path),
        description="child",
        child_instance_id=child_instance_id,
    )
    # parent-side delegated request row: request and end share the wire id.
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


async def _resume_and_close(home: Path, cwd: Path, session_id: str) -> None:
    runtime = _runtime(home, cwd)
    await runtime.resume_session(session_id)
    await runtime.close()


def _parent_ends(home: Path, session_id: str) -> list[dict]:
    store = SessionManager(home).open(session_id).store
    try:
        return [
            entry.data
            for entry in store.replay()
            if entry.type == "approval_history"
            and entry.data.get("event") == "approval_end"
        ]
    finally:
        store.close()


def _child_resolution(home: Path, session_id: str) -> dict | None:
    root = SessionManager(home).open(session_id).store
    try:
        child = ConversationStore(
            root.session_dir / "agents",
            session_id="0",
            cwd=Path(root.cwd),
            _storage_lease=root._storage_lease,
        )
        try:
            return child.approval_resolution_data("call-c")
        finally:
            child.close()
    finally:
        root.close()


@pytest.mark.asyncio
async def test_restart_recovery_repairs_delegated_child_and_parent_end(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    session_id = await asyncio.to_thread(_seed_delegated, home, tmp_path)

    await _resume_and_close(home, tmp_path, session_id)

    # Fails on base: before this change recovery never resolved the child approval
    # nor wrote the parent end, so both assertions below are empty/None.
    ends = await asyncio.to_thread(_parent_ends, home, session_id)
    assert len(ends) == 1
    assert ends[0]["request_id"] == "wire-c"
    assert ends[0]["data"]["decision"] == "abort"
    assert ends[0]["data"]["reason"] == "server_restarted"

    child = await asyncio.to_thread(_child_resolution, home, session_id)
    assert child is not None
    assert child["decision"] == "abort"
    assert child["reason"] == "server_restarted"


@pytest.mark.asyncio
async def test_restart_recovery_delegated_repair_is_idempotent(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    session_id = await asyncio.to_thread(_seed_delegated, home, tmp_path)

    await _resume_and_close(home, tmp_path, session_id)
    await _resume_and_close(home, tmp_path, session_id)

    ends = await asyncio.to_thread(_parent_ends, home, session_id)
    # A second restart must not write a duplicate parent end.
    assert len(ends) == 1
