"""Restart recovery of a delegated child approval and its parent end."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
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
from zeta.storage_format import FEATURE_EPOCHS, StorageFeature

pytest_plugins = ("tests.support.socket_fixtures",)

_RESTART_EPOCH = FEATURE_EPOCHS[StorageFeature.APPROVAL_RESTART_REASON]


async def _restart_and_read_ends(
    home: Path, cwd: Path, session_id: str
) -> tuple[list[dict], dict]:
    """Drive restart over the real ``zeta serve`` socket and read the root
    approval ends back from ``session_history``.

    The ``resume`` RPC runs the same recovery a crash-restart runs; reading the
    ends back over the wire proves the delegated end landed on the root session
    history, not an intermediate child store.
    """
    server = ZetaServer(
        provider="codex",
        home=home,
        socket_path=_socket_path(cwd),
        backend_factory=lambda provider, model, home: (FakeBackend([]), model or "offline"),
    )
    await server.start()
    writer = None
    try:
        reader, writer = await asyncio.open_unix_connection(str(server.socket_path))
        await request(
            reader,
            writer,
            1,
            "hello",
            {
                "protocol_version": "1.2",
                "features": ["session_replay_seq", "approval_restart_reason"],
            },
        )
        resumed = (
            await request(reader, writer, 2, "resume", {"session_id": session_id})
        )[-1]
        history = (
            await request(
                reader, writer, 3, "session_history", {"session_id": session_id}
            )
        )[-1]["result"]
        ends = [
            row for row in history["messages"] if row.get("event") == "approval_end"
        ]
        return ends, resumed
    finally:
        if writer is not None:
            await close_serve(server, writer)
        await server.close()


def _open_root_child(
    home: Path, cwd: Path, epoch: int | None = None
) -> tuple[ConversationStore, ConversationStore, ToolCall]:
    """Open a root session with one delegated child and the root-side request row.

    Serve writes every delegated ``approval_request``/``approval_end`` row to the
    root store keyed by the child's ``agent_instance_id``. The returned stores are
    left open for the caller to finish the child state and close.
    """
    if epoch is not None:
        # Pin the home manifest before first open so the observed epoch is exact.
        home.mkdir(parents=True, exist_ok=True)
        (home / "storage-format.json").write_text(
            json.dumps({"writer_epoch": epoch}) + "\n"
        )
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
    return root, child, call


def _seed_delegated(home: Path, cwd: Path, epoch: int | None = None) -> str:
    root, child, _call = _open_root_child(home, cwd, epoch)
    child.close()
    session_id = root.session_id
    root.close()
    return session_id


def _seed_child_allow(home: Path, cwd: Path) -> str:
    """The child already resolved ``allow`` before the crash; no parent end yet."""
    root, child, _call = _open_root_child(home, cwd)
    child.resolve_approval("call-c", "allow")
    child.close()
    session_id = root.session_id
    root.close()
    return session_id


def _seed_existing_parent_end(home: Path, cwd: Path) -> str:
    """A fully-closed delegated approval: the parent end is already present."""
    root, child, call = _open_root_child(home, cwd)
    child.resolve_approval("call-c", "abort", reason="server_restarted")
    root.append_approval_history(
        {
            "event": "approval_end",
            "request_id": "wire-c",
            "tool_call": call.to_dict(),
            "data": {"decision": "abort", "reason": "server_restarted"},
            "reason": "server_restarted",
        }
    )
    child.close()
    session_id = root.session_id
    root.close()
    return session_id


def _seed_grandchild(home: Path, cwd: Path) -> str:
    """orchestrator -> lead -> worker: the worker (a grandchild) has an unresolved
    delegated approval whose request row lives on the ROOT store."""
    opened = SessionManager(home).create(provider="codex", model="offline", cwd=cwd)
    root = opened.store
    lead_call = ToolCall("call-lead", "agent", {"prompt": "lead", "description": "lead"})
    root.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(lead_call)]))
    lead_id = f"{root.session_id}:0"
    lead_path = root.session_dir / "agents" / "0"
    root.register_agent_child(
        lead_call,
        child_session_path=str(lead_path),
        description="lead",
        child_instance_id=lead_id,
    )
    lead = ConversationStore(
        root.session_dir / "agents",
        session_id="0",
        cwd=cwd,
        _storage_lease=root._storage_lease,
    )
    lead.activate_runtime_lease()
    worker_call = ToolCall("call-worker", "agent", {"prompt": "w", "description": "worker"})
    lead.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(worker_call)]))
    worker_id = f"{lead_id}:0"
    worker_path = lead.session_dir / "agents" / "0"
    lead.register_agent_child(
        worker_call,
        child_session_path=str(worker_path),
        description="worker",
        child_instance_id=worker_id,
    )
    worker = ConversationStore(
        lead.session_dir / "agents",
        session_id="0",
        cwd=cwd,
        _storage_lease=root._storage_lease,
    )
    worker.activate_runtime_lease()
    req = ToolCall("req-g", "bash", {"command": "true"})
    worker.append_approval_request("req-g", req)
    # Serve routes the grandchild's delegated request to the ROOT store, keyed by
    # the worker's own agent_instance_id.
    root.append_approval_history(
        {
            "event": "approval_request",
            "request_id": "wire-g",
            "tool_call": req.to_dict(),
            "delegated": True,
            "agent_instance_id": worker_id,
        }
    )
    worker.close()
    lead.close()
    session_id = root.session_id
    root.close()
    return session_id


def _resolution_at(
    home: Path, session_id: str, agents_rel: str, request_id: str
) -> dict | None:
    root = SessionManager(home).open(session_id).store
    try:
        store = ConversationStore(
            root.session_dir / agents_rel,
            session_id="0",
            cwd=Path(root.cwd),
            _storage_lease=root._storage_lease,
        )
        try:
            return store.approval_resolution_now(request_id)
        finally:
            store.close()
    finally:
        root.close()


@pytest.mark.asyncio
async def test_restart_recovery_repairs_delegated_child_and_parent_end(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    session_id = await asyncio.to_thread(_seed_delegated, home, tmp_path)

    ends, _resumed = await _restart_and_read_ends(home, tmp_path, session_id)

    # Fails on base: before this change recovery never resolved the child approval
    # nor wrote the parent end, so both assertions below are empty/None.
    assert len(ends) == 1
    assert ends[0]["request_id"] == "wire-c"
    assert ends[0]["data"]["decision"] == "abort"
    assert ends[0]["data"]["reason"] == "server_restarted"

    child = await asyncio.to_thread(_resolution_at, home, session_id, "agents", "call-c")
    assert child is not None
    assert child["decision"] == "abort"
    assert child["reason"] == "server_restarted"


@dataclass(frozen=True)
class _EndCase:
    seed: Callable[[Path, Path], str]
    check: Callable[[list[dict], object], None]


def _check_grandchild(ends: list[dict], resolution: object) -> None:
    grand = [e for e in ends if e["request_id"] == "wire-g"]
    # Fails on the old base (74e71775): the grandchild end was repaired against
    # the intermediate child store, so the root session history never got it and
    # session_history over the socket returns no wire-g end.
    assert len(grand) == 1, f"root grandchild end rows: {grand}"
    assert grand[0]["data"]["decision"] == "abort"
    assert grand[0]["data"]["reason"] == "server_restarted"
    assert resolution is not None
    assert resolution["decision"] == "abort"
    assert resolution["reason"] == "server_restarted"


def _check_child_allow(ends: list[dict], resolution: object) -> None:
    child = [e for e in ends if e["request_id"] == "wire-c"]
    assert len(child) == 1
    # A user allow is copied, never rewritten to a restart abort.
    assert child[0]["data"]["decision"] == "allow"
    assert "reason" not in child[0]["data"]


def _check_no_duplicate_end(ends: list[dict], resolution: object) -> None:
    child = [e for e in ends if e["request_id"] == "wire-c"]
    # Zero new rows: with the idempotency guard removed, a second end is written.
    assert len(child) == 1, f"duplicate parent ends: {child}"


@pytest.mark.parametrize(
    ("name", "case"),
    [
        ("grandchild", _EndCase(_seed_grandchild, _check_grandchild)),
        ("preserves_child_allow", _EndCase(_seed_child_allow, _check_child_allow)),
        ("existing_end_not_duplicated", _EndCase(_seed_existing_parent_end, _check_no_duplicate_end)),
    ],
)
@pytest.mark.asyncio
async def test_restart_recovery_delegated_end_states(
    tmp_path: Path, name: str, case: _EndCase
) -> None:
    home = tmp_path / "home"
    session_id = await asyncio.to_thread(case.seed, home, tmp_path)

    ends, _resumed = await _restart_and_read_ends(home, tmp_path, session_id)

    resolution = (
        await asyncio.to_thread(
            _resolution_at, home, session_id, "agents/0/agents", "req-g"
        )
        if name == "grandchild"
        else None
    )
    case.check(ends, resolution)


@pytest.mark.asyncio
async def test_restart_recovery_low_epoch_writes_reasonless_parent_end(
    tmp_path: Path,
) -> None:
    # Below the feature epoch, the parent end is a reasonless abort that older
    # readers accept; the child resolution carries no reason either.
    home = tmp_path / "home"
    session_id = await asyncio.to_thread(
        _seed_delegated, home, tmp_path, _RESTART_EPOCH - 1
    )

    ends, _resumed = await _restart_and_read_ends(home, tmp_path, session_id)
    assert len(ends) == 1
    assert ends[0]["data"]["decision"] == "abort"
    assert "reason" not in ends[0]["data"]
    assert "reason" not in ends[0]

    child = await asyncio.to_thread(_resolution_at, home, session_id, "agents", "call-c")
    assert child is not None and child["decision"] == "abort"
    assert "reason" not in child
