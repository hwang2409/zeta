from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.support.fake_backend import FakeBackend
from zeta.agent.conversation_channel import ConversationChannel
from zeta.core.session import SessionManager
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    Message,
    MessageOrigin,
    MessageRole,
    TextContent,
    ToolCall,
    ToolUseContent,
    with_message_origin,
)
from zeta.server.runtime import ServerRuntime


def _runtime(home: Path, cwd: Path) -> ServerRuntime:
    return ServerRuntime(
        home,
        cwd=cwd,
        provider="codex",
        backend_factory=lambda provider, model, home: (
            FakeBackend([]),
            model or "offline",
        ),
    )


def _seed_session(home: Path, cwd: Path, *, children: int = 0) -> str:
    opened = SessionManager(home).create(provider="codex", model="offline", cwd=cwd)
    store = opened.store
    calls = [
        ToolCall(
            f"call-{index}",
            "agent",
            {"prompt": "work", "description": f"child {index}"},
        )
        for index in range(children)
    ]
    if calls:
        store.append_message(
            Message(MessageRole.ASSISTANT, [ToolUseContent(call) for call in calls])
        )
    for index, call in enumerate(calls):
        store.register_agent_child(
            call,
            child_session_path=str(store.session_dir / "missing" / str(index)),
            description=f"child {index}",
            background=index != 0,
            child_instance_id=f"{store.session_id}:{index}",
        )
    session_id = store.session_id
    store.close()
    return session_id


@pytest.mark.asyncio
async def test_resume_large_root_without_children_does_not_reload_root(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    session_id = await asyncio.to_thread(_seed_session, home, tmp_path)

    def fill_root() -> Path:
        opened = SessionManager(home).open(session_id)
        for index in range(512):
            opened.store.append_message(
                with_message_origin(
                    Message(
                        MessageRole.USER,
                        [TextContent(f"{index}:" + "x" * 1024)],
                    ),
                    MessageOrigin.USER,
                )
            )
        root_path = opened.store.path
        opened.store.close()
        return root_path

    root_path = await asyncio.to_thread(fill_root)

    full_root_loads = 0
    original = ConversationStore._load_full

    def count_root_loads(store: ConversationStore) -> None:
        nonlocal full_root_loads
        if store.path == root_path:
            full_root_loads += 1
        original(store)

    runtime = _runtime(home, tmp_path)
    with patch.object(ConversationStore, "_load_full", count_root_loads):
        await runtime.resume_session(session_id)
    try:
        assert full_root_loads == 2
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_resume_recovery_runs_on_unpublished_store(tmp_path: Path) -> None:
    home = tmp_path / "home"
    session_id = await asyncio.to_thread(_seed_session, home, tmp_path, children=2)
    runtime = _runtime(home, tmp_path)
    sinks_ready = False
    publications = 0
    original_produce = ConversationChannel.produce_to

    async def checked_produce(self, target_instance_id, item):
        nonlocal publications
        assert sinks_ready
        publications += 1
        return await original_produce(self, target_instance_id, item)

    runtime.set_background_event_sink(lambda _session_id, _event: None)
    original_bind = runtime._bind_background_event_sink

    def bind(state):
        nonlocal sinks_ready
        original_bind(state)
        sinks_ready = True

    with (
        patch.object(runtime, "_bind_background_event_sink", side_effect=bind),
        patch.object(ConversationChannel, "produce_to", checked_produce),
    ):
        await runtime.resume_session(session_id)
    try:
        assert runtime.opened is not None
        assert runtime.opened.store.agent_children() == {}
        results = [
            message.tool_result
            for message in runtime.opened.store.messages()
            if message.tool_result is not None
        ]
        assert [result.tool_call_id for result in results] == ["call-0"]
        notifications = runtime.opened.store.agent_notifications(pending_only=False)
        assert [item.data["child_instance_id"] for item in notifications] == [
            f"{session_id}:1"
        ]
        assert publications == 1
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_resume_recovery_cancel_during_setup_is_safe(tmp_path: Path) -> None:
    home = tmp_path / "home"
    session_id = await asyncio.to_thread(_seed_session, home, tmp_path, children=2)
    runtime = _runtime(home, tmp_path)
    entered = threading.Event()
    release = threading.Event()

    from zeta.agent import background

    original_recover = background.recover_agent_children

    def gated_recover(loop):
        entered.set()
        assert release.wait(2)
        return original_recover(loop)

    with patch.object(background, "recover_agent_children", gated_recover):
        resume = asyncio.create_task(runtime.resume_session(session_id))
        assert await asyncio.to_thread(entered.wait, 2)
        resume.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await resume
        if runtime._setup_reapers:
            await asyncio.gather(*tuple(runtime._setup_reapers))

    await runtime.resume_session(session_id)
    try:
        assert runtime.opened is not None
        assert runtime.opened.store.agent_children() == {}
        notifications = runtime.opened.store.agent_notifications(pending_only=False)
        assert [item.data["child_instance_id"] for item in notifications] == [
            f"{session_id}:1"
        ]
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_resume_recovery_cancel_during_notification_publication_is_safe(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    session_id = await asyncio.to_thread(_seed_session, home, tmp_path, children=3)
    runtime = _runtime(home, tmp_path)
    entered = asyncio.Event()
    release = asyncio.Event()
    original_produce = ConversationChannel.produce_to

    async def gated_produce(self, target_instance_id, item):
        if not entered.is_set():
            entered.set()
            await release.wait()
        return await original_produce(self, target_instance_id, item)

    with patch.object(ConversationChannel, "produce_to", gated_produce):
        resume = asyncio.create_task(runtime.resume_session(session_id))
        await asyncio.wait_for(entered.wait(), 2)
        resume.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await resume

    await runtime.resume_session(session_id)
    try:
        assert runtime.opened is not None
        notifications = runtime.opened.store.agent_notifications(pending_only=False)
        assert [item.data["child_instance_id"] for item in notifications] == [
            f"{session_id}:1",
            f"{session_id}:2",
        ]
    finally:
        await runtime.close()
