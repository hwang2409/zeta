from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from tests.support.credentials import named_account_credentials
from zeta.accounts import AccountId, SessionAccountContext
from zeta.agent import runner as agent_runner
from zeta.agent.presets import GENERAL_PRESET
from zeta.core.session import SessionManager
from zeta.core.store import ConversationStore
from zeta.oauth import OAuthTokens
from zeta.protocol.types import Message, StreamEvent, ToolSchema
from zeta.providers.ollama import OllamaBackend
from zeta.providers.transport import CloseOutcome, close_transport_resources
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog


async def _within(awaitable):
    return await asyncio.wait_for(awaitable, 10)


async def _consume(backend) -> list[StreamEvent]:
    return [event async for event in backend.complete([], [])]


class _Response:
    status_code = 200

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        yield b'{"message":{"content":"ok"},"done":true}\n'


class _Stream:
    def __init__(self, response: _Response | None = None) -> None:
        self.response = response or _Response()

    async def __aenter__(self) -> _Response:
        return self.response

    async def __aexit__(self, *args: object) -> None:
        return None


class _RecordingClient:
    def __init__(self, *, close_error: BaseException | None = None, **_kwargs: object) -> None:
        self.close_count = 0
        self.close_error = close_error
        self.request_count = 0

    def stream(self, *_args: object, **_kwargs: object) -> _Stream:
        self.request_count += 1
        return _Stream()

    async def aclose(self) -> None:
        self.close_count += 1
        if self.close_error is not None:
            raise self.close_error


class _RecordingBackend:
    provider = "ollama"
    model = "qwen3:4b"
    connection_setup_count = 1

    def __init__(self) -> None:
        self.client = _RecordingClient()
        self.retired = False

    def complete(
        self, _messages: list[Message], _tool_schemas: list[ToolSchema]
    ) -> AsyncIterator[StreamEvent]:
        async def empty() -> AsyncIterator[StreamEvent]:
            if False:
                yield StreamEvent  # pragma: no cover

        return empty()

    def retire(self) -> None:
        self.retired = True

    async def aclose(self, *, report_failure=None) -> CloseOutcome:
        return await _within(
            close_transport_resources(
                self.client.aclose(), report_failure=report_failure
            )
        )


@pytest.mark.asyncio
async def test_account_replacement_closes_both_backend_owners(tmp_path: Path) -> None:
    home = tmp_path / "home"
    await _within(
        asyncio.to_thread(
            named_account_credentials(home, "codex", "work").save,
            OAuthTokens("access", "refresh", 9_999_999_999),
        )
    )
    manager = SessionManager(home)
    opened = await _within(
        asyncio.to_thread(
            manager.create,
            provider="codex",
            model="gpt-5.6-sol",
            cwd=tmp_path,
            compaction_budget=8_192,
        )
    )
    old = _RecordingBackend()
    replacement = _RecordingBackend()
    loop = AgentLoop(old, opened.store, skill_catalog=SkillCatalog.empty())
    context = await _within(
        asyncio.to_thread(
            SessionAccountContext.load, home, manager, opened.metadata
        )
    )
    try:
        await _within(
            context.switch_active(
                loop,
                AccountId("codex", "work"),
                boundary=0,
                provider="codex",
                build_backend=lambda _account: replacement,
            )
        )
        assert old.client.close_count == 1
        assert replacement.client.close_count == 0

        await _within(loop.close())
        assert old.client.close_count == 1
        assert replacement.client.close_count == 1
    finally:
        await _within(loop.close())
        opened.store.close()


@pytest.mark.asyncio
async def test_owned_backend_reuses_one_client_and_runtime_closes_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clients: list[_RecordingClient] = []

    def build_client(**kwargs: object) -> _RecordingClient:
        client = _RecordingClient(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(httpx, "AsyncClient", build_client)
    backend = OllamaBackend()
    loop = AgentLoop(
        backend,
        ConversationStore(tmp_path),
        skill_catalog=SkillCatalog.empty(),
    )
    try:
        for _ in range(3):
            await _within(_consume(backend))
        assert len(clients) == 1
        assert backend.connection_setup_count == 1
        assert clients[0].request_count == 3
        assert clients[0].close_count == 0

        await _within(loop.close())
        await _within(loop.close())
        assert clients[0].close_count == 1
    finally:
        await _within(loop.close())
        loop.store.close()


@pytest.mark.asyncio
async def test_replaced_backend_reference_rejects_completion_before_transport(
    tmp_path: Path,
) -> None:
    retired_client = _RecordingClient()
    retired = OllamaBackend(client=retired_client)
    replacement = _RecordingBackend()
    loop = AgentLoop(
        retired,
        ConversationStore(tmp_path),
        skill_catalog=SkillCatalog.empty(),
    )
    try:
        loop.apply_backend_state(loop.prepare_backend_state(replacement))

        with pytest.raises(RuntimeError, match="^completion backend is retired$"):
            await _within(_consume(retired))
        assert retired_client.request_count == 0
    finally:
        await _within(loop.close())
        loop.store.close()


@pytest.mark.asyncio
async def test_double_cancelled_account_replacement_closes_only_retired_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    await _within(
        asyncio.to_thread(
            named_account_credentials(home, "codex", "work").save,
            OAuthTokens("access", "refresh", 9_999_999_999),
        )
    )
    manager = SessionManager(home)
    opened = await _within(
        asyncio.to_thread(
            manager.create,
            provider="codex",
            model="gpt-5.6-sol",
            cwd=tmp_path,
            compaction_budget=8_192,
        )
    )
    old = _RecordingBackend()
    replacement = _RecordingBackend()
    loop = AgentLoop(old, opened.store, skill_catalog=SkillCatalog.empty())
    context = await _within(
        asyncio.to_thread(
            SessionAccountContext.load, home, manager, opened.metadata
        )
    )
    replacement_installed = threading.Event()
    release_worker = threading.Event()
    original_apply_on_loop = context._apply_on_loop

    def hold_worker_after_apply(*args):
        result = original_apply_on_loop(*args)
        replacement_installed.set()
        if not release_worker.wait(10):
            raise TimeoutError("account replacement worker was not released")
        return result

    # test-contract: seam hold the worker after the real event-loop apply callback
    monkeypatch.setattr(
        SessionAccountContext, "_apply_on_loop", staticmethod(hold_worker_after_apply)
    )
    switch = asyncio.create_task(
        context.switch_active(
            loop,
            AccountId("codex", "work"),
            boundary=0,
            provider="codex",
            build_backend=lambda _account: replacement,
        )
    )
    try:
        assert await _within(asyncio.to_thread(replacement_installed.wait, 10))
        assert loop.backend is replacement
        switch.cancel()
        first_cancellation_processed = asyncio.Event()
        asyncio.get_running_loop().call_soon(first_cancellation_processed.set)
        await _within(first_cancellation_processed.wait())
        switch.cancel()
        release_worker.set()
        with pytest.raises(asyncio.CancelledError):
            await _within(switch)

        assert old.client.close_count == 1
        assert replacement.client.close_count == 0
        assert loop.backend is replacement
    finally:
        release_worker.set()
        if not switch.done():
            switch.cancel()
        await _within(asyncio.gather(switch, return_exceptions=True))
        await _within(loop.close())
        opened.store.close()


@pytest.mark.asyncio
async def test_child_registry_clone_gets_its_own_backend_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = _RecordingBackend()
    child = _RecordingBackend()
    parent_loop = AgentLoop(
        parent,
        ConversationStore(tmp_path / "parent"),
        skill_catalog=SkillCatalog.empty(),
    )
    monkeypatch.setattr(
        agent_runner,
        "build_backend",
        lambda *_args, **_kwargs: (child, child.model),
    )
    child_store = ConversationStore(tmp_path / "child")
    child_loop = None
    try:
        resolved, error = agent_runner.resolve_child_backend(parent_loop, None)
        assert error is None
        child_loop, policy = await _within(
            agent_runner._build_child_loop(
                parent_loop,
                child_backend=resolved,
                child_store=child_store,
                child_instance_id="child",
                child_depth=1,
                preset=GENERAL_PRESET,
                model=None,
                cwd_override=None,
                follow_up_loop=False,
                description="test child",
                child_account_context=None,
            )
        )
        assert policy is None
        assert child_loop.backend is child
        assert child_loop.tool_registry is not parent_loop.tool_registry
        await _within(child_loop.close())
        assert child.client.close_count == 1
        assert parent.client.close_count == 0
        await _within(parent_loop.close())
        assert parent.client.close_count == 1
    finally:
        if child_loop is not None:
            await _within(child_loop.close())
        await _within(parent_loop.close())
        child_store.close()
        parent_loop.store.close()


@pytest.mark.asyncio
async def test_runtime_reports_backend_close_failure(tmp_path: Path) -> None:
    failure = RuntimeError("close failed")
    backend = _RecordingBackend()
    backend.client.close_error = failure
    contexts: list[dict[str, object]] = []
    event_loop = asyncio.get_running_loop()
    previous_handler = event_loop.get_exception_handler()

    def capture_report(_loop, context: dict[str, object]) -> None:
        contexts.append(context)

    event_loop.set_exception_handler(capture_report)
    loop = AgentLoop(
        backend,
        ConversationStore(tmp_path),
        skill_catalog=SkillCatalog.empty(),
    )
    try:
        await _within(loop.close())
        callbacks_flushed = event_loop.create_future()
        event_loop.call_soon(callbacks_flushed.set_result, None)
        await _within(callbacks_flushed)
        assert contexts == [
            {
                "exception_type": "builtins.RuntimeError",
                "stage": "completion_backend_runtime_close",
                "index": 0,
                "primary_outcome": "success",
            }
        ]
    finally:
        event_loop.set_exception_handler(previous_handler)
        await _within(loop.close())
        loop.store.close()


@pytest.mark.asyncio
async def test_injected_client_is_never_closed_by_runtime(tmp_path: Path) -> None:
    client = _RecordingClient()
    backend = OllamaBackend(client=client)
    loop = AgentLoop(
        backend,
        ConversationStore(tmp_path),
        skill_catalog=SkillCatalog.empty(),
    )
    try:
        await _within(_consume(backend))
        await _within(loop.close())
        completion_error = None
        try:
            await _within(_consume(backend))
        except RuntimeError as error:
            completion_error = error
        assert (str(completion_error), client.close_count) == (
            "completion backend is retired",
            0,
        )
    finally:
        await _within(loop.close())
        loop.store.close()
