from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from tests.support.credentials import named_account_credentials
from zeta.accounts import AccountId, SessionAccountContext
from zeta.agent import runner as agent_runner
from zeta.core.session import SessionManager
from zeta.core.store import ConversationStore
from zeta.oauth import OAuthTokens
from zeta.protocol.types import Message, StreamEvent, ToolSchema
from zeta.providers.ollama import OllamaBackend
from zeta.providers.transport import CloseOutcome, close_transport_resources
from zeta.runtime.loop import AgentLoop
from zeta.runtime.loop import agent as agent_loop_module
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


class _GatedResponse(_Response):
    def __init__(self, started: asyncio.Event, release: asyncio.Event) -> None:
        self.started = started
        self.release = release

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        self.started.set()
        await _within(self.release.wait())
        yield b'{"message":{"content":"ok"},"done":true}\n'


class _GatedClient(_RecordingClient):
    def __init__(self, started: asyncio.Event, release: asyncio.Event) -> None:
        super().__init__()
        self.started = started
        self.release = release
        self.closed = asyncio.Event()

    def stream(self, *_args: object, **_kwargs: object) -> _Stream:
        self.request_count += 1
        return _Stream(_GatedResponse(self.started, self.release))

    async def aclose(self) -> None:
        await _within(super().aclose())
        self.closed.set()


class _ObservedOllamaBackend(OllamaBackend):
    def __init__(self, close_entered: asyncio.Event) -> None:
        self.close_entered = close_entered
        super().__init__()

    async def aclose(self, *, report_failure=None) -> CloseOutcome:
        self.close_entered.set()
        return await _within(super().aclose(report_failure=report_failure))


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
async def test_cancelled_replacement_close_still_closes_after_active_borrow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    close_entered = asyncio.Event()
    client = _GatedClient(started, release)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: client)
    old = _ObservedOllamaBackend(close_entered)
    replacement = _RecordingBackend()
    loop = AgentLoop(
        old,
        ConversationStore(tmp_path),
        skill_catalog=SkillCatalog.empty(),
    )
    completion = asyncio.create_task(_consume(old))
    try:
        await _within(started.wait())
        retired = loop.apply_backend_state(loop.prepare_backend_state(replacement))
        assert retired is old
        closing = asyncio.create_task(loop.close_backend(old, stage="replacement"))
        await _within(close_entered.wait())
        closing.cancel()
        release.set()
        await _within(completion)
        with pytest.raises(asyncio.CancelledError):
            await _within(closing)
        await _within(client.closed.wait())
        assert client.close_count == 1
        await _within(loop.close())
        assert replacement.client.close_count == 1
    finally:
        release.set()
        completion.cancel()
        await _within(asyncio.gather(completion, return_exceptions=True))
        await _within(loop.close())
        loop.store.close()


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
    resolved, error = agent_runner.resolve_child_backend(parent_loop, None)
    assert error is None
    assert resolved is child
    child_loop = AgentLoop(
        child,
        ConversationStore(tmp_path / "child"),
        skill_catalog=SkillCatalog.empty(),
        agent_depth=1,
        agent_instance_id="child",
        background_owner=parent_loop._background_owner,
    )
    try:
        await _within(child_loop.close())
        assert child.client.close_count == 1
        assert parent.client.close_count == 0
        await _within(parent_loop.close())
        assert parent.client.close_count == 1
    finally:
        await _within(child_loop.close())
        await _within(parent_loop.close())
        child_loop.store.close()
        parent_loop.store.close()


@pytest.mark.asyncio
async def test_runtime_reports_backend_close_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    failure = RuntimeError("close failed")
    backend = _RecordingBackend()
    backend.client.close_error = failure
    reports: list[tuple[BaseException, str, str]] = []

    def report(error: BaseException, *, stage: str, provider: str, **_kwargs) -> None:
        reports.append((error, stage, provider))

    # test-contract: seam capture the p26 reporter handoff without filesystem polling
    monkeypatch.setattr(agent_loop_module, "_schedule_cleanup_failure", report)
    loop = AgentLoop(
        backend,
        ConversationStore(tmp_path),
        skill_catalog=SkillCatalog.empty(),
    )
    try:
        await _within(loop.close())
        assert reports == [(failure, "completion_backend_runtime_close", "ollama")]
    finally:
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
        assert backend.connection_setup_count == 0
        assert client.close_count == 0
    finally:
        await _within(loop.close())
        loop.store.close()
