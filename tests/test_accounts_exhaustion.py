from __future__ import annotations

import asyncio
import base64
import inspect
import json
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import zeta.providers.codex as codex_module
import zeta.providers.transport as transport_module
import zeta.server.accounts as server_accounts
import zeta.tui.slash_handlers as tui_slash_handlers
from tests.support.credentials import named_account_credentials
from tests.support.fake_backend import FakeBackend, ScriptedTurn
from zeta.accounts import (
    AccountId,
    AccountStore,
    AccountUsage,
    SessionAccountContext,
)
from zeta.accounts.exhaustion import (
    SubscriptionLimit,
    classify_subscription_limit,
    recovery_alternatives,
)
from zeta.accounts.usage import (
    UsageSnapshot,
    UsageState,
    UsageWindow,
    format_usage_lines,
    read_usage,
    record_subscription_limit,
)
from zeta.core.loop import AgentLoop
from zeta.core.session import SessionManager
from zeta.core.store import ConversationStore
from zeta.oauth import OAuthTokens
from zeta.protocol.types import MessageOrigin, MessageRole, StreamEventType, TextContent
from zeta.providers.codex import CodexBackend
from zeta.providers.retry_policy import ProviderRetryBudget
from zeta.skills import SkillCatalog
from zeta.tui import TUIApp

FIXTURE = Path(__file__).parent / "fixtures/usage/codex_responses_429_usage_limit_reached.json"
NOW = datetime(2026, 10, 22, 16, 0, tzinfo=UTC)
ACCOUNT = AccountId("codex", "pro")


def _failure(status: int, body: object) -> SimpleNamespace:
    return SimpleNamespace(status_code=status, response_body=json.dumps(body).encode())


def _state(percent: float, *, age: timedelta = timedelta(), provider: str = "codex") -> UsageState:
    kind = "primary" if provider == "codex" else "five_hour"
    return UsageState(
        UsageSnapshot(NOW - age, "usage_endpoint", (UsageWindow(kind, percent, 10_080, NOW + timedelta(days=1)),)),
        "now" if age <= timedelta(minutes=15) else "stale",
    )


def test_codex_usage_limit_reached_fixture_is_classified() -> None:
    payload = json.loads(FIXTURE.read_text())
    result = classify_subscription_limit("codex", ACCOUNT, _failure(429, {"error": payload["error"]}))
    assert result == SubscriptionLimit(ACCOUNT, datetime.fromtimestamp(1793082708, UTC), 10080)


def test_unknown_429_and_changed_schema_are_not_exhaustion() -> None:
    assert classify_subscription_limit("codex", ACCOUNT, _failure(429, {"error": {"type": "other"}})) is None
    assert classify_subscription_limit("codex", ACCOUNT, _failure(429, {"error": {"type": "usage_limit_reached", "resets_at": "soon", "limit_window_minutes": 10080}})) is None
    assert classify_subscription_limit("codex", ACCOUNT, _failure(500, {"error": {"type": "usage_limit_reached", "resets_at": 1793082708, "limit_window_minutes": 10080}})) is None


def test_anthropic_429_is_not_classified_without_fixture() -> None:
    assert classify_subscription_limit("anthropic", AccountId("anthropic", "pro"), _failure(429, {"error": {"type": "usage_limit_reached", "resets_at": 1793082708, "limit_window_minutes": 10080}})) is None


def test_classified_limit_is_not_retried_and_other_retries_unchanged() -> None:
    exhausted = _failure(429, json.loads(FIXTURE.read_text()))
    exhausted.subscription_limit = classify_subscription_limit("codex", ACCOUNT, exhausted)
    budget = ProviderRetryBudget()
    budget.start_attempt("test")
    assert budget.plan(exhausted, owner="test") is None
    assert budget.plan(SimpleNamespace(status_code=429), owner="test") is not None


def test_limit_marks_account_known_exhausted_until_reset(tmp_path: Path) -> None:
    named_account_credentials(tmp_path, "codex", "pro").save(
        OAuthTokens("access", "refresh", 9_999_999_999)
    )
    store = AccountStore(tmp_path)
    limit = SubscriptionLimit(ACCOUNT, NOW + timedelta(hours=1), 10080)
    state = limit.usage_state(NOW)
    assert state.snapshot is not None
    record_subscription_limit(store, ACCOUNT, state.snapshot, NOW)

    persisted = read_usage(store, ACCOUNT, now=NOW)
    assert persisted.known_exhausted(NOW)
    assert format_usage_lines(persisted)[0].startswith("7d 100% · resets ")
    assert not persisted.known_exhausted(NOW + timedelta(hours=1, seconds=1))


def test_offers_switch_only_with_exactly_one_fresh_unexhausted_alternative() -> None:
    a, b, c = AccountId("codex", "a"), AccountId("codex", "b"), AccountId("codex", "c")
    assert recovery_alternatives(a, {a: _state(100), b: _state(20)}, NOW) == (b,)
    assert recovery_alternatives(a, {a: _state(100)}, NOW) == ()
    assert recovery_alternatives(a, {a: _state(100), b: _state(20), c: _state(30)}, NOW) == ()
    assert recovery_alternatives(a, {a: _state(100), b: _state(20, age=timedelta(minutes=16))}, NOW) == ()
    assert recovery_alternatives(a, {a: _state(100), b: _state(100)}, NOW) == ()


def test_switch_never_replays_failed_request() -> None:
    # Recovery returns a selection only; the caller must wait for new user input.
    other = AccountId("codex", "other")
    assert recovery_alternatives(ACCOUNT, {ACCOUNT: _state(100), other: _state(1)}, NOW) == (other,)


def test_child_limit_failure_never_switches_parent() -> None:
    limit = SubscriptionLimit(ACCOUNT, NOW + timedelta(hours=1), 10080)
    assert limit.recovery_event(frozen=True, alternatives=(AccountId("codex", "other"),))["action"] == "none"


def _codex_token(account_id: str) -> str:
    def encode(value: object) -> str:
        return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")

    return ".".join(
        (
            encode({"alg": "none"}),
            encode(
                {
                    "exp": 4_000_000_000,
                    "https://api.openai.com/auth": {"chatgpt_account_id": account_id},
                }
            ),
            encode({"signature": "fixture"}),
        )
    )


def _account_context(
    tmp_path: Path,
) -> tuple[SessionAccountContext, object, AccountId, AccountId]:
    home = tmp_path / "home"
    first = AccountId("codex", "first")
    second = AccountId("codex", "second")
    for account in (first, second):
        named_account_credentials(home, account.provider, account.name).save(
            OAuthTokens(_codex_token(account.name), "refresh", 4_000_000_000)
        )
    store = AccountStore(home)
    store.set_default(first)
    manager = SessionManager(home)
    opened = manager.create(provider="codex", model="gpt-5.6-luna", cwd=tmp_path)
    context = SessionAccountContext.load(home, manager, opened.metadata)
    return context, opened, first, second


def _switch_loop(backend: object) -> SimpleNamespace:
    loop = SimpleNamespace(backend=backend)
    loop.current_backend_state = lambda: loop.backend
    loop.prepare_backend_state = lambda prepared: prepared
    loop.apply_backend_state = lambda prepared: setattr(loop, "backend", prepared)
    loop.set_backend = lambda prepared: setattr(loop, "backend", prepared)
    return loop


def _assert_switch_state(
    context: SessionAccountContext,
    account: AccountId,
    backend: object,
    loop: SimpleNamespace,
    *,
    sequence: int,
    boundary: int = 0,
) -> None:
    assert context.manager is not None
    persisted = context.manager.read_metadata(context.metadata.session_id)
    assert persisted.selected_accounts == {"codex": str(account)}
    assert persisted.account_switch_sequences == ({"codex": sequence} if sequence else {})
    assert persisted.native_context_boundaries == {"codex": boundary}
    assert context.metadata.to_dict() == persisted.to_dict()
    assert context.snapshot()["codex"] == account
    assert context.switch_sequence == sequence
    assert context.active_provider == "codex"
    assert loop.backend is backend


@pytest.mark.asyncio
async def test_switch_cancelled_during_prepare_changes_nothing(
    tmp_path: Path,
) -> None:
    context, opened, first, second = await asyncio.to_thread(
        _account_context, tmp_path
    )
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    first_backend = object()
    loop = _switch_loop(first_backend)

    def build_backend(_account: AccountId) -> object:
        entered.set()
        assert release.wait(5)
        finished.set()
        return object()

    task = asyncio.create_task(
        context.switch_active(
            loop,
            second,
            boundary=0,
            provider="codex",
            build_backend=build_backend,
        )
    )
    assert await asyncio.to_thread(entered.wait, 1)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)
    assert await asyncio.to_thread(finished.wait, 1)

    _assert_switch_state(context, first, first_backend, loop, sequence=0)
    opened.store.close()


@pytest.mark.asyncio
async def test_switch_apply_is_infallible_no_split_when_backend_build_fails(
    tmp_path: Path,
) -> None:
    context, opened, first, second = await asyncio.to_thread(
        _account_context, tmp_path
    )
    first_backend = object()
    loop = _switch_loop(first_backend)

    second_backend = object()
    loop.prepare_backend_state = lambda _backend: (_ for _ in ()).throw(
        RuntimeError("build failed")
    )

    with pytest.raises(RuntimeError, match="build failed"):
        await context.switch_active(
            loop,
            second,
            boundary=3,
            provider="codex",
            build_backend=lambda _account: second_backend,
        )

    _assert_switch_state(context, first, first_backend, loop, sequence=0)
    opened.store.close()


@pytest.mark.asyncio
async def test_switch_persist_failure_keeps_old_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, opened, first, second = await asyncio.to_thread(
        _account_context, tmp_path
    )
    first_backend = object()
    second_backend = object()
    loop = _switch_loop(first_backend)

    assert context.manager is not None
    monkeypatch.setattr(
        context.manager,
        "_write_unlocked",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("persist failed")),
    )
    await context.switch_active(
        loop,
        second,
        boundary=3,
        provider="codex",
        build_backend=lambda _account: second_backend,
    )

    _assert_switch_state(context, first, first_backend, loop, sequence=0)
    opened.store.close()


@pytest.mark.asyncio
async def test_switch_persist_raises_reread_new_applies_new(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, opened, _first, second = await asyncio.to_thread(
        _account_context, tmp_path
    )
    first_backend = object()
    second_backend = object()
    loop = _switch_loop(first_backend)
    assert context.manager is not None
    persist = context.manager._write_unlocked

    def publish_then_fail(*args: object, **kwargs: object) -> None:
        persist(*args, **kwargs)
        raise OSError("published then failed")

    monkeypatch.setattr(context.manager, "_write_unlocked", publish_then_fail)
    await context.switch_active(
        loop,
        second,
        boundary=3,
        provider="codex",
        build_backend=lambda _account: second_backend,
    )

    _assert_switch_state(
        context, second, second_backend, loop, sequence=1, boundary=3
    )
    opened.store.close()


@pytest.mark.asyncio
async def test_switch_persist_raises_reread_fails_marks_unknown_and_blocks_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, opened, first, second = await asyncio.to_thread(
        _account_context, tmp_path
    )
    first_backend = FakeBackend([])
    second_backend = FakeBackend([])
    loop = AgentLoop(first_backend, opened.store, skill_catalog=SkillCatalog.empty())
    loop.manager = context.manager
    loop.session_metadata = context.metadata
    loop.account_context = context
    assert context.manager is not None

    monkeypatch.setattr(
        context.manager,
        "_write_unlocked",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("persist failed")),
    )
    read = context.manager._read
    reads = 0

    def fail_reread(*args: object, **kwargs: object):
        nonlocal reads
        reads += 1
        if reads >= 2:
            raise OSError("reread failed")
        return read(*args, **kwargs)

    monkeypatch.setattr(context.manager, "_read", fail_reread)
    await context.switch_active(
        loop,
        second,
        boundary=3,
        provider="codex",
        build_backend=lambda _account: second_backend,
    )

    assert context.account_state_unknown
    events = [
        event async for event in loop.run_turn("hello", origin=MessageOrigin.USER)
    ]
    assert any(
        event.type is StreamEventType.ACCOUNT_STATE_UNKNOWN for event in events
    )
    error = next(event for event in events if event.type is StreamEventType.ERROR)
    assert error.error is not None and "/accounts" in error.error.message
    assert first_backend.calls == []
    assert second_backend.calls == []
    assert loop.backend is first_backend
    assert context.snapshot()["codex"] == first
    opened.store.close()


@pytest.mark.asyncio
async def test_unknown_account_state_rederives_from_disk_on_next_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, opened, _first, second = await asyncio.to_thread(
        _account_context, tmp_path
    )
    first_backend = FakeBackend([])
    second_backend = FakeBackend([ScriptedTurn(content=[TextContent("ok")])])
    loop = AgentLoop(first_backend, opened.store, skill_catalog=SkillCatalog.empty())
    loop.manager = context.manager
    loop.session_metadata = context.metadata
    loop.account_context = context
    assert context.manager is not None
    persist = context.manager._write_unlocked
    read = context.manager._read
    reads = 0

    def publish_then_fail(*args: object, **kwargs: object) -> None:
        persist(*args, **kwargs)
        raise OSError("published then failed")

    def fail_once(*args: object, **kwargs: object):
        nonlocal reads
        reads += 1
        if reads == 2:
            raise OSError("reread failed")
        return read(*args, **kwargs)

    monkeypatch.setattr(context.manager, "_write_unlocked", publish_then_fail)
    monkeypatch.setattr(context.manager, "_read", fail_once)
    await context.switch_active(
        loop,
        second,
        boundary=3,
        provider="codex",
        build_backend=lambda _account: second_backend,
    )
    assert context.account_state_unknown

    events = [
        event async for event in loop.run_turn("hello", origin=MessageOrigin.USER)
    ]

    assert any(
        event.type is StreamEventType.ACCOUNT_STATE_UNKNOWN for event in events
    )
    assert len(second_backend.calls) == 1
    assert not context.account_state_unknown
    _assert_switch_state(context, second, second_backend, loop, sequence=1, boundary=3)
    opened.store.close()


@pytest.mark.asyncio
async def test_switch_cancel_during_commit_converges_then_reraises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, opened, _first, second = await asyncio.to_thread(
        _account_context, tmp_path
    )
    entered = threading.Event()
    release = threading.Event()
    first_backend = object()
    second_backend = object()
    loop = _switch_loop(first_backend)
    assert context.manager is not None
    persist = context.manager._write_unlocked

    def blocked_persist(*args: object, **kwargs: object) -> None:
        entered.set()
        assert release.wait(5)
        persist(*args, **kwargs)

    monkeypatch.setattr(context.manager, "_write_unlocked", blocked_persist)
    task = asyncio.create_task(
        context.switch_active(
            loop,
            second,
            boundary=0,
            provider="codex",
            build_backend=lambda _account: second_backend,
        )
    )
    assert await asyncio.to_thread(entered.wait, 1)
    assert loop.backend is first_backend
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)

    _assert_switch_state(
        context, second, second_backend, loop, sequence=1, boundary=1
    )
    opened.store.close()


@pytest.mark.asyncio
async def test_overlapping_switches_are_serialized(tmp_path: Path) -> None:
    context, opened, first, second = await asyncio.to_thread(
        _account_context, tmp_path
    )
    first_entered = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()
    loop = _switch_loop(object())
    backends = {first: object(), second: object()}

    def build_backend(account: AccountId) -> object:
        if account == first:
            first_entered.set()
            assert release_first.wait(5)
        else:
            second_entered.set()
        return backends[account]

    first_task = asyncio.create_task(
        context.switch_active(
            loop,
            first,
            boundary=1,
            provider="codex",
            build_backend=build_backend,
        )
    )
    assert await asyncio.to_thread(first_entered.wait, 1)
    second_task = asyncio.create_task(
        context.switch_active(
            loop,
            second,
            boundary=2,
            provider="codex",
            build_backend=build_backend,
        )
    )
    await asyncio.to_thread(second_entered.wait, 0.1)
    release_first.set()
    await asyncio.wait_for(asyncio.gather(first_task, second_task), 5)

    persisted = context.manager.read_metadata(context.metadata.session_id)
    assert persisted.selected_accounts == {"codex": str(second)}
    assert persisted.account_switch_sequences == {"codex": 2}
    assert context.metadata.selected_accounts == persisted.selected_accounts
    assert context.snapshot()["codex"] == second
    assert context.switch_sequence == 2
    assert context.active_provider == "codex"
    assert loop.backend is backends[second]
    opened.store.close()


@pytest.mark.asyncio
async def test_account_switch_never_replays_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, opened, first, second = await asyncio.to_thread(
        _account_context, tmp_path
    )

    def fail_replay(_store: ConversationStore) -> list[object]:
        raise AssertionError("account switching must not replay the transcript")

    monkeypatch.setattr(ConversationStore, "replay", fail_replay)
    monkeypatch.setattr(
        tui_slash_handlers,
        "build_backend",
        lambda *_args, **_kwargs: (SimpleNamespace(model="offline"), "codex"),
    )
    monkeypatch.setattr(
        server_accounts,
        "build_backend",
        lambda *_args, **_kwargs: (SimpleNamespace(model="offline"), "codex"),
    )
    loop = SimpleNamespace(
        account_context=context,
        backend=SimpleNamespace(model="offline"),
        context_assembler=SimpleNamespace(token_budget=100_000),
        store=opened.store,
    )
    loop.current_backend_state = lambda: loop.backend
    loop.prepare_backend_state = lambda backend: backend
    loop.apply_backend_state = lambda backend: setattr(loop, "backend", backend)
    app = SimpleNamespace(
        _accounts_picker=SimpleNamespace(highlighted=second),
        _account_switch_task=None,
        loop=loop,
        provider="codex",
        model="offline",
        _dismiss_accounts_picker=lambda: None,
        _print_system=lambda _message: None,
    )

    task = TUIApp.accounts_picker_select(app)
    assert task is not None
    await task

    async def idle() -> None:
        return None

    response = await server_accounts.request(
        "account_switch",
        SimpleNamespace(loop=loop, metadata=context.metadata),
        idle,
        {"account": str(first)},
    )
    assert response == {
        "account": str(first),
        "replay": False,
        "account_state": "ready",
    }
    assert context.snapshot()["codex"] == first
    opened.store.close()


@pytest.mark.asyncio
async def test_codex_limit_end_to_end_emits_account_limit_once_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, opened, first, second = await asyncio.to_thread(
        _account_context, tmp_path
    )
    now = datetime.now(UTC)
    snapshot = UsageSnapshot(
        now,
        "usage_endpoint",
        (UsageWindow("primary", 25, 10_080, now + timedelta(days=1)),),
    )
    AccountUsage(context.store)._write(
        second, snapshot, now, None, "succeeded", finished=now
    )
    payload = json.loads(FIXTURE.read_text())
    payload.pop("_fixture")
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(429, json=payload, request=request)

    async def fail_retry_sleep(_delay: float) -> None:
        raise AssertionError("classified account limit must not wait to retry")

    retry_provider_completion = transport_module.retry_provider_completion

    def completion_without_retry_sleep(*args: object, **kwargs: object):
        kwargs["sleep"] = fail_retry_sleep
        return retry_provider_completion(*args, **kwargs)

    monkeypatch.setattr(
        codex_module, "retry_provider_completion", completion_without_retry_sleep
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = CodexBackend(
        client=client,
        token_store=named_account_credentials(context.home, "codex", "first"),
        base_url="https://test.invalid/codex/responses",
    )
    loop = AgentLoop(backend, opened.store, skill_catalog=SkillCatalog.empty())
    loop.manager = context.manager
    loop.session_metadata = context.metadata
    loop.account_context = context
    try:
        events = [
            event async for event in loop.run_turn("hello", origin=MessageOrigin.USER)
        ]
        assert requests == 1
        assert not any(
            isinstance(block, TextContent) and block.text
            for message in opened.store.messages()
            if message.role is MessageRole.ASSISTANT
            for block in message.content
        )
        types = [event.type for event in events]
        assert types.count(StreamEventType.ACCOUNT_LIMIT) == 1
        assert types.index(StreamEventType.ACCOUNT_LIMIT) < types.index(
            StreamEventType.ERROR
        )
        assert types.index(StreamEventType.ACCOUNT_LIMIT) < types.index(
            StreamEventType.AGENT_END
        )
        limit_event = next(
            event for event in events if event.type is StreamEventType.ACCOUNT_LIMIT
        )
        assert limit_event.data["alternative"] == str(second)
        assert limit_event.data["action"] == "switch"
        persisted = read_usage(context.store, first)
        assert persisted.snapshot is not None
        assert persisted.snapshot.windows == (
            UsageWindow(
                "primary",
                100.0,
                payload["error"]["limit_window_minutes"],
                datetime.fromtimestamp(payload["error"]["resets_at"], UTC),
            ),
        )

        boundary = opened.store.replay()[-1].seq
        await context.switch_active(
            loop,
            second,
            boundary=boundary,
            provider="codex",
            build_backend=lambda _account: loop.backend,
        )
        assert requests == 1
        assert await asyncio.to_thread(context.resolve, "codex") == second
    finally:
        await client.aclose()
        opened.store.close()


@pytest.mark.asyncio
async def test_limit_recovery_does_not_block_event_loop_while_usage_lock_held(
    tmp_path: Path,
) -> None:
    context, opened, first, _second = await asyncio.to_thread(
        _account_context, tmp_path
    )
    entered = threading.Event()
    release = threading.Event()

    def hold_usage_lock() -> None:
        with context.store.usage_guard(first):
            entered.set()
            assert release.wait(timeout=5)

    holder = threading.Thread(target=hold_usage_lock)
    holder.start()
    assert await asyncio.to_thread(entered.wait, 1)

    payload = json.loads(FIXTURE.read_text())
    failure = _failure(429, {"error": payload["error"]})

    async def recover() -> object:
        result = context.subscription_limit_event(failure)
        return await result if inspect.isawaitable(result) else result

    recovery = asyncio.create_task(recover())

    def release_later() -> None:
        time.sleep(0.25)
        release.set()

    releaser = threading.Thread(target=release_later)
    releaser.start()
    ticks = 0
    while not release.is_set():
        await asyncio.sleep(0.01)
        if not release.is_set():
            ticks += 1
    result = await asyncio.wait_for(recovery, 5)
    holder.join(timeout=1)
    releaser.join(timeout=1)
    try:
        assert ticks >= 3
        assert result is not None
        persisted = read_usage(context.store, first)
        assert persisted.snapshot is not None
        assert persisted.snapshot.windows[0].used_percent == 100
    finally:
        release.set()
        opened.store.close()
