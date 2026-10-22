from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
import json
import multiprocessing
import os
import stat
import threading
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

import zeta.accounts.store as account_store
import zeta.cli.main as cli_main
import zeta.codex as codex_module
import zeta.oauth as oauth_module
import zeta.providers.anthropic as anthropic_module
import zeta.server.login as server_login
from tests.support.credentials import named_account_credentials
from tests.support.fake_backend import FakeBackend, ScriptedTurn
from zeta.accounts import (
    AccountError,
    AccountId,
    AccountLoopThreadLockError,
    AccountStore,
    SessionAccountContext,
)
from zeta.agent.runner import resolve_child_backend
from zeta.automations.models import parse_job
from zeta.automations.store import SQLiteStore
from zeta.core.loop import AgentLoop
from zeta.core.session import SessionManager
from zeta.core.store import ConversationStore
from zeta.oauth import OAuthTokens
from zeta.protocol.types import (
    Message,
    MessageOrigin,
    MessageRole,
    RedactedThinkingContent,
    TextContent,
    ThinkingContent,
)
from zeta.providers.factory import build_backend, credential_store
from zeta.runtime import unattended
from zeta.server.login import Logins
from zeta.skills import SkillCatalog
from zeta.tools import websearch
from zeta.tools.websearch import codex as websearch_codex
from zeta.tui import bootstrap as tui_bootstrap
from zeta.tui import models as tui_models

_SENTINEL = "token-text-must-not-escape"


@pytest.mark.asyncio
async def test_credential_store_while_async_guard_holds_lifecycle_lock_does_not_block_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    credential = named_account_credentials(tmp_path, "codex")
    await asyncio.to_thread(credential.save, tokens())
    monkeypatch.setattr(account_store, "OAUTH_REFRESH_TIMEOUT_SECONDS", 0.05)
    ticks = 0
    stop = asyncio.Event()

    async def ticker() -> None:
        nonlocal ticks
        while not stop.is_set():
            ticks += 1
            await asyncio.sleep(0)

    ticker_task = asyncio.create_task(ticker())
    try:
        async with credential._async_guard(registry=False):
            with pytest.raises(AccountError):
                await asyncio.wait_for(
                    asyncio.to_thread(credential_store, "codex", home=tmp_path),
                    timeout=0.25,
                )
            assert ticks > 1
    finally:
        stop.set()
        await ticker_task


def test_resolve_account_does_not_migrate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy = _write_legacy(tmp_path, "codex")
    store = AccountStore(tmp_path)
    monkeypatch.setattr(store, "migrate", Mock(side_effect=AssertionError("migrated")))

    assert store.resolve_account("codex") == AccountId("codex", "default")
    assert legacy.exists()

    calls = 0
    original_migrate = AccountStore.migrate

    def migrate_once(self: AccountStore) -> None:
        nonlocal calls
        calls += 1
        original_migrate(self)

    monkeypatch.setattr(AccountStore, "migrate", migrate_once)
    monkeypatch.setattr(tui_bootstrap, "env_home", lambda: tmp_path)
    sentinel = object()
    monkeypatch.setattr(
        tui_bootstrap, "_create_app_with_root", lambda *_args: sentinel
    )

    assert tui_bootstrap.create_app(SimpleNamespace(no_session=False)) is sentinel
    assert calls == 1
    assert not legacy.exists()


@pytest.mark.asyncio
async def test_blocking_account_lock_on_loop_thread_raises(tmp_path: Path) -> None:
    store = AccountStore(tmp_path)
    with (
        store._home() as home_fd,
        pytest.raises(AccountLoopThreadLockError, match="event-loop thread"),
        store._lock_at(home_fd, ".accounts.lock"),
    ):
        pass


def tokens() -> OAuthTokens:
    return OAuthTokens(_SENTINEL, "refresh-secret", 9_999_999_999.0)


def _jwt_token(claims: dict[str, object]) -> str:
    def segment(value: dict[str, object]) -> str:
        return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")

    return ".".join((segment({"alg": "none"}), segment(claims), "signature"))


def test_no_writable_credential_store_outside_account_credentials() -> None:
    assert not hasattr(AccountStore, "ensure_" + "account")
    storage_methods = {"save", "access_token"}
    driver_methods = {"read", *storage_methods}
    assert not any(
        inspect.isclass(value) and storage_methods <= set(dir(value))
        for value in vars(oauth_module).values()
    )
    for module, driver_name in (
        (anthropic_module, "AnthropicCredentialDriver"),
        (codex_module, "CodexCredentialDriver"),
    ):
        driver = getattr(module, driver_name)
        assert not driver_methods & set(dir(driver))
        assert not any(isinstance(value, Path) for value in vars(driver()).values())


def test_missing_codex_login_raises_login_required_error(tmp_path: Path) -> None:
    credential = named_account_credentials(tmp_path, "codex")

    async def access() -> str:
        async with httpx.AsyncClient() as client:
            return await credential.access_token(client)

    with pytest.raises(codex_module.CodexLoginRequiredError):
        asyncio.run(access())


def test_missing_anthropic_login_keeps_auth_error_type(tmp_path: Path) -> None:
    credential = named_account_credentials(tmp_path, "anthropic")

    async def access() -> str:
        async with httpx.AsyncClient() as client:
            return await credential.access_token(client)

    with pytest.raises(anthropic_module.AnthropicAuthError):
        asyncio.run(access())


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_provider_driver_refresh_through_account_credentials_keeps_headers_and_rotation(
    provider: str, tmp_path: Path
) -> None:
    account_provider = "anthropic" if provider == "claude" else provider
    old = OAuthTokens("old-access", "old-refresh", 1.0)
    named_account_credentials(tmp_path, account_provider).save(old)
    credential = credential_store(provider, home=tmp_path)
    assert credential is not None
    credential._driver.token_url = "https://refresh.test/token"
    requests: list[httpx.Request] = []

    def refresh(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
            },
        )

    async def run() -> str:
        async with httpx.AsyncClient(transport=httpx.MockTransport(refresh)) as client:
            return await credential.refresh_token(client)

    assert asyncio.run(run()) == "new-access"
    request = requests.pop()
    assert request.headers["accept"] == "application/json"
    if provider == "claude":
        assert request.headers["content-type"] == "application/json"
        assert json.loads(request.content) == {
            "grant_type": "refresh_token",
            "client_id": anthropic_module.CLIENT_ID,
            "refresh_token": "old-refresh",
        }
    else:
        assert request.headers["content-type"] == "application/x-www-form-urlencoded"
        assert request.content.decode() == (
            "grant_type=refresh_token&refresh_token=old-refresh&client_id="
            + codex_module.CODEX_CLIENT_ID
        )
    persisted = credential.read()
    assert persisted is not None
    assert persisted.refresh_token == "new-refresh"


def test_anthropic_email_address_captured_on_refresh(tmp_path: Path) -> None:
    credential = named_account_credentials(tmp_path, "anthropic")
    credential.save(OAuthTokens("old-access", "old-refresh", 1.0))
    credential._driver.token_url = "https://refresh.test/token"

    def refresh(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
                "account": {"email_address": "claude@example.com"},
                "organization": {"name": "Acme"},
            },
        )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(refresh)) as client:
            await credential.refresh_token(client)

    asyncio.run(run())
    persisted = credential.read()
    assert persisted is not None
    assert persisted.email == "claude@example.com"
    assert persisted.workspace == "Acme"


def _write_legacy(home: Path, provider: str) -> Path:
    path = home / f"{provider}-oauth.json"
    path.write_text(json.dumps(tokens().to_dict()))
    return path


def _migration_worker(home: str, start: multiprocessing.synchronize.Event) -> None:
    start.wait()
    AccountStore(home).migrate()


def _login_lock_worker(
    home: str,
    ready: multiprocessing.synchronize.Event,
    release: multiprocessing.synchronize.Event,
) -> None:
    with AccountStore(home).provider_login_lock("anthropic"):
        ready.set()
        release.wait()


@pytest.mark.parametrize("provider", ["anthropic", "codex"])
def test_migration_happy_path(provider: str, tmp_path: Path) -> None:
    legacy = _write_legacy(tmp_path, provider)

    store = AccountStore(tmp_path)
    store.migrate()

    destination = store.credential_path(AccountId(provider, "default"))
    assert json.loads(destination.read_text()) == tokens().to_dict()
    assert not legacy.exists()
    assert store.default(provider) == AccountId(provider, "default")


def test_migration_happy_path_anthropic(tmp_path: Path) -> None:
    test_migration_happy_path("anthropic", tmp_path)


def test_migration_happy_path_codex(tmp_path: Path) -> None:
    test_migration_happy_path("codex", tmp_path)


@pytest.mark.parametrize(
    "step",
    [
        "accounts_directory",
        "provider_directory",
        "account_directory",
        "credential_file",
        "credential_directory_sync",
        "credential_verification",
        "registry_file",
        "registry_directory_sync",
        "legacy_unlink",
        "legacy_directory_sync",
    ],
)
def test_migration_crash_at_each_step_keeps_a_working_login(
    step: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy = _write_legacy(tmp_path, "anthropic")
    counts: dict[str, int] = {}

    patched_owner = account_store

    def wrap(name: str, predicate, owner=account_store):
        nonlocal patched_owner
        patched_owner = owner
        original = getattr(owner, name)

        def crashing(*args, **kwargs):
            result = original(*args, **kwargs)
            counts[name] = counts.get(name, 0) + 1
            if predicate(args, counts[name]):
                raise SystemExit("simulated process death")
            return result

        monkeypatch.setattr(owner, name, crashing)
        return original

    if step == "accounts_directory":
        original = wrap(
            "_mkdir_child_durable", lambda args, count: args[1] == "accounts"
        )
    elif step == "provider_directory":
        original = wrap(
            "_mkdir_child_durable",
            lambda args, count: args[1] == "anthropic" and count == 5,
        )
    elif step == "account_directory":
        original = wrap(
            "_mkdir_child_durable",
            lambda args, count: args[1] == "default" and count == 6,
        )
    elif step == "credential_file":
        original = wrap("_atomic_publish", lambda args, count: args[1] == "oauth.json")
    elif step == "credential_directory_sync":
        original = wrap("_write_token_bytes", lambda args, count: True)
    elif step == "credential_verification":
        original = wrap("_read_tokens", lambda args, count: count == 2)
    elif step == "registry_file":
        original = wrap(
            "_atomic_publish", lambda args, count: args[1] == "accounts.json"
        )
    elif step == "registry_directory_sync":
        original = wrap("_write_registry", lambda args, count: True, AccountStore)
    elif step == "legacy_unlink":
        original = wrap("_unlink_legacy_file", lambda args, count: True)
    else:
        original = wrap("_unlink_legacy", lambda args, count: True)

    with pytest.raises(SystemExit, match="simulated process death"):
        AccountStore(tmp_path).migrate()
    monkeypatch.setattr(
        patched_owner,
        {
            "accounts_directory": "_mkdir_child_durable",
            "provider_directory": "_mkdir_child_durable",
            "account_directory": "_mkdir_child_durable",
            "credential_file": "_atomic_publish",
            "credential_directory_sync": "_write_tokens",
            "credential_verification": "_read_tokens",
            "registry_file": "_atomic_publish",
            "registry_directory_sync": "_write_registry",
            "legacy_unlink": "_unlink_legacy_file",
            "legacy_directory_sync": "_unlink_legacy",
        }[step],
        original,
    )

    store = AccountStore(tmp_path)
    store.migrate()
    destination = store.credential_path(AccountId("anthropic", "default"))
    registered = store.default("anthropic") == AccountId("anthropic", "default")
    assert registered and destination.exists()
    assert not legacy.exists()
    assert json.loads(destination.read_text()) == tokens().to_dict()
    assert not list(tmp_path.rglob("*.tmp"))


def test_concurrent_migration_in_two_processes_keeps_one_login(tmp_path: Path) -> None:
    _write_legacy(tmp_path, "codex")
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    processes = [
        context.Process(target=_migration_worker, args=(str(tmp_path), start))
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    store = AccountStore(tmp_path)
    assert store.list() == [AccountId("codex", "default")]
    assert store.default("codex") == AccountId("codex", "default")
    assert json.loads(store.registry.read_text())["version"] == 1
    assert not list(tmp_path.rglob("*.tmp"))


def test_migration_is_noop_when_already_migrated(tmp_path: Path) -> None:
    legacy = _write_legacy(tmp_path, "anthropic")
    store = AccountStore(tmp_path)
    store.migrate()
    before = (
        store.registry.read_bytes(),
        store.credential_path(AccountId("anthropic", "default")).read_bytes(),
    )

    store.migrate()

    assert not legacy.exists()
    assert before == (
        store.registry.read_bytes(),
        store.credential_path(AccountId("anthropic", "default")).read_bytes(),
    )


def test_migration_does_not_delete_legacy_file_newer_than_migrated_copy(
    tmp_path: Path,
) -> None:
    _write_legacy(tmp_path, "codex")
    store = AccountStore(tmp_path)
    store.migrate()
    destination = store.credential_path(AccountId("codex", "default"))
    destination_mtime = destination.stat().st_mtime_ns
    newer = OAuthTokens("newer-access", "newer-refresh", 10_000_000_000.0)
    legacy = tmp_path / "codex-oauth.json"
    legacy.write_text(json.dumps(newer.to_dict()))
    newer_mtime = destination_mtime + 1_000_000_000
    os.utime(legacy, ns=(newer_mtime, newer_mtime))

    store.migrate()

    assert json.loads(destination.read_text()) == newer.to_dict()
    assert not legacy.exists()


def test_account_files_are_0600_and_dirs_are_private(tmp_path: Path) -> None:
    store = AccountStore(tmp_path)
    named_account_credentials(tmp_path, "codex", "personal").save(tokens())
    with store.provider_login_lock("codex"):
        pass
    store.migrate()

    paths = list(tmp_path.rglob("*"))
    assert all(path.stat().st_mode & 0o777 == 0o700 for path in paths if path.is_dir())
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in paths if path.is_file())


def test_no_token_text_in_errors_logs_or_registry(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    named_account_credentials(tmp_path, "anthropic", "work").save(tokens())

    assert _SENTINEL not in caplog.text
    metadata = [
        path
        for path in tmp_path.rglob("*")
        if path.is_file() and path.name != "oauth.json"
    ]
    assert all(_SENTINEL not in path.read_text() for path in metadata)


def test_concurrent_login_is_rejected_per_provider(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    process = context.Process(
        target=_login_lock_worker, args=(str(tmp_path), ready, release)
    )
    process.start()
    assert ready.wait(timeout=10)
    try:
        with (
            pytest.raises(AccountError, match="login.*already in progress"),
            AccountStore(tmp_path).provider_login_lock("anthropic"),
        ):
            pytest.fail("second login acquired the provider lock")
    finally:
        release.set()
        process.join(timeout=10)
    assert process.exitcode == 0


def test_cli_login_does_not_overwrite_legacy_login_without_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy = _write_legacy(tmp_path, "anthropic")
    expected = json.loads(legacy.read_text())
    monkeypatch.setattr(cli_main, "env_home", lambda: tmp_path)

    async def login_must_not_run(*_args, **_kwargs):
        raise AssertionError("login started before the existing account was checked")

    monkeypatch.setattr(cli_main, "run_login", login_must_not_run)

    with pytest.raises(AccountError, match="already exists"):
        cli_main._run_login("anthropic", "default", False)

    destination = AccountStore(tmp_path).credential_path(
        AccountId("anthropic", "default")
    )
    assert not legacy.exists()
    assert json.loads(destination.read_text()) == expected


def test_accounts_list_shows_migrated_legacy_login(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    legacy = _write_legacy(tmp_path, "codex")
    monkeypatch.setattr(cli_main, "env_home", lambda: tmp_path)
    args = SimpleNamespace(action=None, account=None, json=False)

    assert cli_main._handle_accounts(args) == 0

    assert capsys.readouterr().out == (
        "codex:default  ★ default\n"
        "  usage unknown\n"
    )
    assert not legacy.exists()


def test_named_login_registers_account(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def complete_login(provider, _pkce_values):
        await asyncio.to_thread(provider.credential_store.save, tokens())

    monkeypatch.setattr(cli_main, "env_home", lambda: tmp_path)
    monkeypatch.setattr(cli_main, "run_login", complete_login)

    assert cli_main._run_login("anthropic", "work", False) is None

    store = AccountStore(tmp_path)
    assert store.list() == [AccountId("anthropic", "work")]
    assert (
        json.loads(store.credential_path(AccountId("anthropic", "work")).read_text())
        == tokens().to_dict()
    )


def test_default_set_and_read_back(tmp_path: Path) -> None:
    store = AccountStore(tmp_path)
    personal = AccountId("anthropic", "personal")
    named_account_credentials(tmp_path, "anthropic", "work").save(tokens())
    named_account_credentials(tmp_path, "anthropic", "personal").save(tokens())

    store.set_default(personal)

    assert store.default("anthropic") == personal


def test_default_to_unknown_account_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(AccountError, match="/accounts"):
        AccountStore(tmp_path).set_default(AccountId("anthropic", "work"))


def test_login_then_runtime_reads_the_new_account(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def complete_login(provider, _pkce_values):
        await asyncio.to_thread(provider.credential_store.save, tokens())

    monkeypatch.setattr(cli_main, "env_home", lambda: tmp_path)
    monkeypatch.setattr(cli_main, "run_login", complete_login)

    assert cli_main._run_login("anthropic", "work", False) is None
    backend, _ = build_backend("claude", None, home=tmp_path)

    assert backend.token_store.read() == tokens()
    assert backend.token_store.path == AccountStore(tmp_path).credential_path(
        AccountId("anthropic", "work")
    )


def test_migration_then_every_reader_uses_the_account_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy = _write_legacy(tmp_path, "anthropic")
    monkeypatch.setenv("ZETA_HOME", str(tmp_path))
    AccountStore(tmp_path).migrate()

    primary, _ = build_backend("claude", None, home=tmp_path)
    automation, _ = unattended.build_backend("claude", None, home=tmp_path)
    memory, _ = build_backend("claude", None, home=tmp_path)
    child, error = resolve_child_backend(
        SimpleNamespace(
            backend=None,
            context_assembler=SimpleNamespace(token_budget=200_000),
        ),
        "claude-sonnet-4-6",
    )
    monkeypatch.setattr(
        tui_models.httpx,
        "get",
        lambda *args, **kwargs: SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"data": []},
        ),
    )
    manager = SessionManager(tmp_path)
    opened = manager.create(provider="claude", model="claude-sonnet-4-6", cwd=tmp_path)
    context = SessionAccountContext.load(tmp_path, manager, opened.metadata)
    assert tui_models.load_model_catalog(
        "claude", account_context=context
    ) == frozenset()
    opened.store.close()

    expected = AccountStore(tmp_path).credential_path(
        AccountId("anthropic", "default")
    )
    assert error is None
    assert child is not None
    stores = [
        primary.token_store,
        child.token_store,
        automation.token_store,
        memory.token_store,
        credential_store("claude", home=tmp_path),
    ]
    assert all(store.path == expected and store.read() == tokens() for store in stores)
    assert not legacy.exists()


def test_refresh_after_migration_writes_the_account_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expired = OAuthTokens("old-access", "old-refresh", 1.0)
    fresh = OAuthTokens("fresh-access", "fresh-refresh", 9_999_999_999.0)
    (tmp_path / "anthropic-oauth.json").write_text(json.dumps(expired.to_dict()))
    AccountStore(tmp_path).migrate()
    store = credential_store("claude", home=tmp_path)

    async def return_fresh(_refresh_token: str, _client: httpx.AsyncClient):
        return fresh

    monkeypatch.setattr(store, "refresh", return_fresh)

    async def refresh() -> str:
        async with httpx.AsyncClient() as client:
            return await store.access_token(client)

    assert asyncio.run(refresh()) == fresh.access_token
    assert store.path == AccountStore(tmp_path).credential_path(
        AccountId("anthropic", "default")
    )
    assert store.read() == fresh
    assert not (tmp_path / "anthropic-oauth.json").exists()


def test_orphan_credential_without_pending_record_is_not_registered(
    tmp_path: Path,
) -> None:
    store = AccountStore(tmp_path)
    account = AccountId("anthropic", "deleted")
    path = store.credential_path(account)
    path.parent.mkdir(parents=True)
    orphan = OAuthTokens("codex-access", "codex-refresh", 4_000_000_000)
    path.write_text(json.dumps(orphan.to_dict()) + "\n")
    original = path.read_bytes()

    assert store.resolve_account("anthropic", account) == account
    assert store.list() == []
    assert store.default("anthropic") is None
    assert path.read_bytes() == original


def test_pending_record_digest_mismatch_discards_record_and_keeps_credential(
    tmp_path: Path,
) -> None:
    store = AccountStore(tmp_path)
    account = AccountId("anthropic", "work")
    path = store.credential_path(account)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(tokens().to_dict()) + "\n")
    original = path.read_bytes()
    pending = path.with_name("publish-pending.json")
    pending.write_text(
        json.dumps(
            {
                "provider": account.provider,
                "name": account.name,
                "sha256": hashlib.sha256(b"different credential").hexdigest(),
            }
        )
        + "\n"
    )

    assert store.resolve_account("anthropic", account) == account
    assert store.list() == []
    assert store.default("anthropic") is None
    assert not pending.exists()
    assert path.read_bytes() == original


@pytest.mark.parametrize("publication", ["bootstrap", "login"])
def test_crash_between_credential_and_registry_publish_recovers_account(
    publication: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = OAuthTokens("staged-access", "staged-refresh", 4_000_000_000)
    credential = named_account_credentials(tmp_path, "anthropic")
    bootstrap_calls = 0

    def bootstrap() -> OAuthTokens:
        nonlocal bootstrap_calls
        bootstrap_calls += 1
        return expected

    monkeypatch.setattr(credential._driver, "bootstrap", bootstrap)
    real_write_registry = AccountStore._write_registry

    def crash(*args, **kwargs) -> None:
        raise SystemExit("simulated process death")

    monkeypatch.setattr(AccountStore, "_write_registry", crash)
    with pytest.raises(SystemExit, match="simulated process death"):
        if publication == "bootstrap":
            credential.bootstrap()
        else:
            credential.save(expected)
    monkeypatch.setattr(AccountStore, "_write_registry", real_write_registry)
    pending = credential.path.with_name("publish-pending.json")
    assert pending.exists()

    def bootstrap_must_not_run() -> OAuthTokens:
        raise AssertionError("bootstrap source was consulted during recovery")

    monkeypatch.setattr(credential._driver, "bootstrap", bootstrap_must_not_run)
    assert credential.store.resolve_account(
        "anthropic", credential.account
    ) == AccountId("anthropic", "default")

    async def access() -> str:
        async with httpx.AsyncClient() as client:
            return await credential.access_token(client)

    assert asyncio.run(access()) == expected.access_token
    store = AccountStore(tmp_path)
    assert store.list() == [AccountId("anthropic", "default")]
    assert store.default("anthropic") == AccountId("anthropic", "default")
    assert not pending.exists()
    assert bootstrap_calls == (1 if publication == "bootstrap" else 0)


def test_recovery_never_overwrites_a_corrupt_staged_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    credential = named_account_credentials(tmp_path, "codex")
    path = credential.path
    path.parent.mkdir(parents=True)
    corrupt = b"not valid oauth json\n"
    path.write_bytes(corrupt)
    path.with_name("publish-pending.json").write_text(
        json.dumps(
            {
                "provider": credential.account.provider,
                "name": credential.account.name,
                "sha256": hashlib.sha256(corrupt).hexdigest(),
            }
        )
        + "\n"
    )
    monkeypatch.setattr(
        credential._driver,
        "bootstrap",
        lambda: pytest.fail("bootstrap source was consulted during recovery"),
    )

    async def access() -> str:
        async with httpx.AsyncClient() as client:
            return await credential.access_token(client)

    with pytest.raises(AccountError, match="credentials are invalid"):
        asyncio.run(access())
    assert path.read_bytes() == corrupt
    assert not AccountStore(tmp_path).contains(AccountId("codex", "default"))


def test_bootstrap_refresh_runs_without_the_registry_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expired = OAuthTokens("expired", "refresh", 1.0)
    refreshed = OAuthTokens("fresh", "new-refresh", 4_000_000_000)
    credential = named_account_credentials(tmp_path, "codex")
    other = named_account_credentials(tmp_path, "codex", "other")
    another = named_account_credentials(tmp_path, "codex", "another")
    other.save(tokens())
    another.save(tokens())
    monkeypatch.setattr(credential._driver, "bootstrap", lambda: expired)
    refresh_started = threading.Event()
    release_refresh = threading.Event()
    refresh_done = threading.Event()
    operations_done = threading.Event()
    failures: list[BaseException] = []

    async def refresh(*_args) -> OAuthTokens:
        refresh_started.set()
        await asyncio.to_thread(release_refresh.wait)
        return refreshed

    monkeypatch.setattr(credential, "refresh", refresh)

    def resolve() -> None:
        try:
            async def access() -> None:
                async with httpx.AsyncClient() as client:
                    assert await credential.access_token(client) == refreshed.access_token

            asyncio.run(access())
        except Exception as exc:  # noqa: BLE001 - report thread failures in test
            failures.append(exc)
        finally:
            refresh_done.set()

    def mutate_registry() -> None:
        try:
            store = AccountStore(tmp_path)
            assert AccountId("codex", "default") in store.list()
            store.set_default(AccountId("codex", "another"))
        except Exception as exc:  # noqa: BLE001 - report thread failures in test
            failures.append(exc)
        finally:
            operations_done.set()

    refresh_thread = threading.Thread(target=resolve)
    refresh_thread.start()
    assert refresh_started.wait(timeout=5)
    registry_thread = threading.Thread(target=mutate_registry)
    registry_thread.start()
    try:
        assert operations_done.wait(timeout=1), "registry remained locked during refresh"
    finally:
        release_refresh.set()
        refresh_thread.join(timeout=5)
        registry_thread.join(timeout=5)
    assert refresh_done.is_set()
    assert not failures
    assert AccountStore(tmp_path).default("codex") == AccountId("codex", "another")


def test_bootstrap_writes_credential_registry_row_and_default_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claude_home = tmp_path / ".claude"
    claude_home.mkdir()
    source = claude_home / ".credentials.json"
    source.write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": "bootstrap-access",
                    "refreshToken": "bootstrap-refresh",
                    "expiresAt": 4_000_000_000,
                }
            }
        )
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("ZETA_HOME", raising=False)
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    store = credential_store("claude", home=tmp_path / "zeta")
    assert store is not None

    async def access() -> str:
        async with httpx.AsyncClient() as client:
            return await store.access_token(client)

    assert asyncio.run(access()) == "bootstrap-access"
    registry = AccountStore(tmp_path / "zeta")
    assert registry.list() == [AccountId("anthropic", "default")]
    assert registry.default("anthropic") == AccountId("anthropic", "default")


def test_refresh_racing_replacement_login_keeps_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expired = OAuthTokens("old-access", "old-refresh", 1.0)
    stale = OAuthTokens("stale-refresh", "stale-refresh-token", 4_000_000_000)
    replacement = OAuthTokens("replacement", "replacement-refresh", 4_000_000_000)
    named_account_credentials(tmp_path, "codex").save(expired)
    credential = credential_store("codex", home=tmp_path)
    assert credential is not None
    refresh_started = threading.Event()
    release_refresh = threading.Event()
    replacement_started = threading.Event()
    replacement_done = threading.Event()

    async def refresh(_refresh_token: str, _client: httpx.AsyncClient) -> OAuthTokens:
        refresh_started.set()
        await asyncio.to_thread(release_refresh.wait)
        return stale

    monkeypatch.setattr(credential, "refresh", refresh)

    def replace_login() -> None:
        replacement_started.set()
        named_account_credentials(tmp_path, "codex").save(replacement)
        replacement_done.set()

    async def race() -> None:
        async with httpx.AsyncClient() as client:
            refresh_task = asyncio.create_task(credential.access_token(client))
            await asyncio.to_thread(refresh_started.wait)
            replacement_thread = threading.Thread(target=replace_login)
            replacement_thread.start()
            await asyncio.to_thread(replacement_started.wait)
            # Old code publishes replacement while refresh is paused. The shared
            # lifecycle lock makes replacement wait until refresh publishes.
            await asyncio.to_thread(replacement_done.wait, 0.2)
            release_refresh.set()
            await refresh_task
            replacement_thread.join(timeout=5)
            assert not replacement_thread.is_alive()

    asyncio.run(race())
    assert credential.read() == replacement
    assert not list((tmp_path / "accounts").rglob(".oauth.json.lock"))


def test_account_credential_read_rejects_symlink_and_hardlink(tmp_path: Path) -> None:
    for kind in ("symlink", "hardlink"):
        home = tmp_path / kind
        account = AccountId("codex", "default")
        registry = AccountStore(home)
        named_account_credentials(home, "codex").save(tokens())
        path = registry.credential_path(account)
        credential = credential_store("codex", home=home)
        assert credential is not None
        outside = home / "outside.json"
        outside.write_text(json.dumps(tokens().to_dict()))
        path.unlink()
        if kind == "symlink":
            path.symlink_to(outside)
        else:
            path.hardlink_to(outside)

        with pytest.raises(RuntimeError):
            credential.read()


def test_account_credential_write_fsyncs_file_then_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    named_account_credentials(tmp_path, "codex").save(tokens())
    credential = credential_store("codex", home=tmp_path)
    assert credential is not None
    events: list[str] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def observed_fsync(fd: int) -> None:
        events.append("directory-fsync" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file-fsync")
        real_fsync(fd)

    def observed_replace(*args, **kwargs) -> None:
        events.append("replace")
        real_replace(*args, **kwargs)

    monkeypatch.setattr(account_store.os, "fsync", observed_fsync)
    monkeypatch.setattr(account_store.os, "replace", observed_replace)
    monkeypatch.setattr(oauth_module.os, "fsync", observed_fsync)
    monkeypatch.setattr(oauth_module.os, "replace", observed_replace)

    credential.save(OAuthTokens("new", "new-refresh", 4_000_000_000))

    assert events.index("file-fsync") < events.index("replace")
    assert events.index("replace") < events.index("directory-fsync")


def test_cli_and_serve_login_share_provider_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli_entered = threading.Event()
    release_cli = threading.Event()
    serve_entered = threading.Event()

    async def cli_login(provider, _pkce_values):
        cli_entered.set()
        await asyncio.to_thread(release_cli.wait)
        await asyncio.to_thread(provider.credential_store.save, tokens())

    async def serve_login(*args, **kwargs):
        serve_entered.set()

    monkeypatch.setattr(cli_main, "env_home", lambda: tmp_path)
    monkeypatch.setattr(cli_main, "run_login", cli_login)
    monkeypatch.setattr(server_login, "run_login", serve_login)
    cli_thread = threading.Thread(target=cli_main._run_login, args=("anthropic",))
    cli_thread.start()
    assert cli_entered.wait(timeout=5)
    async def run_serve_login() -> dict:
        return await Logins(tmp_path)._run(
            "claude", asyncio.get_running_loop().create_future()
        )

    try:
        result = asyncio.run(run_serve_login())
    finally:
        release_cli.set()
        cli_thread.join(timeout=5)

    assert not cli_thread.is_alive()
    assert result["state"] == "failed"
    assert not serve_entered.is_set()


def test_accounts_default_invalid_input_is_a_command_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli_main, "env_home", lambda: tmp_path)

    assert cli_main.main(["accounts", "default", "BOGUS:x"]) != 0

    output = capsys.readouterr()
    assert "invalid account id" in output.err.lower()
    assert _SENTINEL not in output.err


def test_new_session_pins_global_default_in_meta(tmp_path: Path) -> None:
    home = tmp_path / "home"
    account = AccountId("anthropic", "work")
    named_account_credentials(home, "anthropic", "work").save(tokens())
    AccountStore(home).set_default(account)
    manager = SessionManager(home)
    opened = manager.create(provider="claude", model="claude-sonnet-4-6", cwd=tmp_path)

    resolver = SessionAccountContext.load(home, manager, opened.metadata)

    assert resolver.resolve("claude") == account
    assert manager.read_metadata(opened.metadata.session_id).selected_accounts == {
        "anthropic": "anthropic:work"
    }
    opened.store.close()


def test_resumed_session_without_selection_pins_default_at_load(tmp_path: Path) -> None:
    home = tmp_path / "home"
    manager = SessionManager(home)
    opened = manager.create(provider="codex", model="gpt-5.6-luna", cwd=tmp_path)
    opened.store.close()
    account = AccountId("codex", "personal")
    named_account_credentials(home, "codex", "personal").save(tokens())
    AccountStore(home).set_default(account)

    resumed = manager.open(opened.metadata.session_id)
    resolver = SessionAccountContext.load(home, manager, resumed.metadata)

    assert resolver.resolve("codex") == account
    assert manager.read_metadata(resumed.metadata.session_id).selected_accounts == {
        "codex": "codex:personal"
    }
    resumed.store.close()


def test_session_with_missing_pinned_account_fails_naming_accounts(tmp_path: Path) -> None:
    home = tmp_path / "home"
    manager = SessionManager(home)
    opened = manager.create(provider="codex", model="gpt-5.6-luna", cwd=tmp_path)
    manager.record_account_selection(
        opened.metadata, AccountId("codex", "deleted"), boundary=0
    )

    with pytest.raises(AccountError, match=r"codex:deleted.*?/accounts"):
        SessionAccountContext.load(home, manager, opened.metadata)
    opened.store.close()


@pytest.mark.asyncio
async def test_switch_changes_only_current_session_not_other_sessions_or_live_children(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    first = AccountId("codex", "first")
    second = AccountId("codex", "second")
    await asyncio.to_thread(
        named_account_credentials(home, "codex", "first").save, tokens()
    )
    await asyncio.to_thread(
        named_account_credentials(home, "codex", "second").save, tokens()
    )
    await asyncio.to_thread(AccountStore(home).set_default, first)
    manager = SessionManager(home)
    current = await asyncio.to_thread(
        manager.create, provider="codex", model="gpt-5.6-luna", cwd=tmp_path
    )
    other = await asyncio.to_thread(
        manager.create, provider="codex", model="gpt-5.6-luna", cwd=tmp_path
    )
    current_resolver = await asyncio.to_thread(
        SessionAccountContext.load, home, manager, current.metadata
    )
    other_resolver = await asyncio.to_thread(
        SessionAccountContext.load, home, manager, other.metadata
    )
    child_accounts = current_resolver.snapshot()
    loop = SimpleNamespace(backend=None)
    loop.set_backend = lambda backend: setattr(loop, "backend", backend)

    await current_resolver.switch_active(
        loop,
        second,
        boundary=7,
        provider="codex",
        build_backend=lambda _account: None,
    )

    assert await asyncio.to_thread(current_resolver.resolve, "codex") == second
    assert await asyncio.to_thread(other_resolver.resolve, "codex") == first
    assert child_accounts["codex"] == first
    await current_resolver.switch_active(
        loop,
        first,
        boundary=7,
        provider="codex",
        build_backend=lambda _account: None,
    )
    current_metadata = manager.read_metadata(current.metadata.session_id)
    assert current_metadata.account_switch_sequences == {"codex": 2}
    assert current_metadata.native_context_boundaries == {"codex": 8}
    assert manager.read_metadata(other.metadata.session_id).selected_accounts == {
        "codex": "codex:first"
    }
    current.store.close()
    other.store.close()


def test_child_inherits_parent_pinned_account(tmp_path: Path) -> None:
    home = tmp_path / "home"
    account = AccountId("anthropic", "work")
    named_account_credentials(home, "anthropic", "work").save(tokens())
    manager = SessionManager(home)
    opened = manager.create(provider="claude", model="claude-sonnet-4-6", cwd=tmp_path)
    manager.record_account_selection(opened.metadata, account, boundary=0)
    resolver = SessionAccountContext.load(home, manager, opened.metadata)
    loop = SimpleNamespace(
        backend=None,
        context_assembler=SimpleNamespace(token_budget=200_000),
        account_context=resolver,
    )

    child, error = resolve_child_backend(loop, "claude-sonnet-4-6")

    assert error is None
    assert child.token_store.account == account
    opened.store.close()


def test_every_backend_path_uses_session_resolver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    anthropic = AccountId("anthropic", "work")
    codex = AccountId("codex", "personal")
    named_account_credentials(home, "anthropic", "work").save(tokens())
    named_account_credentials(home, "codex", "personal").save(tokens())
    manager = SessionManager(home)
    opened = manager.create(provider="claude", model="claude-sonnet-4-6", cwd=tmp_path)
    manager.record_account_selection(opened.metadata, anthropic, boundary=0)
    resolver = SessionAccountContext.load(home, manager, opened.metadata)
    selected = resolver.snapshot()

    primary, _ = build_backend("claude", None, home=home, account=selected["anthropic"])
    captured: list[tuple[AccountId | None, Path | None]] = []

    async def capture(_query, _signal, *, account=None, home=None):
        captured.append((account, home))
        return websearch_codex.CodexSearchResult("answer", [])

    monkeypatch.setattr(websearch.codex, "search", capture)
    registry = SimpleNamespace(account_context=resolver)
    asyncio.run(
        websearch._websearch(
            registry,
            {"query": "query"},
            SimpleNamespace(aborted=False),
        )
    )
    loop = SimpleNamespace(
        backend=primary,
        context_assembler=SimpleNamespace(token_budget=200_000),
        account_context=resolver,
    )
    child, error = resolve_child_backend(loop, "gpt-5.6-luna")

    assert primary.token_store.account == anthropic
    assert error is None
    assert child.token_store.account == codex
    assert captured == [(codex, home)]
    assert manager.read_metadata(opened.metadata.session_id).selected_accounts == {
        "anthropic": "anthropic:work",
        "codex": "codex:personal",
    }
    opened.store.close()


def test_account_switch_filters_stale_native_artifacts(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "session", cwd=tmp_path)
    store.append_message(
        Message(
            MessageRole.ASSISTANT,
            [
                ThinkingContent("summary", "encrypted"),
                RedactedThinkingContent("redacted"),
                TextContent("answer"),
            ],
            metadata={"account_provider": "codex", "account_switch_sequence": 0},
        )
    )
    backend = FakeBackend([ScriptedTurn(content=[TextContent("next")])])
    loop = AgentLoop(backend, store, skill_catalog=SkillCatalog.empty())
    loop.account_context = SimpleNamespace(
        active_provider="codex", switch_sequence=1
    )

    async def run() -> None:
        async for _ in loop.run_turn("continue", origin=MessageOrigin.USER):
            pass

    asyncio.run(run())

    stale = next(message for message in backend.calls[0][0] if message.role == MessageRole.ASSISTANT)
    assert stale.content == [ThinkingContent("summary"), TextContent("answer")]
    store.close()

@pytest.mark.asyncio
async def test_grandchild_inherits_child_snapshot_not_global_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.agent import runner as agent_runner
    from zeta.core.abort import AbortGenerationRegistry
    from zeta.protocol.types import ToolCall
    from zeta.skills.agent_catalog import discover_packaged_agents

    home = tmp_path / "home"
    first = AccountId("codex", "first")
    second = AccountId("codex", "second")
    await asyncio.to_thread(
        named_account_credentials(home, "codex", "first").save, tokens()
    )
    await asyncio.to_thread(
        named_account_credentials(home, "codex", "second").save, tokens()
    )
    await asyncio.to_thread(AccountStore(home).set_default, first)
    manager = SessionManager(home)
    opened = manager.create(provider="codex", model="gpt-5.6-luna", cwd=tmp_path)
    resolver = await asyncio.to_thread(
        SessionAccountContext.load, home, manager, opened.metadata
    )
    parent = AgentLoop(
        FakeBackend([]),
        opened.store,
        skill_catalog=SkillCatalog.empty(),
        agent_catalog=discover_packaged_agents(),
    )
    parent.account_context = resolver
    parent.tool_registry.account_context = resolver

    async def consume_child(child_loop: AgentLoop, _prompt: str, **_kwargs: object):
        await asyncio.to_thread(AccountStore(home).set_default, second)
        grandchild, error = await asyncio.to_thread(
            resolve_child_backend, child_loop, "gpt-5.6-luna"
        )
        assert error is None
        assert grandchild.token_store.account == first
        return {"content": [{"type": "text", "text": "done"}], "isError": False}

    monkeypatch.setattr(agent_runner, "consume_child", consume_child)
    call = ToolCall(
        "child", "agent", {"prompt": "work", "description": "child", "background": False}
    )
    try:
        result = await parent._run_agent_tool(
            call, call.arguments, AbortGenerationRegistry().new_generation(), None
        )
        assert result["isError"] is False, result
    finally:
        await parent.close()
        opened.store.close()


@pytest.mark.asyncio
async def test_live_child_websearch_ignores_later_parent_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.agent import runner as agent_runner
    from zeta.core.abort import AbortGenerationRegistry
    from zeta.protocol.types import ToolCall
    from zeta.skills.agent_catalog import discover_packaged_agents

    home = tmp_path / "home"
    first = AccountId("codex", "first")
    second = AccountId("codex", "second")
    await asyncio.to_thread(
        named_account_credentials(home, "codex", "first").save, tokens()
    )
    await asyncio.to_thread(
        named_account_credentials(home, "codex", "second").save, tokens()
    )
    await asyncio.to_thread(AccountStore(home).set_default, first)
    manager = SessionManager(home)
    opened = manager.create(provider="claude", model="claude-sonnet-4-6", cwd=tmp_path)
    resolver = await asyncio.to_thread(
        SessionAccountContext.load, home, manager, opened.metadata
    )
    parent = AgentLoop(
        FakeBackend([]), opened.store, skill_catalog=SkillCatalog.empty(),
        agent_catalog=discover_packaged_agents(),
    )
    parent.account_context = resolver
    parent.tool_registry.account_context = resolver
    captured: list[AccountId | None] = []

    async def capture(_query, _signal, *, account=None, home=None):
        captured.append(account)
        return websearch_codex.CodexSearchResult("answer", [])

    async def consume_child(child_loop: AgentLoop, _prompt: str, **_kwargs: object):
        await resolver.switch_active(
            parent,
            second,
            boundary=0,
            provider="claude",
            build_backend=lambda _account: None,
        )
        await websearch._websearch(
            child_loop.tool_registry,
            {"query": "query"},
            SimpleNamespace(aborted=False),
        )
        return {"content": [{"type": "text", "text": "done"}], "isError": False}

    monkeypatch.setattr(websearch.codex, "search", capture)
    monkeypatch.setattr(agent_runner, "consume_child", consume_child)
    call = ToolCall(
        "child", "agent", {"prompt": "work", "description": "child", "background": False}
    )
    try:
        result = await parent._run_agent_tool(
            call, call.arguments, AbortGenerationRegistry().new_generation(), None
        )
        assert result["isError"] is False
        assert captured == [first]
    finally:
        await parent.close()
        opened.store.close()


def test_model_catalog_uses_session_pinned_account(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_account = AccountId("anthropic", "session")
    default_account = AccountId("anthropic", "global")
    named_account_credentials(tmp_path, "anthropic", "session").save(tokens())
    named_account_credentials(tmp_path, "anthropic", "global").save(
        OAuthTokens("wrong", "wrong-refresh", 9_999_999_999.0)
    )
    AccountStore(tmp_path).set_default(default_account)
    requests: list[httpx.Request] = []

    def request(url: str, **kwargs: object):
        requests.append(httpx.Request("GET", url, headers=kwargs["headers"]))
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"data": [{"id": "claude-sonnet-4-6"}]},
        )

    monkeypatch.setattr(tui_models.httpx, "get", request)
    manager = SessionManager(tmp_path)
    opened = manager.create(provider="claude", model="claude-sonnet-4-6", cwd=tmp_path)
    manager.record_account_selection(opened.metadata, session_account, boundary=0)
    context = SessionAccountContext.load(tmp_path, manager, opened.metadata)
    assert tui_models.load_model_catalog(
        "claude", account_context=context
    ) == frozenset({"claude-sonnet-4-6"})
    opened.store.close()
    assert requests[0].headers["authorization"] == f"Bearer {_SENTINEL}"


def test_api_key_mode_skips_named_accounts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.accounts.session import SessionAccountContext

    monkeypatch.setenv("ZETA_ALLOW_API_KEY", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
    manager = SessionManager(tmp_path)
    opened = manager.create(provider="claude", model="claude-sonnet-4-6", cwd=tmp_path)
    context = SessionAccountContext.load(tmp_path, manager, opened.metadata)

    assert context.auth_mode == "api-key"
    assert context.backend_kwargs("claude") == {"home": tmp_path}
    assert opened.metadata.selected_accounts == {}
    assert context.switching_unavailable == (
        "this session uses an API key; account switching is unavailable"
    )

    from zeta.tui.app import TUIApp

    loop = AgentLoop(FakeBackend([]), opened.store, skill_catalog=SkillCatalog.empty())
    loop.account_context = context
    app = TUIApp(loop, provider="claude", model="claude-sonnet-4-6")
    rendered: list[object] = []
    app._presenter.print_unit = rendered.append
    app._handle_slash_output(asyncio.run(app.slash_accounts("")))
    assert rendered[0].plain == (
        "system · this session uses an API key; account switching is unavailable"
    )
    opened.store.close()


def test_label_is_display_only_and_id_stays_stable(tmp_path: Path) -> None:
    home = tmp_path / "home"
    account = AccountId("codex", "work")
    named_account_credentials(home, "codex", "work").save(tokens())
    store = AccountStore(home)
    store.set_default(account)
    manager = SessionManager(home)
    opened = manager.create(provider="codex", model="offline", cwd=tmp_path)
    manager.record_account_selection(opened.metadata, account, boundary=0)
    opened.store.close()
    job = parse_job(
        "stable-account",
        {
            "prompt": "test",
            "trigger": {"kind": "schedule", "cron": "0 8 * * 1-5"},
            "servers": [],
            "allow": [],
            "provider": "codex",
            "model": "offline",
            "cwd": str(tmp_path),
        },
    )
    with SQLiteStore(home) as automations:
        revision = automations.draft(job)
        automations.approve(job.name, revision.revision, None, datetime.now(UTC))
        assert automations.get(job.name).job.account == str(account)
    credential = store.credential_path(account)
    credential_inode = credential.stat().st_ino

    with store.lifecycle_guard(account):
        assert store.rename(account, "henry") == "renamed codex:work to codex:henry"

    assert store.list() == [account]
    assert store.default("codex") == account
    assert store.resolve_reference("codex", "henry") == account
    assert store.resolve_reference("codex", "work") == account
    assert [(item.label, item.account) for item in store.list_display()] == [
        ("henry", account)
    ]
    assert credential.stat().st_ino == credential_inode
    assert manager.read_metadata(opened.metadata.session_id).selected_accounts == {
        "codex": "codex:work"
    }
    with SQLiteStore(home) as automations:
        assert automations.get(job.name).job.account == str(account)
    assert named_account_credentials(home, "codex", "work").read() == tokens()
    assert not (home / ".account-rename.json").exists()


def _read_registry_with_548_rules(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if (
        not isinstance(value, dict)
        or value.get("version") != 1
        or not isinstance(value.get("accounts"), dict)
        or not isinstance(value.get("defaults"), dict)
    ):
        raise ValueError
    accounts = dict(value["accounts"])
    defaults = dict(value["defaults"])
    for key, metadata in accounts.items():
        account = AccountId(*key.split(":", 1))
        if key != str(account) or metadata != {
            "provider": account.provider,
            "name": account.name,
        }:
            raise ValueError
    for provider, key in defaults.items():
        if (
            provider not in {"anthropic", "codex"}
            or key not in accounts
            or not key.startswith(f"{provider}:")
        ):
            raise ValueError
    return {"version": 1, "accounts": accounts, "defaults": defaults}


def test_registry_written_by_head_is_readable_by_548_reader(tmp_path: Path) -> None:
    account = AccountId("codex", "work")
    named_account_credentials(tmp_path, "codex", "work").save(tokens())
    store = AccountStore(tmp_path)
    store.rename(account, "main")

    value = _read_registry_with_548_rules(store.registry)
    accounts = value["accounts"]
    defaults = value["defaults"]
    assert [AccountId(*key.split(":", 1)) for key in accounts] == [account]
    assert AccountId(*defaults["codex"].split(":", 1)) == account
    resolved = AccountId("codex", "work") if "codex:work" in accounts else None
    assert resolved == account


def test_login_publish_rejects_label_claimed_during_oauth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = AccountId("codex", "work")
    named_account_credentials(tmp_path, "codex", "work").save(tokens())
    monkeypatch.setattr(cli_main, "env_home", lambda: tmp_path)
    oauth_started = threading.Event()
    publish = threading.Event()
    failures: list[BaseException] = []

    async def login(provider, _pkce_values) -> None:
        oauth_started.set()
        await asyncio.to_thread(publish.wait)
        provider.credential_store.save(tokens())

    monkeypatch.setattr(cli_main, "run_login", login)

    def run_login() -> None:
        try:
            cli_main._run_login("codex", "new")
        except AccountError as exc:
            failures.append(exc)

    thread = threading.Thread(target=run_login)
    thread.start()
    assert oauth_started.wait(1)
    AccountStore(tmp_path).rename(work, "new")
    publish.set()
    thread.join(2)

    assert not thread.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], AccountError)
    store = AccountStore(tmp_path)
    assert store.resolve_reference("codex", "new") == work
    assert store.list() == [work]
    assert not store.credential_path(AccountId("codex", "new")).exists()


def test_label_unique_across_ids_and_labels(tmp_path: Path) -> None:
    store = AccountStore(tmp_path)
    work = AccountId("codex", "work")
    personal = AccountId("codex", "personal")
    named_account_credentials(tmp_path, "codex", "work").save(tokens())
    named_account_credentials(tmp_path, "codex", "personal").save(tokens())

    with pytest.raises(AccountError, match="already exists"):
        store.rename(work, "personal")
    store.rename(personal, "home")
    with pytest.raises(AccountError, match="already exists"):
        store.rename(work, "home")
    with pytest.raises(ValueError, match="invalid account label"):
        store.rename(work, "Not Valid")


def test_cli_resolves_label_or_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    account = AccountId("codex", "work")
    named_account_credentials(tmp_path, "codex", "work").save(tokens())
    store = AccountStore(tmp_path)
    store.rename(account, "henry")
    monkeypatch.setattr(cli_main, "env_home", lambda: tmp_path)

    assert cli_main.main(["accounts", "default", "codex:henry"]) == 0
    assert store.default("codex") == account
    assert cli_main.main(["accounts", "default", "codex:work"]) == 0
    assert store.default("codex") == account


def test_refresh_keeps_workspace_when_response_has_only_email(tmp_path: Path) -> None:
    credential = named_account_credentials(tmp_path, "anthropic")
    credential.save(
        OAuthTokens("old-access", "old-refresh", 1.0, workspace="Existing workspace")
    )

    async def refresh(_refresh_token: str, _client: httpx.AsyncClient) -> OAuthTokens:
        return OAuthTokens(
            "new-access", "new-refresh", 9_999_999_999.0, email="new@example.com"
        )

    credential.refresh = refresh  # type: ignore[method-assign]

    async def run() -> None:
        async with httpx.AsyncClient() as client:
            await credential.refresh_token(client)

    asyncio.run(run())
    assert credential.read() == OAuthTokens(
        "new-access",
        "new-refresh",
        9_999_999_999.0,
        email="new@example.com",
        workspace="Existing workspace",
    )


def test_refresh_keeps_email_when_response_has_only_workspace(tmp_path: Path) -> None:
    credential = named_account_credentials(tmp_path, "anthropic")
    credential.save(
        OAuthTokens("old-access", "old-refresh", 1.0, email="old@example.com")
    )

    async def refresh(_refresh_token: str, _client: httpx.AsyncClient) -> OAuthTokens:
        return OAuthTokens(
            "new-access", "new-refresh", 9_999_999_999.0, workspace="New workspace"
        )

    credential.refresh = refresh  # type: ignore[method-assign]

    async def run() -> None:
        async with httpx.AsyncClient() as client:
            await credential.refresh_token(client)

    asyncio.run(run())
    assert credential.read() == OAuthTokens(
        "new-access",
        "new-refresh",
        9_999_999_999.0,
        email="old@example.com",
        workspace="New workspace",
    )


def test_codex_email_from_access_token_profile_claim(tmp_path: Path) -> None:
    access_token = _jwt_token(
        {"https://api.openai.com/profile": {"email": "henry@example.com"}}
    )
    credential = named_account_credentials(tmp_path, "codex")
    credential.save(OAuthTokens(access_token, "refresh", 9_999_999_999))

    listed = AccountStore(tmp_path).list_display()

    assert [(str(item.account), item.detail) for item in listed] == [
        ("codex:default", "henry@example.com")
    ]


def test_list_shows_email_metadata_when_present_offline(tmp_path: Path) -> None:
    parsed = OAuthTokens.from_mapping(
        {
            "access_token": "access",
            "refresh_token": "refresh",
            "expires_at": 9_999_999_999,
            "account": {"email_address": "claude@example.com"},
            "organization": {"name": "Acme"},
        },
        error_type=RuntimeError,
    )
    assert parsed.email == "claude@example.com"
    assert parsed.workspace == "Acme"
    identity = OAuthTokens.from_mapping(
        {
            "access_token": "access",
            "refresh_token": "refresh",
            "expires_at": 9_999_999_999,
            "id_token": _jwt_token({"email": "codex@example.com"}),
        },
        error_type=RuntimeError,
    )
    assert identity.email == "codex@example.com"
    codex = OAuthTokens(
        "access", "refresh", 9_999_999_999, email="henry@example.com"
    )
    anthropic = OAuthTokens(
        "access", "refresh", 9_999_999_999, workspace="Acme"
    )
    named_account_credentials(tmp_path, "codex", "work").save(codex)
    named_account_credentials(tmp_path, "anthropic", "team").save(anthropic)

    listed = AccountStore(tmp_path).list_display()

    assert [(str(item.account), item.detail) for item in listed] == [
        ("anthropic:team", "Acme"),
        ("codex:work", "henry@example.com"),
    ]


def test_login_prompts_for_label_and_rejects_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli_main, "env_home", lambda: tmp_path)
    monkeypatch.setattr(cli_main.sys.stdin, "isatty", lambda: True)
    labels = iter(["work", "work"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(labels))
    called: list[str] = []

    async def login(provider, _pkce_values) -> None:
        called.append(provider.credential_store.account.name)
        await asyncio.to_thread(provider.credential_store.save, tokens())

    monkeypatch.setattr(cli_main, "run_login", login)
    assert cli_main.main(["login", "codex"]) == 0
    assert cli_main.main(["accounts", "login", "codex"]) == 1
    assert called == ["work"]
    assert "already exists" in capsys.readouterr().err
