from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import multiprocessing
import os
import stat
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import zeta.accounts.store as account_store
import zeta.cli.main as cli_main
import zeta.codex as codex_module
import zeta.oauth as oauth_module
import zeta.providers.anthropic as anthropic_module
import zeta.server.login as server_login
from tests.support.credentials import named_account_credentials
from zeta.accounts import AccountError, AccountId, AccountStore
from zeta.agent.runner import resolve_child_backend
from zeta.oauth import OAuthTokens
from zeta.providers.factory import build_backend, credential_store
from zeta.runtime import unattended
from zeta.server.login import Logins
from zeta.tui import models as tui_models

_SENTINEL = "token-text-must-not-escape"


def tokens() -> OAuthTokens:
    return OAuthTokens(_SENTINEL, "refresh-secret", 9_999_999_999.0)


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

    assert capsys.readouterr().out == "codex:default\n"
    assert not legacy.exists()


def test_named_login_registers_account(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def complete_login(provider, _pkce_values):
        provider.credential_store.save(tokens())

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
        provider.credential_store.save(tokens())

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
    assert tui_models.load_model_catalog("claude", home=tmp_path) == frozenset()

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
        provider.credential_store.save(tokens())

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
