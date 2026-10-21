from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import pytest

import zeta.accounts.store as account_store
import zeta.cli.main as cli_main
from zeta.accounts import AccountError, AccountId, AccountStore
from zeta.oauth import OAuthTokens

_SENTINEL = "token-text-must-not-escape"


def tokens() -> OAuthTokens:
    return OAuthTokens(_SENTINEL, "refresh-secret", 9_999_999_999.0)


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
        original = wrap("_write_tokens", lambda args, count: True)
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
    store.ensure_account(AccountId("codex", "personal"), tokens())
    with store.provider_login_lock("codex"):
        pass
    store.migrate()

    paths = list(tmp_path.rglob("*"))
    assert all(path.stat().st_mode & 0o777 == 0o700 for path in paths if path.is_dir())
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in paths if path.is_file())


def test_no_token_text_in_errors_logs_or_registry(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = AccountStore(tmp_path)
    store.ensure_account(AccountId("anthropic", "work"), tokens())
    with pytest.raises(AccountError) as raised:
        store.ensure_account(AccountId("anthropic", "work"), tokens())

    assert _SENTINEL not in str(raised.value)
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
    work = AccountId("anthropic", "work")
    personal = AccountId("anthropic", "personal")
    store.ensure_account(work, tokens())
    store.ensure_account(personal, tokens())

    store.set_default(personal)

    assert store.default("anthropic") == personal


def test_default_to_unknown_account_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(AccountError, match="/accounts"):
        AccountStore(tmp_path).set_default(AccountId("anthropic", "work"))
