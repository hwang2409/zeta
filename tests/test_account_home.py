import asyncio
import shutil
from pathlib import Path

import httpx
import pytest

import zeta.core.session as core_session
from tests.support.credentials import named_account_credentials
from zeta.accounts import AccountId, AccountStore
from zeta.config.settings import load_settings
from zeta.core.session import SessionManager, env_home
from zeta.oauth import OAuthTokens
from zeta.project_registry import ProjectRegistry
from zeta.providers.factory import build_backend, credential_store


def _tokens(access: str, refresh: str, expires_at: float = 4_000_000_000) -> OAuthTokens:
    return OAuthTokens(access, refresh, expires_at)


def test_provider_credentials_use_account_home_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zeta_home = tmp_path / "zeta"
    account_home = tmp_path / "accounts"
    expected = _tokens("access-b", "refresh-b")
    named_account_credentials(account_home, "anthropic", "work").save(expected)
    monkeypatch.setenv("ZETA_HOME", str(zeta_home))
    monkeypatch.setenv("ZETA_ACCOUNT_HOME", str(account_home))

    backend, _ = build_backend("claude", None, home=zeta_home)

    assert backend.token_store.read() == expected
    assert backend.token_store.path == AccountStore(account_home).credential_path(
        AccountId("anthropic", "work")
    )
    assert not (zeta_home / "accounts.json").exists()
    assert not (zeta_home / "accounts").exists()


@pytest.mark.asyncio
async def test_refresh_writes_token_and_lock_to_account_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zeta_home = tmp_path / "zeta"
    account_home = tmp_path / "accounts"
    expired = _tokens("old-access", "old-refresh", 1.0)
    fresh = _tokens("new-access", "new-refresh")
    named_account_credentials(account_home, "anthropic", "work").save(expired)
    monkeypatch.setenv("ZETA_HOME", str(zeta_home))
    monkeypatch.setenv("ZETA_ACCOUNT_HOME", str(account_home))
    store = await asyncio.to_thread(credential_store, "claude", home=zeta_home)
    assert store is not None
    shutil.rmtree(account_home / "account-locks")

    async def return_fresh(_refresh_token: str, _client: httpx.AsyncClient) -> OAuthTokens:
        return fresh

    monkeypatch.setattr(store, "refresh", return_fresh)
    async with httpx.AsyncClient() as client:
        assert await store.access_token(client) == fresh.access_token

    assert await asyncio.to_thread(store.read) == fresh
    assert store.path.is_relative_to(account_home)
    assert (account_home / "account-locks/anthropic/work/lifecycle.lock").is_file()
    assert not (zeta_home / "accounts").exists()
    assert not (zeta_home / "account-locks").exists()


def test_non_account_state_stays_in_zeta_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zeta_home = tmp_path / "zeta"
    account_home = tmp_path / "accounts"
    zeta_home.mkdir()
    account_home.mkdir()
    (zeta_home / "settings.toml").write_text('provider = "codex"\n')
    (account_home / "settings.toml").write_text('provider = "anthropic"\n')
    monkeypatch.setenv("ZETA_HOME", str(zeta_home))
    monkeypatch.setenv("ZETA_ACCOUNT_HOME", str(account_home))

    assert core_session.account_home() == account_home
    assert env_home() == zeta_home
    opened = SessionManager(env_home()).create(
        provider="codex", model="test", cwd=tmp_path
    )
    opened.store.close()
    ProjectRegistry(env_home() / "projects").create_project("demo", "test")

    assert load_settings(home=env_home()).settings.provider == "codex"
    assert (zeta_home / "sessions").is_dir()
    assert (zeta_home / "projects").is_dir()
    assert not (account_home / "sessions").exists()
    assert not (account_home / "projects").exists()


def test_account_home_expands_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ZETA_ACCOUNT_HOME", "~/shared-accounts")

    assert core_session.account_home() == tmp_path / "shared-accounts"


@pytest.mark.parametrize("value", ["", "relative/accounts"])
def test_invalid_account_home_fails_loudly(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZETA_ACCOUNT_HOME", value)

    with pytest.raises(ValueError, match="ZETA_ACCOUNT_HOME must be an absolute path"):
        core_session.account_home()
