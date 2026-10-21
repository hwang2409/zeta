import json
from pathlib import Path

import pytest

from zeta.accounts import AccountError, AccountId, AccountStore
from zeta.oauth import OAuthTokens


def tokens() -> OAuthTokens:
    return OAuthTokens("access-secret", "refresh-secret", 9_999_999_999.0)


@pytest.mark.parametrize("provider,legacy", [("anthropic", "anthropic-oauth.json"), ("codex", "codex-oauth.json")])
def test_legacy_migration(provider: str, legacy: str, tmp_path: Path) -> None:
    tmp_path.joinpath(legacy).write_text(json.dumps(tokens().to_dict()))
    AccountStore(tmp_path).migrate()
    destination = tmp_path / "accounts" / provider / "default" / "oauth.json"
    assert json.loads(destination.read_text()) == tokens().to_dict()
    assert not tmp_path.joinpath(legacy).exists()
    assert destination.stat().st_mode & 0o777 == 0o600
    assert tmp_path.joinpath("accounts.json").stat().st_mode & 0o777 == 0o600


def test_migration_is_idempotent(tmp_path: Path) -> None:
    legacy = tmp_path / "anthropic-oauth.json"
    legacy.write_text(json.dumps(tokens().to_dict()))
    store = AccountStore(tmp_path)
    store.migrate()
    store.migrate()
    assert str(AccountId("anthropic", "default")) in {str(item) for item in store.list()}


def test_default_requires_accounts_command(tmp_path: Path) -> None:
    with pytest.raises(AccountError, match="/accounts"):
        AccountStore(tmp_path).set_default(AccountId("anthropic", "work"))


def test_named_account_and_default(tmp_path: Path) -> None:
    store = AccountStore(tmp_path)
    account = AccountId("anthropic", "Work")
    store.ensure_account(account, tokens())
    assert store.default("anthropic") == AccountId("anthropic", "work")
    with pytest.raises(AccountError, match="--replace"):
        store.ensure_account(account, tokens())


def test_account_dirs_are_private(tmp_path: Path) -> None:
    store = AccountStore(tmp_path)
    store.ensure_account(AccountId("codex", "personal"), tokens())
    assert (tmp_path / "accounts" / "codex" / "personal").stat().st_mode & 0o777 == 0o700
    assert not any("access-secret" in path.read_text() for path in tmp_path.glob("accounts.json"))
