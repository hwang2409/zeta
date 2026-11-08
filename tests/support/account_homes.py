from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from zeta.accounts import AccountId, AccountStore
from zeta.oauth import OAuthTokens

from .credentials import named_account_credentials


def build_account_home(
    home: Path,
    accounts: Iterable[AccountId],
    *,
    available: Iterable[AccountId] = (),
    defaults: Iterable[AccountId] = (),
) -> AccountStore:
    """Build a stored-format-valid account home with selected credentials missing."""

    available_set = set(available)
    store = AccountStore(home)
    for account in accounts:
        named_account_credentials(home, account.provider, account.name).save(
            OAuthTokens(f"access-{account.name}", "refresh", 9_999_999_999)
        )
        if account not in available_set:
            store.credential_path(account).unlink()
    for account in defaults:
        store.set_default(account)
    return store
