from __future__ import annotations

import hashlib
import stat
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


def home_snapshot(home: Path) -> tuple[dict[str, tuple[str, int]], set[tuple[str, int]]]:
    """Every file (sha256, mode) and directory (mode) under home, by relative path."""

    files: dict[str, tuple[str, int]] = {}
    directories: set[tuple[str, int]] = set()
    for path in home.rglob("*"):
        relative = str(path.relative_to(home))
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            directories.add((relative, stat.S_IMODE(mode)))
        else:
            files[relative] = (hashlib.sha256(path.read_bytes()).hexdigest(), mode)
    return files, directories
