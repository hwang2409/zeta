"""Shared named-account login entry point for CLI and interactive surfaces."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from ..core.login_flow import (
    LoginError,
    LoginPortInUseError,
    LoginPublicationError,
    run_login,
)
from ..core.session import account_home
from ..providers.login import build_login_provider, pkce_values
from .store import AccountError, AccountId, AccountStore


def login_failure_message(exc: Exception) -> str:
    """Return a credential-safe cause and recovery action for login failures."""

    prefix = "login failed: "
    if isinstance(exc, LoginPortInUseError | LoginPublicationError):
        return prefix + str(exc)
    if isinstance(exc, AccountError):
        cause = str(exc)
        if cause == "a login for this provider is already in progress":
            cause += "; close the other login and retry"
        return prefix + cause
    if isinstance(exc, LoginError):
        return (
            prefix + "the provider rejected the login or returned an invalid callback; "
            "retry in a new browser window"
        )
    if isinstance(exc, OSError):
        return (
            prefix
            + "local login setup failed; check browser and account storage access, "
            "then retry"
        )
    if isinstance(exc, ValueError):
        return prefix + "invalid login settings; check the provider and account label"
    return prefix + f"unexpected {type(exc).__name__}; check zeta logs and retry"


def login_named_account(
    home: Path,
    provider: str,
    label: str,
    *,
    replace_existing: bool = False,
    login: Callable[..., Awaitable[Any]] = run_login,
) -> str | None:
    store = AccountStore(account_home(home))
    with store.provider_login_lock(provider):
        store.migrate()
        try:
            account = store.resolve_reference(provider, label)
        except AccountError:
            account = AccountId(provider, label)
        else:
            if not replace_existing:
                raise AccountError(
                    f"account {provider}:{label} already exists; use --replace"
                )
        return asyncio.run(
            login(build_login_provider(provider, store.home, account.name), pkce_values)
        )
