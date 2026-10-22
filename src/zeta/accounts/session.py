"""Resolve and persist named accounts for one session."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .store import AccountError, AccountId, AccountStore

if TYPE_CHECKING:
    from ..core.session import SessionManager, SessionMetadata


def account_provider(provider: str) -> str | None:
    if provider == "claude":
        return "anthropic"
    return provider if provider in {"anthropic", "codex"} else None


@dataclass(slots=True)
class SessionAccountResolver:
    """Own the pinned named-account map for one loaded session."""

    store: AccountStore
    manager: SessionManager
    metadata: SessionMetadata

    @classmethod
    def load(
        cls,
        home: str | Path,
        manager: SessionManager,
        metadata: SessionMetadata,
    ) -> SessionAccountResolver:
        resolver = cls(AccountStore(home), manager, metadata)
        for value in metadata.selected_accounts.values():
            account = AccountId(*value.split(":", 1))
            if not resolver.store.contains(account):
                raise AccountError(
                    f"session account {account} no longer exists; choose another with /accounts"
                )
        provider = account_provider(metadata.provider)
        if provider is not None:
            resolver.resolve(provider)
        return resolver

    def resolve(self, provider: str) -> AccountId:
        canonical = account_provider(provider)
        if canonical is None:
            raise ValueError("provider does not use a named account")
        selected = self.metadata.selected_accounts.get(canonical)
        if selected is not None:
            account = AccountId(*selected.split(":", 1))
            if not self.store.contains(account):
                raise AccountError(
                    f"session account {account} no longer exists; choose another with /accounts"
                )
            return account
        self.store.migrate()
        account = self.store.default(canonical)
        if account is None:
            return AccountId(canonical, "default")
        self.manager.record_account_selection(self.metadata, account, boundary=0)
        return account

    def snapshot(self) -> dict[str, AccountId]:
        return {
            provider: AccountId(*value.split(":", 1))
            for provider, value in self.metadata.selected_accounts.items()
        }

    def switch(self, account: AccountId, *, boundary: int) -> None:
        if not self.store.contains(account):
            raise AccountError(f"account {account} does not exist; use /accounts")
        self.manager.record_account_selection(
            self.metadata, account, boundary=boundary, switched=True
        )
