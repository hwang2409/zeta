"""Own authentication and account state for one loaded session."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .store import AccountError, AccountId, AccountStore

if TYPE_CHECKING:
    from ..core.session import SessionManager, SessionMetadata


def account_provider(provider: str) -> str | None:
    if provider == "claude":
        return "anthropic"
    return provider if provider in {"anthropic", "codex"} else None


def api_key_mode(provider: str) -> bool:
    return provider in {"claude", "anthropic"} and os.environ.get(
        "ZETA_ALLOW_API_KEY"
    ) == "1"


@dataclass(slots=True)
class SessionAccountContext:
    """Resolve credentials and keep provider transitions local to one session."""

    home: Path
    store: AccountStore
    manager: SessionManager | None
    metadata: SessionMetadata
    _pins: dict[str, AccountId] = field(default_factory=dict)
    _frozen: bool = False
    _active_provider: str | None = None
    _switch_sequences: dict[str, int] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        home: str | Path,
        manager: SessionManager,
        metadata: SessionMetadata,
        *,
        validate: bool = True,
    ) -> SessionAccountContext:
        path = Path(home)
        pins = {
            provider: AccountId(*value.split(":", 1))
            for provider, value in metadata.selected_accounts.items()
        }
        context = cls(
            path,
            AccountStore(path),
            manager,
            metadata,
            pins,
            _active_provider=account_provider(metadata.provider),
            _switch_sequences=dict(metadata.account_switch_sequences),
        )
        if validate:
            context._validate_pins()
        if (
            validate
            and context._active_provider is not None
            and not api_key_mode(metadata.provider)
        ):
            context.resolve(metadata.provider)
        return context

    @property
    def auth_mode(self) -> str:
        return (
            "api-key"
            if api_key_mode(self._active_provider or self.metadata.provider)
            else "named-account"
        )

    @property
    def switching_unavailable(self) -> str | None:
        if self.auth_mode == "api-key":
            return "this session uses an API key; account switching is unavailable"
        return None

    @property
    def active_provider(self) -> str | None:
        return self._active_provider

    @property
    def switch_sequence(self) -> int:
        if self._active_provider is None:
            return 0
        return self._switch_sequences.get(self._active_provider, 0)

    def _validate_pins(self) -> None:
        for provider, account in self._pins.items():
            if api_key_mode(provider):
                continue
            if not self.store.contains(account):
                raise AccountError(
                    f"session account {account} no longer exists; choose another with /accounts"
                )

    def resolve(self, provider: str) -> AccountId:
        canonical = account_provider(provider)
        if canonical is None:
            raise ValueError("provider does not use a named account")
        if api_key_mode(provider):
            raise ValueError("API-key authentication does not use a named account")
        selected = self._pins.get(canonical)
        if selected is not None:
            if not self._frozen and not self.store.contains(selected):
                raise AccountError(
                    f"session account {selected} no longer exists; choose another with /accounts"
                )
            return selected
        if self._frozen:
            raise AccountError(
                f"child account snapshot has no {canonical} account; start a new child"
            )
        selected = self.store.resolve_account(provider)
        if self.store.contains(selected):
            self._pins[canonical] = selected
            assert self.manager is not None
            self.manager.record_account_selection(self.metadata, selected, boundary=0)
        return selected

    def backend_kwargs(self, provider: str) -> dict[str, object]:
        result: dict[str, object] = {"home": self.home}
        canonical = account_provider(provider)
        if canonical is not None and not api_key_mode(provider):
            self.resolve(provider)
            if (account := self._pins.get(canonical)) is not None:
                result["account"] = account
        return result

    def snapshot(self) -> dict[str, AccountId]:
        return dict(self._pins)

    def child_snapshot(self, provider: str | None) -> SessionAccountContext:
        needed = {"codex"}
        canonical = account_provider(provider or "")
        if canonical is not None:
            needed.add(canonical)
        child_pins = self.snapshot()
        for name in needed:
            source = "claude" if name == "anthropic" else name
            if not api_key_mode(source):
                child_pins[name] = self.resolve(source)
        return SessionAccountContext(
            self.home,
            self.store,
            None,
            self.metadata,
            child_pins,
            _frozen=True,
            _active_provider=canonical,
            _switch_sequences=dict(self._switch_sequences),
        )

    def switch(self, account: AccountId, *, boundary: int) -> None:
        if self.switching_unavailable is not None:
            raise AccountError(self.switching_unavailable)
        if self._frozen:
            raise AccountError("child account snapshots cannot be changed")
        if not self.store.contains(account):
            raise AccountError(f"account {account} does not exist; use /accounts")
        assert self.manager is not None
        self.manager.record_account_selection(
            self.metadata, account, boundary=boundary, switched=True
        )
        self._pins = {**self._pins, account.provider: account}
        self._switch_sequences = dict(self.metadata.account_switch_sequences)

    def activate(self, loop: Any, backend: Any, provider: str) -> tuple[Any, str | None]:
        previous = (loop.backend, self._active_provider)
        loop.set_backend(backend)
        self._active_provider = account_provider(provider)
        return previous

    def restore(self, loop: Any, state: tuple[Any, str | None]) -> None:
        backend, self._active_provider = state
        loop.set_backend(backend)
