from __future__ import annotations

from pathlib import Path
from threading import Thread
from typing import Any

from zeta.accounts import AccountCredentials, AccountId, AccountStore
from zeta.codex import CODEX_TOKEN_URL, CodexCredentialDriver
from zeta.providers.anthropic import TOKEN_URL, AnthropicCredentialDriver


class FixtureAccountCredentials(AccountCredentials):
    """Keep synchronous credential setup out of async test event loops."""

    def _thread_call(self, method: Any, *args: Any, **kwargs: Any) -> Any:
        result: list[Any] = []
        error: list[Exception] = []

        def run() -> None:
            try:
                result.append(method(*args, **kwargs))
            except BaseException as exc:  # noqa: BLE001 - preserve test exceptions
                error.append(exc)

        thread = Thread(target=run)
        thread.start()
        thread.join()
        if error:
            raise error[0]
        return result[0] if result else None

    def read(self, **kwargs: Any):
        return self._thread_call(super().read, **kwargs)

    def save(self, *args: Any, **kwargs: Any) -> None:
        self._thread_call(super().save, *args, **kwargs)


def _test_home(path: str | Path) -> Path:
    credential_path = Path(path)
    return credential_path.parent / f".{credential_path.name}-account"


def named_account_credentials(
    home: str | Path,
    provider: str,
    name: str = "default",
    *,
    token_url: str | None = None,
) -> AccountCredentials:
    account = AccountId(provider, name)
    driver = (
        AnthropicCredentialDriver(token_url=token_url or TOKEN_URL)
        if provider == "anthropic"
        else CodexCredentialDriver(token_url=token_url or CODEX_TOKEN_URL)
    )
    return FixtureAccountCredentials(AccountStore(home), account, driver)


def anthropic_credentials(
    path: str | Path,
    *,
    token_url: str = TOKEN_URL,
) -> AccountCredentials:
    store = AccountStore(_test_home(path))
    return FixtureAccountCredentials(
        store,
        AccountId("anthropic", "default"),
        AnthropicCredentialDriver(token_url=token_url),
    )


def codex_credentials(
    path: str | Path,
    *,
    token_url: str = CODEX_TOKEN_URL,
) -> AccountCredentials:
    store = AccountStore(_test_home(path))
    return FixtureAccountCredentials(
        store,
        AccountId("codex", "default"),
        CodexCredentialDriver(token_url=token_url),
    )
