from __future__ import annotations

from pathlib import Path

from zeta.accounts import AccountCredentials, AccountId, AccountStore
from zeta.codex import CODEX_TOKEN_URL, CodexCredentialDriver
from zeta.providers.anthropic import TOKEN_URL, AnthropicCredentialDriver


def _test_home(path: str | Path) -> Path:
    credential_path = Path(path)
    return credential_path.parent / f".{credential_path.name}-account"


def anthropic_credentials(
    path: str | Path,
    *,
    token_url: str = TOKEN_URL,
) -> AccountCredentials:
    store = AccountStore(_test_home(path))
    return AccountCredentials(
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
    return AccountCredentials(
        store,
        AccountId("codex", "default"),
        CodexCredentialDriver(token_url=token_url),
    )
