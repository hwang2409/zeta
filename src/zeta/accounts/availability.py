from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .store import AccountId, AccountStore


@dataclass(frozen=True, slots=True)
class AccountDisplay:
    account: AccountId
    label: str
    detail: str | None = None
    usage: Any = None
    available: bool = True

    def __str__(self) -> str:
        return f"{self.account.provider}:{self.label}"


def catalog_display(value: Mapping[str, Any]) -> list[AccountDisplay]:
    from .store import AccountId

    return [
        AccountDisplay(
            AccountId(*key.split(":", 1)),
            value["labels"].get(key, key.split(":", 1)[1]),
        )
        for key in value["accounts"]
    ]


def list_display(store: AccountStore) -> list[AccountDisplay]:
    from .store import _read_account_tokens
    from .usage import read_usage

    result = []
    catalog = store.list_catalog()
    with store._home() as home_fd:
        for item in catalog:
            tokens = _read_account_tokens(home_fd, item.account)
            details = (
                tuple(filter(None, (tokens.email, tokens.workspace))) if tokens else ()
            )
            result.append(
                AccountDisplay(
                    item.account,
                    item.label,
                    " · ".join(details) or None,
                    read_usage(store, item.account),
                    tokens is not None,
                )
            )
    return result
