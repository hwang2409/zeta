"""Test generation readiness without admitting a child request."""

from __future__ import annotations

import hashlib
from contextlib import aclosing
from datetime import UTC, datetime
from enum import Enum, auto

from ..accounts.store import AccountCredentials
from ..accounts.usage import AccountUsage, UsageState, normalize_usage
from ..protocol.types import Message, MessageRole, StreamEventType, TextContent
from .anthropic import AnthropicApiKeyCredential
from .completion import retryable_provider_error


class Readiness(Enum):
    READY = auto()
    RETRY = auto()
    STOP = auto()


def recovery_account(backend):
    credentials = getattr(backend, "token_store", None)
    if isinstance(credentials, AccountCredentials):
        return credentials.account
    if isinstance(credentials, AnthropicApiKeyCredential):
        return "anthropic", hashlib.sha256(credentials.api_key.encode()).digest()
    return backend.provider, getattr(backend, "base_url", None)


async def _exhausted(backend, credentials: AccountCredentials) -> bool:
    usage = AccountUsage(credentials.store)
    async with backend._lease.borrow("provider recovery usage") as client:
        response = await usage._request(credentials.account, client)
    if response.status_code in {401, 403}:
        response.raise_for_status()
    if not response.is_success:
        return False
    now = datetime.now(UTC)
    try:
        snapshot = normalize_usage(credentials.account.provider, response.json(), now)
    except (TypeError, ValueError):
        return False
    return UsageState(snapshot, "now").known_exhausted(now)


async def probe_readiness(backend) -> Readiness:
    try:
        if isinstance(credentials := getattr(backend, "token_store", None), AccountCredentials) and await _exhausted(backend, credentials):
            return Readiness.RETRY
        messages = (Message(MessageRole.USER, [TextContent("x")]),)
        async with aclosing(backend._complete_once(messages, (), probe=True)) as events:
            async for event in events:
                if event.type == StreamEventType.ERROR:
                    return Readiness.RETRY if retryable_provider_error(event.error, event.data) else Readiness.STOP
                if event.type == StreamEventType.MESSAGE_UPDATE and (event.delta or event.content):
                    return Readiness.READY
    except Exception as exc:  # noqa: BLE001 - classify only; never persist a provider's error body
        status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
        if status in {401, 403}:
            return Readiness.STOP
        return Readiness.RETRY if retryable_provider_error(exc) else Readiness.STOP
    return Readiness.RETRY
