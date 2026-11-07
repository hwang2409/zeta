"""Synchronization helpers for login timing tests."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from urllib.parse import urlencode
from urllib.request import urlopen

import httpx
import pytest

from zeta.core import login_flow
from zeta.providers import login as provider_login


def reject_provider_token_exchange(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    *,
    invalid_response: bool,
) -> None:
    """Drive the real provider exchange against a rejected HTTP response."""

    real_client = httpx.AsyncClient
    status = 200 if invalid_response else 401

    def rejected_client(*_args: object, **_kwargs: object) -> httpx.AsyncClient:
        return real_client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    status,
                    content=b"secret-token-body",
                    request=request,
                )
            )
        )

    monkeypatch.setattr(login_flow.httpx, "AsyncClient", rejected_client)
    if provider == "anthropic":
        redirect_name = "ANTHROPIC_REDIRECT_URI"
        builder_name = "build_anthropic_authorization_url"
        path = "/callback"
    else:
        redirect_name = "DEFAULT_CODEX_REDIRECT_URI"
        builder_name = "build_codex_authorization_url"
        path = "/auth/callback"
    monkeypatch.setattr(provider_login, redirect_name, f"http://localhost:0{path}")
    real_builder = getattr(provider_login, builder_name)

    def authorize(state: str, challenge: str, redirect_uri: str) -> str:
        def callback() -> None:
            with urlopen(
                f"{redirect_uri}?{urlencode({'code': 'code', 'state': state})}",
                timeout=2,
            ) as response:
                assert response.status == 200

        threading.Thread(target=callback, daemon=True).start()
        return real_builder(state, challenge, redirect_uri)

    monkeypatch.setattr(provider_login, builder_name, authorize)


def block_loop_until_publication_decision_expires(
    monkeypatch: pytest.MonkeyPatch,
    loop: asyncio.AbstractEventLoop,
) -> tuple[Callable[[], None], threading.Event]:
    """Keep the loop blocked until the worker observes decision expiry."""

    decision_expired = threading.Event()
    loop_blocked = threading.Event()
    real_is_set = login_flow._PublicationArbiter.is_set

    def observed_is_set(arbiter: login_flow._PublicationArbiter) -> bool:
        result = real_is_set(arbiter)
        if arbiter.decision_failed:
            decision_expired.set()
        return result

    def block_loop() -> None:
        loop_blocked.set()
        assert decision_expired.wait(timeout=2)

    monkeypatch.setattr(login_flow._PublicationArbiter, "is_set", observed_is_set)
    return block_loop, loop_blocked
