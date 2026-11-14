from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

import zeta.providers.factory as factory_module
from tests.support.completion import completion_control
from tests.support.credentials import named_account_credentials
from zeta.accounts import AccountCredentials
from zeta.protocol.types import Message, MessageRole, TextContent
from zeta.providers.anthropic import (
    AnthropicApiKeyCredential,
    AnthropicAuthError,
    OAuthTokens,
)
from zeta.providers.codex import DEFAULT_CODEX_MODEL
from zeta.providers.factory import API_KEY_OPT_IN_VAR, build_backend

_SSE = (
    'event: message_start\n'
    'data: {"type":"message_start","message":{"id":"m","model":"c",'
    '"role":"assistant","usage":{"input_tokens":1}}}\n\n'
    'event: message_delta\n'
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
    '"usage":{"output_tokens":1}}\n\n'
    'event: message_stop\n'
    'data: {"type":"message_stop"}\n'
)


@pytest.mark.asyncio
async def test_settings_file_enables_oauth_compat_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A settings.toml opt-in makes a claude request carry the billing block,
    even with no interactive-shell env var set. Fails on main (setting ignored).
    """

    monkeypatch.delenv("ZETA_ANTHROPIC_OAUTH_COMPAT", raising=False)
    monkeypatch.delenv(API_KEY_OPT_IN_VAR, raising=False)
    (tmp_path / "settings.toml").write_text("[anthropic]\noauth_compat = true\n")
    store = named_account_credentials(tmp_path, "anthropic")
    store.save(OAuthTokens("access-test", "refresh-test", 4_000_000_000))

    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=_SSE
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    real_backend = factory_module.AnthropicBackend
    monkeypatch.setattr(
        factory_module,
        "AnthropicBackend",
        lambda **kwargs: real_backend(client=client, **kwargs),
    )

    backend, _ = await asyncio.to_thread(build_backend, "claude", None, home=tmp_path)
    try:
        await anext(
            backend.complete(
                [Message(MessageRole.USER, [TextContent("abcdefghijklmnop")])],
                [],
                control=completion_control(),
            )
        )
    finally:
        await client.aclose()

    system = json.loads(requests[0].content)["system"]
    assert system[0]["text"].startswith("x-anthropic-billing-header:")
    assert requests[0].headers["user-agent"] == "claude-cli/2.1.280"


def test_codex_default_matches_backend_default(tmp_path: Path) -> None:
    backend, model = build_backend("codex", None, home=tmp_path)

    assert model == backend.model == DEFAULT_CODEX_MODEL


def test_bare_api_key_without_opt_in_still_uses_oauth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The critical regression: ANTHROPIC_API_KEY alone must never flip auth mode."""

    monkeypatch.delenv(API_KEY_OPT_IN_VAR, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-be-ignored")

    backend, model = build_backend("claude", None, home=tmp_path)

    assert isinstance(backend.token_store, AccountCredentials)
    assert model == "claude-sonnet-4-6"


def test_opt_in_with_key_selects_api_key_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(API_KEY_OPT_IN_VAR, "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")

    backend, _ = build_backend("claude", None, home=tmp_path)

    assert isinstance(backend.token_store, AnthropicApiKeyCredential)
    assert backend.token_store.api_key == "sk-ant-test-key"


def test_opt_in_without_key_fails_loudly_naming_both_routes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(API_KEY_OPT_IN_VAR, "1")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(AnthropicAuthError, match="ANTHROPIC_API_KEY") as excinfo:
        build_backend("claude", None, home=tmp_path)

    assert "zeta login" in str(excinfo.value)
    assert API_KEY_OPT_IN_VAR in str(excinfo.value)


def test_opt_in_value_must_be_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A truthy-looking but non-canonical value does not opt in (fail-safe default)."""

    monkeypatch.setenv(API_KEY_OPT_IN_VAR, "true")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")

    backend, _ = build_backend("claude", None, home=tmp_path)

    assert isinstance(backend.token_store, AccountCredentials)
