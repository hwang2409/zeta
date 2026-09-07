from __future__ import annotations

from pathlib import Path

import pytest

from zeta.providers.anthropic import (
    AnthropicApiKeyCredential,
    AnthropicAuthError,
    AnthropicCredentialStore,
)
from zeta.providers.factory import (
    API_KEY_OPT_IN_VAR,
    anthropic_api_key_store,
    build_backend,
)


def test_bare_api_key_without_opt_in_still_uses_oauth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The critical regression: ANTHROPIC_API_KEY alone must never flip auth mode."""

    monkeypatch.delenv(API_KEY_OPT_IN_VAR, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-be-ignored")

    backend, model = build_backend("claude", None, home=tmp_path)

    assert isinstance(backend.token_store, AnthropicCredentialStore)
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

    assert isinstance(backend.token_store, AnthropicCredentialStore)


def test_persisted_login_api_key_wins_over_oauth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ZETA-88 `zeta login` API-key choice is enough on its own, no env vars."""

    monkeypatch.delenv(API_KEY_OPT_IN_VAR, raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    anthropic_api_key_store(tmp_path).save("sk-ant-from-login")

    backend, _ = build_backend("claude", None, home=tmp_path)

    assert isinstance(backend.token_store, AnthropicApiKeyCredential)
    assert backend.token_store.api_key == "sk-ant-from-login"


def test_env_opt_in_outranks_persisted_login_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Automation's env-var opt-in must not depend on prior `zeta login` state."""

    monkeypatch.setenv(API_KEY_OPT_IN_VAR, "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
    anthropic_api_key_store(tmp_path).save("sk-ant-from-login")

    backend, _ = build_backend("claude", None, home=tmp_path)

    assert backend.token_store.api_key == "sk-ant-from-env"
