"""Runtime provider model catalogs used by the terminal UI."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx

from ..core.session import env_home
from ..providers.anthropic import AnthropicCredentialStore


MODEL_CATALOGS = {"fake": frozenset({"offline", "faster"})}
_MODEL_CATALOG_TIMEOUT = 2.0
_WRONG_PROVIDER_PREFIXES = {
    "claude": ("gpt-", "o1", "o3", "o4", "gemini-", "llama-"),
    "codex": ("claude-", "gemini-", "llama-"),
    "fake": ("claude-", "gpt-", "gemini-", "llama-"),
}


def validate_model_name(provider: str, model: str) -> None:
    if not model or any(character.isspace() for character in model):
        raise ValueError("model must be one nonempty word")
    if model.startswith(_WRONG_PROVIDER_PREFIXES.get(provider, ())):
        raise ValueError(f"model {model!r} has a wrong-provider prefix for {provider}")


def _model_names_from_payload(payload: Any) -> frozenset[str] | None:
    if not isinstance(payload, dict):
        return None
    entries = next(
        (payload[key] for key in ("data", "models") if key in payload), None
    )
    if not isinstance(entries, list):
        return None
    names = {
        item
        if isinstance(item, str)
        else next(
            (
                item[key]
                for key in ("id", "slug", "model", "name")
                if isinstance(item, dict) and isinstance(item.get(key), str)
            ),
            None,
        )
        for item in entries
    }
    return frozenset(name for name in names if isinstance(name, str) and name)


def _load_codex_model_catalog() -> frozenset[str] | None:
    configured_home = os.environ.get("CODEX_HOME")
    codex_home = Path(configured_home) if configured_home else Path.home() / ".codex"
    candidates = (
        codex_home / "models_cache.json",
        codex_home / "models.json",
        Path.home() / ".config" / "codex" / "models.json",
        Path(__file__).with_name("models.json"),
    )
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, list):
            payload = {"data": payload}
        catalog = _model_names_from_payload(payload)
        if catalog is not None:
            return catalog
    return None


def load_model_catalog(
    provider: str, *, home: str | Path | None = None
) -> frozenset[str] | None:
    """Load a provider catalog, returning None when it cannot be obtained."""

    if provider == "fake":
        return MODEL_CATALOGS["fake"]
    if provider == "codex":
        return _load_codex_model_catalog()
    if provider != "claude":
        return None

    auth_home = Path(home) if home is not None else env_home()
    try:
        token_store = AnthropicCredentialStore(auth_home / "anthropic-oauth.json")
        tokens = token_store.read() or token_store.bootstrap()
        if tokens is None or not tokens.is_valid():
            return None
        response = httpx.get(
            "https://api.anthropic.com/v1/models",
            headers={
                "accept": "application/json",
                "anthropic-version": "2023-06-01",
                "authorization": f"Bearer {tokens.access_token}",
            },
            timeout=_MODEL_CATALOG_TIMEOUT,
        )
        response.raise_for_status()
        return _model_names_from_payload(response.json())
    except (
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
        httpx.HTTPError,
    ):
        return None
