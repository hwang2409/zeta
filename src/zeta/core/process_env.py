"""Environment policy for child processes."""

from __future__ import annotations

import os
from collections.abc import Mapping

# These names cover the sensitive-name basket used by auth redaction, plus
# credentials that commonly appear in a developer environment.
CREDENTIAL_ENV_NAMES = (
    "AUTHORIZATION",
    "PROXY_AUTHORIZATION",
    "WWW_AUTHENTICATE",
    "AUTHENTICATION",
    "X_API_KEY",
    "X_AUTH_TOKEN",
    "X_AMZ_SECURITY_TOKEN",
    "X_AMZ_SIGNATURE",
    "X_GOOG_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "SEC_WEBSOCKET_KEY",
    "SEC_WEBSOCKET_ACCEPT",
    "COOKIE",
    "SET_COOKIE",
    "PASSWORD",
    "PASSWD",
    "SECRET",
    "CLIENT_SECRET",
    "ACCESS_TOKEN",
    "REFRESH_TOKEN",
    "ID_TOKEN",
    "TOKEN",
    "API_KEY",
    "APIKEY",
    "PERSONAL_ACCESS_TOKEN",
    "BEARER_TOKEN",
    "SIGNING_SECRET",
    "WEBHOOK_SECRET",
    "SECRET_KEY",
    "CREDENTIALS",
    "WEBSOCKET_KEY",
    "FORM_ID_TOKEN",
    "SESSION_TOKEN",
    "PRIVATE_KEY",
    "CLIENT_ASSERTION",
    "DEVICE_CODE",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "X_AMZ_CREDENTIAL",
    "X_GOOG_CREDENTIAL",
    "X_GOOG_SIGNATURE",
    "AUTH_TOKEN",
    "API_SECRET",
    "CONSUMER_SECRET",
    "SIGNING_KEY",
    "GITHUB_TOKEN",
    "SESSION_COOKIE",
    "MY_PASSWORD",
    "OAUTH_BEARER",
    "ZETA_ALLOW_API_KEY",
)


def _normalize_env_name(name: str) -> str:
    return name.upper().replace("_", "").replace("-", "")


_DENIED_ENV_NAMES = frozenset(_normalize_env_name(name) for name in CREDENTIAL_ENV_NAMES)
_DENIED_ENV_NAMES |= {_normalize_env_name("ZETA_HOME")}


def subprocess_env(overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return a scrubbed parent environment with explicit overrides applied."""

    environment = {
        name: value
        for name, value in os.environ.items()
        if _normalize_env_name(name) not in _DENIED_ENV_NAMES
    }
    if overrides is not None:
        environment.update(overrides)
    return environment
