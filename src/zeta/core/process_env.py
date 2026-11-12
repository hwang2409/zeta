"""Environment policy for child processes."""

from __future__ import annotations

import os
import tempfile
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


def subprocess_env(overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return a scrubbed parent environment with explicit overrides applied."""

    environment = {
        name: value
        for name, value in os.environ.items()
        if _normalize_env_name(name) not in _DENIED_ENV_NAMES
    }
    # Child processes launched by tools are distinguishable from direct user
    # commands. This blocks accidental model authorization through the shell.
    environment["ZETA_TOOL_SUBPROCESS"] = "1"
    environment.setdefault("GIT_TERMINAL_PROMPT", "0")
    environment.setdefault("DEBIAN_FRONTEND", "noninteractive")
    environment.setdefault("PAGER", "cat")
    environment.setdefault("GIT_PAGER", "cat")
    # A tool shell inherits the user's $TMUX/$TMUX_PANE, so a bare `tmux ...`
    # (including `kill-server`) controls the host server that hosts every zeta
    # session. Drop those and point tmux at a private per-home socket dir so a
    # bare agent tmux only reaches its own server. To drive the host server on
    # purpose, target it explicitly, e.g. `tmux -S /private/tmp/tmux-$(id -u)/default`.
    environment.pop("TMUX", None)
    environment.pop("TMUX_PANE", None)
    if "TMUX_TMPDIR" not in environment:
        environment["TMUX_TMPDIR"] = _tmux_tmpdir()
    if overrides is not None:
        environment.update(overrides)
    return environment


def _tmux_tmpdir() -> str:
    """Return a private, per-home tmux socket dir, creating it 0700.

    On any filesystem error (read-only home, a symlink or file at the path, an
    unowned dir) fall back to a per-process temp dir so a tool spawn never
    fails and still never reaches the host tmux server.
    """

    from .session import env_home

    try:
        path = env_home() / "run" / "tmux"
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o700)
        return str(path)
    except OSError:
        global _FALLBACK_TMUX_TMPDIR
        if _FALLBACK_TMUX_TMPDIR is None:
            _FALLBACK_TMUX_TMPDIR = tempfile.mkdtemp(prefix="zeta-tmux-")
        return _FALLBACK_TMUX_TMPDIR


_FALLBACK_TMUX_TMPDIR: str | None = None
