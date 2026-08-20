"""Provider-neutral OAuth storage for zeta."""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import secrets
import time
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote_plus

import httpx


_SENSITIVE_ERROR_NAMES = frozenset(
    {
        "authorization",
        "proxy_authorization",
        "www_authenticate",
        "authentication",
        "x_api_key",
        "x_auth_token",
        "x_amz_security_token",
        "x_amz_signature",
        "x_goog_api_key",
        "anthropic_api_key",
        "openai_api_key",
        "sec_websocket_key",
        "sec_websocket_accept",
        "cookie",
        "set_cookie",
        "password",
        "passwd",
        "secret",
        "client_secret",
        "access_token",
        "refresh_token",
        "id_token",
        "form_id_token",
        "token",
        "api_key",
        "apikey",
        "websocket_key",
    }
)
_FIELD_NAME_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.%+-"
)
_BARE_TOKEN_PREFIXES = ("access-token-", "access_token-", "refresh-token-", "refresh_token-")
_TOKEN_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)


def _is_sensitive_name(name: str) -> bool:
    normalized = unquote_plus(name).strip().strip("\"'").lower().replace("-", "_")
    return normalized in _SENSITIVE_ERROR_NAMES


def _line_parts(line: str) -> tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith(("\n", "\r")):
        return line[:-1], line[-1]
    return line, ""


def _field_name_before(line: str, separator: int) -> str:
    prefix = line[:separator].rstrip()
    prefix = prefix.rstrip("\"'")
    end = len(prefix)
    start = end
    while start and prefix[start - 1] in _FIELD_NAME_CHARS:
        start -= 1
    return prefix[start:end]


def _sensitive_separator(line: str) -> tuple[int, str] | None:
    for index, character in enumerate(line):
        if character not in ":=":
            continue
        name = _field_name_before(line, index)
        if _is_sensitive_name(name):
            return index, character
    return None


def _looks_like_header(line: str) -> bool:
    separator = line.find(":")
    if separator <= 0:
        return False
    name = line[:separator].strip().strip("\"'")
    return bool(name) and all(character in _FIELD_NAME_CHARS for character in name)


def _multipart_name(line: str) -> str | None:
    if "content-disposition" not in line.lower():
        return None
    for segment in line.split(";"):
        key, separator, value = segment.partition("=")
        if separator and key.strip().lower() == "name":
            return value.strip().strip("\"'")
    return None


def _redact_multipart(text: str) -> str:
    lines = text.splitlines(keepends=True)
    result: list[str] = []
    in_headers = True
    sensitive_part = False
    for line in lines:
        content, ending = _line_parts(line)
        if content.lstrip().startswith("--"):
            in_headers = True
            sensitive_part = False
            result.append(line)
        elif in_headers:
            result.append(line)
            name = _multipart_name(content)
            if name is not None:
                sensitive_part = _is_sensitive_name(name)
            if not content.strip():
                in_headers = False
        elif sensitive_part:
            result.append(f"[redacted]{ending}")
        else:
            result.append(line)
    return "".join(result)


def _redact_error_text(text: str) -> str:
    if "content-disposition" in text.lower():
        text = _redact_multipart(text)

    def redact_bare_tokens(line: str) -> str:
        lower = line.lower()
        if not any(prefix in lower for prefix in _BARE_TOKEN_PREFIXES):
            return line
        result: list[str] = []
        cursor = 0
        while cursor < len(line):
            prefix = next(
                (
                    candidate
                    for candidate in _BARE_TOKEN_PREFIXES
                    if lower.startswith(candidate, cursor)
                    and (cursor == 0 or line[cursor - 1] not in _TOKEN_CHARS)
                ),
                None,
            )
            if prefix is None:
                result.append(line[cursor])
                cursor += 1
                continue
            end = cursor + len(prefix)
            while end < len(line) and line[end] in _TOKEN_CHARS:
                end += 1
            result.append("[redacted]")
            cursor = end
        return "".join(result)

    result: list[str] = []
    redact_continuation = False
    position = 0
    while position < len(text):
        line_end = text.find("\n", position)
        if line_end == -1:
            line_end = len(text)
        else:
            line_end += 1
        line = text[position:line_end]
        content, ending = _line_parts(line)
        if redact_continuation:
            if text.find(":", position) == -1:
                result.append("[redacted]")
                break
            if not content.strip():
                redact_continuation = False
                result.append(line)
                position = line_end
                continue
            if _looks_like_header(content):
                redact_continuation = False
            else:
                result.append(f"[redacted]{ending}")
                position = line_end
                continue
        found = _sensitive_separator(content)
        if found is None:
            result.append(redact_bare_tokens(line))
        else:
            separator, kind = found
            result.append(f"{content[: separator + 1]}[redacted]{ending}")
            redact_continuation = kind == ":"
        position = line_end
    return "".join(result)


def _redact_error_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: "[redacted]"
            if isinstance(key, str) and _is_sensitive_name(key)
            else _redact_error_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_error_value(item) for item in value]
    if isinstance(value, str):
        return _redact_error_text(value)
    return value


def error_body_excerpt(body: bytes, *, limit: int = 300) -> str:
    """Return a short, whitespace-collapsed provider error without secrets."""

    try:
        value = _redact_error_value(json.loads(body))
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        text = body.decode("utf-8", errors="replace")
        text = _redact_error_text(text)
    return " ".join(text.split())[:limit]


@dataclass(frozen=True, slots=True)
class OAuthTokens:
    access_token: str
    refresh_token: str
    expires_at: float

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        error_type: type[RuntimeError],
    ) -> OAuthTokens:
        access_token = _first_string(value, "access_token", "accessToken", "access")
        refresh_token = _first_string(value, "refresh_token", "refreshToken", "refresh")
        expires_at = value.get("expires_at", value.get("expiresAt", value.get("expires", 0)))
        if not access_token or not refresh_token:
            raise error_type("OAuth credentials are incomplete")
        if type(expires_at) not in {int, float}:
            raise error_type("OAuth expiry is invalid")
        if expires_at > 100_000_000_000:
            expires_at /= 1000
        return cls(access_token, refresh_token, float(expires_at))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
        }

    def is_valid(self, *, skew: float = 60) -> bool:
        return self.expires_at > time.time() + skew


def _first_string(value: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        candidate = value.get(key)
        if type(candidate) is str and candidate:
            return candidate
    return None


class OAuthCredentialStore:
    """Own one zeta OAuth file and serialize refreshes across processes."""

    auth_error_type: type[RuntimeError]
    http_error_type: type[RuntimeError]
    provider_label: str

    def __init__(self, path: str | Path, *, token_url: str) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_name(f".{self.path.name}.lock")
        self.token_url = token_url
        self._async_refresh_lock = asyncio.Lock()

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        with self.lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def read(self) -> OAuthTokens | None:
        with self._lock():
            return self._read_unlocked()

    def _read_unlocked(self) -> OAuthTokens | None:
        if not self.path.exists():
            return None
        try:
            with self.path.open() as handle:
                value = json.load(handle)
            os.chmod(self.path, 0o600)
            if not isinstance(value, Mapping):
                raise TypeError
            return OAuthTokens.from_mapping(value, error_type=self.auth_error_type)
        except self.auth_error_type:
            raise
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise self.auth_error_type(
                f"zeta's {self.provider_label} OAuth store is invalid"
            ) from exc

    def save(self, tokens: OAuthTokens) -> None:
        with self._lock():
            self._save_unlocked(tokens)

    def _save_unlocked(self, tokens: OAuthTokens) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        temporary = self.path.with_name(f".{self.path.name}.{secrets.token_hex(8)}.tmp")
        try:
            with temporary.open("w") as handle:
                json.dump(tokens.to_dict(), handle, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.path)
            os.chmod(self.path, 0o600)
        finally:
            temporary.unlink(missing_ok=True)

    async def access_token(self, client: httpx.AsyncClient) -> str:
        async with self._async_refresh_lock, self._async_refresh_lock_file():
            tokens = self._read_unlocked()
            from_bootstrap = tokens is None
            if tokens is None:
                tokens = self.bootstrap()
                if tokens is None:
                    raise self.auth_error_type(
                        f"no {self.provider_label} OAuth login found; log in first"
                    )
            if tokens.is_valid():
                if from_bootstrap:
                    self._save_unlocked(tokens)
                return tokens.access_token
            refreshed = await self.refresh(tokens.refresh_token, client)
            self._save_unlocked(refreshed)
            return refreshed.access_token

    @asynccontextmanager
    async def _async_refresh_lock_file(self) -> AsyncIterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        handle = self.lock_path.open("a+")
        try:
            await asyncio.to_thread(fcntl.flock, handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                await asyncio.to_thread(fcntl.flock, handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def bootstrap(self) -> OAuthTokens | None:
        raise NotImplementedError

    async def refresh(self, refresh_token: str, client: httpx.AsyncClient) -> OAuthTokens:
        raise NotImplementedError
