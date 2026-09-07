"""Provider-neutral OAuth storage for zeta."""

from __future__ import annotations

import asyncio
import base64
import fcntl
import hashlib
import io
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
        "proxyauthorization",
        "wwwauthenticate",
        "authentication",
        "xapikey",
        "xauthtoken",
        "xamzsecuritytoken",
        "xamzsignature",
        "xgoogapikey",
        "anthropicapikey",
        "openaiapikey",
        "secwebsocketkey",
        "secwebsocketaccept",
        "cookie",
        "setcookie",
        "password",
        "passwd",
        "secret",
        "clientsecret",
        "accesstoken",
        "refreshtoken",
        "idtoken",
        "formidtoken",
        "sessiontoken",
        "privatekey",
        "clientassertion",
        "devicecode",
        "awsaccesskeyid",
        "awssecretaccesskey",
        "xamzcredential",
        "xgoogcredential",
        "xgoogsignature",
        "authtoken",
        "apisecret",
        "consumersecret",
        "signingkey",
        "credentials",
        "personalaccesstoken",
        "bearertoken",
        "signingsecret",
        "webhooksecret",
        "secretkey",
        "token",
        "apikey",
        "websocketkey",
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
    decoded = unquote_plus(name).strip().strip("\"'")
    normalized: list[str] = []
    for index, character in enumerate(decoded):
        if not character.isalnum():
            continue
        if character.isupper() and index:
            previous = decoded[index - 1]
            following = decoded[index + 1] if index + 1 < len(decoded) else ""
            if previous.islower() or previous.isdigit() or (
                previous.isupper() and following.islower()
            ):
                normalized.append("_")
        normalized.append(character.lower())
    return "".join(normalized).replace("_", "") in _SENSITIVE_ERROR_NAMES


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


def _header_value(headers: list[str], name: str) -> str | None:
    for header in headers:
        separator = header.find(":")
        if separator > 0 and header[:separator].strip().lower() == name:
            return header[separator + 1 :].strip()
    return None


def _header_parameter(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    clean = _strip_mime_comments(value)
    if '"' not in clean:
        segments = clean.split(";")
    else:
        segments = []
        fragment: list[str] = []
        quoted = False
        escaped = False
        for character in clean:
            if escaped:
                fragment.append(character)
                escaped = False
            elif quoted and character == "\\":
                fragment.append(character)
                escaped = True
            elif character == '"':
                fragment.append(character)
                quoted = not quoted
            elif character == ";" and not quoted:
                segments.append("".join(fragment))
                fragment = []
            else:
                fragment.append(character)
        segments.append("".join(fragment))
    for segment in segments:
        key, separator, parameter = segment.partition("=")
        if separator and key.strip().lower() == name:
            parameter = parameter.strip()
            if len(parameter) >= 2 and parameter[0] == parameter[-1] == '"':
                return parameter[1:-1]
            return parameter
    return None


def _strip_mime_comments(value: str) -> str:
    if "(" not in value and ")" not in value:
        return value
    result: list[str] = []
    depth = 0
    quoted = False
    escaped = False
    for character in value:
        if escaped:
            if depth == 0:
                result.append(character)
            escaped = False
        elif quoted:
            result.append(character)
            if character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
            result.append(character)
        elif character == "(":
            depth += 1
        elif character == ")" and depth:
            depth -= 1
        elif depth == 0:
            result.append(character)
    return "".join(result)


def _multipart_name(headers: list[str]) -> str | None:
    disposition = _header_value(headers, "content-disposition")
    return _header_parameter(disposition, "name")


def _boundary_kind(line: str, boundary: str) -> str | None:
    content, _ = _line_parts(line)
    content = content.rstrip(" \t")
    if content == f"--{boundary}":
        return "open"
    if content == f"--{boundary}--":
        return "close"
    return None


def _read_headers(lines: list[str], start: int) -> tuple[list[str], int]:
    headers: list[str] = []
    fragments: list[str] = []
    index = start
    while index < len(lines):
        content, _ = _line_parts(lines[index])
        if not content:
            if fragments:
                headers.append(" ".join(fragments))
            return headers, index + 1
        if content[:1] in " \t" and fragments:
            fragments.append(content.strip())
        else:
            if fragments:
                headers.append(" ".join(fragments))
            fragments = [content]
        index += 1
    if fragments:
        headers.append(" ".join(fragments))
    return headers, index


@dataclass(slots=True)
class _MultipartFrame:
    boundary: str
    inherited_sensitive: bool
    awaiting_boundary: bool = False
    part_sensitive: bool = False


def _redact_multipart(text: str, *, depth_observer: list[int] | None = None) -> str:
    lines = text.splitlines(keepends=True)
    if not lines:
        return text

    headers, header_end = _read_headers(lines, 0)
    boundary = _header_parameter(_header_value(headers, "content-type"), "boundary")
    first_boundary = header_end
    if boundary is None:
        first_boundary = next(
            (
                index
                for index, line in enumerate(lines)
                if (
                    (content := _line_parts(line)[0]).startswith("--")
                    and not content.startswith("-- ")
                )
            ),
            len(lines),
        )
        if first_boundary == len(lines):
            return text
        first_line = _line_parts(lines[first_boundary])[0].rstrip(" \t")
        boundary = first_line[2:]
    if not boundary:
        return text
    while (
        first_boundary < len(lines)
        and _boundary_kind(lines[first_boundary], boundary) != "open"
    ):
        first_boundary += 1
    if first_boundary == len(lines):
        return text

    boundary_stack = [_MultipartFrame(boundary, False)]
    index = first_boundary
    while index < len(lines) and boundary_stack:
        line = lines[index]
        match_index: int | None = None
        match_kind: str | None = None
        if line.startswith("--"):
            for depth in range(len(boundary_stack) - 1, -1, -1):
                kind = _boundary_kind(line, boundary_stack[depth].boundary)
                if kind is not None:
                    match_index = depth
                    match_kind = kind
                    break
        if match_index is not None:
            if match_kind == "close":
                del boundary_stack[match_index:]
                index += 1
            else:
                del boundary_stack[match_index + 1 :]
                frame = boundary_stack[-1]
                frame.awaiting_boundary = False
                headers, index = _read_headers(lines, index + 1)
                frame.part_sensitive = frame.inherited_sensitive or _is_sensitive_name(
                    _multipart_name(headers) or ""
                )
                nested_boundary = _header_parameter(
                    _header_value(headers, "content-type"), "boundary"
                )
                if nested_boundary:
                    boundary_stack.append(
                        _MultipartFrame(nested_boundary, frame.part_sensitive, True)
                    )
                    if depth_observer is not None:
                        depth_observer.append(len(boundary_stack))
            continue

        frame = boundary_stack[-1]
        if frame.awaiting_boundary:
            if frame.inherited_sensitive:
                ending = line[len(line.rstrip("\r\n")) :]
                lines[index] = f"[redacted]{ending}"
        elif frame.part_sensitive:
            ending = line[len(line.rstrip("\r\n")) :]
            lines[index] = f"[redacted]{ending}"
        index += 1
    return "".join(lines)


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
    for line in io.StringIO(text):
        content, ending = _line_parts(line)
        if redact_continuation:
            if not content.strip():
                redact_continuation = False
                result.append(line)
                continue
            if _looks_like_header(content):
                redact_continuation = False
            else:
                result.append(f"[redacted]{ending}")
                break
        found = _sensitive_separator(content)
        if found is None:
            result.append(redact_bare_tokens(line))
        else:
            separator, kind = found
            result.append(f"{content[: separator + 1]}[redacted]{ending}")
            redact_continuation = kind == ":"
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
        text = body[: max(limit * 4, 4096)].decode("utf-8", errors="replace")
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


@dataclass(frozen=True, slots=True)
class PKCEParameters:
    """The verifier, challenge, and state for one OAuth authorization."""

    verifier: str
    challenge: str
    state: str


def build_pkce_parameters() -> PKCEParameters:
    """Create the PKCE values required by an OAuth authorization request."""

    verifier = secrets.token_urlsafe(32)
    challenge = hashlib.sha256(verifier.encode("ascii")).digest()
    return PKCEParameters(
        verifier=verifier,
        challenge=base64.urlsafe_b64encode(challenge).decode("ascii").rstrip("="),
        state=secrets.token_urlsafe(32),
    )


def _first_string(value: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        candidate = value.get(key)
        if type(candidate) is str and candidate:
            return candidate
    return None


def atomic_write_secret(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a 0600 JSON secret file via temp+rename, creating a 0700 parent.

    Shared by every zeta credential store (OAuth tokens, the ZETA-88
    persisted API key) so file permissions and atomicity stay identical
    across secret kinds instead of being reimplemented per store.
    """

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(dict(payload), handle, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


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
        atomic_write_secret(self.path, tokens.to_dict())

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
            return await self._refresh_unlocked(tokens, client)

    async def refresh_token(self, client: httpx.AsyncClient) -> str:
        async with self._async_refresh_lock, self._async_refresh_lock_file():
            tokens = self._read_unlocked()
            if tokens is None:
                tokens = self.bootstrap()
                if tokens is None:
                    raise self.auth_error_type(
                        f"no {self.provider_label} OAuth login found; log in first"
                    )
            return await self._refresh_unlocked(tokens, client)

    async def _refresh_unlocked(
        self, tokens: OAuthTokens, client: httpx.AsyncClient
    ) -> str:
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
