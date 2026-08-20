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

import httpx


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
