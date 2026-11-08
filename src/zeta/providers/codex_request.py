"""Bounded, replayable request bodies for large Codex image payloads."""

from __future__ import annotations

import json
import secrets
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from .codex_payload import _CodexImageURL

_THRESHOLD = 24 * 1024 * 1024
_CHUNK = 64 * 1024


class _Body(httpx.AsyncByteStream):
    def __init__(self, parts: tuple[bytes | str, ...]):
        self._parts = parts

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for part in self._parts:
            if isinstance(part, bytes):
                if part:
                    yield part
                continue
            for start in range(0, len(part), _CHUNK):
                yield part[start : start + _CHUNK].encode("ascii")

    async def aclose(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class PreparedCodexRequest:
    headers: dict[str, str]
    content: httpx.AsyncByteStream


def _marked(value: object, active: set[int], images: list[str]) -> bool:
    if type(value) is dict:
        identity = id(value)
        if identity in active:
            return False
        active.add(identity)
        try:
            return all(
                _marked(key, active, images) and _marked(item, active, images)
                for key, item in value.items()
            )
        finally:
            active.remove(identity)
    if type(value) in (list, tuple):
        identity = id(value)
        if identity in active:
            return False
        active.add(identity)
        try:
            return all(_marked(item, active, images) for item in value)
        finally:
            active.remove(identity)
    if type(value) is _CodexImageURL:
        images.append(value)
    return True


def _clone(value: object, replacements: dict[int, list[str]]) -> object:
    if type(value) is dict:
        return {_clone(key, replacements): _clone(item, replacements) for key, item in value.items()}
    if type(value) is list:
        return [_clone(item, replacements) for item in value]
    if type(value) is tuple:
        return tuple(_clone(item, replacements) for item in value)
    if type(value) is _CodexImageURL:
        return replacements[id(value)].pop(0)
    return value


def _attempt(payload: dict[str, Any], images: list[str]) -> PreparedCodexRequest | None:
    replacements: dict[int, list[str]] = {}
    tokens_by_image: list[str] = []
    for image in images:
        token = "__zeta_codex_image_" + secrets.token_urlsafe(18) + "__"
        replacements.setdefault(id(image), []).append(token)
        tokens_by_image.append(token)
    encoded = json.dumps(
        _clone(payload, {key: values.copy() for key, values in replacements.items()}),
        ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    tokens = [json.dumps(token, ensure_ascii=False, separators=(",", ":")).encode("utf-8") for token in tokens_by_image]
    offsets: list[tuple[int, int, str]] = []
    cursor = 0
    for token, image in zip(tokens, images, strict=True):
        position = encoded.find(token, cursor)
        if position < 0 or encoded.find(token, position + 1) >= 0:
            return None
        offsets.append((position, position + len(token), image))
        cursor = position + len(token)
    parts: list[bytes | str] = []
    cursor = 0
    for start, end, image in offsets:
        parts.extend((encoded[cursor:start], b'"', image, b'"'))
        cursor = end
    parts.append(encoded[cursor:])
    length = sum(map(len, parts))
    return PreparedCodexRequest(
        {"Content-Type": "application/json", "Content-Length": str(length)},
        _Body(tuple(parts)),
    )


def prepare_large_image_request(payload: dict[str, Any]) -> PreparedCodexRequest | None:
    images: list[str] = []
    if not _marked(payload, set(), images):
        return None
    if not images or sum(len(image) for image in images) < _THRESHOLD:
        return None
    for _ in range(3):
        prepared = _attempt(payload, images)
        if prepared is not None:
            return prepared
    return None
