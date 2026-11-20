"""Shared keyset cursors and frame-bounded result pages for serve reads."""

from __future__ import annotations

import base64
import binascii
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

_VERSION = 1
_MAX_CURSOR_CHARS = 4096
_BASE64URL = re.compile(r"[A-Za-z0-9_-]+\Z")
_HEX = re.compile(r"(?:[0-9a-f]{2})+\Z")


class InvalidParams(ValueError):
    """Malformed params or a cursor that does not belong to these params."""


class TooLarge(Exception):
    """Not even one item fits the caller's size budget."""


@dataclass(frozen=True)
class Key:
    """One order field of the page: never null; ``kind`` is ``int``, ``str``, or ``bytes``."""

    column: str
    kind: type
    descending: bool = False


@dataclass(frozen=True)
class Resume:
    """Where a page starts: the scope the cursor is bound to, its pinned ints, and the last returned key."""

    scope: tuple[object, ...]
    pinned: tuple[int, ...]
    position: tuple[object, ...] | None
    emitted: int


def resume(cursor: object, scope: Sequence[object], pinned: Sequence[int], order: Sequence[Key]) -> Resume:
    """The first page (``cursor`` None) pins ``pinned``; a later page takes them from its cursor.

    ``scope`` is the method and every filter, selector, and sort param as the client sent them; a cursor
    made for another scope, or a malformed one, raises ``InvalidParams``.
    """
    if cursor is None:
        return Resume(tuple(scope), tuple(pinned), None, 0)
    if not isinstance(cursor, str) or not 0 < len(cursor) <= _MAX_CURSOR_CHARS or not _BASE64URL.fullmatch(cursor):
        raise InvalidParams("invalid cursor")
    try:
        value = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode())
    except (binascii.Error, UnicodeDecodeError, ValueError):
        raise InvalidParams("invalid cursor") from None
    if (
        not isinstance(value, list) or len(value) != 5 or value[0] != _VERSION or value[1] != list(scope)
        or not isinstance(value[2], list) or len(value[2]) != len(pinned) or not all(_count(item) for item in value[2])
        or not isinstance(value[3], list) or len(value[3]) != len(order) or not _count(value[4])
    ):
        raise InvalidParams("invalid cursor")
    position = tuple(_decode_key(item, key) for item, key in zip(value[3], order, strict=True))
    return Resume(tuple(scope), tuple(value[2]), position, value[4])


def page_room(start: Resume, limit: int, total: int | None) -> int:
    if total is None:
        return limit
    if start.emitted >= total:
        raise InvalidParams("invalid cursor")
    return min(limit, total - start.emitted)


def page_result(
    items: list[Any],
    position: Callable[[int], Sequence[object]],
    start: Resume,
    *,
    more: bool,
    total: int | None,
    fits: Callable[[dict[str, Any]], bool],
    shape: Callable[[list[Any], str | None], dict[str, Any]],
    scan_end: Sequence[object] | None = None,
) -> dict[str, Any]:
    """Return the largest fitting prefix; a full page may resume after a bounded scan."""
    def result(count: int) -> dict[str, Any]:
        cursor = None
        if (more or count < len(items)) and (total is None or start.emitted + count < total):
            key = scan_end if scan_end is not None and count == len(items) else position(count - 1)
            cursor = _encode(start, key, start.emitted + count)
        return shape(items[:count], cursor)

    return _fit_prefix(items, result, fits, shape)


def offset_page(
    items: list[Any], offset: int, limit: int, *,
    project: Callable[[Any], Any],
    fits: Callable[[dict[str, Any]], bool],
    shape: Callable[[list[Any], int | None], dict[str, Any]],
) -> dict[str, Any]:
    """Fit an offset page of an already ordered collection to the response frame."""
    page = [project(item) for item in items[offset:offset + limit]]

    def result(count: int) -> dict[str, Any]:
        end = offset + count
        return shape(page[:count], end if end < len(items) else None)

    return _fit_prefix(page, result, fits, shape)


def _fit_prefix(items, result, fits, shape):
    size = len(items)
    if fits(full := result(size)):
        return full

    # The item-only frame grows with count, but cursors can shrink as the key
    # changes. Find the upper bound without cursors, then check real frames.
    if size and fits(shape(items, None)):
        high = size
    else:
        low, high = 0, size
        while high - low > 1:
            middle = (low + high) // 2
            if fits(shape(items[:middle], None)):
                low = middle
            else:
                high = middle
        high = low
    for count in range(min(high, size - 1), 0, -1):
        if fits(candidate := result(count)):
            return candidate
    raise TooLarge("one item does not fit the frame")


def descending_page(
    rows: list[dict[str, Any]], order: Sequence[Key], start: Resume, limit: int,
    *, fits: Callable[[dict[str, Any]], bool], shape: Callable[[list[Any], str | None], dict[str, Any]],
    total: int | None = None,
) -> dict[str, Any]:
    """Page in descending key order, without copying cursor or frame rules."""
    def position(row: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(row[key.column] for key in order)

    after = sorted(
        (row for row in rows if start.position is None or position(row) < start.position),
        key=position, reverse=True,
    )
    room = page_room(start, limit, total)
    return page_result(
        after[:room], lambda index: position(after[index]), start,
        more=len(after) > room, total=total, fits=fits, shape=shape,
    )


def _encode(start: Resume, keys: Sequence[object], emitted: int) -> str:
    position = [key.hex() if isinstance(key, bytes) else key for key in keys]
    text = json.dumps([_VERSION, list(start.scope), list(start.pinned), position, emitted], separators=(",", ":"))
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


def _decode_key(item: object, key: Key) -> object:
    if key.kind is int and _count(item):
        return item
    if key.kind is str and isinstance(item, str):
        return item
    if key.kind is bytes and isinstance(item, str) and _HEX.match(item):
        return bytes.fromhex(item)
    raise InvalidParams("invalid cursor")


def _count(value: object) -> bool:
    return type(value) is int and value >= 0
