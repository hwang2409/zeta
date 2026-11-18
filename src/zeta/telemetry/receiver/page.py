"""Keyset pages for telemetry reads: the one owner of cursors, page order, count caps, and the frame cut.

A list read names its rows with a SQL query and its order with ``Key`` columns. ``resume`` checks a
client's cursor; ``keyset_page`` runs the query after the cursor's position in that exact order, keeps
at most ``limit`` rows (and at most ``total`` over all pages), and returns the largest prefix whose
result frame fits. Its ``next_cursor`` points after the last item it returned whenever more rows exist.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

_VERSION = 1
_MAX_CURSOR_CHARS = 4096
_HEX = re.compile(r"(?:[0-9a-f]{2})+\Z")


class InvalidParams(ValueError):
    """Malformed params or a cursor that does not belong to these params."""


class TooLarge(Exception):
    """Not even one item fits the caller's size budget."""


@dataclass(frozen=True)
class Key:
    """One order column of the page query: never null; ``kind`` is ``int``, ``str``, or ``bytes``."""

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
    if not isinstance(cursor, str) or not 0 < len(cursor) <= _MAX_CURSOR_CHARS or "=" in cursor:
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


def keyset_page(
    connection: sqlite3.Connection,
    query: str,
    args: Sequence[object],
    order: Sequence[Key],
    start: Resume,
    *,
    limit: int,
    fits: Callable[[dict[str, Any]], bool],
    shape: Callable[[list[Any], str | None], dict[str, Any]],
    project: Callable[[list[sqlite3.Row]], list[Any]] = lambda rows: [dict(row) for row in rows],
    total: int | None = None,
) -> dict[str, Any]:
    """One page of ``query``'s rows after ``start``, in ``order``, as ``shape(items, next_cursor)``.

    ``project`` turns the page's rows into items once. Raises ``TooLarge`` when one item does not fit.
    """
    if total is not None and start.emitted >= total:
        raise InvalidParams("invalid cursor")
    room = limit if total is None else min(limit, total - start.emitted)
    sql = f"SELECT * FROM ({query})"
    values = list(args)
    if start.position is not None:
        predicate, bound = _after(order, start.position)
        sql += f" WHERE {predicate}"
        values += bound
    sql += " ORDER BY " + ", ".join(f"{key.column} {'DESC' if key.descending else 'ASC'}" for key in order) + " LIMIT ?"
    rows = connection.execute(sql, [*values, room + 1]).fetchall()
    more = len(rows) > room
    rows = rows[:room]
    items = project(rows)

    def result(count: int) -> dict[str, Any]:
        cursor = None
        if (more or count < len(rows)) and (total is None or start.emitted + count < total):
            cursor = _encode(start, rows[count - 1], order, start.emitted + count)
        return shape(items[:count], cursor)

    if fits(full := result(len(rows))):
        return full
    low, high = 0, len(rows)  # result(low) fits once low >= 1; result(high) does not
    while high - low > 1:
        middle = (low + high) // 2
        if fits(result(middle)):
            low = middle
        else:
            high = middle
    if low == 0:
        raise TooLarge("one item does not fit the frame")
    return result(low)


def _after(order: Sequence[Key], position: Sequence[object]) -> tuple[str, list[object]]:
    """Rows strictly after ``position`` in ``order``; the leading bound lets SQLite seek an index."""
    first = order[0]
    terms, values = [f"{first.column} {'<=' if first.descending else '>='} ?"], [position[0]]
    alternatives = []
    for index, key in enumerate(order):
        equal = [f"{earlier.column} = ?" for earlier in order[:index]]
        alternatives.append("(" + " AND ".join([*equal, f"{key.column} {'<' if key.descending else '>'} ?"]) + ")")
        values += [*position[:index], position[index]]
    terms.append("(" + " OR ".join(alternatives) + ")")
    return " AND ".join(terms), values


def _encode(start: Resume, row: sqlite3.Row, order: Sequence[Key], emitted: int) -> str:
    position = [row[key.column].hex() if key.kind is bytes else row[key.column] for key in order]
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
