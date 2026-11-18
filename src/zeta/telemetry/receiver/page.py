"""SQLite keyset queries over the shared serve pager."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from typing import Any

from ...read_page import Key, Resume, page_result, page_room


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
    room = page_room(start, limit, total)
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

    return page_result(
        project(rows), lambda index: tuple(rows[index][key.column] for key in order),
        start, more=more, total=total, fits=fits, shape=shape,
    )


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
