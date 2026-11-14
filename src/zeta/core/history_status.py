from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING

from .checkpoints import ConversationEntry

if TYPE_CHECKING:
    from .slash import CompactionSummary

StatusProjection = tuple[int, int, tuple["CompactionSummary", ...]]


def active_ids(rows: Sequence[object]) -> set[str]:
    by_id = {str(entry.id): entry for entry in rows}
    active: set[str] = set()
    current = rows[-1] if rows else None
    while current is not None:
        if current.id in active:
            break
        active.add(current.id)
        current = by_id.get(current.parent_id) if current.parent_id else None
    return active


def facts(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    return {
        key: getattr(value, key)
        for key in (
            "st_dev", "st_ino", "byte_boundary", "last_seq", "last_id",
            "last_parent", "prefix_digest", "reset_generation", "repair_generation",
        )
        if hasattr(value, key)
    }


def stored_facts(connection: sqlite3.Connection) -> dict[str, object] | None:
    row = connection.execute("SELECT value FROM meta WHERE key='facts'").fetchone()
    return facts(json.loads(row[0])) if row is not None else None


def warm_facts(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool, list, tuple, dict)):
        return value
    if is_dataclass(value):
        return {"kind": type(value).__name__, "value": asdict(value)}
    raise TypeError(f"unsupported warm row facts: {type(value).__name__}")


def _value(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _entry(row: tuple[object, ...]) -> ConversationEntry:
    return ConversationEntry(
        seq=int(row[0]),
        id=str(row[1]),
        parent_id=str(row[2]) if row[2] is not None else None,
        lane=str(row[3]),
        type=str(row[4]),
        data=json.loads(str(row[5])),
    )


def write(connection: sqlite3.Connection) -> StatusProjection:
    from ..context_accounting import message_token_count
    from .slash import compaction_history

    rows = connection.execute(
        "SELECT seq,id,parent_id,lane,type,data FROM entries "
        "WHERE tui_group IS NOT NULL ORDER BY seq"
    )
    history = compaction_history(tuple(_entry(row) for row in rows), message_token_count)
    _store(connection, history)
    marker_count, checkpoint_count = connection.execute(
        "SELECT SUM(type='compaction'),SUM(type='checkpoint') "
        "FROM entries WHERE tui_group IS NOT NULL"
    ).fetchone()
    return int(marker_count or 0), int(checkpoint_count or 0), history


def update(
    connection: sqlite3.Connection,
    current: StatusProjection | None,
    entries: Iterable[object],
) -> StatusProjection:
    from ..context_accounting import message_token_count
    from .slash import compaction_summary

    if current is None:
        return write(connection)
    marker_count, checkpoint_count, history = current
    summaries = list(history)
    for raw_entry in entries:
        entry = (
            raw_entry
            if isinstance(raw_entry, ConversationEntry)
            else ConversationEntry.from_dict(raw_entry)
        )
        marker_count += int(entry.type == "compaction")
        checkpoint_count += int(entry.type == "checkpoint")
        if entry.type != "compaction":
            continue
        replaces = tuple(entry.data.get("replaces", ()))
        replacement_clause = (
            f" OR id IN ({','.join('?' for _ in replaces)})" if replaces else ""
        )
        rows = connection.execute(
            "SELECT seq,id,parent_id,lane,type,data FROM entries "
            "WHERE tui_group IS NOT NULL AND "
            "((type='message' AND seq>=? AND seq<=?)"
            f"{replacement_clause}) ORDER BY seq",
            (
                int(entry.data["source_seq_start"]),
                int(entry.data["source_seq_end"]),
                *replaces,
            ),
        )
        source_entries = tuple(_entry(row) for row in rows)
        user_turn_row = connection.execute(
            "SELECT COUNT(*) FROM entries WHERE tui_group IS NOT NULL "
            "AND type='message' AND seq<=? "
            "AND json_extract(data,'$.message.role')='user'",
            (entry.seq,),
        ).fetchone()
        user_turn = int(user_turn_row[0]) if user_turn_row is not None else 0
        summaries.append(
            compaction_summary(
                entry,
                source_entries,
                message_token_count,
                max(1, user_turn),
            )
        )
    result = marker_count, checkpoint_count, tuple(summaries)
    _store(connection, result[2])
    return result


def load(connection: sqlite3.Connection) -> StatusProjection | None:
    from .slash import CompactionSummary

    row = connection.execute(
        "SELECT value FROM meta WHERE key='slash_status'"
    ).fetchone()
    if row is None:
        return None
    marker_count, checkpoint_count = connection.execute(
        "SELECT SUM(type='compaction'),SUM(type='checkpoint') "
        "FROM entries WHERE tui_group IS NOT NULL"
    ).fetchone()
    return (
        int(marker_count or 0),
        int(checkpoint_count or 0),
        tuple(CompactionSummary(**item) for item in json.loads(row[0])),
    )


def _store(
    connection: sqlite3.Connection, history: tuple[CompactionSummary, ...]
) -> None:
    connection.execute(
        "INSERT INTO meta(key,value) VALUES('slash_status',?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (_value([asdict(item) for item in history]),),
    )
