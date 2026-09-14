"""SQLite automation metadata; conversation history belongs to normal sessions."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Protocol, Self
from zoneinfo import ZoneInfoNotFoundError

from ..core.session import env_home
from .models import (
    DueOccurrence,
    Job,
    JobState,
    PollEvent,
    RunRecord,
    instant,
    parse_job,
    timestamp,
)


class AutomationStore(Protocol):
    def jobs(self) -> tuple[JobState, ...]: ...
    def get(self, name: str) -> JobState: ...
    def claim(self, occurrence: DueOccurrence) -> str | None: ...
    def consume(
        self, run_id: str, occurrence: DueOccurrence, events: tuple[PollEvent, ...]
    ) -> tuple[PollEvent, ...]: ...
    def attach_session(self, run_id: str, session_id: str) -> None: ...
    def finish(
        self, run_id: str, status: str, detail: str = "", delivery: str | None = None
    ) -> None: ...


class SQLiteStore:
    def __init__(self, home: Path | None = None) -> None:
        directory = (home or env_home()) / "automations"
        directory.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(directory / "automations.sqlite3", timeout=5)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS revisions (
                name TEXT NOT NULL, revision INTEGER NOT NULL, document TEXT NOT NULL,
                source TEXT NOT NULL, PRIMARY KEY(name, revision));
            CREATE TABLE IF NOT EXISTS jobs (
                name TEXT PRIMARY KEY, revision INTEGER NOT NULL, enabled INTEGER NOT NULL DEFAULT 0,
                approved_at TEXT, recipient TEXT, last_run TEXT, last_check TEXT, last_due TEXT);
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, revision INTEGER NOT NULL,
                due_at TEXT NOT NULL, status TEXT NOT NULL, session_id TEXT,
                detail TEXT NOT NULL DEFAULT '', delivery TEXT NOT NULL DEFAULT '',
                UNIQUE(name, revision, due_at));
            CREATE TABLE IF NOT EXISTS events (
                name TEXT NOT NULL, event_id TEXT NOT NULL, run_id TEXT NOT NULL,
                PRIMARY KEY(name, event_id));
            CREATE TABLE IF NOT EXISTS errors (name TEXT PRIMARY KEY, detail TEXT NOT NULL);
        """)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self.db.close()

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            yield

    def draft(self, job: Job, *, source: str = "tool") -> JobState:
        document = json.dumps(job.document(), sort_keys=True)
        with self._transaction():
            row = self.db.execute(
                "SELECT revision FROM jobs WHERE name=?", (job.name,)
            ).fetchone()
            revision = row["revision"] + 1 if row else 1
            self.db.execute(
                "INSERT INTO revisions VALUES (?, ?, ?, ?)",
                (job.name, revision, document, source),
            )
            self.db.execute(
                """INSERT INTO jobs(name, revision) VALUES (?, ?)
                ON CONFLICT(name) DO UPDATE SET revision=excluded.revision, enabled=0,
                approved_at=NULL, recipient=NULL, last_run=NULL, last_check=NULL, last_due=NULL""",
                (job.name, revision),
            )
            self.db.execute("DELETE FROM errors WHERE name=?", (job.name,))
        return self.get(job.name)

    def _state(self, row: sqlite3.Row) -> JobState:
        def date(field: str) -> datetime | None:
            return instant(row[field]) if row[field] else None

        return JobState(
            parse_job(row["name"], json.loads(row["document"])),
            row["revision"],
            date("approved_at"),
            row["recipient"],
            date("last_run"),
            date("last_check"),
            date("last_due"),
            bool(row["enabled"]),
        )

    def jobs(self) -> tuple[JobState, ...]:
        rows = self.db.execute("""SELECT j.*, r.document FROM jobs j JOIN revisions r
            ON j.name=r.name AND j.revision=r.revision ORDER BY j.name""").fetchall()
        result = []
        for row in rows:
            try:
                result.append(self._state(row))
            except (ValueError, TypeError, KeyError, ZoneInfoNotFoundError):
                # Read-only: tick must never mutate even when an entry is damaged.
                continue
        return tuple(result)

    def get(self, name: str) -> JobState:
        row = self.db.execute(
            """SELECT j.*, r.document FROM jobs j JOIN revisions r
            ON j.name=r.name AND j.revision=r.revision WHERE j.name=?""",
            (name,),
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown automation: {name}")
        return self._state(row)

    def approve(self, name: str, revision: int, recipient: str, now: datetime) -> None:
        if not recipient or recipient[0] not in "CUGD" or not recipient.isalnum():
            raise ValueError(
                "approval requires a resolved Slack user or conversation ID"
            )
        with self._transaction():
            state = self.get(name)
            if state.revision != revision:
                raise ValueError("draft changed; inspect and approve the new revision")
            if state.enabled:
                raise ValueError("automation is already armed")
            self.db.execute(
                """UPDATE jobs SET enabled=1, approved_at=?, recipient=?,
                last_run=?, last_check=?, last_due=NULL WHERE name=? AND revision=?""",
                (
                    timestamp(now),
                    recipient,
                    timestamp(now),
                    timestamp(now),
                    name,
                    revision,
                ),
            )

    def disable(self, name: str) -> None:
        with self._transaction():
            self.get(name)
            self.db.execute("UPDATE jobs SET enabled=0 WHERE name=?", (name,))

    def claim(self, occurrence: DueOccurrence) -> str | None:
        from .tick import tick

        with self._transaction():
            if occurrence not in tick(self, occurrence.checked_at):
                return None
            running = self.db.execute(
                "SELECT 1 FROM runs WHERE name=? AND status IN ('claimed','running','sending')",
                (occurrence.name,),
            ).fetchone()
            if running:
                return None
            run_id = uuid.uuid4().hex
            row = self.db.execute(
                """INSERT OR IGNORE INTO runs(id,name,revision,due_at,status)
                VALUES (?,?,?,?, 'claimed')""",
                (
                    run_id,
                    occurrence.name,
                    occurrence.revision,
                    timestamp(occurrence.due_at),
                ),
            )
            if not row.rowcount:
                return None
            self.db.execute(
                "UPDATE jobs SET last_due=?, last_check=? WHERE name=?",
                (
                    timestamp(occurrence.due_at),
                    timestamp(occurrence.checked_at),
                    occurrence.name,
                ),
            )
            return run_id

    def attach_session(self, run_id: str, session_id: str) -> None:
        with self._transaction():
            self.db.execute(
                "UPDATE runs SET session_id=?, status='running' WHERE id=?",
                (session_id, run_id),
            )

    def consume(
        self, run_id: str, occurrence: DueOccurrence, events: tuple[PollEvent, ...]
    ) -> tuple[PollEvent, ...]:
        with self._transaction():
            claimed = []
            for event in events:
                inserted = self.db.execute(
                    "INSERT OR IGNORE INTO events VALUES (?,?,?)",
                    (occurrence.name, event.id, run_id),
                ).rowcount
                if inserted:
                    claimed.append(event)
            self.db.execute(
                "UPDATE jobs SET last_run=? WHERE name=? AND revision=?",
                (
                    timestamp(occurrence.checked_at),
                    occurrence.name,
                    occurrence.revision,
                ),
            )
            return tuple(claimed)

    def finish(
        self, run_id: str, status: str, detail: str = "", delivery: str | None = None
    ) -> None:
        with self._transaction():
            self.db.execute(
                "UPDATE runs SET status=?, detail=?, delivery=COALESCE(?, delivery) WHERE id=?",
                (status, detail, delivery, run_id),
            )

    def fail_unfinished(self, run_id: str, detail: str) -> None:
        """Retain uncertainty if an unexpected worker exception follows a send."""
        with self._transaction():
            self.db.execute(
                """UPDATE runs SET status=CASE WHEN status='sending'
                THEN 'uncertain' ELSE 'failed' END, detail=?
                WHERE id=? AND status IN ('claimed','running','sending')""",
                (detail, run_id),
            )

    def recover(self) -> None:
        with self._transaction():
            self.db.execute(
                "UPDATE runs SET status='uncertain', detail='daemon stopped during delivery; not replayed' WHERE status='sending'"
            )
            self.db.execute(
                "UPDATE runs SET status='interrupted', detail='daemon stopped; not replayed' WHERE status IN ('claimed','running')"
            )

    def runs(self, name: str) -> tuple[RunRecord, ...]:
        rows = self.db.execute(
            "SELECT * FROM runs WHERE name=? ORDER BY rowid DESC LIMIT 20", (name,)
        )
        return tuple(RunRecord(**dict(row)) for row in rows)

    def record_error(self, name: str, detail: str) -> None:
        with self._transaction():
            self.db.execute(
                "INSERT OR REPLACE INTO errors VALUES (?, ?)", (name, detail)
            )

    def errors(self) -> tuple[str, ...]:
        errors = [
            f"{row['name']}: {row['detail']}"
            for row in self.db.execute("SELECT * FROM errors ORDER BY name")
        ]
        names = {state.job.name for state in self.jobs()}
        errors.extend(
            f"{row['name']}: malformed stored job (disabled for scheduling)"
            for row in self.db.execute("SELECT name FROM jobs")
            if row["name"] not in names
        )
        return tuple(errors)
