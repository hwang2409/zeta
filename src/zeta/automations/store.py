"""SQLite automation metadata; conversation history belongs to normal sessions."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
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
from .trigger import Webhook


@dataclass(frozen=True)
class WebhookCredentials:
    secret: bytes
    token: str


@dataclass(frozen=True)
class WebhookDelivery:
    id: str
    name: str
    revision: int
    body: bytes
    headers: dict[str, str]
    accepted_at: datetime


@dataclass(frozen=True)
class ClaimedWebhook:
    delivery: WebhookDelivery
    occurrence: DueOccurrence
    run_id: str


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


class PendingWebhookLimitError(RuntimeError):
    """The durable pending-delivery budget for a job is exhausted."""


class SQLiteStore:
    def __init__(
        self,
        home: Path | None = None,
        *,
        max_pending_count: int = 100,
        max_pending_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        if max_pending_count <= 0 or max_pending_bytes <= 0:
            raise ValueError("pending webhook limits must be positive")
        self.max_pending_count = max_pending_count
        self.max_pending_bytes = max_pending_bytes
        directory = (home or env_home()) / "automations"
        directory.mkdir(parents=True, exist_ok=True)
        os.chmod(directory, 0o700)
        self.path = directory / "automations.sqlite3"
        self.db = sqlite3.connect(self.path, timeout=5, check_same_thread=False)
        os.chmod(self.path, 0o600)
        self.db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
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
                delivery_id TEXT UNIQUE);
            CREATE TABLE IF NOT EXISTS events (
                name TEXT NOT NULL, event_id TEXT NOT NULL, run_id TEXT NOT NULL,
                PRIMARY KEY(name, event_id));
            CREATE TABLE IF NOT EXISTS errors (name TEXT PRIMARY KEY, detail TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS webhook_credentials (
                name TEXT PRIMARY KEY, secret BLOB NOT NULL, token TEXT NOT NULL UNIQUE);
            CREATE TABLE IF NOT EXISTS webhook_deliveries (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, revision INTEGER NOT NULL,
                event_key TEXT NOT NULL, body_hash TEXT NOT NULL, body BLOB NOT NULL,
                headers TEXT NOT NULL, accepted_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending', run_id TEXT,
                UNIQUE(name, event_key));
        """)
        self._migrate_runs_schema()
        self.db.executescript("""
            CREATE UNIQUE INDEX IF NOT EXISTS runs_scheduled_occurrence
            ON runs(name, revision, due_at) WHERE delivery_id IS NULL;
            CREATE INDEX IF NOT EXISTS webhook_deliveries_pending
            ON webhook_deliveries(name, status);
            CREATE INDEX IF NOT EXISTS webhook_deliveries_body_dedupe
            ON webhook_deliveries(name, body_hash, accepted_at);
        """)

    def _migrate_runs_schema(self) -> None:
        columns = {row["name"] for row in self.db.execute("PRAGMA table_info(runs)")}
        if "delivery_id" in columns:
            return
        with self.db:
            self.db.execute("ALTER TABLE runs RENAME TO runs_legacy")
            self.db.execute("""CREATE TABLE runs (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, revision INTEGER NOT NULL,
                due_at TEXT NOT NULL, status TEXT NOT NULL, session_id TEXT,
                detail TEXT NOT NULL DEFAULT '', delivery TEXT NOT NULL DEFAULT '',
                delivery_id TEXT UNIQUE)""")
            self.db.execute("""INSERT INTO runs
                (id,name,revision,due_at,status,session_id,detail,delivery)
                SELECT id,name,revision,due_at,status,session_id,detail,delivery
                FROM runs_legacy""")
            self.db.execute("DROP TABLE runs_legacy")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self.db.close()

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        with self._lock, self.db:
            self.db.execute("BEGIN IMMEDIATE")
            yield

    def _fsync(self) -> None:
        for suffix in ("", "-wal"):
            try:
                descriptor = os.open(str(self.path) + suffix, os.O_RDONLY)
            except FileNotFoundError:
                continue
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

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
        with self._lock:
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
        with self._lock:
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
            if isinstance(state.job.trigger, Webhook):
                self.db.execute(
                    "INSERT OR IGNORE INTO webhook_credentials VALUES (?, ?, ?)",
                    (name, secrets.token_bytes(32), secrets.token_urlsafe(24)),
                )

    def disable(self, name: str) -> None:
        with self._transaction():
            self.get(name)
            self.db.execute("UPDATE jobs SET enabled=0 WHERE name=?", (name,))

    def claim(self, occurrence: DueOccurrence) -> str | None:
        from .tick import tick

        with self._transaction():
            state = self.get(occurrence.name)
            if isinstance(state.job.trigger, Webhook) or occurrence not in tick(
                self, occurrence.checked_at
            ):
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

    def webhook_credentials(self, name: str) -> WebhookCredentials:
        with self._lock:
            row = self.db.execute(
                "SELECT secret, token FROM webhook_credentials WHERE name=?", (name,)
            ).fetchone()
        if row is None:
            raise ValueError(f"webhook credentials do not exist for: {name}")
        return WebhookCredentials(bytes(row["secret"]), row["token"])

    def rotate_webhook_secret(self, name: str) -> bytes:
        value = secrets.token_bytes(32)
        with self._transaction():
            self.webhook_credentials(name)
            self.db.execute(
                "UPDATE webhook_credentials SET secret=? WHERE name=?", (value, name)
            )
        self._fsync()
        return value

    def rotate_webhook_url(self, name: str) -> str:
        value = secrets.token_urlsafe(24)
        with self._transaction():
            self.webhook_credentials(name)
            self.db.execute(
                "UPDATE webhook_credentials SET token=? WHERE name=?", (value, name)
            )
        self._fsync()
        return value

    def resolve_webhook_token(
        self, token: str
    ) -> tuple[JobState, WebhookCredentials] | None:
        with self._lock:
            row = self.db.execute(
                """SELECT name FROM webhook_credentials WHERE token=?""", (token,)
            ).fetchone()
            if row is None:
                return None
            try:
                state = self.get(row["name"])
            except ValueError:
                return None
            if not state.enabled or not isinstance(state.job.trigger, Webhook):
                return None
            return state, self.webhook_credentials(state.job.name)

    def accept_webhook(
        self,
        name: str,
        revision: int,
        body: bytes,
        headers: dict[str, str],
        accepted_at: datetime,
        *,
        delivery_id: str | None = None,
        dedupe_window_seconds: float = 300,
    ) -> bool:
        digest = hashlib.sha256(body).hexdigest()
        if delivery_id is not None and (
            not delivery_id or len(delivery_id.encode("utf-8")) > 256
        ):
            raise ValueError("delivery id must be a bounded nonempty string")
        with self._transaction():
            state = self.get(name)
            if (
                not state.enabled
                or state.revision != revision
                or not isinstance(state.job.trigger, Webhook)
            ):
                return False
            if delivery_id is not None:
                event_key = "id:" + delivery_id
                if self.db.execute(
                    "SELECT 1 FROM webhook_deliveries WHERE name=? AND event_key=?",
                    (name, event_key),
                ).fetchone():
                    return False
            else:
                previous = self.db.execute(
                    """SELECT accepted_at FROM webhook_deliveries
                    WHERE name=? AND body_hash=? ORDER BY accepted_at DESC LIMIT 1""",
                    (name, digest),
                ).fetchone()
                if previous is not None:
                    age = (accepted_at - instant(previous["accepted_at"])).total_seconds()
                    if age <= dedupe_window_seconds:
                        return False
                event_key = f"body:{digest}:{uuid.uuid4().hex}"
            pending = self.db.execute(
                """SELECT COUNT(*) AS count, COALESCE(SUM(length(body)), 0) AS bytes
                FROM webhook_deliveries WHERE name=? AND status='pending'""",
                (name,),
            ).fetchone()
            if (
                pending["count"] >= self.max_pending_count
                or pending["bytes"] + len(body) > self.max_pending_bytes
            ):
                raise PendingWebhookLimitError(
                    f"pending webhook limit reached for: {name}"
                )
            self.db.execute(
                """INSERT INTO webhook_deliveries
                (id,name,revision,event_key,body_hash,body,headers,accepted_at)
                VALUES (?,?,?,?,?,?,?,?)""",
                (
                    uuid.uuid4().hex,
                    name,
                    revision,
                    event_key,
                    digest,
                    body,
                    json.dumps(headers, sort_keys=True),
                    timestamp(accepted_at),
                ),
            )
        self._fsync()
        return True

    def _delivery(self, row: sqlite3.Row) -> WebhookDelivery:
        return WebhookDelivery(
            row["id"],
            row["name"],
            row["revision"],
            bytes(row["body"]),
            json.loads(row["headers"]),
            instant(row["accepted_at"]),
        )

    def pending_webhooks(self) -> tuple[WebhookDelivery, ...]:
        with self._lock:
            rows = self.db.execute(
                "SELECT * FROM webhook_deliveries WHERE status='pending' ORDER BY rowid"
            ).fetchall()
        return tuple(self._delivery(row) for row in rows)

    def claim_webhook(self, checked_at: datetime) -> ClaimedWebhook | None:
        """Atomically bind the oldest runnable delivery to a newly-created run."""
        with self._transaction():
            while True:
                row = self.db.execute(
                    "SELECT * FROM webhook_deliveries WHERE status='pending' ORDER BY rowid LIMIT 1"
                ).fetchone()
                if row is None:
                    return None
                delivery = self._delivery(row)
                state = self.get(delivery.name)
                running = self.db.execute(
                    """SELECT 1 FROM runs WHERE name=?
                    AND status IN ('claimed','running','sending')""",
                    (delivery.name,),
                ).fetchone()
                if running:
                    return None
                if (
                    not state.enabled
                    or state.revision != delivery.revision
                    or not isinstance(state.job.trigger, Webhook)
                ):
                    run_id = uuid.uuid4().hex
                    self.db.execute(
                        """INSERT INTO runs
                        (id,name,revision,due_at,status,detail,delivery_id)
                        VALUES (?,?,?,?, 'skipped',
                        'approved revision changed before execution', ?)""",
                        (
                            run_id,
                            delivery.name,
                            delivery.revision,
                            timestamp(delivery.accepted_at),
                            delivery.id,
                        ),
                    )
                    self.db.execute(
                        """UPDATE webhook_deliveries
                        SET status='done', run_id=?, body=X'', headers='{}' WHERE id=?""",
                        (run_id, delivery.id),
                    )
                    continue
                occurrence = DueOccurrence(
                    delivery.name,
                    delivery.revision,
                    delivery.accepted_at,
                    checked_at,
                    state.last_run or delivery.accepted_at,
                )
                run_id = uuid.uuid4().hex
                self.db.execute(
                    """INSERT INTO runs
                    (id,name,revision,due_at,status,delivery_id)
                    VALUES (?,?,?,?, 'claimed', ?)""",
                    (
                        run_id,
                        delivery.name,
                        delivery.revision,
                        timestamp(delivery.accepted_at),
                        delivery.id,
                    ),
                )
                self.db.execute(
                    """UPDATE webhook_deliveries SET status='claimed', run_id=?
                    WHERE id=? AND status='pending'""",
                    (run_id, delivery.id),
                )
                return ClaimedWebhook(delivery, occurrence, run_id)

    def interrupt_oldest_webhook(self, detail: str) -> None:
        """Quarantine one delivery after an unexpected claim/store failure."""
        with self._transaction():
            row = self.db.execute(
                """SELECT id,name,revision,accepted_at FROM webhook_deliveries
                WHERE status='pending' ORDER BY rowid LIMIT 1"""
            ).fetchone()
            if row is None:
                return
            run_id = uuid.uuid4().hex
            self.db.execute(
                """INSERT INTO runs
                (id,name,revision,due_at,status,detail,delivery_id)
                VALUES (?,?,?,?, 'interrupted', ?, ?)""",
                (
                    run_id,
                    row["name"],
                    row["revision"],
                    row["accepted_at"],
                    detail,
                    row["id"],
                ),
            )
            self.db.execute(
                """UPDATE webhook_deliveries SET status='interrupted', run_id=?,
                body=X'', headers='{}' WHERE id=?""",
                (run_id, row["id"]),
            )

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
            if status in {
                "completed",
                "failed",
                "canceled",
                "no_match",
                "skipped",
                "uncertain",
                "interrupted",
            }:
                delivery_status = "interrupted" if status == "interrupted" else "done"
                self.db.execute(
                    """UPDATE webhook_deliveries
                    SET status=?, body=X'', headers='{}' WHERE run_id=?""",
                    (delivery_status, run_id),
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
            self.db.execute(
                """UPDATE webhook_deliveries
                SET status='done', body=X'', headers='{}' WHERE run_id=?""",
                (run_id,),
            )

    def recover(self) -> None:
        with self._transaction():
            self.db.execute(
                "UPDATE runs SET status='uncertain', detail='daemon stopped during delivery; not replayed' WHERE status='sending'"
            )
            self.db.execute(
                "UPDATE runs SET status='interrupted', detail='daemon stopped; not replayed' WHERE status IN ('claimed','running')"
            )
            self.db.execute(
                """UPDATE webhook_deliveries
                SET status='interrupted', body=X'', headers='{}'
                WHERE status='claimed'"""
            )

    def runs(self, name: str) -> tuple[RunRecord, ...]:
        rows = self.db.execute(
            """SELECT id,name,revision,due_at,status,session_id,detail,delivery
            FROM runs WHERE name=? ORDER BY rowid DESC LIMIT 20""",
            (name,),
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
