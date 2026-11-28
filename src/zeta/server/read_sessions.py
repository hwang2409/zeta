"""Metadata-only state reads of session owners, without attaching a session."""

from __future__ import annotations

import os
import stat
import time
from datetime import UTC, datetime

from .. import attention_records
from ..attention_decisions import session_attention_counts
from ..core.child_runs import (
    child_run_directories,
    read_child_run_metadata,
    run_row_identity,
)
from ..core.session import (
    list_session_metadata_read_only,
    read_session_metadata,
    validate_session_id,
)
from ..core.session_execution import session_owner_status
from ..core.session_files import (
    SessionError,
    SessionNotFoundError,
    child_directory,
    session_root,
)
from ..read_page import InvalidParams, Key, page_result, page_room, resume
from ..tools._shared.process import read_background_task_records
from .protocol import ErrorDescriptor, ProtocolError
from .read_catalog import validated_state_read

_SESSION_STATES = {"idle", "running", "approval", "failed", "unknown"}
_AGENT_STATES = {"running", "idle", "waiting", "finished", "failed", "cancelled", "unknown"}
_TASK_STATES = {"running", "finished", "failed", "cancelled", "unknown"}
_COLLECTIONS = {
    "agents": ("started_ns", "run_id"),
    "background_tasks": ("started_ns", "task_id"),
    "attention": ("updated_ns", "attention_id"),
}
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _ns(value):
    if value is None:
        return None
    try:
        stamp = datetime.fromisoformat(value)
        instant = stamp if stamp.tzinfo is not None else stamp.replace(tzinfo=UTC)
        delta = instant - _EPOCH
        return max(0, (delta.days * 86400 + delta.seconds) * 10**9 + delta.microseconds * 1000)
    except ValueError:
        return None


def _size(fd):
    try:
        info = os.stat("conversation.jsonl", dir_fd=fd, follow_symlinks=False)
    except FileNotFoundError:
        return 0
    return info.st_size if stat.S_ISREG(info.st_mode) else 0


def _session_status(session_id, fd, active):
    return active.get(session_id) or ("running" if session_owner_status(fd)[0] else "idle")


def _session_row(metadata, fd, status, updated_ns, root, epoch):
    session_id = metadata.session_id
    stored, action_required = session_attention_counts(root, metadata, epoch)
    return {
        "session_id": session_id, "project_id": metadata.project_id,
        "title": metadata.name or None, "profile": None, "model": metadata.model or None,
        "created_ns": _ns(metadata.created_at) or 0, "updated_ns": updated_ns,
        "status": status, "child_count": len(child_run_directories(fd)),
        "background_count": len(read_background_task_records(fd)), "bytes": _size(fd),
        "attention_count": stored, "attention_open": action_required,
    }


def _agent_rows(fd, path, session_id):
    for run_id, lifecycle, background in read_child_run_metadata(fd, path, session_id):
        session_id, parent_run_id = run_row_identity(run_id)
        status = {"completed": "finished", "canceled": "cancelled", "finishing": "waiting"}.get(
            lifecycle["state"], lifecycle["state"],
        )
        yield {
            "run_id": run_id, "session_id": session_id, "parent_run_id": parent_run_id,
            "profile": lifecycle["agent_type"] or None, "status": status, "background": background,
            "started_ns": _ns(lifecycle["started_at"]), "ended_ns": _ns(lifecycle["finished_at"]),
        }


def _task_rows(fd, session_id, active):
    for row in read_background_task_records(fd):
        phase, exit_code = row.get("terminal_phase"), row.get("exit_code")
        if row.get("running") is True:
            status = "running" if active else "unknown"
        elif phase in {"task_kill", "session_shutdown"}:
            status = "cancelled"
        elif type(exit_code) is int:
            status = "finished" if exit_code == 0 else "failed"
        else:
            status = "unknown"
        # Stored times are monotonic, not Unix timestamps; commands and notes are not summaries.
        yield {
            "task_id": row["task_id"], "session_id": session_id,
            "kind": row.get("owner") if row.get("owner") in ("bash", "run_background", "zeta_background") else "run_background",
            "status": status, "started_ns": None, "ended_ns": None, "summary": None,
        }


def _parse(method, params, collection, detail):
    sessions = collection is None and not detail
    if detail:
        collection = params.get("collection")
    selection = "project_id" if sessions else "session_id"
    select = params.get(selection)
    status = params.get("status")
    limit = params.get("limit", 200 if detail else 50)
    allowed = {selection, "limit", "cursor", "collection" if detail else "status"}
    states = _SESSION_STATES if sessions else (_AGENT_STATES if collection == "agents" else _TASK_STATES)
    if (
        params.keys() - allowed or type(limit) is not int or not 1 <= limit <= 200
        or (select is not None and (type(select) is not str or not select))
        or (status is not None and (type(status) is not str or status not in states))
        or (detail and (select is None or type(collection) is not str or collection not in _COLLECTIONS))
    ):
        raise InvalidParams("invalid session state params")
    if not sessions and select is not None:
        try:
            validate_session_id(select)
        except SessionError:
            raise InvalidParams("invalid session id") from None
    fields = ("updated_ns", "session_id") if sessions else _COLLECTIONS[collection]
    order = (Key(fields[0], int, descending=True), Key(fields[1], str, descending=True))
    start = resume(params.get("cursor"), (method.wire, select, collection if detail else status), (), order)
    return collection, select, status, fields, start, limit


def read_session_page(home, method, params, fits, active, *, collection=None, detail=False):
    sessions = collection is None and not detail
    collection, select, status, (stamp, identity), start, limit = _parse(method, params, collection, detail)
    total = 1000 if detail else None
    room = page_room(start, limit, total)
    root = home / "sessions"

    def generation():
        paths = [home / "storage-format.json", root]
        if not sessions and select is not None:
            paths.append(root / select)
            if detail:
                paths.append(root / select / "attention")
        result = []
        for path in paths:
            try:
                info = path.lstat()
            except FileNotFoundError:
                result.append(None)
            else:
                result.append((info.st_ino, info.st_ctime_ns, info.st_mtime_ns))
        return result

    def read_sessions(epoch):
        candidates = []
        for item in list_session_metadata_read_only(root):
            if select is not None and item.project_id != select:
                continue
            updated = _ns(item.updated_at) or 0
            key = updated + 1, item.session_id
            if start.position is None or key < start.position:
                candidates.append((key, item))
        candidates.sort(key=lambda pair: pair[0], reverse=True)
        rows = []
        if candidates:
            with session_root(root) as root_fd:
                for (updated, sid), item in candidates:
                    if status is None and len(rows) == room:
                        return None, rows, True
                    with child_directory(root_fd, sid) as fd:
                        item_status = _session_status(sid, fd, active)
                        if status is not None and item_status != status:
                            continue
                        if len(rows) == room:
                            return None, rows, True
                        rows.append(_session_row(item, fd, item_status, updated - 1, root, epoch))
        return None, rows, False

    def detail_shape(session, items, cursor):
        return {"session": session, "collection": collection, "items": items, "next_cursor": cursor}

    def read(epoch):
        session = None
        rows = []
        if select is not None:
            try:
                metadata = [read_session_metadata(root, select)]
            except (SessionNotFoundError, FileNotFoundError):
                if detail:
                    raise ProtocolError(ErrorDescriptor.SESSION_NOT_FOUND, "resource not found", {"id": select}) from None
                metadata = []
        else:
            metadata = list_session_metadata_read_only(root)
        if not metadata:
            return session, rows, False
        with session_root(root) as root_fd:
            for item in metadata:
                sid = item.session_id
                with child_directory(root_fd, sid) as fd:
                    if detail:
                        session = _session_row(
                            item, fd, _session_status(sid, fd, active), _ns(item.updated_at) or 0, root, epoch,
                        )
                    if collection == "attention":
                        return attention_records.state_attention_page(
                            root / sid, epoch, start, limit, fits=fits,
                            shape=lambda items, cursor, session=session: detail_shape(session, items, cursor),
                        )
                    if collection == "agents":
                        rows.extend(_agent_rows(fd, root / sid, sid))
                    else:
                        rows.extend(_task_rows(fd, sid, sid in active or session_owner_status(fd)[0]))
        return session, rows, False

    result = validated_state_read(home, "sessions", generation, read_sessions if sessions else read)
    if collection == "attention":
        return result
    session, rows, more = result

    def position(row):
        # Zero is the null sentinel; shift real values so even the Unix epoch sorts first.
        return (0 if row[stamp] is None else row[stamp] + 1, row[identity])

    after = sorted(
        (row for row in rows if (status is None or row["status"] == status)
         and (start.position is None or position(row) < start.position)),
        key=position, reverse=True,
    )
    as_of_ns = time.time_ns()

    def shape(items, cursor):
        if detail:
            return detail_shape(session, items, cursor)
        return {"items": items, "next_cursor": cursor, "as_of_ns": as_of_ns}

    return page_result(
        after[:room], lambda index: position(after[index]), start, more=more or len(after) > room,
        total=total, fits=fits,
        shape=shape,
    )
