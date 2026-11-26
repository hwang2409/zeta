"""Serve projections over the project, memory, inbox, and attention owners."""

from collections.abc import Callable
from pathlib import Path
from time import time_ns
from typing import Any

from ... import attention_records, storage_format
from ...attention_decisions import open_attention_items
from ...attention_records import AttentionRecord
from ...memory.transaction import STATE_MEMORY_KINDS
from ...project_inbox import ProjectInbox
from ...project_inbox import is_message_id as _valid_message_id
from ...project_registry import ProjectRegistry
from ...project_schema import PROJECT_ID_PATTERN
from ...read_page import InvalidParams, Key, Resume, descending_page, resume
from ...state_snapshot import stamps, timestamp_ns
from ..protocol import ErrorDescriptor, ProtocolError

_ORDER = {
    "projects": ("updated_ns", "project_id"),
    "memory": ("updated_ns", "memory_id"),
    "inbox": ("created_ns", "message_id"),
    "attention": ("created_ns", "attention_id"),
}
_STATUSES = ("new", "claimed", "done")
_FILTERS = {
    "memory": ("kind", STATE_MEMORY_KINDS),
    "inbox": ("status", _STATUSES),
    "attention": ("status", ("open", "done")),
}
_DETAILS = {"memory_detail": "memory_id", "inbox_detail": "message_id"}


def request(collection: str, params: dict[str, Any]) -> tuple[tuple, Resume | None, int]:
    project_id = params.get("project_id")
    allowed = {"project_id"} if collection != "projects" else set()
    if (collection not in {"projects", "attention"} or project_id is not None) and (
        not isinstance(project_id, str) or not PROJECT_ID_PATTERN.fullmatch(project_id)
    ):
        raise InvalidParams("invalid project id")
    if collection in _DETAILS:
        field = _DETAILS[collection]
        ident = params.get(field)
        if params.keys() != {"project_id", field} or not isinstance(ident, str) or not 0 < len(ident) <= 128:
            raise InvalidParams("invalid detail params")
        if collection == "inbox_detail" and not _valid_message_id(ident):
            raise InvalidParams("invalid message id")
        return (collection, project_id, ident), None, 0
    value = None
    if collection in _FILTERS:
        field, variants = _FILTERS[collection]
        allowed.add(field)
        value = params.get(field)
        if value is not None and value not in variants:
            raise InvalidParams("invalid filter")
    attention_list = None
    if collection == "attention":
        allowed.add("list")
        attention_list = params.get("list")
        if attention_list is not None and (attention_list not in ("action", "fyi") or value == "done"):
            raise InvalidParams("invalid attention list")
    limit = params.get("limit", 50)
    if params.keys() - allowed - {"limit", "cursor"} or type(limit) is not int or not 1 <= limit <= 200:
        raise InvalidParams("invalid page params")
    scope = (collection, project_id, value)
    if attention_list is not None:
        scope += (attention_list,)
    return scope, resume(params.get("cursor"), scope, (), _order(collection)), limit


def _order(collection: str) -> tuple[Key, Key]:
    timestamp, ident = _ORDER[collection]
    return Key(timestamp, int, True), Key(ident, str, True)


def page(collection, snapshot, start, limit, fits):
    if start is None:
        if snapshot is None:
            raise InvalidParams("resource not found")
        return snapshot
    as_of_ns = time_ns()
    return descending_page(
        snapshot, _order(collection), start, limit, fits=fits,
        shape=lambda items, cursor: {"items": items, "next_cursor": cursor, "as_of_ns": as_of_ns},
    )


def snapshot(home: Path, scope: tuple, stable: Callable) -> Any:
    collection, project_id, selector, *attention_list = scope
    resource = collection.removesuffix("_detail")
    manifest = home / "storage-format.json"

    sessions = home / "sessions"
    open_queue = collection == "attention" and (selector == "open" or bool(attention_list))

    def owners(epoch):
        registry = ProjectRegistry.for_state_read(home / "projects", epoch)
        return registry, registry.memory_transactions, ProjectInbox(registry, sessions_root=sessions)

    def generation():
        manifest_stamp = stamps([manifest])
        with storage_format.open_reader(home) as epoch:
            registry, memory, inbox = owners(epoch)
            if collection == "projects":
                projects = registry.state_project_paths()
                state = (
                    stamps([registry.root, *projects, *(p / "sessions.jsonl" for p in projects)]),
                    tuple((memory.state_generation(p.name), inbox.state_generation(p.name)) for p in projects),
                )
            elif collection == "attention":
                state = attention_records.state_attention_home_generation(sessions, include_owner=open_queue)
            else:
                owner = memory if collection.startswith("memory") else inbox
                state = owner.state_generation(project_id)
        return manifest_stamp, state

    def rows():
        with storage_format.open_reader(home) as epoch:
            registry, memory, inbox = owners(epoch)
            if collection == "projects":
                return [_project_row(project, count, registry, inbox)
                        for project, count in registry.state_projects_read_only()]
            if collection == "memory":
                return memory.state_rows(project_id, kind=selector)
            if collection == "memory_detail":
                return memory.state_rows(project_id, memory_id=selector)
            if collection == "inbox":
                return inbox.state_rows(project_id, status=selector)
            if collection == "inbox_detail":
                return inbox.state_rows(project_id, message_id=selector)
            if open_queue:
                return [
                    item.record.state_row()
                    for item in open_attention_items(home, project_id, epoch=epoch)
                    if isinstance(item.record, AttentionRecord)
                    and (not attention_list or item.is_fyi == (attention_list[0] == "fyi"))
                ]
            return attention_records.state_attention_home_rows(sessions, epoch, project_id, selector)

    try:
        return stable(generation, rows)
    except (storage_format.MissingManifest, OSError):
        raise ProtocolError(
            ErrorDescriptor.STATE_READ_UNAVAILABLE, "state unavailable", {"retryable": True},
        ) from None
    except storage_format.StorageFormatError:
        resource = "storage-format.json"
    except ValueError:
        pass
    raise ProtocolError(ErrorDescriptor.CORRUPT_STATE, "stored state is invalid", {"resource": resource})


def _project_row(project, session_count, registry, inbox):
    project_id = project.project_id
    messages = inbox.state_rows(project_id)
    return {
        "project_id": project_id, "name": project.name, "path": project.canonical_integration_root,
        "session_count": session_count, "memory_count": len(registry.memory_transactions.state_rows(project_id)),
        "inbox": {state: sum(row["status"] == state for row in messages) for state in _STATUSES},
        "updated_ns": timestamp_ns(project.updated_at),
    }
