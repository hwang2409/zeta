"""Authenticated, read-only views of one agent subtree."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ...core.store import ConversationStore
from ...protocol.types import ToolCall

if TYPE_CHECKING:
    from ...core.checkpoints import ConversationEntry


def _caller_handle(store: ConversationStore) -> str:
    return store.agent_handle() or store.session_id


def _root_path(store: ConversationStore, caller: str) -> Path:
    path = store.session_dir.absolute()
    for _ in range(caller.count(":")):
        path = path.parent.parent
    return path


def _segments(root_id: str, handle: object) -> tuple[str, ...] | None:
    if type(handle) is not str or not handle.startswith(f"{root_id}:"):
        return None
    parts = handle.split(":")
    if parts[0] != root_id or any(
        not part.isdigit() or int(part) < 1 for part in parts[1:]
    ):
        return None
    return tuple(parts[1:])


def _path_for(root_path: Path, root_id: str, handle: object) -> Path | None:
    if handle == root_id:
        return root_path
    segments = _segments(root_id, handle)
    if segments is None:
        return None
    path = root_path
    for segment in segments:
        path = path / "agents" / segment
    return path


def _agent_call(block: object) -> ToolCall | None:
    if not isinstance(block, dict) or block.get("type") != "tool_use":
        return None
    raw = block.get("tool_call")
    if not isinstance(raw, dict):
        return None
    try:
        call = ToolCall.from_dict(raw)
    except ValueError:
        return None
    return call if call.name.casefold() == "agent" else None


def _active_agent_calls(entries: list[ConversationEntry]) -> dict[str, ToolCall]:
    calls: dict[str, ToolCall] = {}
    for entry in entries:
        message = entry.data.get("message") if entry.type == "message" else None
        if not isinstance(message, dict):
            continue
        for block in message.get("content", []):
            call = _agent_call(block)
            if call is not None:
                calls[call.id] = call
    return calls


def _open_snapshot(path: Path) -> ConversationStore:
    return ConversationStore(
        path.parent,
        session_id=path.name,
        _read_only=True,
        _must_exist=True,
    )


def _logical_parent(handle: str) -> str:
    return handle.rpartition(":")[0]


def _direct_child(parent: str, child: str) -> bool:
    return _logical_parent(child) == parent


def _authenticated_marker_call(
    marker: object, calls: dict[str, ToolCall]
) -> ToolCall | None:
    if not isinstance(marker, dict):
        return None
    raw_call = marker.get("tool_call")
    if not isinstance(raw_call, dict):
        return None
    try:
        call = ToolCall.from_dict(raw_call)
    except ValueError:
        return None
    if call.name.casefold() != "agent" or calls.get(call.id) != call:
        return None
    return call


def _started_child_handle(entry: ConversationEntry, *, parent: str) -> str | None:
    if entry.type != "child_run_started":
        return None
    handle = entry.data.get("run_id")
    expected_parent = parent if ":" in parent else None
    if (
        type(handle) is not str
        or not _direct_child(parent, handle)
        or entry.data.get("parent_run_id") != expected_parent
    ):
        return None
    return handle


class _SubtreeDiscovery:
    def __init__(self, store: ConversationStore, requested_handle: str | None) -> None:
        self.store = store
        self.requested_handle = requested_handle
        self.caller = _caller_handle(store)
        self.root_id = self.caller.split(":", 1)[0]
        self.root_path = _root_path(store, self.caller)
        self.found: dict[str, Path] = {}
        self.opened: dict[str, ConversationStore] = {}
        self.pending = [self.caller]

    def discover(self) -> dict[str, Path]:
        self._validate_request()
        try:
            while self.pending:
                parent = self.pending.pop()
                parent_store = (
                    self.store if parent == self.caller else self._snapshot(parent)
                )
                if parent_store is not None:
                    self._visit(parent, parent_store)
        finally:
            for opened_store in self.opened.values():
                opened_store.close()

        if self.requested_handle is None:
            return self.found
        path = self.found.get(self.requested_handle)
        if path is None:
            raise ValueError("unknown child handle")
        return {self.requested_handle: path}

    def _validate_request(self) -> None:
        if self.requested_handle is None:
            return
        requested = _segments(self.root_id, self.requested_handle)
        caller = _segments(self.root_id, self.caller) or ()
        if (
            requested is None
            or len(requested) <= len(caller)
            or requested[: len(caller)] != caller
        ):
            raise ValueError("unknown child handle")

    def _snapshot(self, handle: str) -> ConversationStore | None:
        existing = self.opened.get(handle)
        if existing is not None:
            return existing
        path = _path_for(self.root_path, self.root_id, handle)
        if path is None:
            return None
        try:
            snapshot = _open_snapshot(path)
        except (OSError, ValueError):
            return None
        self.opened[handle] = snapshot
        return snapshot

    def _visit(self, parent: str, parent_store: ConversationStore) -> None:
        entries = parent_store.replay()
        calls = _active_agent_calls(entries)
        self._add_receipt_edges(parent, parent_store)
        self._add_started_edges(parent, entries, calls)
        self._add_marker_edges(parent, parent_store, calls)

    def _add_receipt_edges(self, parent: str, parent_store: ConversationStore) -> None:
        parent_path = _path_for(self.root_path, self.root_id, parent)
        for handle, stored_path in parent_store.active_agent_receipts():
            if _segments(self.root_id, handle) is not None and _direct_child(
                parent, handle
            ):
                self._add_edge(handle, stored_path, parent=parent)
                continue
            stored = Path(stored_path).absolute()
            if (
                self.requested_handle is None
                and parent_path is not None
                and stored.parent == parent_path / "agents"
            ):
                self.found[handle] = stored

    def _add_started_edges(
        self,
        parent: str,
        entries: list[ConversationEntry],
        calls: dict[str, ToolCall],
    ) -> None:
        for entry in entries:
            handle = _started_child_handle(entry, parent=parent)
            if handle is None:
                continue
            child_store = self._snapshot(handle)
            marker = (
                getattr(child_store, "_agent_parent", None) if child_store else None
            )
            if not isinstance(marker, dict) or marker.get("tool_call_id") not in calls:
                continue
            child_path = _path_for(self.root_path, self.root_id, handle)
            if child_path is not None:
                self._add_edge(handle, str(child_path), parent=parent)

    def _add_marker_edges(
        self,
        parent: str,
        parent_store: ConversationStore,
        parent_calls: dict[str, ToolCall],
    ) -> None:
        for marker in parent_store.agent_children().values():
            if not isinstance(marker, dict):
                continue
            handle = marker.get("child_instance_id")
            if type(handle) is not str:
                continue
            logical_parent = _logical_parent(handle)
            logical_store = (
                parent_store
                if logical_parent == parent
                else self._snapshot(logical_parent)
            )
            if logical_store is None:
                continue
            calls = (
                parent_calls
                if logical_store is parent_store
                else _active_agent_calls(logical_store.replay())
            )
            if _authenticated_marker_call(marker, calls) is not None:
                self._add_edge(
                    handle,
                    marker.get("child_session_path"),
                    parent=logical_parent,
                )

    def _add_edge(self, handle: object, stored_path: object, *, parent: str) -> None:
        if type(handle) is not str or type(stored_path) is not str:
            return
        path = _path_for(self.root_path, self.root_id, handle)
        if path is None or Path(stored_path).absolute() != path:
            return
        if not handle.startswith(f"{self.caller}:"):
            return
        requested = self.requested_handle
        if requested is not None and not (
            requested == handle
            or requested.startswith(f"{handle}:")
            or handle.startswith(f"{requested}:")
        ):
            return
        if handle not in self.found:
            self.found[handle] = path
            if _direct_child(parent, handle):
                self.pending.append(handle)


def _discover(
    store: ConversationStore, requested_handle: str | None
) -> dict[str, Path]:
    return _SubtreeDiscovery(store, requested_handle).discover()


def read_output_result(
    store: ConversationStore,
    *,
    requested_handle: str,
    offset: int,
    limit: int | None,
    max_bytes: int,
) -> dict[str, object]:
    from . import _read_agent_output_at_path

    path = _discover(store, requested_handle)[requested_handle]
    return _read_agent_output_at_path(
        store,
        path,
        requested_handle=requested_handle,
        offset=offset,
        limit=limit,
        max_bytes=max_bytes,
    )


def read_status_result(
    store: ConversationStore,
    *,
    requested_handle: str | None,
    offset: int,
    limit: int | None,
    max_bytes: int,
) -> dict[str, object]:
    from . import (
        _agent_error,
        _fit_status_row,
        _is_finished_agent_status,
        _project_agent_status,
        _read_agent_output_at_path,
        _sanitize_status_reason,
        _status_response,
        _unknown_agent_status,
        encode_json,
        idle_resource_usage,
        text_block,
    )

    receipts = _discover(store, requested_handle)
    fresh_store = _open_snapshot(store.session_dir.absolute())
    try:
        notifications = fresh_store.agent_notifications(pending_only=False)
    finally:
        fresh_store.close()
    children = []
    for handle, child_path in receipts.items():
        try:
            _read_agent_output_at_path(
                store,
                child_path,
                requested_handle=handle,
                offset=0,
                limit=1,
                max_bytes=max_bytes,
            )
            item = _project_agent_status(
                store,
                handle,
                child_path,
                notifications=notifications,
                requested_handle=requested_handle,
            )
        except Exception as exc:  # noqa: BLE001 - damage is isolated per leaf
            item = _unknown_agent_status(handle, reason=_sanitize_status_reason(exc))
        children.append(item)
    children.sort(key=lambda child: str(child["started_at"]))

    if requested_handle is None:
        finished_omitted = sum(_is_finished_agent_status(child) for child in children)
        children = [child for child in children if not _is_finished_agent_status(child)]
    else:
        finished_omitted = 0
    total = len(children)
    root_idle_usage = idle_resource_usage(children)
    raw_page = children[offset : offset + limit if limit is not None else None]
    page = []
    for index, child in enumerate(raw_page):
        child_offset = offset + index
        child_truncated = child_offset + 1 < total
        fitted = _fit_status_row(
            child,
            offset=child_offset,
            total=total,
            truncated=child_truncated,
            next_offset=child_offset + 1 if child_truncated else None,
            finished_omitted=finished_omitted,
            max_bytes=max_bytes,
        )
        if fitted is None:
            break
        page.append(fitted)

    while page:
        next_offset = offset + len(page)
        truncated = next_offset < total
        result = _status_response(
            page,
            offset=offset,
            total=total,
            truncated=truncated,
            next_offset=next_offset if truncated else None,
            finished_omitted=finished_omitted,
            idle_usage=root_idle_usage,
        )
        try:
            if len(encode_json(result)) <= max_bytes:
                return result
        except (OverflowError, RecursionError, TypeError, ValueError, UnicodeError):
            pass
        page.pop()
    if offset < total:
        return _agent_error("status item exceeds response limit", max_bytes)
    count = len(children)
    label = "child" if count == 1 else "children"
    result = _status_response(
        [],
        offset=offset,
        total=total,
        truncated=False,
        next_offset=None,
        finished_omitted=finished_omitted,
        idle_usage=root_idle_usage,
    )
    if count == 1:
        result["content"] = [
            text_block(
                f"agent status: {count} {label}"
                + (
                    f"\n{finished_omitted} finished children omitted "
                    "(query by handle for results)"
                    if finished_omitted
                    else ""
                )
            )
        ]
    try:
        return (
            result
            if len(encode_json(result)) <= max_bytes
            else _agent_error("response exceeds response limit", max_bytes)
        )
    except (OverflowError, RecursionError, TypeError, ValueError, UnicodeError):
        return _agent_error("response exceeds response limit", max_bytes)
