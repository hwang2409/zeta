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


def _active_agent_calls(entries: list[ConversationEntry]) -> dict[str, ToolCall]:
    calls: dict[str, ToolCall] = {}
    for entry in entries:
        if entry.type != "message":
            continue
        message = entry.data.get("message")
        if not isinstance(message, dict):
            continue
        for block in message.get("content", []):
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            raw = block.get("tool_call")
            if not isinstance(raw, dict):
                continue
            try:
                call = ToolCall.from_dict(raw)
            except ValueError:
                continue
            if call.name.casefold() == "agent":
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


def _discover(
    store: ConversationStore, requested_handle: str | None
) -> dict[str, Path]:
    caller = _caller_handle(store)
    root_id = caller.split(":", 1)[0]
    root_path = _root_path(store, caller)
    requested_segments = (
        _segments(root_id, requested_handle) if requested_handle else None
    )
    caller_segments = _segments(root_id, caller) or ()
    if requested_handle is not None and (
        requested_segments is None
        or len(requested_segments) <= len(caller_segments)
        or requested_segments[: len(caller_segments)] != caller_segments
    ):
        raise ValueError("unknown child handle")

    found: dict[str, Path] = {}
    opened: dict[str, ConversationStore] = {}
    pending = [caller]

    def snapshot(handle: str) -> ConversationStore | None:
        existing = opened.get(handle)
        if existing is not None:
            return existing
        path = _path_for(root_path, root_id, handle)
        if path is None:
            return None
        try:
            existing = _open_snapshot(path)
        except (OSError, ValueError):
            return None
        opened[handle] = existing
        return existing

    def add(handle: object, stored_path: object, *, parent: str) -> None:
        if type(handle) is not str or type(stored_path) is not str:
            return
        path = _path_for(root_path, root_id, handle)
        if path is None or Path(stored_path).absolute() != path:
            return
        if not handle.startswith(f"{caller}:"):
            return
        if requested_handle is not None and not (
            requested_handle == handle
            or requested_handle.startswith(f"{handle}:")
            or handle.startswith(f"{requested_handle}:")
        ):
            return
        if handle not in found:
            found[handle] = path
            if _direct_child(parent, handle):
                pending.append(handle)

    try:
        resident_entries = store.replay()
        resident_calls = _active_agent_calls(resident_entries)
        for handle, stored_path_value in store.active_agent_receipts():
            stored_path = Path(stored_path_value).absolute()
            if _segments(root_id, handle) is not None and _direct_child(caller, handle):
                add(handle, str(stored_path), parent=caller)
            elif (
                requested_handle is None
                and stored_path.parent == store.session_dir.absolute() / "agents"
            ):
                found[handle] = stored_path
        for marker in store.agent_children().values():
            if not isinstance(marker, dict):
                continue
            raw_call = marker.get("tool_call")
            try:
                marker_call = ToolCall.from_dict(raw_call) if isinstance(raw_call, dict) else None
            except ValueError:
                marker_call = None
            if (
                marker_call is not None
                and resident_calls.get(marker_call.id) == marker_call
            ):
                add(
                    marker.get("child_instance_id"),
                    marker.get("child_session_path"),
                    parent=caller,
                )

        while pending:
            parent = pending.pop()
            parent_store = snapshot(parent)
            if parent_store is None:
                continue
            entries = parent_store.replay()
            calls = _active_agent_calls(entries)

            for handle, stored_path in parent_store.active_agent_receipts():
                if _segments(root_id, handle) is not None and _direct_child(parent, handle):
                    add(handle, stored_path, parent=parent)
                elif requested_handle is None:
                    parent_path = _path_for(root_path, root_id, parent)
                    stored = Path(stored_path).absolute()
                    if parent_path is not None and stored.parent == parent_path / "agents":
                        found[handle] = stored

            for entry in entries:
                if entry.type != "child_run_started":
                    continue
                handle = entry.data.get("run_id")
                if type(handle) is not str or not _direct_child(parent, handle):
                    continue
                expected_parent = parent if ":" in parent else None
                if entry.data.get("parent_run_id") != expected_parent:
                    continue
                child_path = _path_for(root_path, root_id, handle)
                child_store = snapshot(handle)
                marker = (
                    getattr(child_store, "_agent_parent", None) if child_store else None
                )
                call_id = (
                    marker.get("tool_call_id") if isinstance(marker, dict) else None
                )
                if child_path is not None and call_id in calls:
                    add(handle, str(child_path), parent=parent)

            for marker in parent_store.agent_children().values():
                if not isinstance(marker, dict):
                    continue
                handle = marker.get("child_instance_id")
                if type(handle) is not str:
                    continue
                logical_parent = _logical_parent(handle)
                logical_store = snapshot(logical_parent)
                if logical_store is None:
                    continue
                raw_call = marker.get("tool_call")
                if not isinstance(raw_call, dict):
                    continue
                try:
                    marker_call = ToolCall.from_dict(raw_call)
                except ValueError:
                    continue
                if marker_call.name.casefold() != "agent":
                    continue
                logical_calls = _active_agent_calls(logical_store.replay())
                if logical_calls.get(marker_call.id) != marker_call:
                    continue
                add(handle, marker.get("child_session_path"), parent=logical_parent)
    finally:
        for opened_store in opened.values():
            opened_store.close()

    if requested_handle is not None:
        path = found.get(requested_handle)
        if path is None:
            raise ValueError("unknown child handle")
        return {requested_handle: path}
    return found


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
