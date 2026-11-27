"""One bounded projection of child-run lifecycle state."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..accounts.status import project_account_status
from ..protocol.types import StructuredContentValue, ToolCall
from .agent_progress import live_agent_lifecycle
from .agent_state import (
    BrokenAgentLifecycle,
    LoadedAgentLifecycle,
    MissingAgentLifecycle,
    _parse_agent_state,
    read_agent_lifecycle_state,
)
from .checkpoints import ConversationEntry, load_session_json
from .child_lifecycle import select_legacy_child_lifecycle
from .session_execution import execution_owner_state
from .session_files import child_directory, read_session_file
from .store import ConversationStore
from .store._child_runs import parse_child_run_id, run_row_identity

TERMINAL_AGENT_STATES = frozenset({"completed", "failed", "canceled"})
NONTERMINAL_AGENT_STATES = frozenset({"running", "idle", "finishing", "suspended"})
_VALID_AGENT_STATES = NONTERMINAL_AGENT_STATES | TERMINAL_AGENT_STATES
_MISSING = object()
_STATUS_REASON_FALLBACK = "child status unavailable"
_STATUS_REASON_MAX_BYTES = 8192


def _bounded_text(value: str, max_bytes: int) -> str:
    if len(value.encode("utf-8")) <= max_bytes:
        return value
    return value.encode("utf-8")[:max_bytes].decode("utf-8", "ignore")


def _agent_status_elapsed(
    started_at: str,
    finished_at: str | None,
    *,
    started_monotonic: float | None = None,
    monotonic_pid: int | None = None,
    stored_elapsed: object = _MISSING,
) -> float:
    try:
        started = datetime.fromisoformat(started_at)
        ended = (
            datetime.fromisoformat(finished_at)
            if finished_at is not None
            else datetime.now(UTC)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid child timestamps") from exc
    if finished_at is not None and stored_elapsed is not _MISSING:
        elapsed = max(0.0, float(stored_elapsed))
    elif type(started_monotonic) in {int, float} and monotonic_pid == os.getpid():
        elapsed = max(0.0, time.monotonic() - started_monotonic)
    else:
        elapsed = max(0.0, (ended - started).total_seconds())
    if not math.isfinite(elapsed):
        raise ValueError("invalid child metadata elapsed")
    return elapsed


def _status_int(value: object, field: str, default: int = 0, minimum: int = 0) -> int:
    if value is _MISSING:
        value = default
    if type(value) is not int or value < minimum:
        raise ValueError(f"invalid child metadata {field}")
    return value


def _status_text(value: object, field: str, default: str = "") -> str:
    if value is _MISSING:
        return default
    if type(value) is not str:
        if field == "final_result":
            raise TypeError("invalid child final_result")
        raise TypeError(f"invalid child metadata {field}")
    return value


def _sanitize_status_reason(value: object) -> str:
    try:
        text = str(value)
        sanitized = "".join(
            character if ord(character) >= 32 and ord(character) != 127 else " "
            for character in text
        ).strip()
    except BaseException:  # noqa: BLE001 - formatting must not break the boundary
        return _STATUS_REASON_FALLBACK
    return _bounded_text(sanitized or _STATUS_REASON_FALLBACK, _STATUS_REASON_MAX_BYTES)


def _canonical_child_path(session_dir: Path, value: object) -> Path | None:
    if type(value) is not str or not value:
        return None
    try:
        session_root = session_dir.absolute()
        child_path = Path(value).absolute()
        relative = child_path.relative_to(session_root)
    except (OSError, RuntimeError, ValueError):
        return None
    parts = relative.parts
    if (
        len(parts) < 2
        or len(parts) % 2
        or any(parts[index] != "agents" for index in range(0, len(parts), 2))
        or any(not parts[index].isdigit() for index in range(1, len(parts), 2))
    ):
        return None
    return child_path


def active_agent_receipts(store: object) -> dict[str, Path]:
    receipt_data = getattr(store, "active_agent_receipts", None)
    session_dir = getattr(store, "session_dir", None)
    if not callable(receipt_data) or not isinstance(session_dir, Path):
        raise TypeError("agent status is unavailable outside an agent session")
    receipts: dict[str, Path] = {}

    def add_receipt(handle: object, child_path: object) -> None:
        if type(handle) is not str or not handle:
            return
        canonical_path = _canonical_child_path(session_dir, child_path)
        if canonical_path is not None:
            receipts.setdefault(handle, canonical_path)

    for handle, child_path in receipt_data():
        add_receipt(handle, child_path)

    agent_children = getattr(store, "agent_children", None)
    if callable(agent_children):
        for marker in agent_children().values():
            if type(marker) is not dict:
                continue
            raw_tool_call = marker.get("tool_call")
            if type(raw_tool_call) is not dict:
                continue
            try:
                tool_call = ToolCall.from_dict(raw_tool_call)
            except ValueError:
                continue
            if tool_call.name.casefold() != "agent":
                continue
            add_receipt(
                marker.get("child_instance_id"),
                marker.get("child_session_path"),
            )
    return receipts


def marker_supports_recovery(marker: dict[str, Any]) -> bool:
    # Reading validated markers with a resume descriptor implies the recovery epoch.
    return bool(marker.get("background") and "resume" in marker)


def project_orphaned_lifecycle(
    lifecycle: dict[str, Any], owner: str | None, marker: dict[str, Any],
    *, child_path: Path, completed: bool = False,
) -> dict[str, Any]:
    if (
        owner is None and lifecycle.get("state") in NONTERMINAL_AGENT_STATES - {"suspended"}
        and marker_supports_recovery(marker) and not completed and not lifecycle.get("finished_at")
    ):
        with ConversationStore(child_path.parent, session_id=child_path.name, _read_only=True, _must_exist=True) as child:
            if (
                child.child_recovery_enabled and child.agent_canceled() is None
                and child.pending_prompt_queue.explicit_finish_report_snapshot() is None
            ):
                return {**lifecycle, "state": "suspended", "current_step": "suspended (parent crashed)"}
    return lifecycle


def read_child_lifecycle(
    store: ConversationStore,
    child_path: Path,
    *,
    read_state: Callable[..., object] = read_agent_lifecycle_state,
) -> tuple[object, str | None]:
    with ExitStack() as cleanup:
        directory_fd = parent_fd = store.directory_fd
        for component in child_path.relative_to(store.session_dir.absolute()).parts:
            if component == "agents":
                parent_fd = directory_fd
            directory_fd = cleanup.enter_context(child_directory(directory_fd, component))
        result = read_state(directory_fd, child_path / "agent_lifecycle.json")
        owner = execution_owner_state(directory_fd)
        if owner is not None or not isinstance(result, LoadedAgentLifecycle):
            return result, owner
        lifecycle = result.value
        if lifecycle.get("state") not in NONTERMINAL_AGENT_STATES - {"suspended"}:
            return result, owner
        run_id = lifecycle.get("handle")
        if parent_fd == store.directory_fd:
            markers = store.agent_children()
            completed = run_id in store.agent_completion_notifications_by_child()
        else:
            try:
                parent_state = load_session_json(read_session_file(parent_fd, "session_state.json"))
            except FileNotFoundError:
                return result, owner
            markers = _parse_agent_state(parent_state, child_path.parent.parent / "session_state.json")["agent_children"]
            completed = False
        marker = markers.get(run_id, {})

        projected = project_orphaned_lifecycle(lifecycle, owner, marker, child_path=child_path, completed=completed)
        return LoadedAgentLifecycle(projected), owner


def unreadable_agent_status(
    handle: str, *, reason: str, started_at: object = None
) -> dict[str, StructuredContentValue]:
    return {
        "handle": handle,
        "state": "failed",
        "started_at": (started_at if type(started_at) is str else "unknown"),
        "finished_at": "unknown",
        "elapsed": 0.0,
        "turns_used": 0,
        "tool_calls": 0,
        "current_step": "unreadable",
        "depth": 0,
        "agent_type": "general",
        "description": "",
        "reason": f"unreadable: {reason}",
    }


def _lost_agent_status(
    item: dict[str, StructuredContentValue], *, requested: bool
) -> dict[str, StructuredContentValue]:
    lost = dict(item)
    lost.pop("account_status", None)
    lost.update(
        state="failed",
        finished_at="unknown",
        elapsed=0.0,
        current_step="lost",
        reason="lost: owner process exited",
    )
    if requested:
        lost["final_result"] = "lost: owner process exited"
    return lost


def project_agent_status(
    lifecycle: object,
    handle: str,
    *,
    notifications: Sequence[ConversationEntry],
    requested_handle: str | None = None,
) -> dict[str, StructuredContentValue]:
    if type(lifecycle) is not dict:
        raise ValueError("lifecycle data is not an object")
    if lifecycle.get("handle") != handle:
        raise ValueError("child handle mismatch")

    started_at = lifecycle.get("started_at", "unknown")
    finished_value = lifecycle.get("finished_at", _MISSING)
    finished_at = None if finished_value is _MISSING else finished_value
    state = _status_text(lifecycle.get("state", _MISSING), "state", "unknown")
    if state not in _VALID_AGENT_STATES:
        raise ValueError("invalid child metadata state")
    if state in TERMINAL_AGENT_STATES and finished_at is None:
        raise ValueError("terminal state has no finished_at")
    if state in NONTERMINAL_AGENT_STATES and finished_at is not None:
        raise ValueError("non-terminal state has finished_at")

    elapsed = _agent_status_elapsed(
        started_at,
        finished_at,
        started_monotonic=lifecycle.get("started_monotonic"),
        monotonic_pid=lifecycle.get("monotonic_pid"),
        stored_elapsed=lifecycle.get("elapsed", _MISSING),
    )
    turns_used = _status_int(lifecycle.get("turns_used", _MISSING), "turns_used")
    tool_calls = _status_int(lifecycle.get("tool_calls", _MISSING), "tool_calls")
    depth = _status_int(lifecycle.get("depth", _MISSING), "depth", minimum=1)
    current_step = _status_text(
        lifecycle.get("current_step", _MISSING), "current_step", "unknown"
    )
    agent_type = _status_text(
        lifecycle.get("agent_type", _MISSING), "agent_type", "general"
    )
    description = _status_text(lifecycle.get("description", _MISSING), "description")
    cwd = _status_text(lifecycle.get("cwd", _MISSING), "cwd", "")
    item: dict[str, StructuredContentValue] = {
        "handle": handle,
        "state": state,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed": elapsed,
        "turns_used": turns_used,
        "tool_calls": tool_calls,
        "current_step": current_step,
        "depth": depth,
        "agent_type": agent_type,
        "description": description,
        "cwd": cwd,
    }
    if state in NONTERMINAL_AGENT_STATES and (
        account_status := project_account_status(lifecycle.get("account_status"))
    ) is not None:
        item["account_status"] = dict(account_status)
    if state == "idle":
        idle_since = lifecycle.get("idle_since")
        if type(idle_since) is str:
            item["idle_since"] = idle_since
    report_status = next(
        (
            entry.data.get("report_status")
            for entry in reversed(notifications)
            if entry.data.get("child_instance_id") == handle
            and entry.data.get("report_status") in {"done", "blocked", "failed"}
        ),
        None,
    )
    if isinstance(report_status, str):
        item["report_status"] = report_status
    final_result = (
        _status_text(lifecycle.get("final_result", _MISSING), "final_result")
        if finished_at is not None
        else ""
    )
    if state == "failed" and final_result.startswith(("lost:", "unreadable:")):
        item["reason"] = final_result
    if finished_at is not None and requested_handle is not None:
        item["final_result"] = final_result
    return item


def read_agent_status(
    store: object,
    *,
    requested_handle: str | None = None,
    running_handles: frozenset[str] | None = None,
    read_lifecycle: Callable[[ConversationStore, Path], tuple[object, str | None]] = read_child_lifecycle,
    project_status: Callable[..., dict[str, StructuredContentValue]] = project_agent_status,
) -> list[dict[str, StructuredContentValue]]:
    session_dir = getattr(store, "session_dir", None)
    if not isinstance(session_dir, Path):
        raise TypeError("agent status is unavailable outside an agent session")
    active_receipts = active_agent_receipts(store)
    if requested_handle is not None and requested_handle not in active_receipts:
        raise ValueError("unknown child handle")
    children: list[dict[str, StructuredContentValue]] = []
    notifications = getattr(store, "agent_notifications", lambda **_kwargs: [])(
        pending_only=False
    )
    for handle, child_path in active_receipts.items():
        if requested_handle is not None and handle != requested_handle:
            continue
        try:
            lifecycle_read, owner_state = read_lifecycle(store, child_path)
            if isinstance(lifecycle_read, MissingAgentLifecycle):
                item = unreadable_agent_status(handle, reason="missing lifecycle record")
            elif isinstance(lifecycle_read, BrokenAgentLifecycle):
                item = unreadable_agent_status(handle, reason=lifecycle_read.reason)
            elif isinstance(lifecycle_read, LoadedAgentLifecycle):
                live = live_agent_lifecycle(child_path / "agent_lifecycle.json")
                lifecycle = select_legacy_child_lifecycle(lifecycle_read.value, live)
                item = project_status(
                    lifecycle,
                    handle,
                    notifications=notifications,
                    requested_handle=requested_handle,
                )
                if (
                    running_handles is not None
                    and handle not in running_handles
                    and owner_state in {None, "recovery"}
                    and item["state"] in NONTERMINAL_AGENT_STATES - {"suspended"}
                ):
                    item = _lost_agent_status(item, requested=requested_handle is not None)
            else:
                raise TypeError("unknown lifecycle read result")
        except Exception as exc:  # noqa: BLE001 - every child needs a safe row
            children.append(
                unreadable_agent_status(
                    handle,
                    reason=_sanitize_status_reason(exc),
                )
            )
            continue
        children.append(item)
    return sorted(children, key=lambda child: str(child["started_at"]))


def child_run_for_features(row: dict[str, Any], *, child_recovery: bool) -> dict[str, Any]:
    """Project a child run row, not its event or history envelope."""
    if not child_recovery and row.get("status") == "suspended":
        return {**row, "status": "running"}
    return row


def project_child_run_row(
    item: dict[str, Any],
    *,
    session_id: str,
    parent_run_id: str | None,
    accepts_follow_ups: bool,
) -> dict[str, StructuredContentValue]:
    account = (
        project_account_status(item.get("account_status"))
        if item["state"] in NONTERMINAL_AGENT_STATES else None
    )
    return {
        "run_id": str(item["handle"]),
        **({"account_status": dict(account)} if account is not None else {}),
        "session_id": session_id,
        "parent_run_id": parent_run_id,
        "description": _bounded_text(item["description"], 3_998),
        "agent_type": _bounded_text(item["agent_type"], 3_998),
        "status": _bounded_text(item["state"], 3_998),
        "accepts_follow_ups": accepts_follow_ups,
        "turns_used": item["turns_used"],
        "tool_count": item["tool_calls"],
        "current_step": _bounded_text(item["current_step"], 3_998),
        "started_at": _bounded_text(item["started_at"], 3_998),
        "finished_at": (
            _bounded_text(item["finished_at"], 3_998)
            if item["finished_at"] is not None
            else None
        ),
        "elapsed_seconds": item["elapsed"],
    }


def child_run_rows(
    store: ConversationStore,
    *,
    running_handles: frozenset[str] | None = None,
) -> list[dict[str, StructuredContentValue]]:
    """Return the public bounded row shared by `/runs` and serve."""
    markers = {
        marker.get("child_instance_id"): marker
        for marker in store.agent_children().values()
        if type(marker) is dict and type(marker.get("child_instance_id")) is str
    }
    rows: list[dict[str, StructuredContentValue]] = []
    for item in read_agent_status(store, running_handles=running_handles):
        run_id = str(item["handle"])
        session_id, parent_run_id = run_row_identity(run_id)
        marker = markers.get(run_id, {})
        row = project_child_run_row(
            item,
            session_id=session_id,
            parent_run_id=parent_run_id,
            accepts_follow_ups=marker.get("accepts_follow_ups") is True,
        )
        rows.append(row)
    return rows


def child_run_directories(directory_fd: int) -> list[str]:
    """List the numeric child slots without opening any child transcript."""
    try:
        with child_directory(directory_fd, "agents") as agents_fd, os.scandir(agents_fd) as entries:
            return [
                entry.name for entry in entries
                if entry.name.isascii() and entry.name.isdecimal()
                and not entry.name.startswith("0") and entry.is_dir(follow_symlinks=False)
            ]
    except FileNotFoundError:
        return []


def read_child_run_metadata(directory_fd: int, session_dir: Path, session_id: str):
    """Yield child metadata with a nullable background flag from the parent marker."""
    def descend(parent_fd: int, parent_dir: Path, parent_id: str):
        children = child_run_directories(parent_fd)
        if not children:
            return
        try:
            value = load_session_json(read_session_file(parent_fd, "session_state.json"))
            markers = _parse_agent_state(value, parent_dir / "session_state.json")["agent_children"]
        except FileNotFoundError:
            markers = {}
        background_by_id = {
            marker.get("child_instance_id"): marker.get("background", False)
            for marker in markers.values()
        }
        with child_directory(parent_fd, "agents") as agents_fd:
            for name in children:
                run_id = f"{parent_id}:{name}"
                parse_child_run_id(session_id, run_id)
                path = parent_dir / "agents" / name
                with child_directory(agents_fd, name) as child_fd:
                    loaded = read_agent_lifecycle_state(child_fd, path / "agent_lifecycle.json")
                    if isinstance(loaded, LoadedAgentLifecycle) and loaded.value.get("handle") == run_id:
                        lifecycle = select_legacy_child_lifecycle(
                            loaded.value, live_agent_lifecycle(path / "agent_lifecycle.json")
                        )
                        if isinstance(lifecycle, dict) and lifecycle.get("handle") == run_id:
                            yield run_id, lifecycle, background_by_id.get(run_id)
                    yield from descend(child_fd, path, run_id)

    yield from descend(directory_fd, session_dir, session_id)
