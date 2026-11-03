"""Identity and durable payload rules for child runs."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..checkpoints import ConversationEntry
    from ._store import ConversationStore

_MAX_RUN_ID_BYTES = 256
_MAX_RUN_DEPTH = 64
_RUN_SEGMENT = re.compile(r"[1-9][0-9]*\Z")
_MAX_STORED_TEXT_BYTES = 1024
_MAX_TIMESTAMP_BYTES = 64


def _bounded_utf8(value: str, limit: int) -> str:
    return value.encode("utf-8")[:limit].decode("utf-8", "ignore")


def parse_child_run_id(parent_session_id: str, run_id: str) -> tuple[int, str | None]:
    """Validate a public run id and return its leaf number and parent id."""
    if type(parent_session_id) is not str or not parent_session_id:
        raise ValueError("invalid parent session id")
    if type(run_id) is not str or len(run_id.encode("utf-8")) > _MAX_RUN_ID_BYTES:
        raise ValueError("invalid child run id")
    prefix = f"{parent_session_id}:"
    if not run_id.startswith(prefix):
        raise ValueError("child run id has the wrong parent")
    segments = run_id[len(prefix) :].split(":")
    if not 1 <= len(segments) <= _MAX_RUN_DEPTH:
        raise ValueError("invalid child run id depth")
    if any(_RUN_SEGMENT.fullmatch(segment) is None for segment in segments):
        raise ValueError("invalid child run id segment")
    return int(segments[-1]), run_id.rpartition(":")[0] if len(segments) > 1 else None


def validate_child_run_started_data(data: Mapping[str, Any]) -> None:
    """Validate the bounded internal start-row payload."""
    run_id = data.get("run_id")
    parent_run_id = data.get("parent_run_id")
    name = data.get("name")
    task = data.get("task")
    started_at = data.get("started_at")
    if (
        set(data) - {"run_id", "parent_run_id", "name", "task", "started_at"}
        or type(run_id) is not str
        or not run_id
        or len(run_id.encode("utf-8")) > _MAX_RUN_ID_BYTES
        or type(parent_run_id) not in {str, type(None)}
        or parent_run_id == ""
        or type(task) is not str
        or not task
        or len(task.encode("utf-8")) > _MAX_STORED_TEXT_BYTES
        or type(started_at) is not str
        or not started_at
        or len(started_at.encode("utf-8")) > _MAX_TIMESTAMP_BYTES
        or (
            name is not None
            and (
                type(name) is not str
                or not name
                or len(name.encode("utf-8")) > _MAX_STORED_TEXT_BYTES
            )
        )
    ):
        raise ValueError("invalid child run started payload")
    parts = run_id.split(":")
    if len(parts) < 2 or len(parts) - 1 > _MAX_RUN_DEPTH or any(
        _RUN_SEGMENT.fullmatch(part) is None for part in parts[1:]
    ):
        raise ValueError("invalid child run id")
    expected_parent = ":".join(parts[:-1]) if len(parts) > 2 else None
    if parent_run_id != expected_parent:
        raise ValueError("invalid child run parent")


class ChildRunStoreMixin:
    def append_child_run_started(
        self: ConversationStore,
        run_id: str,
        parent_run_id: str | None,
        name: str | None,
        task: str,
        started_at: str,
    ) -> ConversationEntry:
        data: dict[str, Any] = {
            "run_id": run_id,
            "parent_run_id": parent_run_id,
            "task": _bounded_utf8(task, _MAX_STORED_TEXT_BYTES),
            "started_at": started_at,
        }
        if name is not None:
            data["name"] = _bounded_utf8(name, _MAX_STORED_TEXT_BYTES)
        return self._append_row("child_run_started", data)

    async def append_child_run_started_async(
        self: ConversationStore,
        run_id: str,
        parent_run_id: str | None,
        name: str | None,
        task: str,
        started_at: str,
        *,
        on_persisted: Callable[[], None] | None = None,
    ) -> ConversationEntry:
        """Append the spawn row off-loop and publish it before returning."""
        return await self._to_thread_durable(
            self.append_child_run_started,
            run_id,
            parent_run_id,
            name,
            task,
            started_at,
            _on_persisted=on_persisted,
        )
