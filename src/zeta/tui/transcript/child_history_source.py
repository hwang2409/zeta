"""Bounded, no-sidecar history reads for one focused child session."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from ...core.store import ConversationStore
from ...core.store._rows import MessageFacts
from ...protocol.types import (
    FAILED_TURN_MARKER,
    MESSAGE_ORIGIN_METADATA,
    Message,
    MessageOrigin,
    MessageRole,
    ToolUseContent,
    is_passive_harness_message,
)
from ..cards.agent_sync import bounded_transcript_message
from ..user import displayed_user_text
from .paged_history import (
    HistoryCursor,
    HistorySearchResult,
    TUIHistoryPage,
    TUIHistoryUnit,
    TUIPaintSegment,
)
from .root_history_source import _project_entry


class ChildLogHistorySource:
    """Adapt a read-only child log to the shared TUI history seam."""

    prompt_limit = 128
    prompt_bytes = 256 * 1024
    row_limit = 128
    byte_limit = 1024 * 1024

    def __init__(self, path: Path) -> None:
        self.path = path
        self.replay_provider: str | None = None
        self._store: ConversationStore | None = None
        self._closed = False
        self._operation_lock = asyncio.Lock()
        self.pending_prompts: tuple[str, ...] = ()

    async def close(self) -> None:
        self._closed = True
        async with self._operation_lock:
            pass

    def _run_sync(self, operation: Any, *args: Any) -> Any:
        store = ConversationStore(
            self.path.parent,
            session_id=self.path.name,
            _read_only=True,
            _observer_snapshot=True,
            _must_exist=True,
        )
        self._store = store
        try:
            return operation(*args)
        finally:
            self._store = None
            store.close()

    def _refresh_refs(self) -> tuple[Any, ...]:
        if self._closed:
            return ()
        self._store.refresh()
        refs = self._store.branch_refs()
        pending_refs = [
            ref
            for ref in refs
            if ref.type in {"pending_prompt", "pending_prompt_ack"}
        ]
        pending: dict[str, str] = {}
        for entry in self._store.stream_rows(pending_refs):
            if entry.type == "pending_prompt":
                pending[entry.id] = str(entry.data.get("text", ""))
            else:
                pending.pop(str(entry.data.get("prompt_id", "")), None)
        self.pending_prompts = tuple(pending.values())
        return refs

    def _dependency(self, refs: tuple[Any, ...], position: int, message: Message) -> Any | None:
        if message.role is MessageRole.ASSISTANT and message.metadata.get(FAILED_TURN_MARKER):
            for ref in reversed(refs[:position]):
                if (
                    ref.type == "message"
                    and isinstance(ref.facts, MessageFacts)
                    and ref.facts.is_user_forkpoint
                ):
                    return self._store.entry(ref)
        result = message.tool_result
        if result is None:
            return None
        for ref in reversed(refs[:position]):
            if ref.type != "message":
                continue
            entry = self._store.entry(ref)
            candidate = Message.from_dict(
                bounded_transcript_message(entry.data["message"])
            )
            if any(
                isinstance(block, ToolUseContent)
                and block.tool_call.id == result.tool_call_id
                for block in candidate.content
            ):
                return entry
        return None

    def _bundle(
        self, refs: tuple[Any, ...], position: int
    ) -> tuple[list[TUIHistoryUnit], int]:
        ref = refs[position]
        entry = self._store.entry(ref)
        projected: list[tuple[str, Any]] = []
        if entry.type == "message":
            raw_message = entry.data["message"]
            if is_passive_harness_message(raw_message):
                return [], 0
            bounded = bounded_transcript_message(raw_message)
            entry = replace(entry, data={**entry.data, "message": bounded})
            message = Message.from_dict(bounded)
            dependency = self._dependency(refs, position, message)
            if dependency is not None:
                projected.append((f"{dependency.id}:dependency:{entry.id}", dependency))
        projected.append((entry.id, entry))
        units: list[TUIHistoryUnit] = []
        decoded_bytes = 0
        for entry_id, item in projected:
            value = _project_entry(item)
            units.append(TUIHistoryUnit(entry_id, value))
            decoded_bytes += len(str(value).encode("utf-8", errors="replace"))
        return units, decoded_bytes

    def _page_from_refs(
        self,
        refs: tuple[Any, ...],
        cursor: HistoryCursor | None,
        direction: Literal["older", "newer"],
    ) -> TUIHistoryPage:
        boundary = len(refs) if cursor is None else int(cursor.value)
        positions = (
            range(boundary - 1, -1, -1)
            if direction == "older"
            else range(boundary, len(refs))
        )
        selected: list[tuple[list[TUIHistoryUnit], int]] = []
        row_count = decoded_bytes = consumed = 0
        for position in positions:
            units, size = self._bundle(refs, position)
            if selected and (
                row_count + len(units) > self.row_limit
                or decoded_bytes + size > self.byte_limit
            ):
                break
            selected.append((units, size))
            row_count += len(units)
            decoded_bytes += size
            consumed += 1
        if direction == "older":
            selected.reverse()
            start = boundary - consumed
            stop = boundary
        else:
            start = boundary
            stop = boundary + consumed
        units = tuple(unit for bundle, _size in selected for unit in bundle)
        segments = (TUIPaintSegment(units, decoded_bytes=decoded_bytes),) if units else ()
        endpoint = refs[-1].id if refs else None
        return TUIHistoryPage(
            segments,
            older=HistoryCursor(str(start)) if start > 0 else None,
            newer=HistoryCursor(str(stop)) if stop < len(refs) else None,
            endpoint_id=endpoint,
        )

    def _page_sync(
        self,
        cursor: HistoryCursor | None,
        direction: Literal["older", "newer"],
    ) -> TUIHistoryPage:
        return self._page_from_refs(self._refresh_refs(), cursor, direction)

    async def page(
        self,
        cursor: HistoryCursor | None = None,
        *,
        direction: Literal["older", "newer"] = "older",
    ) -> TUIHistoryPage:
        async with self._operation_lock:
            return await asyncio.to_thread(
                self._run_sync, self._page_sync, cursor, direction
            )

    def _entry_page_sync(self, entry_id: str) -> TUIHistoryPage:
        refs = self._refresh_refs()
        position = next(
            (index for index, ref in enumerate(refs) if ref.id == entry_id), None
        )
        if position is None:
            return TUIHistoryPage(())
        return self._page_from_refs(
            refs, HistoryCursor(str(position + 1)), "older"
        )

    async def reveal(self, entry_id: str) -> TUIHistoryPage:
        async with self._operation_lock:
            return await asyncio.to_thread(
                self._run_sync, self._entry_page_sync, entry_id
            )

    def _prompt_refs(self, refs: tuple[Any, ...]) -> list[Any]:
        return [
            ref
            for ref in refs
            if ref.type == "message"
            and isinstance(ref.facts, MessageFacts)
            and ref.facts.is_user
        ]

    def _prompt_page_from_refs(
        self,
        users: list[Any],
        cursor: HistoryCursor | None,
        direction: Literal["older", "newer"],
    ) -> tuple[tuple[tuple[str, str], ...], HistoryCursor | None, HistoryCursor | None]:
        boundary = len(users) if cursor is None else int(cursor.value)
        if direction == "older":
            start = max(0, boundary - self.prompt_limit)
            selected = users[start:boundary]
        else:
            start = boundary
            selected = users[start : start + self.prompt_limit]
        rows: list[tuple[str, str]] = []
        size = scanned = 0
        scan = reversed(selected) if direction == "older" else iter(selected)
        for ref in scan:
            raw_message = self._store.entry(ref).data["message"]
            if is_passive_harness_message(raw_message):
                continue
            message = Message.from_dict(bounded_transcript_message(raw_message))
            scanned += 1
            if message.metadata.get(MESSAGE_ORIGIN_METADATA) != MessageOrigin.USER:
                continue
            text = displayed_user_text(message)
            encoded = len(text.encode())
            if rows and size + encoded > self.prompt_bytes:
                scanned -= 1
                break
            rows.append((ref.id, text))
            size += encoded
        if direction == "older":
            rows.reverse()
            start = boundary - scanned
            consumed = boundary
        else:
            consumed = start + scanned
        return (
            tuple(rows),
            HistoryCursor(str(start)) if start > 0 else None,
            HistoryCursor(str(consumed)) if consumed < len(users) else None,
        )

    def _prompt_page_sync(
        self,
        cursor: HistoryCursor | None,
        direction: Literal["older", "newer"],
    ) -> tuple[tuple[tuple[str, str], ...], HistoryCursor | None, HistoryCursor | None]:
        users = self._prompt_refs(self._refresh_refs())
        return self._prompt_page_from_refs(users, cursor, direction)

    async def prompt_page(
        self,
        cursor: HistoryCursor | None = None,
        *,
        direction: Literal["older", "newer"] = "older",
    ) -> tuple[tuple[tuple[str, str], ...], HistoryCursor | None, HistoryCursor | None]:
        async with self._operation_lock:
            return await asyncio.to_thread(
                self._run_sync, self._prompt_page_sync, cursor, direction
            )

    async def prompt_pages(self) -> AsyncIterator[tuple[tuple[str, str], ...]]:
        async with self._operation_lock:
            users = await asyncio.to_thread(
                self._run_sync, lambda: self._prompt_refs(self._refresh_refs())
            )
        cursor: HistoryCursor | None = None
        while True:
            async with self._operation_lock:
                rows, older, _newer = await asyncio.to_thread(
                    self._run_sync,
                    self._prompt_page_from_refs,
                    users,
                    cursor,
                    "older",
                )
            if rows:
                yield rows
            if older is None:
                return
            cursor = older

    def _adjacent_user_sync(
        self, entry_id: str, direction: Literal["older", "newer"]
    ) -> str | None:
        users = [
            ref
            for ref in self._refresh_refs()
            if ref.type == "message"
            and isinstance(ref.facts, MessageFacts)
            and ref.facts.is_user_forkpoint
        ]
        positions = {ref.id: index for index, ref in enumerate(users)}
        position = positions.get(entry_id)
        if position is None:
            return None
        target = position + (-1 if direction == "older" else 1)
        return users[target].id if 0 <= target < len(users) else None

    async def adjacent_visible_user(
        self, entry_id: str, *, direction: Literal["older", "newer"]
    ) -> str | None:
        async with self._operation_lock:
            return await asyncio.to_thread(
                self._run_sync, self._adjacent_user_sync, entry_id, direction
            )

    def _search_sync(
        self, query: str, limit: int, cancel: threading.Event
    ) -> tuple[HistorySearchResult, ...]:
        needle = query.casefold()
        matches: list[HistorySearchResult] = []
        refs = self._refresh_refs()
        for entry in self._store.stream_rows(reversed(refs), batch=256):
            if cancel.is_set():
                break
            raw_message = entry.data.get("message")
            if isinstance(raw_message, dict):
                if is_passive_harness_message(raw_message):
                    continue
                entry = replace(
                    entry,
                    data={
                        **entry.data,
                        "message": bounded_transcript_message(raw_message),
                    },
                )
            if needle in str(_project_entry(entry)).casefold():
                matches.append(HistorySearchResult(entry.id))
                if len(matches) == limit:
                    break
        return tuple(matches)

    async def search(
        self, query: str, *, limit: int = 200
    ) -> tuple[HistorySearchResult, ...]:
        cancel = threading.Event()
        async with self._operation_lock:
            worker = asyncio.create_task(
                asyncio.to_thread(
                    self._run_sync, self._search_sync, query, limit, cancel
                )
            )
            try:
                return await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancel.set()
                raise
