"""Bounded root-session history reads for the terminal UI."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any, Literal

from ...core.store._rows import MessageFacts
from ...protocol.types import (
    MESSAGE_ORIGIN_METADATA,
    Message,
    MessageOrigin,
    ToolUseContent,
)
from ..user import displayed_user_text
from .paged_history import (
    HistoryCursor,
    TUIHistoryPage,
    TUIHistoryUnit,
    TUIPaintSegment,
)


@dataclass(frozen=True, slots=True)
class TUIReplayEntry:
    seq: int
    id: str
    type: str
    data: dict[str, Any]
    message: Message | None = None


def _project_entry(entry: Any) -> TUIReplayEntry:
    message = None
    if entry.type == "message":
        parsed = Message.from_dict(entry.data["message"])
        content = [
            replace(
                block,
                tool_call=replace(
                    block.tool_call,
                    arguments=copy.deepcopy(block.tool_call.arguments),
                ),
            )
            if isinstance(block, ToolUseContent)
            else block
            for block in parsed.content
        ]
        tool_result = parsed.tool_result
        if tool_result is not None:
            tool_result = replace(
                tool_result,
                structured_content=copy.deepcopy(tool_result.structured_content),
            )
        message = replace(
            parsed,
            content=content,
            tool_result=tool_result,
            metadata=copy.deepcopy(parsed.metadata),
        )
        data: dict[str, Any] = {}
    elif entry.type in {
        "checkpoint", "fork", "compaction", "notification", "notification_ack",
        "notification_tui_presented", "turn_end", "warning",
    }:
        data = copy.deepcopy(entry.data)
    else:
        data = {}
    return TUIReplayEntry(entry.seq, entry.id, entry.type, data, message)


class RootTUIHistorySource:
    """Adapt the disposable root index to bounded, off-loop TUI reads."""

    prompt_limit = 128
    prompt_bytes = 256 * 1024

    def __init__(self, store: Any) -> None:
        self._store = store

    def _page_sync(
        self,
        cursor: HistoryCursor | None,
        direction: Literal["older", "newer"],
    ) -> TUIHistoryPage:
        if direction == "newer":
            offset = int(cursor.value) if cursor is not None else 0
            request = {"offset": offset}
        else:
            before = int(cursor.value) if cursor is not None else None
            request = {"backward": True, "before": before}
        indexed = self._store.tui_history_index_page(request)
        if indexed is None:
            return TUIHistoryPage(())
        entries = tuple(entry for group in indexed.groups for entry in group)
        entries = tuple(
            entry
            for entry in entries
            if entry.type != "notification"
            or (
                self._store.is_agent_notification_active(entry.id)
                and not self._store.is_agent_notification_presented_to_tui(entry.id)
            )
        )
        seen_calls: set[str] = set()
        projected: list[tuple[str, Any]] = []
        for entry in entries:
            if entry.type == "message":
                message = Message.from_dict(entry.data["message"])
                result = message.tool_result
                if result is not None and result.tool_call_id not in seen_calls:
                    dependency = self._store.tui_tool_call_dependency(
                        result.tool_call_id, entry.seq
                    )
                    if dependency is not None:
                        projected.append(
                            (
                                f"{dependency.id}:dependency:{entry.id}",
                                dependency,
                            )
                        )
                        seen_calls.add(result.tool_call_id)
                for block in message.content:
                    if isinstance(block, ToolUseContent):
                        seen_calls.add(block.tool_call.id)
            projected.append((entry.id, entry))
        segments: list[TUIPaintSegment] = []
        current: list[TUIHistoryUnit] = []
        current_bytes = 0
        for entry_id, entry in projected:
            unit = TUIHistoryUnit(entry_id, _project_entry(entry))
            size = len(repr(entry.data).encode())
            if current and (len(current) >= 128 or current_bytes + size > 1024 * 1024):
                segments.append(TUIPaintSegment(tuple(current), decoded_bytes=current_bytes))
                current = []
                current_bytes = 0
            current.append(unit)
            current_bytes += size
        if current:
            segments.append(TUIPaintSegment(tuple(current), decoded_bytes=current_bytes))
        stop = indexed.start + len(indexed.groups)
        return TUIHistoryPage(
            tuple(segments),
            older=HistoryCursor(str(indexed.start)) if indexed.start > 0 else None,
            newer=(
                HistoryCursor(str(stop))
                if stop < indexed.total
                else None
            ),
            endpoint_id=indexed.endpoint,
            repair_generation=indexed.repair_generation,
        )

    async def page(
        self,
        cursor: HistoryCursor | None = None,
        *,
        direction: Literal["older", "newer"] = "older",
    ) -> TUIHistoryPage:
        return await self._store._to_thread_store(
            lambda: self._page_sync(cursor, direction)
        )

    def _entry_page_sync(self, entry_id: str) -> TUIHistoryPage:
        refs = self._store.branch_refs()
        target = next((ref for ref in refs if ref.id == entry_id), None)
        if target is None:
            return TUIHistoryPage(())
        position = next(index for index, ref in enumerate(refs) if ref.id == entry_id)
        return self._page_sync(HistoryCursor(str(position + 1)), "older")

    async def reveal(self, entry_id: str) -> TUIHistoryPage:
        return await self._store._to_thread_store(
            lambda: self._entry_page_sync(entry_id)
        )

    def _prompt_page_sync(
        self,
        cursor: HistoryCursor | None,
        direction: Literal["older", "newer"],
    ) -> tuple[tuple[tuple[str, str], ...], HistoryCursor | None, HistoryCursor | None]:
        refs = self._store.branch_refs()
        users = [
            ref for ref in refs
            if ref.type == "message"
            and isinstance(ref.facts, MessageFacts)
            and ref.facts.is_user
        ]
        boundary = len(users) if cursor is None else int(cursor.value)
        if direction == "older":
            start = max(0, boundary - self.prompt_limit)
            selected = users[start:boundary]
        else:
            start = boundary
            selected = users[start:start + self.prompt_limit]
        rows: list[tuple[str, str]] = []
        size = 0
        scanned = 0
        scan = reversed(selected) if direction == "older" else iter(selected)
        for ref in scan:
            message = Message.from_dict(self._store.entry(ref).data["message"])
            if message.metadata.get(MESSAGE_ORIGIN_METADATA) != MessageOrigin.USER:
                scanned += 1
                continue
            text = displayed_user_text(message)
            encoded = len(text.encode())
            if rows and size + encoded > self.prompt_bytes:
                break
            rows.append((ref.id, text))
            size += encoded
            scanned += 1
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

    async def prompt_page(
        self,
        cursor: HistoryCursor | None = None,
        *,
        direction: Literal["older", "newer"] = "older",
    ) -> tuple[tuple[tuple[str, str], ...], HistoryCursor | None, HistoryCursor | None]:
        return await self._store._to_thread_store(
            lambda: self._prompt_page_sync(cursor, direction)
        )

    def _visible_users(self) -> list[Any]:
        return [
            ref for ref in self._store.branch_refs()
            if ref.type == "message"
            and isinstance(ref.facts, MessageFacts)
            and ref.facts.is_user_forkpoint
        ]

    async def adjacent_visible_user(
        self, entry_id: str, *, direction: Literal["older", "newer"]
    ) -> str | None:
        def read() -> str | None:
            users = self._visible_users()
            positions = {ref.id: index for index, ref in enumerate(users)}
            position = positions.get(entry_id)
            if position is None:
                return None
            target = position + (-1 if direction == "older" else 1)
            return users[target].id if 0 <= target < len(users) else None

        return await self._store._to_thread_store(read)

    async def search(self, query: str, *, limit: int = 200) -> tuple[str, ...]:
        def read() -> tuple[str, ...]:
            needle = query.casefold()
            matches: list[str] = []
            for ref in reversed(self._store.branch_refs()):
                entry = self._store.entry(ref)
                if needle in repr(entry.data).casefold():
                    matches.append(entry.id)
                    if len(matches) >= limit:
                        break
            return tuple(matches)

        return await self._store._to_thread_store(read)
