"""Self-contained OptChat-style tree and incremental view experiment.

This module has no production wiring. Its interface accepts decoded Zeta messages,
keeps an in-memory summary tree, renders a bounded summary view, and supports
OptChat's ``zoom(id, n)`` addressing.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from zeta.media.images import image_description
from zeta.protocol.types import (
    ImageContent,
    Message,
    MessageRole,
    RedactedThinkingContent,
    TextContent,
    ThinkingContent,
    ToolUseContent,
)

NODE_BYTES = 512
TOOL_RESULT_CAP_CHARS = 30_000
OptKind = Literal["user", "talk", "tool", "echo", "note"]
SummaryFn = Callable[[str, str], str]


@dataclass(frozen=True, slots=True)
class LogMessage:
    """One OptChat log item derived from a Zeta transcript message."""

    id: int
    source_seq: int
    kind: OptKind
    text: str

    @property
    def line(self) -> str:
        return f"{self.kind}: {self.text}"


@dataclass(frozen=True, slots=True)
class Node:
    """A binary summary-tree node; ``index`` is its coordinate at ``level``."""

    level: int
    index: int
    text: str

    @property
    def start(self) -> int:
        return self.index << self.level

    @property
    def count(self) -> int:
        return 1 << self.level

    @property
    def address(self) -> tuple[int, int]:
        return self.start, self.count


@dataclass(frozen=True, slots=True)
class CompactorStats:
    model_calls: int
    free_nodes: int
    nodes: int
    input_bytes: int
    output_bytes: int
    oversize_nodes: int


@dataclass(frozen=True, slots=True)
class ZoomResult:
    start: int
    count: int
    lines: tuple[str, ...]
    verbatim: bool


class DeterministicCompactor:
    """Extractive stand-in used to measure tree shape without model calls."""

    def __init__(self, node_bytes: int = NODE_BYTES) -> None:
        self.node_bytes = node_bytes

    def __call__(self, _context: str, source: str) -> str:
        flat = " ".join(source.split())
        encoded = flat.encode("utf-8")
        if len(encoded) <= self.node_bytes:
            return flat
        marker = " … "
        marker_size = len(marker.encode())
        left_budget = (self.node_bytes - marker_size) * 2 // 3
        right_budget = self.node_bytes - marker_size - left_budget
        left = _utf8_prefix(encoded, left_budget)
        right = _utf8_suffix(encoded, right_budget)
        return f"{left}{marker}{right}"


class OptChatView:
    """Build the exact binary tree and incremental never-split view from §3/§5.2.

    Nodes are built eagerly in dependency order because offline compaction is
    synchronous. The view still changes only by appending a level-0 part and
    merging the most-due eligible adjacent pair. It never splits a merged part.
    """

    def __init__(
        self,
        *,
        view_bytes: int = 128_000,
        summarize: SummaryFn | None = None,
        recent_verbatim: int = 0,
    ) -> None:
        if view_bytes <= 0:
            raise ValueError("view_bytes must be positive")
        if recent_verbatim < 0:
            raise ValueError("recent_verbatim must not be negative")
        self.view_bytes = view_bytes
        self.recent_verbatim = recent_verbatim
        self._summarize = summarize or DeterministicCompactor()
        self.messages: list[LogMessage] = []
        self.nodes: dict[tuple[int, int], Node] = {}
        self.parts: list[tuple[int, int]] = []
        self._model_calls = 0
        self._free_nodes = 0
        self._input_bytes = 0
        self._output_bytes = 0
        self._oversize_nodes = 0

    def append(self, kind: OptKind, text: str, *, source_seq: int) -> int:
        message_id = len(self.messages)
        message = LogMessage(message_id, source_seq, kind, text)
        self.messages.append(message)
        self._build(0, message_id, message.line)
        self.parts.append((0, message_id))
        self._build_ancestors(message_id)
        self._fit()
        return message_id

    def extend_zeta(self, records: Iterable[tuple[int, Message]]) -> None:
        for seq, message in records:
            for kind, text in map_zeta_message(message):
                self.append(kind, text, source_seq=seq)

    def render(self) -> bytes:
        lines = ["<chat>"]
        for key in self.parts:
            node = self.nodes[key]
            lines.append(f"{node.start}+{node.count}|{_flatten(node.text)}")
        lines.append("</chat>")
        if self.recent_verbatim:
            lines.append("<recent-verbatim-variant>")
            for message in self.messages[-self.recent_verbatim :]:
                lines.append(f"{message.id}+1|{_flatten(message.line)}")
            lines.append("</recent-verbatim-variant>")
        return "\n".join(lines).encode("utf-8")

    def zoom(self, start: int, count: int) -> ZoomResult:
        if count < 1 or count & (count - 1):
            raise ValueError("count must be a positive power of two")
        if start < 0 or start % count:
            raise ValueError("start must be non-negative and aligned to count")
        level = count.bit_length() - 1
        node = self.nodes.get((level, start // count))
        if node is None:
            raise KeyError(f"unknown node {start}+{count}")
        if count == 1:
            message = self.messages[start]
            return ZoomResult(start, count, (message.line,), True)
        child_level = level - 1
        left = self.nodes[(child_level, node.index * 2)]
        right = self.nodes[(child_level, node.index * 2 + 1)]
        return ZoomResult(
            start,
            count,
            (
                f"{left.start}+{left.count}|{_flatten(left.text)}",
                f"{right.start}+{right.count}|{_flatten(right.text)}",
            ),
            False,
        )

    def covering_node(self, message_id: int) -> Node:
        if not 0 <= message_id < len(self.messages):
            raise IndexError(message_id)
        for key in self.parts:
            node = self.nodes[key]
            if node.start <= message_id < node.start + node.count:
                return node
        raise RuntimeError("view does not tile the log")

    def reference_depth(self, message_id: int, needles: Sequence[str]) -> int | None:
        """Return zooms needed before an exact reference appears in a line.

        Zero means the visible view line contains a signature. One means one
        zoom reveals it. ``None`` means only the verbatim leaf has it or the
        deterministic summary removed it at every level.
        """

        folded = tuple(value.casefold() for value in needles if value)
        if not folded:
            return None
        node = self.covering_node(message_id)
        depth = 0
        while True:
            text = node.text.casefold()
            if any(value in text for value in folded):
                return depth
            if node.level == 0:
                message_text = self.messages[message_id].line.casefold()
                return depth if any(value in message_text for value in folded) else None
            depth += 1
            child_level = node.level - 1
            child_count = 1 << child_level
            child_index = message_id // child_count
            node = self.nodes[(child_level, child_index)]

    @property
    def stats(self) -> CompactorStats:
        return CompactorStats(
            model_calls=self._model_calls,
            free_nodes=self._free_nodes,
            nodes=len(self.nodes),
            input_bytes=self._input_bytes,
            output_bytes=self._output_bytes,
            oversize_nodes=self._oversize_nodes,
        )

    def _build(self, level: int, index: int, source: str) -> None:
        key = (level, index)
        if key in self.nodes:
            return
        if len(source.encode("utf-8")) <= NODE_BYTES:
            text = source
            self._free_nodes += 1
        else:
            context = "\n".join(self.nodes[key].text for key in self.parts)
            self._input_bytes += len(context.encode()) + len(source.encode())
            text = self._summarize(context, source).strip()
            self._model_calls += 1
        size = len(text.encode("utf-8"))
        self._output_bytes += size
        self._oversize_nodes += size > NODE_BYTES
        self.nodes[key] = Node(level, index, text)

    def _build_ancestors(self, message_id: int) -> None:
        level = 0
        index = message_id
        while index % 2 == 1:
            left_key = (level, index - 1)
            right_key = (level, index)
            left = self.nodes[left_key]
            right = self.nodes[right_key]
            parent_source = f"{_flatten(left.text)}\n{_flatten(right.text)}"
            index //= 2
            level += 1
            self._build(level, index, parent_source)

    def _fit(self) -> None:
        total_messages = len(self.messages)
        size = sum(len(self.nodes[key].text.encode("utf-8")) for key in self.parts)
        while size > self.view_bytes:
            best_position: int | None = None
            best_due = -1.0
            for position, (left_key, right_key) in enumerate(
                zip(self.parts, self.parts[1:], strict=False)
            ):
                left_level, left_index = left_key
                right_level, right_index = right_key
                parent_key = (left_level + 1, left_index // 2)
                if (
                    left_level != right_level
                    or left_index % 2
                    or right_index != left_index + 1
                    or parent_key not in self.nodes
                ):
                    continue
                start = left_index << left_level
                due = (total_messages - start) / (1 << (left_level + 2))
                if due > best_due:
                    best_due = due
                    best_position = position
            if best_position is None:
                return
            left_key = self.parts[best_position]
            right_key = self.parts[best_position + 1]
            parent_key = (left_key[0] + 1, left_key[1] // 2)
            size -= len(self.nodes[left_key].text.encode())
            size -= len(self.nodes[right_key].text.encode())
            size += len(self.nodes[parent_key].text.encode())
            self.parts[best_position : best_position + 2] = [parent_key]


def map_zeta_message(message: Message) -> list[tuple[OptKind, str]]:
    """Map one provider-neutral Zeta message into OptChat log kinds."""

    if message.tool_result is not None:
        return [("echo", _cap_tool_result(message.tool_result.content))]
    if message.role is MessageRole.COMPACTION:
        default_kind: OptKind = "note"
    elif message.role in {MessageRole.USER, MessageRole.APPROVAL}:
        default_kind = "user"
    elif message.role is MessageRole.ASSISTANT:
        default_kind = (
            "echo" if message.metadata.get("response_state") == "synthetic" else "talk"
        )
    elif message.role is MessageRole.SYSTEM:
        default_kind = (
            "echo" if message.metadata.get("zeta_event") == "agent_notifications" else "note"
        )
    else:
        default_kind = "note"

    mapped: list[tuple[OptKind, str]] = []
    for block in message.content:
        if isinstance(block, TextContent):
            mapped.append((default_kind, block.text))
        elif isinstance(block, ToolUseContent):
            call = block.tool_call
            mapped.append(
                (
                    "tool",
                    f"{call.name} {json.dumps(call.arguments, sort_keys=True, separators=(',', ':'))}",
                )
            )
        elif isinstance(block, ImageContent):
            mapped.append((default_kind, image_description(block.to_dict(), detailed=True)))
        elif isinstance(block, (ThinkingContent, RedactedThinkingContent)):
            continue
    if not mapped:
        mapped.append((default_kind, "[empty message]"))
    return mapped


def _cap_tool_result(text: str) -> str:
    if len(text) <= TOOL_RESULT_CAP_CHARS:
        return text
    half = TOOL_RESULT_CAP_CHARS // 2
    return f"{text[:half]}\n[... capped ...]\n{text[-half:]}"


def _flatten(text: str) -> str:
    return " ".join(text.splitlines())


def _utf8_prefix(value: bytes, size: int) -> str:
    return value[:size].decode("utf-8", errors="ignore")


def _utf8_suffix(value: bytes, size: int) -> str:
    return value[-size:].decode("utf-8", errors="ignore")
