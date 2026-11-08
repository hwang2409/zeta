"""Cooperative context token accounting and canonical digests."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterable
from math import ceil
from typing import Any

from ..protocol.types import Message, MessageRole

IMAGE_TOKEN_ESTIMATE = 1024
_JSON_STRING_CHUNK = 65_536
_TOKEN_YIELD_INTERVAL = 8
_token_count_state = threading.local()


def cooperative_pause() -> None:
    """Yield the GIL only while work runs in cooperative mode."""

    if getattr(_token_count_state, "cooperative", False):
        time.sleep(0.0001)


def cooperative_call(function: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Run CPU work with bounded GIL holds for event-loop responsiveness."""

    previous = getattr(_token_count_state, "cooperative", False)
    _token_count_state.cooperative = True
    try:
        return function(*args, **kwargs)
    finally:
        _token_count_state.cooperative = previous


def _json_string_length(value: str) -> int:
    if len(value) <= _JSON_STRING_CHUNK:
        return len(json.dumps(value))
    length = 2
    for start in range(0, len(value), _JSON_STRING_CHUNK):
        length += len(json.dumps(value[start : start + _JSON_STRING_CHUNK])) - 2
        cooperative_pause()
    return length


def compact_json_length(value: Any) -> int:
    """Return canonical JSON character length without one long GIL hold."""

    if isinstance(value, str):
        return _json_string_length(value)
    if isinstance(value, list):
        return 2 + max(0, len(value) - 1) + sum(
            compact_json_length(item) for item in value
        )
    if isinstance(value, dict):
        keys = sorted(value)
        return (
            2
            + max(0, len(keys) - 1)
            + sum(
                compact_json_length(key) + 1 + compact_json_length(value[key])
                for key in keys
            )
        )
    return len(json.dumps(value, separators=(",", ":")))


def message_token_count(message: Message) -> int:
    """Estimate text tokens and charge a small fixed amount per image.

    Base64 is transport data, not text. Without image dimensions, use a fixed
    estimate that keeps images near the 4 MiB transport cap usable. Tool results
    count the union of fields sent by Anthropic, Codex, and Ollama.
    """

    calls = getattr(_token_count_state, "calls", 0) + 1
    _token_count_state.calls = calls
    if calls % _TOKEN_YIELD_INTERVAL == 0:
        cooperative_pause()

    value = message.to_dict()
    if message.role is MessageRole.TOOL_RESULT and message.tool_result is not None:
        result = message.tool_result
        value = {
            "role": message.role.value,
            "tool_result": {
                "tool_call_id": result.tool_call_id,
                "content": result.content,
                "is_error": result.is_error,
                **(
                    {"content_blocks": result.content_blocks}
                    if result.content_blocks is not None
                    else {}
                ),
            },
        }
    image_count = 0
    content = value.get("content")
    if isinstance(content, list):
        for index, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == "image":
                content[index] = {
                    key: item for key, item in block.items() if key != "data"
                }
                image_count += 1
    tool_result = value.get("tool_result")
    if isinstance(tool_result, dict):
        blocks = tool_result.get("content_blocks")
        if isinstance(blocks, list):
            tool_result["content_blocks"] = [
                {key: item for key, item in block.items() if key != "data"}
                if isinstance(block, dict) and block.get("type") == "image"
                else block
                for block in blocks
            ]
            image_count += sum(
                isinstance(block, dict) and block.get("type") == "image"
                for block in blocks
            )
    return max(
        1,
        ceil(compact_json_length(value) / 4)
        + image_count * IMAGE_TOKEN_ESTIMATE,
    )


class MessageTokenCounts:
    """Reuse token counts of messages decoded from unchanged persisted entries.

    A count is reused only for the exact message object that `track` bound to
    its source entry, and only for the same counter. Resident store entries are
    append-only, so changed content is a new entry object and misses.
    """

    def __init__(self) -> None:
        self._sources: dict[int, tuple[Message, object]] = {}
        self._counts: dict[int, tuple[object, Callable[[Message], int], int]] = {}

    def track(self, pairs: Iterable[tuple[object, Message]]) -> None:
        """Bind decoded messages to their sources; keep counts for these only."""

        sources = {id(message): (message, source) for source, message in pairs}
        counts = self._counts
        self._counts = {
            key: counts[key]
            for _message, source in sources.values()
            if (key := id(source)) in counts and counts[key][0] is source
        }
        self._sources = sources

    def count(self, message: Message, counter: Callable[[Message], int]) -> int:
        bound = self._sources.get(id(message))
        if bound is None or bound[0] is not message:
            return counter(message)
        source = bound[1]
        cached = self._counts.get(id(source))
        if cached is not None and cached[0] is source and cached[1] is counter:
            return cached[2]
        tokens = counter(message)
        self._counts[id(source)] = (source, counter, tokens)
        return tokens
