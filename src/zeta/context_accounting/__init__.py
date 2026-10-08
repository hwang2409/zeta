"""Cooperative context token accounting and canonical digests."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable, Iterator, Sequence
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


def compact_json_chunks(value: Any) -> Iterator[str]:
    """Yield canonical JSON in chunks that bound worker GIL holds."""

    if isinstance(value, str):
        if len(value) <= _JSON_STRING_CHUNK:
            yield json.dumps(value)
            return
        yield '"'
        for start in range(0, len(value), _JSON_STRING_CHUNK):
            yield json.dumps(value[start : start + _JSON_STRING_CHUNK])[1:-1]
            cooperative_pause()
        yield '"'
        return
    if isinstance(value, list):
        yield "["
        for index, item in enumerate(value):
            if index:
                yield ","
            yield from compact_json_chunks(item)
        yield "]"
        return
    if isinstance(value, dict):
        yield "{"
        for index, key in enumerate(sorted(value)):
            if index:
                yield ","
            yield from compact_json_chunks(key)
            yield ":"
            yield from compact_json_chunks(value[key])
        yield "}"
        return
    yield json.dumps(value, separators=(",", ":"))


def context_digest(messages: Sequence[Message]) -> str:
    """Return the stable digest for an assembled context."""

    digest = hashlib.sha256()
    values = [message.to_dict() for message in messages]
    for chunk in compact_json_chunks(values):
        digest.update(chunk.encode())
    return digest.hexdigest()
