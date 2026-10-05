from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.context.replay import active_message_records, replay_records
from zeta.core.context import _message_token_count
from zeta.protocol.types import (
    Message,
    MessageRole,
    TextContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
)


def _message(role: MessageRole, text: str) -> Message:
    return Message(role, [TextContent(text)])


def _write_log(path: Path, messages: list[Message]) -> None:
    rows = [{"type": "header", "data": {"schema": 1}}]
    parent = None
    for seq, message in enumerate(messages, 1):
        entry_id = f"entry-{seq}"
        rows.append(
            {
                "seq": seq,
                "id": entry_id,
                "parent_id": parent,
                "lane": "main",
                "type": "message",
                "data": {"message": message.to_dict()},
            }
        )
        parent = entry_id
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_active_message_records_is_read_only_and_ignores_torn_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "conversation.jsonl"
    messages = [_message(MessageRole.USER, "hello")]
    _write_log(path, messages)
    with path.open("ab") as handle:
        handle.write(b'{"type":"message"')
    before = path.stat()

    real_open = open

    def read_only_open(name: object, mode: str = "r", *args: object, **kwargs: object):
        assert mode == "rb"
        return real_open(name, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", read_only_open)
    records = active_message_records(path)

    after = path.stat()
    assert records == [(1, messages[0])]
    assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns)


def test_replay_calculates_request_and_prefix_metrics() -> None:
    records = [
        (1, _message(MessageRole.USER, "first")),
        (2, _message(MessageRole.ASSISTANT, "answer")),
        (3, _message(MessageRole.USER, "second")),
    ]
    result = replay_records(records, 100_000)
    first = _message_token_count(records[0][1])
    second_request = sum(_message_token_count(message) for _, message in records)

    assert result.requests == 2
    assert result.evictions == 0
    assert result.total_input_tokens == first + second_request
    assert result.mean_tokens == (first + second_request) / 2
    assert result.p50_tokens == (first + second_request) / 2
    assert result.max_tokens == second_request
    assert result.unchanged_prefix_fraction == 1.0
    assert result.cached_token_share == first / result.total_input_tokens


def test_replay_fits_large_tool_results_with_production_assembler_logic() -> None:
    call = ToolCall("call-large", "bash", {"command": "generate output"})
    records = [
        (1, _message(MessageRole.USER, "start")),
        (2, Message(MessageRole.ASSISTANT, [ToolUseContent(call)])),
        (
            3,
            Message(
                MessageRole.USER,
                [],
                tool_result=ToolResult("call-large", "large output " * 100_000),
            ),
        ),
    ]

    result = replay_records(records, 10_000)

    assert result.requests == 2
    assert result.max_tokens <= 10_000


def test_replay_counts_later_reference_to_evicted_result() -> None:
    call = ToolCall("call-1", "read", {"path": "/tmp/needle.txt"})
    records = [
        (1, _message(MessageRole.USER, "inspect the incident")),
        (2, Message(MessageRole.ASSISTANT, [ToolUseContent(call)])),
        (
            3,
            Message(
                MessageRole.USER,
                [],
                tool_result=ToolResult(
                    "call-1",
                    ("routine output " * 3000) + "/tmp/needle.txt Incident-1234",
                ),
            ),
        ),
        (4, _message(MessageRole.USER, "continue")),
        (5, _message(MessageRole.ASSISTANT, "done")),
        (6, _message(MessageRole.USER, "re-open /tmp/needle.txt for Incident-1234")),
    ]

    result = replay_records(records, 2_000)

    assert result.evictions >= 1
    assert result.references.evicted_items >= 1
    assert result.references.later_referenced_items >= 1
    assert result.references.path_matches >= 1
    assert result.references.identifier_matches >= 1
