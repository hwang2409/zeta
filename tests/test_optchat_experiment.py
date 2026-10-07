from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.optchat.eviction_replay import active_message_records
from evals.optchat.real_compactor import _scale_line
from evals.optchat.replay import replay_optchat
from evals.optchat.strategy import (
    DeterministicCompactor,
    OptChatView,
    map_zeta_message,
)
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


def test_active_reader_is_binary_read_only_and_ignores_torn_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "conversation.jsonl"
    message = _message(MessageRole.USER, "hello")
    _write_log(path, [message])
    with path.open("ab") as handle:
        handle.write(b'{"type":"message"')
    before = path.stat()
    real_open = open

    def read_only_open(name: object, mode: str = "r", *args: object, **kwargs: object):
        assert mode == "rb"
        return real_open(name, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", read_only_open)
    assert active_message_records(path) == [(1, message)]
    after = path.stat()
    assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns)


def test_tree_uses_binary_addresses_and_zoom_reaches_verbatim_message() -> None:
    view = OptChatView(view_bytes=20)
    for seq in range(1, 5):
        view.append("user", f"message-{seq}", source_seq=seq)

    assert [(view.nodes[key].start, view.nodes[key].count) for key in view.parts] == [(0, 4)]
    assert view.zoom(0, 4).lines[0].startswith("0+2|")
    assert view.zoom(0, 2).lines[0].startswith("0+1|")
    leaf = view.zoom(0, 1)
    assert leaf.verbatim
    assert leaf.lines == ("user: message-1",)


def test_reference_depth_counts_final_zoom_from_summary_leaf_to_verbatim() -> None:
    view = OptChatView(view_bytes=128_000, summarize=lambda _context, _source: "omitted")
    message_id = view.append("user", "Needle-1234 " * 100, source_seq=1)

    assert view.reference_depth(message_id, ("Needle-1234",)) == 1


def test_fit_merges_most_due_pair_and_never_splits() -> None:
    view = OptChatView(view_bytes=35)
    for seq in range(1, 5):
        view.append("user", f"m{seq:02d}", source_seq=seq)
    before = tuple(view.parts)
    view.append("user", "m05", source_seq=5)

    assert all(part in view.parts or part[0] < max(level for level, _ in view.parts) for part in before)
    assert sum(view.nodes[key].count for key in view.parts) == 5
    assert [view.nodes[key].start for key in view.parts] == sorted(
        view.nodes[key].start for key in view.parts
    )


def test_deterministic_compactor_is_utf8_safe_and_bounded() -> None:
    result = DeterministicCompactor()("", "user: " + "é" * 1000)

    assert len(result.encode("utf-8")) <= 512
    assert "…" in result
    assert len(_scale_line().encode("utf-8")) == 512


def test_zeta_mapping_covers_text_calls_results_and_synthetic_receipts() -> None:
    call = ToolCall("call-1", "read", {"path": "/tmp/a"})
    assistant = Message(
        MessageRole.ASSISTANT,
        [TextContent("working"), ToolUseContent(call)],
    )
    result = Message(
        MessageRole.TOOL_RESULT,
        [],
        tool_result=ToolResult("call-1", "done"),
    )
    receipt = Message(
        MessageRole.ASSISTANT,
        [TextContent("receipt")],
        metadata={"response_state": "synthetic"},
    )

    assert [kind for kind, _ in map_zeta_message(assistant)] == ["talk", "tool"]
    assert map_zeta_message(result) == [("echo", "done")]
    assert map_zeta_message(receipt) == [("echo", "receipt")]


def test_replay_renders_view_before_new_user_message() -> None:
    records = [
        (1, _message(MessageRole.USER, "first")),
        (2, _message(MessageRole.ASSISTANT, "answer")),
        (3, _message(MessageRole.USER, "second")),
    ]

    requests, stats = replay_optchat(records, view_bytes=128_000)

    assert len(requests) == 2
    assert b"first" not in requests[0].body.split(b"<new-message>")[0]
    assert b"first" in requests[1].body.split(b"<new-message>")[0]
    assert stats["messages"] == 3
