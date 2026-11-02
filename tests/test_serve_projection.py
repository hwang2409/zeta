from dataclasses import dataclass

from zeta.protocol.types import (
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    ToolCall,
    ToolResult,
)
from zeta.server.ergonomics import (
    assistant_message,
    tool_end,
    tool_output,
    tool_start,
    transcript_row,
)
from zeta.server.protocol import FrameCodec


def _params(event: str, fields: dict[str, object]) -> bytes:
    return FrameCodec().notification(event, "session", **fields)


def test_live_projection_golden_params_are_byte_exact() -> None:
    call = ToolCall("call", "bash", {"command": "printf 'x'"})
    cases = [
        ("assistant_message", assistant_message(StreamEvent(StreamEventType.MESSAGE_END, message=Message(MessageRole.ASSISTANT, []))), b'{"jsonrpc":"2.0","method":"event","params":{"event":"assistant_message","session_id":"session","message":{"role":"assistant","content":[]}}}\n'),
        ("tool_start", tool_start(StreamEvent(StreamEventType.TOOL_EXECUTION_START, tool_call=call, data={"x": None})), b'{"jsonrpc":"2.0","method":"event","params":{"event":"tool_start","session_id":"session","tool_call":{"id":"call","name":"bash","arguments":{"command":"printf \'x\'"}},"data":{"x":null}}}\n'),
        ("tool_output", tool_output(StreamEvent(StreamEventType.TOOL_EXECUTION_UPDATE, tool_call=None, delta=None, data={})), b'{"jsonrpc":"2.0","method":"event","params":{"event":"tool_output","session_id":"session","tool_call":null,"output":"","data":{}}}\n'),
        ("tool_end", tool_end(StreamEvent(StreamEventType.TOOL_EXECUTION_END, tool_call=call, tool_result=ToolResult("call", "done", False), data={})), b'{"jsonrpc":"2.0","method":"event","params":{"event":"tool_end","session_id":"session","tool_call":{"id":"call","name":"bash","arguments":{"command":"printf \'x\'"}},"tool_result":{"tool_call_id":"call","content":"done","is_error":false},"data":{}}}\n'),
    ]
    for name, fields, expected in cases:
        assert _params(name, fields) == expected


@dataclass
class _Entry:
    seq: int
    id: str
    parent_id: str | None
    lane: str
    type: str
    data: dict[str, object]


def test_transcript_row_projection_redacts_private_content_and_internal_rows() -> None:
    entry = _Entry(4, "e4", "e3", "main", "message", {"message": {"role": "assistant", "content": [{"type": "tool_use", "tool_call": {"id": "c", "name": "x", "arguments": {"secret": "no"}}}, {"type": "thinking", "body": "private"}]}})
    row = transcript_row(entry)
    assert row["display"]["content"][0]["tool_call"]["arguments"] == {}
    assert transcript_row(_Entry(5, "e5", "e4", "main", "checkpoint", {})) is None


def test_transcript_row_projection_uses_complete_structural_fallback_for_oversized_row() -> None:
    entry = _Entry(4, "e4", None, "main", "message", {"message": {"role": "user", "content": [{"type": "text", "text": "x" * 10000}]}})
    row = transcript_row(entry)
    assert row["display"] is None
    assert row["truncated"] is True
    assert row["original_bytes"]
    assert row["recall_history"]


def test_transcript_row_projection_outline_has_no_display() -> None:
    entry = _Entry(4, "e4", None, "main", "message", {"message": {"role": "user", "content": []}})
    assert transcript_row(entry, "outline") == {"seq": 4, "entry_id": "e4", "parent_entry_id": None, "lane": "main", "type": "message", "at": None}
