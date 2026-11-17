import json
from types import SimpleNamespace

from zeta.core.retention.computer import ComputerScreenshotRetention
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    ToolCall,
    ToolResult,
)
from zeta.server.event_contract import (
    assistant_message,
    tool_end,
    tool_output,
    tool_start,
)
from zeta.server.handlers.history_projection import _history_entry_rows
from zeta.server.protocol import EventDescriptor, FrameCodec


def _params(event: str, fields: dict[str, object]) -> bytes:
    return FrameCodec().notification(EventDescriptor(event), "session", **fields)


def test_computer_tool_end_and_history_omit_screenshot_bytes(tmp_path) -> None:
    marker = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAHnOcQAAAAABJRU5ErkJggg=="
    result = ToolResult(
        "shot",
        "screenshot 1x1",
        content_blocks=[
            {"type": "image", "data": marker, "mimeType": "image/png", "width": 1, "height": 1, "size": 70}
        ],
        structured_content={"computer": {"has_screenshot": True, "control": "idle"}},
    )
    call = ToolCall("shot", "computer", {"action": "screenshot"})
    live = _params(
        "tool_end",
        tool_end(
            StreamEvent(
                StreamEventType.TOOL_EXECUTION_END,
                tool_call=call,
                tool_result=result,
                data={},
            )
        ),
    )
    assert marker.encode() not in live
    assert b'"width":1,"height":1,"size":70,"data_omitted":true' in live
    assert b'"control":"idle"' in live

    store = ConversationStore(tmp_path)
    retention = ComputerScreenshotRetention()
    try:
        store.append_message(
            Message(
                MessageRole.TOOL_RESULT,
                [],
                tool_result=retention.durable(result),
            )
        )
        entry = store.entry(store.branch_refs()[0])
        rows = _history_entry_rows(entry, {}, fidelity=True)
        encoded = FrameCodec().response("history", {"groups": [rows]})
        assert marker.encode() not in encoded
        assert b'"width":1,"height":1,"size":70,"data_omitted":true' in encoded
        assert b'"control":"idle"' in encoded
    finally:
        store.close()


def test_live_projection_golden_params_are_byte_exact() -> None:
    call = ToolCall("call", "bash", {"command": "printf 'x'"})
    cases = [
        (
            "assistant_message",
            assistant_message(
                StreamEvent(
                    StreamEventType.MESSAGE_END,
                    message=Message(MessageRole.ASSISTANT, []),
                )
            ),
            b'{"jsonrpc":"2.0","method":"event","params":{"event":"assistant_message","session_id":"session","message":{"role":"assistant","content":[]}}}\n',
        ),
        (
            "tool_start",
            tool_start(
                StreamEvent(
                    StreamEventType.TOOL_EXECUTION_START,
                    tool_call=call,
                    data={"x": None},
                )
            ),
            b'{"jsonrpc":"2.0","method":"event","params":{"event":"tool_start","session_id":"session","tool_call":{"id":"call","name":"bash","arguments":{"command":"printf \'x\'"}},"data":{"x":null}}}\n',
        ),
        (
            "tool_output",
            tool_output(
                StreamEvent(
                    StreamEventType.TOOL_EXECUTION_UPDATE,
                    tool_call=None,
                    delta=None,
                    data={},
                )
            ),
            b'{"jsonrpc":"2.0","method":"event","params":{"event":"tool_output","session_id":"session","tool_call":null,"output":"","data":{}}}\n',
        ),
        (
            "tool_end",
            tool_end(
                StreamEvent(
                    StreamEventType.TOOL_EXECUTION_END,
                    tool_call=call,
                    tool_result=ToolResult("call", "done", False),
                    data={},
                )
            ),
            b'{"jsonrpc":"2.0","method":"event","params":{"event":"tool_end","session_id":"session","tool_call":{"id":"call","name":"bash","arguments":{"command":"printf \'x\'"}},"tool_result":{"tool_call_id":"call","content":"done","is_error":false},"data":{}}}\n',
        ),
    ]
    for name, fields, expected in cases:
        assert _params(name, fields) == expected


def _computer_event(control: str, *, bundle: object = "com.example.app", shortcut: object = "ctrl+opt+cmd+esc") -> StreamEvent:
    call = ToolCall("computer-call", "computer", {"action": "type", "text": "secret input"})
    structured = {
        "computer": {
            "action": "type",
            "control": control,
            "kill_switch": shortcut,
            "frontmost_bundle_id": bundle,
            "has_screenshot": True,
            "thumbnail": "raw-base64",
        },
        "unrelated": "private result",
    }
    return StreamEvent(
        StreamEventType.TOOL_EXECUTION_END,
        tool_call=call,
        tool_result=ToolResult(
            "computer-call",
            "private result",
            control != "idle",
            content_blocks=[
                {"type": "text", "text": "private result"},
                {"type": "image", "data": "raw-base64", "mimeType": "image/png"},
            ],
            structured_content=structured,
        ),
        data={"unrelated": "private event"},
    )


def _wire(event: StreamEvent, *, start: bool) -> dict[str, object]:
    descriptor = EventDescriptor.TOOL_START if start else EventDescriptor.TOOL_END
    fields = tool_start(event) if start else tool_end(event)
    frame = FrameCodec().notification(descriptor, "session", **fields)
    assert frame is not None
    return json.loads(frame)["params"]


def test_computer_start_end_frame_projection_is_bounded_and_uses_existing_names() -> None:
    for control in ("active", "idle", "denied", "killed"):
        end_event = _computer_event(control)
        start_event = StreamEvent(
            StreamEventType.TOOL_EXECUTION_START,
            tool_call=end_event.tool_call,
            data={"unrelated": "private event"},
        )
        start = _wire(start_event, start=True)
        end = _wire(end_event, start=False)

        assert start["event"] == "tool_start"
        assert end["event"] == "tool_end"
        assert start["computer"] == {
            "action": "type",
            "control": "active",
            "kill_switch": "ctrl+opt+cmd+esc",
        }
        assert end["computer"] == {
            "action": "type",
            "control": control,
            "kill_switch": "ctrl+opt+cmd+esc",
            "frontmost_bundle_id": "com.example.app",
        }
        assert start["tool_call"]["arguments"] == {"action": "type"}
        assert end["tool_call"]["arguments"] == {"action": "type"}
        assert end["tool_result"] == {
            "tool_call_id": "computer-call",
            "is_error": control != "idle",
            "content_blocks": [
                {
                    "type": "image",
                    "mimeType": "image/png",
                    "size": 7,
                    "data_omitted": True,
                }
            ],
        }
        wire_text = json.dumps((start, end))
        for excluded in ("secret input", "raw-base64", "thumbnail", "private result", "private event", "has_screenshot"):
            assert excluded not in wire_text


def test_computer_projection_omits_malformed_optional_values_and_extras() -> None:
    event = _computer_event("denied", bundle="b" * 20_000, shortcut={"bad": True})
    oversized = _wire(event, start=False)
    assert oversized["computer"] == {"action": "type", "control": "denied"}
    assert event.tool_result is not None
    event = StreamEvent(
        event.type,
        tool_call=event.tool_call,
        tool_result=ToolResult(
            event.tool_result.tool_call_id,
            event.tool_result.content,
            event.tool_result.is_error,
            content_blocks=event.tool_result.content_blocks,
            structured_content={
                "computer": "malformed",
                "oversized": "x" * 20_000,
                "thumbnail": "raw-base64",
            },
        ),
    )
    params = _wire(event, start=False)
    assert params["computer"] == {"action": "type", "control": "idle"}
    assert len(json.dumps(params)) < 1000


def test_computer_history_matches_live_state_and_stubs_images() -> None:
    event = _computer_event("denied", bundle="com.denied.exact")
    live = _wire(event, start=False)
    result = event.tool_result
    assert result is not None
    entry = SimpleNamespace(
        id="entry",
        type="message",
        data={
            "message": {
                "role": "tool",
                "content": [{"type": "tool_use", "tool_call": event.tool_call.to_dict()}],
                "tool_result": result.to_dict(),
            }
        },
    )
    rows = _history_entry_rows(
        entry,
        {},
        {},
        include_delivery_id=False,
        fidelity=True,
        content_read=False,
        approval_origin=False,
    )
    history_result = rows[0]["tool_result"]
    assert history_result["structured_content"] == {"computer": live["computer"]}
    assert history_result["content_blocks"] == [
        {"type": "image", "mimeType": "image/png", "size": 7, "data_omitted": True}
    ]
    history_wire = json.dumps(rows)
    assert "raw-base64" not in history_wire
    assert "private result" not in history_wire


def test_child_history_does_not_inherit_parent_computer_state() -> None:
    child = SimpleNamespace(
        id="child-entry",
        type="message",
        data={"message": {"role": "assistant", "content": [{"type": "text", "text": "child only"}]}},
    )
    rows = _history_entry_rows(
        child,
        {},
        {},
        include_delivery_id=False,
        fidelity=True,
        content_read=False,
        approval_origin=False,
    )
    assert "computer" not in json.dumps(rows)


def test_bounded_tool_output_golden_matches_origin_main() -> None:
    event = StreamEvent(
        StreamEventType.TOOL_EXECUTION_UPDATE,
        tool_call=None,
        delta="x" * 9000,
        data={},
    )
    expected = b'{"jsonrpc":"2.0","method":"event","params":{"event":"tool_output","session_id":"session","tool_call":null,"output":"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx... [truncated: 1000 bytes]","data":{}}}\n'
    assert _params("tool_output", tool_output(event)) == expected
