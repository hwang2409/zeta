from zeta.types import (
    ErrorInfo,
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
)


def test_stream_event_serialization_round_trip() -> None:
    event = StreamEvent(
        StreamEventType.MESSAGE_UPDATE,
        message=Message(
            MessageRole.ASSISTANT,
            [ThinkingContent("plan"), TextContent("answer")],
        ),
        content=ToolUseContent(ToolCall("call-1", "read", {"path": "a"})),
        tool_result=ToolResult("call-1", "ok"),
        error=ErrorInfo("sample", "message"),
        data={"turn": 1},
    )

    assert StreamEvent.from_dict(event.to_dict()) == event
