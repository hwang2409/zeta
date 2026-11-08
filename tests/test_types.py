from zeta.protocol.types import (
    ErrorInfo,
    ImageContent,
    Message,
    MessageRole,
    RedactedThinkingContent,
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


def test_signed_and_redacted_thinking_round_trip() -> None:
    message = Message(
        MessageRole.ASSISTANT,
        [ThinkingContent("plan", "sig-1"), RedactedThinkingContent("opaque")],
    )

    assert Message.from_dict(message.to_dict()) == message


def test_thinking_wire_body_marks_display_safe_text() -> None:
    assert ThinkingContent("summary").to_dict() == {
        "type": "thinking",
        "text": "summary",
        "body": "summary",
    }
    assert ThinkingContent("", "opaque").to_dict() == {
        "type": "thinking",
        "text": "",
        "signature": "opaque",
    }


def test_image_validation_provenance_is_not_public_state() -> None:
    data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNg+M8AAAAEAAEBouDEsAAAAABJRU5ErkJggg=="
    reconstructed = Message.from_dict(
        {
            "role": "user",
            "content": [{"type": "image", "data": data, "mimeType": "image/png"}],
        }
    ).content[0]
    direct = ImageContent(data, "image/png")
    assert reconstructed == direct
    assert hash(reconstructed) == hash(direct)
    assert "validation" not in repr(reconstructed)
    assert reconstructed.to_dict() == direct.to_dict()
    assert (
        Message.from_dict(
            {"role": "user", "content": [reconstructed.to_dict()]}
        ).content[0]
        == reconstructed
    )


def test_tool_result_round_trip_preserves_mixed_content_blocks() -> None:
    result = ToolResult(
        "call-1",
        "[image block]\n[resource: file:///tmp/note.txt]",
        content_blocks=[
            {
                "type": "image",
                "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNg+M8AAAAEAAEBouDEsAAAAABJRU5ErkJggg==",
                "mimeType": "image/png",
            },
            {
                "type": "resource",
                "resource": {
                    "uri": "file:///tmp/note.txt",
                    "text": "note",
                },
            },
        ],
    )

    assert ToolResult.from_dict(result.to_dict()) == result
