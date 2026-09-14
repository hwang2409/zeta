import asyncio
import base64
import io
from pathlib import Path

import pytest
from rich.console import Console

from zeta.core.context import ContextAssembler
from zeta.core.store import ConversationStore
from zeta.providers.anthropic import build_messages_payload
from zeta.providers.codex import build_responses_payload
from zeta.tools import ToolRegistry
from zeta.tui.render import render_event
from zeta.types import (
    ImageContent,
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolResult,
)

PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6360f8cf00000004000101a2e0c4b00000000049454e44ae426082"
)
IMAGE_FIXTURES = (
    ("png", "image/png", PNG),
    ("jpeg", "image/jpeg", b"\xff\xd8\xff"),
    ("gif", "image/gif", b"GIF89a\x01\x00\x01\x00"),
    (
        "webp",
        "image/webp",
        b"RIFF"
        + (26).to_bytes(4, "little")
        + b"WEBPVP8X"
        + (10).to_bytes(4, "little")
        + b"\x00" * 10,
    ),
)


@pytest.mark.asyncio
@pytest.mark.parametrize(("format_name", "mime_type", "data"), IMAGE_FIXTURES)
async def test_read_detects_images_by_magic_bytes(
    tmp_path: Path, format_name: str, mime_type: str, data: bytes
) -> None:
    path = tmp_path / f"renamed.{format_name}.bin"
    path.write_bytes(data)

    result = await ToolRegistry(tmp_path).execute(
        ToolCall("read-image", "read", {"path": path.name})
    )

    assert result["isError"] is False
    image = result["content"][1]
    assert image["type"] == "image"
    assert image["mimeType"] == mime_type
    assert base64.b64decode(image["data"]) == data
    assert result["structuredContent"]["format"] == format_name


@pytest.mark.asyncio
async def test_read_keeps_text_behavior_for_non_images(tmp_path: Path) -> None:
    path = tmp_path / "note.bin"
    data = "one\r\ntwo\n三".encode()
    path.write_bytes(data)

    result = await ToolRegistry(tmp_path).execute(
        ToolCall("read-text", "read", {"path": path.name})
    )

    assert result["isError"] is False
    block = result["content"][0]
    assert block["text"] == "one\ntwo\n三"
    assert block["full_size"] == len("one\ntwo\n三".encode())
    assert block["truncated"] is False
    assert result["structuredContent"]["sha256"]


@pytest.mark.asyncio
async def test_read_rejects_oversized_images_with_size_and_cap(tmp_path: Path) -> None:
    path = tmp_path / "large.png"
    path.write_bytes(PNG + b"x" * (4 * 1024 * 1024))

    result = await ToolRegistry(tmp_path).execute(
        ToolCall("read-large-image", "read", {"path": path.name})
    )

    assert result["isError"] is True
    message = result["content"][0]["text"]
    assert "image is 4194374 bytes" in message
    assert "cap is 4194304 bytes (4 MiB)" in message


@pytest.mark.asyncio
@pytest.mark.parametrize("argument", ["offset", "limit"])
async def test_read_rejects_paging_arguments_for_images(
    tmp_path: Path, argument: str
) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(PNG)

    result = await ToolRegistry(tmp_path).execute(
        ToolCall("read-paged-image", "read", {"path": path.name, argument: 1})
    )

    assert result["isError"] is True
    assert result["content"][0]["text"] == (
        "offset and limit are not supported for image reads"
    )


def _image_tool_result(data: bytes = PNG) -> ToolResult:
    encoded = base64.b64encode(data).decode("ascii")
    return ToolResult(
        "read-call",
        "",
        content_blocks=[
            {
                "type": "text",
                "text": "filename=screenshot.png bytes=70 format=png",
                "truncated": False,
                "full_size": 43,
            },
            {
                "type": "image",
                "data": encoded,
                "mimeType": "image/png",
                "path": "/tmp/screenshot.png",
                "size": len(data),
            },
        ],
        structured_content={
            "filename": "screenshot.png",
            "bytes": len(data),
            "format": "png",
        },
    )


def test_anthropic_payload_contains_tool_result_image_bytes() -> None:
    payload = build_messages_payload(
        [Message(MessageRole.TOOL_RESULT, tool_result=_image_tool_result())],
        [],
        model="claude-test",
        max_tokens=4096,
        thinking_budget=2048,
    )

    content = payload["messages"][0]["content"][0]["content"]
    image = next(block for block in content if block["type"] == "image")
    assert image["source"] == {
        "type": "base64",
        "media_type": "image/png",
        "data": base64.b64encode(PNG).decode("ascii"),
    }


def test_codex_payload_injects_tool_image_as_input_image() -> None:
    payload = build_responses_payload(
        [Message(MessageRole.TOOL_RESULT, tool_result=_image_tool_result())],
        [],
        model="codex-test",
    )

    output = payload["input"][0]
    assert output["type"] == "function_call_output"
    assert output["output"] == "filename=screenshot.png bytes=70 format=png"
    image = payload["input"][1]["content"][0]
    assert image == {
        "type": "input_image",
        "image_url": "data:image/png;base64," + base64.b64encode(PNG).decode("ascii"),
    }


def test_codex_pasted_image_path_keeps_image_bytes_in_user_content() -> None:
    encoded = base64.b64encode(PNG).decode("ascii")
    payload = build_responses_payload(
        [
            Message(
                MessageRole.USER,
                [TextContent("inspect this"), ImageContent(encoded, "image/png")],
            )
        ],
        [],
        model="codex-test",
    )

    image = payload["input"][0]["content"][1]
    assert image == {
        "type": "input_image",
        "image_url": "data:image/png;base64," + encoded,
    }


def test_image_tool_result_persists_and_replays_byte_identically(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="image-session")
    message = Message(
        MessageRole.TOOL_RESULT,
        tool_result=_image_tool_result(),
    )
    store.append_message(message)
    persisted = message.to_dict()

    reopened = ConversationStore(tmp_path, session_id=store.session_id)
    replayed = reopened.messages()[0]
    assert replayed.to_dict() == persisted
    assert replayed.tool_result is not None
    assert replayed.tool_result.content_blocks == message.tool_result.content_blocks

    context = ContextAssembler(reopened)
    assembled = asyncio.run(context.assemble())
    assert assembled[0].to_dict() == persisted


def test_tui_renders_compact_image_read_card() -> None:
    rendered = render_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=ToolCall("read-call", "read", {"path": "screenshot.png"}),
            tool_result=_image_tool_result(),
        )
    )
    output = io.StringIO()
    Console(file=output, width=100, force_terminal=False).print(rendered)

    assert "filename=screenshot.png bytes=70 format=png" in output.getvalue()
