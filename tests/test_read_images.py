import asyncio
import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from zeta.core.context import ContextAssembler
from zeta.core.store import ConversationStore
from zeta.images import IMAGE_DEGRADATION_WARNING, detect_image_media_type
from zeta.loop import _validated_tool_result
from zeta.providers.anthropic import build_messages_payload
from zeta.providers.codex import build_responses_payload
from zeta.tools import ToolRegistry
from zeta.tools.read import IMAGE_MAX_BYTES
from zeta.tui.checkpoints import CheckpointTranscriptMixin
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
    ("jpeg", "image/jpeg", b"\xff\xd8\xff\xd9"),
    ("gif", "image/gif", b"GIF89a\x01\x00\x01\x00\x00\x00\x00;"),
    (
        "webp",
        "image/webp",
        b"RIFF"
        + (22).to_bytes(4, "little")
        + b"WEBPVP8X"
        + (10).to_bytes(4, "little")
        + b"\x00" * 10,
    ),
)

INVALID_IMAGE_FIXTURES = (
    ("image/png", PNG[:24]),
    ("image/jpeg", b"\xff\xd8\xff"),
    ("image/gif", b"GIF89a\x01\x00\x01\x00"),
    (
        "image/webp",
        b"RIFF"
        + (23).to_bytes(4, "little")
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
async def test_read_falls_back_for_webp_lookalike_text(tmp_path: Path) -> None:
    path = tmp_path / "note.bin"
    path.write_bytes(b"RIFFxxxxWEBPthis is UTF-8 text\n")

    result = await ToolRegistry(tmp_path).execute(
        ToolCall("read-lookalike", "read", {"path": path.name})
    )

    assert result["isError"] is False
    assert result["content"][0]["text"] == "RIFFxxxxWEBPthis is UTF-8 text"


def test_png_lookalike_fails_full_image_validation() -> None:
    data = b"\x89PNG\r\n\x1a\nthis is UTF-8 text"

    assert detect_image_media_type(data) == "image/png"
    assert detect_image_media_type(data, complete=True) is None


@pytest.mark.parametrize(("mime_type", "data"), INVALID_IMAGE_FIXTURES)
def test_complete_image_validation_requires_container_structure(
    mime_type: str, data: bytes
) -> None:
    assert detect_image_media_type(data) == mime_type
    assert detect_image_media_type(data, complete=True) is None


def _oversized_webp() -> bytes:
    chunk_size = IMAGE_MAX_BYTES
    riff_size = chunk_size + 12
    payload = b"\x2f\x00\x00\x00\x00" + b"x" * (chunk_size - 5)
    return (
        b"RIFF"
        + riff_size.to_bytes(4, "little")
        + b"WEBPVP8L"
        + chunk_size.to_bytes(4, "little")
        + payload
    )


def _oversized_lookalike() -> bytes:
    prefix = b"RIFFxxxxWEBPthis is UTF-8 text\n"
    return prefix + b"x" * (IMAGE_MAX_BYTES + 1 - len(prefix))


def _oversized_invalid_webp() -> bytes:
    data = _oversized_webp()
    return data[:12] + b"NOPE" + data[16:]


def _no_eoi_progressive_jpeg() -> bytes:
    return b"\xff\xd8\xff\xc2\x00\x11" + b"progressive JPEG without EOI"


DECISION_TABLE_CASES = [
    pytest.param("row-1-text", b"plain text\n", {}, "text", id="row-1-text"),
    pytest.param(
        "row-2-oversized-valid-png",
        PNG + b"x" * (IMAGE_MAX_BYTES + 1 - len(PNG)),
        {},
        "size",
        id="row-2-oversized-valid-png",
    ),
    pytest.param(
        "row-2-oversized-invalid-webp",
        _oversized_invalid_webp(),
        {},
        "size",
        id="row-2-oversized-invalid-webp",
    ),
    pytest.param(
        "row-2-oversized-lookalike",
        _oversized_lookalike(),
        {},
        "size",
        id="row-2-oversized-lookalike",
    ),
    pytest.param(
        "row-3-oversized-invalid-webp-with-paging",
        _oversized_invalid_webp(),
        {"offset": 1},
        "paging",
        id="row-3-oversized-with-paging",
    ),
]
DECISION_TABLE_CASES.extend(
    pytest.param(
        f"row-4-{format_name}-with-paging",
        data,
        {"limit": 1},
        "paging",
        id=f"row-4-{format_name}-with-paging",
    )
    for format_name, _mime_type, data in IMAGE_FIXTURES
)
DECISION_TABLE_CASES.extend(
    pytest.param(
        f"row-5-{format_name}-trailing-data",
        data + b"trailing metadata",
        {},
        "image",
        id=f"row-5-{format_name}-trailing-data",
    )
    for format_name, _mime_type, data in IMAGE_FIXTURES
)
DECISION_TABLE_CASES.extend(
    [
        pytest.param(
            "row-6-invalid-png",
            PNG[:24],
            {},
            "fallback",
            id="row-6-invalid-png",
        ),
        pytest.param(
            "row-6-no-eoi-progressive-jpeg",
            _no_eoi_progressive_jpeg(),
            {},
            "fallback",
            id="row-6-no-eoi-progressive-jpeg",
        ),
        pytest.param(
            "row-6-invalid-gif",
            b"GIF89a\x01\x00\x01\x00lookalike",
            {},
            "fallback",
            id="row-6-invalid-gif",
        ),
        pytest.param(
            "row-6-invalid-webp-lookalike",
            b"RIFFxxxxWEBPthis is UTF-8 text\n",
            {},
            "fallback",
            id="row-6-invalid-webp-lookalike",
        ),
    ]
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case_name", "data", "arguments", "expected"), DECISION_TABLE_CASES
)
async def test_image_read_decision_table(
    tmp_path: Path,
    case_name: str,
    data: bytes,
    arguments: dict[str, int],
    expected: str,
) -> None:
    path = tmp_path / f"{case_name}.bin"
    path.write_bytes(data)

    result = await ToolRegistry(tmp_path).execute(
        ToolCall(case_name, "read", {"path": path.name, **arguments})
    )

    if expected == "size":
        assert result["isError"] is True
        assert result["content"][0]["text"] == (
            f"image is {len(data)} bytes; cap is "
            f"{IMAGE_MAX_BYTES} bytes (4 MiB)"
        )
    elif expected == "paging":
        assert result["isError"] is True
        assert result["content"][0]["text"] == (
            "offset and limit are not supported for image reads"
        )
    elif expected == "image":
        assert result["isError"] is False
        assert result["content"][1]["type"] == "image"
        assert base64.b64decode(result["content"][1]["data"]) == data
    elif expected == "text":
        assert result["isError"] is False
        assert result["content"][0]["text"] == "plain text"
    else:
        assert all(block["type"] != "image" for block in result["content"])

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
async def test_read_rejects_oversized_webp_before_sample_validation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "large.webp"
    path.write_bytes(_oversized_webp())

    result = await ToolRegistry(tmp_path).execute(
        ToolCall("read-large-webp", "read", {"path": path.name})
    )

    assert result["isError"] is True
    message = result["content"][0]["text"]
    assert "image is 4194324 bytes" in message
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


@pytest.mark.asyncio
async def test_read_image_near_cap_fits_default_context_budget(tmp_path: Path) -> None:
    path = tmp_path / "near-cap.png"
    path.write_bytes(PNG + b"x" * (IMAGE_MAX_BYTES - len(PNG)))

    result = await ToolRegistry(tmp_path).execute(
        ToolCall("read-near-cap", "read", {"path": path.name})
    )
    assert result["isError"] is False

    blocks = result["content"]
    receipt = blocks[0]["text"]
    store = ConversationStore(tmp_path, session_id="near-cap-session")
    store.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            tool_result=ToolResult(
                "read-near-cap",
                receipt,
                content_blocks=blocks,
                structured_content=result["structuredContent"],
            ),
        )
    )

    assembled = await ContextAssembler(store).assemble()

    assert assembled[0].tool_result is not None
    assert assembled[0].tool_result.content_blocks is not None
    assert len(assembled[0].tool_result.content_blocks[1]["data"]) > 5_000_000


def _image_tool_result(data: bytes = PNG) -> ToolResult:
    encoded = base64.b64encode(data).decode("ascii")
    return ToolResult(
        "read-call",
        "filename=screenshot.png bytes=70 format=png",
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


@pytest.mark.asyncio
async def test_tui_renders_compact_image_read_card(tmp_path: Path) -> None:
    path = tmp_path / "screenshot.png"
    path.write_bytes(PNG)
    raw_result = await ToolRegistry(tmp_path).execute(
        ToolCall("read-call", "read", {"path": path.name})
    )
    tool_result = _validated_tool_result(raw_result, "read-call")

    rendered = render_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=ToolCall("read-call", "read", {"path": "screenshot.png"}),
            tool_result=tool_result,
        )
    )
    output = io.StringIO()
    Console(file=output, width=100, force_terminal=False).print(rendered)

    assert output.getvalue().count("filename=screenshot.png bytes=70 format=png") == 1


@pytest.mark.parametrize("corruption", ["missing", "invalid", "truncated"])
def test_corrupt_stored_image_block_keeps_receipt(
    tmp_path: Path, corruption: str
) -> None:
    store = ConversationStore(tmp_path, session_id="corrupt-image")
    store.append_message(
        Message(MessageRole.TOOL_RESULT, tool_result=_image_tool_result())
    )
    rows = store.path.read_text().splitlines()
    row = json.loads(rows[1])
    block = row["data"]["message"]["tool_result"]["content_blocks"][1]
    if corruption == "missing":
        del block["data"]
    elif corruption == "invalid":
        block["data"] = "not-base64"
    else:
        block["data"] = base64.b64encode(PNG[:24]).decode("ascii")
    rows[1] = json.dumps(row)
    store.path.write_text("\n".join(rows) + "\n")

    reopened = ConversationStore(tmp_path, session_id=store.session_id)

    result = reopened.messages()[0].tool_result
    assert result is not None
    assert result.content == (
        "filename=screenshot.png bytes=70 format=png\n"
        f"{IMAGE_DEGRADATION_WARNING}"
    )
    assert result.content_blocks == [
        {
            "type": "text",
            "text": "filename=screenshot.png bytes=70 format=png",
            "truncated": False,
            "full_size": 43,
        },
        {
            "type": "text",
            "text": IMAGE_DEGRADATION_WARNING,
            "truncated": False,
            "full_size": len(IMAGE_DEGRADATION_WARNING.encode("utf-8")),
        },
    ]


def test_tui_resume_renders_receipt_for_corrupt_stored_image(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="tui-corrupt-image")
    store.append_message(
        Message(MessageRole.TOOL_RESULT, tool_result=_image_tool_result())
    )
    rows = store.path.read_text().splitlines()
    row = json.loads(rows[1])
    del row["data"]["message"]["tool_result"]["content_blocks"][1]["data"]
    rows[1] = json.dumps(row)
    store.path.write_text("\n".join(rows) + "\n")
    reopened = ConversationStore(tmp_path, session_id=store.session_id)

    app = object.__new__(CheckpointTranscriptMixin)
    printed: list[object] = []
    app.loop = SimpleNamespace(store=reopened)
    app._presenter = SimpleNamespace(clear=lambda: None)
    app._failed_turn = None
    app._print_unit = printed.append
    app._print_system = printed.append

    app._rebuild_transcript()

    assert len(printed) == 1
    output = io.StringIO()
    Console(file=output, width=100, force_terminal=False).print(printed[0])
    assert IMAGE_DEGRADATION_WARNING in output.getvalue()
