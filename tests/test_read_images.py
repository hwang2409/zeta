import asyncio
import base64
import binascii
import io
import json
import os
import resource
import struct
import subprocess
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace
from typing import Self

import pytest
from PIL import Image
from rich.console import Console

from zeta.core.context import ContextAssembler
from zeta.core.store import ConversationStore
from zeta.media import image_normalization
from zeta.media.image_policy import (
    ANTHROPIC_IMAGE_POLICY,
    CODEX_IMAGE_POLICY,
    OLLAMA_IMAGE_POLICY,
    WireLimitUnit,
)
from zeta.media.images import IMAGE_DEGRADATION_WARNING, detect_image_media_type
from zeta.protocol.types import (
    ImageContent,
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolResult,
)
from zeta.providers.anthropic import build_messages_payload
from zeta.providers.codex import build_responses_payload
from zeta.runtime.loop.agent import _validated_tool_result
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry
from zeta.tui.checkpoints import CheckpointTranscriptMixin
from zeta.tui.render import render_event

LEGACY_IMAGE_SIZE = 4 * 1024 * 1024
OPENAI_IMAGE_URL_MAX_LENGTH = 20_971_520


def _image_bytes(format_name: str) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (1, 1), "red").save(output, format=format_name)
    return output.getvalue()


PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6360f8cf00000004000101a2e0c4b00000000049454e44ae426082"
)
IMAGE_FIXTURES = (
    ("png", "image/png", PNG),
    ("jpeg", "image/jpeg", _image_bytes("JPEG")),
    ("gif", "image/gif", _image_bytes("GIF")),
    ("webp", "image/webp", _image_bytes("WEBP")),
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


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF)
    )


def _write_compressed_png(path: Path, width: int, height: int) -> None:
    compressor = zlib.compressobj(level=9)
    with path.open("wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        handle.write(
            _png_chunk(
                b"IHDR",
                struct.pack(">IIBBBBB", width, height, 1, 0, 0, 0, 0),
            )
        )
        row = b"\x00" * (1 + (width + 7) // 8)
        for _ in range(height):
            if compressed := compressor.compress(row):
                handle.write(_png_chunk(b"IDAT", compressed))
        if compressed := compressor.flush():
            handle.write(_png_chunk(b"IDAT", compressed))
        handle.write(_png_chunk(b"IEND", b""))


def _write_huge_compressed_png(path: Path) -> None:
    _write_compressed_png(path, 50_000, 50_000)


def _webp_data(chunk_type: bytes, chunk_data: bytes) -> bytes:
    chunk = (
        chunk_type
        + len(chunk_data).to_bytes(4, "little")
        + chunk_data
        + (b"\x00" if len(chunk_data) % 2 else b"")
    )
    body = b"WEBP" + chunk
    return b"RIFF" + len(body).to_bytes(4, "little") + body


READ_WEBP_FIXTURES = (
    (
        "vp8",
        _webp_data(b"VP8 ", b"\x00\x00\x00\x9d\x01\x2a\x01\x00\x01\x00"),
    ),
    ("vp8l", _webp_data(b"VP8L", b"/\x00\x00\x00\x00")),
    ("animated-vp8x", _webp_data(b"VP8X", b"\x02" + b"\x00" * 9)),
)


@pytest.mark.asyncio
@pytest.mark.parametrize(("format_name", "mime_type", "data"), IMAGE_FIXTURES)
async def test_read_detects_images_by_magic_bytes(
    tmp_path: Path, format_name: str, mime_type: str, data: bytes
) -> None:
    path = tmp_path / f"renamed.{format_name}.bin"
    path.write_bytes(data)

    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
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

    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
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

    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
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
    chunk_size = LEGACY_IMAGE_SIZE
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
    return prefix + b"x" * (LEGACY_IMAGE_SIZE + 1 - len(prefix))


def _oversized_invalid_webp() -> bytes:
    data = _oversized_webp()
    return data[:12] + b"NOPE" + data[16:]


def _no_eoi_progressive_jpeg() -> bytes:
    return b"\xff\xd8\xff\xc2\x00\x11" + b"progressive JPEG without EOI"


def _jpeg_eoi_in_app_payload() -> bytes:
    return b"\xff\xd8\xff\xe1\x00\x05ab\xff\xd9"


def _oversized_truncated_png() -> bytes:
    data = PNG[:24]
    return data + b"x" * (LEGACY_IMAGE_SIZE + 1 - len(data))


def _malformed_webp_chunks() -> bytes:
    return (
        b"RIFF"
        + (26).to_bytes(4, "little")
        + b"WEBPVP8X"
        + (10).to_bytes(4, "little")
        + b"\x00" * 10
        + b"NOPE"
    )


def _leading_junk_webp() -> bytes:
    junk = b"JUNK" + (0).to_bytes(4, "little")
    codec = b"VP8X" + (10).to_bytes(4, "little") + b"\x00" * 10
    body = b"WEBP" + junk + codec
    return b"RIFF" + len(body).to_bytes(4, "little") + body


def _oversized(data: bytes) -> bytes:
    return data + b"x" * (LEGACY_IMAGE_SIZE + 1 - len(data))


DECISION_TABLE_CASES = [
    pytest.param("row-1-text", b"plain text\n", {}, "text", id="row-1-text"),
]
DECISION_TABLE_CASES.extend(
    pytest.param(
        f"row-3-oversized-valid-{format_name}-with-paging",
        _oversized(data),
        {"offset": 1},
        "paging",
        id=f"row-3-oversized-valid-{format_name}-with-paging",
    )
    for format_name, _mime_type, data in IMAGE_FIXTURES
)
DECISION_TABLE_CASES.extend(
    pytest.param(
        f"row-3-oversized-invalid-{format_name}-with-paging",
        _oversized(invalid_data),
        {"offset": 1},
        "paging",
        id=f"row-3-oversized-invalid-{format_name}-with-paging",
    )
    for (format_name, _mime_type, _valid_data), (_invalid_mime, invalid_data) in zip(
        IMAGE_FIXTURES, INVALID_IMAGE_FIXTURES, strict=True
    )
)
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
        f"row-4-{format_name}-invalid-with-paging",
        data,
        {"limit": 1},
        "paging",
        id=f"row-4-{format_name}-invalid-with-paging",
    )
    for (format_name, _image_mime_type, _valid_data), (_mime_type, data) in zip(
        IMAGE_FIXTURES, INVALID_IMAGE_FIXTURES, strict=True
    )
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
            (True, "decode"),
            id="row-6-invalid-png",
        ),
        pytest.param(
            "row-6-no-eoi-progressive-jpeg",
            _no_eoi_progressive_jpeg(),
            {},
            (True, "decode"),
            id="row-6-no-eoi-progressive-jpeg",
        ),
        pytest.param(
            "row-6-jpeg-eoi-in-app-payload",
            _jpeg_eoi_in_app_payload(),
            {},
            (True, "decode"),
            id="row-6-jpeg-eoi-in-app-payload",
        ),
        pytest.param(
            "row-6-invalid-gif",
            b"GIF89a\x01\x00\x01\x00\x00\x00;",
            {},
            (False, "GIF89a\x01\x00\x01\x00\x00\x00;"),
            id="row-6-invalid-gif",
        ),
        pytest.param(
            "row-6-invalid-webp-lookalike",
            b"RIFFxxxxWEBPthis is UTF-8 text\n",
            {},
            (False, "RIFFxxxxWEBPthis is UTF-8 text"),
            id="row-6-invalid-webp-lookalike",
        ),
        pytest.param(
            "row-6-malformed-webp-chunks",
            _malformed_webp_chunks(),
            {},
            (False, _malformed_webp_chunks().decode()),
            id="row-6-malformed-webp-chunks",
        ),
        pytest.param(
            "row-6-leading-junk-webp",
            _leading_junk_webp(),
            {},
            (False, _leading_junk_webp().decode()),
            id="row-6-leading-junk-webp",
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

    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall(case_name, "read", {"path": path.name, **arguments})
    )

    if expected == "size":
        assert result["isError"] is True
        assert result["content"][0]["text"] == (
            f"image is {len(data)} bytes; cap is "
            f"{LEGACY_IMAGE_SIZE} bytes (4 MiB)"
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
        assert isinstance(expected, tuple)
        expected_error, expected_text = expected
        assert result["isError"] is expected_error
        if expected_text == "decode":
            assert result["content"][0]["text"].startswith("could not decode image:")
        else:
            assert result["content"][0]["text"] == expected_text


@pytest.mark.asyncio
@pytest.mark.parametrize(("case_name", "data"), READ_WEBP_FIXTURES)
async def test_read_detects_webp_codecs(
    tmp_path: Path, case_name: str, data: bytes
) -> None:
    path = tmp_path / f"{case_name}.bin"
    path.write_bytes(data)

    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall(f"read-{case_name}", "read", {"path": path.name})
    )

    assert result["isError"] is False
    assert result["content"][1]["type"] == "image"
    assert result["content"][1]["mimeType"] == "image/webp"
    assert base64.b64decode(result["content"][1]["data"]) == data
    assert result["structuredContent"]["format"] == "webp"

@pytest.mark.asyncio
@pytest.mark.parametrize("argument", ["offset", "limit"])
async def test_read_rejects_paging_arguments_for_images(
    tmp_path: Path, argument: str
) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(PNG)

    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("read-paged-image", "read", {"path": path.name, argument: 1})
    )

    assert result["isError"] is True
    assert result["content"][0]["text"] == (
        "offset and limit are not supported for image reads"
    )


@pytest.mark.asyncio
async def test_read_image_near_cap_fits_default_context_budget(tmp_path: Path) -> None:
    path = tmp_path / "near-cap.png"
    path.write_bytes(PNG + b"x" * (LEGACY_IMAGE_SIZE - len(PNG)))

    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
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
    raw_result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
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

    assert output.getvalue().count("filename=screenshot.png original=1x1 70B png sent=1x1 70B png") == 1


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
    app._dismiss_model_picker = lambda: None
    app._failed_turn = None
    app._print_unit = printed.append
    app._print_system = printed.append

    app._rebuild_transcript()

    assert len(printed) == 1
    output = io.StringIO()
    Console(file=output, width=100, force_terminal=False).print(printed[0])
    assert IMAGE_DEGRADATION_WARNING in output.getvalue()


def _save_image(path: Path, image: Image.Image, format_name: str, **kwargs: object) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=format_name, **kwargs)
    data = buffer.getvalue()
    path.write_bytes(data)
    return data


def _jpeg_with_late_large_sof() -> bytes:
    source = _image_bytes("JPEG")
    sof = next(
        index
        for index in range(2, len(source) - 9)
        if source[index] == 0xFF and source[index + 1] in range(0xC0, 0xC4)
    )
    # The SOF stores precision, height, and width after its two-byte length.
    source = (
        source[: sof + 5]
        + (30_000).to_bytes(2, "big")
        + (40_000).to_bytes(2, "big")
        + source[sof + 9 :]
    )
    app = b"\xff\xe1" + (65_535).to_bytes(2, "big") + b"x" * 65_533
    return source[:2] + app + app + source[2:]


@pytest.mark.asyncio
async def test_vips_path_converts_display_p3_to_srgb_and_preserves_untagged(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyvips", reason="libvips is required to exercise the vips path")
    profile_path = Path("/System/Library/ColorSync/Profiles/Display P3.icc")
    if not profile_path.is_file():
        pytest.skip("no Display P3 ICC profile is available on this host")
    from PIL import ImageCms

    p3_profile = profile_path.read_bytes()
    srgb_profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
    source = Image.new("RGB", (9000, 100), (220, 80, 20))
    source.info["icc_profile"] = p3_profile
    expected = ImageCms.profileToProfile(
        source,
        ImageCms.ImageCmsProfile(io.BytesIO(p3_profile)),
        srgb_profile,
        outputMode="RGB",
    ).getpixel((100, 50))
    tagged_path = tmp_path / "display-p3.png"
    _save_image(tagged_path, source, "PNG", icc_profile=p3_profile)
    untagged_path = tmp_path / "untagged.png"
    _save_image(untagged_path, Image.new("RGB", source.size, (220, 80, 20)), "PNG")

    async def read(path: Path) -> Image.Image:
        result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
            ToolCall("read-image", "read", {"path": path.name})
        )
        assert result["isError"] is False, result
        return Image.open(io.BytesIO(base64.b64decode(result["content"][1]["data"]))).convert(
            "RGB"
        )

    tagged = await read(tagged_path)
    untagged = await read(untagged_path)
    assert all(abs(actual - wanted) <= 8 for actual, wanted in zip(tagged.getpixel((100, 50)), expected))
    assert all(abs(actual - wanted) <= 8 for actual, wanted in zip(untagged.getpixel((100, 50)), (220, 80, 20)))


@pytest.mark.asyncio
async def test_jpeg_late_sof_over_one_gigapixel_metadata_only(tmp_path: Path) -> None:
    path = tmp_path / "late-sof.jpg"
    path.write_bytes(_jpeg_with_late_large_sof())
    result = await image_normalization.prepare_image(
        os.open(path, os.O_RDONLY),
        file_size=path.stat().st_size,
        policy=ANTHROPIC_IMAGE_POLICY,
    )
    assert result.error is None
    assert result.image is not None
    assert result.image.data is None
    assert (result.image.original_width, result.image.original_height) == (40_000, 30_000)


@pytest.mark.asyncio
async def test_read_large_image_downscales_not_errors(tmp_path: Path) -> None:
    path = tmp_path / "large.jpg"
    image = Image.new("RGB", (8000, 6000), "#4976a3")
    original = _save_image(path, image, "JPEG", quality=95)
    del image
    padding = 12 * 1024 * 1024 - len(original)
    assert padding > 0
    path.write_bytes(original + b"\x00" * padding)
    original = path.read_bytes()

    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("read-large-image", "read", {"path": path.name})
    )

    assert result["isError"] is False, result
    details = result["structuredContent"]
    assert details["original"] == {
        "bytes": len(original),
        "width": 8000,
        "height": 6000,
        "format": "jpeg",
    }
    assert details["sent"]["bytes"] <= ANTHROPIC_IMAGE_POLICY.max_raw_bytes("image/jpeg")
    assert details["sent"]["width"] <= 8000
    assert details["sent"]["height"] <= 8000
    assert details["original_path"] == str(path)
    assert details["original_unchanged"] is True
    assert path.read_bytes() == original


def test_image_limits_match_provider_documentation() -> None:
    assert ANTHROPIC_IMAGE_POLICY.max_wire_size == 10_000_000
    assert ANTHROPIC_IMAGE_POLICY.wire_limit_unit is WireLimitUnit.BASE64_CHARACTERS
    assert ANTHROPIC_IMAGE_POLICY.max_dimension == 8000
    assert CODEX_IMAGE_POLICY.max_wire_size == OPENAI_IMAGE_URL_MAX_LENGTH
    assert CODEX_IMAGE_POLICY.wire_limit_unit is WireLimitUnit.DATA_URL_CHARACTERS
    assert CODEX_IMAGE_POLICY.max_dimension is None
    assert OLLAMA_IMAGE_POLICY.max_wire_size is None
    assert OLLAMA_IMAGE_POLICY.max_dimension is None


@pytest.mark.asyncio
async def test_read_uses_active_provider_limits(tmp_path: Path) -> None:
    path = tmp_path / "codex-within-limit.png"
    original = PNG + b"\x00" * (6 * 1024 * 1024 - len(PNG))
    path.write_bytes(original)

    result = await ToolRegistry(
        tmp_path,
        skill_catalog=SkillCatalog.empty(),
        image_policy=CODEX_IMAGE_POLICY,
    ).execute(ToolCall("read-codex-image", "read", {"path": path.name}))

    assert result["isError"] is False
    assert base64.b64decode(result["content"][1]["data"]) == original
    assert result["structuredContent"]["original"] == result["structuredContent"]["sent"]


@pytest.mark.asyncio
async def test_small_image_no_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "small-fast.png"
    original = _save_image(path, Image.new("RGB", (40, 30), "navy"), "PNG")

    async def unexpected_process(*args: object, **kwargs: object) -> object:
        raise AssertionError("small unchanged images must not start a subprocess")

    monkeypatch.setattr(
        image_normalization.asyncio, "create_subprocess_exec", unexpected_process
    )
    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("read-small-fast", "read", {"path": path.name})
    )

    assert result["isError"] is False
    assert base64.b64decode(result["content"][1]["data"]) == original


@pytest.mark.asyncio
async def test_read_small_image_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "small.png"
    original = _save_image(path, Image.new("RGB", (40, 30), "navy"), "PNG")

    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("read-small-image", "read", {"path": path.name})
    )

    assert result["isError"] is False
    assert base64.b64decode(result["content"][1]["data"]) == original
    details = result["structuredContent"]
    assert details["original"] == details["sent"]
    assert details["original_unchanged"] is True


@pytest.mark.asyncio
async def test_read_transparent_png_large(tmp_path: Path) -> None:
    path = tmp_path / "transparent.png"
    image = Image.new("RGBA", (9000, 100), (20, 40, 60, 0))
    for x in range(0, image.width, 2):
        image.paste((200, 80, 30, 160), (x, 0, x + 1, image.height))
    original = _save_image(path, image, "PNG")

    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("read-transparent-image", "read", {"path": path.name})
    )

    assert result["isError"] is False
    sent = base64.b64decode(result["content"][1]["data"])
    with Image.open(io.BytesIO(sent)) as normalized:
        assert normalized.mode == "RGBA"
        assert "A" in normalized.getbands()
        assert max(normalized.size) <= 8000
    assert path.read_bytes() == original


@pytest.mark.asyncio
async def test_read_huge_dimension_image_bounded_memory(tmp_path: Path) -> None:
    path = tmp_path / "huge.jpg"
    image = Image.new("RGB", (50_000, 2_100), "#345678")
    _save_image(path, image, "JPEG", quality=75)
    del image

    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("read-huge-image", "read", {"path": path.name})
    )

    assert result["isError"] is False
    assert result["structuredContent"]["original"]["width"] == 50_000
    assert max(
        result["structuredContent"]["sent"]["width"],
        result["structuredContent"]["sent"]["height"],
    ) <= 8000


@pytest.mark.asyncio
async def test_canceling_image_normalization_kills_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "image.png"
    _write_compressed_png(path, 9_000, 1)
    started = asyncio.Event()
    killed = asyncio.Event()

    class HangingProcess:
        returncode: int | None = None

        async def communicate(self) -> tuple[bytes, bytes]:
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        def kill(self) -> None:
            self.returncode = -9
            killed.set()

        async def wait(self) -> int:
            return self.returncode or 0

    async def create_process(*args: object, **kwargs: object) -> HangingProcess:
        return HangingProcess()

    monkeypatch.setattr(
        image_normalization.asyncio,
        "create_subprocess_exec",
        create_process,
    )
    task = asyncio.create_task(
        image_normalization.prepare_image(
            os.open(path, os.O_RDONLY),
            file_size=path.stat().st_size,
            policy=ANTHROPIC_IMAGE_POLICY,
        )
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert killed.is_set()


@pytest.mark.asyncio
async def test_large_png_lookalike_validation_is_bounded(tmp_path: Path) -> None:
    path = tmp_path / "lookalike.png"
    with path.open("wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\nnot an image")
        handle.seek(64 * 1024 * 1024)
        handle.write(b"x")
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("read-lookalike", "read", {"path": path.name})
    )

    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert result["isError"] is True
    assert rss_after - rss_before < 32 * 1024 * 1024


@pytest.mark.asyncio
async def test_large_screenshot_png_returns_pixels(tmp_path: Path) -> None:
    path = tmp_path / "screenshot-12k.png"
    _write_compressed_png(path, 12_000, 12_000)

    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("read-large-screenshot", "read", {"path": path.name})
    )

    assert result["isError"] is False
    assert len(result["content"]) == 2
    assert result["content"][1]["type"] == "image"
    assert result["structuredContent"]["sent"]["bytes"] > 0


@pytest.mark.asyncio
async def test_png_normalization_rss_bounded_macos_and_linux(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "screenshot-10k.png"
    _write_compressed_png(path, 10_000, 10_000)
    samples: list[int] = []
    sample_rss = image_normalization._process_rss_bytes

    def record_rss(process_id: int) -> int:
        value = sample_rss(process_id)
        samples.append(value)
        return value

    monkeypatch.setattr(image_normalization, "_process_rss_bytes", record_rss)
    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("read-rss-screenshot", "read", {"path": path.name})
    )

    assert result["isError"] is False
    assert len(result["content"]) == 2
    assert 0 < max(samples) < image_normalization.WORKER_RSS_BUDGET_BYTES


@pytest.mark.asyncio
async def test_over_one_gigapixel_returns_metadata_only(tmp_path: Path) -> None:
    path = tmp_path / "over-one-gigapixel.png"
    _write_huge_compressed_png(path)

    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("read-pathological-png", "read", {"path": path.name})
    )

    assert result["isError"] is False
    assert len(result["content"]) == 1
    assert "1 gigapixel" in result["content"][0]["text"]
    assert result["structuredContent"]["sha256"]
    assert result["structuredContent"]["path"] == str(path)


def test_worker_rss_tolerates_exit_while_reading_procfs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class VanishedStatus:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def __iter__(self) -> Self:
            return self

        def __next__(self) -> str:
            raise ProcessLookupError(3, "No such process")

    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: VanishedStatus())
    monkeypatch.setattr(
        image_normalization.subprocess,
        "check_output",
        lambda *args, **kwargs: "",
    )

    assert image_normalization._process_rss_bytes(1234) == 0


@pytest.mark.asyncio
async def test_watchdog_kills_runaway_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "runaway.png"
    _write_compressed_png(path, 9_000, 1)
    killed = asyncio.Event()

    class RunawayProcess:
        pid = 1234
        returncode: int | None = None

        async def communicate(self) -> tuple[bytes, bytes]:
            await killed.wait()
            return b"", b""

        def kill(self) -> None:
            self.returncode = -9
            killed.set()

        async def wait(self) -> int:
            await killed.wait()
            return self.returncode or 0

    async def create_process(*args: object, **kwargs: object) -> RunawayProcess:
        return RunawayProcess()

    monkeypatch.setattr(
        image_normalization.asyncio, "create_subprocess_exec", create_process
    )
    monkeypatch.setattr(
        image_normalization, "_process_rss_bytes", lambda _pid: 2**63
    )
    result = await image_normalization.prepare_image(
        os.open(path, os.O_RDONLY),
        file_size=path.stat().st_size,
        policy=ANTHROPIC_IMAGE_POLICY,
    )

    assert killed.is_set()
    assert result.image is not None
    assert result.image.data is None
    assert "RSS safety budget" in (result.image.note or "")


@pytest.mark.asyncio
async def test_watchdog_accepts_worker_exit_during_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExitedProcess:
        pid = 1234
        returncode: int | None = None

        def kill(self) -> None:
            raise ProcessLookupError

    monkeypatch.setattr(
        image_normalization, "_process_rss_bytes", lambda _pid: 2**63
    )

    limit = await image_normalization._watch_worker(ExitedProcess())  # type: ignore[arg-type]

    assert limit is None


@pytest.mark.asyncio
async def test_huge_png_bounded_memory(tmp_path: Path) -> None:
    path = tmp_path / "huge.png"
    _write_huge_compressed_png(path)
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("read-huge-png", "read", {"path": path.name})
    )

    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert result["isError"] is False
    assert len(result["content"]) == 1
    assert "could not be sent" in result["content"][0]["text"]
    assert result["structuredContent"]["original"]["width"] == 50_000
    assert result["structuredContent"]["original"]["height"] == 50_000
    assert rss_after - rss_before < 128 * 1024 * 1024


def test_importing_registry_does_not_import_pillow() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import zeta.tools.registry; assert not any(name == 'PIL' or name.startswith('PIL.') for name in sys.modules)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.asyncio
async def test_codex_data_url_stays_within_wire_limit(tmp_path: Path) -> None:
    data = _image_bytes("JPEG") + b"x" * (16 * 1024 * 1024)
    path = tmp_path / "large.jpg"
    path.write_bytes(data)
    result = await ToolRegistry(
        tmp_path,
        skill_catalog=SkillCatalog.empty(),
        image_policy=CODEX_IMAGE_POLICY,
    ).execute(ToolCall("read-large", "read", {"path": path.name}))
    assert result["isError"] is False
    payload = build_responses_payload(
        [
            Message(
                MessageRole.TOOL_RESULT,
                tool_result=_validated_tool_result(result, "read-large"),
            )
        ],
        [],
        model="codex-test",
    )
    image_url = payload["input"][1]["content"][0]["image_url"]
    assert len(image_url) <= OPENAI_IMAGE_URL_MAX_LENGTH


@pytest.mark.asyncio
async def test_exif_orientation_is_applied_before_sizing(tmp_path: Path) -> None:
    output = io.BytesIO()
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", (9000, 100), "red").save(output, "JPEG", exif=exif)
    path = tmp_path / "rotated.jpg"
    path.write_bytes(output.getvalue())
    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("read-rotated", "read", {"path": path.name})
    )

    assert result["isError"] is False
    assert result["structuredContent"]["sent"]["height"] == 8000
    assert result["structuredContent"]["sent"]["width"] < 100


@pytest.mark.asyncio
async def test_animated_gif_is_flattened_to_first_frame(tmp_path: Path) -> None:
    output = io.BytesIO()
    frames = [Image.new("RGB", (2, 2), color) for color in ("red", "blue")]
    frames[0].save(output, "GIF", save_all=True, append_images=frames[1:], loop=0)
    path = tmp_path / "animated.gif"
    path.write_bytes(output.getvalue())
    result = await ToolRegistry(
        tmp_path,
        skill_catalog=SkillCatalog.empty(),
        image_policy=CODEX_IMAGE_POLICY,
    ).execute(ToolCall("read-gif", "read", {"path": path.name}))

    assert result["isError"] is False
    sent = base64.b64decode(result["content"][1]["data"])
    with Image.open(io.BytesIO(sent)) as image:
        assert getattr(image, "n_frames", 1) == 1


@pytest.mark.asyncio
async def test_read_corrupt_image_errors_cleanly(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"not an image")

    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("read-corrupt-image", "read", {"path": path.name})
    )

    assert result["isError"] is True
    assert result["content"][0]["text"].startswith("could not decode image:")
