import base64
import hashlib
from pathlib import Path

import pytest

from zeta.tools import ToolRegistry
from zeta.tools.registry import validate_tool_result
from zeta.types import (
    StructuredContentValue,
    StructuredToolResult,
    ToolCall,
    ToolTextBlock,
)


def _text_block(result: StructuredToolResult) -> ToolTextBlock:
    content = result["content"]
    assert type(content) is list
    block = content[0]
    assert type(block) is dict
    return block


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (
            {
                "content": [],
                "isError": False,
            },
            "missing top-level keys",
        ),
        (
            {
                "content": [],
                "isError": False,
                "structuredContent": None,
                "extra": True,
            },
            "unexpected top-level keys",
        ),
        (
            {
                "content": [{"type": "image"}],
                "isError": False,
                "structuredContent": None,
            },
            "invalid image shape",
        ),
        (
            {
                "content": [],
                "isError": False,
                "structuredContent": None,
                "content_blocks": [],
            },
            "legacy content_blocks",
        ),
    ],
)
def test_validate_tool_result_rejects_malformed_shapes(
    result: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_tool_result(result)


@pytest.mark.parametrize(
    "resource",
    [
        {"uri": "file:///tmp/note.txt"},
        {"uri": "file:///tmp/note.txt", "text": "note", "blob": "bm90ZQ=="},
    ],
)
def test_validate_tool_result_requires_one_resource_payload(
    resource: dict[str, str],
) -> None:
    with pytest.raises(ValueError, match="text or blob"):
        validate_tool_result(
            {
                "content": [{"type": "resource", "resource": resource}],
                "isError": False,
                "structuredContent": None,
            }
        )


def test_validate_tool_result_rejects_invalid_image_base64() -> None:
    with pytest.raises(ValueError, match="valid base64"):
        validate_tool_result(
            {
                "content": [
                    {"type": "image", "data": "not base64!", "mimeType": "image/png"}
                ],
                "isError": False,
                "structuredContent": None,
            }
        )


@pytest.mark.parametrize(
    ("mime_type", "data"),
    [("image/png", b"not a png"), ("image/jpeg", b"\x89PNG\r\n\x1a\n")],
)
def test_validate_tool_result_rejects_image_media_mismatch(
    mime_type: str, data: bytes
) -> None:
    with pytest.raises(ValueError, match="does not match media type"):
        validate_tool_result(
            {
                "content": [
                    {
                        "type": "image",
                        "data": base64.b64encode(data).decode(),
                        "mimeType": mime_type,
                    }
                ],
                "isError": False,
                "structuredContent": None,
            }
        )


@pytest.mark.asyncio
async def test_success_result_uses_mcp_content_shape(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path, register_builtin=False)
    registry.register("echo", lambda arguments: "hello")

    result = await registry.execute(ToolCall("call-1", "echo", {}))

    assert set(result) == {"content", "isError", "structuredContent"}
    assert result["isError"] is False
    assert result["structuredContent"] is None
    assert _text_block(result) == {
        "type": "text",
        "text": "hello",
        "truncated": False,
        "full_size": 5,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("handler_kind", ["exception", "malformed"])
async def test_handler_errors_are_capped(
    tmp_path: Path, handler_kind: str
) -> None:
    oversized_message = "x" * 20
    if handler_kind == "exception":

        def handler(arguments: dict[str, object]) -> str:
            raise ValueError(oversized_message)

    else:

        def handler(arguments: dict[str, object]) -> dict[str, object]:
            return {
                "content": "invalid",
                "isError": False,
                "structuredContent": None,
            }

    registry = ToolRegistry(tmp_path, max_output_chars=4, register_builtin=False)
    registry.register("failure", handler)

    result = await registry.execute(ToolCall("failure-1", "failure", {}))

    expected = (
        oversized_message
        if handler_kind == "exception"
        else "invalid tool handler result: content must be an array"
    )
    assert result["isError"] is True
    assert result["content"][0] == {
        "type": "text",
        "text": expected[:4],
        "truncated": True,
        "full_size": len(expected),
    }


@pytest.mark.asyncio
async def test_result_cap_is_aggregate_across_text_blocks(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path, max_output_chars=5, register_builtin=False)
    registry.register(
        "multi",
        lambda arguments: {
            "content": [
                {
                    "type": "text",
                    "text": "abc",
                    "truncated": False,
                    "full_size": 3,
                },
                {
                    "type": "text",
                    "text": "defgh",
                    "truncated": False,
                    "full_size": 5,
                },
            ],
            "isError": False,
            "structuredContent": None,
        },
    )

    result = await registry.execute(ToolCall("multi-1", "multi", {}))

    assert result["content"] == [
        {
            "type": "text",
            "text": "abc",
            "truncated": False,
            "full_size": 3,
        },
        {
            "type": "text",
            "text": "de",
            "truncated": True,
            "full_size": 5,
        },
    ]


@pytest.mark.asyncio
async def test_mixed_mcp_content_caps_text_and_preserves_other_blocks(
    tmp_path: Path,
) -> None:
    image = {
        "type": "image",
        "data": base64.b64encode(
            bytes.fromhex(
                "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
                "0000000d49444154789c6360f8cf00000004000101a2e0c4b00000000049454e44ae426082"
            )
        ).decode(),
        "mimeType": "image/png",
        "annotations": {"audience": ["user"]},
    }
    resource = {
        "type": "resource",
        "resource": {
            "uri": "file:///tmp/note.txt",
            "mimeType": "text/plain",
            "text": "resource body",
        },
    }
    registry = ToolRegistry(tmp_path, max_output_chars=4, register_builtin=False)
    registry.register(
        "mixed",
        lambda arguments: {
            "content": [
                {
                    "type": "text",
                    "text": "oversized",
                    "truncated": False,
                    "full_size": 9,
                },
                image,
                resource,
            ],
            "isError": False,
            "structuredContent": None,
        },
    )

    result = await registry.execute(ToolCall("mixed-1", "mixed", {}))

    assert result["isError"] is False
    assert result["content"] == [
        {
            "type": "text",
            "text": "over",
            "truncated": True,
            "full_size": 9,
        },
        image,
        resource,
    ]
    assert result["content"][1] is not image


@pytest.mark.asyncio
async def test_failure_result_uses_mcp_error_shape(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path, register_builtin=False)

    result = await registry.execute(ToolCall("call-1", "missing", {}))

    assert result["isError"] is True
    assert result["structuredContent"] is None
    block = _text_block(result)
    assert block["type"] == "text"
    assert block["text"] == "unknown tool: missing"


@pytest.mark.asyncio
async def test_dispatch_rejects_cyclic_structured_content(tmp_path: Path) -> None:
    structured_content: dict[str, StructuredContentValue] = {}
    cycle: list[StructuredContentValue] = [structured_content]
    structured_content["cycle"] = cycle
    handler_result: StructuredToolResult = {
        "content": [
            {"type": "text", "text": "bad", "truncated": False, "full_size": 3}
        ],
        "isError": False,
        "structuredContent": structured_content,
    }
    registry = ToolRegistry(tmp_path, register_builtin=False)
    registry.register("cycle", lambda arguments: handler_result)

    result = await registry.execute(ToolCall("cycle-1", "cycle", {}))

    assert result["isError"] is True
    assert result["content"][0]["text"] == (
        "invalid tool handler result: cyclic structuredContent"
    )


@pytest.mark.asyncio
async def test_dispatch_accepts_aliased_structured_content(tmp_path: Path) -> None:
    shared: list[StructuredContentValue] = []
    structured_content: dict[str, StructuredContentValue] = {
        "left": shared,
        "right": shared,
    }
    handler_result: StructuredToolResult = {
        "content": [
            {"type": "text", "text": "good", "truncated": False, "full_size": 4}
        ],
        "isError": False,
        "structuredContent": structured_content,
    }
    registry = ToolRegistry(tmp_path, register_builtin=False)
    registry.register("alias", lambda arguments: handler_result)

    result = await registry.execute(ToolCall("alias-1", "alias", {}))

    assert result["isError"] is False
    assert result["content"][0]["text"] == "good"


@pytest.mark.asyncio
async def test_dispatch_rejects_deep_structured_content(tmp_path: Path) -> None:
    structured_content: dict[str, StructuredContentValue] = {}
    current = structured_content
    for _ in range(100):
        child: dict[str, StructuredContentValue] = {}
        current["child"] = child
        current = child
    handler_result: StructuredToolResult = {
        "content": [
            {"type": "text", "text": "bad", "truncated": False, "full_size": 3}
        ],
        "isError": False,
        "structuredContent": structured_content,
    }
    registry = ToolRegistry(tmp_path, register_builtin=False)
    registry.register("deep", lambda arguments: handler_result)

    result = await registry.execute(ToolCall("deep-1", "deep", {}))

    assert result["isError"] is True
    assert result["content"][0]["text"] == (
        "invalid tool handler result: structuredContent depth > 32"
    )


@pytest.mark.asyncio
async def test_capped_text_exposes_truncation_metadata(tmp_path: Path) -> None:
    file_path = tmp_path / "large.txt"
    file_path.write_text("abcdefgh", encoding="utf-8")
    registry = ToolRegistry(tmp_path, max_output_chars=4)

    result = await registry.execute(
        ToolCall("call-1", "read", {"path": file_path.name})
    )

    assert result["isError"] is False
    assert _text_block(result) == {
        "type": "text",
        "text": "abcd",
        "truncated": True,
        "full_size": 8,
    }


@pytest.mark.asyncio
async def test_builtin_tools_populate_structured_content(tmp_path: Path) -> None:
    file_path = tmp_path / "note.txt"
    file_path.write_text("one\ntwo\n", encoding="utf-8")
    registry = ToolRegistry(tmp_path)

    read_result = await registry.execute(
        ToolCall("read-1", "read", {"path": "note.txt"})
    )
    bash_result = await registry.execute(
        ToolCall("bash-1", "bash", {"cmd": "printf bash"})
    )
    exec_result = await registry.execute(
        ToolCall("exec-1", "exec", {"command": "true"})
    )

    assert read_result["structuredContent"] == {
        "path": str(file_path),
        "sha256": hashlib.sha256(file_path.read_bytes()).hexdigest(),
        "line_count": 2,
    }
    assert bash_result["structuredContent"] == {
        "stdout": "bash",
        "stderr": "",
        "exit_code": 0,
        "cwd_after": str(tmp_path),
    }
    assert exec_result["structuredContent"] == {
        "exit_code": 0,
        "cwd": str(tmp_path),
    }
