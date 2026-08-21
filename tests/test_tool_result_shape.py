import hashlib
from pathlib import Path

import pytest

from zeta.tools import ToolRegistry
from zeta.tools.registry import validate_tool_result
from zeta.types import StructuredToolResult, ToolCall, ToolTextBlock


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
            "invalid shape",
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
async def test_failure_result_uses_mcp_error_shape(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path, register_builtin=False)

    result = await registry.execute(ToolCall("call-1", "missing", {}))

    assert result["isError"] is True
    assert result["structuredContent"] is None
    block = _text_block(result)
    assert block["type"] == "text"
    assert block["text"] == "unknown tool: missing"


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
    list_result = await registry.execute(
        ToolCall("list-1", "list", {"path": "."})
    )
    exec_result = await registry.execute(
        ToolCall("exec-1", "exec", {"command": "true"})
    )

    assert read_result["structuredContent"] == {
        "path": str(file_path),
        "sha256": hashlib.sha256(file_path.read_bytes()).hexdigest(),
        "line_count": 2,
    }
    assert list_result["structuredContent"] == {
        "root": str(tmp_path),
        "entry_count": 1,
        "full_size": 1,
        "truncated": False,
    }
    assert exec_result["structuredContent"] == {
        "exit_code": 0,
        "cwd": str(tmp_path),
    }
