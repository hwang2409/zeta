from pathlib import Path

import pytest

import zeta.tools as tools_package
from zeta.tools import ToolRegistry
from zeta.types import ToolCall


def _use_tool_path(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setattr(tools_package, "__path__", [str(path)])


@pytest.mark.asyncio
async def test_registry_discovers_and_executes_tool_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_name = "fixture_discovered_tool"
    (tmp_path / f"{module_name}.py").write_text(
        """
async def handle(arguments, abort_signal):
    value = arguments[\"value\"]
    return {
        \"content\": [{
            \"type\": \"text\",
            \"text\": value,
            \"truncated\": False,
            \"full_size\": len(value.encode(\"utf-8\")),
        }],
        \"isError\": False,
        \"structuredContent\": {\"value\": value},
    }

def register(registry):
    registry.register(
        \"fixture\",
        handle,
        description=\"fixture tool\",
        parameters={
            \"type\": \"object\",
            \"properties\": {\"value\": {\"type\": \"string\"}},
            \"required\": [\"value\"],
            \"additionalProperties\": False,
        },
    )
""",
        encoding="utf-8",
    )
    _use_tool_path(monkeypatch, tmp_path)

    registry = ToolRegistry(tmp_path)
    result = await registry.execute(ToolCall("fixture-call", "fixture", {"value": "ok"}))

    assert result["isError"] is False
    assert result["content"][0]["text"] == "ok"
    assert result["structuredContent"] == {"value": "ok"}


def test_registry_ignores_helpers_and_discovers_in_name_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "z_tool.py").write_text(
        'def register(registry):\n    registry.register("z", lambda arguments: "z")\n',
        encoding="utf-8",
    )
    (tmp_path / "a_tool.py").write_text(
        'def register(registry):\n    registry.register("a", lambda arguments: "a")\n',
        encoding="utf-8",
    )
    (tmp_path / "helper.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "_helper.py").write_text(
        'def register(registry):\n    raise AssertionError("must be skipped")\n',
        encoding="utf-8",
    )
    _use_tool_path(monkeypatch, tmp_path)

    registry = ToolRegistry(tmp_path)

    assert [definition.name for definition in registry.definitions] == ["a", "z"]
    assert "helper" not in registry.definitions_by_name


def test_registry_names_malformed_tool_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_name = "malformed_discovered_tool"
    (tmp_path / f"{module_name}.py").write_text(
        "register = 42\n", encoding="utf-8"
    )
    _use_tool_path(monkeypatch, tmp_path)

    with pytest.raises(TypeError, match=module_name):
        ToolRegistry(tmp_path)
