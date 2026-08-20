import asyncio
import shlex
import sys
from pathlib import Path

import pytest

from zeta.fake import FakeBackend, ScriptedTurn
from zeta.loop import AgentLoop
from zeta.store import ConversationStore
from zeta.tools import ToolAbortSignal, ToolRegistry
from zeta.types import MessageRole, TextContent, ToolCall, ToolResult


def _python_command(source: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(source)}"


def _descendant_command(marker: Path, delay: float = 0.3) -> str:
    child = (
        "import pathlib,time; "
        f"time.sleep({delay}); pathlib.Path({str(marker)!r}).write_text('alive')"
    )
    parent = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}]); time.sleep(5)"
    )
    return _python_command(parent)


@pytest.mark.asyncio
async def test_registry_validates_arguments_before_running_handler(tmp_path: Path) -> None:
    called = False

    def handler(arguments: dict[str, object]) -> str:
        nonlocal called
        called = True
        return "ran"

    registry = ToolRegistry(tmp_path, register_builtin=False)
    registry.register(
        "typed",
        handler,
        parameters={
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
            "additionalProperties": False,
        },
    )

    result = await registry.execute(ToolCall("call-1", "typed", {"count": "one"}))

    assert result.is_error
    assert "invalid arguments" in result.content
    assert not called


def test_registry_rejects_unsupported_schema_constructs(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path, register_builtin=False)

    with pytest.raises(ValueError, match="unsupported schema type"):
        registry.register(
            "union",
            lambda arguments: "ran",
            parameters={"type": ["string", "null"]},
        )
    with pytest.raises(ValueError, match="unsupported schema keywords"):
        registry.register(
            "alternative",
            lambda arguments: "ran",
            parameters={
                "type": "string",
                "anyOf": [{"minLength": 2}],
            },
        )


@pytest.mark.asyncio
async def test_paths_outside_session_cwd_are_allowed(tmp_path: Path) -> None:
    outside = tmp_path.parent / "zeta-outside.txt"
    outside.write_text("outside", encoding="utf-8")
    outside_dir = tmp_path.parent / "zeta-outside-dir"
    outside_dir.mkdir()
    (outside_dir / "nested.txt").write_text("nested", encoding="utf-8")
    link = tmp_path / "outside-link"
    link.symlink_to(outside)
    dir_link = tmp_path / "outside-dir-link"
    dir_link.symlink_to(outside_dir, target_is_directory=True)
    registry = ToolRegistry(tmp_path)

    absolute_result = await registry.execute(
        ToolCall("read-1", "read", {"path": str(outside)})
    )
    parent_result = await registry.execute(
        ToolCall("list-1", "list", {"path": "../", "depth": 1})
    )
    symlink_result = await registry.execute(
        ToolCall("read-2", "read", {"path": "outside-link"})
    )
    symlink_dir_result = await registry.execute(
        ToolCall("list-2", "list", {"path": "outside-dir-link"})
    )

    assert absolute_result.content == "outside"
    assert "zeta-outside.txt" in parent_result.content
    assert symlink_result.content == "outside"
    assert "outside-dir-link/nested.txt" in symlink_dir_result.content


@pytest.mark.asyncio
async def test_builtin_tools_read_list_and_exec_use_session_cwd(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "note.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    registry = ToolRegistry(tmp_path)

    read_result = await registry.execute(
        ToolCall("read-1", "read", {"path": "nested/note.txt", "offset": 1, "limit": 1})
    )
    list_result = await registry.execute(
        ToolCall("list-1", "list", {"path": ".", "depth": 2})
    )
    exec_result = await registry.execute(
        ToolCall("exec-1", "exec", {"command": "pwd"})
    )

    assert read_result.content == "two"
    assert "nested/" in list_result.content
    assert "nested/note.txt" in list_result.content
    assert str(tmp_path) in exec_result.content
    assert "not a sandbox" in registry.schemas[2]["description"]


@pytest.mark.asyncio
async def test_exec_timeout_kills_and_reaps_descendants(tmp_path: Path) -> None:
    marker = tmp_path / "timeout-child-alive"
    registry = ToolRegistry(tmp_path)

    result = await registry.execute(
        ToolCall(
            "exec-timeout",
            "exec",
            {"command": _descendant_command(marker), "timeout": 0.05},
        )
    )
    await asyncio.sleep(0.4)

    assert result.is_error
    assert "command timed out" in result.content
    assert not marker.exists()


@pytest.mark.asyncio
async def test_exec_cancellation_kills_and_reaps_descendants(tmp_path: Path) -> None:
    marker = tmp_path / "cancel-child-alive"
    registry = ToolRegistry(tmp_path)
    task = asyncio.create_task(
        registry.execute(
            ToolCall(
                "exec-cancel",
                "exec",
                {"command": _descendant_command(marker), "timeout": 5},
            )
        )
    )
    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.4)

    assert not marker.exists()


@pytest.mark.asyncio
async def test_exec_abort_kills_process_group_and_returns_canceled_result(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "abort-child-alive"
    abort_signal = ToolAbortSignal()
    registry = ToolRegistry(tmp_path, abort_signal=abort_signal)
    task = asyncio.create_task(
        registry.execute(
            ToolCall(
                "exec-abort",
                "exec",
                {"command": _descendant_command(marker), "timeout": 5},
            )
        )
    )
    await asyncio.sleep(0.05)
    abort_signal.abort()

    result = await asyncio.wait_for(task, timeout=0.5)
    await asyncio.sleep(0.4)

    assert result == ToolResult("exec-abort", "tool execution canceled", True)
    assert not marker.exists()


@pytest.mark.asyncio
async def test_exec_output_cap_includes_final_content_boundary(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path)

    result = await registry.execute(
        ToolCall("exec-cap", "exec", {"command": "printf 1234567890", "max_output": 5})
    )

    assert not result.is_error
    assert len(result.content) == 5


@pytest.mark.asyncio
async def test_abort_cancels_calls_after_the_signal_is_set(tmp_path: Path) -> None:
    abort_signal = ToolAbortSignal()
    called: list[str] = []
    registry = ToolRegistry(
        tmp_path,
        abort_signal=abort_signal,
        register_builtin=False,
    )

    async def handler(
        arguments: dict[str, str],
        signal: ToolAbortSignal,
    ) -> str:
        called.append(arguments["value"])
        if arguments["value"] == "first":
            signal.abort()
        await asyncio.sleep(0)
        return arguments["value"]

    registry.register("step", handler)
    results = await registry.execute_many(
        [
            ToolCall("call-1", "step", {"value": "first"}),
            ToolCall("call-2", "step", {"value": "second"}),
        ]
    )

    assert [result.content for result in results] == ["first", "tool execution canceled"]
    assert called == ["first"]
    assert results[1].is_error


@pytest.mark.asyncio
async def test_parallel_safe_calls_overlap_and_keep_call_order(tmp_path: Path) -> None:
    finished: list[str] = []
    registry = ToolRegistry(tmp_path, register_builtin=False)

    async def worker(arguments: dict[str, str]) -> str:
        if arguments["value"] == "slow":
            await asyncio.sleep(0.03)
        finished.append(arguments["value"])
        return arguments["value"]

    registry.register(
        "work",
        worker,
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        parallel_safe=True,
    )
    results = await registry.execute_many(
        [
            ToolCall("call-1", "work", {"value": "slow"}),
            ToolCall("call-2", "work", {"value": "fast"}),
        ]
    )

    assert finished == ["fast", "slow"]
    assert [result.content for result in results] == ["slow", "fast"]


@pytest.mark.asyncio
async def test_pre_execution_hook_can_allow_and_deny(tmp_path: Path) -> None:
    seen: list[tuple[str, dict[str, object]]] = []

    def hook(name: str, arguments: dict[str, object]) -> bool:
        seen.append((name, arguments))
        return arguments.get("allow") is True

    registry = ToolRegistry(tmp_path, pre_execute_hook=hook, register_builtin=False)
    registry.register(
        "gated",
        lambda arguments: "allowed",
        parameters={
            "type": "object",
            "properties": {"allow": {"type": "boolean"}},
            "required": ["allow"],
        },
    )

    allowed = await registry.execute(ToolCall("call-1", "gated", {"allow": True}))
    denied = await registry.execute(ToolCall("call-2", "gated", {"allow": False}))

    assert allowed.content == "allowed"
    assert denied.is_error
    assert denied.content == "tool execution denied"
    assert seen == [("gated", {"allow": True}), ("gated", {"allow": False})]


@pytest.mark.asyncio
async def test_agent_loop_executes_tool_calls_through_registry(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("from registry", encoding="utf-8")
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[ToolCall("call-1", "read", {"path": "note.txt"})]),
            ScriptedTurn(content=[TextContent("done")]),
        ]
    )
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    registry = ToolRegistry(tmp_path)

    events = [
        event
        async for event in AgentLoop(backend, store, registry=registry).run_turn("read it")
    ]

    result_message = store.messages()[2]
    assert result_message.role is MessageRole.TOOL_RESULT
    assert result_message.tool_result is not None
    assert result_message.tool_result.content == "from registry"
    assert events[-1].type.value == "agent_end"
    assert backend.calls[0][1][0]["name"] == "read"


@pytest.mark.asyncio
async def test_agent_loop_mapping_tools_still_validate_through_registry(
    tmp_path: Path,
) -> None:
    called = False

    def typed(arguments: dict[str, object]) -> str:
        nonlocal called
        called = True
        return "ran"

    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[ToolCall("call-1", "typed", {"count": "bad"})]),
            ScriptedTurn(content=[TextContent("done")]),
        ]
    )
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    schema = {
        "name": "typed",
        "parameters": {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        },
    }

    [
        event
        async for event in AgentLoop(
            backend,
            store,
            tools={"typed": typed},
            tool_schemas=[schema],
        ).run_turn("go")
    ]

    result = store.messages()[2].tool_result
    assert result is not None and result.is_error
    assert "invalid arguments" in result.content
    assert not called
