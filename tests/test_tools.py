import asyncio
import hashlib
import math
import shlex
import sys
from pathlib import Path

import pytest

import zeta.tools.exec as exec_module
import zeta.tools.list as list_module
import zeta.tools.read as read_module
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.loop import AgentLoop
from zeta.core.store import ConversationStore
from zeta.tools import ToolAbortSignal, ToolRegistry
from zeta.types import MessageRole, StreamEventType, TextContent, ToolCall, ToolResult


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


async def _collect_loop(loop: AgentLoop) -> list[object]:
    return [event async for event in loop.run_turn("go")]


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

    assert result["isError"] is True
    assert "invalid arguments" in result["content"][0]["text"]
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
    with pytest.raises(ValueError, match="schema must contain JSON data"):
        registry.register(
            "non-json",
            lambda arguments: "ran",
            parameters={"type": "object", "const": object()},
        )
    with pytest.raises(ValueError, match="schema must contain JSON data"):
        registry.register(
            "tuple",
            lambda arguments: "ran",
            parameters={"type": "object", "properties": {"value": {"enum": [("x",)]}}},
        )


@pytest.mark.asyncio
async def test_paths_outside_session_cwd_are_allowed(tmp_path: Path) -> None:
    session_cwd = tmp_path / "session"
    session_cwd.mkdir()
    outside = tmp_path / "zeta-outside.txt"
    outside.write_text("outside", encoding="utf-8")
    outside_dir = tmp_path / "zeta-outside-dir"
    outside_dir.mkdir()
    (outside_dir / "nested.txt").write_text("nested", encoding="utf-8")
    link = session_cwd / "outside-link"
    link.symlink_to(outside)
    dir_link = session_cwd / "outside-dir-link"
    dir_link.symlink_to(outside_dir, target_is_directory=True)
    registry = ToolRegistry(session_cwd)

    absolute_result = await registry.execute(
        ToolCall("read-1", "read", {"path": str(outside)})
    )
    # See ZETA-P2 for the underlying list-cap fragility this rewrite works around.
    parent_result = await registry.execute(
        ToolCall("list-1", "list", {"path": "../", "depth": 1})
    )
    symlink_result = await registry.execute(
        ToolCall("read-2", "read", {"path": "outside-link"})
    )
    symlink_dir_result = await registry.execute(
        ToolCall("list-2", "list", {"path": "outside-dir-link"})
    )

    assert absolute_result["isError"] is False
    assert absolute_result["content"][0]["text"] == "outside"
    assert parent_result["isError"] is False
    assert "zeta-outside.txt" in parent_result["content"][0]["text"]
    assert symlink_result["isError"] is False
    assert symlink_result["content"][0]["text"] == "outside"
    assert symlink_dir_result["isError"] is False
    assert "outside-dir-link/nested.txt" in symlink_dir_result["content"][0]["text"]


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

    assert read_result["isError"] is False
    assert read_result["content"][0]["text"] == "two"
    assert list_result["isError"] is False
    assert "nested/" in list_result["content"][0]["text"]
    assert "nested/note.txt" in list_result["content"][0]["text"]
    assert exec_result["isError"] is False
    assert str(tmp_path) in exec_result["content"][0]["text"]
    assert "not a sandbox" in registry.schemas[2]["description"]


@pytest.mark.asyncio
async def test_write_creates_file_with_structured_result(tmp_path: Path) -> None:
    content = "héllo\n"
    registry = ToolRegistry(tmp_path)

    result = await registry.execute(
        ToolCall("write-new", "write", {"path": "note.txt", "content": content})
    )

    file_path = tmp_path / "note.txt"
    assert result["isError"] is False
    assert result["content"][0]["text"] == f"wrote 7 bytes to {file_path}"
    assert result["structuredContent"] == {
        "path": str(file_path),
        "bytes_written": 7,
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "was_created": True,
        "was_overwritten": False,
    }
    assert file_path.read_bytes() == content.encode("utf-8")


@pytest.mark.asyncio
async def test_write_reports_overwrite(tmp_path: Path) -> None:
    file_path = tmp_path / "note.txt"
    file_path.write_text("old", encoding="utf-8")
    registry = ToolRegistry(tmp_path)

    result = await registry.execute(
        ToolCall("write-overwrite", "write", {"path": "note.txt", "content": "new"})
    )

    assert result["isError"] is False
    assert result["structuredContent"] == {
        "path": str(file_path),
        "bytes_written": 3,
        "sha256": hashlib.sha256(b"new").hexdigest(),
        "was_created": False,
        "was_overwritten": True,
    }
    assert file_path.read_text(encoding="utf-8") == "new"


@pytest.mark.asyncio
async def test_write_rejects_missing_parent_by_default(tmp_path: Path) -> None:
    target = tmp_path / "missing" / "note.txt"
    registry = ToolRegistry(tmp_path)

    result = await registry.execute(
        ToolCall("write-missing-parent", "write", {"path": str(target), "content": "x"})
    )

    assert result["isError"] is True
    assert str(target.parent) in result["content"][0]["text"]
    assert not target.parent.exists()
    assert not target.exists()


@pytest.mark.asyncio
async def test_write_can_create_missing_parents(tmp_path: Path) -> None:
    target = tmp_path / "missing" / "note.txt"
    registry = ToolRegistry(tmp_path)

    result = await registry.execute(
        ToolCall(
            "write-create-parent",
            "write",
            {"path": str(target), "content": "x", "create_parents": True},
        )
    )

    assert result["isError"] is False
    assert target.read_text(encoding="utf-8") == "x"
    assert target.parent.is_dir()


@pytest.mark.asyncio
async def test_write_rejects_path_outside_session_cwd(tmp_path: Path) -> None:
    session_cwd = tmp_path / "session"
    session_cwd.mkdir()
    outside = tmp_path / "outside.txt"
    registry = ToolRegistry(session_cwd)

    result = await registry.execute(
        ToolCall("write-outside", "write", {"path": str(outside), "content": "x"})
    )

    assert result["isError"] is True
    assert "outside session cwd" in result["content"][0]["text"]
    assert not outside.exists()


@pytest.mark.asyncio
async def test_write_rejects_invalid_utf8_content_before_writing(
    tmp_path: Path,
) -> None:
    target = tmp_path / "invalid.txt"
    registry = ToolRegistry(tmp_path)

    result = await registry.execute(
        ToolCall(
            "write-invalid-utf8", "write", {"path": str(target), "content": "\ud800"}
        )
    )

    assert result["isError"] is True
    assert "valid UTF-8" in result["content"][0]["text"]
    assert not target.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {"path": "note.txt"},
        {"path": "note.txt", "content": "x", "create_parents": "yes"},
        {"path": "note.txt", "content": "x", "extra": True},
    ],
)
async def test_registry_rejects_malformed_write_arguments(
    tmp_path: Path,
    arguments: dict[str, object],
) -> None:
    registry = ToolRegistry(tmp_path)

    result = await registry.execute(ToolCall("write-invalid", "write", arguments))

    assert result["isError"] is True
    assert "invalid arguments" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_exec_retains_only_bounded_output_from_large_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captures = []
    real_capture = exec_module._BoundedOutput

    class TrackingCapture(real_capture):
        def __init__(self, limit: int) -> None:
            super().__init__(limit)
            captures.append(self)

    monkeypatch.setattr(exec_module, "_BoundedOutput", TrackingCapture)
    registry = ToolRegistry(tmp_path)
    result = await registry.execute(
        ToolCall(
            "exec-large",
            "exec",
            {
                "command": _python_command(
                    "import sys; sys.stdout.write('x' * 2000000)"
                ),
                "max_output": 64,
            },
        )
    )

    assert result["isError"] is False
    assert len(result["content"][0]["text"]) == 64
    assert result["content"][0]["truncated"] is True
    assert len(captures) == 2
    assert all(capture.retained_bytes <= 64 for capture in captures)
    assert sum(capture.retained_bytes for capture in captures) <= 128


@pytest.mark.asyncio
async def test_exec_full_size_is_stable_for_capped_utf8_output(tmp_path: Path) -> None:
    command = _python_command("import sys; sys.stdout.write('é')")
    uncapped = await ToolRegistry(tmp_path).execute(
        ToolCall("exec-utf8-full", "exec", {"command": command})
    )
    capped = await ToolRegistry(tmp_path).execute(
        ToolCall(
            "exec-utf8-capped",
            "exec",
            {"command": command, "max_output": 1},
        )
    )

    uncapped_block = uncapped["content"][0]
    capped_block = capped["content"][0]
    assert uncapped_block["full_size"] == capped_block["full_size"]
    assert uncapped_block["full_size"] == len(uncapped_block["text"].encode("utf-8"))


@pytest.mark.asyncio
async def test_read_retains_only_bounded_output_from_large_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captures = []
    real_capture = read_module._BoundedText

    class TrackingCapture(real_capture):
        def __init__(self, limit: int) -> None:
            super().__init__(limit)
            captures.append(self)

    monkeypatch.setattr(read_module, "_BoundedText", TrackingCapture)
    (tmp_path / "large.txt").write_text("x\n" * 1_000_000, encoding="utf-8")
    real_open = Path.open
    open_count = 0

    def tracking_open(
        file_path: Path,
        *args: object,
        **kwargs: object,
    ) -> object:
        nonlocal open_count
        if file_path == tmp_path / "large.txt":
            open_count += 1
        return real_open(file_path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "open", tracking_open)
    registry = ToolRegistry(tmp_path, max_output_chars=64)

    result = await registry.execute(
        ToolCall("read-large", "read", {"path": "large.txt"})
    )

    assert result["isError"] is False
    assert len(result["content"][0]["text"]) == 64
    assert result["content"][0]["truncated"] is True
    assert len(captures) == 1
    assert captures[0].retained_chars <= 64
    assert open_count == 1


@pytest.mark.asyncio
async def test_list_retains_only_bounded_output_from_large_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captures = []
    real_capture = list_module._BoundedText

    class TrackingCapture(real_capture):
        def __init__(self, limit: int) -> None:
            super().__init__(limit)
            captures.append(self)

    monkeypatch.setattr(list_module, "_BoundedText", TrackingCapture)
    for index in range(1_000):
        (tmp_path / f"file-{index:04d}.txt").write_text("x", encoding="utf-8")
    registry = ToolRegistry(tmp_path, max_output_chars=64)

    result = await registry.execute(ToolCall("list-large", "list", {"path": "."}))

    assert result["isError"] is False
    assert len(result["content"][0]["text"]) == 64
    assert result["content"][0]["truncated"] is True
    assert len(captures) == 1
    assert captures[0].retained_chars <= 64


@pytest.mark.asyncio
async def test_list_bounds_directory_working_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for index in range(5_000):
        (tmp_path / f"file-{index:04d}.txt").write_text("x", encoding="utf-8")
    calls: list[int] = []
    real_nsmallest = list_module.heapq.nsmallest

    def tracking_nsmallest(
        count: int,
        iterable: object,
        *,
        key: object,
    ) -> list[Path]:
        calls.append(count)
        return real_nsmallest(count, iterable, key=key)  # type: ignore[arg-type]

    monkeypatch.setattr(list_module.heapq, "nsmallest", tracking_nsmallest)
    registry = ToolRegistry(tmp_path, max_output_chars=32)

    result = await registry.execute(ToolCall("list-wide", "list", {"path": "."}))

    assert result["isError"] is False
    assert len(result["content"][0]["text"]) == 32
    assert result["content"][0]["truncated"] is True
    assert calls == [32]
    full_listing = "\n".join(f"file-{index:04d}.txt" for index in range(5_000))
    assert result["content"][0]["full_size"] == len(full_listing.encode("utf-8"))
    assert result["structuredContent"] == {
        "root": str(tmp_path),
        "entry_count": 5_000,
        "full_size": result["content"][0]["full_size"],
        "truncated": result["content"][0]["truncated"],
    }


@pytest.mark.asyncio
async def test_list_truncation_metadata_matches_capped_content(tmp_path: Path) -> None:
    for index in range(100):
        (tmp_path / f"file-{index:03d}.txt").write_text("x", encoding="utf-8")
    registry = ToolRegistry(tmp_path, max_output_chars=32)

    result = await registry.execute(ToolCall("list-capped", "list", {"path": "."}))

    block = result["content"][0]
    structured = result["structuredContent"]
    assert block["truncated"] is True
    assert structured["truncated"] is True
    assert structured["full_size"] == block["full_size"]


@pytest.mark.asyncio
async def test_list_truncation_metadata_matches_full_content(tmp_path: Path) -> None:
    for index in range(601):
        directory = tmp_path / f"directory-{index:03d}"
        directory.mkdir()
        (directory / "file.txt").write_text("x", encoding="utf-8")
    registry = ToolRegistry(tmp_path, max_output_chars=100_000)

    result = await registry.execute(
        ToolCall("list-complete", "list", {"path": ".", "depth": 2})
    )

    block = result["content"][0]
    structured = result["structuredContent"]
    assert structured["entry_count"] == 1_202
    assert block["truncated"] is False
    assert structured["truncated"] is False
    assert structured["full_size"] == block["full_size"]


@pytest.mark.asyncio
async def test_list_abort_returns_canceled_result_during_traversal(tmp_path: Path) -> None:
    for index in range(256):
        (tmp_path / f"file-{index:03d}.txt").write_text("x", encoding="utf-8")
    abort_signal = ToolAbortSignal()
    registry = ToolRegistry(tmp_path, abort_signal=abort_signal)

    async def abort_soon() -> None:
        await asyncio.sleep(0)
        abort_signal.abort()

    abort_task = asyncio.create_task(abort_soon())
    result = await registry.execute(ToolCall("list-abort", "list", {"depth": 1}))
    await abort_task

    assert result["isError"] is True
    assert result["content"][0]["text"] == "tool execution canceled"


@pytest.mark.asyncio
async def test_read_abort_returns_canceled_result_during_scan(tmp_path: Path) -> None:
    (tmp_path / "large.txt").write_text("x\n" * 1_000_000, encoding="utf-8")
    abort_signal = ToolAbortSignal()
    registry = ToolRegistry(
        tmp_path,
        abort_signal=abort_signal,
        max_output_chars=3_000_000,
    )

    async def abort_soon() -> None:
        await asyncio.sleep(0)
        abort_signal.abort()

    abort_task = asyncio.create_task(abort_soon())
    result = await registry.execute(ToolCall("read-abort", "read", {"path": "large.txt"}))
    await abort_task

    assert result["isError"] is True
    assert result["content"][0]["text"] == "tool execution canceled"


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

    assert result["isError"] is True
    assert "command timed out" in result["content"][0]["text"]
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

    assert result["isError"] is True
    assert result["content"][0]["text"] == "tool execution canceled"
    assert not marker.exists()


@pytest.mark.asyncio
async def test_exec_output_cap_includes_final_content_boundary(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path)

    result = await registry.execute(
        ToolCall("exec-cap", "exec", {"command": "printf 1234567890", "max_output": 5})
    )

    assert result["isError"] is False
    assert len(result["content"][0]["text"]) == 5
    assert result["content"][0]["truncated"] is True


@pytest.mark.asyncio
async def test_exec_abort_wins_when_completion_and_abort_are_ready_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    abort_signal = ToolAbortSignal()
    registry = ToolRegistry(tmp_path, abort_signal=abort_signal)
    real_wait = exec_module.asyncio.wait

    async def forced_tie(tasks, *, return_when):
        await asyncio.sleep(0.1)
        abort_signal.abort()
        await asyncio.sleep(0)
        task_set = set(tasks)
        done = {task for task in task_set if task.done()}
        if len(done) < 2:
            return await real_wait(task_set, return_when=return_when)
        return done, task_set - done

    monkeypatch.setattr(exec_module.asyncio, "wait", forced_tie)
    result = await registry.execute(
        ToolCall(
            "exec-race",
            "exec",
            {"command": _python_command("import time; time.sleep(0.01)")},
        )
    )

    assert result["isError"] is True
    assert result["content"][0]["text"] == "tool execution canceled"


@pytest.mark.asyncio
async def test_argument_finiteness_covers_undeclared_and_default_fields(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(tmp_path, register_builtin=False)
    registry.register(
        "permissive",
        lambda arguments: "ran",
        parameters={"type": "object"},
    )
    registry.register("default", lambda arguments: "ran")

    undeclared = await registry.execute(
        ToolCall("undeclared", "permissive", {"extra": math.nan})
    )
    default = await registry.execute(
        ToolCall("default", "default", {"extra": math.inf})
    )

    assert undeclared["isError"] is True
    assert default["isError"] is True


@pytest.mark.asyncio
async def test_registered_schema_copies_cannot_disable_validation(tmp_path: Path) -> None:
    called = False

    def handler(arguments: dict[str, object]) -> str:
        nonlocal called
        called = True
        return "ran"

    registry = ToolRegistry(tmp_path, register_builtin=False)
    definition = registry.register(
        "typed",
        handler,
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    )
    definition.parameters.clear()
    registry.definitions_by_name["typed"].parameters["properties"].clear()
    registry.schemas[0]["parameters"]["properties"].clear()

    result = await registry.execute(ToolCall("typed", "typed", {}))

    assert result["isError"] is True
    assert "required" in result["content"][0]["text"]
    assert not called


@pytest.mark.asyncio
async def test_numeric_validation_rejects_nonfinite_and_bool_enum_values(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(tmp_path, register_builtin=False)
    registry.register(
        "number",
        lambda arguments: "number",
        parameters={
            "type": "object",
            "properties": {
                "value": {"type": "number", "minimum": 0, "maximum": 10}
            },
        },
    )
    registry.register(
        "enum",
        lambda arguments: "enum",
        parameters={
            "type": "object",
            "properties": {"value": {"enum": [0, 1]}},
        },
    )

    nan_result = await registry.execute(
        ToolCall("nan", "number", {"value": math.nan})
    )
    bool_result = await registry.execute(
        ToolCall("bool", "enum", {"value": True})
    )
    int_result = await registry.execute(
        ToolCall("int", "enum", {"value": 1})
    )
    assert nan_result["isError"] is True
    assert bool_result["isError"] is True
    assert int_result["content"][0]["text"] == "enum"


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

    assert [result["content"][0]["text"] for result in results] == [
        "first",
        "tool execution canceled",
    ]
    assert called == ["first"]
    assert results[1]["isError"] is True


@pytest.mark.asyncio
async def test_abort_signal_stays_set_for_an_active_handler(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path, register_builtin=False)
    started = asyncio.Event()
    observed: list[bool] = []

    async def handler(
        arguments: dict[str, object],
        abort_signal: ToolAbortSignal,
    ) -> str:
        del arguments
        started.set()
        await abort_signal.wait()
        observed.append(abort_signal.is_set())
        await asyncio.sleep(0)
        observed.append(abort_signal.is_set())
        return "canceled"

    registry.register("wait", handler)
    task = asyncio.create_task(registry.execute(ToolCall("active", "wait", {})))
    await asyncio.wait_for(started.wait(), timeout=1)

    registry.abort()

    assert await asyncio.wait_for(task, timeout=1) == {
        "content": [
            {
                "type": "text",
                "text": "canceled",
                "truncated": False,
                "full_size": 8,
            }
        ],
        "isError": False,
        "structuredContent": None,
    }
    assert observed == [True, True]


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
    assert [result["content"][0]["text"] for result in results] == ["slow", "fast"]


@pytest.mark.asyncio
async def test_execute_many_abort_cancels_every_parallel_handler(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(tmp_path, register_builtin=False)
    calls = [ToolCall("parallel-a", "wait", {}), ToolCall("parallel-b", "wait", {})]
    started = asyncio.Event()
    started_count = 0
    generations: list[int] = []

    async def handler(
        arguments: dict[str, object],
        abort_signal: ToolAbortSignal,
    ) -> str:
        nonlocal started_count
        del arguments
        started_count += 1
        generations.append(abort_signal.generation)
        if started_count == len(calls):
            started.set()
        await abort_signal.wait()
        return "canceled"

    registry.register("wait", handler, parallel_safe=True)
    task = asyncio.create_task(registry.execute_many(calls))
    await asyncio.wait_for(started.wait(), timeout=1)

    registry.abort()

    assert await asyncio.wait_for(task, timeout=1) == [
        {
            "content": [
                {
                    "type": "text",
                    "text": "canceled",
                    "truncated": False,
                    "full_size": 8,
                }
            ],
            "isError": False,
            "structuredContent": None,
        }
        for _ in calls
    ]
    assert generations == [generations[0], generations[0]]


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

    assert allowed["isError"] is False
    assert allowed["content"][0]["text"] == "allowed"
    assert denied["isError"] is True
    assert denied["content"][0]["text"] == "tool execution denied"
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


@pytest.mark.asyncio
async def test_agent_loop_mapping_tools_do_not_expose_builtins(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall("call-1", "exec", {"command": "printf unsafe"})
                ]
            ),
            ScriptedTurn(content=[TextContent("done")]),
        ]
    )
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)

    await _collect_loop(
        AgentLoop(
            backend,
            store,
            tools={"safe_only": lambda arguments: "safe"},
            tool_schemas=[{"name": "safe_only"}],
        )
    )

    result = store.messages()[2].tool_result
    assert result is not None
    assert result.content == "unknown tool: exec"
    assert result.is_error


@pytest.mark.asyncio
async def test_agent_loop_keeps_boundary_abort_for_pending_tools(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[ToolCall("call-1", "step", {})]),
            ScriptedTurn(content=[TextContent("done")]),
        ]
    )
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    registry = ToolRegistry(tmp_path, register_builtin=False)
    registry.register("step", lambda arguments: "ran")
    aborted = False

    async for event in AgentLoop(backend, store, registry=registry).run_turn("go"):
        if event.type is StreamEventType.MESSAGE_END and not aborted:
            registry.abort()
            aborted = True

    result = store.messages()[2].tool_result
    assert result is not None
    assert result == ToolResult("call-1", "tool execution canceled", True)


@pytest.mark.asyncio
async def test_agent_loop_refreshes_abort_signal_each_turn(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall("call-1", "step", {"value": "abort"}),
                    ToolCall("call-2", "step", {"value": "canceled"}),
                ]
            ),
            ScriptedTurn(tool_calls=[ToolCall("call-3", "step", {"value": "next"})]),
            ScriptedTurn(content=[TextContent("done")]),
        ]
    )
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)

    async def step(
        arguments: dict[str, str],
        abort_signal: ToolAbortSignal,
    ) -> str:
        if arguments["value"] == "abort":
            abort_signal.abort()
        return arguments["value"]

    await _collect_loop(
        AgentLoop(backend, store, tools={"step": step})
    )

    results = [message.tool_result for message in store.messages() if message.tool_result]
    assert [result.content for result in results] == [
        "abort",
        "tool execution canceled",
        "next",
    ]


@pytest.mark.asyncio
async def test_registry_abort_cancels_loop_batch_and_next_tool(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall("call-1", "exec", {"command": "sleep 5"}),
                    ToolCall("call-2", "read", {"path": "missing.txt"}),
                ]
            ),
            ScriptedTurn(content=[TextContent("done")]),
        ]
    )
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    started = asyncio.Event()

    def hook(name: str, arguments: dict[str, object]) -> bool:
        if name == "exec":
            started.set()
        return True

    registry = ToolRegistry(tmp_path, pre_execute_hook=hook)
    task = asyncio.create_task(_collect_loop(AgentLoop(backend, store, registry=registry)))
    await started.wait()
    await asyncio.sleep(0.05)
    registry.abort()
    await task

    results = [message.tool_result for message in store.messages() if message.tool_result]
    assert [result.content for result in results] == [
        "tool execution canceled",
        "tool execution canceled",
    ]
