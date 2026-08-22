import asyncio
import errno
import fcntl
import hashlib
import math
import os
import shlex
import shutil
import sys
import threading
from pathlib import Path

import pytest

import zeta.tools.exec as exec_module
import zeta.tools.read as read_module
import zeta.tools.write as write_module
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
    symlink_result = await registry.execute(
        ToolCall("read-2", "read", {"path": "outside-link"})
    )

    assert absolute_result["isError"] is False
    assert absolute_result["content"][0]["text"] == "outside"
    assert symlink_result["isError"] is False
    assert symlink_result["content"][0]["text"] == "outside"


@pytest.mark.asyncio
async def test_builtin_tools_read_and_exec_use_session_cwd(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "note.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    registry = ToolRegistry(tmp_path)

    read_result = await registry.execute(
        ToolCall("read-1", "read", {"path": "nested/note.txt", "offset": 1, "limit": 1})
    )
    exec_result = await registry.execute(
        ToolCall("exec-1", "exec", {"command": "pwd"})
    )

    assert read_result["isError"] is False
    assert read_result["content"][0]["text"] == "two"
    assert exec_result["isError"] is False
    assert str(tmp_path) in exec_result["content"][0]["text"]
    assert "not a sandbox" in registry.schemas[2]["description"]


@pytest.mark.asyncio
async def test_bash_captures_stdout_stderr_and_exit_code(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path)

    result = await registry.execute(
        ToolCall(
            "bash-output",
            "bash",
            {"cmd": "printf out; printf err >&2; exit 7"},
        )
    )

    assert result["isError"] is True
    assert result["structuredContent"] == {
        "stdout": "out",
        "stderr": "err",
        "exit_code": 7,
        "cwd_after": str(tmp_path),
    }
    assert "stdout:\nout\nstderr:\nerr" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_bash_persists_cwd_across_calls(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path)

    changed = await registry.execute(
        ToolCall("bash-cd", "bash", {"cmd": "cd /tmp"})
    )
    current = await registry.execute(ToolCall("bash-pwd", "bash", {"cmd": "pwd"}))

    assert changed["structuredContent"]["cwd_after"] == "/tmp"
    assert current["structuredContent"]["stdout"].strip() == "/tmp"
    assert registry.bash_cwd == "/tmp"


@pytest.mark.asyncio
async def test_bash_persistent_cwd_isolated_between_sessions(tmp_path: Path) -> None:
    first_store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    first = ToolRegistry(tmp_path, session_store=first_store)
    await first.execute(ToolCall("first-cd", "bash", {"cmd": "cd /tmp"}))

    second_store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    second = ToolRegistry(tmp_path, session_store=second_store)
    result = await second.execute(ToolCall("second-pwd", "bash", {"cmd": "pwd"}))

    assert result["structuredContent"]["stdout"].strip() == str(tmp_path)


@pytest.mark.asyncio
async def test_bash_cwd_channel_rejects_user_forgery(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path)

    result = await registry.execute(
        ToolCall(
            "bash-channel-attack",
            "bash",
            {"cmd": 'printf "/tmp\\n" > "$3"; trap - EXIT; exit 0'},
        )
    )

    assert result["isError"] or result["structuredContent"]["cwd_after"] == str(tmp_path)


@pytest.mark.asyncio
async def test_bash_failed_persistence_keeps_registry_state_on_replace_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    registry = ToolRegistry(tmp_path, session_store=store)
    before_state = store.state_path.read_bytes()

    def fail_replace(source: object, destination: object) -> None:
        del source, destination
        raise OSError("injected state replace failure")

    monkeypatch.setattr("zeta.core.store.os.replace", fail_replace)
    result = await registry.execute(
        ToolCall("bash-state-failure", "bash", {"cmd": "cd /tmp"})
    )

    assert result["isError"] is True
    assert result["structuredContent"] is None
    assert registry.bash_cwd == str(tmp_path)
    assert store.bash_cwd == str(tmp_path)
    assert store.state_path.read_bytes() == before_state

    monkeypatch.undo()
    current = await registry.execute(ToolCall("bash-state-old", "bash", {"cmd": "pwd"}))
    assert current["structuredContent"]["stdout"].strip() == str(tmp_path)


@pytest.mark.asyncio
async def test_bash_rejects_per_call_cwd_outside_sandbox(tmp_path: Path) -> None:
    marker = tmp_path / "should-not-run"
    registry = ToolRegistry(tmp_path)

    result = await registry.execute(
        ToolCall(
            "bash-outside-cwd",
            "bash",
            {"cmd": f"touch {shlex.quote(str(marker))}", "cwd": str(tmp_path.parent)},
        )
    )

    assert result["isError"] is True
    assert result["structuredContent"] is None
    assert not marker.exists()


@pytest.mark.asyncio
async def test_bash_persists_cwd_after_failed_command(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path)

    failed = await registry.execute(
        ToolCall("bash-failed-cd", "bash", {"cmd": "cd /tmp; exit 3"})
    )
    current = await registry.execute(ToolCall("bash-failed-pwd", "bash", {"cmd": "pwd"}))

    assert failed["isError"] is True
    assert failed["structuredContent"] == {
        "stdout": "",
        "stderr": "",
        "exit_code": 3,
        "cwd_after": "/tmp",
    }
    assert current["structuredContent"]["stdout"].strip() == "/tmp"


@pytest.mark.asyncio
async def test_bash_persists_explicit_cd_from_override(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    registry = ToolRegistry(tmp_path)

    result = await registry.execute(
        ToolCall("bash-explicit-cd", "bash", {"cmd": "cd /tmp", "cwd": str(nested)})
    )
    current = await registry.execute(ToolCall("bash-explicit-pwd", "bash", {"cmd": "pwd"}))

    assert result["structuredContent"]["cwd_after"] == "/tmp"
    assert current["structuredContent"]["stdout"].strip() == "/tmp"


@pytest.mark.asyncio
async def test_bash_abort_kills_process_group_and_reaps_descendants(tmp_path: Path) -> None:
    marker = tmp_path / "child-alive"
    abort_signal = ToolAbortSignal()
    registry = ToolRegistry(tmp_path, abort_signal=abort_signal)
    task = asyncio.create_task(
        registry.execute(
            ToolCall("bash-abort", "bash", {"cmd": _descendant_command(marker)})
        )
    )

    await asyncio.sleep(0.05)
    abort_signal.abort()
    result = await task
    await asyncio.sleep(0.5)

    assert result["isError"] is True
    assert result["content"][0]["text"] == "tool execution canceled"
    assert not marker.exists()


@pytest.mark.asyncio
async def test_bash_task_cancellation_kills_process_group(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "child-canceled"
    registry = ToolRegistry(tmp_path)
    task = asyncio.create_task(
        registry.execute(
            ToolCall("bash-cancel", "bash", {"cmd": _descendant_command(marker)})
        )
    )

    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.5)

    assert not marker.exists()


@pytest.mark.asyncio
async def test_bash_invalid_start_cwd_falls_back_without_persisting(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    registry = ToolRegistry(tmp_path)
    registry.bash_cwd = str(missing)

    result = await registry.execute(ToolCall("bash-missing-cwd", "bash", {"cmd": "pwd"}))

    assert result["isError"] is True
    assert result["structuredContent"]["cwd_after"] == str(missing)
    assert registry.bash_cwd == str(missing)


@pytest.mark.asyncio
async def test_bash_cwd_override_does_not_persist_without_cd(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    registry = ToolRegistry(tmp_path)

    result = await registry.execute(
        ToolCall("bash-override", "bash", {"cmd": "pwd", "cwd": str(nested)})
    )
    current = await registry.execute(ToolCall("bash-default", "bash", {"cmd": "pwd"}))

    assert result["structuredContent"]["cwd_after"] == str(nested)
    assert current["structuredContent"]["stdout"].strip() == str(tmp_path)
    assert registry.bash_cwd == str(tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"cmd": ""},
        {"cmd": 1},
        {"cmd": "pwd", "cwd": 1},
        {"cmd": "pwd", "extra": True},
    ],
)
async def test_bash_rejects_malformed_arguments(
    tmp_path: Path,
    arguments: dict[str, object],
) -> None:
    result = await ToolRegistry(tmp_path).execute(
        ToolCall("bash-invalid", "bash", arguments)
    )

    assert result["isError"] is True
    assert result["structuredContent"] is None


def test_list_is_not_registered(tmp_path: Path) -> None:
    assert "list" not in ToolRegistry(tmp_path).definitions_by_name


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
async def test_edit_replaces_unique_string_with_structured_result(tmp_path: Path) -> None:
    file_path = tmp_path / "note.txt"
    file_path.write_text("before: old\n", encoding="utf-8")
    registry = ToolRegistry(tmp_path)

    result = await registry.execute(
        ToolCall(
            "edit-unique",
            "edit",
            {"path": "note.txt", "old_string": "old", "new_string": "new"},
        )
    )

    updated = b"before: new\n"
    assert result["isError"] is False
    assert result["content"][0]["text"] == (
        f"edited {file_path}: 12 bytes → 12 bytes"
    )
    assert result["structuredContent"] == {
        "path": str(file_path),
        "bytes_before": 12,
        "bytes_after": 12,
        "sha256_after": hashlib.sha256(updated).hexdigest(),
    }
    assert file_path.read_bytes() == updated


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "old_string", "message"),
    [
        ("one", "missing", "old_string not found in note.txt"),
        (
            "old and old",
            "old",
            "old_string found 2 times in note.txt; must be unique",
        ),
    ],
)
async def test_edit_requires_one_match(
    tmp_path: Path,
    content: str,
    old_string: str,
    message: str,
) -> None:
    file_path = tmp_path / "note.txt"
    file_path.write_text(content, encoding="utf-8")
    registry = ToolRegistry(tmp_path)

    result = await registry.execute(
        ToolCall(
            "edit-match-count",
            "edit",
            {"path": "note.txt", "old_string": old_string, "new_string": "new"},
        )
    )

    assert result["isError"] is True
    assert result["content"][0]["text"] == message
    assert result["structuredContent"] is None
    assert file_path.read_text(encoding="utf-8") == content


@pytest.mark.asyncio
async def test_edit_rejects_overlapping_matches(tmp_path: Path) -> None:
    file_path = tmp_path / "note.txt"
    file_path.write_text("aaa", encoding="utf-8")
    registry = ToolRegistry(tmp_path)

    result = await registry.execute(
        ToolCall(
            "edit-overlap",
            "edit",
            {"path": "note.txt", "old_string": "aa", "new_string": "X"},
        )
    )

    assert result["isError"] is True
    assert result["content"][0]["text"] == (
        "old_string found 2 times in note.txt; must be unique"
    )
    assert file_path.read_text(encoding="utf-8") == "aaa"


@pytest.mark.asyncio
async def test_edit_preserves_utf8_and_reports_byte_lengths(tmp_path: Path) -> None:
    file_path = tmp_path / "unicode.txt"
    file_path.write_text("café: 世界\n", encoding="utf-8")
    registry = ToolRegistry(tmp_path)

    result = await registry.execute(
        ToolCall(
            "edit-unicode",
            "edit",
            {"path": "unicode.txt", "old_string": "世界", "new_string": "мир"},
        )
    )

    updated = "café: мир\n".encode()
    assert result["isError"] is False
    assert result["structuredContent"] == {
        "path": str(file_path),
        "bytes_before": len("café: 世界\n".encode()),
        "bytes_after": len(updated),
        "sha256_after": hashlib.sha256(updated).hexdigest(),
    }
    assert file_path.read_bytes() == updated


@pytest.mark.asyncio
async def test_edit_rejects_path_outside_session_cwd(tmp_path: Path) -> None:
    session_cwd = tmp_path / "session"
    session_cwd.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("old", encoding="utf-8")
    registry = ToolRegistry(session_cwd)

    result = await registry.execute(
        ToolCall(
            "edit-outside",
            "edit",
            {"path": str(outside), "old_string": "old", "new_string": "new"},
        )
    )

    assert result["isError"] is True
    assert "escaped sandbox" in result["content"][0]["text"]
    assert outside.read_text(encoding="utf-8") == "old"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {"path": "note.txt", "old_string": "old"},
        {"path": "note.txt", "new_string": "new"},
        {
            "path": "note.txt",
            "old_string": "old",
            "new_string": "new",
            "extra": True,
        },
        {"path": "note.txt", "old_string": 1, "new_string": "new"},
    ],
)
async def test_registry_rejects_malformed_edit_arguments(
    tmp_path: Path,
    arguments: dict[str, object],
) -> None:
    registry = ToolRegistry(tmp_path)

    result = await registry.execute(ToolCall("edit-invalid", "edit", arguments))

    assert result["isError"] is True
    assert "invalid arguments" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_write_getpath_failure_does_not_truncate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "getpath-failure.txt"
    target.write_text("original", encoding="utf-8")
    registry = ToolRegistry(tmp_path)

    def fail_getpath(file_descriptor: int) -> str:
        raise OSError("injected F_GETPATH failure")

    monkeypatch.setattr(write_module, "_path_from_fd", fail_getpath)
    result = await registry.execute(
        ToolCall("write-getpath-failure", "write", {"path": target.name, "content": "new"})
    )

    assert result["isError"] is True
    assert target.read_text(encoding="utf-8") == "original"


@pytest.mark.asyncio
async def test_write_fdopen_failure_closes_raw_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "fdopen-failure.txt"
    registry = ToolRegistry(tmp_path)
    raw_fds: list[int] = []

    def fail_fdopen(file_descriptor: int, mode: str) -> object:
        raw_fds.append(file_descriptor)
        raise OSError("injected fdopen failure")

    monkeypatch.setattr(write_module.os, "fdopen", fail_fdopen)
    result = await registry.execute(
        ToolCall("write-fdopen-failure", "write", {"path": target.name, "content": "x"})
    )

    assert result["isError"] is True
    assert len(raw_fds) == 1
    with pytest.raises(OSError) as error:
        fcntl.fcntl(raw_fds[0], fcntl.F_GETFD)
    assert error.value.errno == errno.EBADF


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
    assert result["structuredContent"] == {
        "path": str(target),
        "bytes_written": 1,
        "sha256": hashlib.sha256(b"x").hexdigest(),
        "was_created": True,
        "was_overwritten": False,
    }
    assert target.read_text(encoding="utf-8") == "x"
    assert target.parent.is_dir()


@pytest.mark.asyncio
async def test_write_race_reports_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "raced.txt"
    registry = ToolRegistry(tmp_path)
    original_open = write_module.os.open

    def racing_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if path == "raced.txt" and dir_fd is not None and flags & write_module.os.O_EXCL:
            creator = threading.Thread(
                target=lambda: target.write_bytes(b"concurrent")
            )
            creator.start()
            creator.join()
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(write_module.os, "open", racing_open)
    result = await registry.execute(
        ToolCall("write-race", "write", {"path": "raced.txt", "content": "x"})
    )

    assert result["isError"] is False
    assert result["structuredContent"] == {
        "path": str(target),
        "bytes_written": 1,
        "sha256": hashlib.sha256(b"x").hexdigest(),
        "was_created": False,
        "was_overwritten": True,
    }
    assert target.read_bytes() == b"x"


@pytest.mark.asyncio
async def test_write_overwrite_symlink_race_stays_in_sandbox(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    target = sandbox / "target"
    target.write_bytes(b"inside")
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside")
    stop = threading.Event()

    def swap_target() -> None:
        while not stop.is_set():
            try:
                target.unlink()
            except (FileNotFoundError, IsADirectoryError, PermissionError):
                pass
            try:
                target.symlink_to(outside)
            except FileExistsError:
                pass

    swapper = threading.Thread(target=swap_target)
    swapper.start()
    try:
        registry = ToolRegistry(tmp_path)
        for index in range(50):
            result = await registry.execute(
                ToolCall(
                    f"write-overwrite-race-{index}",
                    "write",
                    {"path": "sandbox/target", "content": "x"},
                )
            )
            if not result["isError"]:
                assert result["structuredContent"]["path"] == str(target)
            assert outside.read_bytes() == b"outside"
    finally:
        stop.set()
        swapper.join()
        if target.is_symlink():
            target.unlink()
        if not target.exists():
            target.write_bytes(b"inside")


@pytest.mark.asyncio
async def test_write_create_parents_symlink_race_stays_in_sandbox(
    tmp_path: Path,
) -> None:
    sandbox = tmp_path / "sandbox"
    intermediate = sandbox / "a" / "b"
    intermediate.mkdir(parents=True)
    outside = tmp_path / "outside"
    (outside / "c").mkdir(parents=True)
    outside_target = outside / "c" / "target"
    outside_target.write_bytes(b"outside")
    stop = threading.Event()

    def swap_intermediate() -> None:
        while not stop.is_set():
            shutil.rmtree(intermediate, ignore_errors=True)
            try:
                intermediate.symlink_to(outside, target_is_directory=True)
            except FileExistsError:
                pass
            try:
                intermediate.unlink()
            except (FileNotFoundError, IsADirectoryError, PermissionError):
                pass
            try:
                intermediate.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass

    swapper = threading.Thread(target=swap_intermediate)
    swapper.start()
    try:
        registry = ToolRegistry(tmp_path)
        for index in range(50):
            result = await registry.execute(
                ToolCall(
                    f"write-create-parents-race-{index}",
                    "write",
                    {
                        "path": "sandbox/a/b/c/target",
                        "content": "x",
                        "create_parents": True,
                    },
                )
            )
            if not result["isError"]:
                assert Path(result["structuredContent"]["path"]).is_relative_to(
                    sandbox
                )
            assert outside_target.read_bytes() == b"outside"
    finally:
        stop.set()
        swapper.join()


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
    assert "escaped sandbox" in result["content"][0]["text"]
    assert not outside.exists()


@pytest.mark.asyncio
async def test_write_rejects_replaced_session_cwd(tmp_path: Path) -> None:
    session_cwd = tmp_path / "session"
    session_cwd.mkdir()
    attack = tmp_path / "attack"
    attack.mkdir()
    registry = ToolRegistry(session_cwd)

    os.rename(session_cwd, tmp_path / "session-original")
    session_cwd.symlink_to(attack, target_is_directory=True)
    result = await registry.execute(
        ToolCall(
            "write-replaced-cwd",
            "write",
            {
                "path": "sandbox/file.txt",
                "content": "x",
                "create_parents": True,
            },
        )
    )

    assert result["isError"] is True
    assert "session cwd was replaced" in result["content"][0]["text"]
    assert not (attack / "sandbox" / "file.txt").exists()


@pytest.mark.asyncio
async def test_write_rejects_replaced_session_cwd_identity(tmp_path: Path) -> None:
    session_cwd = tmp_path / "session"
    session_cwd.mkdir()
    registry = ToolRegistry(session_cwd)

    os.rename(session_cwd, tmp_path / "session-original")
    session_cwd.mkdir()
    result = await registry.execute(
        ToolCall(
            "write-replaced-cwd-identity",
            "write",
            {
                "path": "sandbox/file.txt",
                "content": "x",
                "create_parents": True,
            },
        )
    )

    assert result["isError"] is True
    assert "session cwd was replaced" in result["content"][0]["text"]
    assert not (session_cwd / "sandbox" / "file.txt").exists()


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
    real_fdopen = read_module.os.fdopen
    open_count = 0

    def tracking_fdopen(
        file_descriptor: int,
        mode: str,
        *args: object,
        **kwargs: object,
    ) -> object:
        nonlocal open_count
        open_count += 1
        return real_fdopen(file_descriptor, mode, *args, **kwargs)

    monkeypatch.setattr(read_module.os, "fdopen", tracking_fdopen)
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
