from __future__ import annotations

import asyncio
import os
import stat
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Self

import pytest

import zeta.tools.edit as edit_module
from zeta.protocol.types import StructuredToolResult, ToolCall
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry

ToolInvocation = Callable[[ToolRegistry, str], Awaitable[StructuredToolResult]]


async def _write(registry: ToolRegistry, path: str) -> StructuredToolResult:
    return await registry.execute(
        ToolCall("write-atomic", "write", {"path": path, "content": "new content"})
    )


async def _edit(registry: ToolRegistry, path: str) -> StructuredToolResult:
    return await registry.execute(
        ToolCall(
            "edit-atomic",
            "edit",
            {"path": path, "old_string": "old", "new_string": "new"},
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("invoke", [_write, _edit])
async def test_file_work_does_not_block_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invoke: ToolInvocation,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("old content", encoding="utf-8")
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    entered = threading.Event()
    release = threading.Event()
    loop_progressed = threading.Event()
    wrote_after_progress: list[bool] = []
    main_thread = threading.get_ident()
    original_fdopen = os.fdopen

    class BlockingWriter:
        def __init__(self, handle: Any) -> None:
            self._handle = handle

        def __enter__(self) -> Self:
            self._handle.__enter__()
            return self

        def __exit__(self, *args: object) -> object:
            return self._handle.__exit__(*args)

        def write(self, content: bytes) -> int:
            entered.set()
            if threading.get_ident() != main_thread:
                release.wait()
            wrote_after_progress.append(loop_progressed.is_set())
            return self._handle.write(content)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._handle, name)

    def blocking_fdopen(
        fd: int, mode: str = "r", *args: object, **kwargs: object
    ) -> Any:
        handle = original_fdopen(fd, mode, *args, **kwargs)
        return BlockingWriter(handle) if "w" in mode or "+" in mode else handle

    monkeypatch.setattr(os, "fdopen", blocking_fdopen)

    async def prove_progress() -> None:
        await asyncio.to_thread(entered.wait)
        loop_progressed.set()
        release.set()

    try:
        progress = asyncio.create_task(prove_progress())
        result = await invoke(registry, target.name)
        await progress

        assert wrote_after_progress == [True]
        assert result["isError"] is False
    finally:
        release.set()
        await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("change_kind", ["content", "identity"])
async def test_edit_rejects_concurrent_change_before_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change_kind: str,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("old content", encoding="utf-8")
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    original_count = edit_module._count_overlapping
    changed = False

    def change_after_read(content: str, old_string: str) -> int:
        nonlocal changed
        if not changed:
            changed = True
            if change_kind == "content":
                target.write_text("concurrent content", encoding="utf-8")
            else:
                replacement = tmp_path / "replacement"
                replacement.write_text("concurrent content", encoding="utf-8")
                os.replace(replacement, target)
        return original_count(content, old_string)

    monkeypatch.setattr(edit_module, "_count_overlapping", change_after_read)

    try:
        result = await _edit(registry, target.name)

        assert result["isError"] is True
        assert "changed while edit was in progress" in result["content"][0]["text"]
        assert target.read_text(encoding="utf-8") == "concurrent content"
    finally:
        await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("invoke", [_write, _edit])
async def test_tools_atomically_replace_existing_file(
    tmp_path: Path,
    invoke: ToolInvocation,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("old content", encoding="utf-8")
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())

    try:
        with target.open("rb") as old_handle:
            result = await invoke(registry, target.name)
            new_content = target.read_bytes()
            old_content = old_handle.read()

        assert result["isError"] is False
        assert old_content == b"old content"
        assert new_content == b"new content"
    finally:
        await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("invoke", [_write, _edit])
async def test_tools_preserve_existing_file_mode(
    tmp_path: Path,
    invoke: ToolInvocation,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("old content", encoding="utf-8")
    target.chmod(0o640)
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())

    try:
        result = await invoke(registry, target.name)

        assert result["isError"] is False
        assert stat.S_IMODE(target.stat().st_mode) == 0o640
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_write_keeps_default_mode_for_new_file(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    reference.touch()
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())

    try:
        result = await _write(registry, "target.txt")

        assert result["isError"] is False
        assert stat.S_IMODE((tmp_path / "target.txt").stat().st_mode) == stat.S_IMODE(
            reference.stat().st_mode
        )
    finally:
        await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ["write", "edit"])
@pytest.mark.parametrize("target_kind", ["symlink", "fifo", "hardlink"])
async def test_tools_reject_unsafe_target_types(
    tmp_path: Path,
    tool: str,
    target_kind: str,
) -> None:
    target = tmp_path / "target"
    fifo_reader: int | None = None
    if target_kind == "symlink":
        source = tmp_path / "source"
        source.write_text("old content", encoding="utf-8")
        target.symlink_to(source)
    elif target_kind == "fifo":
        os.mkfifo(target)
        fifo_reader = os.open(target, os.O_RDONLY | os.O_NONBLOCK)
    else:
        source = tmp_path / "source"
        source.write_text("old content", encoding="utf-8")
        target.hardlink_to(source)

    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    try:
        result = await (
            _write(registry, target.name)
            if tool == "write"
            else _edit(registry, target.name)
        )

        assert result["isError"] is True
        if target_kind == "hardlink":
            assert "multiple hard links" in result["content"][0]["text"]
    finally:
        if fifo_reader is not None:
            os.close(fifo_reader)
        await registry.close()
