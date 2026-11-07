from __future__ import annotations

import asyncio
import hashlib
import os
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
async def test_edit_rechecks_path_after_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("old content", encoding="utf-8")
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    digest_ready = threading.Event()
    release_digest = threading.Event()
    original_file_digest = hashlib.file_digest

    def blocked_file_digest(*args: Any, **kwargs: Any) -> Any:
        digest = original_file_digest(*args, **kwargs)
        digest_ready.set()
        release_digest.wait()
        return digest

    monkeypatch.setattr(hashlib, "file_digest", blocked_file_digest)
    operation = asyncio.create_task(_edit(registry, target.name))
    digest_wait = asyncio.create_task(asyncio.to_thread(digest_ready.wait))
    try:
        done, _pending = await asyncio.wait(
            {operation, digest_wait}, return_when=asyncio.FIRST_COMPLETED
        )
        if digest_wait in done:
            replacement = tmp_path / "replacement"
            replacement.write_text("concurrent content", encoding="utf-8")
            os.replace(replacement, target)
            release_digest.set()

        result = await operation

        assert result["isError"] is True
        assert "changed while edit was in progress" in result["content"][0]["text"]
        assert target.read_text(encoding="utf-8") == "concurrent content"
    finally:
        digest_ready.set()
        release_digest.set()
        digest_wait.cancel()
        await asyncio.gather(digest_wait, return_exceptions=True)
        await registry.close()


@pytest.mark.asyncio
async def test_write_does_not_publish_new_file_before_content_is_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.txt"
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    entered = threading.Event()
    release = threading.Event()
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
            return self._handle.write(content)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._handle, name)

    def blocking_fdopen(
        fd: int, mode: str = "r", *args: object, **kwargs: object
    ) -> Any:
        handle = original_fdopen(fd, mode, *args, **kwargs)
        return BlockingWriter(handle) if "w" in mode or "+" in mode else handle

    monkeypatch.setattr(os, "fdopen", blocking_fdopen)
    operation = asyncio.create_task(_write(registry, target.name))
    try:
        await asyncio.to_thread(entered.wait)

        assert not target.exists()

        release.set()
        result = await operation
        assert result["isError"] is False
        assert target.read_text(encoding="utf-8") == "new content"
    finally:
        release.set()
        await asyncio.gather(operation, return_exceptions=True)
        await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("invoke", [_write, _edit])
async def test_abort_waits_for_file_worker_and_prevents_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invoke: ToolInvocation,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("old content", encoding="utf-8")
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    entered = threading.Event()
    release = threading.Event()
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
            return self._handle.write(content)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._handle, name)

    def blocking_fdopen(
        fd: int, mode: str = "r", *args: object, **kwargs: object
    ) -> Any:
        handle = original_fdopen(fd, mode, *args, **kwargs)
        return BlockingWriter(handle) if "w" in mode or "+" in mode else handle

    monkeypatch.setattr(os, "fdopen", blocking_fdopen)
    operation = asyncio.create_task(invoke(registry, target.name))
    try:
        await asyncio.to_thread(entered.wait)
        registry.abort()
        loop = asyncio.get_running_loop()
        for _ in range(4):
            turn = loop.create_future()
            loop.call_soon(turn.set_result, None)
            await turn

        assert not operation.done()

        release.set()
        result = await operation
        assert result["isError"] is True
        assert result["content"][0]["text"] == "tool execution canceled"
        assert target.read_text(encoding="utf-8") == "old content"
    finally:
        release.set()
        await asyncio.gather(operation, return_exceptions=True)
        await registry.close()
