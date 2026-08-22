import os
from contextlib import contextmanager
from pathlib import Path

import pytest

import zeta.tools._sandbox as sandbox_module
import zeta.tools.read as read_module
from zeta.tools import ToolRegistry
from zeta.types import ToolCall


@pytest.mark.asyncio
async def test_write_rejects_hard_link_target(tmp_path: Path) -> None:
    outside = tmp_path / "victim"
    outside.write_text("outside", encoding="utf-8")
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    os.link(outside, sandbox / "target")

    result = await ToolRegistry(sandbox).execute(
        ToolCall("hard-link", "write", {"path": "target", "content": "changed"})
    )

    assert result["isError"] is True
    assert "multiple hard links" in result["content"][0]["text"]
    assert outside.read_text(encoding="utf-8") == "outside"


@pytest.mark.asyncio
async def test_read_allows_hard_link_target(tmp_path: Path) -> None:
    outside = tmp_path / "victim"
    outside.write_text("outside", encoding="utf-8")
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    os.link(outside, sandbox / "target")

    result = await ToolRegistry(sandbox).execute(
        ToolCall("hard-link-read", "read", {"path": "target"})
    )

    assert result["isError"] is False
    assert result["content"][0]["text"] == "outside"


@pytest.mark.asyncio
async def test_read_uses_verified_fd_after_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    target = sandbox / "target"
    target.write_text("original", encoding="utf-8")
    moved = sandbox / "moved"
    registry = ToolRegistry(sandbox)
    real_open_target = read_module.open_target

    @contextmanager
    def open_and_rename(
        registry: ToolRegistry,
        raw_path: str,
        *,
        flags: int,
        mode: int = 0o644,
        create_parents: bool = False,
    ):
        with real_open_target(
            registry,
            raw_path,
            flags=flags,
            mode=mode,
            create_parents=create_parents,
        ) as target_info:
            os.rename(target, moved)
            yield target_info

    def fail_path_reopen(*args: object, **kwargs: object) -> object:
        raise AssertionError("internal read reopened the target by path")

    monkeypatch.setattr(read_module, "open_target", open_and_rename)
    monkeypatch.setattr(read_module.Path, "open", fail_path_reopen)

    result = await registry.execute(
        ToolCall("read-after-rename", "read", {"path": "target"})
    )

    assert result["isError"] is False
    assert result["content"][0]["text"] == "original"
    assert result["structuredContent"]["path"] == str(target)
    monkeypatch.undo()
    assert moved.read_text(encoding="utf-8") == "original"


def test_path_from_fd_returns_absolute_path(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("content", encoding="utf-8")
    file_descriptor = os.open(target, os.O_RDONLY)
    try:
        assert sandbox_module._path_from_fd(file_descriptor) == str(target)
    finally:
        os.close(file_descriptor)


def test_ancestor_symlink_is_rejected(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "file").write_text("outside", encoding="utf-8")
    (sandbox / "dir").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="escaped sandbox"):
        with sandbox_module.open_target(
            ToolRegistry(sandbox),
            "dir/file",
            flags=os.O_RDONLY | os.O_CLOEXEC,
        ):
            pass


@pytest.mark.asyncio
async def test_post_walk_ancestry_check_rejects_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = tmp_path / "sandbox"
    inner = sandbox / "inner"
    inner.mkdir(parents=True)
    (inner / "target").write_text("inside", encoding="utf-8")
    moved = tmp_path / "inner-moved"
    registry = ToolRegistry(sandbox)
    real_verify = sandbox_module._verify_ancestry
    calls = 0

    def verify(file_descriptor: int, identity: tuple[int, int]) -> None:
        nonlocal calls
        real_verify(file_descriptor, identity)
        calls += 1
        if calls == 2:
            os.rename(inner, moved)

    monkeypatch.setattr(sandbox_module, "_verify_ancestry", verify)
    result = await registry.execute(
        ToolCall("rename-escape", "write", {"path": "inner/target", "content": "changed"})
    )

    assert result["isError"] is True
    assert "escaped sandbox" in result["content"][0]["text"]
    assert (moved / "target").read_text(encoding="utf-8") == "inside"
