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

    with (
        pytest.raises(ValueError, match="escaped session cwd"),
        sandbox_module.open_target(
            ToolRegistry(sandbox),
            "dir/file",
            flags=os.O_RDONLY | os.O_CLOEXEC,
        ),
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

    def verify(
        file_descriptor: int,
        identity: tuple[int, int],
        policy: sandbox_module.SandboxPolicy,
    ) -> None:
        nonlocal calls
        real_verify(file_descriptor, identity, policy)
        calls += 1
        if calls == 2:
            os.rename(inner, moved)

    monkeypatch.setattr(sandbox_module, "_verify_ancestry", verify)
    result = await registry.execute(
        ToolCall("rename-escape", "write", {"path": "inner/target", "content": "changed"})
    )

    assert result["isError"] is True
    assert "parent directory does not exist" in result["content"][0]["text"]
    assert (moved / "target").read_text(encoding="utf-8") == "inside"


def test_expand_user_path_expands_home() -> None:
    expanded = sandbox_module.expand_user_path("~/notes.txt")
    home = os.path.expanduser("~")
    assert expanded.startswith((home + os.sep, home))
    assert not expanded.startswith("~")


def test_expand_user_path_leaves_plain_paths_alone() -> None:
    assert sandbox_module.expand_user_path("relative/path") == "relative/path"
    assert sandbox_module.expand_user_path("/absolute/path") == "/absolute/path"


def test_expand_user_path_rejects_unknown_user() -> None:
    with pytest.raises(ValueError, match="use an absolute path"):
        sandbox_module.expand_user_path("~definitelynotarealuser-zeta68/x")


def test_sandbox_policy_resolves_tilde_and_classifies_roots(tmp_path: Path) -> None:
    policy = sandbox_module.SandboxPolicy(tmp_path)
    inside = policy.resolve("nested/file.txt")
    assert inside.in_cwd is True
    assert inside.absolute == tmp_path / "nested" / "file.txt"

    home_relative = policy.resolve("~/anywhere.txt")
    assert home_relative.absolute == Path(os.path.expanduser("~/anywhere.txt"))
    # Home is (almost always) outside the sandbox in tests.
    assert home_relative.in_cwd is False

    outside = policy.resolve(str(tmp_path.parent / "sibling.txt"))
    assert outside.in_cwd is False
    assert outside.absolute == tmp_path.parent / "sibling.txt"


def test_sandbox_policy_describe_roots_names_cwd(tmp_path: Path) -> None:
    policy = sandbox_module.SandboxPolicy(tmp_path)
    described = policy.describe_roots()
    assert str(tmp_path) in described
    assert "session cwd" in described


@pytest.mark.asyncio
async def test_write_expands_tilde_home_in_target_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    session_cwd = tmp_path / "session"
    session_cwd.mkdir()
    registry = ToolRegistry(session_cwd)

    result = await registry.execute(
        ToolCall("write-tilde", "write", {"path": "~/hello.txt", "content": "hi"})
    )

    assert result["isError"] is False, result["content"][0]["text"]
    assert (home / "hello.txt").read_text(encoding="utf-8") == "hi"
    assert result["structuredContent"]["path"] == str(home / "hello.txt")


@pytest.mark.asyncio
async def test_edit_expands_tilde_home_in_target_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    (home / "note.txt").write_text("old", encoding="utf-8")
    session_cwd = tmp_path / "session"
    session_cwd.mkdir()
    registry = ToolRegistry(session_cwd)

    result = await registry.execute(
        ToolCall(
            "edit-tilde",
            "edit",
            {"path": "~/note.txt", "old_string": "old", "new_string": "new"},
        )
    )

    assert result["isError"] is False, result["content"][0]["text"]
    assert (home / "note.txt").read_text(encoding="utf-8") == "new"


@pytest.mark.asyncio
async def test_read_expands_tilde_home_in_target_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    (home / "greeting.txt").write_text("hi", encoding="utf-8")
    session_cwd = tmp_path / "session"
    session_cwd.mkdir()
    registry = ToolRegistry(session_cwd)

    result = await registry.execute(
        ToolCall("read-tilde", "read", {"path": "~/greeting.txt"})
    )

    assert result["isError"] is False, result["content"][0]["text"]
    assert result["content"][0]["text"] == "hi"


@pytest.mark.asyncio
async def test_bash_cwd_expands_tilde_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    session_cwd = tmp_path / "session"
    session_cwd.mkdir()
    registry = ToolRegistry(session_cwd)

    result = await registry.execute(
        ToolCall("bash-tilde-cwd", "bash", {"cmd": "pwd", "cwd": "~"})
    )

    assert result["isError"] is False
    assert result["structuredContent"]["stdout"].strip() == str(home)
    assert result["structuredContent"]["cwd_after"] == str(home)


@pytest.mark.asyncio
async def test_write_cross_root_allow_regression_pin_for_audit(
    tmp_path: Path,
) -> None:
    """Audit round-3 shape: write to an absolute path outside session cwd."""

    session_cwd = tmp_path / "session"
    session_cwd.mkdir()
    outside_dir = tmp_path / "elsewhere"
    outside_dir.mkdir()
    target = outside_dir / "artifact.html"
    registry = ToolRegistry(session_cwd)

    result = await registry.execute(
        ToolCall(
            "write-cross-root",
            "write",
            {"path": str(target), "content": "<html></html>"},
        )
    )

    assert result["isError"] is False, result["content"][0]["text"]
    assert target.read_text(encoding="utf-8") == "<html></html>"


@pytest.mark.asyncio
async def test_write_cross_root_can_create_parents_outside_cwd(
    tmp_path: Path,
) -> None:
    session_cwd = tmp_path / "session"
    session_cwd.mkdir()
    target = tmp_path / "new" / "sub" / "artifact.txt"
    registry = ToolRegistry(session_cwd)

    result = await registry.execute(
        ToolCall(
            "write-cross-root-create",
            "write",
            {"path": str(target), "content": "hi", "create_parents": True},
        )
    )

    assert result["isError"] is False, result["content"][0]["text"]
    assert target.read_text(encoding="utf-8") == "hi"


@pytest.mark.asyncio
async def test_symlink_escape_error_names_session_cwd(tmp_path: Path) -> None:
    """Refusal error text must name the allowed root so the model can retry."""

    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "note").write_text("outside", encoding="utf-8")
    (sandbox / "dir").symlink_to(outside, target_is_directory=True)
    registry = ToolRegistry(sandbox)

    result = await registry.execute(
        ToolCall(
            "escape-error",
            "write",
            {"path": "dir/note", "content": "changed"},
        )
    )

    assert result["isError"] is True
    message = result["content"][0]["text"]
    assert "escaped session cwd" in message
    assert str(sandbox) in message
    assert "retry with an absolute path" in message
    assert (outside / "note").read_text(encoding="utf-8") == "outside"


@pytest.mark.asyncio
async def test_read_cross_root_absolute_path_regression_pin(
    tmp_path: Path,
) -> None:
    session_cwd = tmp_path / "session"
    session_cwd.mkdir()
    outside = tmp_path / "elsewhere.txt"
    outside.write_text("outside content", encoding="utf-8")
    registry = ToolRegistry(session_cwd)

    result = await registry.execute(
        ToolCall("read-cross-root", "read", {"path": str(outside)})
    )

    assert result["isError"] is False
    assert result["content"][0]["text"] == "outside content"
