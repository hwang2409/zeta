"""event-hook controls for native extensions."""

from __future__ import annotations

import asyncio
import shlex
import sys
from pathlib import Path

import pytest

from zeta.core.hooks import load_hooks
from zeta.extensions.api import ExtensionAPI, ExtensionError
from zeta.extensions.hooks import (
    CompositeToolHook,
    ExtensionHook,
    run_extension_hooks,
)
from zeta.extensions.host import LoadingExtension
from zeta.extensions.manifest import parse_manifest
from zeta.protocol.types import ToolCall
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry


class _TrackingOwner:
    def __init__(self) -> None:
        self.tasks: set[asyncio.Task[object]] = set()

    def __call__(self, coroutine: object) -> asyncio.Task[object]:
        task = asyncio.create_task(coroutine)  # type: ignore[arg-type]
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    async def close(self) -> None:
        for task in tuple(self.tasks):
            task.cancel()
        await asyncio.wait_for(
            asyncio.gather(*self.tasks, return_exceptions=True), 10
        )


def _task_owner(coroutine: object) -> asyncio.Task[object]:
    return asyncio.create_task(coroutine)  # type: ignore[arg-type]


def _loading() -> LoadingExtension:
    manifest = parse_manifest(
        '[extension]\nname = "hooks"\nversion = "1"\napi_version = 1\nentry = "m:activate"\n'
    )
    return LoadingExtension(manifest, "user", Path("/extensions/hooks"))


class _Host:
    home = Path.home()
    project_dir = None


def test_hooks_reject_sync_handlers_at_registration() -> None:
    extension = _loading()
    api = ExtensionAPI(_Host(), extension)  # type: ignore[arg-type]
    with pytest.raises(ExtensionError, match="must be an async function"):
        api.on("pre_tool", lambda _tool, _args: None)


@pytest.mark.asyncio
async def test_pre_tool_denies_and_ignores_returned_replacement() -> None:
    async def replacement(_tool: str, args: dict[str, object]) -> dict[str, object]:
        args["mutated_copy"] = True
        return {"arguments": {**args, "safe": True}}

    async def deny(_tool: str, _args: dict[str, object]) -> str:
        return "blocked"

    notices: list[str] = []
    args: dict[str, object] = {"safe": False}
    hook = CompositeToolHook(
        None,
        [
            ExtensionHook("hooks", "pre_tool", replacement),
            ExtensionHook("hooks", "pre_tool", deny),
        ],
        notices.append,
        _task_owner,
    )
    assert await hook("bash", args) == "blocked"
    assert args == {"safe": False}
    assert len(notices) == 1
    assert "replacement arguments; ignored" in notices[0]


@pytest.mark.asyncio
async def test_tool_hook_composes_file_hooks_before_extensions_and_denies(tmp_path: Path) -> None:
    import json

    marker = tmp_path / "order"
    source = f"from pathlib import Path; Path({str(marker)!r}).open('a').write('file')"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(source)}"
    (tmp_path / "hooks.toml").write_text(
        "[[hook]]\nevent = \"pre_tool\"\ncommand = "
        f"{json.dumps(command)}\n"
        "[[hook]]\nevent = \"pre_tool\"\ncommand = "
        f"{json.dumps(command)}\n",
        encoding="utf-8",
    )
    manager = load_hooks(tmp_path)
    events: list[str] = []

    async def observe(_tool: str, _args: dict[str, object]) -> None:
        events.append("extension-1")

    async def observe_again(_tool: str, _args: dict[str, object]) -> None:
        events.append("extension-2")

    def capture(_arguments: dict[str, object]) -> str:
        events.append("tool")
        return "ran"

    def compose(base: object, hooks: list[ExtensionHook]) -> ToolRegistry:
        registry = ToolRegistry(
            tmp_path, register_builtin=False, skill_catalog=SkillCatalog.empty(),
            pre_execute_hook=CompositeToolHook(
                base, hooks, lambda _: None, _task_owner
            ),
        )
        registry.register("capture", capture)
        return registry

    registry = compose(manager.pre_tool, [
        ExtensionHook("hooks", "pre_tool", observe),
        ExtensionHook("hooks", "pre_tool", observe_again),
    ])
    try:
        result = await registry.execute(ToolCall("order", "capture", {}))
        assert result["isError"] is False
        assert marker.read_text() == "filefile"
        assert events == ["extension-1", "extension-2", "tool"]
    finally:
        await registry.close()

    registry = compose(lambda _tool, _args: "file denied", [
        ExtensionHook("hooks", "pre_tool", observe),
    ])
    try:
        denied = await registry.execute(ToolCall("file-deny", "capture", {}))
        assert denied["isError"] is True
        assert events == ["extension-1", "extension-2", "tool"]
    finally:
        await registry.close()

    async def deny(_tool: str, _args: dict[str, object]) -> str:
        return "extension denied"

    registry = compose(manager.pre_tool, [ExtensionHook("hooks", "pre_tool", deny)])
    try:
        denied = await registry.execute(ToolCall("extension-deny", "capture", {}))
        assert denied["isError"] is True
        assert events == ["extension-1", "extension-2", "tool"]
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_cancellation_resistant_timeout_releases_call_and_warns_again() -> None:
    release = asyncio.Event()
    cancelled = asyncio.Event()

    async def ignores_cancel(_tool: str, _args: dict[str, object]) -> None:
        try:
            await asyncio.wait_for(release.wait(), 10)
        except asyncio.CancelledError:
            cancelled.set()
            await asyncio.wait_for(release.wait(), 10)

    notices: list[str] = []
    extension_hook = ExtensionHook("hooks", "pre_tool", ignores_cancel, timeout_seconds=0.001)
    hook = CompositeToolHook(None, [extension_hook], notices.append, _task_owner)
    first = asyncio.create_task(hook("bash", {}))
    try:
        await asyncio.wait_for(cancelled.wait(), 10)
        assert first.done()
        assert notices
        assert await hook("bash", {}) is True
        assert len(notices) == 2
    finally:
        release.set()


@pytest.mark.asyncio
async def test_hook_tasks_are_owned_and_drained_on_close() -> None:
    owner = _TrackingOwner()
    started = asyncio.Event()
    release = asyncio.Event()

    async def post_tool(_name: str, _result: str) -> None:
        started.set()
        await asyncio.wait_for(release.wait(), 10)

    hook_task = asyncio.create_task(
        run_extension_hooks(
            [ExtensionHook("hooks", "post_tool", post_tool)],
            "post_tool",
            ("capture", "ok"),
            lambda _: None,
            owner,
        )
    )
    await asyncio.wait_for(started.wait(), 10)
    await owner.close()
    assert all(task.done() for task in owner.tasks)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(hook_task, 10)


@pytest.mark.asyncio
async def test_turn_cancellation_does_not_wait_for_hook() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def ignores_cancel(_tool: str, _args: dict[str, object]) -> None:
        started.set()
        try:
            await asyncio.wait_for(release.wait(), 10)
        except asyncio.CancelledError:
            await asyncio.wait_for(release.wait(), 10)

    hook = CompositeToolHook(
        None, [ExtensionHook("hooks", "pre_tool", ignores_cancel)], lambda _: None,
        _task_owner,
    )
    task = asyncio.create_task(hook("bash", {}))
    await asyncio.wait_for(started.wait(), 10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 10)
    release.set()


@pytest.mark.asyncio
async def test_pre_tool_runs_once_across_prepare_and_execute(tmp_path: Path) -> None:
    calls = 0

    def capture(_arguments: dict[str, object]) -> str:
        return "ok"

    def count(_tool: str, _args: dict[str, object]) -> bool:
        nonlocal calls
        calls += 1
        return True

    registry = ToolRegistry(
        tmp_path, register_builtin=False, skill_catalog=SkillCatalog.empty(),
        pre_execute_hook=CompositeToolHook(
            None, [ExtensionHook("hooks", "pre_tool", count)], lambda _: None,
            _task_owner,
        ),
    )
    registry.register("capture", capture, parameters={"type": "object"})
    try:
        call = ToolCall("call-once", "capture", {})
        assert await registry.prepare_approval(call) is None
        result = await registry.execute(call)
        assert result["isError"] is False
        assert calls == 1
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_timeout_disables_hook_and_warns() -> None:
    gate = asyncio.Event()

    async def hangs(_tool: str, _args: dict[str, object]) -> None:
        await asyncio.wait_for(gate.wait(), 10)

    notices: list[str] = []
    extension_hook = ExtensionHook("hooks", "pre_tool", hangs, timeout_seconds=0.001)
    hook = CompositeToolHook(None, [extension_hook], notices.append, _task_owner)
    assert await asyncio.wait_for(hook("bash", {}), 10) is True
    assert extension_hook.disabled
    assert len(notices) == 1
    gate.set()
