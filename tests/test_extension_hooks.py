"""event-hook controls for native extensions."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from zeta.extensions.api import ExtensionAPI, ExtensionError
from zeta.extensions.hooks import CompositeToolHook, ExtensionHook
from zeta.extensions.host import LoadingExtension
from zeta.extensions.manifest import parse_manifest
from zeta.protocol.types import ToolCall
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry


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
async def test_pre_tool_can_deny_and_modify_arguments() -> None:
    async def modify(_tool: str, args: dict[str, object]) -> dict[str, object]:
        return {"arguments": {**args, "safe": True}}

    async def deny(_tool: str, _args: dict[str, object]) -> str:
        return "blocked"

    notices: list[str] = []
    args: dict[str, object] = {"safe": False}
    hook = CompositeToolHook(
        None,
        [
            ExtensionHook("hooks", "pre_tool", modify),
            ExtensionHook("hooks", "pre_tool", deny),
        ],
        notices.append,
    )
    assert await hook("bash", args) == "blocked"
    assert args == {"safe": True}
    assert notices == []


@pytest.mark.asyncio
async def test_cancellation_resistant_timeout_releases_call_and_warns_again() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def ignores_cancel(_tool: str, _args: dict[str, object]) -> None:
        started.set()
        try:
            await asyncio.wait_for(release.wait(), 10)
        except asyncio.CancelledError:
            await asyncio.wait_for(release.wait(), 10)

    notices: list[str] = []
    extension_hook = ExtensionHook("hooks", "pre_tool", ignores_cancel, timeout_seconds=0.001)
    hook = CompositeToolHook(None, [extension_hook], notices.append)
    first = asyncio.create_task(hook("bash", {}))
    await asyncio.wait_for(started.wait(), 10)
    assert await asyncio.wait_for(first, 10) is True
    assert notices
    assert await hook("bash", {}) is True
    assert len(notices) == 2
    release.set()


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
        None, [ExtensionHook("hooks", "pre_tool", ignores_cancel)], lambda _: None
    )
    task = asyncio.create_task(hook("bash", {}))
    await asyncio.wait_for(started.wait(), 10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 10)
    release.set()


@pytest.mark.asyncio
async def test_registry_executes_validated_modified_arguments(tmp_path: Path) -> None:
    seen: list[dict[str, object]] = []

    def capture(arguments: dict[str, object]) -> str:
        seen.append(arguments)
        return "ok"

    async def modify(_tool: str, args: dict[str, object]) -> dict[str, object]:
        return {"arguments": {**args, "value": "changed"}}

    registry = ToolRegistry(
        tmp_path, register_builtin=False, skill_catalog=SkillCatalog.empty(),
        pre_execute_hook=CompositeToolHook(None, [ExtensionHook("hooks", "pre_tool", modify)], lambda _: None),
    )
    registry.register("capture", capture, parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]})
    try:
        result = await registry.execute(ToolCall("call-1", "capture", {"value": "original"}))
        assert result["isError"] is False
        assert seen == [{"value": "changed"}]
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_registry_rejects_invalid_modified_arguments(tmp_path: Path) -> None:
    called = False

    def capture(_arguments: dict[str, object]) -> str:
        nonlocal called
        called = True
        return "bad"

    async def modify(_tool: str, _args: dict[str, object]) -> dict[str, object]:
        return {"arguments": {"value": 7}}

    registry = ToolRegistry(
        tmp_path, register_builtin=False, skill_catalog=SkillCatalog.empty(),
        pre_execute_hook=CompositeToolHook(None, [ExtensionHook("hooks", "pre_tool", modify)], lambda _: None),
    )
    registry.register(
        "capture", capture,
        parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
    )
    try:
        result = await registry.execute(ToolCall("call-1", "capture", {"value": "original"}))
        assert result["isError"] is True
        assert result["structuredContent"]["error"]["kind"] == "invalid_arguments"
        assert called is False
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_timeout_disables_hook_and_warns() -> None:
    gate = asyncio.Event()

    async def hangs(_tool: str, _args: dict[str, object]) -> None:
        await asyncio.wait_for(gate.wait(), 10)

    notices: list[str] = []
    extension_hook = ExtensionHook("hooks", "pre_tool", hangs, timeout_seconds=0.001)
    hook = CompositeToolHook(None, [extension_hook], notices.append)
    assert await asyncio.wait_for(hook("bash", {}), 10) is True
    assert extension_hook.disabled
    assert len(notices) == 1
    gate.set()
