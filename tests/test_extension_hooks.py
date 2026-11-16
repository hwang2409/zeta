"""event-hook controls for native extensions."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from zeta.extensions.api import ExtensionAPI, ExtensionError
from zeta.extensions.hooks import CompositeToolHook, ExtensionHook
from zeta.extensions.host import LoadingExtension
from zeta.extensions.manifest import parse_manifest


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
async def test_timeout_disables_hook_and_warns() -> None:
    gate = asyncio.Event()

    async def hangs(_tool: str, _args: dict[str, object]) -> None:
        await gate.wait()

    notices: list[str] = []
    extension_hook = ExtensionHook("hooks", "pre_tool", hangs, timeout_seconds=0.001)
    hook = CompositeToolHook(None, [extension_hook], notices.append)
    assert await hook("bash", {}) is True
    assert extension_hook.disabled
    assert len(notices) == 1
    gate.set()
