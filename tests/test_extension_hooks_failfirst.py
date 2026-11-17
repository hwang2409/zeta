"""Behavior regressions for private extension-hook ownership."""

from __future__ import annotations

import asyncio
import builtins
import contextvars
import gc
import subprocess
import sys
import textwrap
import threading
import weakref
from pathlib import Path

import pytest

from zeta.extensions import apply_extensions
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry


def _task_owner(coroutine: object) -> asyncio.Task[object]:
    return asyncio.create_task(coroutine)  # type: ignore[arg-type]


def _write_extension(home: Path, source: str) -> None:
    directory = home / "extensions" / "hooks"
    directory.mkdir(parents=True)
    (directory / "zeta-extension.toml").write_text(
        '[extension]\nname="hooks"\nversion="1"\napi_version=1\nentry="ext:activate"\n',
        encoding="utf-8",
    )
    (directory / "ext.py").write_text(source, encoding="utf-8")


@pytest.mark.skipif(sys.version_info < (3, 14), reason="Thread(context=) starts in 3.14")
def test_python314_inherited_thread_context_is_empty(tmp_path: Path) -> None:
    script = tmp_path / "probe.py"
    script.write_text(
        textwrap.dedent(
            """
            import asyncio
            import contextvars
            import tempfile
            from pathlib import Path
            from zeta.extensions import apply_extensions
            from zeta.skills import SkillCatalog
            from zeta.tools import ToolRegistry

            ambient = contextvars.ContextVar("ambient")

            async def main():
                root = Path(tempfile.mkdtemp())
                ext = root / "home/extensions/hooks"
                ext.mkdir(parents=True)
                (ext / "zeta-extension.toml").write_text(
                    '[extension]\\nname="hooks"\\nversion="1"\\napi_version=1\\nentry="ext:activate"\\n'
                )
                (ext / "ext.py").write_text(
                    'from __main__ import ambient\\n'
                    'def activate(zx):\\n'
                    '    async def hook(tool, args):\\n'
                    '        if ambient.get(None) is not None: return "inherited"\\n'
                    '    zx.on("pre_tool", hook)\\n'
                )
                registry = ToolRegistry(root, skill_catalog=SkillCatalog.empty())
                host = apply_extensions(
                    registry, home=root / "home", project_dir=None,
                    builtin_names=registry.registered_names,
                    task_owner=asyncio.create_task,
                )
                token = ambient.set(object())
                try:
                    result = await host.pre_tool_hook()("bash", {})
                    assert result is True, "hook daemon inherited the session ambient context"
                finally:
                    ambient.reset(token)
                    close = getattr(host, "aclose", None)
                    if close is None:
                        host.close()
                    else:
                        await close()
                    await registry.close()

            asyncio.run(main())
            """
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, "-X", "thread_inherit_context=1", str(script)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.asyncio
async def test_resistant_hook_does_not_retain_session_context(tmp_path: Path) -> None:
    class Sentinel:
        pass

    home = tmp_path / "home"
    release = threading.Event()
    builtins._zeta_context_release = release
    _write_extension(
        home,
        """
import asyncio
import builtins

def activate(zx):
    async def hook(tool, arguments):
        while not builtins._zeta_context_release.is_set():
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                pass
    zx.on("pre_tool", hook)
""",
    )
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    host = apply_extensions(
        registry, home=home, project_dir=None, builtin_names=registry.registered_names,
        task_owner=_task_owner,
    )
    host._hooks[0].timeout_seconds = 0.01
    host.set_event_sink(lambda _event: True, asyncio.get_running_loop())
    ambient: contextvars.ContextVar[object] = contextvars.ContextVar("ambient")
    sentinel = Sentinel()
    reference = weakref.ref(sentinel)
    token = ambient.set(sentinel)
    try:
        assert await host.pre_tool_hook()("bash", {}) is True
    finally:
        ambient.reset(token)
        del sentinel
    try:
        gc.collect()
        assert reference() is None, "resistant hook retained the session ambient context"
    finally:
        release.set()
        close = getattr(host, "aclose", None)
        if close is None:
            host.close()
        else:
            await close()
        await registry.close()


@pytest.mark.asyncio
async def test_close_claim_warns_before_sink_drop(tmp_path: Path) -> None:
    home = tmp_path / "home"
    started = threading.Event()
    release = threading.Event()
    _write_extension(
        home,
        """
import asyncio
import builtins

def activate(zx):
    async def hook(tool, arguments):
        builtins._zeta_hook_started.set()
        await asyncio.to_thread(builtins._zeta_hook_release.wait)
    zx.on("pre_tool", hook)
""",
    )
    builtins._zeta_hook_started = started
    builtins._zeta_hook_release = release
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    host = apply_extensions(
        registry, home=home, project_dir=None, builtin_names=registry.registered_names,
        task_owner=_task_owner,
    )
    notices = []
    host.set_event_sink(lambda event: notices.append(event) or True, asyncio.get_running_loop())
    invocation = asyncio.create_task(host.pre_tool_hook()("bash", {}))
    assert await asyncio.to_thread(started.wait, 1)
    close = getattr(host, "aclose", None)
    if close is None:
        host.close()
    else:
        await close()
    release.set()
    await asyncio.wait_for(invocation, 10)
    assert notices, "close dropped the admitted hook warning before sink teardown"
    await registry.close()
