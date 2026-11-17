"""Private-loop ownership and behavior for native extension hooks."""

from __future__ import annotations

import ast
import asyncio
import contextvars
import gc
import threading
import weakref
from pathlib import Path

import pytest

from tests.support.owned import await_or_fail
from zeta.extensions import apply_extensions
from zeta.extensions.hooks import CompositeToolHook, ExtensionHook, HookRunner
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry


async def _notice(_extension: str, _message: str) -> None:
    return None


@pytest.mark.asyncio
async def test_private_runner_value_context_and_thread() -> None:
    runner = HookRunner()
    ambient: contextvars.ContextVar[object] = contextvars.ContextVar("ambient")
    sentinel = object()
    token = ambient.set(sentinel)
    session_thread = threading.get_ident()

    async def inspect_context() -> tuple[object | None, int]:
        return ambient.get(None), threading.get_ident()

    try:
        outcome = await runner.invoke(inspect_context, (), 1)
        assert outcome.kind == "value"
        assert outcome.value == (None, outcome.value[1])
        assert outcome.value[1] != session_thread
    finally:
        ambient.reset(token)
        await runner.aclose()


@pytest.mark.asyncio
async def test_blocking_hook_does_not_stall_session_loop() -> None:
    runner = HookRunner()
    entered = threading.Event()
    release = threading.Event()
    ticks = 0

    async def blocks() -> None:
        entered.set()
        release.wait(10)

    # test-contract: latency proves the session loop advances during timeout
    invocation = asyncio.create_task(runner.invoke(blocks, (), 0.05))
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        async with asyncio.timeout(10):
            while not invocation.done():
                ticks += 1
                await asyncio.sleep(0)
        assert ticks > 0
        assert (await await_or_fail(invocation, what="blocking hook timeout")).kind == "timeout"
    finally:
        release.set()
        await runner.aclose()


@pytest.mark.asyncio
async def test_cancellation_resistant_close_is_bounded() -> None:
    # test-contract: latency close is bounded despite cancellation resistance
    runner = HookRunner(close_grace=0.01)
    started = threading.Event()
    release = threading.Event()

    async def resists() -> None:
        started.set()
        try:
            # test-contract: latency task must still run after cancellation
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            release.wait(10)

    invocation = asyncio.create_task(runner.invoke(resists, (), 60))
    assert await asyncio.to_thread(started.wait, 1)
    result = await asyncio.wait_for(runner.aclose(), 10)
    assert result.resistant is True
    assert (await asyncio.wait_for(invocation, 10)).kind == "closing"
    release.set()


def test_cancellation_resistant_close_lets_outer_asyncio_run_exit() -> None:
    completed = threading.Event()

    def run() -> None:
        async def scenario() -> None:
            runner = HookRunner(close_grace=0.01)
            started = threading.Event()
            release = threading.Event()

            async def resists() -> None:
                started.set()
                try:
                    # test-contract: latency task must still run after cancellation
                    await asyncio.sleep(3600)
                except asyncio.CancelledError:
                    release.wait(10)

            invocation = asyncio.create_task(runner.invoke(resists, (), 60))
            assert await asyncio.to_thread(started.wait, 10)
            assert (await asyncio.wait_for(runner.aclose(), 10)).resistant is True
            assert (await asyncio.wait_for(invocation, 10)).kind == "closing"
            release.set()

        asyncio.run(scenario())
        completed.set()

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(10)
    assert completed.is_set(), "asyncio.run waited for the private resistant hook"


def test_runner_cross_loop_posts_have_one_empty_context_helper() -> None:
    extension_sources = Path(__file__).parents[1] / "src/zeta/extensions"
    posts: list[tuple[Path, ast.Call]] = []
    combined = ""
    for source in extension_sources.glob("*.py"):
        text = source.read_text(encoding="utf-8")
        combined += text
        tree = ast.parse(text)
        posts.extend(
            (source, node) for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "call_soon_threadsafe"
        )
    assert [(path.name, node.func.attr) for path, node in posts] == [
        ("hooks.py", "call_soon_threadsafe")
    ]
    assert any(keyword.arg == "context" for keyword in posts[0][1].keywords)
    hook_tree = ast.parse(
        (extension_sources / "hooks.py").read_text(encoding="utf-8")
    )
    callback_tasks = [
        node for node in ast.walk(hook_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "create_task"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "loop"
    ]
    assert len(callback_tasks) == 1
    assert any(keyword.arg == "context" for keyword in callback_tasks[0].keywords)
    assert "run_coroutine_threadsafe" not in combined
    assert "asyncio.run(" not in combined
    assert "asyncio.Runner" not in combined
    assert ".join(" not in (extension_sources / "hooks.py").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_queued_cross_loop_post_does_not_retain_ambient_context() -> None:
    class Sentinel:
        pass

    runner = HookRunner(close_grace=0.01)
    entered = threading.Event()
    release = threading.Event()

    async def blocks() -> None:
        entered.set()
        release.wait(10)

    # test-contract: latency timeout queues cancellation behind the blocker
    invocation = asyncio.create_task(runner.invoke(blocks, (), 0.01))
    assert await asyncio.to_thread(entered.wait, 1)
    sentinel = Sentinel()
    reference = weakref.ref(sentinel)
    variable: contextvars.ContextVar[object] = contextvars.ContextVar("queued")
    token = variable.set(sentinel)
    await await_or_fail(invocation, what="queued hook timeout")
    variable.reset(token)
    del sentinel
    gc.collect()
    assert reference() is None, "a queued private-loop post retained ambient context"
    release.set()
    await runner.aclose()


@pytest.mark.asyncio
async def test_public_async_registration_runs_once_and_notify_marshals(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    extension = home / "extensions" / "hooks"
    extension.mkdir(parents=True)
    (extension / "zeta-extension.toml").write_text(
        '[extension]\nname="hooks"\nversion="1"\napi_version=1\nentry="ext:activate"\n',
        encoding="utf-8",
    )
    (extension / "ext.py").write_text(
        """
def activate(zx):
    async def safety(tool, arguments):
        zx.ui.notify(f"checked {tool}")
    zx.on("pre_tool", safety)
""",
        encoding="utf-8",
    )
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    host = apply_extensions(
        registry, home=home, project_dir=None, builtin_names=registry.registered_names
    )
    events = []
    sink_thread = []

    def sink(event):
        events.append(event)
        sink_thread.append(threading.get_ident())
        return True

    host.set_event_sink(sink, asyncio.get_running_loop())
    hook = host.pre_tool_hook()
    try:
        assert await hook("bash", {"command": "true"}) is True
        await asyncio.sleep(0)
        assert [event.data["text"] for event in events] == ["checked bash"]
        assert sink_thread == [threading.get_ident()]
    finally:
        await host.aclose()
        await registry.close()


@pytest.mark.asyncio
async def test_resistant_daemon_does_not_retain_closed_host(tmp_path: Path) -> None:
    home = tmp_path / "home"
    extension = home / "extensions" / "hooks"
    extension.mkdir(parents=True)
    (extension / "zeta-extension.toml").write_text(
        '[extension]\nname="hooks"\nversion="1"\napi_version=1\nentry="ext:activate"\n',
        encoding="utf-8",
    )
    (extension / "ext.py").write_text(
        """
import asyncio
import builtins

def activate(zx):
    async def safety(tool, arguments):
        zx.ui.notify("started")
        builtins._zeta_weak_started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            builtins._zeta_weak_release.wait(10)
    zx.on("pre_tool", safety)
""",
        encoding="utf-8",
    )
    started = threading.Event()
    import builtins
    release = threading.Event()
    builtins._zeta_weak_started = started
    builtins._zeta_weak_release = release
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    host = apply_extensions(
        registry, home=home, project_dir=None, builtin_names=registry.registered_names
    )
    host.set_event_sink(lambda _event: True, asyncio.get_running_loop())
    invocation = asyncio.create_task(host.pre_tool_hook()("bash", {}))
    assert await asyncio.to_thread(started.wait, 10)
    await host.aclose()
    assert (await asyncio.wait_for(invocation, 10)) is True
    reference = weakref.ref(host)
    del invocation
    del host
    gc.collect()
    assert reference() is None, "resistant hook retained the closed extension host"
    release.set()
    await registry.close()


@pytest.mark.asyncio
async def test_close_waits_for_admitted_warning_before_sink_drop(tmp_path: Path) -> None:
    home = tmp_path / "home"
    extension = home / "extensions" / "hooks"
    extension.mkdir(parents=True)
    (extension / "zeta-extension.toml").write_text(
        '[extension]\nname="hooks"\nversion="1"\napi_version=1\nentry="ext:activate"\n',
        encoding="utf-8",
    )
    (extension / "ext.py").write_text(
        """
def activate(zx):
    async def safety(tool, arguments):
        raise RuntimeError("boom")
    zx.on("pre_tool", safety)
""",
        encoding="utf-8",
    )
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    host = apply_extensions(
        registry, home=home, project_dir=None, builtin_names=registry.registered_names
    )
    notices = []
    host.set_event_sink(lambda event: notices.append(event) or True, asyncio.get_running_loop())
    entered = asyncio.Event()
    release = asyncio.Event()
    original_notice = host._hook_notice

    async def gated_notice(extension_name: str, detail: str) -> None:
        entered.set()
        await await_or_fail(release.wait(), what="warning release")
        await original_notice(extension_name, detail)

    host._hook_notice = gated_notice
    runner_closed = asyncio.Event()
    original_runner_close = host._runner.aclose

    async def gated_runner_close() -> object:
        result = await original_runner_close()
        runner_closed.set()
        return result

    host._runner.aclose = gated_runner_close
    invocation = asyncio.create_task(host.pre_tool_hook()("bash", {}))
    await await_or_fail(entered.wait(), what="hook warning admission")
    close = asyncio.create_task(host.aclose())
    await await_or_fail(runner_closed.wait(), what="private runner close")
    await asyncio.sleep(0)
    assert not close.done(), "close passed an admitted warning outcome"
    release.set()
    await await_or_fail(invocation, what="admitted hook outcome")
    await await_or_fail(close, what="extension host close")
    assert notices, "close dropped the admitted warning before sink teardown"
    await registry.close()


@pytest.mark.asyncio
async def test_rejected_warnings_coalesce_and_retry_once(tmp_path: Path) -> None:
    home = tmp_path / "home"
    extension = home / "extensions" / "hooks"
    extension.mkdir(parents=True)
    (extension / "zeta-extension.toml").write_text(
        '[extension]\nname="hooks"\nversion="1"\napi_version=1\nentry="ext:activate"\n',
        encoding="utf-8",
    )
    (extension / "ext.py").write_text(
        """
def activate(zx):
    async def safety(tool, arguments):
        raise RuntimeError("boom")
    zx.on("pre_tool", safety)
""",
        encoding="utf-8",
    )
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    host = apply_extensions(
        registry, home=home, project_dir=None, builtin_names=registry.registered_names
    )
    accepted = False
    notices = []

    def sink(event: object) -> bool:
        notices.append(event)
        return accepted

    host.set_event_sink(sink, asyncio.get_running_loop())
    hook = host.pre_tool_hook()
    assert await hook("bash", {}) is True
    assert await hook("bash", {}) is True
    assert host._safety_pending == {"hooks": 2}
    accepted = True
    await host._flush_safety("hooks")
    assert host._safety_pending == {}
    assert notices[-1].data["text"] == "2 extension hook calls allowed unchecked"
    await host.aclose()
    await registry.close()


@pytest.mark.asyncio
async def test_pre_tool_deny_and_copied_arguments() -> None:
    runner = HookRunner()

    async def mutate_then_deny(_tool: str, arguments: dict[str, object]) -> str:
        arguments["changed"] = True
        return "blocked"

    arguments: dict[str, object] = {}
    hook = CompositeToolHook(
        None, [ExtensionHook("hooks", "pre_tool", mutate_then_deny)], _notice, runner
    )
    try:
        assert await hook("bash", arguments) == "blocked"
        assert arguments == {}
    finally:
        await runner.aclose()


@pytest.mark.asyncio
async def test_close_completion_between_snapshot_and_sink_drop_is_delivered(
    tmp_path: Path,
) -> None:
    """A completion already won before close must not become a warning."""
    runner = HookRunner()
    release = threading.Event()

    async def finishes() -> str:
        await asyncio.to_thread(release.wait, 10)
        return "done"

    invocation = asyncio.create_task(runner.invoke(finishes, (), 1))
    release.set()
    outcome = await await_or_fail(invocation, what="hook completion")
    close = await runner.aclose()
    assert outcome.kind == "value"
    assert close.resistant is False
