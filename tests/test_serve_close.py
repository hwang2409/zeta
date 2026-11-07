from __future__ import annotations

import ast
import asyncio
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.support.serve_close import close_serve

_TESTS_ROOT = Path(__file__).resolve().parent
_CLOSE_SERVE_CONDITION_FIX = (
    "call close_serve unconditionally; pass optional writers as "
    "*[w for w in (...) if w is not None]"
)


def _is_close_serve_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (isinstance(func, ast.Name) and func.id == "close_serve") or (
        isinstance(func, ast.Attribute) and func.attr == "close_serve"
    )


_SERVER_START_HELPERS = frozenset({"start_server"})


def _call_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _is_server_acquisition(node: ast.stmt) -> bool:
    value: ast.AST | None = None
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr)):
        value = node.value
    if value is None:
        return False
    calls = [call for call in ast.walk(value) if isinstance(call, ast.Call)]
    names = {_call_name(call) for call in calls}
    names.discard(None)
    if any(name in _SERVER_START_HELPERS for name in names):
        return True
    return any(name == "start_server" for name in names)


def _conditional_close_serve_lines(tree: ast.AST) -> set[int]:
    """Line numbers of close_serve calls that are not owned by server setup."""
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not _is_close_serve_call(node):
            continue
        ancestors: list[ast.AST] = []
        parent = parents.get(node)
        function: ast.AST | None = None
        while parent is not None:
            ancestors.append(parent)
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function = parent
                break
            parent = parents.get(parent)
        if function is None:
            lines.add(node.lineno)
            continue
        try_ancestor = next(
            (
                ancestor
                for ancestor in ancestors
                if isinstance(ancestor, ast.Try)
                and any(node in ast.walk(statement) for statement in ancestor.finalbody)
            ),
            None,
        )
        if try_ancestor is None:
            lines.add(node.lineno)
            continue
        try_index = ancestors.index(try_ancestor)
        other_ancestors = ancestors[try_index + 1 :]
        if any(
            isinstance(ancestor, (ast.If, ast.IfExp, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith))
            for ancestor in other_ancestors
        ):
            lines.add(node.lineno)
            continue
        if any(
            isinstance(ancestor, ast.Try)
            and not any(node in ast.walk(statement) for statement in ancestor.finalbody)
            for ancestor in other_ancestors
        ):
            lines.add(node.lineno)
            continue
        # The owning try must be the next statement after server acquisition.
        block = parents.get(try_ancestor)
        if not hasattr(block, "body") or try_ancestor not in block.body:
            lines.add(node.lineno)
            continue
        index = block.body.index(try_ancestor)
        if index == 0:
            function_decorators = getattr(function, "decorator_list", [])
            if any(
                isinstance(decorator, ast.Name)
                and decorator.id == "asynccontextmanager"
                for decorator in function_decorators
            ):
                lines.add(node.lineno)
            continue
        if _is_server_acquisition(block.body[index - 1]):
            continue
        if any(_is_server_acquisition(statement) for statement in block.body[:index]):
            lines.add(node.lineno)
    return lines


@pytest.mark.parametrize(
    ("source", "expected_offender"),
    [
        pytest.param("async def test():\n    close_serve(s) if ready else None\n", True, id="ternary"),
        pytest.param("async def test():\n    for _ in ():\n        close_serve(s)\n", True, id="for"),
        pytest.param("async def test():\n    while ready:\n        close_serve(s)\n", True, id="while"),
        pytest.param("async def test():\n    try:\n        close_serve(s)\n    finally:\n        pass\n", True, id="try-body"),
        pytest.param("async def test():\n    try:\n        pass\n    except Exception:\n        close_serve(s)\n", True, id="except-handler"),
        pytest.param("async def test():\n    if ready:\n        try:\n            pass\n        finally:\n            close_serve(s)\n", True, id="if-around-finally"),
        pytest.param("async def test():\n    close_serve(s)\n", True, id="top-level-after-assert"),
        pytest.param("async def test():\n    async with resource:\n        close_serve(s)\n", True, id="async-with-without-helper"),
        pytest.param("async def test():\n    try:\n        if ready:\n            try:\n                pass\n            finally:\n                close_serve(s)\n    finally:\n        pass\n", True, id="unreachable-nested-finally"),
        pytest.param("async def test():\n    try:\n        pass\n    finally:\n        close_serve(s)\n", False, id="finally"),
        pytest.param(
            "async def test():\n"
            "    server = await start_server()\n"
            "    try:\n"
            "        pass\n"
            "    finally:\n"
            "        await close_serve(server)\n",
            False,
            id="adjacent-acquisition",
        ),
        pytest.param(
            "async def test():\n"
            "    server = await start_server()\n"
            "    await can_fail()\n"
            "    try:\n"
            "        pass\n"
            "    finally:\n"
            "        await close_serve(server)\n",
            True,
            id="await-between-acquisition-and-try",
        ),
        pytest.param(
            "async def test():\n"
            "    server = await start_server()\n"
            "    if ready:\n"
            "        return\n"
            "    try:\n"
            "        pass\n"
            "    finally:\n"
            "        await close_serve(server)\n",
            True,
            id="return-between-acquisition-and-try",
        ),
        pytest.param(
            "async def test():\n"
            "    @asynccontextmanager\n"
            "    async def serving(server):\n"
            "        try:\n"
            "            yield\n"
            "        finally:\n"
            "            await close_serve(server)\n"
            "    async with serving(s):\n"
            "        pass\n",
            True,
            id="generic-asynccontextmanager",
        ),
    ],
)
def test_close_serve_guard_shapes(source: str, expected_offender: bool) -> None:
    offenders = _conditional_close_serve_lines(ast.parse(source))
    assert bool(offenders) is expected_offender


def test_close_serve_is_never_called_under_a_condition() -> None:
    offenders: list[str] = []
    for path in sorted(_TESTS_ROOT.rglob("*.py")):
        if path.name == "test_serve_close.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        root = _TESTS_ROOT.parent
        offenders.extend(
            f"{path.relative_to(root)}:{line}"
            for line in sorted(_conditional_close_serve_lines(tree))
        )
    assert not offenders, (
        "close_serve must not be nested inside an if/elif/else body "
        f"({', '.join(offenders)}): {_CLOSE_SERVE_CONDITION_FIX}"
    )


@dataclass
class _Server:
    fail: bool = False

    def __post_init__(self) -> None:
        self.started = asyncio.Event()
        self.closed = False

    async def close(self) -> None:
        self.started.set()
        await asyncio.sleep(0)
        self.closed = True
        if self.fail:
            raise RuntimeError("server close failed")


class _Writer:
    def __init__(self, server: _Server, *, fail: bool = False) -> None:
        self.server = server
        self.fail = fail
        self.closed = False
        self.wait_started_with_server_shutdown = False

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        self.wait_started_with_server_shutdown = self.server.started.is_set()
        assert self.wait_started_with_server_shutdown, (
            "serve shutdown must start before waiting for a peer writer to close"
        )
        if self.fail:
            raise RuntimeError("writer close failed")


class _CancellationServer:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False
        self.task: asyncio.Task[None] | None = None
        self.owner_task: asyncio.Task[None] | None = None
        self.cancel_owner = False
        self.cancel_during_server_wait = False
        self.trigger = asyncio.Event()

    async def close(self) -> None:
        self.task = asyncio.current_task()
        self.started.set()
        if self.cancel_owner and self.owner_task is not None:
            self.owner_task.cancel()
        if self.cancel_during_server_wait:
            await asyncio.wait_for(self.trigger.wait(), 10)
            if self.owner_task is not None:
                self.owner_task.cancel()
        await asyncio.wait_for(self.release.wait(), 10)
        self.closed = True


class _CancellationWriter:
    def __init__(self, server: _CancellationServer, *, wait: bool) -> None:
        self.server = server
        self.wait = wait
        self.waiting = asyncio.Event()
        self.finished = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.closed = False

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        self.waiting.set()
        if self.wait:
            try:
                await asyncio.wait_for(self.finished.wait(), 10)
            except asyncio.CancelledError:
                self.cancelled.set()
                self.server.trigger.set()
                raise
        else:
            self.finished.set()


@pytest.mark.asyncio
async def test_close_serve_starts_server_shutdown_before_peer_wait() -> None:
    server = _Server()
    writer = _Writer(server)

    await close_serve(server, writer)

    assert server.closed
    assert writer.closed
    assert writer.wait_started_with_server_shutdown


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("writer_count", "writer_failure", "server_failure"),
    [
        pytest.param(1, False, False, id="success"),
        pytest.param(1, True, False, id="writer-wait-failure"),
        pytest.param(1, False, True, id="server-close-failure"),
        pytest.param(2, False, False, id="multiple-writers"),
    ],
)
async def test_close_serve_cleanup_outcomes(
    writer_count: int, writer_failure: bool, server_failure: bool
) -> None:
    server = _Server(fail=server_failure)
    writers = [
        _Writer(server, fail=writer_failure and index == 0)
        for index in range(writer_count)
    ]

    expected_error = (
        "writer close failed" if writer_failure else "server close failed" if server_failure else ""
    )
    context = (
        pytest.raises(RuntimeError, match=expected_error)
        if expected_error
        else nullcontext()
    )
    with context:
        await close_serve(server, *writers)

    assert server.closed
    assert all(writer.closed for writer in writers)
    assert all(writer.wait_started_with_server_shutdown for writer in writers)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["first-yield", "writer-wait", "server-wait"])
async def test_close_serve_drains_server_when_cancelled(phase: str) -> None:
    server = _CancellationServer()
    writer = _CancellationWriter(server, wait=phase == "writer-wait")
    close_task = asyncio.create_task(close_serve(server, writer))
    server.owner_task = close_task
    server.cancel_owner = phase == "first-yield"
    server.cancel_during_server_wait = phase == "writer-wait"
    try:
        if phase == "first-yield":
            await asyncio.wait_for(server.started.wait(), 10)
        elif phase == "writer-wait":
            await asyncio.wait_for(writer.waiting.wait(), 10)
            close_task.cancel()
            await asyncio.wait_for(writer.cancelled.wait(), 10)
        else:
            await asyncio.wait_for(writer.finished.wait(), 10)
            close_task.cancel()
        assert not close_task.done(), "cleanup must drain the server before returning"
        server.release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(close_task, 10)
        assert server.closed
        assert close_task.done()
        assert server.task is not None and server.task.done()
    finally:
        server.release.set()
        if not close_task.done():
            close_task.cancel()
        await asyncio.wait_for(asyncio.gather(close_task, return_exceptions=True), 10)
