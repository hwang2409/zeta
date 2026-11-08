from __future__ import annotations

import ast
import asyncio
import subprocess
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


def _conditional_close_serve_lines(tree: ast.AST) -> set[int]:
    """Line numbers of close_serve calls nested in an if/elif/else body.

    close_serve owns serve-test cleanup; gating it behind a condition is the
    recurring bug where an early failure skips the server/writer close and
    leaks descriptors into the next test (PR #881 rounds 1-2). The guard has no
    exemptions: a call that needs a condition is rewritten to the unconditional
    form instead.
    """
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            for branch in (node.body, node.orelse):
                for statement in branch:
                    for inner in ast.walk(statement):
                        if _is_close_serve_call(inner):
                            lines.add(inner.lineno)
    return lines


def _test_files() -> list[Path]:
    root = _TESTS_ROOT.parent
    result = subprocess.run(
        ["git", "ls-files", "--", "tests/*.py", "tests/**/*.py"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if not result.stdout.splitlines():
        raise RuntimeError("no tracked test files found")
    return [root / path for path in result.stdout.splitlines()]


def test_close_serve_is_never_called_under_a_condition() -> None:
    offenders: list[str] = []
    for path in _test_files():
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


def test_close_serve_guard_ignores_untracked_runtime_probe() -> None:
    probe = _TESTS_ROOT / "test_zz_probe_tmp.py"
    probe.write_text("def test_probe():\n    pass\n", encoding="utf-8")
    try:
        files = _test_files()
        assert _TESTS_ROOT / "test_serve_close.py" in files
        assert probe not in files
    finally:
        probe.unlink()


def test_test_files_fails_on_git_listing_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, command, stderr="git listing failed")

    monkeypatch.setattr(subprocess, "run", run)

    with pytest.raises(subprocess.CalledProcessError) as error:
        _test_files()
    assert error.value.stderr == "git listing failed"


def test_test_files_rejects_empty_git_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(subprocess, "run", run)

    with pytest.raises(RuntimeError, match="no tracked test files found"):
        _test_files()


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

    outcome: BaseException | None = None
    try:
        await close_serve(server, *writers)
    except RuntimeError as error:
        outcome = error

    assert server.closed
    assert all(writer.closed for writer in writers)
    assert all(writer.wait_started_with_server_shutdown for writer in writers)
    if writer_failure:
        assert isinstance(outcome, RuntimeError)
        assert str(outcome) == "writer close failed"
    elif server_failure:
        assert isinstance(outcome, RuntimeError)
        assert str(outcome) == "server close failed"
    else:
        assert outcome is None


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
