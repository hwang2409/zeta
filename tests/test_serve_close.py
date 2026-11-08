from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.support.serve_close import close_serve


def _run_owner_test(pytester: pytest.Pytester, source: str):
    pytester.makeconftest('pytest_plugins = ("tests.conftest", "pytest_asyncio")')
    pytester.makeini("[pytest]\nasyncio_mode = auto\n")
    pytester.makepyfile(test_owner=source)
    root = str(Path(__file__).resolve().parents[1])
    old_pythonpath = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = (
        root
        if not old_pythonpath
        else os.pathsep.join((root, old_pythonpath))
    )
    try:
        return pytester.runpytest_subprocess("-q")
    finally:
        if old_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = old_pythonpath


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


def test_server_owner_closes_after_body_failure(pytester: pytest.Pytester) -> None:
    result = _run_owner_test(
        pytester,
        """
import pytest
from pathlib import Path
from zeta.server import ZetaServer

pytestmark = pytest.mark.usefixtures("close_direct_test_runtime_owners")

@pytest.mark.asyncio
async def test_leaked_server(tmp_path):
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    await server.start()
    original_close = server.close
    async def close_and_prove():
        await original_close()
        assert not server.socket_path.exists()
        assert server._server is None
        Path("closed").write_text("closed")
    server.close = close_and_prove
    assert False, "original body failure"
""",
    )
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*AssertionError: original body failure*"])
    assert (pytester.path / "closed").read_text() == "closed"


def test_app_owner_closes_after_body_failure(pytester: pytest.Pytester) -> None:
    result = _run_owner_test(
        pytester,
        """
import asyncio
from pathlib import Path
import pytest
from tests.support.fake_backend import FakeBackend
from zeta.core.store import ConversationStore
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.tui.app import TUIApp

pytestmark = pytest.mark.usefixtures("close_direct_test_runtime_owners")

@pytest.mark.asyncio
async def test_leaked_app(tmp_path):
    loop = AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions"), skill_catalog=SkillCatalog.empty())
    app = TUIApp(loop, provider="codex", model="offline")
    cancelled = asyncio.Event()
    async def active():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
    app._active_task = asyncio.create_task(active())
    original_close = app.close
    async def close_and_prove():
        await original_close()
        assert cancelled.is_set()
        Path("closed").write_text("closed")
    app.close = close_and_prove
    assert False, "original body failure"
""",
    )
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*AssertionError: original body failure*"])
    assert (pytester.path / "closed").read_text() == "closed"


def test_app_owner_explicit_close_is_idempotent(pytester: pytest.Pytester) -> None:
    result = _run_owner_test(
        pytester,
        """
import pytest
from tests.support.fake_backend import FakeBackend
from zeta.core.store import ConversationStore
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.tui.app import TUIApp

pytestmark = pytest.mark.usefixtures("close_direct_test_runtime_owners")

@pytest.mark.asyncio
async def test_explicit_close(tmp_path):
    app = TUIApp(AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions"), skill_catalog=SkillCatalog.empty()), provider="codex", model="offline")
    await app.close()
""",
    )
    result.assert_outcomes(passed=1)


def test_app_owner_reports_bounded_close_failure(pytester: pytest.Pytester) -> None:
    result = _run_owner_test(
        pytester,
        """
import asyncio
import pytest
from tests.support.fake_backend import FakeBackend
from zeta.core.store import ConversationStore
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.tui.app import TUIApp

pytestmark = pytest.mark.usefixtures("close_direct_test_runtime_owners")

@pytest.mark.asyncio
async def test_hanging_app_close(tmp_path):
    app = TUIApp(AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions"), skill_catalog=SkillCatalog.empty()), provider="codex", model="offline")
    async def hang():
        await asyncio.Event().wait()
    app.close = hang
""",
    )
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(["*hang: TUIApp.close did not finish within 10 s*"])


def test_two_apps_and_server_close_in_dependency_order(pytester: pytest.Pytester) -> None:
    result = _run_owner_test(
        pytester,
        """
from pathlib import Path
import pytest
from tests.support.fake_backend import FakeBackend
from zeta.core.store import ConversationStore
from zeta.runtime.loop import AgentLoop
from zeta.server import ZetaServer
from zeta.skills import SkillCatalog
from zeta.tui.app import TUIApp

pytestmark = pytest.mark.usefixtures("close_direct_test_runtime_owners")

def app(tmp_path):
    return TUIApp(AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions"), skill_catalog=SkillCatalog.empty()), provider="codex", model="offline")

@pytest.mark.asyncio
async def test_owners(tmp_path):
    first = app(tmp_path / "first")
    second = app(tmp_path / "second")
    for value in (first, second):
        original = value.close
        async def close(original=original):
            await original()
            with Path("order").open("a") as marker:
                marker.write("a")
        value.close = close
    server = ZetaServer(home=tmp_path / "server", port=0, provider="codex")
    await server.start()
    original = server.close
    async def close_server():
        with Path("order").open("a") as marker:
            marker.write("s")
        await original()
    server.close = close_server
""",
    )
    result.assert_outcomes(passed=1)
    assert (pytester.path / "order").read_text() == "aas"


def test_server_owner_explicit_close_is_idempotent(pytester: pytest.Pytester) -> None:
    result = _run_owner_test(
        pytester,
        """
import pytest
from zeta.server import ZetaServer

pytestmark = pytest.mark.usefixtures("close_direct_test_runtime_owners")

@pytest.mark.asyncio
async def test_explicit_close(tmp_path):
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    await server.start()
    await server.close()
""",
    )
    result.assert_outcomes(passed=1)


def test_server_owner_reports_bounded_close_failure(pytester: pytest.Pytester) -> None:
    result = _run_owner_test(
        pytester,
        """
import asyncio
import pytest
from zeta.server import ZetaServer

pytestmark = pytest.mark.usefixtures("close_direct_test_runtime_owners")

@pytest.mark.asyncio
async def test_hanging_close(tmp_path):
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    async def hang():
        await asyncio.Event().wait()
    server.close = hang
""",
    )
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(["*ZetaServer.close did not finish within 10 s*"])


def test_server_owner_closes_all_servers(pytester: pytest.Pytester) -> None:
    result = _run_owner_test(
        pytester,
        """
import pytest
from pathlib import Path
from zeta.server import ZetaServer

pytestmark = pytest.mark.usefixtures("close_direct_test_runtime_owners")

@pytest.mark.asyncio
async def test_two_servers(tmp_path):
    servers = [
        ZetaServer(home=tmp_path / str(index), port=0, provider="codex")
        for index in range(2)
    ]
    for server in servers:
        await server.start()
        original_close = server.close
        async def close_and_prove(server=server, original_close=original_close):
            await original_close()
            assert server._server is None
            with Path("closed").open("a") as marker:
                marker.write("x")
        server.close = close_and_prove
""",
    )
    result.assert_outcomes(passed=1)
    assert (pytester.path / "closed").read_text() == "xx"


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
