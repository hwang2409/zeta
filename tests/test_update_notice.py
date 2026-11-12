from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.support.server_client import read_until, send
from tests.support.sockets import short_socket_path
from tests.test_server import _connect
from zeta.runtime.update_notice import (
    ProcessUpdateMonitor,
    UpdateChecker,
    source_revision,
)
from zeta.server import ZetaServer
from zeta.tui import layout


def test_checker_reports_each_transition_from_running_revision() -> None:
    revisions = iter(("old", "new", "newer"))
    updates: list[tuple[str, str]] = []

    async def on_update(old: str, new: str) -> None:
        updates.append((old, new))

    checker = UpdateChecker(
        Path("/not-used"), on_update, read_revision=lambda _: next(revisions)
    )
    assert asyncio.run(checker.tick())
    assert asyncio.run(checker.tick())
    assert updates == [("old", "new"), ("old", "newer")]


@pytest.mark.asyncio
async def test_failing_subscriber_does_not_stop_other_subscribers(tmp_path: Path) -> None:
    monitor = ProcessUpdateMonitor(tmp_path / "package.py")
    received: list[tuple[str, str]] = []

    async def failing(_old: str, _new: str) -> None:
        raise RuntimeError("subscriber failed")

    async def receiving(old: str, new: str) -> None:
        received.append((old, new))

    monitor.subscribe(failing)
    monitor.subscribe(receiving)
    await monitor._publish("old", "new")
    assert received == [("old", "new")]
    assert monitor.current == ("old", "new")


def test_checker_silently_retries_a_failed_read() -> None:
    revisions = iter(("old", None, "new"))
    updates: list[tuple[str, str]] = []

    async def on_update(old: str, new: str) -> None:
        updates.append((old, new))

    checker = UpdateChecker(
        Path("/not-used"), on_update, read_revision=lambda _: next(revisions)
    )
    assert not asyncio.run(checker.tick())
    assert asyncio.run(checker.tick())
    assert updates == [("old", "new")]


def test_checker_reports_one_revision_change_without_sleep() -> None:
    revisions = iter(("old", "new", "new"))
    updates: list[tuple[str, str]] = []

    async def on_update(old: str, new: str) -> None:
        updates.append((old, new))

    checker = UpdateChecker(
        Path("/not-used"), on_update, read_revision=lambda _: next(revisions)
    )
    assert asyncio.run(checker.tick())
    assert not asyncio.run(checker.tick())
    assert updates == [("old", "new")]


def test_process_monitor_restarts_after_new_event_loop(monkeypatch) -> None:
    revision = ["old"]
    monitor = ProcessUpdateMonitor(
        Path("/not-used"),
    )
    monitor._checker._read_revision = lambda _: revision[0]
    received: list[tuple[str, str]] = []

    async def first_loop() -> None:
        await monitor.tick()
        monitor.start()

    asyncio.run(first_loop())

    async def second_loop() -> None:
        updated = asyncio.Event()

        async def on_update(old: str, new: str) -> None:
            received.append((old, new))
            updated.set()

        monitor.subscribe(on_update)
        revision[0] = "new"
        monitor.start()
        await asyncio.wait_for(updated.wait(), 10)

    asyncio.run(second_loop())
    assert received == [("old", "new")]
    assert monitor.current == ("old", "new")

    class Output:
        def get_size(self):
            return type("Size", (), {"columns": 80})()

    class PromptApp:
        output = Output()

    class App:
        _remote_view = None

    monkeypatch.setattr(layout, "process_update_monitor", lambda _package: monitor)
    monkeypatch.setattr(layout, "get_app", lambda: PromptApp())
    assert "zeta updated · /restart" in "".join(
        text for _style, text in layout.status_toolbar(App())
    )


def test_source_revision_reads_detached_and_packed_heads(tmp_path: Path) -> None:
    package = tmp_path / "src" / "zeta" / "__init__.py"
    package.parent.mkdir(parents=True)
    package.write_text("")
    git_dir = tmp_path / ".git"
    (git_dir / "refs" / "heads").mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "refs" / "heads" / "main").write_text("loose\n")
    assert source_revision(package) == "loose"
    (git_dir / "refs" / "heads" / "main").unlink()
    (git_dir / "packed-refs").write_text("# pack-refs with: peeled fully-peeled\npacked refs/heads/main\n")
    assert source_revision(package) == "packed"
    (git_dir / "HEAD").write_text("detached\n")
    assert source_revision(package) == "detached"


def test_source_revision_reads_worktree_git_file(tmp_path: Path) -> None:
    package = tmp_path / "worktree" / "src" / "zeta" / "__init__.py"
    package.parent.mkdir(parents=True)
    package.write_text("")
    common = tmp_path / "common" / ".git"
    worktree_git = common / "worktrees" / "linked"
    worktree_git.mkdir(parents=True)
    (tmp_path / "worktree" / ".git").write_text(
        f"gitdir: {worktree_git}\n"
    )
    (worktree_git / "HEAD").write_text("ref: refs/heads/linked\n")
    (worktree_git / "commondir").write_text("../..\n")
    (common / "refs" / "heads").mkdir(parents=True)
    (common / "refs" / "heads" / "linked").write_text("worktree\n")
    assert source_revision(package) == "worktree"


def test_no_git_checkout_uses_a_source_fingerprint(tmp_path: Path) -> None:
    package = tmp_path / "package.py"
    package.write_text("value = 1")
    first = source_revision(package)
    package.write_text("value = 22")
    assert first is not None
    assert source_revision(package) != first


def test_updated_footer_uses_the_persistent_notice(monkeypatch) -> None:
    class Output:
        def get_size(self):
            return type("Size", (), {"columns": 80})()

    class PromptApp:
        output = Output()

    class App:
        _remote_view = None

    monitor = type("Monitor", (), {"current": ("old", "new")})()
    monkeypatch.setattr(layout, "process_update_monitor", lambda _package: monitor)
    monkeypatch.setattr(layout, "get_app", lambda: PromptApp())
    assert "zeta updated · /restart" in "".join(
        text for _style, text in layout.status_toolbar(App())
    )


@pytest.mark.asyncio
async def test_fresh_serve_publishes_update_without_opened_session(tmp_path: Path) -> None:
    server = ZetaServer(
        home=tmp_path,
        socket_path=short_socket_path(tmp_path),
        provider="ollama",
        model="offline",
    )
    reader, writer = await _connect(server)
    try:
        await send(
            reader,
            writer,
            1,
            "hello",
            {"protocol_version": "1.2", "role": "controller", "features": ["zeta_updated"]},
        )
        await read_until(
            reader, lambda frame: frame.get("id") == 1, what="hello response"
        )
        await send(reader, writer, 2, "ping")
        await read_until(reader, lambda frame: frame.get("id") == 2, what="ping response")
        await server.publish_update("old", "new")
        frame = await read_until(
            reader,
            lambda frame: frame.get("params", {}).get("event") == "zeta_updated",
            what="zeta update event",
        )
        frame = next(frame for frame in frame if frame.get("params", {}).get("event") == "zeta_updated")
        assert frame["params"]["event"] == "zeta_updated"
        assert frame["params"]["data"] == {"old_revision": "old", "new_revision": "new"}
    finally:
        writer.close()
        await writer.wait_closed()
        await server.close()
