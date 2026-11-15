"""armed continuation, arm wake, history dedupe, and start-card dedupe (s4c F1/F3/F4/F8)."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import time
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

import zeta.server.server as server_module
from tests.support.fake_backend import FakeBackend
from tests.support.owned import owned_app
from tests.support.retention_owner import _age_tree, _create_closed
from tests.support.serve_close import close_serve
from tests.test_server import _connect, _event, _request
from zeta.cli.main import _attach_retention_owner
from zeta.core.session import SessionManager
from zeta.runtime.loop import AgentLoop
from zeta.server import ZetaServer
from zeta.session_retention import (
    read_retention_history,
    retention_authority,
)
from zeta.session_retention.activation import make_retention_owner
from zeta.session_retention.pin import set_retention_pin
from zeta.skills import SkillCatalog
from zeta.storage_format import open_writer
from zeta.tui.app import TUIApp
from zeta.tui.cards.retention import render_retention_card

_YEAR_NS = 365 * 86_400 * 1_000_000_000


def _future() -> int:
    # a year ahead, so lock files the scan itself creates cannot make a root young
    return time.time_ns() + _YEAR_NS


class _IdleSignal(asyncio.Event):
    """The owner's wake event; it reports each time the owner goes idle."""

    def __init__(self) -> None:
        super().__init__()
        self.idle = asyncio.Event()

    def wait(self):
        self.idle.set()
        # test-contract: seam the owner awaits this coroutine under its own wake deadline
        return super().wait()


async def _settle(event: asyncio.Event) -> None:
    # a guard only: every assertion below checks state, never elapsed time
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(event.wait(), 30)


def _without_progress(row: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key not in {"timestamp", "progress_counts"}}


def _epoch_11_home(tmp_path: Path, old_roots: int) -> tuple[Path, list[str]]:
    home = tmp_path / "home"
    open_writer(home).close()
    (home / "storage-format.json").write_text('{"writer_epoch":11}\n')
    manager = SessionManager(home)
    roots = [_create_closed(manager, [], name=f"old-{index}") for index in range(old_roots)]
    for root in roots:
        _age_tree(home / "sessions" / root, days=100)
    return home, roots


@pytest.mark.asyncio
async def test_armed_owner_deletes_every_eligible_root_past_a_pinned_one(tmp_path: Path) -> None:
    home, roots = await asyncio.to_thread(_epoch_11_home, tmp_path, 4)
    lease = await asyncio.to_thread(open_writer, home)
    owner = make_retention_owner(home, lease, lambda: 30, now_ns=_future)
    signal = owner._wake = _IdleSignal()
    published: list[dict[str, object]] = []
    owner.set_publish(published.append)
    try:
        owner.start()
        await _settle(signal.idle)
        preview = published[-1]
        assert preview["mode"] == "preview" and sorted(preview["root_ids"]) == sorted(roots)
        # pinned after the preview, and not first, so it sits between eligible roots
        pinned = preview["root_ids"][1]
        await asyncio.to_thread(set_retention_pin, home, pinned, lease, pinned=True)
        assert (await asyncio.to_thread(retention_authority, home, lease, days=lambda: 30)).applied
        signal.idle.clear()
        owner.wake()
        await _settle(signal.idle)

        deleted = {root for row in published if row["mode"] == "deleted" for root in row["root_ids"]}
        assert deleted == set(roots) - {pinned}, f"{len(deleted)} of 3 eligible roots deleted"
        assert sorted(path.name for path in (home / "sessions").iterdir()) == [pinned]
        # the chain ends with a fresh complete preview, then waits for the next wake
        assert published[-1]["mode"] == "preview" and published[-1]["root_ids"] == []
    finally:
        await owner.stop()
        await asyncio.to_thread(lease.close)


@pytest.mark.asyncio
async def test_armed_deletion_steps_append_no_duplicate_history_rows(tmp_path: Path) -> None:
    home, [root] = await asyncio.to_thread(_epoch_11_home, tmp_path, 1)
    lease = await asyncio.to_thread(open_writer, home)
    owner = make_retention_owner(home, lease, lambda: 30, now_ns=_future)
    signal = owner._wake = _IdleSignal()
    published: list[dict[str, object]] = []
    owner.set_publish(published.append)
    try:
        owner.start()
        await _settle(signal.idle)
        assert (await asyncio.to_thread(retention_authority, home, lease, days=lambda: 30)).applied
        signal.idle.clear()
        owner.wake()
        await _settle(signal.idle)
        assert any(row["mode"] == "deleted" and row["root_ids"] == [root] for row in published)
        history = await asyncio.to_thread(read_retention_history, home)
        repeats = [
            (index, row["mode"]) for index, row in enumerate(history[1:], start=1)
            if _without_progress(history[index - 1]) == _without_progress(row)
        ]
        assert repeats == [], f"{len(repeats)} repeated rows in {len(history)}"
    finally:
        await owner.stop()
        await asyncio.to_thread(lease.close)


def _tui_app(home: Path) -> tuple[TUIApp, StringIO]:
    manager = SessionManager(home)
    current = manager.create(provider="codex", model="offline", cwd=home, auto_project=False)
    loop = AgentLoop(FakeBackend([]), current.store, skill_catalog=SkillCatalog.empty())
    rendered = StringIO()
    app = TUIApp(
        loop, provider="codex", model="offline", zeta_home=home,
        console=Console(file=rendered, force_terminal=False, width=200),
    )
    return app, rendered


def _tui_owner(app: TUIApp, home: Path, published: list[dict[str, object]], deleted: asyncio.Event | None = None):
    owner = make_retention_owner(home, app.loop.store._storage_lease, lambda: 30, now_ns=_future)
    _attach_retention_owner(app, owner)

    # the publish adapter cli/main.py installs for the TUI
    def publish(projection: dict[str, object]) -> None:
        published.append(projection)
        if deleted is not None and projection["mode"] == "deleted":
            deleted.set()
        if projection.get("mode") != "none":
            app._print_unit(render_retention_card(projection))

    owner.set_publish(publish)
    return owner


@pytest.mark.asyncio
async def test_tui_arm_wakes_owner_and_deletes_without_restart(tmp_path: Path) -> None:
    home, [root] = await asyncio.to_thread(_epoch_11_home, tmp_path, 1)
    app, _rendered = await asyncio.to_thread(_tui_app, home)
    published: list[dict[str, object]] = []
    deleted = asyncio.Event()
    async with owned_app(app):
        owner = _tui_owner(app, home, published, deleted)
        signal = owner._wake = _IdleSignal()
        try:
            owner.start()
            await _settle(signal.idle)
            assert published[-1]["mode"] == "preview" and published[-1]["root_ids"] == [root]
            output = await app._slash_commands.dispatch(app, "/retention arm")
            # the test never wakes the owner; the arm does
            await _settle(deleted)
            assert [row["root_ids"] for row in published if row["mode"] == "deleted"] == [[root]]
            assert not (home / "sessions" / root).exists()
            assert output.startswith("retention armed: deleting 1 sessions older than 30 days")
        finally:
            await owner.stop()


@pytest.mark.asyncio
async def test_tui_start_shows_no_card_when_nothing_changed(tmp_path: Path) -> None:
    home, _roots = await asyncio.to_thread(_epoch_11_home, tmp_path, 0)
    lease = await asyncio.to_thread(open_writer, home)
    try:
        # an earlier process: complete preview, then arm it
        earlier = make_retention_owner(home, lease, lambda: 30, now_ns=_future)
        _projections, failure = await asyncio.to_thread(earlier._run_episode)
        assert failure is None
        await earlier.stop()
        assert (await asyncio.to_thread(retention_authority, home, lease, days=lambda: 30)).applied
    finally:
        await asyncio.to_thread(lease.close)
    assert read_retention_history(home)[-1]["mode"] == "armed"

    app, rendered = await asyncio.to_thread(_tui_app, home)
    async with owned_app(app):
        owner = _tui_owner(app, home, [])
        signal = owner._wake = _IdleSignal()
        try:
            owner.start()
            await _settle(signal.idle)
            assert signal.idle.is_set()
        finally:
            await owner.stop()
    assert "retention ·" not in rendered.getvalue()


async def _retention_events_until(reader, done) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    with contextlib.suppress(TimeoutError):
        async with asyncio.timeout(30):
            while not events or not done(events[-1]):
                events.append((await _event(reader, "session_retention"))["data"])
    return events


@pytest.mark.asyncio
async def test_serve_arm_wakes_owner_and_deletes_without_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home, [root] = await asyncio.to_thread(_epoch_11_home, tmp_path, 1)
    monkeypatch.setattr(
        server_module, "make_retention_owner",
        functools.partial(make_retention_owner, now_ns=_future),
    )
    server = ZetaServer(home=home, cwd=tmp_path, port=0, provider="codex")
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.1", "features": ["session_retention"]})
        sid = (await _request(reader, writer, 2, "new_session", {"provider": "codex"}))[-1]["result"]["session"]["session_id"]
        events = await _retention_events_until(reader, lambda data: data["preview_id"] is not None)
        assert events[-1]["mode"] == "preview" and events[-1]["root_ids"] == [root]
        await _request(reader, writer, 3, "slash_run", {"session_id": sid, "text": "/retention arm"})
        events = await _retention_events_until(reader, lambda data: data["mode"] == "deleted")
        assert [data["root_ids"] for data in events if data["mode"] == "deleted"] == [[root]]
        assert not (home / "sessions" / root).exists()
    finally:
        await close_serve(server, writer)
