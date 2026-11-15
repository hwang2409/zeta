"""armed continuation, arm wake, history dedupe, and start-card dedupe (s4c F1/F3/F4/F8)."""

from __future__ import annotations

import asyncio
import contextlib
import errno
import fcntl
import functools
import os
import re
import time
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

import zeta.server.server as server_module
import zeta.session_retention.transaction as transaction_module
from tests.support.fake_backend import FakeBackend
from tests.support.owned import owned_app
from tests.support.retention_owner import _age_tree, _create_closed
from tests.support.serve_close import close_serve
from tests.test_server import _connect, _event, _request
from zeta.cli.main import _attach_retention_owner
from zeta.core.session import SessionManager
from zeta.locking import RETENTION_HOME_ID, domain_lock_name
from zeta.runtime.loop import AgentLoop
from zeta.server import ZetaServer
from zeta.session_retention import (
    RetentionBounds,
    RetentionPlanner,
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


class _NoSpaceForTransactions:
    """transaction.py's ``os``: only the per-transaction txid mkdir fails, with ENOSPC."""

    def __init__(self) -> None:
        self.attempts = 0

    def __getattr__(self, name: str) -> object:
        return getattr(os, name)

    def mkdir(self, path, mode=0o777, *, dir_fd=None) -> None:
        if dir_fd is not None and re.fullmatch("[0-9a-f]{32}", str(path)):
            self.attempts += 1
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
        os.mkdir(path, mode, dir_fd=dir_fd)


class _Request:
    """The real user-open request handshake lock, held the way a user open holds it."""

    def __init__(self, home: Path) -> None:
        self.fd = os.open(
            home / ".domain-locks" / domain_lock_name(RETENTION_HOME_ID, "user-open-request"),
            os.O_RDWR,
        )

    def hold(self) -> None:
        fcntl.flock(self.fd, fcntl.LOCK_SH)

    def close(self) -> None:
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        os.close(self.fd)


# row: (old roots, what happens after the arm). every row ends with the owner idle.
_OWNER_ROWS = {
    "deleted": 2,
    "pinned-head": 3,
    "opened-head": 3,
    "repeated-outcome": 1,
    "error-after-changed": 1,
    "final-walk-yield": 2,
    "final-walk-budget": 3,
    "user-open-yield": 2,
}


@pytest.mark.asyncio
@pytest.mark.parametrize("row", tuple(_OWNER_ROWS))
async def test_armed_owner_continue_stop_or_yield_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, row: str,
) -> None:
    """When the real owner continues, ends the episode and waits, or yields (s4c F1)."""
    home, roots = await asyncio.to_thread(_epoch_11_home, tmp_path, _OWNER_ROWS[row])
    lease = await asyncio.to_thread(open_writer, home)
    clock_offset = [0]
    owner = make_retention_owner(
        home, lease, lambda: 30, now_ns=lambda: _future() + clock_offset[0],
    )
    signal = owner._wake = _IdleSignal()
    published: list[dict[str, object]] = []
    chain_done = asyncio.Event()

    def publish(projection: dict[str, object]) -> None:
        published.append(projection)
        if projection["mode"] == "preview" and projection["preview_id"] and projection["root_ids"] == []:
            chain_done.set()

    owner.set_publish(publish)
    passes = [0]
    original_pass = RetentionPlanner.run_pass

    def counted_pass(planner, **kwargs):
        passes[0] += 1
        # a spin guard, not a proof: the owner fails this pass, so even a spinning base goes idle
        if passes[0] > 500:
            raise RuntimeError("owner spun past 500 passes")
        result = original_pass(planner, **kwargs)
        if row == "repeated-outcome" and getattr(result, "mode", "") is None and result.changed:
            # the root id is published again right after its quarantine step
            (home / "sessions" / roots[0]).mkdir(exist_ok=True)
        return result

    monkeypatch.setattr(RetentionPlanner, "run_pass", counted_pass)
    episodes = [0]
    original_episode = owner._run_episode

    def counted_episode():
        episodes[0] += 1
        return original_episode()

    owner._run_episode = counted_episode
    in_final_walk: list[str] = []
    original_walk = RetentionPlanner._walk

    def walk(planner, flight, budget):
        # fires once, in the first walk after the arm: the transaction's final candidate walk
        if in_final_walk and flight.root_id == in_final_walk[0]:
            in_final_walk.clear()
            if row == "final-walk-yield":
                request.hold()
            else:
                clock_offset[0] += RetentionBounds().worker_ns
        return original_walk(planner, flight, budget)

    # test-contract: seam times a real user-open request or a 30 s clock step inside the final walk
    monkeypatch.setattr(RetentionPlanner, "_walk", walk)
    no_space = _NoSpaceForTransactions()
    if row == "error-after-changed":
        monkeypatch.setattr(transaction_module, "os", no_space)
    request = opened = None
    try:
        owner.start()
        await _settle(signal.idle)
        preview = published[-1]
        assert preview["mode"] == "preview" and sorted(preview["root_ids"]) == sorted(roots)
        head = preview["root_ids"][0]
        request = await asyncio.to_thread(_Request, home)
        if row == "pinned-head":
            await asyncio.to_thread(set_retention_pin, home, head, lease, pinned=True)
        elif row == "opened-head":
            # the user holds this session open for the whole chain
            opened = await asyncio.to_thread(SessionManager(home).open, head)
        elif row in {"final-walk-yield", "final-walk-budget"}:
            in_final_walk.append(head)
        elif row == "user-open-yield":
            request.hold()
        assert (await asyncio.to_thread(retention_authority, home, lease, days=lambda: 30)).applied
        signal.idle.clear()
        before = passes[0]
        owner.wake()
        await _settle(signal.idle)
        armed_passes = passes[0] - before

        def deleted() -> set[str]:
            return {root for item in published if item["mode"] == "deleted" for root in item["root_ids"]}

        errors = [error for item in published for error in item.get("errors") or ()]
        left = sorted(path.name for path in (home / "sessions").iterdir())
        if row in {"deleted", "pinned-head", "opened-head", "final-walk-budget"}:
            kept = set() if row == "deleted" else {head}
            # one wake deletes every other root: a rejected head does not end the chain
            assert deleted() == set(roots) - kept, f"{len(deleted())} of {len(roots) - len(kept)} eligible roots deleted"
            assert left == sorted(kept)
            # the armed set is done: the chain ends on a fresh complete preview, then waits
            assert published[-1]["mode"] == "preview" and published[-1]["preview_id"] is not None
            if row == "final-walk-budget":
                from zeta.session_retention import FINAL_WALK_BUDGET_REASON

                assert errors == [FINAL_WALK_BUDGET_REASON], errors
                # the kept root is old and closed, so the next preview lists it again
                assert published[-1]["root_ids"] == [head]
            else:
                assert published[-1]["root_ids"] == []
        elif row == "repeated-outcome":
            # quarantine, then the republished id blocks the root: once a step, once a repeat
            assert armed_passes == 3, f"{armed_passes} passes"
            assert left == [head] and errors == ["id_republished"], errors
        elif row == "error-after-changed":
            # the failed transaction ends the episode; it is not progress to repeat at once
            assert armed_passes == 1, f"{armed_passes} passes, {no_space.attempts} failed mkdirs"
            assert errors == [str(OSError(errno.ENOSPC, os.strerror(errno.ENOSPC)))], errors
            assert left == roots
            signal.idle.clear()
            owner.wake()
            await _settle(signal.idle)
            # the next wake retries once; the same error is not published again
            assert (passes[0] - before, no_space.attempts) == (2, 2)
            assert [error for item in published for error in item.get("errors") or ()] == errors
        else:
            # a user open yields: no error card, nothing deleted while the request is held
            assert errors == [] and deleted() == set(), (errors, deleted())
            assert left == sorted(roots)
            if row == "user-open-yield":
                # held at episode start: each retry is one episode with no pass, then a wait
                held = (episodes[0], passes[0])
                signal.idle.clear()
                await _settle(signal.idle)
                assert (episodes[0], passes[0]) == (held[0] + 1, held[1]), (held, episodes[0], passes[0])
            # released: the owner resumes by itself, with no wake
            await asyncio.to_thread(request.close)
            request = None
            await _settle(chain_done)
            assert deleted() == set(roots), f"{len(deleted())} of {len(roots)} roots deleted after the user open"
            assert [error for item in published for error in item.get("errors") or ()] == []
    finally:
        if request is not None:
            await asyncio.to_thread(request.close)
        if opened is not None:
            await asyncio.to_thread(opened.store.close)
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


_RESTART = "restart every zeta process on every host that pushes or pulls sessions to this home before arming"
_KEEPS = "arming keeps deleting closed, unpinned sessions older than 30 days until /retention disarm"


@pytest.mark.asyncio
async def test_tui_retention_text_before_arming_and_after_a_completed_chain(tmp_path: Path) -> None:
    home, [root] = await asyncio.to_thread(_epoch_11_home, tmp_path, 1)
    app, _rendered = await asyncio.to_thread(_tui_app, home)
    published: list[dict[str, object]] = []
    async with owned_app(app):
        owner = _tui_owner(app, home, published)
        signal = owner._wake = _IdleSignal()
        try:
            owner.start()
            await _settle(signal.idle)
            assert published[-1]["mode"] == "preview" and published[-1]["root_ids"] == [root]
            unarmed = await app._slash_commands.dispatch(app, "/retention")
            assert _RESTART in unarmed and _KEEPS in unarmed
            signal.idle.clear()
            arm = await app._slash_commands.dispatch(app, "/retention arm")
            assert _KEEPS in arm
            await _settle(signal.idle)
            assert not (home / "sessions" / root).exists()
            # the chain ended on a fresh preview row; the arm file still stands
            assert read_retention_history(home)[-1]["mode"] == "preview"
            armed = await app._slash_commands.dispatch(app, "/retention")
            assert "before arming" not in armed and _KEEPS not in armed
        finally:
            await owner.stop()


@pytest.mark.asyncio
async def test_serve_retention_text_before_arming_and_after_a_completed_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home, [root] = await asyncio.to_thread(_epoch_11_home, tmp_path, 1)
    monkeypatch.setattr(
        server_module, "make_retention_owner",
        functools.partial(make_retention_owner, now_ns=_future),
    )
    server = ZetaServer(home=home, cwd=tmp_path, port=0, provider="codex")
    reader, writer = await _connect(server)

    async def slash(request_id: int, text: str) -> str:
        reply = await _request(reader, writer, request_id, "slash_run", {"session_id": sid, "text": text})
        return reply[-1]["result"]["text"]

    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.1", "features": ["session_retention"]})
        sid = (await _request(reader, writer, 2, "new_session", {"provider": "codex"}))[-1]["result"]["session"]["session_id"]
        events = await _retention_events_until(reader, lambda data: data["preview_id"] is not None)
        assert events[-1]["mode"] == "preview" and events[-1]["root_ids"] == [root]
        unarmed = await slash(3, "/retention")
        assert _RESTART in unarmed and _KEEPS in unarmed
        assert _KEEPS in await slash(4, "/retention arm")
        await _retention_events_until(reader, lambda data: data["mode"] == "deleted")
        # the chain ends on a fresh complete preview while the arm file still stands
        events = await _retention_events_until(
            reader, lambda data: data["mode"] == "preview" and data["preview_id"] is not None
        )
        assert events[-1]["root_ids"] == [] and not (home / "sessions" / root).exists()
        armed = await slash(5, "/retention")
        assert "before arming" not in armed and _KEEPS not in armed
    finally:
        await close_serve(server, writer)
