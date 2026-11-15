"""skips are shown as skips, and bare `/retention` below the epoch says how to enable it (s4c F6/F7)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.support.owned import owned_app
from tests.support.retention_owner import _age_tree, _create_closed
from tests.support.serve_close import close_serve
from tests.test_server import _connect, _request
from tests.test_session_retention_continuation import (
    _IdleSignal,
    _retention_events_until,
    _settle,
    _tui_app,
)
from zeta.cli.main import _attach_retention_owner
from zeta.core.session import SessionManager
from zeta.server import ZetaServer
from zeta.session_retention.activation import make_retention_owner
from zeta.session_retention.pin import set_retention_pin
from zeta.storage_format import open_writer
from zeta.tui.cards.retention import render_retention_card

_HINT = "retention unavailable: upgrade the home before arming"
# the surface's own open session is the in-use root
_SKIPS = {"pinned": 1, "not old enough": 1, "session is open or in use": 1}


def _home(tmp_path: Path, epoch: int = 11) -> Path:
    """An old pinned root and a young root; the surface adds its own open session."""
    home = tmp_path / "home"
    open_writer(home).close()
    (home / "storage-format.json").write_text(f'{{"writer_epoch":{epoch}}}\n')
    if epoch == 11:
        manager = SessionManager(home)
        pinned = _create_closed(manager, [], name="old-pinned")
        _age_tree(home / "sessions" / pinned, days=100)
        _create_closed(manager, [], name="young")
        with open_writer(home) as lease:
            set_retention_pin(home, pinned, lease, pinned=True)
    return home


@pytest.mark.asyncio
async def test_tui_preview_card_shows_kept_roots_as_skips(tmp_path: Path) -> None:
    home = await asyncio.to_thread(_home, tmp_path)
    app, rendered = await asyncio.to_thread(_tui_app, home)
    published: list[dict[str, object]] = []
    async with owned_app(app):
        # the real clock: the young root must stay young
        owner = make_retention_owner(home, app.loop.store._storage_lease, lambda: 30)
        _attach_retention_owner(app, owner)

        def publish(projection: dict[str, object]) -> None:
            # the publish adapter cli/main.py installs for the TUI
            published.append(projection)
            if projection.get("mode") != "none":
                app._print_unit(render_retention_card(projection))

        owner.set_publish(publish)
        signal = owner._wake = _IdleSignal()
        try:
            owner.start()
            await _settle(signal.idle)
        finally:
            await owner.stop()
    preview = published[-1]
    assert preview["mode"] == "preview" and preview["preview_id"] is not None
    assert (preview["skip_counts"], preview["errors"], preview["root_ids"]) == (_SKIPS, [], [])
    card = rendered.getvalue()
    assert [line.strip(" │") for line in card.splitlines() if "skipped:" in line] == [
        "skipped: 1 not old enough", "skipped: 1 pinned", "skipped: 1 session is open or in use",
    ]
    assert "error:" not in card


@pytest.mark.asyncio
async def test_serve_preview_event_counts_kept_roots_as_skips(tmp_path: Path) -> None:
    home = await asyncio.to_thread(_home, tmp_path)
    server = ZetaServer(home=home, cwd=tmp_path, port=0, provider="codex")
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.1", "features": ["session_retention"]})
        await _request(reader, writer, 2, "new_session", {"provider": "codex"})
        events = await _retention_events_until(reader, lambda data: data["preview_id"] is not None)
        assert events and events[-1]["preview_id"] is not None
        preview = events[-1]
        assert (preview["skip_counts"], preview["errors"], preview["root_ids"]) == (_SKIPS, [], [])
        assert all(not data["errors"] for data in events)
    finally:
        await close_serve(server, writer)


@pytest.mark.asyncio
async def test_bare_retention_below_the_epoch_says_upgrade_on_both_surfaces(tmp_path: Path) -> None:
    home = await asyncio.to_thread(_home, tmp_path, 10)
    app, _rendered = await asyncio.to_thread(_tui_app, home)
    async with owned_app(app):
        tui = await app._slash_commands.dispatch(app, "/retention")
    server = ZetaServer(home=home, cwd=tmp_path, port=0, provider="codex")
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.1", "features": ["session_retention"]})
        sid = (await _request(reader, writer, 2, "new_session", {"provider": "codex"}))[-1]["result"]["session"]["session_id"]
        serve = (await _request(reader, writer, 3, "slash_run", {"session_id": sid, "text": "/retention"}))[-1]["result"]["text"]
    finally:
        await close_serve(server, writer)
    assert (tui, serve) == (_HINT, _HINT)
    assert not (home / "retention-history.jsonl").exists()
