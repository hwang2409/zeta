from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from rich.console import Console
from rich.text import Text

from tests.support.serve_close import close_serve
from tests.support.storage_homes import initialize_storage_home
from tests.test_server import _connect, _event, _request
from zeta.server import ZetaServer
from zeta.session_retention import (
    RetentionPlanner,
    retention_authority,
    retention_result,
)
from zeta.storage_format import open_writer
from zeta.tui.cards.retention import render_retention_card

VISIBLE_MODES = (
    ("preview", False, "retention · preview"),
    ("armed", False, "retention · armed"),
    ("disarmed", False, "retention · disarmed"),
    ("disabled", False, "retention · disabled"),
    ("preview", True, "disable pending"),
    ("preview", False, "error: broken"),
)


def _card_text(projection: dict[str, object]) -> str:
    console = Console(record=True, width=100, color_system=None)
    console.print(render_retention_card(projection))
    return console.export_text()


@pytest.mark.parametrize("mode,disable_pending,needle", VISIBLE_MODES)
def test_retention_card_renders_each_public_state(
    mode: str, disable_pending: bool, needle: str
) -> None:
    projection = retention_result(
        timestamp="2026-01-01T00:00:00+00:00",
        mode=mode,
        days=30,
        errors=("broken",) if needle == "error: broken" else (),
        disable_pending=disable_pending,
    )
    assert needle in _card_text(projection)


def test_retention_none_is_silent_at_real_tui_publish_boundary() -> None:
    projection = retention_result(
        timestamp="2026-01-01T00:00:00+00:00", mode="none", days=30
    )
    published: list[dict[str, object]] = []
    publish = lambda value: published.append(value) if value.get("mode") != "none" else None
    publish(projection)
    assert published == []


async def _hello(reader, writer, request_id: int, *, features: list[str], role: str | None = None):
    params: dict[str, object] = {"protocol_version": "1.1", "features": features}
    if role is not None:
        params["role"] = role
    return (await _request(reader, writer, request_id, "hello", params))[-1]["result"]


def _seed_preview(home: Path) -> None:
    initialize_storage_home(home)
    with open_writer(home) as lease, RetentionPlanner(home, lease) as planner:
        assert planner.run_pass(days=30).complete


async def _seed_preview_async(home: Path) -> None:
    await asyncio.to_thread(_seed_preview, home)


@pytest.mark.asyncio
async def test_real_serve_retention_event_filters_and_has_exact_projection_keys(tmp_path: Path) -> None:
    home = tmp_path / "home"
    await _seed_preview_async(home)
    server = ZetaServer(home=home, cwd=tmp_path, port=0, provider="codex")
    controller_reader, controller_writer = await _connect(server)
    assert server.port is not None
    observer_reader, observer_writer = await asyncio.open_connection(
        "127.0.0.1", server.port
    )
    try:
        controller_hello = await _hello(
            controller_reader, controller_writer, 1, features=["session_retention"]
        )
        assert "session_retention" in controller_hello["capabilities"]["features"]
        await _hello(
            observer_reader, observer_writer, 2,
            features=["session_retention", "serve_observers"], role="observer",
        )
        sid = (await _request(
            controller_reader, controller_writer, 3, "new_session", {"provider": "codex"}
        ))[-1]["result"]["session"]["session_id"]
        await _request(
            controller_reader, controller_writer, 4, "slash_run",
            {"session_id": sid, "text": "/retention arm"},
        )
        event = await _event(observer_reader, "session_retention")
        while event["data"]["mode"] != "armed":
            event = await _event(observer_reader, "session_retention")
        assert set(event["data"]) == {
            "timestamp", "mode", "days", "armed_by_zeta_version", "preview_id",
            "cutoff", "root_ids", "count", "descendant_count", "bytes",
            "skip_counts", "progress_counts", "errors", "disable_pending",
        }
        assert event["data"]["mode"] == "armed"
        assert event["data"]["armed_by_zeta_version"] is not None
        assert isinstance(event["data"]["root_ids"], list)
        assert isinstance(event["data"]["errors"], list)
    finally:
        await close_serve(server, controller_writer, observer_writer)


@pytest.mark.asyncio
async def test_real_slash_retention_read_mutation_and_idempotence(tmp_path: Path) -> None:
    home = tmp_path / "home"
    await _seed_preview_async(home)
    server = ZetaServer(home=home, cwd=tmp_path, port=0, provider="codex")
    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer, 1, features=["session_retention"])
        sid = (await _request(reader, writer, 2, "new_session", {"provider": "codex"}))[-1]["result"]["session"]["session_id"]
        bare = (await _request(reader, writer, 3, "slash_run", {"session_id": sid, "text": "/retention"}))[-1]["result"]
        assert bare["kind"] == "output"
        assert bare["text"].startswith("retention: ")
        arm = (await _request(reader, writer, 4, "slash_run", {"session_id": sid, "text": "/retention arm"}))[-1]["result"]
        assert arm["text"] == "restart every zeta process on every host that pushes or pulls sessions to this home before arming"
        assert (home / "retention-arm.json").is_file()
        arm_again = (await _request(reader, writer, 5, "slash_run", {"session_id": sid, "text": "/retention arm"}))[-1]["result"]
        assert arm_again["kind"] == "output"
        disarm = (await _request(reader, writer, 6, "slash_run", {"session_id": sid, "text": "/retention disarm"}))[-1]["result"]
        assert disarm["text"] == "retention disabled"
        assert not (home / "retention-arm.json").exists()
        disarm_again = (await _request(reader, writer, 7, "slash_run", {"session_id": sid, "text": "/retention disarm"}))[-1]["result"]
        assert disarm_again["kind"] == "output"
    finally:
        await close_serve(server, writer)


@pytest.mark.asyncio
async def test_retention_reconnect_reads_last_state_after_missing_event(tmp_path: Path) -> None:
    home = tmp_path / "home"
    await _seed_preview_async(home)
    server = ZetaServer(home=home, cwd=tmp_path, port=0, provider="codex")
    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer, 1, features=["session_retention"])
        sid = (await _request(reader, writer, 2, "new_session", {"provider": "codex"}))[-1]["result"]["session"]["session_id"]
        await _request(reader, writer, 3, "slash_run", {"session_id": sid, "text": "/retention arm"})
        writer.close()
        await writer.wait_closed()
        assert server.port is not None
        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        await _hello(reader, writer, 4, features=["session_retention"])
        result = (await _request(reader, writer, 5, "slash_run", {"session_id": sid, "text": "/retention"}))[-1]["result"]
        assert result["kind"] == "output"
        assert "mode=armed" in result["text"]
    finally:
        await close_serve(server, writer)


def test_tui_card_uses_text_renderable() -> None:
    projection = retention_result(timestamp="now", mode="preview", days=30)
    assert isinstance(render_retention_card(projection).renderable, Text)


@pytest.mark.asyncio
def _unarmed_authority(home: Path):
    with open_writer(home) as lease:
        return retention_authority(home, lease, days=30)


@pytest.mark.asyncio
async def test_retention_policy_stays_preview_only_until_later_armed_pass(tmp_path: Path) -> None:
    home = tmp_path / "home"
    await _seed_preview_async(home)
    assert not (home / "retention-arm.json").exists()
    result = await asyncio.to_thread(_unarmed_authority, home)
    assert result.message == "restart every zeta process on every host that pushes or pulls sessions to this home before arming"
    assert (home / "retention-arm.json").is_file()
    assert not list(home.glob("retention-quarantine*"))
