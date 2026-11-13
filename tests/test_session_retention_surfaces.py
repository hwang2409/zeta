from __future__ import annotations

import asyncio
import fcntl
import multiprocessing
import os
from pathlib import Path

import pytest
from rich.console import Console
from rich.text import Text

from tests.support.serve_close import close_serve
from tests.support.storage_homes import initialize_storage_home
from tests.test_server import _connect, _event, _request
from zeta.core.session import SessionManager
from zeta.locking import RETENTION_HOME_ID, domain_lock_name
from zeta.protocol.types import (
    Message,
    MessageOrigin,
    MessageRole,
    TextContent,
    ToolCall,
    ToolUseContent,
    with_message_origin,
)
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


SERVE_RETENTION_STATES = ("none", "idle", "mid-preview", "crashed", "armed-mid-pass")
SERVE_RETENTION_ROUTES = (
    "resume", "resume-history-backward", "resume-approval-setup", "resume-yolo",
    "resume-corrupt-transcript", "resume-reconnect", "list-full", "list-paged",
    "list-project", "delete", "rename",
)


def _seed_matrix_session(home: Path, cwd: Path, route: str) -> str:
    """Build the same durable fixtures used by the server resume regressions."""
    initialize_storage_home(home)
    opened = SessionManager(home).create(
        provider="codex", model="offline", cwd=cwd
    )
    sid = opened.metadata.session_id
    if route == "resume-history-backward":
        for index in range(6):
            opened.store.append_message(
                with_message_origin(
                    Message(MessageRole.USER, [TextContent(f"prefix {index}")]),
                    MessageOrigin.USER,
                )
            )
    elif route == "resume-approval-setup":
        call = ToolCall("matrix-call", "read", {"path": str(cwd / "input")})
        opened.store.append_message_with_approval_requests(
            Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
            [(call.id, call)],
        )
    elif route == "resume-corrupt-transcript":
        path = opened.store.path
        opened.store.close()
        with path.open("ab") as stream:
            stream.write(b'{"private":"must-not-leak"\n')
        return sid
    opened.store.close()
    if route == "resume-yolo":
        (cwd / "settings.toml").write_text("yolo = true\\n")
    return sid


def _retention_lock_path(home: Path, domain: str) -> Path:
    return home / ".domain-locks" / domain_lock_name(RETENTION_HOME_ID, domain)


def _hold_retention_root(home_text: str, session_id: str, ready, command, crashed: bool) -> None:
    home = Path(home_text)
    active = os.open(_retention_lock_path(home, "planner-active"), os.O_RDWR)
    root = os.open(home / "sessions" / session_id, os.O_RDONLY)
    fcntl.flock(active, fcntl.LOCK_SH)
    fcntl.flock(root, fcntl.LOCK_EX)
    ready.send(True)
    if crashed:
        return
    from zeta.session_retention.handshake import planner_request_pending

    while not planner_request_pending(home) and not command.poll():
        pass
    fcntl.flock(root, fcntl.LOCK_UN)
    fcntl.flock(active, fcntl.LOCK_UN)
    os.close(root)
    os.close(active)


def _start_retention_holder(home: Path, session_id: str, *, crashed: bool = False):
    lock_dir = home / ".domain-locks"
    lock_dir.mkdir(mode=0o700, exist_ok=True)
    for domain in ("planner-active", "user-open-request"):
        _retention_lock_path(home, domain).touch(exist_ok=True)
    context = multiprocessing.get_context("spawn")
    ready_recv, ready_send = context.Pipe(False)
    command_recv, command_send = context.Pipe(False)
    process = context.Process(
        target=_hold_retention_root,
        args=(str(home), session_id, ready_send, command_recv, crashed),
    )
    process.start()
    assert ready_recv.recv()
    return process, command_send


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_state", SERVE_RETENTION_STATES)
@pytest.mark.parametrize("route", SERVE_RETENTION_ROUTES)
async def test_real_serve_access_matrix_never_busy_or_partial(
    tmp_path: Path, owner_state: str, route: str,
) -> None:
    """Every serve access keeps its normal result while preview owns a root."""
    home = tmp_path / "home"
    sid = await asyncio.to_thread(_seed_matrix_session, home, tmp_path, route)
    if owner_state == "armed-mid-pass":
        (home / "retention-arm.json").write_text("{}\n")
    server = ZetaServer(home=home, cwd=tmp_path, port=0, provider="codex")
    reader, writer = await _connect(server)
    holder = command = None
    try:
        await _hello(
            reader,
            writer,
            1,
            features=["list_sessions_paging", "projects", "session_replay_seq"],
        )
        if owner_state not in {"none", "idle"}:
            holder, command = _start_retention_holder(
                home, sid, crashed=owner_state == "crashed"
            )
        if route.startswith("resume") and route != "resume-reconnect":
            frames = await _request(reader, writer, 2, "resume", {"session_id": sid})
            result = next(frame for frame in reversed(frames) if "result" in frame or "error" in frame)
            if route == "resume-corrupt-transcript":
                assert result["error"]["data"]["code"] == "corrupt_transcript"
            else:
                assert result.get("result", {}).get("session", {}).get("session_id") == sid, result
                if route == "resume-history-backward":
                    pages: list[tuple[int, dict[str, object]]] = []
                    offset = 0
                    while True:
                        page = (await _request(reader, writer, f"page-{offset}", "session_history", {"session_id": sid, "offset": offset}))[-1]["result"]
                        pages.append((offset, page))
                        if page["next_offset"] is None:
                            break
                        offset = page["next_offset"]
                    forward = [row for _, page in pages for row in page["messages"]]
                    backward_pages: dict[int, list[dict[str, object]]] = {}
                    for offset, _ in reversed(pages):
                        backward_pages[offset] = (await _request(reader, writer, f"back-{offset}", "session_history", {"session_id": sid, "offset": offset}))[-1]["result"]["messages"]
                    assert [row for offset, _ in pages for row in backward_pages[offset]] == forward
        elif route == "resume-reconnect":
            resumed = (await _request(reader, writer, 2, "resume", {"session_id": sid}))[-1]
            assert resumed.get("result", {}).get("session", {}).get("session_id") == sid, resumed
            first_response = (await _request(reader, writer, 3, "session_history_and_subscribe", {"session_id": sid}))[-1]
            assert "result" in first_response, first_response
            first = first_response["result"]
            assert first["replay"]["history_seq"] == 0
            writer.close()
            await writer.wait_closed()
            reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
            await _hello(reader, writer, 3, features=["session_replay_seq"])
            result = (await _request(reader, writer, 4, "session_history_and_subscribe", {"session_id": sid}))[-1]
            assert result.get("result", {}).get("replay", {}).get("history_seq") == 0, result
        else:
            method = "list_sessions" if route.startswith("list-") else route
            params = {"session_id": sid}
            if route == "list-full":
                params = {}
            elif route == "list-paged":
                params = {"offset": 0, "limit": 1}
            elif route == "list-project":
                params = {"project_id": "missing-project"}
            elif route == "delete":
                method, params = "delete_session", {"session_id": sid}
            elif route == "rename":
                method, params = "rename_session", {"session_id": sid, "name": "matrix"}
            result = next(frame for frame in reversed(await _request(reader, writer, 2, method, params)) if "result" in frame or "error" in frame)
            if route == "list-project":
                assert result.get("error", {}).get("code") == -32602, result
                assert result["error"]["data"]["code"] == "project_not_found"
            else:
                assert "error" not in result, result
                assert result.get("result") is not None
            if route.startswith("list-") and route != "list-project":
                rows = result["result"]["sessions"]
                assert sid in {row["session_id"] for row in rows}
            if route == "rename":
                assert result["result"]["session"]["session_id"] == sid
        if owner_state == "mid-preview" and route != "list-project":
            assert holder is not None
            assert not holder.is_alive()
            if route in {"list-full", "list-paged"}:
                assert sid in {row["session_id"] for row in rows}
    finally:
        if holder is not None and holder.is_alive():
            command.send(True)
            holder.join(10)
        await close_serve(server, writer)
        if holder is not None:
            assert holder.exitcode == 0
        if owner_state == "mid-preview":
            await asyncio.to_thread(_seed_preview, home)
            assert (home / "retention-state.json").is_file()
