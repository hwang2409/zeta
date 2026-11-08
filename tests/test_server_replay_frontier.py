"""Replay frames that a joiner's history already shows are not sent again.

Every test runs a real ``zeta serve`` over a unix socket and a websocket with a
real store. A replay frame with a history row carries ``covers``, the seq of that
row; a stream drops it when the row is inside the history it joined with.
"""
from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tests.support.fake_backend import FakeBackend, PerActorFakeBackend, ScriptedTurn
from tests.support.owned import await_or_fail
from tests.support.replay_transport import (
    ReplayClient,
    Transport,
    connect_replay,
    negotiate,
    replay_server,
)
from zeta.core.store import _row_history
from zeta.model_input import ModelInputEnvelope
from zeta.protocol.types import (
    Message,
    MessageOrigin,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
)
from zeta.providers.completion import owned_completion
from zeta.server import ZetaServer
from zeta.server.protocol import EventDescriptor
from zeta.server.session_replay import DeferredFrame, rows

pytest_plugins = ("tests.support.socket_fixtures",)

TRANSPORTS: tuple[Transport, ...] = ("unix", "websocket")
CONTROLLER_FEATURES = ("delivery_id", "user_message_event", "user_message_delivery_id", "model_input_ids")
Route = tuple[EventDescriptor, str]


@dataclass
class Serve:
    server: ZetaServer
    transport: Transport
    controller: ReplayClient
    observer: ReplayClient
    session_id: str
    clients: list[ReplayClient] = field(default_factory=list)
    # (event, fields, publisher facts, covers) of every replay-routed publication
    published: list[tuple[EventDescriptor, dict[str, Any], rows.RowFacts, Any]] = field(
        default_factory=list
    )
    next_id: int = 100

    def request_id(self) -> int:
        self.next_id += 1
        return self.next_id

    @property
    def store(self) -> Any:
        return self.server.runtime.opened.store

    async def client(self) -> ReplayClient:
        client = await connect_replay(self.server, self.transport)
        self.clients.append(client)
        await negotiate(client, observer=True, extra=CONTROLLER_FEATURES, protocol_version="1.2")
        return client

    async def join(self, client: ReplayClient) -> dict[str, Any]:
        response = await client.request(
            self.request_id(), "session_history_and_subscribe", {"session_id": self.session_id}
        )
        assert "error" not in response, response
        return response["result"]

    async def approve(self, request_id: str) -> None:
        response = await self.controller.request(
            self.request_id(), "approve", {"request_id": request_id}
        )
        assert response["result"]["accepted"] is True, response

    async def live(self, client: ReplayClient) -> list[dict[str, Any]]:
        """Every frame the client received before a later status response."""
        status = self.request_id()
        await client.send(status, "status", {})
        frames: list[dict[str, Any]] = []
        while (frame := await client.read()).get("id") != status:
            if "params" in frame:
                frames.append(frame["params"])
        return frames

    def records(self, route: Route) -> list[tuple[dict[str, Any], rows.RowFacts, Any]]:
        event, variant = route
        return [
            (fields, facts, covers)
            for published, fields, facts, covers in self.published
            if published is event and rows.variant(event, fields, facts) == variant
        ]


async def _serve(tmp_path: Path, transport: Transport, backend: Any) -> Serve:
    server = replay_server(tmp_path, transport, lambda provider, model, home: (backend, model or "offline"))
    await server.start()
    controller = await connect_replay(server, transport)
    observer = await connect_replay(server, transport)
    await negotiate(controller, extra=CONTROLLER_FEATURES, protocol_version="1.2")
    created = await controller.request(2, "new_session", {"provider": "codex"})
    await negotiate(observer, observer=True, protocol_version="1.2")
    serve = Serve(server, transport, controller, observer, created["result"]["session"]["session_id"])
    assert server.connections is not None
    original = server.connections._replay_covers

    def recorded(
        event: EventDescriptor, session_id: str | None, fields: dict[str, Any], replay_row: Any, *rest: Any
    ) -> Any:
        covers = original(event, session_id, fields, replay_row, *rest)
        # a publish-first frame is recorded when its deferred copy is admitted
        if not isinstance(covers, DeferredFrame):
            serve.published.append((event, dict(fields), replay_row or rows.RowFacts(), covers))
        return covers

    admit_deferred = server.connections._admit_deferred

    def recorded_deferred(record: DeferredFrame, covers: Any) -> None:
        serve.published.append((record.event, dict(record.fields), record.facts, covers))
        admit_deferred(record, covers)

    # test-contract: seam observe each publication's internal covers before the wire strips it
    server.connections._replay_covers = recorded  # type: ignore[method-assign]
    # test-contract: seam observe each deferred copy's covers when its row is found
    server.connections._admit_deferred = recorded_deferred  # type: ignore[method-assign]
    return serve


async def _close(serve: Serve) -> None:
    for client in (serve.controller, serve.observer, *serve.clients):
        await client.close()
    await serve.server.close()


def _texts(rows_: list[dict[str, Any]]) -> list[str]:
    return [
        block["text"]
        for row in rows_
        for block in row.get("content") or []
        if isinstance(block, dict) and isinstance(block.get("text"), str)
    ]


def _history_rows_for(
    event: EventDescriptor, fields: dict[str, Any], facts: rows.RowFacts, history: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The history rows that present the same effect as one live frame."""
    data = fields.get("data") or {}
    if isinstance(data, dict) and "notification_id" in data:
        return [row for row in history if row["id"] == data["notification_id"]]
    if event in {EventDescriptor.APPROVAL_REQUEST, EventDescriptor.APPROVAL_END}:
        return [
            row
            for row in history
            if row.get("event") == event.value and row.get("request_id") == fields["request_id"]
        ]
    if event is EventDescriptor.DELIVERY_STATUS:
        return [
            row
            for row in history
            if row.get("event") == event.value and row.get("delivery_id") == fields["delivery_id"]
        ]
    if event is EventDescriptor.USER_MESSAGE and facts.delivery_id is None:
        return [row for row in history if row.get("role") == "user" and any(fields["text"] in text for text in _texts([row]))]
    if event is EventDescriptor.ASSISTANT_MESSAGE:
        return [
            row
            for row in history
            if row.get("role") == "assistant" and _texts([row]) == _texts([fields["message"]])
        ]
    if event is EventDescriptor.USER_MESSAGE:
        return [
            row
            for row in history
            if row.get("role") == "user" and row.get("delivery_id") == facts.delivery_id
        ]
    if event is EventDescriptor.TOOL_END:
        call_id = fields["tool_result"]["tool_call_id"]
        return [
            row
            for row in history
            if isinstance(row.get("tool_result"), dict)
            and row["tool_result"].get("tool_call_id") == call_id
        ]
    raise AssertionError(f"no history-row matcher for {event.value}")


async def _wait_event(client: ReplayClient, name: str, **match: Any) -> dict[str, Any]:
    while True:
        params = await client.event(name)
        if all(_lookup(params, key) == value for key, value in match.items()):
            return params


def _lookup(params: dict[str, Any], dotted: str) -> Any:
    value: Any = params
    for part in dotted.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return value


# --- scenarios: each drives one real serve flow with the observer already joined ---


def _text_backend(*texts: str) -> FakeBackend:
    return FakeBackend([ScriptedTurn([TextContent(text)]) for text in texts])


class BlockingBackend:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    @owned_completion
    async def complete(self, messages: Any, tool_schemas: Any) -> Any:
        del messages, tool_schemas
        self.started.set()
        await await_or_fail(self.release.wait(), what="blocked turn release")
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, [TextContent("done")]),
        )


async def _delivery(serve: Serve) -> None:
    await serve.controller.request(serve.request_id(), "send", {"text": "delivery-marker", "delivery_id": "d-1"})
    await _wait_event(serve.observer, "user_message", text="delivery-marker")
    await _wait_event(serve.observer, "agent_end")


async def _send_plain(serve: Serve) -> None:
    await serve.controller.request(serve.request_id(), "send", {"text": "plain-marker"})
    await _wait_event(serve.observer, "user_message", text="plain-marker")
    await _wait_event(serve.observer, "agent_end")


PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6360f8cf00000004000101a2e0c4b00000000049454e44ae426082"
)


async def _send_images(serve: Serve) -> None:
    image = {"name": "shot.png", "mime_type": "image/png", "data": base64.b64encode(PNG).decode()}
    await serve.controller.request(
        serve.request_id(), "send_images", {"session_id": serve.session_id, "text": "image-marker", "images": [image]}
    )
    await _wait_event(serve.observer, "user_message", text="image-marker")
    await _wait_event(serve.observer, "agent_end")


async def _send_skill(serve: Serve) -> None:
    input_id = serve.server.control._model_inputs.register(
        serve.session_id,
        ModelInputEnvelope("skill-marker expanded body", "skill-marker", MessageOrigin.SKILL_EXPANSION),
    )
    await serve.controller.request(serve.request_id(), "send", {"input_id": input_id})
    await _wait_event(serve.observer, "user_message", text="skill-marker")
    await _wait_event(serve.observer, "agent_end")


class TruncatedBackend:
    """A reply cut off mid-stream: its row is written as ``failed``."""

    @owned_completion
    async def complete(self, messages: Any, tool_schemas: Any) -> Any:
        del messages, tool_schemas
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, [TextContent("failed-marker")]),
            data={"truncated": True},
        )


async def _assistant_failed(serve: Serve) -> None:
    await serve.controller.request(serve.request_id(), "send", {"text": "go"})
    await _wait_event(serve.observer, "assistant_message")
    await _wait_event(serve.observer, "agent_end")


async def _delivery_after_newer_row(serve: Serve) -> None:
    """The echo is admitted after a newer row installed, so its row is not the newest."""
    release = asyncio.Event()
    entered = asyncio.Event()
    original = serve.server.control.user_message

    async def held(*args: Any, **kwargs: Any) -> None:
        entered.set()
        await await_or_fail(release.wait(), what="release user echo")
        await original(*args, **kwargs)

    # test-contract: seam hold the echo after its delivery row installed
    serve.server.control.user_message = held  # type: ignore[method-assign]
    send = asyncio.create_task(
        serve.controller.request(serve.request_id(), "send", {"text": "delivery-marker", "delivery_id": "d-1"})
    )
    try:
        await await_or_fail(entered.wait(), what="delivery row installed")
        serve.store.append_task_notification(task_id="newer", command="true", exit_code=0)
        release.set()
        await await_or_fail(send, what="send response")
    finally:
        release.set()
        serve.server.control.user_message = original  # type: ignore[method-assign]
    await _wait_event(serve.observer, "user_message", text="delivery-marker")
    await _wait_event(serve.observer, "agent_end")


def _read_call(tmp_path: Path, call_id: str = "call-fg") -> ToolCall:
    target = tmp_path / f"{call_id}.txt"
    target.write_text("data", encoding="utf-8")
    return ToolCall(call_id, "read", {"path": str(target)})


async def _foreground_tool(serve: Serve) -> None:
    await serve.controller.request(serve.request_id(), "send", {"text": "run"})
    request = await _wait_event(serve.observer, "approval_request")
    await serve.approve(request["request_id"])
    await _wait_event(serve.observer, "tool_end", **{"tool_call.id": "call-fg"})
    await _wait_event(serve.observer, "agent_end")


def _foreground_tool_backend(tmp_path: Path) -> FakeBackend:
    return FakeBackend(
        [ScriptedTurn(tool_calls=[_read_call(tmp_path)]), ScriptedTurn([TextContent("done")])]
    )


def _background_backend(tmp_path: Path) -> PerActorFakeBackend:
    agent = ToolCall(
        "agent-bg", "agent", {"prompt": "child-work", "description": "child", "background": True}
    )
    return PerActorFakeBackend(
        [
            ScriptedTurn(tool_calls=[agent]),
            ScriptedTurn([TextContent("parent done")]),
            ScriptedTurn([TextContent("noted")]),
        ],
        {
            "child-work": [
                ScriptedTurn(tool_calls=[_read_call(tmp_path, "call-child")]),
                ScriptedTurn([TextContent("child done")]),
            ]
        },
        parent_prompt="delegate",
    )


async def _background(serve: Serve) -> None:
    await serve.controller.request(serve.request_id(), "send", {"text": "delegate"})
    request = await _wait_event(serve.observer, "approval_request", delegated=True)
    await serve.approve(request["request_id"])
    terminal = await _wait_event(serve.observer, "tool_end", **{"tool_call.id": "agent-bg"})
    while "notification_id" not in terminal["data"]:
        terminal = await _wait_event(serve.observer, "tool_end", **{"tool_call.id": "agent-bg"})
    await _wait_event(
        serve.observer, "sub_agent_receipt", **{"data.notification_id": terminal["data"]["notification_id"]}
    )


NOTIFICATIONS: dict[EventDescriptor, Callable[[Any], None]] = {
    EventDescriptor.SUB_AGENT_RECEIPT: lambda store: store.append_agent_notification(
        "child-done", child_session_path="/tmp/child", description="child", status="completed", text="done"
    ),
    EventDescriptor.CHILD_IDLE: lambda store: store.append_child_idle_notification(
        "child-idle", child_session_path="/tmp/child", description="child", text="idle"
    ),
    EventDescriptor.TASK_EXIT_NOTIFICATION: lambda store: store.append_task_notification(
        task_id="task-1", command="true", exit_code=0
    ),
    EventDescriptor.PROJECT_INBOX_NOTIFICATION: lambda store: store.append_inbox_notification_if_absent(
        "project-1", ["message-1"]
    ),
    EventDescriptor.CHILD_QUESTION: lambda store: store.append_child_question(
        child_instance_id="child-q", question_id="question-1", question="which?"
    ),
    EventDescriptor.CHILD_QUESTION_WITHDRAWN: lambda store: (
        store.append_child_question(
            child_instance_id="child-w", question_id="question-2", question="which?"
        ),
        store.close_child_questions("child-w", reason="finished"),
    ),
}


def _notification(event: EventDescriptor) -> Callable[[Serve, Any], Awaitable[None]]:
    async def run(serve: Serve, backend: Any) -> None:
        NOTIFICATIONS[event](serve.store)
        await serve.controller.request(serve.request_id(), "send", {"text": "go"})
        await _wait_event(serve.observer, event.value)
        await _wait_event(serve.observer, "agent_end")

    return run


async def _dropped_steer(serve: Serve, backend: BlockingBackend) -> None:
    await serve.controller.request(serve.request_id(), "send", {"text": "go"})
    await await_or_fail(backend.started.wait(), what="turn running")
    steered = await serve.controller.request(
        serve.request_id(), "steer", {"text": "steer-marker", "delivery_id": "steer-1"}
    )
    assert steered["result"]["accepted"] is True, steered
    aborted = await serve.controller.request(serve.request_id(), "abort", {})
    assert aborted["result"]["aborted"] is True, aborted
    await _wait_event(serve.observer, "delivery_status", delivery_id="steer-1")


async def _synthetic_abort(serve: Serve) -> None:
    """A failing store makes the session unhealthy; the next abort has no row."""
    calls = [ToolCall(f"call-{index}", "read", {"path": f"/tmp/input-{index}"}) for index in (1, 2)]
    serve.store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call) for call in calls]),
        [(call.id, call) for call in calls],
    )
    control = serve.server.control
    for call in calls:
        await control._event(StreamEvent(StreamEventType.TOOL_APPROVAL_START, tool_call=call))

    async def failing(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk failed")

    # test-contract: seam the durable approval write fails
    serve.store.resolve_approval_async = failing
    response = await serve.controller.request(serve.request_id(), "approve", {"request_id": "call-1"})
    assert "error" in response, response
    assert control.unhealthy()
    await await_or_fail(control.request_approval_abort("call-2"), what="abort second approval")
    await _wait_event(serve.observer, "approval_end", request_id="call-2")


@dataclass(frozen=True)
class Scenario:
    name: str
    backend: Callable[[Path], Any]
    run: Callable[[Serve, Any], Awaitable[None]]


SCENARIOS = {
    "delivery_idle": Scenario("delivery_idle", lambda tmp: _text_backend("ok"), lambda s, b: _delivery(s)),
    "delivery_after_newer_row": Scenario(
        "delivery_after_newer_row", lambda tmp: _text_backend("ok"), lambda s, b: _delivery_after_newer_row(s)
    ),
    "send_plain": Scenario("send_plain", lambda tmp: _text_backend("ok"), lambda s, b: _send_plain(s)),
    "send_images": Scenario("send_images", lambda tmp: _text_backend("ok"), lambda s, b: _send_images(s)),
    "send_skill": Scenario("send_skill", lambda tmp: _text_backend("ok"), lambda s, b: _send_skill(s)),
    "assistant_failed": Scenario(
        "assistant_failed", lambda tmp: TruncatedBackend(), lambda s, b: _assistant_failed(s)
    ),
    "foreground_tool": Scenario("foreground_tool", _foreground_tool_backend, lambda s, b: _foreground_tool(s)),
    "background_child": Scenario("background_child", _background_backend, lambda s, b: _background(s)),
    "dropped_steer": Scenario("dropped_steer", lambda tmp: BlockingBackend(), _dropped_steer),
    "synthetic_abort": Scenario("synthetic_abort", lambda tmp: FakeBackend([]), lambda s, b: _synthetic_abort(s)),
    **{
        f"notification_{event.value}": Scenario(
            f"notification_{event.value}",
            lambda tmp: _text_backend("ok"),
            _notification(event),
        )
        for event in NOTIFICATIONS
    },
}

# which scenarios produce each row-backed route; a new table entry without one fails P
PREMISE_SCENARIOS: dict[Route, tuple[str, ...]] = {
    (EventDescriptor.USER_MESSAGE, "delivered"): ("delivery_idle", "delivery_after_newer_row"),
    (EventDescriptor.USER_MESSAGE, "undelivered"): ("send_plain", "send_images", "send_skill"),
    (EventDescriptor.ASSISTANT_MESSAGE, ""): ("delivery_idle", "assistant_failed"),
    (EventDescriptor.TOOL_END, "foreground"): ("foreground_tool",),
    (EventDescriptor.TOOL_END, "background_terminal"): ("background_child",),
    (EventDescriptor.APPROVAL_REQUEST, "foreground"): ("foreground_tool",),
    (EventDescriptor.APPROVAL_REQUEST, "delegated"): ("background_child",),
    (EventDescriptor.APPROVAL_END, "foreground"): ("foreground_tool",),
    (EventDescriptor.APPROVAL_END, "delegated"): ("background_child",),
    (EventDescriptor.DELIVERY_STATUS, ""): ("dropped_steer",),
    **{(event, "durable"): (f"notification_{event.value}",) for event in NOTIFICATIONS},
}
ROW_BACKED = [
    route for route, source in rows.ROUTES.items() if source is not rows.LIVE
]
PREMISE_CASES = [
    (route, scenario) for route in ROW_BACKED for scenario in PREMISE_SCENARIOS.get(route, ("missing",))
]
NO_ROW_CASES: list[tuple[Route, str]] = [
    ((EventDescriptor.APPROVAL_END, "write_failed"), "synthetic_abort"),
    ((EventDescriptor.TOOL_END, "child"), "background_child"),
    ((EventDescriptor.TOOL_START, ""), "background_child"),
    ((EventDescriptor.USER_MESSAGE, "steer"), "dropped_steer"),
    ((EventDescriptor.TURN_ABORTED, ""), "dropped_steer"),
]


def _case_id(case: tuple[Route, str]) -> str:
    (event, variant), scenario = case
    return f"{event.value}-{variant or 'any'}-{scenario}"


async def _run_scenario(tmp_path: Path, transport: Transport, name: str) -> tuple[Serve, list[dict[str, Any]]]:
    scenario = SCENARIOS[name]
    backend = scenario.backend(tmp_path)
    serve = await _serve(tmp_path, transport, backend)
    try:
        await serve.join(serve.observer)
        await scenario.run(serve, backend)
        later = await serve.client()
        history = (await serve.join(later))["messages"]
    except BaseException:
        if isinstance(backend, BlockingBackend):
            backend.release.set()
        await _close(serve)
        raise
    return serve, history


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.parametrize("case", PREMISE_CASES, ids=[_case_id(case) for case in PREMISE_CASES])
async def test_premise_row_backed_frame_covers_its_history_row(
    tmp_path: Path, transport: Transport, case: tuple[Route, str]
) -> None:
    route, scenario = case
    assert scenario != "missing", f"route table entry {route} has no premise scenario"
    serve, history = await _run_scenario(tmp_path, transport, scenario)
    try:
        seqs = {entry.id: entry.seq for entry in serve.store.entries}
        records = serve.records(route)
        assert records, f"{scenario} published no {route} frame"
        for fields, facts, covers in records:
            matched = _history_rows_for(route[0], fields, facts, history)
            assert len({row["id"] for row in matched}) == 1, (route, fields, matched)
            assert covers == seqs[matched[0]["id"]], (route, covers, matched[0])
    finally:
        await _close(serve)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.parametrize("case", NO_ROW_CASES, ids=[_case_id(case) for case in NO_ROW_CASES])
async def test_premise_frame_without_history_row_covers_none(
    tmp_path: Path, transport: Transport, case: tuple[Route, str]
) -> None:
    route, scenario = case
    serve, _history = await _run_scenario(tmp_path, transport, scenario)
    try:
        records = serve.records(route)
        assert records, f"{scenario} published no {route} frame"
        assert [covers for _fields, _facts, covers in records] == [None] * len(records)
    finally:
        await _close(serve)


UNROUTED = "replay event has no route table entry"


@pytest.fixture(autouse=True)
def fail_on_unrouted_event(caplog: pytest.LogCaptureFixture) -> Any:
    """A publication with no route table entry fails the test that produced it."""
    caplog.set_level(logging.WARNING)
    yield
    unrouted = [record.getMessage() for record in caplog.records if UNROUTED in record.getMessage()]
    assert not unrouted, unrouted


def test_every_serve_event_kind_has_a_route_or_is_never_published() -> None:
    routed = {event for event, _variant in rows.ROUTES}
    assert set(EventDescriptor) - routed - rows.NOT_PUBLISHED == set()
    assert routed & rows.NOT_PUBLISHED == set()


async def _turn_start(serve: Serve) -> None:
    await serve.server.control._notify(EventDescriptor.TURN_START, serve.session_id, data={})


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_unrouted_event_ends_the_stream_and_is_logged(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    serve = await _serve(tmp_path, transport, FakeBackend([]))
    try:
        stream_id = (await serve.join(serve.observer))["replay"]["stream_id"]
        monkeypatch.delitem(rows.ROUTES, (EventDescriptor.TURN_START, ""))
        await _turn_start(serve)
        ended = await serve.observer.event("replay_stream_ended")
        assert (ended["stream_id"], ended["reason"]) == (stream_id, "evicted")
        assert [r.getMessage() for r in caplog.records if UNROUTED in r.getMessage()] == [
            f"{UNROUTED}: turn_start/-"
        ]
        caplog.clear()  # the expected failure; the autouse check rejects any other
    finally:
        await _close(serve)


async def _publish_tool_end(serve: Serve, call_id: str) -> None:
    await serve.server.control._notify(
        EventDescriptor.TOOL_END,
        serve.session_id,
        tool_call={"id": call_id, "name": "read", "arguments": {}},
        tool_result={"tool_call_id": call_id, "content": "late", "is_error": False},
        data={},
    )


def _tool_ends(frames: list[dict[str, Any]], call_id: str) -> int:
    return sum(
        frame["event"] == "tool_end" and frame["tool_result"]["tool_call_id"] == call_id for frame in frames
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_missing_row_with_whole_log_in_view_is_sent_live_and_logged(
    tmp_path: Path, transport: Transport, caplog: pytest.LogCaptureFixture
) -> None:
    serve = await _serve(tmp_path, transport, FakeBackend([]))
    try:
        await serve.join(serve.observer)
        await _publish_tool_end(serve, "ghost-call")
        assert _tool_ends(await serve.live(serve.observer), "ghost-call") == 1
        assert "replay row missing for tool_end; sent as live-only" in caplog.text
    finally:
        await _close(serve)


def _tool_rows(store: Any, call_id: str) -> None:
    call = ToolCall(call_id, "read", {"path": "/tmp/input"})
    store.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(call)]))
    store.append_message(
        Message(MessageRole.TOOL_RESULT, [TextContent("ok")], tool_result=ToolResult(call_id, "ok"))
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_row_older_than_lookup_window_ends_older_streams_only(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    serve = await _serve(tmp_path, transport, FakeBackend([]))
    try:
        old_stream = (await serve.join(serve.observer))["replay"]["stream_id"]
        monkeypatch.setattr(_row_history, "RECENT_ROW_WINDOW", 4)
        _tool_rows(serve.store, "late-call")
        for index in range(5):
            serve.store.append_task_notification(task_id=f"filler-{index}", command="true", exit_code=0)
        later = await serve.client()
        history = (await serve.join(later))["messages"]
        await _publish_tool_end(serve, "late-call")
        frames = await serve.live(serve.observer)
        ended = [(f["stream_id"], f["reason"]) for f in frames if f["event"] == "replay_stream_ended"]
        assert ended == [(old_stream, "evicted")], frames
        in_history = sum(
            (row.get("tool_result") or {}).get("tool_call_id") == "late-call" for row in history
        )
        assert in_history + _tool_ends(await serve.live(later), "late-call") == 1
    finally:
        await _close(serve)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_store_reset_ends_streams_instead_of_dropping_renumbered_rows(
    tmp_path: Path, transport: Transport
) -> None:
    serve = await _serve(tmp_path, transport, _text_backend("ok"))
    try:
        for index in range(6):
            serve.store.append_task_notification(task_id=f"before-{index}", command="true", exit_code=0)
        stream_id = (await serve.join(serve.observer))["replay"]["stream_id"]
        log = serve.store.path
        lines = log.read_bytes().splitlines(keepends=True)
        # an external rewrite drops the newest rows; the next write reloads and renumbers
        log.write_bytes(b"".join(lines[:3]))
        await serve.controller.request(
            serve.request_id(), "send", {"text": "after-reset", "delivery_id": "d-reset"}
        )
        await _wait_event(serve.controller, "agent_end")
        frames = await serve.live(serve.observer)
        ended = [(f["stream_id"], f["reason"]) for f in frames if f["event"] == "replay_stream_ended"]
        echoed = [f.get("text") for f in frames if f["event"] == "user_message"]
        assert ended == [(stream_id, "session_replaced")], f"ended={ended} live echo={echoed}"
        history = (await serve.join(serve.observer))["messages"]
        live = [
            frame.get("text") for frame in await serve.live(serve.observer) if frame["event"] == "user_message"
        ]
        assert (_texts(history) + live).count("after-reset") == 1
    finally:
        await _close(serve)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_store_reset_during_join_read_fails_the_join(tmp_path: Path, transport: Transport) -> None:
    """J3: a reset between a join's start and its end renumbers rows; the join is stale."""
    serve = await _serve(tmp_path, transport, _text_backend("ok"))
    connections = serve.server.connections
    assert connections is not None
    run_read = connections.run_read
    entered, release = asyncio.Event(), asyncio.Event()

    async def held(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        await await_or_fail(release.wait(), what="release held history read")
        return await run_read(*args, **kwargs)

    try:
        for index in range(6):
            serve.store.append_task_notification(task_id=f"before-{index}", command="true", exit_code=0)
        generation = serve.store.reset_generation()
        # test-contract: seam hold the join's history read after the join began
        connections.run_read = held  # type: ignore[method-assign]
        join = asyncio.create_task(
            serve.observer.request(
                serve.request_id(), "session_history_and_subscribe", {"session_id": serve.session_id}
            )
        )
        await await_or_fail(entered.wait(), what="join history read held")
        connections.run_read = run_read  # type: ignore[method-assign]
        log = serve.store.path
        # an external rewrite drops the newest rows; the next write reloads and renumbers
        log.write_bytes(b"".join(log.read_bytes().splitlines(keepends=True)[:3]))
        serve.store.append_task_notification(task_id="after-reset", command="true", exit_code=0)
        assert serve.store.reset_generation() != generation
        release.set()
        response = await await_or_fail(join, what="join response")
        assert response.get("error", {}).get("code") == -32009, response
        later = await serve.client()
        history = (await serve.join(later))["messages"]
        ids = [row["id"] for row in history]
        assert len(ids) == len(set(ids)), ids
        assert sum("after-reset" in str(row) for row in history) == 1, history
    finally:
        release.set()
        connections.run_read = run_read  # type: ignore[method-assign]
        await _close(serve)
