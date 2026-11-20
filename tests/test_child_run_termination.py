"""One table for how a child run ends (design-child-termination-owner §6).

Every row drives a real parent ``AgentLoop`` (or serve) and checks the same
durable records: D1 child lifecycle state, D2 cancel marker, D4 parent marker,
D5 parent receipt, plus no running owner entry, open child store, or pending task.
Rows that fail today are strict xfails that must fail at a durable-outcome
check (``TerminalMismatch``: D1, D5, D2, D4, or a cancel reply; each reason
names which); a hang, leak, or setup error is a real failure.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.support.agent_lifecycle import watch_delegated_registrations
from tests.support.gated_store import gate_store
from tests.support.owned import await_or_fail
from tests.support.replay_transport import (
    ReplayClient,
    connect_replay,
    negotiate,
    replay_server,
)
from zeta.agent import runner
from zeta.agent.presets import AGENT_PRESETS
from zeta.core.agent_progress import register_child_run_sink
from zeta.core.approval import ApprovalPolicy
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    CompletionBackend,
    Message,
    MessageOrigin,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolResult,
    ToolSchema,
    ToolUseContent,
    with_message_origin,
)
from zeta.providers.completion import owned_completion
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog

pytest_plugins = ("tests.support.socket_fixtures",)
SERVE_FEATURES = ("child_run_control_v1", "child_run_events", "child_run_history", "child_runs")
PROMPTS = {"start": "parent", "child prompt": "child", "grandchild prompt": "grandchild"}
STATUS = {"completed": "completed", "canceled": "canceled", "error": "failed", "failed": "failed"}


class TerminalMismatch(AssertionError):
    """D1 or D5 differs from the row's expected outcome."""


def _text(text: str) -> list:
    return [TextContent(text)]


def _call(name: str, call_id: str, **arguments: object) -> list:
    return [ToolUseContent(ToolCall(call_id, name, dict(arguments)))]


@dataclass
class Actor:
    turns: list[list | BaseException]
    gate_call: int | None = None
    calls: int = 0
    started: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)


class Script(CompletionBackend):
    """Route each model call to its actor by the actor's prompt."""

    def __init__(self, actors: dict[str, Actor]) -> None:
        self.actors = actors

    @owned_completion
    async def complete(
        self, messages: Sequence[Message], tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        del tool_schemas
        name = next(
            PROMPTS[block.text]
            for message in messages
            if message.role is MessageRole.USER
            for block in message.content
            if isinstance(block, TextContent) and block.text in PROMPTS
        )
        actor = self.actors[name]
        index, actor.calls = actor.calls, actor.calls + 1
        if index == actor.gate_call:
            actor.started.set()
            # wait_for, not await_or_fail: a canceled call must not leave the wait behind.
            await asyncio.wait_for(actor.release.wait(), 10)
        blocks = actor.turns[index] if index < len(actor.turns) else _text(f"{name} done")
        if isinstance(blocks, BaseException):
            raise blocks
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for block in blocks:
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=block)
        yield StreamEvent(
            StreamEventType.MESSAGE_END, message=Message(MessageRole.ASSISTANT, blocks),
        )


@dataclass
class Run:
    tmp_path: Path
    mode: str  # fg | bg | explicit
    store: ConversationStore
    loop: AgentLoop
    backend: Script
    cleanup: AsyncExitStack
    turn: asyncio.Task | None = None
    client: ReplayClient | None = None
    close: Callable[[], Awaitable[object]] | None = None
    baseline: set[asyncio.Task] = field(default_factory=set)
    request_id: int = 10
    spawn_id: int | None = None
    policy: ApprovalPolicy | None = None

    @property
    def child(self) -> Actor:
        return self.backend.actors["child"]

    @property
    def grandchild(self) -> Actor:
        return self.backend.actors["grandchild"]

    @property
    def handle(self) -> str:
        return f"{self.store.session_id}:1"

    @property
    def child_dir(self) -> Path:
        return self.store.session_dir / "agents" / "1"

    def start(self) -> asyncio.Task:
        async def turn() -> None:
            if self.client is not None:
                self.spawn_id = await self.spawn()
                return
            async for _ in self.loop.run_turn("start", origin=MessageOrigin.USER):
                pass

        self.turn = asyncio.create_task(turn())
        return self.turn

    async def close_parent(self) -> None:
        await await_or_fail(
            self.close() if self.close is not None else self.loop.close(), what="parent close",
        )

    async def agent_cancel(self, handle: str | None = None) -> str:
        if self.client is not None:
            return await self.serve_cancel()
        result = await self.loop.tool_registry.execute(
            ToolCall("cancel", "agent_cancel", {"handle": handle or self.handle})
        )
        return (result.get("structuredContent") or {}).get("status") or result["content"][0]["text"]

    async def serve_cancel(self) -> str:
        """``child_run_cancel``: the state answered, or the error reason."""
        reply, = await self.serve_cancels(1)
        return reply

    async def serve_cancels(self, count: int) -> list[str]:
        """Send ``count`` cancels in one tick, then read each reply."""
        assert self.client is not None
        ids = []
        for _ in range(count):
            self.request_id += 1
            ids.append(self.request_id)
            await self.client.send(self.request_id, "child_run_cancel", {
                "session_id": self.store.session_id, "run_id": self.handle,
                "reason": "user_cancelled",
            })
        replies = [await self.reply(request_id) for request_id in ids]
        return [
            reply["error"]["data"]["reason"] if "error" in reply else reply["result"]["data"]["state"]
            for reply in replies
        ]

    async def spawn(self) -> int:
        """Send ``child_run_spawn``; read the reply later with ``reply``."""
        assert self.client is not None
        self.request_id += 1
        await self.client.send(self.request_id, "child_run_spawn", {
            "session_id": self.store.session_id, "agent_type": "general",
            "task": "child prompt", "cwd": str(self.tmp_path),
        })
        return self.request_id

    async def reply(self, request_id: int) -> dict:
        assert self.client is not None
        for frame in self.client.buffered:
            if frame.get("id") == request_id:
                self.client.buffered.remove(frame)
                return frame
        return await self.client.response(request_id)

    async def finish_turn(self) -> BaseException | None:
        """Await the parent turn; return its exception (a cancel is not a test failure)."""
        assert self.turn is not None
        results = await await_or_fail(
            asyncio.gather(self.turn, return_exceptions=True), what="parent turn ends",
        )
        if self.spawn_id is not None:
            # A spawn the close interrupted gets no reply: the connection is gone.
            spawn_id, self.spawn_id = self.spawn_id, None
            results = await await_or_fail(
                asyncio.gather(self.reply(spawn_id), return_exceptions=True), what="spawn reply",
            )
        return results[0] if isinstance(results[0], BaseException) else None

    async def settle(self) -> None:
        await await_or_fail(self.loop._background_owner.wait(), what="child tree settles")


def _agent_arguments(mode: str, **extra: object) -> dict[str, object]:
    arguments: dict[str, object] = {"prompt": "child prompt", "description": "child run"}
    if mode == "bg":
        arguments["background"] = True
    if mode == "explicit":
        arguments["agent_type"] = "run"
    return {**arguments, **extra}


async def _open_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    child: Actor,
    *,
    grandchild: Actor | None = None,
    approval: bool = False,
    **agent_extra: object,
) -> Run:
    if mode == "explicit":
        monkeypatch.setitem(
            AGENT_PRESETS, "run", replace(AGENT_PRESETS["run"], explicit_finish=True),
        )
    parent = Actor([_call("agent", "child", **_agent_arguments(mode, **agent_extra))])
    backend = Script({
        "parent": parent, "child": child,
        "grandchild": grandchild or Actor([_text("grandchild done")]),
    })
    cleanup = AsyncExitStack()
    store = ConversationStore(tmp_path, session_id="parent")
    cleanup.callback(store.close)
    policy = ApprovalPolicy(store=store) if approval else None
    loop = AgentLoop(backend, store, skill_catalog=SkillCatalog.empty(), approval_policy=policy)
    loop.explicit_finish = mode == "explicit"
    cleanup.push_async_callback(loop.close)
    return Run(
        tmp_path, mode, store, loop, backend, cleanup, baseline=set(asyncio.all_tasks()),
        policy=policy,
    )


async def _open_serve(tmp_path: Path, child: Actor) -> Run:
    backend = Script({"parent": Actor([]), "child": child, "grandchild": Actor([])})
    cleanup = AsyncExitStack()
    server = replay_server(tmp_path, "unix", lambda *_args: (backend, "offline"))
    await server.start()
    cleanup.push_async_callback(server.close)
    client = await connect_replay(server, "unix")
    cleanup.push_async_callback(client.close)
    await negotiate(client, extra=SERVE_FEATURES, protocol_version="1.2")
    created = await client.request(2, "new_session", {"provider": "codex"})
    assert "result" in created, created
    loop = server.runtime.loop
    return Run(
        tmp_path, "serve", loop.store, loop, backend, cleanup, client=client,
        close=server.close, baseline=set(asyncio.all_tasks()),
    )


async def _open(tmp_path, monkeypatch, mode: str, child: Actor, **options) -> Run:
    if mode == "serve":
        assert not options, options
        return await _open_serve(tmp_path, child)
    return await _open_run(tmp_path, monkeypatch, mode, child, **options)


def _child_store(path: Path) -> ConversationStore:
    return ConversationStore(path.parent, session_id=path.name, _read_only=True)


def _d1(path: Path) -> str | None:
    if not path.exists():
        return None
    child = _child_store(path)
    try:
        lifecycle = child.agent_lifecycle()
        return None if lifecycle is None else lifecycle["state"]
    finally:
        child.close()


def _d5_entry(store: ConversationStore, mode: str) -> tuple[str | None, str]:
    """(status, text) of the parent's receipt for the child: the background
    notification, else the agent tool result (foreground, or a setup-phase end)."""
    if mode != "fg":
        notes = [
            entry.data for entry in store.agent_notifications(pending_only=False)
            if entry.data.get("kind") != "child_question" and entry.data.get("status")
        ]
        if notes:
            return STATUS[notes[-1]["status"]], notes[-1].get("text", "")
    result = store.tool_result("child")
    if result is None or (result.structured_content or {}).get("status") == "running":
        return None, ""
    if result.is_canceled:
        return "canceled", result.content
    return ("failed" if result.is_error else "completed"), result.content


def _d5(store: ConversationStore, mode: str) -> str | None:
    return _d5_entry(store, mode)[0]


def _expect_terminal(d1: str | None, d5: str | None, expected: str | None) -> None:
    if d1 != expected:
        raise TerminalMismatch(f"D1 is {d1!r}, expected {expected!r} (D5 {d5!r})")
    if d5 != expected:
        raise TerminalMismatch(f"D5 is {d5!r}, expected {expected!r} (D1 {d1!r})")


def _expect(label: str, ok: bool, detail: object = "") -> None:
    """A durable outcome other than D1/D5 (D2, D4, a cancel reply) is wrong."""
    if not ok:
        raise TerminalMismatch(f"{label}: {detail}")


async def _parent_state(run) -> tuple[dict, tuple[str | None, str]]:
    """(D4, D5) of the parent after its loop closed; serve's store is reopened."""
    if run.mode != "serve":
        return run.store.agent_children(), _d5_entry(run.store, run.mode)

    def read() -> tuple[dict, tuple[str | None, str]]:
        session_dir = run.store.session_dir
        view = ConversationStore(session_dir.parent, session_id=session_dir.name)
        try:
            return view.agent_children(), _d5_entry(view, run.mode)
        finally:
            view.close()

    return await await_or_fail(asyncio.to_thread(read), what="read the parent state")


async def _d4(run) -> dict:
    return (await _parent_state(run))[0]


async def _d5_after(run) -> tuple[str | None, str]:
    return (await _parent_state(run))[1]


def _d2(path: Path) -> bool:
    child = _child_store(path)
    try:
        return child.agent_canceled() is not None
    finally:
        child.close()


async def _closed(run: Run) -> None:
    """Close the parent; nothing of the child tree may stay running or open."""
    await run.close_parent()
    owner = run.loop._background_owner
    assert not owner.running, "owner still runs a child"
    open_stores = [store.session_dir for store in owner._stores if store.directory_fd != -1]
    assert not open_stores, f"child stores still open: {open_stores}"
    leaked = [
        task for task in asyncio.all_tasks()
        if task not in run.baseline and not task.done() and task is not asyncio.current_task()
    ]
    assert not leaked, f"pending tasks: {leaked}"


async def _verify(run: Run, expected: str | None) -> None:
    """Close the parent, then check every durable record of the child's end.

    ``expected=None``: the child never passed its marker (no D1, no D4)."""
    await _closed(run)
    d1 = _d1(run.child_dir)
    d4, (d5, _) = await _parent_state(run)
    if expected is None:
        _expect("D1", d1 is None, d1)
    else:
        _expect_terminal(d1, d5, expected)
        _expect("D2", _d2(run.child_dir) == (expected == "canceled"), f"present={_d2(run.child_dir)}")
    _expect("D4", not d4, d4)


async def _restart(run: Run) -> tuple[str | None, str | None]:
    """Resume the parent session in a fresh loop; return (D1, D5) after recovery."""
    store = ConversationStore(run.tmp_path, session_id="parent")
    loop = AgentLoop(Script({}), store, skill_catalog=SkillCatalog.empty())
    try:
        await await_or_fail(loop.activate(), what="recovery")
        return _d1(run.child_dir), _d5(store, run.mode)
    finally:
        await await_or_fail(loop.close(), what="close recovered parent")
        store.close()


# rows ------------------------------------------------------------------------

Row = Callable[[Path, pytest.MonkeyPatch, str], Awaitable[None]]
ROWS: dict[str, Row] = {}


def row(name: str) -> Callable[[Row], Row]:
    def register(fn: Row) -> Row:
        ROWS[name] = fn
        return fn

    return register


@row("success")
async def _success(tmp_path, monkeypatch, mode):
    if mode == "explicit":
        child = Actor([_call("finish", "finish", status="done", summary="child finished")])
    else:
        child = Actor([_text("child complete")])
    run = await _open_run(tmp_path, monkeypatch, mode, child)
    async with run.cleanup:
        run.start()
        assert await run.finish_turn() is None
        await run.settle()
        await _verify(run, "completed")


@row("fail-child")
async def _fail_child(tmp_path, monkeypatch, mode):
    run = await _open_run(tmp_path, monkeypatch, mode, Actor([RuntimeError("model down")]))
    async with run.cleanup:
        run.start()
        assert await run.finish_turn() is None
        await run.settle()
        await _verify(run, "failed")


def _held(monkeypatch, run: Run, name: str, **options):
    """Gate ``name`` on the child's own store."""
    return gate_store(
        monkeypatch, name, when=lambda store, *_: store.session_dir == run.child_dir, **options,
    )


def _terminal_d1(monkeypatch, run: Run) -> asyncio.Event:
    """Set when any writer makes the child's D1 terminal."""
    written = asyncio.Event()
    loop = asyncio.get_running_loop()

    def check(store: ConversationStore) -> None:
        if (store.agent_lifecycle() or {}).get("finished_at") is not None:
            loop.call_soon_threadsafe(written.set)

    _held(monkeypatch, run, "_write_agent_lifecycle", then=check)
    return written


def _hold_open_child_store(monkeypatch):
    # test-contract: seam hold the joined store work that opens the child store.
    return gate_store(
        monkeypatch, "_to_thread_store", hold=True,
        when=lambda _store, args, _kwargs: args[0].__name__ == "open_child_store",
    )


def _gated_child() -> Actor:
    return Actor([_text("late")], gate_call=0)


@row("cancel@open_store")
async def _cancel_open_store(tmp_path, monkeypatch, mode):
    # No serve mode: serve reads one connection's requests in order, so a
    # child_run_cancel cannot reach the server while its spawn is in setup.
    run = await _open(tmp_path, monkeypatch, mode, Actor([_text("child complete")]))
    async with run.cleanup:
        held = _hold_open_child_store(monkeypatch)
        run.start()
        await await_or_fail(held.entered.wait(), what="child store open")
        run.loop.abort()
        await run.finish_turn()
        await _verify(run, None)
        _expect("D5", (await _d5_after(run))[0] == "canceled", await _d5_after(run))


@row("close@open_store")
async def _close_open_store(tmp_path, monkeypatch, mode):
    run = await _open(tmp_path, monkeypatch, mode, Actor([_text("child complete")]))
    async with run.cleanup:
        held = _hold_open_child_store(monkeypatch)
        run.start()
        await await_or_fail(held.entered.wait(), what="child store open")
        await run.close_parent()
        await run.finish_turn()
        await _verify(run, None)


SETUP_POINTS = (
    "lineage", "child_start", "account_pin", "lifecycle_start",
    "base_prompt", "live_config", "step_delivery",
)


def _pause_setup(monkeypatch, loop: AgentLoop, point: str) -> asyncio.Event:
    """Pause the child's setup at one named await after its D4 marker."""
    entered = asyncio.Event()

    async def pause() -> None:
        entered.set()
        await asyncio.wait_for(asyncio.Event().wait(), 10)

    original_offload = runner.await_joined_offload

    async def offload(fn, *args, **kwargs):
        if (point, fn.__name__) in {("lineage", "publish"), ("account_pin", "child_snapshot"),
                                    ("base_prompt", "_child_base_system_prompt")}:
            await pause()
        result = await original_offload(fn, *args, **kwargs)
        if (point, fn.__name__) == ("lifecycle_start", "start_lifecycle"):
            await pause()
        return result

    # test-contract: seam pause at the real setup await, not a timed race.
    monkeypatch.setattr(runner, "await_joined_offload", offload)
    def publish(link):
        pass

    def child_snapshot(provider):
        return None

    if point == "lineage":
        loop._child_lineage_publisher = SimpleNamespace(publish=publish)
    if point == "account_pin":
        # Only the child setup may see this stand-in account context: install it at D4.
        accounts = SimpleNamespace(
            child_snapshot=child_snapshot, active_provider=None, switch_sequence=0,
        )
        gate_store(
            monkeypatch, "register_agent_child", when=lambda store, *_: store is loop.store,
            then=lambda _store: setattr(loop, "account_context", accounts),
        )
    if point == "child_start":
        original_start = runner.append_child_start

        async def append_child_start(*args, **kwargs):
            await pause()
            return await original_start(*args, **kwargs)

        monkeypatch.setattr(runner, "append_child_start", append_child_start)
    for name, method in (("live_config", "refresh_live_config"), ("step_delivery", "_ensure_step_delivery")):
        if point != name:
            continue
        original = getattr(AgentLoop, method)

        async def child_pause(self, *args, _original=original, **kwargs):
            if self.agent_depth == 1:
                await pause()
            return await _original(self, *args, **kwargs)

        monkeypatch.setattr(AgentLoop, method, child_pause)
    return entered


@row("cancel@after_marker")
async def _cancel_after_marker(tmp_path, monkeypatch, mode):
    """Bugs 1 and 3: a cancel after D4 and before the child runs."""
    mode, point = mode.split(":")
    run = await _open(tmp_path, monkeypatch, mode, Actor([_text("child complete")]))
    async with run.cleanup:
        entered = _pause_setup(monkeypatch, run.loop, point)
        run.start()
        await await_or_fail(entered.wait(), what=f"setup at {point}")
        assert run.store.agent_children(), "D4 is written before the pause"
        if mode == "serve":
            await run.close_parent()
        else:
            run.loop.abort()
        await run.finish_turn()
        await run.settle()
        await _verify(run, "canceled")


@row("cancel@anchor_write")
async def _cancel_anchor_write(tmp_path, monkeypatch, mode):
    run = await _open(tmp_path, monkeypatch, mode, Actor([_text("child complete")]))
    async with run.cleanup:
        held = gate_store(
            monkeypatch, "_commit_address_async", hold=True,
            when=lambda store, *_: store is run.store,
        )
        run.start()
        await await_or_fail(held.entered.wait(), what="anchor write")
        closing = asyncio.create_task(run.close_parent())
        held.release()
        await await_or_fail(closing, what="close during the anchor write")
        await run.finish_turn()
        await _verify(run, "canceled")


@row("cancel@model_turn")
async def _cancel_model_turn(tmp_path, monkeypatch, mode):
    run = await _open(tmp_path, monkeypatch, mode, _gated_child())
    async with run.cleanup:
        run.start()
        await await_or_fail(run.child.started.wait(), what="child provider call")
        assert await run.agent_cancel() in {"cancellation_confirmed", "cancelling"}
        await run.finish_turn()
        await run.settle()
        await _verify(run, "canceled")


@row("cancel@turn_limit")
async def _cancel_turn_limit(tmp_path, monkeypatch, mode):
    """A cancel claimed in the last turn of a turn-limited child; the child's
    turn-limit D1 lands before the cancel write."""
    child = Actor([_call("read", "read-1", path="missing.txt")], gate_call=0)
    run = await _open(tmp_path, monkeypatch, mode, child, max_turns=1)
    async with run.cleanup:
        _held(monkeypatch, run, "mark_agent_canceled_async", hold=True).released = (
            _terminal_d1(monkeypatch, run)
        )
        run.start()
        await await_or_fail(run.child.started.wait(), what="child last turn")
        # agent_cancel's path without its 1 s confirmation wait (the write is held).
        assert run.loop._background_owner.cancel(run.handle)
        run.child.release.set()
        await run.finish_turn()
        await run.settle()
        await _verify(run, "canceled")


@row("abort@generator_close")
async def _abort_generator_close(tmp_path, monkeypatch, mode):
    run = await _open(tmp_path, monkeypatch, mode, _gated_child())
    async with run.cleanup:
        events = run.loop.run_turn("start", origin=MessageOrigin.USER)
        while not run.child.started.is_set():
            await await_or_fail(anext(events), what="parent turn event")
        await await_or_fail(events.aclose(), what="parent turn generator close")
        await run.settle()
        await _verify(run, "canceled")


@row("cancel@question_pending")
async def _cancel_question_pending(tmp_path, monkeypatch, mode):
    child = Actor([_call("ask_parent", "ask", question="may i?"), _text("late")], gate_call=1)
    run = await _open(tmp_path, monkeypatch, mode, child)
    async with run.cleanup:
        run.start()
        await await_or_fail(run.child.started.wait(), what="child waits after asking")
        assert any(
            entry.data.get("kind") == "child_question" for entry in run.store.agent_notifications()
        )
        assert await run.agent_cancel() == "cancellation_confirmed"
        await run.finish_turn()
        await run.settle()
        await _verify(run, "canceled")
        pending = [e.data for e in run.store.agent_notifications() if e.data.get("kind") == "child_question"]
        _expect("question withdrawn", not pending, pending)


@row("cancel@approval")
async def _cancel_approval(tmp_path, monkeypatch, mode):
    child = Actor([_call("bash", "bash-1", command="echo hi")])
    run = await _open(tmp_path, monkeypatch, mode, child, approval=True)
    async with run.cleanup:
        registered = watch_delegated_registrations(monkeypatch, run.policy)
        run.start()
        await await_or_fail(registered.wait(), what="child approval pending")
        assert await run.agent_cancel() == "cancellation_confirmed"
        await run.finish_turn()
        await run.settle()
        await _verify(run, "canceled")


@row("cancel@deciding")
async def _cancel_deciding(tmp_path, monkeypatch, mode):
    """The child decided completed; a cancel arrives while its end is written."""
    run = await _open(tmp_path, monkeypatch, mode, Actor([_text("child complete")]))
    replies: list[asyncio.Task] = []
    async with run.cleanup:
        writer = "finish_agent_lifecycle" if mode == "fg" else "finalize_background_agent"
        _held(monkeypatch, run, writer, then=lambda _store: replies.append(
            asyncio.ensure_future(run.agent_cancel())
        ))
        run.start()
        await run.finish_turn()
        await run.settle()
        reply, = await await_or_fail(asyncio.gather(*replies), what="cancel reply")
        await _verify(run, "completed")
        _expect("cancel reply", reply == "already_finished", reply)


@row("cancel-vs-explicit-finish")
async def _cancel_vs_explicit_finish(tmp_path, monkeypatch, mode):
    child = Actor([_call("finish", "finish", status="done", summary="child finished")])
    run = await _open(tmp_path, monkeypatch, mode, child)
    replies: list[asyncio.Task] = []
    async with run.cleanup:
        _held(monkeypatch, run, "accept_explicit_finish", then=lambda _store: replies.append(
            asyncio.ensure_future(run.agent_cancel())
        ))
        run.start()
        await run.finish_turn()
        await run.settle()
        reply, = await await_or_fail(asyncio.gather(*replies), what="cancel reply")
        await _verify(run, "completed")
        _expect("cancel reply", reply == "already_finished", reply)


@row("explicit-cancel-first")
async def _explicit_cancel_first(tmp_path, monkeypatch, mode):
    child = Actor([_call("finish", "finish", status="done", summary="too late")], gate_call=0)
    run = await _open(tmp_path, monkeypatch, mode, child)
    async with run.cleanup:
        run.start()
        await await_or_fail(run.child.started.wait(), what="child about to finish")
        assert await run.agent_cancel() == "cancellation_confirmed"
        run.child.release.set()
        await run.finish_turn()
        await run.settle()
        await _verify(run, "canceled")


@row("abort-after-complete")
async def _abort_after_complete(tmp_path, monkeypatch, mode):
    run = await _open(tmp_path, monkeypatch, mode, Actor([_text("child complete")]))
    async with run.cleanup:
        _held(monkeypatch, run, "finish_agent_lifecycle", then=lambda _store: run.loop.abort())
        run.start()
        await run.finish_turn()
        await run.settle()
        await _verify(run, "completed")


@row("close@running")
async def _close_running(tmp_path, monkeypatch, mode):
    child = Actor([
        _call("agent", "grandchild", prompt="grandchild prompt", description="gc", background=True),
        _text("late"),
    ], gate_call=1)
    grandchild = Actor([_text("late")], gate_call=0)
    run = await _open(tmp_path, monkeypatch, mode, child, grandchild=grandchild)
    async with run.cleanup:
        run.start()
        await await_or_fail(run.child.started.wait(), what="child second call")
        await await_or_fail(run.grandchild.started.wait(), what="grandchild call")
        await run.close_parent()
        await run.finish_turn()
        await _verify(run, "canceled")
        _expect("grandchild D1", _d1(run.child_dir / "agents" / "1") == "canceled")


@row("cancel@finish_gate")
async def _cancel_finish_gate(tmp_path, monkeypatch, mode):
    """Bug 4: the grandchild's cancel write frees the gate before the child's lands."""
    child = Actor([
        _call("agent", "grandchild", prompt="grandchild prompt", description="gc", background=True),
        _text("child complete"),
    ])
    grandchild = Actor([_text("late")], gate_call=0)
    run = await _open(tmp_path, monkeypatch, mode, child, grandchild=grandchild)
    child_d1 = _held(monkeypatch, run, "finish_agent_lifecycle")
    grandchild_cancel = gate_store(
        monkeypatch, "mark_agent_canceled_async",
        when=lambda store, *_: store.session_dir == run.child_dir / "agents" / "1",
    )
    # The grandchild's cancel write lands first; in fg the child also writes its
    # own D1 first (the RCA order). bg has no child D1 writer before the cancel.
    _held(monkeypatch, run, "mark_agent_canceled_async", hold=True).released = (
        child_d1.done if mode == "fg" else grandchild_cancel.done
    )
    async with run.cleanup:
        run.start()
        await await_or_fail(run.grandchild.started.wait(), what="grandchild call")
        run.loop._background_owner.cancel_subtree(run.handle)
        await run.finish_turn()
        await run.settle()
        await _verify(run, "canceled")
        _expect("grandchild D1", _d1(run.child_dir / "agents" / "1") == "canceled")


@row("fail@D1_cancel_write")
async def _fail_d1_cancel_write(tmp_path, monkeypatch, mode):
    run = await _open(tmp_path, monkeypatch, mode, _gated_child())
    async with run.cleanup:
        _held(monkeypatch, run, "mark_agent_canceled_async", error=OSError("disk full"))
        run.start()
        await await_or_fail(run.child.started.wait(), what="child provider call")
        await run.agent_cancel()
        run.child.release.set()
        assert await run.finish_turn() is None, "the parent tool must not raise"
        await run.settle()
        await _closed(run)
        status, text = await _d5_after(run)
        _expect_terminal("canceled", status, "canceled")
        _expect("D5 notice", "terminal state not saved" in text, text)
        _expect("D4 kept", bool(await _d4(run)))
        _expect_terminal(*await _restart(run), "canceled")


@row("fail@D1_finish_write")
async def _fail_d1_finish_write(tmp_path, monkeypatch, mode):
    run = await _open(tmp_path, monkeypatch, mode, Actor([_text("child complete")]))
    async with run.cleanup:
        writer = "finish_agent_lifecycle" if mode == "fg" else "finalize_background_agent"
        _held(monkeypatch, run, writer, error=OSError("disk full"))
        run.start()
        assert await run.finish_turn() is None, "the parent tool must not raise"
        await run.settle()
        await _closed(run)
        status, text = await _d5_after(run)
        _expect_terminal("completed", status, "completed")
        _expect("D5 notice", "terminal state not saved" in text, text)
        _expect("D4 kept", bool(await _d4(run)))
        _expect_terminal(*await _restart(run), "completed")


@row("fail@D5_write")
async def _fail_d5_write(tmp_path, monkeypatch, mode):
    run = await _open(tmp_path, monkeypatch, mode, Actor([_text("child complete")]))
    async with run.cleanup:
        if mode == "fg":
            gate_store(
                monkeypatch, "append_message", error=OSError("disk full"), once=True,
                when=lambda store, args, _kw: store is run.store
                and args[0].tool_result is not None and args[0].tool_result.tool_call_id == "child",
            )
        else:
            gate_store(
                monkeypatch, "append_agent_notification", error=OSError("disk full"), once=True,
                when=lambda store, *_: store is run.store,
            )
        run.start()
        await run.finish_turn()
        await run.settle()
        await _closed(run)
        _expect_terminal(_d1(run.child_dir), "completed", "completed")
        _expect_terminal(*await _restart(run), "completed")


@row("fail@D6_publish")
async def _fail_d6_publish(tmp_path, monkeypatch, mode):
    run = await _open(tmp_path, monkeypatch, mode, Actor([_text("child complete")]))
    async with run.cleanup:
        def broken(kind, *_):
            if kind == "ended":
                raise RuntimeError("event sink down")

        run.cleanup.callback(register_child_run_sink(broken))
        run.start()
        error = await run.finish_turn()
        await run.settle()
        await _verify(run, "completed")
        _expect("parent turn", error is None, repr(error))


def _crashed_parent(tmp_path: Path, mode: str, *, d1: str | None, d5: bool = False,
                    report: bool = False) -> None:
    """Write the durable state a crash leaves: D4, a running or final D1, maybe D5."""
    store = ConversationStore(tmp_path, session_id="parent")
    child = ConversationStore(store.session_dir / "agents", session_id="1")
    try:
        call = ToolCall("child", "agent", _agent_arguments(mode))
        store.append_message(with_message_origin(
            Message(MessageRole.USER, _text("start")), MessageOrigin.USER,
        ))
        store.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(call)]))
        child.mark_agent_parent(call.id)
        child.start_agent_lifecycle(
            handle="parent:1", started_at="2026-11-20T00:00:00Z", depth=1,
            agent_type="run" if mode == "explicit" else "general", description="child run",
        )
        if mode != "fg":
            store.append_message(Message(
                MessageRole.TOOL_RESULT, _text("background agent started"),
                tool_result=ToolResult("child", "background agent started", structured_content={
                    "turns_used": 0, "child_session_path": str(child.session_dir),
                    "status": "running", "child_instance_id": "parent:1",
                    "description": "child run",
                }),
            ))
        store.allocate_agent_index()
        store.register_agent_child(
            call, child_session_path=str(child.session_dir), description="child run",
            background=mode != "fg", child_instance_id="parent:1",
            accepts_follow_ups=mode == "explicit", explicit_finish=mode == "explicit",
        )
        if d1 is not None:
            child.finish_agent_lifecycle(
                d1, final_result=f"child complete · 1 turns · 0.1s · 0 tool calls · "
                f"error=false · canceled={str(d1 == 'canceled').lower()}",
            )
        if d5:
            store.append_message(Message(
                MessageRole.TOOL_RESULT, _text("child complete"),
                tool_result=ToolResult("child", "child complete"),
            ))
        if report:
            asyncio.run(child.accept_explicit_finish(
                {"status": "done", "summary": "child finished"}, frozenset(), None,
            ))
    finally:
        child.close()
        store.close()


async def _restart_row(tmp_path, mode, expected, **state) -> None:
    await await_or_fail(asyncio.to_thread(_crashed_parent, tmp_path, mode, **state), what="crash state")
    run = SimpleNamespace(tmp_path=tmp_path, mode=mode, child_dir=tmp_path / "parent" / "agents" / "1")
    _expect_terminal(*await _restart(run), expected)


@row("restart@D4_only")
async def _restart_d4_only(tmp_path, monkeypatch, mode):
    await _restart_row(tmp_path, mode, "canceled", d1=None)


@row("restart@D1_done_no_D5")
async def _restart_d1_done(tmp_path, monkeypatch, mode):
    await _restart_row(tmp_path, mode, "completed", d1="completed")


@row("restart@D5_no_D1")
async def _restart_d5_no_d1(tmp_path, monkeypatch, mode):
    await _restart_row(tmp_path, mode, "completed", d1=None, d5=True)


@row("restart@D3_finish")
async def _restart_d3_finish(tmp_path, monkeypatch, mode):
    await _restart_row(tmp_path, mode, "completed", d1=None, report=True)


@row("dup-cancel")
async def _dup_cancel(tmp_path, monkeypatch, mode):
    run = await _open(tmp_path, monkeypatch, mode, _gated_child())
    async with run.cleanup:
        writes = _held(monkeypatch, run, "mark_agent_canceled_async")
        run.start()
        await await_or_fail(run.child.started.wait(), what="child provider call")
        if mode == "serve":
            first, second = await run.serve_cancels(2)
        else:
            first, second = await await_or_fail(
                asyncio.gather(run.agent_cancel(), run.agent_cancel()), what="two cancels",
            )
        await run.finish_turn()
        await run.settle()
        after = await run.agent_cancel()
        await _verify(run, "canceled")
        _expect("one cancel write", len(writes.calls) == 1, len(writes.calls))
        # serve answers one request at a time: the second sees the decided state.
        _expect("same reply", first == second or mode == "serve", (first, second))
        _expect("reply after decide", after in {"already_finished", "canceled"}, after)


@row("dup-propose")
async def _dup_propose(tmp_path, monkeypatch, mode):
    run = await _open(tmp_path, monkeypatch, mode, Actor([_text("child complete")]))
    async with run.cleanup:
        # finalize_background_agent runs finish_agent_lifecycle as an off-loop
        # command; count only the writers that a run's end invokes.
        finished = gate_store(
            monkeypatch, "finish_agent_lifecycle",
            when=lambda store, *_: store.session_dir == run.child_dir
            and threading.current_thread() is threading.main_thread(),
        )
        finalized = _held(monkeypatch, run, "finalize_background_agent")
        run.start()
        await run.finish_turn()
        await run.settle()
        await _verify(run, "completed")
        writes = len(finished.calls) + len(finalized.calls)
        _expect("one terminal D1 write", writes == 1, writes)


@row("dup-close")
async def _dup_close(tmp_path, monkeypatch, mode):
    run = await _open(tmp_path, monkeypatch, "fg", _gated_child())
    async with run.cleanup:
        run.start()
        await await_or_fail(run.child.started.wait(), what="child provider call")
        await await_or_fail(
            asyncio.gather(run.loop.close(), run.loop.close()), what="two concurrent closes",
        )
        await run.finish_turn()
        await _verify(run, "canceled")
        await await_or_fail(run.loop.close(), what="close after close")


# table -----------------------------------------------------------------------

def _xfail(reason: str) -> pytest.MarkDecorator:
    return pytest.mark.xfail(strict=True, raises=TerminalMismatch, reason=reason)


_SETUP_D4 = "setup cancel after D4 writes D1/D5 canceled but leaves the D4 marker (fails at D4)"
XFAIL: dict[tuple[str, str], str] = {
    **{("cancel@after_marker", f"{mode}:{point}"): _SETUP_D4
       for mode in ("fg", "bg") for point in SETUP_POINTS},
    **{("cancel@after_marker", f"serve:{point}"):
       "a serve spawn closed in setup writes D1 canceled and no parent receipt (fails at D5)"
       for point in SETUP_POINTS},
    ("cancel@turn_limit", "fg"): "the turn-limit D1 completed lands before the claimed cancel (fails at D1)",
    ("cancel@finish_gate", "fg"):
        "bug 4: the child finishes after the grandchild's cancel write and before its own (fails at D1)",
    ("cancel@deciding", "fg"): "a cancel during the child's D1 write answers unknown child handle (fails at the reply)",
    ("cancel@deciding", "bg"): "a cancel during finalize waits 1 s and answers unconfirmed (fails at the reply)",
    ("cancel-vs-explicit-finish", "explicit"):
        "a cancel after the D3 finish row waits 1 s and answers unconfirmed (fails at the reply)",
    ("abort-after-complete", "fg"): "an abort after the child's D1 completed still writes D5 canceled (fails at D5)",
    ("fail@D1_cancel_write", "fg"): "a failed cancel write lets the child complete; D5 completed (fails at D5)",
    ("fail@D1_cancel_write", "bg"): "a failed cancel write lets the child complete; D5 completed (fails at D5)",
    ("fail@D1_finish_write", "fg"): "a failed D1 write turns D5 into an error (fails at D5)",
    ("fail@D1_finish_write", "bg"): "a failed finalize write leaves no notification (fails at D5)",
    ("fail@D6_publish", "fg"): "a raising child_ended sink fails the parent turn before D4 is cleared (fails at D4)",
    ("restart@D5_no_D1", "fg"): "recovery leaves D1 running when D5 exists (fails at D1; slice 4)",
}

MODES = {
    "success": ("fg", "bg", "explicit"),
    "fail-child": ("fg", "bg"),
    "cancel@open_store": ("fg", "bg"),
    "close@open_store": ("fg", "bg", "serve"),
    "cancel@after_marker": tuple(
        f"{mode}:{point}" for mode in ("fg", "bg", "serve") for point in SETUP_POINTS
    ),
    "cancel@anchor_write": ("serve",),
    "cancel@model_turn": ("fg", "bg"),
    "cancel@turn_limit": ("fg",),
    "abort@generator_close": ("fg",),
    "cancel@question_pending": ("explicit",),
    "cancel@approval": ("fg",),
    "cancel@finish_gate": ("fg", "bg"),
    "cancel@deciding": ("fg", "bg"),
    "cancel-vs-explicit-finish": ("explicit",),
    "explicit-cancel-first": ("explicit",),
    "abort-after-complete": ("fg",),
    "close@running": ("fg", "bg"),
    "fail@D1_cancel_write": ("fg", "bg"),
    "fail@D1_finish_write": ("fg", "bg"),
    "fail@D5_write": ("fg", "bg"),
    "fail@D6_publish": ("fg", "bg"),
    "restart@D4_only": ("fg", "bg"),
    "restart@D1_done_no_D5": ("fg", "bg"),
    "restart@D5_no_D1": ("fg",),
    "restart@D3_finish": ("explicit",),
    "dup-cancel": ("fg", "bg", "serve"),
    "dup-propose": ("bg",),
    "dup-close": ("root",),
}
assert set(MODES) == set(ROWS)
CASES = [
    pytest.param(name, mode, id=f"{name}-{mode}", marks=(
        [_xfail(XFAIL[name, mode])] if (name, mode) in XFAIL else []
    ))
    for name, modes in MODES.items()
    for mode in modes
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "mode"), CASES)
async def test_child_run_termination(tmp_path, monkeypatch, name, mode):
    await ROWS[name](tmp_path, monkeypatch, mode)
