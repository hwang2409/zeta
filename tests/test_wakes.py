from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from zeta.agent.wakes import (
    MAX_PARENT_ADMISSIONS_PER_PROCESS,
    MAX_PENDING_WAKES,
    MAX_WAKE_RUN_COMPLETIONS,
    WakeAdmission,
    WakeScheduler,
)
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    MESSAGE_ORIGIN_METADATA,
    MessageOrigin,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
)
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry
from zeta.tools.wake import register_wake
from zeta.tui.app import TUIApp


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class Ids:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> str:
        self.value += 1
        return str(self.value)


async def _execute(registry: ToolRegistry, call: ToolCall) -> dict[str, object]:
    return await asyncio.wait_for(registry.execute(call), timeout=10)


async def _collect(stream: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    async def consume() -> list[StreamEvent]:
        return [event async for event in stream]

    try:
        return await asyncio.wait_for(consume(), timeout=10)
    finally:
        await asyncio.wait_for(stream.aclose(), timeout=10)


async def _close_scheduler(
    scheduler: WakeScheduler, registry: ToolRegistry | None = None
) -> None:
    try:
        await asyncio.wait_for(scheduler.close(), timeout=10)
    finally:
        if registry is not None:
            await asyncio.wait_for(registry.close(), timeout=10)


def _scheduler(
    session_class: str, clock: Clock, wakes: list[None] | None = None
) -> WakeScheduler:
    sink = wakes if wakes is not None else []
    return WakeScheduler(
        session_class,  # type: ignore[arg-type]
        on_due=lambda: sink.append(None),
        wall_clock=clock,
        id_factory=Ids(),
    )


@pytest.mark.asyncio
async def test_wake_tool_schedules_lists_cancels_and_enforces_pending_bound(
    tmp_path: Path,
) -> None:
    """Mutation: lower the minimum delay; the 9-second error assertion fails."""

    clock = Clock()
    scheduler = _scheduler("top_level", clock)
    registry = ToolRegistry(
        tmp_path, register_builtin=False, skill_catalog=SkillCatalog.empty()
    )
    register_wake(registry, scheduler)
    try:
        too_soon = await _execute(
            registry, ToolCall("soon", "wake", {"after_seconds": 9, "note": "too soon"})
        )
        assert too_soon["isError"] is True
        assert "between 10 and 604800" in too_soon["content"][0]["text"]

        due_times: list[str] = []
        wake_ids: list[str] = []
        for index in range(MAX_PENDING_WAKES):
            result = await _execute(
                registry,
                ToolCall(
                    f"schedule-{index}",
                    "wake",
                    {"after_seconds": 20 - index, "note": f"note {index}"},
                ),
            )
            assert result["isError"] is False
            assert result["structuredContent"]["durability"] == "process_local"
            assert (
                "lost when this zeta session process restarts"
                in result["content"][0]["text"]
            )
            due_times.append(result["structuredContent"]["due_at"])
            wake_ids.append(result["structuredContent"]["wake_id"])

        overflow = await _execute(
            registry,
            ToolCall(
                "overflow",
                "wake",
                {"after_seconds": 30, "note": "one too many"},
            ),
        )
        assert overflow["isError"] is True
        assert "at most 8 wakes" in overflow["content"][0]["text"]

        listed = await _execute(registry, ToolCall("list", "wake", {"action": "list"}))
        listed_wakes = listed["structuredContent"]["wakes"]
        assert [item["due_at"] for item in listed_wakes] == sorted(due_times)

        canceled = await _execute(
            registry,
            ToolCall(
                "cancel",
                "wake",
                {"action": "cancel", "wake_id": wake_ids[0]},
            ),
        )
        assert canceled["structuredContent"] == {
            "action": "cancel",
            "wake_id": wake_ids[0],
            "canceled": True,
        }
    finally:
        await _close_scheduler(scheduler, registry)


@pytest.mark.asyncio
async def test_top_level_external_run_allows_only_eight_root_wakes() -> None:
    """Mutation: raise the root limit; the ninth passive assertion fails."""

    clock = Clock()
    scheduler = _scheduler("top_level", clock)
    try:
        scheduler.begin_external_run()
        roots = [
            scheduler.schedule(after_seconds=10, note=f"root {index}")
            for index in range(1, 9)
        ]
        scheduler.cancel(roots[0].record.wake_id)
        ninth = scheduler.schedule(after_seconds=10, note="root 9")

        assert all(result.autonomous for result in roots)
        assert len({result.record.chain_id for result in roots}) == 8
        assert ninth.autonomous is False
        assert ninth.reason == "top_level_chain_limit"
    finally:
        await _close_scheduler(scheduler)


@pytest.mark.asyncio
async def test_top_level_wakes_stop_after_three_generations() -> None:
    """Mutation: allow generation 4; the final passive assertion fails."""

    clock = Clock()
    scheduler = _scheduler("top_level", clock)
    try:
        scheduler.begin_external_run()
        scheduled = scheduler.schedule(after_seconds=10, note="generation 1")
        assert scheduled.record.generation == 1
        for expected_generation in (1, 2, 3):
            clock.advance(10)
            assert scheduler.poll_due()
            activation = scheduler.admit_due()
            assert activation is not None and activation.admission is not None
            admission = activation.admission
            assert admission.records[0].generation == expected_generation
            scheduler.begin_wake_run(admission)
            successor = scheduler.schedule(
                after_seconds=10, note=f"generation {expected_generation + 1}"
            )
            if expected_generation < 3:
                assert successor.autonomous is True
            else:
                assert successor.autonomous is False
                assert successor.reason == "top_level_chain_limit"
        clock.advance(10)
        assert scheduler.poll_due()
        passive = scheduler.admit_due()
        assert passive is not None
        assert passive.admission is None
        assert passive.passive[0].generation == 4
    finally:
        await _close_scheduler(scheduler)


@pytest.mark.asyncio
async def test_parent_owned_lifetime_admits_exactly_96_runs() -> None:
    """Mutation: use <= for the lifetime check; a 97th admission appears."""

    clock = Clock()
    scheduler = _scheduler("parent_owned", clock)
    try:
        admissions: list[WakeAdmission] = []
        for index in range(MAX_PARENT_ADMISSIONS_PER_PROCESS):
            result = scheduler.schedule(after_seconds=10, note=f"run {index + 1}")
            assert result.autonomous is True
            clock.advance(10)
            assert scheduler.poll_due()
            activation = scheduler.admit_due()
            assert activation is not None and activation.admission is not None
            admissions.append(activation.admission)
        assert len({item.admission_id for item in admissions}) == 96
        assert scheduler.admission_count == 96

        denied = scheduler.schedule(after_seconds=10, note="run 97")
        assert denied.autonomous is False
        assert denied.reason == "parent_lifetime_limit"
        clock.advance(10)
        assert scheduler.poll_due()
        activation = scheduler.admit_due()
        assert activation is not None
        assert activation.admission is None
        assert activation.passive[0].passive_reason == "parent_lifetime_limit"
        assert scheduler.admission_count == 96
    finally:
        await _close_scheduler(scheduler)


@pytest.mark.asyncio
async def test_due_wakes_coalesce_and_later_wake_stays_out_of_closed_cohort() -> None:
    """Mutation: select records after dispatch; wake b joins the first cohort."""

    clock = Clock()
    scheduler = _scheduler("parent_owned", clock)
    try:
        first = scheduler.schedule(after_seconds=10, note="a")
        peer = scheduler.schedule(after_seconds=10, note="a peer")
        later = scheduler.schedule(after_seconds=20, note="b")
        clock.advance(10)
        assert scheduler.poll_due()
        first_activation = scheduler.admit_due()
        assert first_activation is not None and first_activation.admission is not None
        assert {record.wake_id for record in first_activation.admission.records} == {
            first.record.wake_id,
            peer.record.wake_id,
        }
        assert scheduler.admission_count == 1

        clock.advance(10)
        assert scheduler.poll_due()
        second_activation = scheduler.admit_due()
        assert second_activation is not None and second_activation.admission is not None
        assert [record.wake_id for record in second_activation.admission.records] == [
            later.record.wake_id
        ]
        assert scheduler.admission_count == 2
    finally:
        await _close_scheduler(scheduler)


@pytest.mark.asyncio
async def test_timer_rechecks_utc_after_one_sixty_second_slice() -> None:
    """Mutation: wait for the full UTC delta; the slice assertion fails."""

    clock = Clock()
    wait_started = asyncio.Event()
    release_wait = asyncio.Event()
    hold_timer = asyncio.Event()
    fired = asyncio.Event()
    waits: list[float] = []

    async def controlled_wait(_changed: asyncio.Event, timeout: float) -> None:
        waits.append(timeout)
        wait_started.set()
        await asyncio.wait_for(
            (release_wait if len(waits) == 1 else hold_timer).wait(), timeout=10
        )

    scheduler = WakeScheduler(
        "top_level",
        on_due=fired.set,
        wall_clock=clock,
        id_factory=Ids(),
        wait_for_change=controlled_wait,
    )
    try:
        scheduler.schedule(after_seconds=7 * 24 * 60 * 60, note="after suspend")
        await asyncio.wait_for(wait_started.wait(), timeout=10)
        assert waits == [60.0]

        clock.advance(7 * 24 * 60 * 60)
        release_wait.set()
        await asyncio.wait_for(fired.wait(), timeout=10)
        assert scheduler.poll_due() is False
    finally:
        release_wait.set()
        await _close_scheduler(scheduler)


@pytest.mark.asyncio
async def test_due_wake_runs_through_production_harness_notice_path(
    tmp_path: Path,
) -> None:
    """Mutation: pass the wake as user-role input; the role assertion fails."""

    backend = FakeBackend([ScriptedTurn([TextContent("reminder handled")])])
    store = ConversationStore(tmp_path / "sessions")
    loop = AgentLoop(backend, store, skill_catalog=SkillCatalog.empty())
    old_scheduler = loop.wake_scheduler
    assert old_scheduler is not None
    await asyncio.wait_for(old_scheduler.close(), timeout=10)
    clock = Clock()
    scheduler = _scheduler("top_level", clock)
    loop.wake_scheduler = scheduler
    try:
        scheduled = scheduler.schedule(after_seconds=10, note="check staging")
        clock.advance(10)

        notice = loop.notification_system_message()
        assert notice is not None
        assert notice.role is MessageRole.SYSTEM
        assert (
            notice.metadata[MESSAGE_ORIGIN_METADATA]
            == MessageOrigin.HARNESS_NUDGE.value
        )
        assert notice.metadata["zeta_event"] == "wake"
        assert notice.metadata["wake_ids"] == [scheduled.record.wake_id]
        assert loop.schedule_notification_turn()

        await _collect(loop.run_notification_turn())

        assert len(backend.calls) == 1
        delivered = next(
            message
            for message in backend.calls[0][0]
            if message.metadata.get("zeta_event") == "wake"
        )
        assert delivered.role is MessageRole.SYSTEM
        assert (
            delivered.metadata[MESSAGE_ORIGIN_METADATA]
            == MessageOrigin.HARNESS_NUDGE.value
        )
        assert all(
            message.role is not MessageRole.USER for message in backend.calls[0][0]
        )

        output = StringIO()
        app = TUIApp(
            loop,
            provider="codex",
            model="offline",
            console=Console(file=output, force_terminal=False),
        )
        app._rebuild_transcript()
        assert "[zeta] wake reminder:" in output.getvalue()
    finally:
        try:
            await asyncio.wait_for(loop.close(), timeout=10)
        finally:
            store.close()


@pytest.mark.asyncio
async def test_passive_wake_is_presented_without_provider_dispatch(
    tmp_path: Path,
) -> None:
    """Mutation: dispatch passive due wakes to the provider; the call assertion fails."""

    backend = FakeBackend([ScriptedTurn([TextContent("ordinary reply")])])
    store = ConversationStore(tmp_path / "sessions")
    loop = AgentLoop(backend, store, skill_catalog=SkillCatalog.empty())
    old_scheduler = loop.wake_scheduler
    assert old_scheduler is not None
    await asyncio.wait_for(old_scheduler.close(), timeout=10)
    clock = Clock()
    scheduler = _scheduler("top_level", clock)
    loop.wake_scheduler = scheduler
    try:
        roots = [
            scheduler.schedule(after_seconds=10, note=f"root {index}")
            for index in range(8)
        ]
        scheduler.cancel(roots[0].record.wake_id)
        passive = scheduler.schedule(after_seconds=10, note="passive reminder")
        assert passive.autonomous is False
        for root in roots[1:]:
            scheduler.cancel(root.record.wake_id)
        clock.advance(10)

        notice = loop.notification_system_message()
        assert notice is not None
        assert loop.schedule_notification_turn()
        events = await _collect(loop.run_notification_turn())

        assert backend.calls == []
        presented = next(
            event.message
            for event in events
            if event.type is StreamEventType.MESSAGE_END
        )
        assert presented is not None
        assert presented.role is MessageRole.SYSTEM
        assert presented.metadata["zeta_event"] == "wake"
        assert "passive reason: top_level_chain_limit" in presented.content[0].text
    finally:
        try:
            await asyncio.wait_for(loop.close(), timeout=10)
        finally:
            store.close()


@pytest.mark.asyncio
async def test_wake_run_stops_after_eight_provider_completions(tmp_path: Path) -> None:
    """Mutation: raise the wake completion cap; the call-count assertion fails."""

    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[ToolCall(f"todo-{index}", "todo", {"items": []})])
            for index in range(MAX_WAKE_RUN_COMPLETIONS + 1)
        ]
    )
    store = ConversationStore(tmp_path / "sessions")
    loop = AgentLoop(backend, store, skill_catalog=SkillCatalog.empty())
    old_scheduler = loop.wake_scheduler
    assert old_scheduler is not None
    await asyncio.wait_for(old_scheduler.close(), timeout=10)
    clock = Clock()
    scheduler = _scheduler("top_level", clock)
    loop.wake_scheduler = scheduler
    try:
        scheduler.schedule(after_seconds=10, note="bounded run")
        clock.advance(10)
        assert loop.notification_system_message() is not None
        assert loop.schedule_notification_turn()

        events = await _collect(loop.run_notification_turn())

        assert len(backend.calls) == MAX_WAKE_RUN_COMPLETIONS
        assert events[-2].error is not None
        assert events[-2].error.code == "max_turns"
        assert events[-2].error.message.endswith(": 8")
    finally:
        try:
            await asyncio.wait_for(loop.close(), timeout=10)
        finally:
            store.close()


@pytest.mark.asyncio
async def test_agent_loop_registers_wake_and_discards_it_on_close(
    tmp_path: Path,
) -> None:
    """Mutation: omit the production scheduler close; the closed assertion fails."""

    store = ConversationStore(tmp_path / "sessions")
    loop = AgentLoop(FakeBackend([]), store, skill_catalog=SkillCatalog.empty())
    scheduler = loop.wake_scheduler
    assert scheduler is not None
    try:
        result = await _execute(
            loop.tool_registry,
            ToolCall(
                "schedule",
                "wake",
                {"after_seconds": 10, "note": "lost on loop close"},
            ),
        )
        assert result["isError"] is False
        assert scheduler.owned_work().pending_ids
    finally:
        try:
            await asyncio.wait_for(loop.close(), timeout=10)
        finally:
            store.close()

    assert scheduler.closed


@pytest.mark.asyncio
async def test_agent_loop_starts_each_user_turn_with_fresh_root_allowance(
    tmp_path: Path,
) -> None:
    """Mutation: omit begin_external_run in run_turn; the new root stays passive."""

    store = ConversationStore(tmp_path / "sessions")
    loop = AgentLoop(FakeBackend([]), store, skill_catalog=SkillCatalog.empty())
    scheduler = loop.wake_scheduler
    assert scheduler is not None
    stream = None
    try:
        roots = [
            scheduler.schedule(after_seconds=10, note=f"root {index}")
            for index in range(8)
        ]
        scheduler.cancel(roots[0].record.wake_id)
        ninth = scheduler.schedule(after_seconds=10, note="root 9")
        assert ninth.autonomous is False
        scheduler.cancel(ninth.record.wake_id)

        stream = loop.run_turn("ordinary user turn", origin=MessageOrigin.USER)
        await asyncio.wait_for(stream.aclose(), timeout=10)
        stream = None

        new_root = scheduler.schedule(after_seconds=10, note="new root")
        assert new_root.autonomous is True
        assert new_root.record.generation == 1
    finally:
        try:
            if stream is not None:
                await asyncio.wait_for(stream.aclose(), timeout=10)
        finally:
            try:
                await asyncio.wait_for(loop.close(), timeout=10)
            finally:
                store.close()


@pytest.mark.asyncio
async def test_close_discards_process_local_state_and_stops_owner_checks() -> None:
    """Mutation: retain records during close; the lost-state assertion fails."""

    clock = Clock()
    owner_checks: list[None] = []
    scheduler = WakeScheduler(
        "top_level",
        on_due=lambda: None,
        wall_clock=clock,
        assert_owned=lambda: owner_checks.append(None),
        id_factory=Ids(),
    )
    scheduler.schedule(after_seconds=10, note="lost on restart")
    assert scheduler.owned_work().pending_ids

    lost = await asyncio.wait_for(scheduler.close(), timeout=10)

    assert lost == 1
    assert len(owner_checks) == 1
    assert scheduler.closed
