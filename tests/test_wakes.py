from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from zeta.agent.wakes import (
    MAX_PARENT_ADMISSIONS_PER_PROCESS,
    MAX_PENDING_WAKES,
    WakeAdmission,
    WakeScheduler,
)
from zeta.protocol.types import ToolCall
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry
from zeta.tools.wake import register_wake


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
        too_soon = await registry.execute(
            ToolCall("soon", "wake", {"after_seconds": 9, "note": "too soon"})
        )
        assert too_soon["isError"] is True
        assert "between 10 and 604800" in too_soon["content"][0]["text"]

        due_times: list[str] = []
        wake_ids: list[str] = []
        for index in range(MAX_PENDING_WAKES):
            result = await registry.execute(
                ToolCall(
                    f"schedule-{index}",
                    "wake",
                    {"after_seconds": 20 - index, "note": f"note {index}"},
                )
            )
            assert result["isError"] is False
            assert result["structuredContent"]["durability"] == "process_local"
            assert (
                "lost when this zeta session process restarts"
                in result["content"][0]["text"]
            )
            due_times.append(result["structuredContent"]["due_at"])
            wake_ids.append(result["structuredContent"]["wake_id"])

        overflow = await registry.execute(
            ToolCall(
                "overflow",
                "wake",
                {"after_seconds": 30, "note": "one too many"},
            )
        )
        assert overflow["isError"] is True
        assert "at most 8 wakes" in overflow["content"][0]["text"]

        listed = await registry.execute(ToolCall("list", "wake", {"action": "list"}))
        listed_wakes = listed["structuredContent"]["wakes"]
        assert [item["due_at"] for item in listed_wakes] == sorted(due_times)

        canceled = await registry.execute(
            ToolCall(
                "cancel",
                "wake",
                {"action": "cancel", "wake_id": wake_ids[0]},
            )
        )
        assert canceled["structuredContent"] == {
            "action": "cancel",
            "wake_id": wake_ids[0],
            "canceled": True,
        }
    finally:
        await scheduler.close()
        await registry.close()


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
        await scheduler.close()


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
        await scheduler.close()


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
        await scheduler.close()


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
        await scheduler.close()


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
        async with asyncio.timeout(10):
            await (release_wait if len(waits) == 1 else hold_timer).wait()

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
        await scheduler.close()


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

    lost = await scheduler.close()

    assert lost == 1
    assert len(owner_checks) == 1
    assert scheduler.closed
