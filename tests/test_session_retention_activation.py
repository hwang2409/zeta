from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

import zeta.session_retention as retention_module
from tests.support.retention_owner import _age_tree, _create_closed
from tests.support.storage_homes import initialize_storage_home
from zeta.core.session import SessionManager
from zeta.session_retention import (
    RetentionBounds,
    RetentionPlanner,
    read_retention_history,
    retention_authority,
)
from zeta.session_retention.activation import make_retention_owner
from zeta.storage_format import open_writer


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "days", "expected"),
    [
        ("fresh preview", 30, "preview"),
        ("disabled owner", 0, "disarmed"),
    ],
)
async def test_owner_lifecycle_table(
    tmp_path: Path, name: str, days: int, expected: str
) -> None:
    del name
    home = tmp_path / "home"
    initialize_storage_home(home)
    (home / "storage-format.json").write_text('{"writer_epoch":11}\n')
    published: list[dict[str, object]] = []

    lease = await asyncio.to_thread(open_writer, home)
    try:
        if days == 0:
            def seed_arm():
                with RetentionPlanner(home, lease) as planner:
                    preview = planner.run_pass(days=30)
                assert preview.complete
                return retention_authority(home, lease, days=30).applied
            assert await asyncio.to_thread(seed_arm)

        owner = make_retention_owner(home, lease, lambda: days)
        projections, failure = await asyncio.to_thread(owner._run_episode)
        assert failure is None, failure
        published.extend(projections)
        await owner.stop()

        assert published
        result = published[-1]
        assert (result.mode if hasattr(result, "mode") else result["mode"]) == expected
    finally:
        await asyncio.to_thread(lease.close)

@pytest.mark.asyncio
async def test_owner_corrupt_history_survives_and_records_error(tmp_path: Path) -> None:
    home = tmp_path / "home"
    initialize_storage_home(home)
    (home / "storage-format.json").write_text('{"writer_epoch":11}\n')
    (home / "retention-history.jsonl").write_text("not json\n")
    lease = await asyncio.to_thread(open_writer, home)
    try:
        owner = make_retention_owner(home, lease, lambda: 30)
        projections, failure = await asyncio.to_thread(owner._run_episode)
        assert not projections
        assert failure is not None
        published = asyncio.Event()
        owner.set_publish(lambda _projection: published.set())
        owner.start()
        await asyncio.wait_for(published.wait(), 10)
        assert owner.task is not None and not owner.task.done()
        await owner.stop()
        assert '"mode":"none"' in (home / "retention-history.jsonl").read_text()
    finally:
        await asyncio.to_thread(lease.close)


@pytest.mark.asyncio
async def test_owner_publish_failure_does_not_stop_owner(tmp_path: Path) -> None:
    home = tmp_path / "home"
    initialize_storage_home(home)
    (home / "storage-format.json").write_text('{"writer_epoch":11}\n')
    lease = await asyncio.to_thread(open_writer, home)
    try:
        owner = make_retention_owner(home, lease, lambda: 30)
        owner.set_publish(lambda _projection: (_ for _ in ()).throw(RuntimeError("publish")))
        owner.start()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert owner.task is not None and not owner.task.done()
        await owner.stop()
    finally:
        await asyncio.to_thread(lease.close)


@pytest.mark.asyncio
async def test_two_owners_append_without_lost_rows(tmp_path: Path) -> None:
    home = tmp_path / "home"
    initialize_storage_home(home)
    (home / "storage-format.json").write_text('{"writer_epoch":11}\n')
    lease = await asyncio.to_thread(open_writer, home)
    try:
        first = make_retention_owner(home, lease, lambda: 30)
        second = make_retention_owner(home, lease, lambda: 30)
        results = await asyncio.wait_for(
            asyncio.gather(
                asyncio.to_thread(first._run_episode), asyncio.to_thread(second._run_episode)
            ),
            10,
        )
        assert all(failure is None for _projections, failure in results), results
        rows = (home / "retention-history.jsonl").read_text().splitlines()
        assert len(rows) >= 2
        assert {json.loads(row)["mode"] for row in rows} >= {"preview"}
        assert all(json.loads(row)["preview_id"] for row in rows)
        state = json.loads((home / "retention-state.json").read_text())
        assert state["preview"]["preview_id"] == json.loads(rows[-1])["preview_id"]
        await first.stop()
        await second.stop()
    finally:
        await asyncio.to_thread(lease.close)

@pytest.mark.asyncio
async def test_owner_cancel_mid_episode_releases_lease(tmp_path: Path, monkeypatch) -> None:
    import threading

    home = tmp_path / "home"
    initialize_storage_home(home)
    (home / "storage-format.json").write_text('{"writer_epoch":11}\n')
    lease = await asyncio.to_thread(open_writer, home)
    entered = threading.Event()
    release = threading.Event()
    original = RetentionPlanner.run_pass

    def blocked(planner, *, days):
        entered.set()
        assert release.wait(10)
        return original(planner, days=days)

    monkeypatch.setattr(RetentionPlanner, "run_pass", blocked)
    owner = make_retention_owner(home, lease, lambda: 30)
    try:
        owner.start()
        await asyncio.wait_for(asyncio.to_thread(entered.wait, 10), 10)
        stopping = asyncio.create_task(owner.stop())
        await asyncio.sleep(0)
        release.set()
        await asyncio.wait_for(stopping, 10)
        replacement = await asyncio.to_thread(open_writer, home)
        replacement.close()
    finally:
        release.set()
        await asyncio.to_thread(lease.close)


@pytest.mark.asyncio
async def test_owner_repeats_bounded_preview_without_daily_wake(tmp_path: Path) -> None:
    home = tmp_path / "home"
    initialize_storage_home(home)
    (home / "storage-format.json").write_text('{"writer_epoch":11}\n')
    manager = SessionManager(home)
    await asyncio.to_thread(lambda: [_create_closed(manager, [], name=f"root-{index}") for index in range(4)])
    lease = await asyncio.to_thread(open_writer, home)
    try:
        owner = make_retention_owner(
            home, lease, lambda: 30,
            bounds=RetentionBounds(roots_advanced=1, metadata_candidates=1),
        )
        projections, failure = await asyncio.to_thread(owner._run_episode)
        assert failure is None
        previews = [item for item in projections if item["mode"] == "preview"]
        assert len(previews) >= 3
        assert previews[-1].complete
        await owner.stop()
    finally:
        await asyncio.to_thread(lease.close)


@pytest.mark.asyncio
async def test_armed_cleanup_publishes_deleted_and_removes_root(tmp_path: Path) -> None:
    home = tmp_path / "home"
    initialize_storage_home(home)
    (home / "storage-format.json").write_text('{"writer_epoch":11}\n')
    manager = SessionManager(home)
    opened = await asyncio.to_thread(_create_closed, manager, [], name="old")
    _age_tree(home / "sessions" / opened, days=100)
    lease = await asyncio.to_thread(open_writer, home)
    try:
        owner = make_retention_owner(
            home, lease, lambda: 30,
            now_ns=lambda: time.time_ns() + 365 * 86_400 * 1_000_000_000,
        )
        projections, failure = await asyncio.to_thread(owner._run_episode)
        assert failure is None
        preview = next(item for item in projections if item["mode"] == "preview" and item.complete)
        assert preview["root_ids"] == [opened]
        armed = await asyncio.to_thread(retention_authority, home, lease, days=30)
        assert armed.applied
        projections, failure = await asyncio.to_thread(owner._run_episode)
        assert failure is None
        deleted = [item for item in projections if item["mode"] == "deleted"]
        assert deleted and any(item["root_ids"] == [opened] for item in deleted)
        assert not (home / "sessions" / opened).exists()
        assert any(row["mode"] == "deleted" and row["root_ids"] == [opened] for row in read_retention_history(home))
        await owner.stop()
    finally:
        await asyncio.to_thread(lease.close)

@pytest.mark.asyncio
async def test_disarm_deadline_persists_pending_and_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    initialize_storage_home(home)
    (home / "storage-format.json").write_text('{"writer_epoch":11}\n')
    lease = await asyncio.to_thread(open_writer, home)
    try:
        def preview_pass():
            with RetentionPlanner(home, lease) as planner:
                return planner.run_pass(days=30)
        preview = await asyncio.to_thread(preview_pass)
        assert preview.complete
        assert await asyncio.to_thread(retention_authority, home, lease, days=30)

        def deadline(*_args, **_kwargs):
            raise TimeoutError

        monkeypatch.setattr(retention_module, "_try_authority", deadline)
        pending = await asyncio.to_thread(retention_authority, home, lease, days=0)
        assert not pending.applied and pending.projection["disable_pending"]
        assert (home / "retention-arm.json").exists()
        assert read_retention_history(home)[-1]["disable_pending"]
        monkeypatch.undo()
        cleared = await asyncio.to_thread(retention_authority, home, lease, days=0)
        assert cleared.applied and cleared.projection["mode"] == "disarmed"
        assert not (home / "retention-arm.json").exists()
    finally:
        await asyncio.to_thread(lease.close)


@pytest.mark.asyncio
async def test_armed_days_change_previews_before_deletion(tmp_path: Path) -> None:
    home = tmp_path / "home"
    initialize_storage_home(home)
    (home / "storage-format.json").write_text('{"writer_epoch":11}\n')
    manager = SessionManager(home)
    opened = await asyncio.to_thread(_create_closed, manager, [], name="old")
    _age_tree(home / "sessions" / opened, days=100)
    lease = await asyncio.to_thread(open_writer, home)
    try:
        clock = lambda: time.time_ns() + 365 * 86_400 * 1_000_000_000
        owner = make_retention_owner(home, lease, lambda: 30, now_ns=clock)
        first, failure = await asyncio.to_thread(owner._run_episode)
        assert failure is None and any(item.complete for item in first)
        assert (await asyncio.to_thread(retention_authority, home, lease, days=30)).applied
        current_days = [60]
        owner = make_retention_owner(home, lease, lambda: current_days[0], now_ns=clock)
        projections, failure = await asyncio.to_thread(owner._run_episode)
        assert failure is None
        assert any(item["mode"] == "preview" and item.complete and item["days"] == 60 for item in projections)
        assert (home / "sessions" / opened).exists()
        assert (home / "retention-arm.json").exists()
        await owner.stop()
    finally:
        await asyncio.to_thread(lease.close)
