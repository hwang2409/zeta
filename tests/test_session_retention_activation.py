from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.support.storage_homes import initialize_storage_home
from zeta.session_retention import (
    RetentionPlanner,
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
                    assert planner.run_pass(days=30).complete
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
