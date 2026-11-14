from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.support.storage_homes import initialize_storage_home
from zeta.session_retention import (
    RetentionBounds,
    RetentionPlanner,
    retention_authority,
)
from zeta.session_retention.activation import RetentionActivationOwner
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

        owner = RetentionActivationOwner(
            lambda: RetentionPlanner(
                home,
                lease,
                bounds=RetentionBounds(
                    metadata_candidates=2,
                    roots_advanced=1,
                    entries_per_root=8,
                    entries_total=16,
                ),
            ),
            lambda: days,
            wake_seconds=60,
            publish=None,
            home=home,
            activation_enabled=True,
            disable=lambda: retention_authority(home, lease, days=0).projection,
        )
        projections, failure = await asyncio.to_thread(owner._run_episode)
        assert failure is None, failure
        published.extend(projections)
        await owner.stop()

        assert published
        result = published[-1]
        assert (result.mode if hasattr(result, "mode") else result["mode"]) == (None if expected == "preview" else expected)
    finally:
        await asyncio.to_thread(lease.close)
