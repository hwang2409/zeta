from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.support.serve_close import close_serve
from tests.support.storage_homes import initialize_storage_home
from tests.test_server import _connect, _request
from tests.test_session_retention_surfaces import _hello
from zeta.server import ZetaServer
from zeta.session_retention import (
    RetentionPlanner,
    disarm_retention,
    retention_authority,
)
from zeta.storage_format import open_writer


async def _slash(home: Path, text: str) -> dict[str, object]:
    server = ZetaServer(home=home, cwd=home, port=0, provider="codex")
    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer, 1, features=["session_retention"])
        sid = (await _request(reader, writer, 2, "new_session", {"provider": "codex"}))[-1]["result"]["session"]["session_id"]
        return (await _request(reader, writer, 3, "slash_run", {"session_id": sid, "text": text}))[-1]["result"]
    finally:
        writer.close()
        await writer.wait_closed()
        await close_serve(server)


@pytest.mark.asyncio
@pytest.mark.parametrize(("epoch", "writes"), [(11, True), (10, False)])
async def test_public_authority_table(tmp_path: Path, epoch: int, writes: bool) -> None:
    home = tmp_path / "home"
    initialize_storage_home(home)
    (home / "storage-format.json").write_text(f'{{"writer_epoch":{epoch}}}\n')
    if epoch == 11:
        def seed_preview() -> None:
            lease = open_writer(home)
            try:
                planner = RetentionPlanner(home, lease)
                try:
                    assert planner.run_pass(days=30).complete
                finally:
                    planner.close()
            finally:
                lease.close()
        await asyncio.to_thread(seed_preview)
    arm = home / "retention-arm.json"
    history = home / "retention-history.jsonl"
    arm_result = await _slash(home, "/retention arm")
    disarm_result = await _slash(home, "/retention disarm")
    if writes:
        assert arm_result["kind"] == "output"
        assert disarm_result["kind"] == "output"
        assert "disabled" in disarm_result["text"]
        assert not arm.exists()
        assert history.exists()
        assert '"mode":"disarmed"' in history.read_text()
    else:
        assert not arm.exists()
        assert not history.exists()

@pytest.mark.asyncio
async def test_authority_zero_to_thirty_requires_fresh_preview(tmp_path: Path) -> None:
    home = tmp_path / "home"
    initialize_storage_home(home)
    (home / "storage-format.json").write_text('{"writer_epoch":11}\n')

    def cycle() -> tuple[bool, bool, bool]:
        lease = open_writer(home)
        try:
            planner = RetentionPlanner(home, lease)
            try:
                assert planner.run_pass(days=30).complete
            finally:
                planner.close()
            # an unarmed completed preview is also invalidated by disabling retention
            assert disarm_retention(home, lease).applied
            stale = retention_authority(home, lease, days=lambda: 30).applied
            planner = RetentionPlanner(home, lease)
            try:
                fresh = planner.run_pass(days=30).complete
            finally:
                planner.close()
            reapplied = retention_authority(home, lease, days=lambda: 30).applied
            return stale, fresh, reapplied
        finally:
            lease.close()

    stale, fresh, reapplied = await asyncio.to_thread(cycle)
    assert not stale
    assert fresh
    assert reapplied


@pytest.mark.asyncio
async def test_slash_arm_with_days_zero_disarms(tmp_path: Path) -> None:
    home = tmp_path / "home"
    initialize_storage_home(home)
    (home / "storage-format.json").write_text('{"writer_epoch":11}\n')

    def seed_armed() -> None:
        with open_writer(home) as lease:
            with RetentionPlanner(home, lease) as planner:
                assert planner.run_pass(days=30).complete
            assert retention_authority(home, lease, days=lambda: 30).applied

    await asyncio.to_thread(seed_armed)
    (home / "settings.toml").write_text("[retention]\ndays = 0\n")
    result = await _slash(home, "/retention arm")
    # serve's own owner may disarm first; either way the arm is gone and nothing was armed
    assert result["text"] in {"retention disabled", "retention already disabled"}
    assert not (home / "retention-arm.json").exists()
    assert '"mode":"armed"' not in (home / "retention-history.jsonl").read_text().splitlines()[-1]
