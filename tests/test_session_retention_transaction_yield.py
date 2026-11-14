from __future__ import annotations

import asyncio
import fcntl
import multiprocessing
import os
import threading
from pathlib import Path

import pytest

import zeta.session_retention.transaction as transaction_module
from tests.support.retention_owner import _age_tree, _create_closed, _run_sweep
from tests.support.storage_homes import initialize_storage_home
from zeta.core.session import SessionManager
from zeta.locking import RETENTION_HOME_ID, domain_lock_name
from zeta.session_retention import (
    RetentionPlanner,
    read_retention_history,
    retention_authority,
)
from zeta.session_retention.activation import make_retention_owner
from zeta.session_retention.handshake import planner_request_pending
from zeta.storage_format import open_writer


def _hold_user_request(path: str, ready, release) -> None:
    fd = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        ready.send(True)
        release.recv()
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


async def _armed_home(tmp_path: Path) -> tuple[Path, SessionManager, str, object]:
    home = tmp_path / "home"
    initialize_storage_home(home)
    (home / "storage-format.json").write_text('{"writer_epoch":11}\n')
    manager = SessionManager(home)
    session_id = await asyncio.to_thread(_create_closed, manager, [], name="old")
    _age_tree(home / "sessions" / session_id, days=100)
    lease = await asyncio.to_thread(open_writer, home)

    def seed() -> None:
        with RetentionPlanner(home, lease) as planner:
            result, _reasons = _run_sweep(planner)
        assert result.complete
        assert retention_authority(home, lease, days=30).applied

    await asyncio.to_thread(seed)
    return home, manager, session_id, lease


@pytest.mark.asyncio
@pytest.mark.parametrize("request_kind", ["user", "stop"])
async def test_real_transaction_checkpoint_yield_has_no_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request_kind: str
) -> None:
    home, manager, session_id, lease = await _armed_home(tmp_path)
    checkpoint = threading.Event()
    second_checkpoint = threading.Event()
    request_seen = threading.Event()
    stop_seen = threading.Event()
    charge_calls = 0
    original_charge = transaction_module._Budget.charge

    def charge(budget, size=0):
        nonlocal charge_calls
        charge_calls += 1
        if charge_calls == 1:
            checkpoint.set()
            if request_kind == "user":
                assert request_seen.wait(10)
            else:
                assert stop_seen.wait(10)
            raise transaction_module.TransactionYield
        elif charge_calls == 2:
            second_checkpoint.set()
        return original_charge(budget, size)

    monkeypatch.setattr(transaction_module._Budget, "charge", charge)
    baseline_rows = read_retention_history(home)
    owner = make_retention_owner(home, lease, lambda: 30)
    episodes: list[object] = []
    original_episode = owner._run_episode

    def record_episode():
        result = original_episode()
        episodes.append(result)
        return result

    owner._run_episode = record_episode
    published: list[object] = []
    owner.set_publish(lambda projection: published.append(projection))
    owner.start()
    opened = None
    request_process = None
    release_request = None
    request_released = False
    try:
        await asyncio.wait_for(asyncio.to_thread(checkpoint.wait, 10), 10)
        if request_kind == "user":
            ready_request, ready_child = multiprocessing.Pipe(False)
            release_child, release_request = multiprocessing.Pipe(False)
            request_process = multiprocessing.Process(
                target=_hold_user_request,
                args=(
                    str(home / ".domain-locks" / domain_lock_name(RETENTION_HOME_ID, "user-open-request")),
                    ready_child,
                    release_child,
                ),
            )
            request_process.start()
            assert await asyncio.to_thread(ready_request.recv)
            assert planner_request_pending(home)
            request_seen.set()
            open_task = asyncio.create_task(asyncio.to_thread(manager.open, session_id))
            opened = await asyncio.wait_for(open_task, 10)
            opened.store.close()
            release_request.send(True)
            request_released = True
            await asyncio.to_thread(request_process.join, 10)
            owner.wake()
            await asyncio.wait_for(asyncio.to_thread(second_checkpoint.wait, 10), 10)
            await owner.stop()
        else:
            stop_task = asyncio.create_task(owner.stop())
            stop_seen.set()
            await asyncio.wait_for(stop_task, 10)

        assert episodes
        rows = read_retention_history(home)
        new_rows = rows[len(baseline_rows) :]
        assert all(row.get("mode") not in {"yielded", "none"} for row in new_rows)
        assert all(not row.get("errors") for row in new_rows)
        assert all(
            (item.get("mode") if isinstance(item, dict) else getattr(item, "mode", None))
            not in {"yielded", "none"}
            for item in published
        )
        if request_kind == "stop":
            assert len(rows) == 1
            assert not published
    finally:
        if release_request is not None:
            if not request_released:
                release_request.send(True)
            release_request.close()
        if request_process is not None:
            request_process.join(10)
            if request_process.is_alive():
                request_process.terminate()
                request_process.join(10)
        if opened is not None:
            opened.store.close()
        await owner.stop()
        await asyncio.to_thread(lease.close)
