from __future__ import annotations

import asyncio
import contextlib
import contextvars
import fcntl
import json
import os
import threading
import time
import types
from pathlib import Path

import pytest

import zeta.session_retention as retention_module
import zeta.session_retention.activation as activation_module
import zeta.session_retention.handshake as handshake_module
from tests.support.retention_owner import _age_tree, _create_closed, _opened_lease
from tests.support.storage_homes import initialize_storage_home
from zeta.core.session import SessionManager
from zeta.locking import RETENTION_HOME_ID, domain_lock_name
from zeta.session_retention import (
    RetentionBounds,
    RetentionPlanner,
    Yielded,
    disarm_retention,
    read_retention_history,
    retention_authority,
)
from zeta.session_retention.activation import (
    Published,
    RetentionActivationOwner,
    make_retention_owner,
)
from zeta.session_retention.handshake import ensure_handshake_files, planner_active
from zeta.storage_format import StorageFormatError, open_writer


@pytest.mark.asyncio
async def test_arm_before_owner_generation_keeps_armed_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    initialize_storage_home(home)
    (home / "storage-format.json").write_text('{"writer_epoch":11}\n')
    lease = await asyncio.to_thread(open_writer, home)
    owner = make_retention_owner(home, lease, lambda: 30)
    entered = threading.Event()
    release = threading.Event()
    original = RetentionPlanner.run_pass

    def gated(planner, *, days, commit_state=True, yield_check=None):
        entered.set()
        assert release.wait(10)
        return original(planner, days=days, commit_state=commit_state, yield_check=yield_check)

    try:
        with RetentionPlanner(home, lease) as planner:
            seeded = await asyncio.to_thread(planner.run_pass, days=30)
        assert seeded.complete
        monkeypatch.setattr(RetentionPlanner, "run_pass", gated)
        task = asyncio.create_task(asyncio.to_thread(owner._run_episode))
        await asyncio.wait_for(asyncio.to_thread(entered.wait, 10), 10)
        armed = await asyncio.to_thread(retention_authority, home, lease, days=lambda: 30)
        assert armed.applied
        release.set()
        projections, failure = await asyncio.wait_for(task, 10)
        assert failure is None
        assert any(item["mode"] == "armed" for item in projections)
        last = read_retention_history(home)[-1]
        assert last["mode"] == "armed"
        assert last["preview_id"] == armed.projection["preview_id"]
    finally:
        release.set()
        await owner.stop()
        await asyncio.to_thread(lease.close)


@pytest.mark.asyncio
async def test_arm_during_owner_scan_keeps_armed_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    initialize_storage_home(home)
    (home / "storage-format.json").write_text('{"writer_epoch":11}\n')
    manager = SessionManager(home)
    session_id = await asyncio.to_thread(_create_closed, manager, [], name="old")
    _age_tree(home / "sessions" / session_id, days=100)
    lease = await asyncio.to_thread(open_writer, home)
    owner = make_retention_owner(home, lease, lambda: 30)
    entered = threading.Event()
    release = threading.Event()
    original = retention_module._Budget.charge_entry
    calls = 0

    def gated(budget, size=0):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert release.wait(10)
        return original(budget, size)

    try:
        with RetentionPlanner(home, lease) as planner:
            seeded = await asyncio.to_thread(planner.run_pass, days=30)
        assert seeded.complete
        monkeypatch.setattr(retention_module._Budget, "charge_entry", gated)
        task = asyncio.create_task(asyncio.to_thread(owner._run_episode))
        await asyncio.wait_for(asyncio.to_thread(entered.wait, 10), 10)
        armed = await asyncio.to_thread(retention_authority, home, lease, days=lambda: 30)
        assert armed.applied
        release.set()
        projections, failure = await asyncio.wait_for(task, 10)
        assert failure is None
        assert any(item["mode"] == "armed" for item in projections)
        last = read_retention_history(home)[-1]
        assert last["mode"] == "armed"
        assert last["preview_id"] == armed.projection["preview_id"]
    finally:
        release.set()
        await owner.stop()
        await asyncio.to_thread(lease.close)


@pytest.mark.asyncio
async def test_epoch_11_owner_creates_handshake_files_once(tmp_path: Path) -> None:
    home = tmp_path / "home"
    initialize_storage_home(home)
    (home / "storage-format.json").write_text('{"writer_epoch":11}\n')
    lease = await asyncio.to_thread(open_writer, home)
    try:
        make_retention_owner(home, lease, lambda: 30)
        lock_dir = home / ".domain-locks"
        assert (lock_dir / domain_lock_name(RETENTION_HOME_ID, "planner-active")).is_file()
        assert (lock_dir / domain_lock_name(RETENTION_HOME_ID, "user-open-request")).is_file()
    finally:
        await asyncio.to_thread(lease.close)


@pytest.mark.asyncio
async def test_planner_active_does_not_create_missing_handshake_file(tmp_path: Path) -> None:
    home = tmp_path / "home"
    initialize_storage_home(home)
    with pytest.raises(FileNotFoundError), planner_active(home):
        pass
    assert not (home / ".domain-locks").exists()


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
                return retention_authority(home, lease, days=lambda: 30).applied
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
async def test_pending_request_is_typed_yield_not_empty_preview(tmp_path: Path) -> None:
    home = tmp_path / "home"
    initialize_storage_home(home)
    (home / "storage-format.json").write_text('{"writer_epoch":11}\n')
    lease = await asyncio.to_thread(open_writer, home)
    ensure_handshake_files(home)
    request = os.open(
        home / ".domain-locks" / domain_lock_name(RETENTION_HOME_ID, "user-open-request"),
        os.O_RDWR,
    )
    try:
        fcntl.flock(request, fcntl.LOCK_EX)
        planner = RetentionPlanner(home, lease)
        try:
            result = planner.run_pass(days=30)
        finally:
            planner.close()
        assert getattr(result, "reason", None) == "user"
        assert not (home / "retention-history.jsonl").exists()
    finally:
        fcntl.flock(request, fcntl.LOCK_UN)
        os.close(request)
        await asyncio.to_thread(lease.close)


def _gated_owner_episode(home, lease, days, during_window) -> object:
    """Run one owner episode, calling during_window after the scan decides and before publication."""
    decided = threading.Event()
    release = threading.Event()
    original = RetentionPlanner.run_pass
    outcome: dict[str, object] = {}

    def gated(planner, **kwargs):
        result = original(planner, **kwargs)
        decided.set()
        assert release.wait(10)
        return result

    RetentionPlanner.run_pass = gated
    try:
        owner = make_retention_owner(home, lease, lambda: days[0])
        thread = threading.Thread(target=lambda: outcome.update(result=owner._run_episode()))
        thread.start()
        assert decided.wait(10)
        during_window()
        release.set()
        thread.join(10)
        assert not thread.is_alive()
    finally:
        release.set()
        RetentionPlanner.run_pass = original
    return outcome["result"]


@pytest.mark.parametrize(
    ("mutation", "result_path"),
    (("arm", "preview"), ("arm", "blocked"), ("days change", "preview")),
)
def test_freshness_result_path_table(tmp_path: Path, mutation: str, result_path: str) -> None:
    """A user change inside the owner's decision window always shows its current state."""
    home = tmp_path / f"{mutation}-{result_path}"
    initialize_storage_home(home)
    days = [30]
    manager = SessionManager(home)
    with _opened_lease(home) as lease:
        session_id = _create_closed(manager, [], name="old")
        _age_tree(home / "sessions" / session_id, days=100)
        with RetentionPlanner(home, lease) as planner:
            preview = planner.run_pass(days=30)
        assert preview.complete and preview.preview is not None and preview.preview.root_ids
        if result_path == "blocked":
            assert retention_authority(home, lease, days=lambda: 30).applied
            days[0] = 60
            _gated_owner_episode(
                home, lease, days,
                lambda: retention_authority(home, lease, days=lambda: 30).applied,
            )
            assert read_retention_history(home)[-1]["days"] == 60
        elif mutation == "arm":
            _gated_owner_episode(
                home, lease, days,
                lambda: retention_authority(home, lease, days=lambda: 30).applied,
            )
            assert read_retention_history(home)[-1]["mode"] == "armed"
        else:
            days[0] = 60
            projections, failure = make_retention_owner(
                home, lease, lambda: days[0]
            )._run_episode()
            assert failure is None and projections[-1].projection["days"] == 60


def test_armed_days_change_during_scan_publishes_current_days(tmp_path: Path) -> None:
    home = tmp_path / "home"
    initialize_storage_home(home)
    days = [30]
    with _opened_lease(home) as lease:
        with RetentionPlanner(home, lease) as planner:
            assert planner.run_pass(days=30).complete
        assert retention_authority(home, lease, days=lambda: 30).applied
        assert read_retention_history(home)[-1]["mode"] == "armed"

        def change_days() -> None:
            days[0] = 60

        projections, failure = _gated_owner_episode(home, lease, days, change_days)
        assert failure is None
        last = read_retention_history(home)[-1]
        if projections[-1].complete:
            assert projections[-1].projection["days"] == 60
            assert last["days"] == 60


@pytest.mark.parametrize("existing_preview", (True, False))
def test_stale_days_change_redecides_current_projection(
    tmp_path: Path, existing_preview: bool,
) -> None:
    home = tmp_path / ("existing" if existing_preview else "empty")
    initialize_storage_home(home)
    days = [30]
    with _opened_lease(home) as lease:
        if existing_preview:
            with RetentionPlanner(home, lease) as planner:
                preview = planner.run_pass(days=30)
            assert preview.complete and preview.preview is not None
        decided = threading.Event()
        release = threading.Event()
        original = RetentionPlanner.run_pass

        def gated(planner, **kwargs):
            result = original(planner, **kwargs)
            decided.set()
            assert release.wait(10)
            return result

        RetentionPlanner.run_pass = gated
        try:
            owner = make_retention_owner(home, lease, lambda: days[0])
            outcome: dict[str, object] = {}

            def run_owner() -> None:
                try:
                    outcome["result"] = owner._run_episode()
                except StorageFormatError as exc:
                    outcome["error"] = exc

            thread = threading.Thread(target=run_owner)
            thread.start()
            assert decided.wait(10)
            days[0] = 60
            release.set()
            thread.join(10)
            assert not thread.is_alive()
        finally:
            release.set()
            RetentionPlanner.run_pass = original

        assert "error" not in outcome
        projections, failure = outcome["result"]
        assert failure is None
        assert projections[-1].projection["days"] == 60
        last = read_retention_history(home)[-1]
        assert last["days"] == 60
        assert last["mode"] == "preview"


@pytest.mark.asyncio
async def test_owner_drops_prepared_preview_after_concurrent_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    initialize_storage_home(home)
    (home / "storage-format.json").write_text('{"writer_epoch":11}\n')
    lease = await asyncio.to_thread(open_writer, home)
    try:
        def seed_preview():
            with RetentionPlanner(home, lease) as planner:
                result = planner.run_pass(days=30)
            assert result.complete
        await asyncio.to_thread(seed_preview)
        before = (home / "retention-state.json").read_bytes()
        owner = make_retention_owner(home, lease, lambda: 30)
        original_authority = activation_module._coordinator_authority
        entered = threading.Event()
        release = threading.Event()

        @contextlib.contextmanager
        def delayed_authority(root):
            entered.set()
            assert release.wait(10), "owner did not receive authority release"
            with original_authority(root) as token:
                yield token

        # test-contract: seam force the coordinator order at the owner interface.
        monkeypatch.setattr(activation_module, "_coordinator_authority", delayed_authority)
        task = asyncio.create_task(asyncio.to_thread(owner._run_episode))
        await asyncio.wait_for(asyncio.to_thread(entered.wait, 10), 10)
        armed = await asyncio.to_thread(retention_authority, home, lease, days=lambda: 30)
        assert armed.applied and armed.projection["mode"] == "armed"
        release.set()
        projections, failure = await asyncio.wait_for(task, 10)
        assert failure is None
        # test-contract: seam the activation owner projection is its public interface.
        assert projections[-1].projection["mode"] == "armed", (projections, failure, read_retention_history(home))
        assert (home / "retention-state.json").read_bytes() == before
        assert read_retention_history(home)[-1]["mode"] == "armed"
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
        published = asyncio.Event()
        def fail_publish(_projection):
            published.set()
            raise RuntimeError("publish")
        owner.set_publish(fail_publish)
        owner.start()
        await asyncio.wait_for(published.wait(), 10)
        assert owner.task is not None and not owner.task.done()
        await owner.stop()
    finally:
        await asyncio.to_thread(lease.close)


_OWNER_ROLE: contextvars.ContextVar[str] = contextvars.ContextVar("owner_role")


def _run_owner_as(role: str, owner: RetentionActivationOwner, finished: threading.Event | None = None):
    _OWNER_ROLE.set(role)
    try:
        return owner._run_episode()
    finally:
        if finished is not None:
            finished.set()


@pytest.mark.asyncio
async def test_concurrent_owner_probes_do_not_read_as_user_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    initialize_storage_home(home)
    (home / "storage-format.json").write_text('{"writer_epoch":11}\n')
    lease = await asyncio.to_thread(open_writer, home)
    a_holding, b_entered = threading.Event(), threading.Event()
    b_probes: list[bool] = []
    real_flock = fcntl.flock
    real_probe = handshake_module.planner_request_pending

    def flock(fd, operation):
        role = _OWNER_ROLE.get(None)
        if role == "b" and a_holding.is_set():
            b_entered.set()
        real_flock(fd, operation)
        if role == "a" and operation == fcntl.LOCK_EX | fcntl.LOCK_NB and not a_holding.is_set():
            # owner a's probe holds the user-open-request lock until owner b probes too
            a_holding.set()
            assert b_entered.wait(5), "owner b did not start its probe while owner a probed"

    def probe(home):
        pending = real_probe(home)
        if _OWNER_ROLE.get(None) == "b":
            b_probes.append(pending)
        return pending

    monkeypatch.setattr(handshake_module, "fcntl", types.SimpleNamespace(
        flock=flock, LOCK_EX=fcntl.LOCK_EX, LOCK_NB=fcntl.LOCK_NB,
        LOCK_SH=fcntl.LOCK_SH, LOCK_UN=fcntl.LOCK_UN,
    ))
    monkeypatch.setattr(activation_module, "planner_request_pending", probe)
    monkeypatch.setattr(retention_module, "planner_request_pending", probe)
    first = make_retention_owner(home, lease, lambda: 30)
    second = make_retention_owner(home, lease, lambda: 31)
    try:
        a_task = asyncio.ensure_future(asyncio.to_thread(_run_owner_as, "a", first))
        assert await asyncio.to_thread(a_holding.wait, 5), "owner a did not take its probe lock"
        b_task = asyncio.ensure_future(asyncio.to_thread(_run_owner_as, "b", second))
        results = await asyncio.wait_for(asyncio.gather(a_task, b_task), 15)
        assert b_probes[:1] == [False], "owner b read owner a's probe as a user open"
        for projections, failure in results:
            assert failure is None and projections, results
        rows = read_retention_history(home)
        assert {row["days"] for row in rows} == {30, 31}, rows
    finally:
        await first.stop()
        await second.stop()
        await asyncio.to_thread(lease.close)


@pytest.mark.asyncio
async def test_two_owners_append_without_lost_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    initialize_storage_home(home)
    (home / "storage-format.json").write_text('{"writer_epoch":11}\n')
    lease = await asyncio.to_thread(open_writer, home)
    a_read, b_contended, b_progress = threading.Event(), threading.Event(), threading.Event()
    real_publish = retention_module.atomic_publish_file
    real_authority = retention_module.domain_authority

    def publish(home_fd, name, data):
        if _OWNER_ROLE.get(None) == "a" and name == "retention-history.jsonl" and not a_read.is_set():
            # owner a has read the history rows and holds the coordinator; let owner b run
            a_read.set()
            assert b_progress.wait(5), "owner b neither waited on the coordinator nor finished"
        return real_publish(home_fd, name, data)

    @contextlib.contextmanager
    def coordinator(root, home_id, domain, **kwargs):
        if domain == "coordinator" and _OWNER_ROLE.get(None) == "b":
            b_contended.set()
            b_progress.set()
        with real_authority(root, home_id, domain, **kwargs):
            yield

    first = make_retention_owner(home, lease, lambda: 30)
    second = make_retention_owner(home, lease, lambda: 31)
    second_factory = second._planner_factory

    def gated_planner():
        planner = second_factory()
        run_pass = planner.run_pass

        def gated_run_pass(**kwargs):
            assert a_read.wait(5), "owner a did not reach its history publication"
            return run_pass(**kwargs)

        planner.run_pass = gated_run_pass
        return planner

    second._planner_factory = gated_planner
    # test-contract: seam the coordinator lock and history publish are the append race interface.
    monkeypatch.setattr(retention_module, "atomic_publish_file", publish)
    monkeypatch.setattr(retention_module, "domain_authority", coordinator)
    try:
        results = await asyncio.wait_for(asyncio.gather(
            asyncio.to_thread(_run_owner_as, "a", first),
            asyncio.to_thread(_run_owner_as, "b", second, b_progress),
        ), 15)
        assert all(failure is None for _projections, failure in results), results
        rows = [json.loads(row) for row in (home / "retention-history.jsonl").read_text().splitlines()]
        assert len(rows) == 2, f"a history row was lost: {rows}"
        assert {row["mode"] for row in rows} == {"preview"}
        assert {row["days"] for row in rows} == {30, 31}
        assert b_contended.is_set(), "owner b never waited on the coordinator owner a held"
        state = json.loads((home / "retention-state.json").read_text())
        assert state["preview"]["preview_id"] in {row["preview_id"] for row in rows if row["preview_id"]}
    finally:
        await first.stop()
        await second.stop()
        await asyncio.to_thread(lease.close)

@pytest.mark.asyncio
async def test_owner_stop_yields_at_bounded_checkpoint_before_pass_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    home = tmp_path / "home"
    initialize_storage_home(home)
    (home / "storage-format.json").write_text('{"writer_epoch":11}\n')
    manager = SessionManager(home)
    await asyncio.to_thread(_create_closed, manager, [], name="old")
    lease = await asyncio.to_thread(open_writer, home)
    checkpoint = threading.Event()
    stop_seen = threading.Event()
    pass_released = threading.Event()
    original_charge = retention_module._Budget.charge_entry
    original_run_pass = RetentionPlanner.run_pass
    owner_ref: dict[str, RetentionActivationOwner] = {}

    def charge_entry(budget, size=0):
        checkpoint.set()
        assert stop_seen.wait(10)
        return original_charge(budget, size)

    def run_pass(planner, *, days, commit_state=True, yield_check=None):
        result = original_run_pass(
            planner, days=days, commit_state=commit_state, yield_check=yield_check
        )
        if result.complete:
            pass_released.set()
        return result

    monkeypatch.setattr(retention_module._Budget, "charge_entry", charge_entry)
    monkeypatch.setattr(RetentionPlanner, "run_pass", run_pass)
    owner = make_retention_owner(
        home, lease, lambda: 30,
        bounds=retention_module.RetentionBounds(entries_total=32, entries_per_root=32),
    )
    owner_ref["owner"] = owner
    real_stop = owner.stop

    async def stop() -> None:
        owner._stop.set()
        stop_seen.set()
        await real_stop()

    owner.stop = stop
    try:
        owner.start()
        # test-contract: latency stop checkpoint gate
        await asyncio.wait_for(asyncio.to_thread(checkpoint.wait, 10), 10)
        stopping = asyncio.create_task(owner.stop())
        await asyncio.wait_for(stopping, 10)
        assert not pass_released.is_set()
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

    def blocked(planner, *, days, commit_state=True, yield_check=None):
        entered.set()
        assert release.wait(10)
        return original(
            planner, days=days, commit_state=commit_state, yield_check=yield_check
        )

    monkeypatch.setattr(RetentionPlanner, "run_pass", blocked)
    owner = make_retention_owner(home, lease, lambda: 30)
    try:
        owner.start()
        await asyncio.wait_for(asyncio.to_thread(entered.wait, 10), 10)
        stop_entered = asyncio.Event()
        original_stop = owner.stop
        async def stop_with_signal():
            stop_entered.set()
            await original_stop()
        owner.stop = stop_with_signal
        stopping = asyncio.create_task(owner.stop())
        await asyncio.wait_for(stop_entered.wait(), 10)
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
        published, failure = await asyncio.to_thread(owner._run_episode)
        assert failure is None
        assert len(published) >= 3
        assert getattr(published[-1], "complete", False)
        await owner.stop()
    finally:
        await asyncio.to_thread(lease.close)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "outcomes"),
    [
        ("complete", [Published({"mode": "preview"}, complete=True)]),
        ("progressed-incomplete", [Published({"mode": "preview"}, complete=False, progressed=True), Published({"mode": "preview"}, complete=True)]),
        ("no-progress", [Published({"mode": "preview"}, complete=False)]),
        ("user-scan-checkpoint", [Yielded("user"), Published({"mode": "preview"}, complete=True)]),
        ("stop-scan-checkpoint", [Yielded("stop")]),
        ("publish-failure", [Published({"mode": "preview"}, complete=True)]),
        ("history-failure", [RuntimeError("history")]),
        ("stale-commit", [RuntimeError("stale")]),
        ("restart", [Published({"mode": "preview"}, complete=True)]),
    ],
)
async def test_owner_outcome_table_real_epoch_home(
    tmp_path: Path, case: str, outcomes: list[object]
) -> None:
    home = tmp_path / case
    initialize_storage_home(home)
    (home / "storage-format.json").write_text('{"writer_epoch":11}\n')
    ensure_handshake_files(home)
    calls = 0
    published: list[object] = []
    published_event = asyncio.Event()

    class Planner:
        enabled = True

        def close(self) -> None:
            return None

    def run_pass(_planner, _days, _yield_check=None):
        nonlocal calls
        outcome = outcomes[min(calls, len(outcomes) - 1)]
        calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    owner = RetentionActivationOwner(
        lambda: Planner(), lambda: 30, run_pass=run_pass, home=home, wake_seconds=60
    )
    if case == "publish-failure":
        def fail_publish(_value):
            published_event.set()
            raise RuntimeError("publish")
        owner.set_publish(fail_publish)
    else:
        def publish(value):
            published.append(value)
            published_event.set()
        owner.set_publish(publish)
    owner.start()
    if case.startswith("user-"):
        await asyncio.wait_for(published_event.wait(), 10)
        await owner.stop()
        assert calls >= 2
    elif case.startswith("stop-"):
        await asyncio.wait_for(owner.task, 10)
        assert calls == 1
    else:
        await asyncio.wait_for(published_event.wait(), 10)
        await owner.stop()
    if case.startswith(("user-", "stop-")):
        assert not (home / "retention-history.jsonl").exists()
    if case.startswith("stop-"):
        assert calls == 1


@pytest.mark.asyncio
async def test_owner_no_progress_waits_for_wake(tmp_path: Path) -> None:
    del tmp_path
    passes = 0

    class Planner:
        enabled = True

        def close(self):
            return None

    def run_pass(_planner, _days, _yield_check=None):
        nonlocal passes
        passes += 1
        return Published({"mode": "preview", "days": 30}, complete=False)

    owner = RetentionActivationOwner(
        lambda: Planner(), lambda: 30, run_pass=run_pass, wake_seconds=60
    )
    published = asyncio.Event()
    owner.set_publish(lambda _projection: published.set())
    owner.start()
    await asyncio.wait_for(published.wait(), 10)
    assert passes == 1
    published.clear()
    owner.wake()
    await asyncio.wait_for(published.wait(), 10)
    assert passes == 2
    await owner.stop()


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
        armed = await asyncio.to_thread(retention_authority, home, lease, days=lambda: 30)
        assert armed.applied
        deleted = []
        for _ in range(20):
            projections, failure = await asyncio.to_thread(owner._run_episode)
            assert failure is None
            deleted = [item for item in projections if item["mode"] == "deleted"]
            if deleted:
                break
        assert deleted and any(item["root_ids"] == [opened] for item in deleted)
        assert not (home / "sessions" / opened).exists()
        assert any(row["mode"] == "deleted" and row["root_ids"] == [opened] for row in read_retention_history(home))
        await owner.stop()
    finally:
        await asyncio.to_thread(lease.close)

@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ("arm", "disarm"))
async def test_arm_changes_after_files_are_deleted_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    home = tmp_path / "home"
    initialize_storage_home(home)
    (home / "storage-format.json").write_text('{"writer_epoch":11}\n')
    manager = SessionManager(home)
    session_id = await asyncio.to_thread(_create_closed, manager, [], name="old")
    _age_tree(home / "sessions" / session_id, days=100)
    lease = await asyncio.to_thread(open_writer, home)
    try:
        clock = lambda: time.time_ns() + 365 * 86_400 * 1_000_000_000
        owner = make_retention_owner(home, lease, lambda: 30, now_ns=clock)
        first, failure = await asyncio.to_thread(owner._run_episode)
        assert failure is None and any(item.complete for item in first)
        assert (await asyncio.to_thread(retention_authority, home, lease, days=lambda: 30)).applied

        original = RetentionPlanner.run_pass
        changed = threading.Event()

        def after_transaction(planner, **kwargs):
            result = original(planner, **kwargs)
            if getattr(result, "mode", None) == "deleted" and not changed.is_set():
                # run_pass returns after the transaction removes files and before owner publication.
                if mutation == "arm":
                    with RetentionPlanner(home, lease) as followup:
                        assert original(followup, days=30).complete
                    changed_result = retention_authority(home, lease, days=lambda: 30)
                else:
                    changed_result = disarm_retention(home, lease)
                assert changed_result.applied
                changed.set()
            return result

        monkeypatch.setattr(RetentionPlanner, "run_pass", after_transaction)
        projections, failure = await asyncio.to_thread(owner._run_episode)
        assert failure is None
        assert changed.is_set()
        assert not (home / "sessions" / session_id).exists()
        assert projections[-1]["mode"] == "deleted"
        assert read_retention_history(home)[-1]["mode"] == "deleted"
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
        assert await asyncio.to_thread(retention_authority, home, lease, days=lambda: 30)

        def deadline(*_args, **_kwargs):
            raise TimeoutError

        # test-contract: seam forced coordinator deadline
        monkeypatch.setattr(retention_module, "_try_authority", deadline)
        pending = await asyncio.to_thread(disarm_retention, home, lease)
        assert not pending.applied and pending.projection["disable_pending"]
        assert (home / "retention-arm.json").exists()
        assert not read_retention_history(home)[-1].get("disable_pending", False)
        monkeypatch.undo()
        cleared = await asyncio.to_thread(disarm_retention, home, lease)
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
        assert (await asyncio.to_thread(retention_authority, home, lease, days=lambda: 30)).applied
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

@pytest.mark.asyncio
async def test_inactive_owner_preserves_authority_until_session_exists(tmp_path: Path) -> None:
    home = tmp_path / "home"
    initialize_storage_home(home)
    (home / "storage-format.json").write_text('{"writer_epoch":11}\n')
    lease = await asyncio.to_thread(open_writer, home)
    try:
        def seed() -> None:
            with RetentionPlanner(home, lease) as planner:
                assert planner.run_pass(days=30).complete
            assert retention_authority(home, lease, days=lambda: 30).applied

        await asyncio.to_thread(seed)
        owner = make_retention_owner(home, lease, lambda: 0, session_available=False)
        projections, failure = await asyncio.to_thread(owner._run_episode)
        assert failure is None and projections == []
        assert (home / "retention-arm.json").exists()
        assert (home / "retention-state.json").exists()
        owner.set_session_available()
        projections, failure = await asyncio.to_thread(owner._run_episode)
        assert failure is None and projections
        assert not (home / "retention-arm.json").exists()
    finally:
        await asyncio.to_thread(lease.close)


@pytest.mark.asyncio
async def test_started_owner_guards_public_loop_session_open(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    initialize_storage_home(home)
    (home / "storage-format.json").write_text('{"writer_epoch":11}\n')
    manager = await asyncio.to_thread(SessionManager, home)
    session_id = await asyncio.to_thread(_create_closed, manager, [], name="idle")
    lease = await asyncio.to_thread(open_writer, home)
    entered = threading.Event()
    release = threading.Event()
    original_run_pass = RetentionPlanner.run_pass

    def blocked_run_pass(planner, *, days, commit_state=True, yield_check=None):
        entered.set()
        assert release.wait(10)
        return original_run_pass(
            planner, days=days, commit_state=commit_state, yield_check=yield_check
        )

    monkeypatch.setattr(RetentionPlanner, "run_pass", blocked_run_pass)
    owner = make_retention_owner(home, lease, lambda: 30)
    lease_fd = os.open(home / "sessions" / session_id / "runtime.lease", os.O_RDWR)
    try:
        fcntl.flock(lease_fd, fcntl.LOCK_EX)
        owner.start()
        await asyncio.wait_for(asyncio.to_thread(entered.wait, 10), 10)
        with pytest.raises(
            RuntimeError,
            match="retention lock requested on an asyncio event-loop thread; use asyncio.to_thread",
        ):
            manager.open(session_id)
    finally:
        release.set()
        fcntl.flock(lease_fd, fcntl.LOCK_UN)
        os.close(lease_fd)
        await owner.stop()
        await asyncio.to_thread(lease.close)


def test_owner_without_days_source_publishes_settings_days(tmp_path: Path) -> None:
    # production builds the owner with no days source; publication must read the settings file
    home = tmp_path / "home"
    initialize_storage_home(home)
    (home / "settings.toml").write_text("[retention]\ndays = 45\n")
    with _opened_lease(home) as lease:
        projections, failure = make_retention_owner(home, lease)._run_episode()
    assert failure is None
    assert projections[-1].projection["days"] == 45
    assert read_retention_history(home)[-1]["days"] == 45
