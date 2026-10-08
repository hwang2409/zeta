from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from zeta.memory.auto import AutoMemoryConfig, AutoMemoryReconciler
from zeta.memory.reconciler import ReconciliationResponse
from zeta.memory.user_authorization import MemoryMutationAuthorization
from zeta.project_memory_commands import run_memory_command
from zeta.project_registry import ProjectRegistry

SESSION = "a" * 32


def _write_transcript(session_dir: Path, count: int = 4) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "conversation.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "seq": seq,
                    "id": f"m{seq}",
                    "parent_id": None if seq == 1 else f"m{seq - 1}",
                    "type": "message",
                    "data": {"message": {"role": "user", "content": [{"type": "text", "text": f"fact {seq}"}]}},
                }
            )
            + "\n"
            for seq in range(1, count + 1)
        ),
        encoding="utf-8",
    )


def _proposal(prompt: str) -> str:
    marker = 'Completed transcript rows:'
    assert marker in prompt
    rows = json.loads(prompt.split(marker, 1)[1].strip())
    start, end = rows[0]["seq"], rows[-1]["seq"]
    session_id = prompt.split("from session\n", 1)[1].split(".", 1)[0].strip()
    return json.dumps(
        {
            "changes": [
                {
                    "file": "decisions.md",
                    "content": f"# Decisions\n\nrange {start}-{end}\n",
                    "sources": [
                        {"session_id": session_id, "seq_start": start, "seq_end": end}
                    ],
                }
            ]
        }
    )


class _IdleClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.waiting = asyncio.Event()
        self.release = asyncio.Event()

    def __call__(self) -> float:
        return self.now

    async def wait(self, wake: asyncio.Event, _timeout: float) -> None:
        self.waiting.set()
        await self.release.wait()
        self.release.clear()
        if wake.is_set():
            return
        raise TimeoutError

    async def advance(self, seconds: float) -> None:
        self.now += seconds
        self.release.set()
        await asyncio.sleep(0)


async def _wait_until(event: asyncio.Event) -> None:
    while not event.is_set():
        await asyncio.sleep(0)
    event.clear()


def _runner(tmp_path: Path, invoke=_proposal, *, transcript_count: int = 4, **config: object):
    home = tmp_path / ".zeta"
    workspace = tmp_path / "repo"
    workspace.mkdir()
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("demo", "scope", workspace)
    registry.initialize_memory(project.project_id)
    session_dir = home / "sessions" / SESSION
    _write_transcript(session_dir, transcript_count)
    notices: list[str] = []
    clock = config.pop("clock", None)
    idle_wait = config.pop("idle_wait", None)
    retry_clock = config.pop("retry_clock", None)
    runner = AutoMemoryReconciler(
        registry=registry,
        project_id=project.project_id,
        session_id=SESSION,
        session_dir=session_dir,
        invoke=invoke,
        config=AutoMemoryConfig(**config),
        notice=notices.append,
        **({"clock": clock} if clock is not None else {}),
        **({"idle_wait": idle_wait} if idle_wait is not None else {}),
        **({"retry_clock": retry_clock} if retry_clock is not None else {}),
    )
    return runner, registry, project.project_id, notices


@pytest.mark.asyncio
async def test_before_eviction_reconciles_exact_range(tmp_path: Path) -> None:
    seen: list[str] = []

    async def invoke(prompt: str) -> str:
        seen.append(prompt)
        return _proposal(prompt)

    runner, registry, project_id, notices = _runner(tmp_path, invoke)
    runner.before_eviction(2, 3)
    await runner.drain()

    assert '"seq": 1' not in seen[0]
    assert '"seq": 2' in seen[0] and '"seq": 3' in seen[0]
    assert '"seq": 4' not in seen[0]
    assert dict(registry.load_memory(project_id))["decisions.md"].endswith("range 2-3\n")
    assert notices == ["memory updated: decisions.md (+1)"]


@pytest.mark.parametrize("text", ["界" * 1_200, "🧪" * 1_200])
@pytest.mark.asyncio
async def test_token_growth_uses_text_estimate_not_utf8_bytes(
    tmp_path: Path, text: str
) -> None:
    runner, registry, project_id, _ = _runner(
        tmp_path, token_threshold=500, transcript_count=1
    )
    row = {"seq": 1, "type": "message", "data": {"text": text}}
    (runner.session_dir / "conversation.jsonl").write_text(
        json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    runner.activity(1)
    await runner.drain()

    assert not registry.memory_log(project_id)


@pytest.mark.asyncio
async def test_token_growth_position_is_durable_and_ignores_provider_usage(
    tmp_path: Path,
) -> None:
    runner, registry, project_id, _ = _runner(
        tmp_path, token_threshold=50, transcript_count=1
    )
    runner.observe_tokens(1_000_000)
    await runner.drain()
    assert not registry.memory_log(project_id)

    _write_transcript(runner.session_dir, 4)
    runner.activity(4)
    await runner.drain()

    position = json.loads(runner.position_path.read_text(encoding="utf-8"))
    assert position["transcript_tokens"] > 0
    replacement = AutoMemoryReconciler(
        registry=registry,
        project_id=project_id,
        session_id=SESSION,
        session_dir=runner.session_dir,
        invoke=_proposal,
        config=runner.config,
    )
    assert replacement._last_reconciled_tokens == position["transcript_tokens"]
    assert len(registry.memory_log(project_id)) == 1


@pytest.mark.asyncio
async def test_idle_trigger(tmp_path: Path) -> None:
    idle_clock = _IdleClock()
    runner, registry, project_id, _ = _runner(
        tmp_path,
        idle_seconds=10,
        minimum_interval=0,
        clock=idle_clock,
        idle_wait=idle_clock.wait,
    )
    runner.activity(1)
    await _wait_until(idle_clock.waiting)
    runner.activity(4)
    await idle_clock.advance(10)
    await _wait_until(idle_clock.waiting)
    assert not registry.memory_log(project_id)
    await idle_clock.advance(10)
    for _ in range(100):
        await asyncio.sleep(0)
        if registry.memory_log(project_id):
            break
    await runner.drain()
    assert len(registry.memory_log(project_id)) == 1

    # Idle is re-armed by later durable activity after a completed reconciliation.
    _write_transcript(runner.session_dir, 5)
    runner.activity(5)
    await _wait_until(idle_clock.waiting)
    await idle_clock.advance(10)
    await _wait_until(idle_clock.waiting)
    await idle_clock.advance(10)
    for _ in range(100):
        await asyncio.sleep(0)
        if len(registry.memory_log(project_id)) == 2:
            break
    await runner.drain()
    assert len(registry.memory_log(project_id)) == 2


@pytest.mark.asyncio
async def test_crash_resume_has_no_duplicate_history(tmp_path: Path) -> None:
    runner, registry, project_id, _ = _runner(tmp_path)
    runner.before_eviction(1, 2)
    await runner.drain()
    # Simulate a crash after apply but before the position publication.
    runner.position_path.unlink()

    replacement = AutoMemoryReconciler(
        registry=registry,
        project_id=project_id,
        session_id=SESSION,
        session_dir=runner.session_dir,
        invoke=_proposal,
        config=runner.config,
    )
    replacement.before_eviction(1, 2)
    await replacement.drain()

    assert len(registry.memory_log(project_id)) == 1
    assert replacement.last_reconciled_seq == 2


@pytest.mark.asyncio
async def test_cas_exhaustion_keeps_pending_range_and_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, registry, project_id, _ = _runner(
        tmp_path, cas_retries=3, minimum_interval=0
    )
    original = registry.compare_and_swap_memory
    calls = 0

    def burst(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        if calls <= 3:
            from zeta.project_registry import ProjectRegistryError

            raise ProjectRegistryError("project memory digest mismatch")
        return original(*args, **kwargs)

    monkeypatch.setattr(registry, "compare_and_swap_memory", burst)
    runner.before_eviction(1, 2)
    await asyncio.wait_for(runner.drain(), timeout=2)

    assert calls == 4
    assert runner.last_reconciled_seq == 2
    assert registry.memory_snapshot(project_id).contents["decisions.md"].endswith(
        "range 1-2\n"
    )


@pytest.mark.asyncio
async def test_close_stops_retrying_persistent_cas_conflicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, registry, _project_id, _ = _runner(
        tmp_path, cas_retries=1, minimum_interval=0
    )
    attempts = 0

    def always_conflicts(*args: object, **kwargs: object):
        nonlocal attempts
        attempts += 1
        from zeta.project_registry import ProjectRegistryError

        raise ProjectRegistryError("project memory digest mismatch")

    monkeypatch.setattr(registry, "compare_and_swap_memory", always_conflicts)
    runner.before_eviction(1, 2)
    while attempts == 0:
        await asyncio.sleep(0)

    await asyncio.wait_for(runner.close(), timeout=0.2)

    assert runner.last_reconciled_seq == 0
    assert not runner.position_path.exists()


@pytest.mark.asyncio
async def test_concurrent_edit_retries_without_clobber(tmp_path: Path) -> None:
    calls = 0
    registry_ref: ProjectRegistry
    project_ref: str

    async def invoke(prompt: str) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            registry_ref.update_memory(project_ref, {"brief.md": "# Brief\n\nHuman edit.\n"})
        return _proposal(prompt)

    runner, registry_ref, project_ref, _ = _runner(tmp_path, invoke)
    runner.before_eviction(1, 2)
    await runner.drain()

    memory = dict(registry_ref.load_memory(project_ref))
    assert calls == 2
    assert memory["brief.md"].endswith("Human edit.\n")
    assert memory["decisions.md"].endswith("range 1-2\n")


@pytest.mark.asyncio
async def test_background_reconcile_does_not_block_event_loop(tmp_path: Path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def invoke(prompt: str) -> str:
        started.set()
        await release.wait()
        return _proposal(prompt)

    runner, _, _, _ = _runner(tmp_path, invoke)
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        for _ in range(5):
            await asyncio.sleep(0)
            ticks += 1

    runner.before_eviction(1, 2)
    await started.wait()
    await ticker()
    assert ticks == 5
    release.set()
    await runner.drain()


@pytest.mark.asyncio
async def test_close_is_deterministic_before_worker_first_runs(tmp_path: Path) -> None:
    runner, _, _, _ = _runner(tmp_path)
    runner.activity(1)

    await asyncio.wait_for(runner.close(), timeout=1)


@pytest.mark.asyncio
async def test_disabled_setting_never_invokes(tmp_path: Path) -> None:
    calls = 0

    async def invoke(prompt: str) -> str:
        nonlocal calls
        calls += 1
        return _proposal(prompt)

    runner, _, _, _ = _runner(tmp_path, invoke, enabled=False, idle_seconds=0.01)
    runner.activity(4)
    runner.before_eviction(1, 3)
    await runner.drain()
    assert calls == 0


@pytest.mark.asyncio
async def test_oversized_row_is_bounded_once_and_preserves_tail_evidence(
    tmp_path: Path,
) -> None:
    fact = "durable-tail-fact-7Q"
    prompts: list[str] = []

    async def invoke(prompt: str) -> str:
        prompts.append(prompt)
        return _proposal(prompt)

    runner, registry, project_id, _ = _runner(
        tmp_path, invoke, transcript_count=1, minimum_interval=0
    )
    row = {
        "seq": 1,
        "type": "notification",
        "data": {"text": "x" * (70 * 1024) + fact},
    }
    (runner.session_dir / "conversation.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )

    runner.before_eviction(1, 1)
    await runner.drain()

    assert len(prompts) == 1
    assert len(prompts[0].encode()) <= 64 * 1024
    assert fact in prompts[0]
    assert "content omitted for automatic memory request size" in prompts[0]
    assert runner.last_reconciled_seq == 1
    assert registry.memory_snapshot(project_id).contents["decisions.md"].endswith(
        "range 1-1\n"
    )


@pytest.mark.asyncio
async def test_oversized_user_row_gets_terminal_receipt_without_publication(
    tmp_path: Path,
) -> None:
    prompts: list[str] = []

    async def invoke(prompt: str) -> str:
        prompts.append(prompt)
        return _proposal(prompt)

    runner, registry, project_id, _ = _runner(
        tmp_path, invoke, transcript_count=2, minimum_interval=0
    )
    oversized = {
        "seq": 1,
        "type": "message",
        "data": {
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "x" * 100_000}],
                "metadata": {"zeta.origin": "user"},
            }
        },
    }
    later = {
        "seq": 2,
        "type": "message",
        "data": {
            "message": {
                "role": "user",
                "content": "later fact",
                "metadata": {"zeta.origin": "user"},
            }
        },
    }
    (runner.session_dir / "conversation.jsonl").write_text(
        json.dumps(oversized) + "\n" + json.dumps(later) + "\n",
        encoding="utf-8",
    )

    runner.before_eviction(1, 2)
    await runner.drain()

    assert len(prompts) == 1
    assert '"seq": 1' not in prompts[0]
    assert registry.memory_log(project_id)[-1]["provenance"]["seq_start"] == 2
    receipt = runner.terminal_receipts()[0]
    assert (receipt.seq_start, receipt.seq_end) == (1, 1)
    assert receipt.validation_summary == "user row exceeds the request limit"
    assert runner.last_failure is not None and runner.last_failure.terminal
    assert runner.last_reconciled_seq == 2


@pytest.mark.asyncio
async def test_fitting_user_row_preserves_middle_fact(tmp_path: Path) -> None:
    fact = "middle-user-fact-9X"
    prompts: list[str] = []

    async def invoke(prompt: str) -> str:
        prompts.append(prompt)
        return '{"changes":[]}'

    runner, _, _, _ = _runner(
        tmp_path, invoke, transcript_count=1, minimum_interval=0
    )
    row = {
        "seq": 1,
        "type": "message",
        "data": {
            "message": {
                "role": "user",
                "content": [
                    {"type": "text", "text": "x" * 30_000 + fact + "y" * 30_000}
                ],
            }
        },
    }
    (runner.session_dir / "conversation.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )

    runner.before_eviction(1, 1)
    await runner.drain()

    assert len(prompts) == 1
    assert len(prompts[0].encode()) <= 64 * 1024
    assert fact in prompts[0]
    assert "content omitted for automatic memory request size" not in prompts[0]
    assert runner.last_reconciled_seq == 1


@pytest.mark.asyncio
async def test_worker_coalesces_activity_while_provider_is_running(tmp_path: Path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    prompts: list[str] = []

    async def invoke(prompt: str) -> str:
        prompts.append(prompt)
        if len(prompts) == 1:
            started.set()
            await release.wait()
        return _proposal(prompt)

    runner, _, _, _ = _runner(
        tmp_path, invoke, transcript_count=2, token_threshold=1, minimum_interval=0
    )
    runner.activity(2)
    await started.wait()
    _write_transcript(runner.session_dir, 8)
    runner.activity(5)
    runner.activity(8)
    release.set()
    await runner.drain()

    assert len(prompts) == 2
    assert '"seq": 8' not in prompts[0]
    assert '"seq": 8' in prompts[1]
    assert runner.last_reconciled_seq == 8


@pytest.mark.asyncio
async def test_notice_uses_explicit_publish_result_after_100_records(
    tmp_path: Path,
) -> None:
    runner, registry, project_id, notices = _runner(tmp_path)
    for seq in range(1, 101):
        snapshot = registry.memory_snapshot(project_id)
        result = registry.compare_and_swap_memory(
            project_id,
            expected_digest=snapshot.digest,
            updates={"decisions.md": f"# Decisions\n\nold {seq}\n"},
            provenance={
                "session_id": "history",
                "seq_start": seq,
                "seq_end": seq,
            },
        )
        assert result.published is True
        assert result.version

    runner.before_eviction(1, 2)
    await runner.drain()

    assert len(registry.memory_log(project_id)) == 100
    assert notices == ["memory updated: decisions.md (+1)"]


def test_memory_export_import_is_logical_and_preserves_provenance(tmp_path: Path) -> None:
    (tmp_path / "source").mkdir()
    _, source, source_id, _ = _runner(tmp_path / "source")
    source_snapshot = source.memory_snapshot(source_id)
    source.compare_and_swap_memory(
        source_id,
        expected_digest=source_snapshot.digest,
        updates={"decisions.md": "# Decisions\n\nremote decision\n"},
        provenance={
            "session_id": "remote-session",
            "seq_start": 7,
            "seq_end": 9,
            "model": "test-model",
            "usage": {"input_tokens": 12},
        },
    )
    exported = source.export_memory(source_id)

    destination_root = tmp_path / "destination"
    destination_workspace = destination_root / "repo"
    destination_workspace.mkdir(parents=True)
    destination = ProjectRegistry(destination_root / ".zeta" / "projects")
    project = destination.create_project("demo", "scope", destination_workspace)
    destination.initialize_memory(project.project_id)
    destination.update_memory(
        project.project_id, {"brief.md": "# Brief\n\nlocal conflict history\n"}
    )
    before = destination.memory_snapshot(project.project_id)

    result = destination.import_memory(
        project.project_id,
        exported,
        expected_digest=before.digest,
    )

    assert result.published is True
    assert dict(result.contents) == exported.contents
    round_trip = destination.export_memory(project.project_id)
    assert round_trip.contents == exported.contents
    assert any(
        record.get("provenance", {}).get("session_id") == "remote-session"
        for record in round_trip.versions
    )
    assert any(record["kind"] == "manual" for record in round_trip.versions)
    assert any(record["kind"] == "import" for record in round_trip.versions)


def test_memory_snapshot_digest_covers_more_than_load_cap(tmp_path: Path) -> None:
    _, registry, project_id, _ = _runner(tmp_path)
    large = "# Brief\n\n" + "x" * (70 * 1024)
    registry.update_memory(project_id, {"brief.md": large})

    snapshot = registry.memory_snapshot(project_id)

    assert snapshot.contents["brief.md"] == large
    assert snapshot.digest == registry.memory_digest(project_id)


@pytest.mark.parametrize("crash_step", ["snapshot", "manifest", "publish"])
def test_memory_transaction_recovers_without_mixed_state(
    tmp_path: Path, crash_step: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, registry, project_id, _ = _runner(tmp_path)
    before = registry.memory_snapshot(project_id)

    def crash(step: str) -> None:
        if step == crash_step:
            raise OSError("simulated crash")

    monkeypatch.setattr(registry, "_memory_transaction_step", crash)
    with pytest.raises(OSError, match="simulated crash"):
        registry.compare_and_swap_memory(
            project_id,
            expected_digest=before.digest,
            updates={
                "brief.md": "# Brief\n\nnew brief\n",
                "decisions.md": "# Decisions\n\nnew decision\n",
            },
            provenance={"session_id": SESSION, "seq_start": 1, "seq_end": 2},
        )

    monkeypatch.setattr(registry, "_memory_transaction_step", lambda step: None)
    recovered = registry.memory_snapshot(project_id)
    if crash_step == "publish":
        assert recovered.contents["brief.md"].endswith("new brief\n")
        assert recovered.contents["decisions.md"].endswith("new decision\n")
        assert len(registry.memory_log(project_id)) == 1
        registry.undo_memory(project_id)
        assert registry.memory_snapshot(project_id).contents == before.contents
    else:
        assert recovered.contents == before.contents
        assert registry.memory_log(project_id) == []


def test_memory_publication_syncs_each_layer_before_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta import project_memory_history
    from zeta.core import session_files

    _, registry, project_id, _ = _runner(tmp_path)
    events: list[str] = []
    real_fsync = project_memory_history.os.fsync
    real_replace = session_files.os.replace
    project_dir = registry.root / project_id

    def label(fd: int) -> str:
        inode = project_memory_history.os.fstat(fd).st_ino
        candidates = (
            project_dir,
            project_dir / "memory-versions",
            project_dir / "memory-versions" / "blobs",
            project_dir / "memory-versions" / "versions",
        )
        for candidate in candidates:
            if candidate.exists() and candidate.stat().st_ino == inode:
                return candidate.name
        return "file"

    def recording_fsync(fd: int) -> None:
        events.append(f"fsync:{label(fd)}")
        real_fsync(fd)

    def recording_replace(src: str, dst: str, **kwargs: object) -> None:
        events.append(f"rename:{dst}")
        real_replace(src, dst, **kwargs)

    monkeypatch.setattr(project_memory_history.os, "fsync", recording_fsync)
    monkeypatch.setattr(session_files.os, "replace", recording_replace)
    registry.update_memory(project_id, {"brief.md": "# Brief\n\ndurable\n"})

    blob_rename = next(i for i, item in enumerate(events) if item.startswith("rename:") and len(item) == 71)
    blob_sync = next(i for i, item in enumerate(events[blob_rename:], blob_rename) if item == "fsync:blobs")
    manifest_rename = next(i for i, item in enumerate(events) if item.startswith("rename:") and item.endswith(".json") and item != "rename:memory-current.json")
    versions_sync = next(i for i, item in enumerate(events[manifest_rename:], manifest_rename) if item == "fsync:versions")
    pointer_rename = events.index("rename:memory-current.json")
    project_sync = next(i for i, item in enumerate(events[pointer_rename:], pointer_rename) if item.startswith("fsync:p_"))

    assert blob_rename < blob_sync < manifest_rename < versions_sync
    assert versions_sync < pointer_rename < project_sync


def test_gc_retains_manifest_and_blobs_referenced_by_undo(tmp_path: Path) -> None:
    _, registry, project_id, _ = _runner(tmp_path)
    snapshot = registry.memory_snapshot(project_id)
    registry.compare_and_swap_memory(
        project_id,
        expected_digest=snapshot.digest,
        updates={"decisions.md": "# Decisions\n\ntarget\n"},
        provenance={"session_id": SESSION, "seq_start": 1, "seq_end": 1},
    )
    registry.undo_memory(project_id)
    undo = registry.memory_log(project_id)[-1]
    target_version = undo["target_version"]

    for seq in range(127):
        registry.update_memory(
            project_id, {"decisions.md": f"# Decisions\n\nmanual {seq}\n"}
        )

    project_dir = registry.root / project_id
    manifest_path = (
        project_dir / "memory-versions" / "versions" / f"{target_version}.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_path.exists()
    for digest in {
        *manifest["snapshot"].values(),
        *manifest["before_snapshot"].values(),
    }:
        assert (project_dir / "memory-versions" / "blobs" / digest).exists()

    snapshot = registry.memory_snapshot(project_id)
    registry.compare_and_swap_memory(
        project_id,
        expected_digest=snapshot.digest,
        updates={"decisions.md": "# Decisions\n\nlatest update\n"},
        provenance={"session_id": SESSION, "seq_start": 1, "seq_end": 1},
    )
    registry.undo_memory(project_id)
    assert registry.memory_snapshot(project_id).contents["decisions.md"].endswith(
        "manual 126\n"
    )


def test_memory_history_retains_recent_undo_after_compaction(tmp_path: Path) -> None:
    _, registry, project_id, _ = _runner(tmp_path)
    for seq in range(1, 140):
        snapshot = registry.memory_snapshot(project_id)
        registry.compare_and_swap_memory(
            project_id,
            expected_digest=snapshot.digest,
            updates={"decisions.md": f"# Decisions\n\nversion {seq}\n"},
            provenance={"session_id": SESSION, "seq_start": seq, "seq_end": seq},
        )

    records = registry.memory_log(project_id, limit=10_000)
    assert len(records) == 128
    assert records[-1]["provenance"]["seq_end"] == 139
    registry.undo_memory(project_id)
    assert registry.memory_snapshot(project_id).contents["decisions.md"].endswith(
        "version 138\n"
    )


def test_memory_undo_restores_previous_version(tmp_path: Path) -> None:
    _runner_instance, registry, project_id, _ = _runner(tmp_path)
    original = dict(registry.load_memory(project_id))["decisions.md"]
    registry.compare_and_swap_memory(
        project_id,
        expected_digest=registry.memory_digest(project_id),
        updates={"decisions.md": "# Decisions\n\nChanged.\n"},
        provenance={"session_id": SESSION, "seq_start": 1, "seq_end": 2},
    )

    restored = dict(registry.undo_memory(project_id))
    assert restored["decisions.md"] == original
    assert registry.memory_log(project_id)[-1]["kind"] == "undo"


@pytest.mark.asyncio
async def test_failed_range_waits_for_retry_while_later_range_continues(
    tmp_path: Path,
) -> None:
    calls: list[int] = []

    async def invoke(prompt: str) -> str:
        rows = json.loads(prompt.split("Completed transcript rows:", 1)[1].strip())
        seq = rows[0]["seq"]
        calls.append(seq)
        if seq == 1:
            return json.dumps(
                {
                    "changes": [
                        {
                            "file": "decisions.md",
                            "content": "# Decisions\n\ninvalid source\n",
                            "sources": [
                                {
                                    "session_id": SESSION,
                                    "seq_start": 999,
                                    "seq_end": 999,
                                }
                            ],
                        }
                    ]
                }
            )
        return _proposal(prompt)

    retry_now = [1_000.0]
    runner, registry, project_id, notices = _runner(
        tmp_path,
        invoke,
        transcript_count=2,
        minimum_interval=0,
        retry_backoff_seconds=10,
        retry_clock=lambda: retry_now[0],
    )
    oversized = {
        "seq": 1,
        "type": "notification",
        "data": {"text": "x" * (100 * 1024)},
    }
    second = {
        "seq": 2,
        "type": "message",
        "data": {"text": "later durable fact"},
    }
    (runner.session_dir / "conversation.jsonl").write_text(
        json.dumps(oversized) + "\n" + json.dumps(second) + "\n",
        encoding="utf-8",
    )

    runner.before_eviction(1, 2)
    await runner.drain()

    assert calls == [1, 1, 2]
    assert runner.last_reconciled_seq == 0
    assert registry.memory_snapshot(project_id).contents["decisions.md"].endswith(
        "range 2-2\n"
    )
    position = json.loads(runner.position_path.read_text(encoding="utf-8"))
    assert position["pending_failures"][0]["attempt_count"] == 1
    assert position["pending_failures"][0]["retry_after"] == 1_010
    assert position["completed_ranges"][0]["seq_start"] == 2

    await runner.close()
    retry_now[0] = 1_010
    runner = AutoMemoryReconciler(
        registry=registry,
        project_id=project_id,
        session_id=SESSION,
        session_dir=runner.session_dir,
        invoke=invoke,
        config=AutoMemoryConfig(
            minimum_interval=0, retry_backoff_seconds=10
        ),
        notice=notices.append,
        retry_clock=lambda: retry_now[0],
    )
    runner.catch_up()
    await runner.drain()

    await runner.close()
    retry_now[0] = 1_030
    runner = AutoMemoryReconciler(
        registry=registry,
        project_id=project_id,
        session_id=SESSION,
        session_dir=runner.session_dir,
        invoke=invoke,
        config=AutoMemoryConfig(
            minimum_interval=0, retry_backoff_seconds=10
        ),
        notice=notices.append,
        retry_clock=lambda: retry_now[0],
    )
    runner.catch_up()
    await runner.drain()

    assert calls == [1, 1, 2, 1, 1, 1, 1]
    assert runner.last_reconciled_seq == 2
    assert runner.last_failure is not None
    assert runner.last_failure.terminal is True
    receipts = runner.terminal_receipts()
    assert len(receipts) == 1
    assert receipts[0].attempt_count == 3
    assert notices[0].startswith("memory update failed:")
    assert notices[1] == "memory updated: decisions.md (+1)"
    assert runner.failure_log_path.read_text(encoding="utf-8").count("\n") == 3


@pytest.mark.asyncio
async def test_provider_failure_records_exact_unit_and_prior_usage(tmp_path: Path) -> None:
    calls: list[int] = []

    async def invoke(prompt: str) -> ReconciliationResponse:
        rows = json.loads(prompt.split("Completed transcript rows:", 1)[1].strip())
        seq = rows[0]["seq"]
        calls.append(seq)
        if seq == 1:
            return ReconciliationResponse('{"changes":[]}', {"input_tokens": 3})
        if calls.count(2) == 1:
            return ReconciliationResponse("not json", {"input_tokens": 7})
        raise RuntimeError("provider unavailable")

    runner, _, _, _ = _runner(
        tmp_path,
        invoke,
        transcript_count=2,
        minimum_interval=0,
        retry_backoff_seconds=60,
    )
    rows = [
        {"seq": seq, "type": "notification", "data": {"text": "x" * 70_000}}
        for seq in (1, 2)
    ]
    (runner.session_dir / "conversation.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    runner.before_eviction(1, 2)
    await runner.drain()

    state = json.loads(runner.position_path.read_text(encoding="utf-8"))
    assert calls == [1, 2, 2]
    assert runner.last_reconciled_seq == 1
    assert len(state["pending_failures"]) == 1
    failure = state["pending_failures"][0]
    assert (failure["seq_start"], failure["seq_end"]) == (2, 2)
    assert failure["usage"] == {"input_tokens": 7}


def test_terminal_receipts_move_to_bounded_archive(tmp_path: Path) -> None:
    runner, _, project_id, _ = _runner(tmp_path, transcript_count=1)

    for seq in range(1, 1_001):
        runner.state.record_failure(
            seq_start=seq,
            seq_end=seq,
            validation_summary="terminal test failure",
            reason="test",
            usage={},
            retry_backoff_seconds=0,
            now=1_000,
            occurred_at="2026-10-07T00:00:00+00:00",
            terminal=True,
        )

    state = json.loads(runner.position_path.read_text(encoding="utf-8"))
    receipts = runner.terminal_receipts()
    listing = run_memory_command(runner.registry, project_id, "retry", runner)
    assert len(state["terminal_receipts"]) == 100
    assert state["completed_ranges"] == []
    assert runner.position_path.stat().st_size < 100_000
    assert runner.state.receipts_path.stat().st_size < 4 * 1024 * 1024
    assert len(receipts) == 1_000
    assert receipts[0].key in listing
    assert receipts[-1].key in listing
    assert runner.state.retry_terminal(receipts[0].key, now=1_001)
    state = json.loads(runner.position_path.read_text(encoding="utf-8"))
    assert state["pending_failures"][0]["key"] == receipts[0].key


@pytest.mark.asyncio
async def test_explicit_retry_requeues_terminal_receipt(tmp_path: Path) -> None:
    valid = False

    async def invoke(prompt: str) -> ReconciliationResponse:
        if valid:
            return ReconciliationResponse(_proposal(prompt), {"input_tokens": 1})
        return ReconciliationResponse("not json", {"input_tokens": 2})

    runner, registry, project_id, _ = _runner(
        tmp_path,
        invoke,
        transcript_count=1,
        minimum_interval=0,
        retry_backoff_seconds=0,
    )
    runner.before_eviction(1, 1)
    await runner.drain()
    receipt = runner.terminal_receipts()[0]
    state = json.loads(runner.position_path.read_text(encoding="utf-8"))
    assert state["terminal_receipts"][0]["usage"] == {"input_tokens": 12}

    listing = run_memory_command(registry, project_id, "retry", runner)
    assert receipt.key in listing
    valid = True
    queued = run_memory_command(
        registry,
        project_id,
        f"retry {receipt.key[:12]}",
        runner,
        MemoryMutationAuthorization.direct_slash(),
    )
    assert queued == f"memory retry queued: {receipt.key}"
    await runner.drain()

    assert runner.terminal_receipts() == ()
    state = json.loads(runner.position_path.read_text(encoding="utf-8"))
    assert state["completed_ranges"] == []
    assert registry.memory_snapshot(project_id).contents["decisions.md"].endswith(
        "range 1-1\n"
    )


@pytest.mark.asyncio
async def test_repair_aggregates_usage_into_committed_provenance(tmp_path: Path) -> None:
    calls = 0

    async def invoke(prompt: str) -> ReconciliationResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ReconciliationResponse("not json", {"input_tokens": 3, "output_tokens": 2})
        return ReconciliationResponse(
            _proposal(prompt), {"input_tokens": 5, "output_tokens": 7}
        )

    runner, registry, project_id, _ = _runner(
        tmp_path, invoke, transcript_count=1, minimum_interval=0
    )
    runner.before_eviction(1, 1)
    await runner.drain()

    provenance = registry.memory_log(project_id)[-1]["provenance"]
    assert provenance["usage"] == {"input_tokens": 8, "output_tokens": 9}


@pytest.mark.asyncio
async def test_failure_writes_persistent_file_log_without_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    async def invoke(_prompt: str) -> ReconciliationResponse:
        return ReconciliationResponse(
            "not json", {"input_tokens": 2, "output_tokens": 1}
        )

    runner, _, _, _ = _runner(
        tmp_path,
        invoke,
        transcript_count=1,
        minimum_interval=0,
        retry_backoff_seconds=60,
    )
    runner.before_eviction(1, 1)
    await runner.drain()

    diagnostics = runner.registry.root.parent / "logs" / "memory-reconciliation.jsonl"
    records = diagnostics.read_text(encoding="utf-8").splitlines()
    assert len(records) == 1
    record = json.loads(records[0])
    assert record["validation_summary"] == "reconciler output is not valid JSON"
    assert record["usage"] == {"input_tokens": 4, "output_tokens": 2}
    assert record["attempt_count"] == 1
    assert capsys.readouterr().err == ""


@pytest.mark.asyncio
async def test_close_with_pending_range_makes_no_backend_call(
    tmp_path: Path,
) -> None:
    calls = 0

    async def invoke(_prompt: str) -> str:
        nonlocal calls
        calls += 1
        return '{"changes":[]}'

    runner, _, _, _ = _runner(
        tmp_path,
        invoke,
        minimum_interval=0,
        shutdown_grace_seconds=0.01,
    )
    runner.before_eviction(1, 4)

    await runner.close()
    await asyncio.sleep(0)

    assert calls == 0
    assert runner.last_reconciled_seq == 0


@pytest.mark.asyncio
async def test_close_cancels_inflight_request_quietly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def invoke(_prompt: str) -> str:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    runner, _, _, notices = _runner(
        tmp_path,
        invoke,
        minimum_interval=0,
        shutdown_grace_seconds=0.01,
    )
    runner.before_eviction(1, 4)
    await started.wait()

    await asyncio.wait_for(runner.close(), timeout=0.2)
    await asyncio.sleep(0)

    assert cancelled.is_set()
    assert notices == []
    assert capsys.readouterr().err == ""


@pytest.mark.asyncio
async def test_resume_catches_up_from_durable_cursor(
    tmp_path: Path,
) -> None:
    runner, registry, project_id, _ = _runner(
        tmp_path, minimum_interval=0, shutdown_grace_seconds=0
    )
    runner.before_eviction(1, 4)
    await runner.close()
    assert runner.last_reconciled_seq == 0

    resumed = AutoMemoryReconciler(
        registry=registry,
        project_id=project_id,
        session_id=SESSION,
        session_dir=runner.session_dir,
        invoke=_proposal,
        config=AutoMemoryConfig(minimum_interval=0),
    )
    resumed.catch_up()
    await resumed.drain()

    assert resumed.last_reconciled_seq == 4
    assert registry.memory_snapshot(project_id).contents["decisions.md"].endswith(
        "range 1-4\n"
    )

@pytest.mark.parametrize("failing_retry_seq", [None, 2], ids=["all-succeed", "one-fails"])
@pytest.mark.asyncio
async def test_repartitioned_retry_replaces_original_failure(
    tmp_path: Path, failing_retry_seq: int | None
) -> None:
    retry_now = [1_000.0]
    calls: list[tuple[int, int]] = []

    async def invoke(prompt: str) -> str:
        rows = json.loads(prompt.split("Completed transcript rows:", 1)[1].strip())
        seq_range = (rows[0]["seq"], rows[-1]["seq"])
        calls.append(seq_range)
        if seq_range == (failing_retry_seq, failing_retry_seq):
            raise RuntimeError("retry failed")
        return _proposal(prompt)

    runner, registry, project_id, _ = _runner(
        tmp_path,
        invoke,
        transcript_count=3,
        minimum_interval=0,
        retry_backoff_seconds=10,
        retry_clock=lambda: retry_now[0],
    )
    rows = [
        {
            "seq": seq,
            "type": "message",
            "data": {
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "x" * 20_000}],
                    "metadata": {"zeta.origin": "user"},
                }
            },
        }
        for seq in (1, 2)
    ]
    rows.append({"seq": 3, "type": "message", "data": {"text": "later fact"}})
    (runner.session_dir / "conversation.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    runner.state.record_failure(
        seq_start=1,
        seq_end=2,
        validation_summary="initial failure",
        reason="eviction",
        usage={},
        retry_backoff_seconds=10,
        now=retry_now[0],
        occurred_at="2026-10-08T00:00:00+00:00",
    )
    runner.state.record_success(
        seq_start=3,
        seq_end=3,
        reason="eviction",
        end_offset=0,
        end_tokens=0,
    )
    registry.update_memory(project_id, {"brief.md": "# Brief\n\n" + "m" * 30_000})

    retry_now[0] = 1_010
    await asyncio.wait_for(runner.drain(), timeout=2)

    assert calls == [(1, 1), (2, 2)]
    state = json.loads(runner.position_path.read_text(encoding="utf-8"))
    expected = [] if failing_retry_seq is None else [(2, 2)]
    assert [
        (item["seq_start"], item["seq_end"])
        for item in state["pending_failures"]
    ] == expected
    if failing_retry_seq is not None:
        failure = state["pending_failures"][0]
        assert failure["attempt_count"] == 2
        assert failure["retry_after"] == 1_030
    await runner.close()


@pytest.mark.asyncio
async def test_drain_bounds_attempts_when_retry_key_stays_due(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    async def invoke(prompt: str) -> str:
        nonlocal calls
        calls += 1
        return _proposal(prompt)

    runner, _, _, _ = _runner(
        tmp_path,
        invoke,
        transcript_count=1,
        minimum_interval=0,
        retry_backoff_seconds=0,
        retry_clock=lambda: 1_000,
    )
    runner.state.record_failure(
        seq_start=1,
        seq_end=1,
        validation_summary="initial failure",
        reason="eviction",
        usage={},
        retry_backoff_seconds=0,
        now=1_000,
        occurred_at="2026-10-08T00:00:00+00:00",
    )
    monkeypatch.setattr(runner.state, "record_retry_outcomes", lambda *args, **kwargs: ())

    await asyncio.wait_for(runner.drain(), timeout=2)

    assert calls == 3
    assert len(runner.state.ready_retries(1_000)) == 1
    await runner.close()


def test_transcript_chunk_skips_delivery_rows_but_advances_offset(tmp_path: Path) -> None:
    runner, _registry, _project_id, _ = _runner(tmp_path, transcript_count=1)
    path = runner.session_dir / "conversation.jsonl"
    rows = [
        {
            "seq": 1,
            "type": "client_delivery",
            "data": {
                "delivery_id": "private-id",
                "method": "steer",
                "status": "queued",
                "outcome": {"accepted": True},
            },
        },
        {"seq": 2, "type": "message", "data": {"text": "visible"}},
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    projected, end_offset, _tokens, scanned_end = runner._transcript_chunk(1, 2)

    assert [row["seq"] for row in projected] == [2]
    assert end_offset == path.stat().st_size
    assert scanned_end == 2


@pytest.mark.asyncio
async def test_delivery_only_range_advances_cursor_without_provider(tmp_path: Path) -> None:
    calls = 0

    async def invoke(prompt: str) -> str:
        nonlocal calls
        calls += 1
        raise AssertionError(prompt)

    runner, registry, project_id, _ = _runner(tmp_path, invoke, transcript_count=0)
    path = runner.session_dir / "conversation.jsonl"
    rows = [
        {"seq": 1, "type": "message", "data": {"text": "user turn"}},
        {"seq": 2, "type": "message", "data": {"text": "assistant turn"}},
        {
            "seq": 3,
            "type": "client_delivery",
            "data": {
                "delivery_id": "private-id",
                "method": "steer",
                "status": "queued",
                "outcome": {"accepted": True},
            },
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    runner.state.record_success(
        seq_start=1,
        seq_end=2,
        reason="test-setup",
        end_offset=0,
        end_tokens=0,
    )
    assert runner.state.consume_early_trigger(2)

    runner.before_eviction(3, 3)
    await runner.drain()

    position = json.loads(runner.position_path.read_text(encoding="utf-8"))
    assert calls == 0
    assert position["seq"] == 3
    assert position["early_trigger_seq"] == 2
    assert position["transcript_bytes"] == path.stat().st_size
    assert position["transcript_tokens"] > 0

    replacement = AutoMemoryReconciler(
        registry=registry,
        project_id=project_id,
        session_id=SESSION,
        session_dir=runner.session_dir,
        invoke=invoke,
        config=runner.config,
    )
    assert replacement.last_reconciled_seq == 3
    assert replacement.state.early_trigger_seq == 2
    replacement.activity(3)
    await replacement.drain()
    assert calls == 0
    await runner.close()
    await replacement.close()
