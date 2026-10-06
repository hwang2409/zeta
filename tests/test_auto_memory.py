from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from zeta.memory.auto import AutoMemoryConfig, AutoMemoryReconciler
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
    runner = AutoMemoryReconciler(
        registry=registry,
        project_id=project.project_id,
        session_id=SESSION,
        session_dir=session_dir,
        invoke=invoke,
        config=AutoMemoryConfig(**config),
        notice=notices.append,
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


@pytest.mark.asyncio
async def test_token_growth_trigger(tmp_path: Path) -> None:
    runner, registry, project_id, _ = _runner(
        tmp_path, token_threshold=50, transcript_count=1
    )
    runner.activity(1)
    await runner.drain()
    assert not registry.memory_log(project_id)

    _write_transcript(runner.session_dir, 4)
    runner.activity(4)
    await runner.drain()
    assert len(registry.memory_log(project_id)) == 1


@pytest.mark.asyncio
async def test_idle_trigger(tmp_path: Path) -> None:
    runner, registry, project_id, _ = _runner(
        tmp_path, idle_seconds=0.01, minimum_interval=0
    )
    runner.activity(1)
    await asyncio.sleep(0.007)
    runner.activity(4)
    await asyncio.sleep(0.007)
    assert not registry.memory_log(project_id)
    await asyncio.sleep(0.02)
    await runner.drain()
    assert len(registry.memory_log(project_id)) == 1

    # Idle is re-armed by later durable activity after a completed reconciliation.
    _write_transcript(runner.session_dir, 5)
    runner.activity(5)
    await asyncio.sleep(0.03)
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
    await asyncio.sleep(0.03)
    await runner.drain()
    assert calls == 0


@pytest.mark.asyncio
async def test_oversized_row_reconciles_every_fragment_before_advancing(
    tmp_path: Path,
) -> None:
    fact = "durable-tail-fact-7Q"
    runner, registry, project_id, _ = _runner(
        tmp_path, transcript_count=1, minimum_interval=0
    )
    row = {
        "seq": 1,
        "type": "message",
        "data": {"text": "x" * (70 * 1024) + fact},
    }
    (runner.session_dir / "conversation.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )
    reached_tail = asyncio.Event()
    release_tail = asyncio.Event()
    prompts: list[str] = []

    async def invoke(prompt: str) -> str:
        prompts.append(prompt)
        if fact not in prompt:
            return '{"changes":[]}'
        reached_tail.set()
        await release_tail.wait()
        return _proposal(prompt)

    runner.invoke = invoke
    runner.before_eviction(1, 1)
    await asyncio.wait_for(reached_tail.wait(), timeout=2)

    durable_position = json.loads(runner.position_path.read_text(encoding="utf-8"))
    assert durable_position["seq"] == 0
    assert durable_position["fragment_seq"] == 1
    assert durable_position["fragment_offset"] > 0

    release_tail.set()
    await runner.drain()

    assert len(prompts) >= 2
    assert any(fact in prompt for prompt in prompts)
    assert runner.last_reconciled_seq == 1
    assert json.loads(runner.position_path.read_text(encoding="utf-8"))["seq"] == 1
    assert registry.memory_snapshot(project_id).contents["decisions.md"].endswith(
        "range 1-1\n"
    )
    records = registry.memory_log(project_id)
    assert records[-1]["provenance"]["fragment_start"] > 0


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
