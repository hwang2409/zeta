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
