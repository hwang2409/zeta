import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.support.fake_backend import FakeBackend
from zeta.core.slash import create_slash_registry
from zeta.core.store import ConversationStore
from zeta.memory.auto import AutoMemoryConfig, AutoMemoryReconciler
from zeta.project_registry import ProjectRegistry
from zeta.runtime.loop import AgentLoop
from zeta.server.slash_commands import ServerSlashSession
from zeta.skills import SkillCatalog
from zeta.tui.app import TUIApp

SESSION = "a" * 32


def _write_transcript(session_dir: Path, count: int) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "conversation.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "seq": seq,
                    "id": f"m{seq}",
                    "parent_id": None if seq == 1 else f"m{seq - 1}",
                    "type": "message",
                    "data": {
                        "message": {
                            "role": "user",
                            "content": [{"type": "text", "text": f"fact {seq}"}],
                        }
                    },
                }
            )
            + "\n"
            for seq in range(1, count + 1)
        ),
        encoding="utf-8",
    )


def _setup(tmp_path: Path, invoke, *, transcript_count: int):
    home = tmp_path / ".zeta"
    workspace = tmp_path / "repo"
    workspace.mkdir()
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("demo", "scope", workspace)
    registry.initialize_memory(project.project_id)
    session_dir = home / "sessions" / SESSION
    _write_transcript(session_dir, transcript_count)
    runner = AutoMemoryReconciler(
        registry=registry,
        project_id=project.project_id,
        session_id=SESSION,
        session_dir=session_dir,
        invoke=invoke,
        config=AutoMemoryConfig(minimum_interval=0, retry_backoff_seconds=0),
    )
    return runner, registry, project.project_id


def _reopen(runner, registry, project_id: str, invoke) -> AutoMemoryReconciler:
    return AutoMemoryReconciler(
        registry=registry,
        project_id=project_id,
        session_id=SESSION,
        session_dir=runner.session_dir,
        invoke=invoke,
        config=AutoMemoryConfig(minimum_interval=0, retry_backoff_seconds=0),
    )


def _proposal(prompt: str) -> str:
    rows = json.loads(prompt.split("Completed transcript rows:", 1)[1].strip())
    start, end = rows[0]["seq"], rows[-1]["seq"]
    return json.dumps(
        {
            "changes": [
                {
                    "file": "decisions.md",
                    "content": f"# Decisions\n\nrange {start}-{end}\n",
                    "sources": [
                        {
                            "session_id": SESSION,
                            "seq_start": start,
                            "seq_end": end,
                        }
                    ],
                }
            ]
        }
    )


def _archived_terminal_row(seq: int) -> bytes:
    return json.dumps(
        {
            "key": f"terminal-{seq}",
            "seq_start": seq,
            "seq_end": seq,
            "attempt_count": 3,
            "validation_summary": "terminal failure",
            "occurred_at": "2026-11-08T00:00:00+00:00",
        },
        separators=(",", ":"),
    ).encode()


@pytest.mark.asyncio
async def test_broken_cursor_stops_reconciliation_and_renders_through_tui_and_serve(
    tmp_path: Path,
) -> None:
    calls = 0
    called = asyncio.Event()

    async def invoke(prompt: str) -> str:
        nonlocal calls
        calls += 1
        called.set()
        return _proposal(prompt)

    runner, registry, project_id = await asyncio.wait_for(
        asyncio.to_thread(_setup, tmp_path, invoke, transcript_count=1), 10
    )
    await runner.close()
    broken_bytes = b'{"seq": not-json}\n'
    runner.position_path.write_bytes(broken_bytes)
    replacement = await asyncio.wait_for(
        asyncio.to_thread(_reopen, runner, registry, project_id, invoke), 10
    )
    loop = AgentLoop(
        FakeBackend([]),
        ConversationStore(tmp_path / "tui-sessions"),
        skill_catalog=SkillCatalog.empty(),
    )
    loop.memory_reconciler = replacement
    app = TUIApp(loop, provider="codex", model="offline")
    runtime = SimpleNamespace(
        loop=loop,
        state=SimpleNamespace(usage_tracker=app._usage_tracker),
        opened=SimpleNamespace(
            metadata=SimpleNamespace(
                session_id=loop.store.session_id,
                provider="codex",
                model="offline",
                context_files=(),
                vim_mode=True,
                plan_mode=False,
            )
        ),
    )
    try:
        replacement.before_eviction(1, 1)
        if replacement._worker_task is not None:
            await asyncio.wait_for(called.wait(), 10)
        tui = create_slash_registry(skill_catalog=SkillCatalog.empty()).dispatch(
            app, "/status"
        )
        serve = create_slash_registry(skill_catalog=SkillCatalog.empty()).dispatch(
            ServerSlashSession(runtime), "/status"
        )

        assert calls == 0
        assert replacement.position_path.read_bytes() == broken_bytes
        assert tui is not None and serve is not None
        for rendered in (tui, serve):
            assert "memory-reconcile.json" in rendered
            assert "malformed JSON" in rendered
            assert "preserved" in rendered
            assert "back up the cursor" in rendered
    finally:
        await app.close()


@pytest.mark.asyncio
async def test_malformed_interior_receipt_stops_without_overwriting(
    tmp_path: Path,
) -> None:
    calls = 0
    called = asyncio.Event()

    async def invoke(prompt: str) -> str:
        nonlocal calls
        calls += 1
        called.set()
        return _proposal(prompt)

    runner, registry, project_id = await asyncio.wait_for(
        asyncio.to_thread(_setup, tmp_path, invoke, transcript_count=3), 10
    )
    await runner.close()
    archive = runner.state.receipts_path
    first = _archived_terminal_row(1)
    broken_bytes = first + b"\n{bad interior}\n" + _archived_terminal_row(2) + b"\n"
    archive.write_bytes(broken_bytes)
    replacement = await asyncio.wait_for(
        asyncio.to_thread(_reopen, runner, registry, project_id, invoke), 10
    )
    try:
        replacement.before_eviction(1, 3)
        if replacement._worker_task is not None:
            await asyncio.wait_for(called.wait(), 10)

        assert calls == 0
        assert archive.read_bytes() == broken_bytes
        assert replacement.automatic_memory_status is not None
        assert "memory-reconcile-receipts.jsonl" in replacement.automatic_memory_status
        assert f"byte offset {len(first) + 1}" in replacement.automatic_memory_status
        assert "back up the archive" in replacement.automatic_memory_status
    finally:
        await replacement.close()


@pytest.mark.asyncio
async def test_torn_final_receipt_is_truncated_without_repeating_terminal_work(
    tmp_path: Path,
) -> None:
    prompts: list[str] = []

    async def invoke(prompt: str) -> str:
        prompts.append(prompt)
        return _proposal(prompt)

    runner, registry, project_id = await asyncio.wait_for(
        asyncio.to_thread(_setup, tmp_path, invoke, transcript_count=2), 10
    )
    await runner.close()
    archive = runner.state.receipts_path
    complete = _archived_terminal_row(1) + b"\n"
    archive.write_bytes(complete + b'{"key":"torn-final"')
    replacement = await asyncio.wait_for(
        asyncio.to_thread(_reopen, runner, registry, project_id, invoke), 10
    )
    try:
        replacement.before_eviction(1, 2)
        await replacement.drain()

        assert archive.read_bytes() == complete
        assert len(prompts) == 1
        assert "fact 1" not in prompts[0]
        assert "fact 2" in prompts[0]
    finally:
        await replacement.close()


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.name == "nt" or not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="mode-bit permission failures are not portable on Windows or as root",
)
async def test_unreadable_cursor_stops_reconciliation(tmp_path: Path) -> None:
    calls = 0
    called = asyncio.Event()

    async def invoke(prompt: str) -> str:
        nonlocal calls
        calls += 1
        called.set()
        return _proposal(prompt)

    runner, registry, project_id = await asyncio.wait_for(
        asyncio.to_thread(_setup, tmp_path, invoke, transcript_count=1), 10
    )
    await asyncio.wait_for(
        asyncio.to_thread(
            runner.state.record_success,
            seq_start=1,
            seq_end=1,
            reason="test",
            end_offset=1,
            end_tokens=1,
        ),
        10,
    )
    await runner.close()
    before = runner.position_path.read_bytes()
    runner.position_path.chmod(0)
    replacement = None
    try:
        replacement = await asyncio.wait_for(
            asyncio.to_thread(_reopen, runner, registry, project_id, invoke), 10
        )
        replacement.before_eviction(1, 1)
        if replacement._worker_task is not None:
            await asyncio.wait_for(called.wait(), 10)

        assert calls == 0
        assert replacement.automatic_memory_status is not None
        assert "could not read" in replacement.automatic_memory_status
    finally:
        runner.position_path.chmod(0o600)
        if replacement is not None:
            await replacement.close()
    assert runner.position_path.read_bytes() == before
