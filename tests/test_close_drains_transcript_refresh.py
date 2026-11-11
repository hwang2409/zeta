"""Close must finish a queued transcript-index refresh, not cancel it.

A one-shot headless session (``zeta -p``) ends right after its last turn. The
turn schedules a transcript-index refresh, then ``AgentLoop.close`` tears the
loop down. When close cancels the queued refresh before it runs, that last turn
never reaches transcript search until another turn or a manual index rebuild.

This test drives the real loop builder, store, and transcript index. It holds the
queued refresh behind a gate until close's drain step releases it, so the refresh
is still queued when close handles tracked tasks. On the old code close cancels
it and the turn never reaches transcript search; with the fix close drains it and
the turn is searchable. The gate removes the natural scheduling race so the
behavior is deterministic on both the base and the fix.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from tests.support.owned import await_or_fail
from zeta.cli.main import build_parser
from zeta.core.session import env_home
from zeta.protocol.types import (
    ASSISTANT_RESPONSE_COMPLETED,
    ASSISTANT_RESPONSE_STATE,
    Message,
    MessageOrigin,
    MessageRole,
    TextContent,
    with_message_origin,
)
from zeta.runtime.loop import _store_writes
from zeta.runtime.loop.agent import AgentLoop
from zeta.transcript_search.index import TranscriptIndex
from zeta.tui.app import create_app


def test_close_indexes_queued_transcript_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / "settings.toml").write_text("[memory]\nauto = false\n", encoding="utf-8")
    monkeypatch.setenv("ZETA_HOME", str(home))
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    monkeypatch.chdir(repo)
    token = "zebraquxindexmarker"

    # Hold the queued refresh until the drain step releases the gate. On the old
    # code close has no drain step, so the refresh stays queued and close cancels
    # it; with the fix the drain releases the gate and awaits the refresh.
    gate = asyncio.Event()
    real_observe = _store_writes._observe_refresh

    async def gated_observe(refresh: object) -> None:
        await gate.wait()
        await real_observe(refresh)  # type: ignore[arg-type]

    monkeypatch.setattr(_store_writes, "_observe_refresh", gated_observe)
    real_drain = getattr(AgentLoop, "_drain_transcript_refreshes", None)
    if real_drain is not None:

        async def drain_then_release(self: AgentLoop) -> None:
            gate.set()
            await real_drain(self)

        monkeypatch.setattr(AgentLoop, "_drain_transcript_refreshes", drain_then_release)

    args = build_parser().parse_args(["--provider", "codex"])
    app = create_app(args)  # off-loop: builder takes the project identity lock
    captured: dict[str, str] = {}

    async def drive() -> None:
        loop = app.loop
        await loop.activate()
        await loop.store.append_message_async(
            with_message_origin(
                Message(MessageRole.USER, [TextContent(f"please remember {token}")]),
                MessageOrigin.USER,
            )
        )
        await loop.store.append_message_async(
            Message(
                MessageRole.ASSISTANT,
                [TextContent("noted")],
                metadata={ASSISTANT_RESPONSE_STATE: ASSISTANT_RESPONSE_COMPLETED},
            )
        )
        assert loop.root_project_id is not None, "session must be a project session"
        captured["project_id"] = loop.root_project_id
        loop._schedule_transcript_index()
        await await_or_fail(loop.close(), what="loop close")
        await await_or_fail(app.close(), what="app close")  # records the owner closed

    asyncio.run(drive())

    # Read the index directly, with no backfill: open_project_index would rebuild
    # an empty index from the session files and hide a dropped refresh.
    project_dir = env_home() / "projects" / captured["project_id"]
    index = TranscriptIndex._reference(project_dir, captured["project_id"])
    assert index.status().unit_count >= 1, "last turn missing from transcript index"
