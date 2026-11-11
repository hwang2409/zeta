from __future__ import annotations

import asyncio
import gc
import logging
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.support.owned import await_or_fail
from zeta.cli.main import build_parser
from zeta.core.session import SessionManager, env_home
from zeta.runtime.headless import run_headless
from zeta.runtime.loop import AgentLoop


def test_resume_with_registry_path_as_file_warns_once_without_unretrieved_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    original_close = AgentLoop.close

    async def close_after_refresh(self: AgentLoop, **kwargs: object) -> tuple[str, ...]:
        # Let a pending transcript-index refresh reach its failure observation
        # before close() cancels tracked tasks; otherwise py3.14 scheduling can
        # cancel it before it logs, dropping the warning this test asserts.
        refresh_tasks = [
            task
            for task in self._tracked_tasks
            if task.get_coro().__qualname__.endswith("_observe_refresh")
            and not task.done()
        ]
        if refresh_tasks:
            await await_or_fail(
                asyncio.gather(*refresh_tasks),
                what="transcript refresh observation",
            )
        return await original_close(self, **kwargs)

    monkeypatch.setattr(AgentLoop, "close", close_after_refresh)
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    monkeypatch.chdir(repo)
    parser = build_parser()
    first = parser.parse_args(["--provider", "codex", "-p", "first turn"])
    assert run_headless(first, first.prompt) == 0
    (session,) = SessionManager(env_home()).list_sessions()
    assert session.project_id is not None
    projects = env_home() / "projects"
    shutil.rmtree(projects)
    projects.write_text("not a directory", encoding="utf-8")
    capsys.readouterr()
    caplog.clear()
    caplog.set_level(logging.WARNING)

    second = parser.parse_args(
        ["--provider", "codex", "--resume", session.session_id, "-p", "second turn"]
    )
    assert run_headless(second, second.prompt) == 0
    gc.collect()  # a dropped failed task reports "never retrieved" when collected

    messages = [record.getMessage() for record in caplog.records]
    assert not [m for m in messages if "never retrieved" in m], messages
    refresh_warnings = [
        m for m in messages if m.startswith("could not refresh transcript index")
    ]
    assert len(refresh_warnings) == 1, messages
    assert str(projects.resolve() / session.project_id) in refresh_warnings[0]
    assert "Not a directory" in refresh_warnings[0]
