from __future__ import annotations

import gc
import logging
import shutil
import subprocess
from pathlib import Path

import pytest

from zeta.cli.main import build_parser
from zeta.core.session import SessionManager, env_home
from zeta.runtime.headless import run_headless


def test_resume_with_registry_path_as_file_warns_once_without_unretrieved_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
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
