from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

LIVE_ZETA_HOME = Path.home() / ".zeta"


def test_console_defaults_are_suite_stable() -> None:
    console = Console(file=StringIO())

    assert console.is_terminal
    assert console.color_system == "truecolor"
    assert not console.no_color


def test_test_home_is_isolated() -> None:
    assert Path(os.environ["ZETA_HOME"]) != LIVE_ZETA_HOME


def test_path_home_follows_home_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = tmp_path / "home"
    monkeypatch.setenv("HOME", str(expected))

    assert Path.home() == expected


def test_teardown_guard_allows_external_live_history_writer(
    tmp_path: Path, live_home_write_guard
) -> None:
    live_home = tmp_path / "live-home"
    history = live_home / "history"
    history.parent.mkdir()
    history.write_text("before\n", encoding="utf-8")
    live_home_write_guard.watch(live_home)

    try:
        external_env = os.environ.copy()
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    "Path(__import__('sys').argv[1]).open('a').write('external\\n')"
                ),
                str(history),
            ],
            env=external_env,
            start_new_session=True,
            check=True,
        )

        live_home_write_guard.assert_clean()
    finally:
        live_home_write_guard.reset()


def test_teardown_guard_rejects_test_process_live_home_write(
    tmp_path: Path, live_home_write_guard
) -> None:
    live_home = tmp_path / "live-home"
    history = live_home / "history"
    history.parent.mkdir()
    live_home_write_guard.watch(live_home)

    try:
        history.write_text("leak\n", encoding="utf-8")

        with pytest.raises(AssertionError, match="tests wrote to live zeta home"):
            live_home_write_guard.assert_clean()
    finally:
        live_home_write_guard.reset()


def test_teardown_guard_rejects_stripped_home_child(
    tmp_path: Path, live_home_write_guard
) -> None:
    live_home = tmp_path / "home" / ".zeta"
    live_home_write_guard.watch(live_home)
    child_env = os.environ.copy()
    child_env.pop("ZETA_HOME")
    child_env["HOME"] = str(live_home.parent)

    try:
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from zeta.core.session import SessionManager; "
                    "SessionManager().create(provider='fake', model='fake')"
                ),
            ],
            check=True,
            env=child_env,
        )

        with pytest.raises(AssertionError, match="tests wrote to live zeta home"):
            live_home_write_guard.assert_clean()
    finally:
        live_home_write_guard.reset()


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is not installed")
def test_teardown_guard_rejects_tmux_pane_without_home(
    tmp_path: Path, live_home_write_guard
) -> None:
    live_home = tmp_path / "live-home"
    history = live_home / "history"
    live_home.mkdir()
    live_home_write_guard.watch(live_home)
    socket = f"zeta-test-{os.getpid()}"
    session = f"zeta-guard-{os.getpid()}"

    try:
        subprocess.run(
            [
                "tmux",
                "-L",
                socket,
                "new-session",
                "-d",
                "-s",
                session,
                "sh",
                "-c",
                (
                    f"unset ZETA_HOME; printf 'pane\\n' >> {shlex.quote(str(history))}; "
                    f"tmux -L {shlex.quote(socket)} wait-for -S zeta-guard-ready"
                ),
            ],
            check=True,
            env=os.environ.copy(),
        )
        subprocess.run(
            ["tmux", "-L", socket, "wait-for", "zeta-guard-ready"],
            check=True,
        )
        assert history.exists()

        with pytest.raises(AssertionError, match="tests wrote to live zeta home"):
            live_home_write_guard.assert_clean()
    finally:
        subprocess.run(
            ["tmux", "-L", socket, "kill-session", "-t", session],
            check=False,
        )
        live_home_write_guard.reset()
