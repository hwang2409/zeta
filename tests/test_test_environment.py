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

COLLECTION_HOME = Path.home()


def test_console_defaults_are_suite_stable() -> None:
    console = Console(file=StringIO())

    assert console.is_terminal
    assert console.color_system == "truecolor"
    assert not console.no_color


def test_test_home_is_isolated() -> None:
    assert not Path(os.environ["ZETA_HOME"]).is_relative_to(COLLECTION_HOME)


def test_teardown_guard_defaults_to_collection_home(live_home_write_guard) -> None:
    assert live_home_write_guard.live_home == COLLECTION_HOME.resolve()
    assert Path.home() == COLLECTION_HOME


@pytest.mark.parametrize(
    ("leak_path", "during_collection"),
    [(None, False), ("unexpected", False), (".zeta/history", False), ("import-leak", True)],
)
def test_session_home_guard_ignores_external_writes_and_catches_home_leaks(
    tmp_path: Path, leak_path: str | None, during_collection: bool
) -> None:
    # Run the actual session hooks and teardown in a child pytest process so
    # intentional leaks fail that session without contaminating this one.
    probe = tmp_path / "probe"
    probe.mkdir()
    external_home = tmp_path / "external-home"
    external_home.mkdir()
    for name in ("conftest.py", "sitecustomize.py"):
        shutil.copyfile(Path(__file__).with_name(name), probe / name)
    (probe / "test_probe.py").write_text(
        "from pathlib import Path\n"
        "COLLECTED_HOME = Path.home()\n"
        f"assert COLLECTED_HOME != Path({str(external_home)!r})\n"
        + (
            "(COLLECTED_HOME / 'import-leak').write_text('collection leak')\n"
            if during_collection
            else ""
        )
        + "def test_probe():\n"
        "    assert Path.home() == COLLECTED_HOME\n"
        f"    external = Path({str(external_home)!r}) / '.zeta/run/serve.sock'\n"
        "    external.parent.mkdir(parents=True)\n"
        "    external.write_text('external writer')\n"
        + (
            f"    leaked = Path.home() / {leak_path!r}\n"
            "    leaked.parent.mkdir(parents=True, exist_ok=True)\n"
            "    leaked.write_text('ignored ZETA_HOME')\n"
            if leak_path is not None and not during_collection
            else ""
        ),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["HOME"] = str(external_home)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--confcutdir", str(probe)],
        cwd=probe,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    output = result.stdout + result.stderr

    if leak_path is None:
        assert result.returncode == 0, output
        assert "1 passed" in output
    else:
        assert result.returncode == 1, output
        assert "1 passed, 1 error" in output
        assert "tests wrote to live zeta home" in output
        assert leak_path in output


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
        live_home_write_guard.declare_external_mutation(
            history, kind="append", expected=b"external\n"
        )
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
            env={},
            start_new_session=True,
            check=True,
        )

        live_home_write_guard.assert_clean()
    finally:
        live_home_write_guard.reset()


def test_teardown_guard_rejects_two_appends_for_one_declaration(
    tmp_path: Path, live_home_write_guard
) -> None:
    live_home = tmp_path / "live-home"
    history = live_home / "history"
    history.parent.mkdir()
    history.write_text("before\n", encoding="utf-8")
    live_home_write_guard.watch(live_home)

    try:
        live_home_write_guard.declare_external_mutation(
            history, kind="append", expected=b"first\n"
        )
        append_command = (
            "from pathlib import Path; "
            "Path(__import__('sys').argv[1]).open('a').write(__import__('sys').argv[2])"
        )
        subprocess.run(
            [sys.executable, "-c", append_command, str(history), "first\n"],
            env={},
            start_new_session=True,
            check=True,
        )
        subprocess.run(
            [sys.executable, "-c", append_command, str(history), "second\n"],
            env={},
            start_new_session=True,
            check=True,
        )

        with pytest.raises(AssertionError, match="tests wrote to live zeta home"):
            live_home_write_guard.assert_clean()
    finally:
        live_home_write_guard.reset()


def test_teardown_guard_allows_declared_append_creating_file(
    tmp_path: Path, live_home_write_guard
) -> None:
    live_home = tmp_path / "live-home"
    history = live_home / "history"
    live_home.mkdir()
    live_home_write_guard.watch(live_home)

    try:
        live_home_write_guard.declare_external_mutation(
            history, kind="append", expected=b"external\n"
        )
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
            env={},
            start_new_session=True,
            check=True,
        )

        live_home_write_guard.assert_clean()
    finally:
        live_home_write_guard.reset()


def test_teardown_guard_allows_two_sequential_declared_appends(
    tmp_path: Path, live_home_write_guard
) -> None:
    live_home = tmp_path / "live-home"
    history = live_home / "history"
    history.parent.mkdir()
    history.write_text("before\n", encoding="utf-8")
    live_home_write_guard.watch(live_home)

    try:
        append_command = (
            "from pathlib import Path; "
            "Path(__import__('sys').argv[1]).open('a').write(__import__('sys').argv[2])"
        )
        live_home_write_guard.declare_external_mutation(
            history, kind="append", expected=b"first\n"
        )
        subprocess.run(
            [sys.executable, "-c", append_command, str(history), "first\n"],
            env={},
            start_new_session=True,
            check=True,
        )
        live_home_write_guard.declare_external_mutation(
            history, kind="append", expected=b"second\n"
        )
        subprocess.run(
            [sys.executable, "-c", append_command, str(history), "second\n"],
            env={},
            start_new_session=True,
            check=True,
        )

        live_home_write_guard.assert_clean()
    finally:
        live_home_write_guard.reset()


def test_teardown_guard_rejects_wrong_declared_append_content(
    tmp_path: Path, live_home_write_guard
) -> None:
    live_home = tmp_path / "live-home"
    history = live_home / "history"
    history.parent.mkdir()
    history.write_text("before\n", encoding="utf-8")
    live_home_write_guard.watch(live_home)

    try:
        live_home_write_guard.declare_external_mutation(
            history, kind="append", expected=b"external\n"
        )
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    "Path(__import__('sys').argv[1]).open('a').write('different\\n')"
                ),
                str(history),
            ],
            env={},
            start_new_session=True,
            check=True,
        )

        with pytest.raises(AssertionError, match="tests wrote to live zeta home"):
            live_home_write_guard.assert_clean()
    finally:
        live_home_write_guard.reset()


def test_teardown_guard_rejects_declared_test_process_write(
    tmp_path: Path, live_home_write_guard
) -> None:
    live_home = tmp_path / "live-home"
    history = live_home / "history"
    history.parent.mkdir()
    history.write_text("before\n", encoding="utf-8")
    live_home_write_guard.watch(live_home)

    try:
        live_home_write_guard.declare_external_mutation(
            history, kind="append", expected=b"test-owned\n"
        )
        with history.open("a", encoding="utf-8") as stream:
            stream.write("test-owned\n")

        with pytest.raises(AssertionError, match="tests wrote to live zeta home"):
            live_home_write_guard.assert_clean()
    finally:
        live_home_write_guard.reset()


def test_teardown_guard_rejects_undeclared_external_live_history_writer(
    tmp_path: Path, live_home_write_guard
) -> None:
    live_home = tmp_path / "live-home"
    history = live_home / "history"
    history.parent.mkdir()
    history.write_text("before\n", encoding="utf-8")
    live_home_write_guard.watch(live_home)

    try:
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    "Path(__import__('sys').argv[1]).open('a').write('external' + chr(92) + 'n')"
                ),
                str(history),
            ],
            env={},
            start_new_session=True,
            check=True,
        )

        with pytest.raises(AssertionError, match="tests wrote to live zeta home"):
            live_home_write_guard.assert_clean()
    finally:
        live_home_write_guard.reset()


def test_teardown_guard_rejects_unattributed_append_before_external_append(
    tmp_path: Path, live_home_write_guard
) -> None:
    live_home = tmp_path / "live-home"
    history = live_home / "history"
    history.parent.mkdir()
    history.write_text("before\n", encoding="utf-8")
    live_home_write_guard.watch(live_home)

    try:
        append_command = (
            "from pathlib import Path; "
            "Path(__import__('sys').argv[1]).open('a').write('external' + chr(92) + 'n')"
        )
        subprocess.run(
            [sys.executable, "-c", append_command, str(history)],
            env={},
            start_new_session=True,
            check=True,
        )

        live_home_write_guard.declare_external_mutation(
            history, kind="append", expected=b"external\n"
        )
        subprocess.run(
            [sys.executable, "-c", append_command, str(history)],
            env={},
            start_new_session=True,
            check=True,
        )

        with pytest.raises(AssertionError, match="tests wrote to live zeta home"):
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
            start_new_session=True,
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


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is not installed")
def test_teardown_guard_rejects_tmux_session_manager_child(
    tmp_path: Path, live_home_write_guard
) -> None:
    live_home = tmp_path / "home" / ".zeta"
    live_home.mkdir(parents=True)
    live_home_write_guard.watch(live_home)
    socket = f"zeta-test-session-manager-{os.getpid()}"
    session = f"zeta-guard-session-manager-{os.getpid()}"
    child_env = os.environ.copy()
    child_env.pop("ZETA_HOME")
    child_env["HOME"] = str(live_home.parent)
    child_code = (
        "from zeta.core.session import SessionManager; "
        "SessionManager().create(provider='fake', model='fake')"
    )
    command = (
        f"{shlex.quote(sys.executable)} -c {shlex.quote(child_code)}; "
        f"tmux -L {shlex.quote(socket)} wait-for -S zeta-session-manager-ready; "
        "sleep 5"
    )

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
                command,
            ],
            check=True,
            env=child_env,
        )
        subprocess.run(
            ["tmux", "-L", socket, "wait-for", "zeta-session-manager-ready"],
            check=True,
        )

        with pytest.raises(AssertionError, match="tests wrote to live zeta home"):
            live_home_write_guard.assert_clean()
    finally:
        subprocess.run(
            ["tmux", "-L", socket, "kill-session", "-t", session],
            check=False,
        )
        live_home_write_guard.reset()
