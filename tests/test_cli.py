from __future__ import annotations

import argparse
import os
import pty
import select
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from zeta.cli import build_parser, main
from zeta.completion import completion_script


def test_completion_parser_accepts_both_shells() -> None:
    parser = build_parser()

    assert parser.parse_args(["completion", "zsh"]).shell == "zsh"
    assert parser.parse_args(["completion", "bash"]).shell == "bash"


def test_completion_scripts_are_deterministic_and_cover_cli_surface() -> None:
    parser = build_parser()
    required = _parser_surface(parser)
    root_flags = _parser_options(parser)
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    serve_flags = _parser_options(subparsers.choices["serve"])
    serve_only_flags = serve_flags - root_flags

    for shell in ("zsh", "bash"):
        script = completion_script(shell)
        assert script == completion_script(shell)
        assert all(value in script for value in required)
        assert all(value in script for value in serve_flags)
        if shell == "bash":
            top_flags = script.split('local top_flags="', 1)[1].split('"', 1)[0]
            assert all(value in top_flags for value in root_flags)
            assert not any(value in top_flags for value in serve_only_flags)
        else:
            root_arguments = script.split("'1:command", 1)[0]
            assert all(value in root_arguments for value in root_flags)
            assert not any(value in root_arguments for value in serve_only_flags)


def _parser_surface(parser: argparse.ArgumentParser) -> set[str]:
    surface: set[str] = set()

    def visit(current: argparse.ArgumentParser) -> None:
        for action in current._actions:
            surface.update(action.option_strings)
            if isinstance(action, argparse._SubParsersAction):
                surface.update(action.choices)
                for subparser in action.choices.values():
                    visit(subparser)

    visit(parser)
    return surface


def _parser_options(parser: argparse.ArgumentParser) -> set[str]:
    return {
        option
        for action in parser._actions
        for option in action.option_strings
    }


def _read_pty(fd: int, timeout: float = 2.0) -> bytes:
    output = b""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ready, _, _ = select.select([fd], [], [], deadline - time.monotonic())
        if not ready:
            break
        try:
            output += os.read(fd, 4096)
        except OSError:
            break
    return output


def _run_zsh_completion(
    script: str, line: bytes, tmp_path: Path, *, cwd: Path | None = None
) -> bytes:
    tmp_path.mkdir()
    zshrc = tmp_path / ".zshrc"
    zshrc.write_text(
        "PS1='ZETA_READY> '\n"
        "autoload -Uz compinit\n"
        f"compinit -d {tmp_path / '.zcompdump'}\n"
        f"source {script}\n"
        "bindkey '^I' expand-or-complete\n",
        encoding="utf-8",
    )
    master, slave = pty.openpty()
    process = None
    try:
        process = subprocess.Popen(
            ["zsh", "-i"],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            env={**os.environ, "HOME": str(tmp_path), "ZDOTDIR": str(tmp_path)},
            cwd=cwd,
        )
        os.close(slave)
        slave = -1
        _read_pty(master)
        os.write(master, line + b"\t")
        output = _read_pty(master)
        process.kill()
        process.wait(timeout=5)
        return output
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if slave >= 0:
            os.close(slave)
        os.close(master)


def _run_bash_completion(script: Path, line: bytes, tmp_path: Path) -> bytes:
    tmp_path.mkdir()
    master, slave = pty.openpty()
    process = None
    try:
        process = subprocess.Popen(
            ["bash", "--noprofile", "--norc", "-i"],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            env={**os.environ, "HOME": str(tmp_path)},
            cwd=tmp_path,
        )
        os.close(slave)
        slave = -1
        _read_pty(master)
        os.write(master, f"source {script}\n".encode())
        _read_pty(master)
        os.write(master, line + b"\t\t")
        return _read_pty(master)
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if slave >= 0:
            os.close(slave)
        os.close(master)


def test_zsh_completion_runs_live_for_global_options(tmp_path) -> None:
    if shutil.which("zsh") is None:
        pytest.skip("zsh is not installed")
    script = tmp_path / "_zeta"
    script.write_text(completion_script("zsh"), encoding="utf-8")

    top_level = _run_zsh_completion(script, b"zeta ", tmp_path / "top")
    session = _run_zsh_completion(
        script, b"zeta --verbose session ", tmp_path / "session"
    )
    provider = _run_zsh_completion(
        script, b"zeta --provider claude ", tmp_path / "provider"
    )
    (tmp_path / "draft.json").touch()
    (tmp_path / "-draft.json").touch()
    import_path = _run_zsh_completion(
        script, b"zeta automation import dra", tmp_path / "import", cwd=tmp_path
    )
    end_of_options_path = _run_zsh_completion(
        script,
        b"zeta automation import -- -dra",
        tmp_path / "end-of-options",
        cwd=tmp_path,
    )

    assert b"bad substitution" not in (
        top_level + session + provider + import_path + end_of_options_path
    )
    assert b"login" in top_level and b"completion" in top_level
    assert b"list" in session and b"rename" in session
    assert b"login" in provider and b"session" in provider
    assert b"draft.json" in import_path
    assert b"-draft.json" in end_of_options_path


def test_bash_completion_runs_live_for_global_options(tmp_path) -> None:
    if shutil.which("bash") is None:
        pytest.skip("bash is not installed")
    script = tmp_path / "zeta"
    script.write_text(completion_script("bash"), encoding="utf-8")
    probe = """
source "$1"
probe() {
    COMP_WORDS=($@)
    COMP_CWORD=$(($# - 1))
    _zeta_completions
    printf '<%s>\n' "${COMPREPLY[*]}"
}
probe zeta ''
probe zeta --verbose session ''
probe zeta --provider claude ''
"""
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", probe, "bash", str(script)],
        check=True,
        capture_output=True,
        text=True,
    )

    lines = result.stdout.splitlines()
    assert "login" in lines[0] and "completion" in lines[0]
    assert "list" in lines[1] and "rename" in lines[1]
    assert "login" in lines[2] and "session" in lines[2]


def test_bash_completion_routes_split_equals_options_in_a_live_shell(tmp_path) -> None:
    if shutil.which("bash") is None:
        pytest.skip("bash is not installed")
    script = tmp_path / "zeta"
    script.write_text(completion_script("bash"), encoding="utf-8")

    output = _run_bash_completion(
        script, b"zeta --provider=claude session ", tmp_path / "probe"
    )

    assert b"list" in output and b"rename" in output
    assert b"--socket" not in output
    assert b"--port" not in output
    assert b"--cwd" not in output


def test_completion_command_prints_without_reading_runtime_state(capsys) -> None:
    assert main(["completion", "bash"]) == 0
    output = capsys.readouterr().out

    assert output.startswith("# install: zeta completion bash")
    assert "complete -F _zeta_completions zeta" in output
