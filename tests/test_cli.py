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

from zeta.cli.main import build_parser, main
from zeta.core.commands.completion import completion_script
from zeta.project_registry import ProjectRegistry


def test_removed_fake_provider_has_clear_cli_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="2"):
        build_parser().parse_args(["--provider", "fake"])

    assert (
        "the fake provider was removed; choose claude, codex or ollama"
        in capsys.readouterr().err
    )


class _TTYInput:
    def __init__(self, value: str) -> None:
        self.value = value

    def isatty(self) -> bool:
        return True

    def readline(self) -> str:
        return self.value


class _TTYOutput:
    def __init__(self) -> None:
        self.value = ""

    def isatty(self) -> bool:
        return True

    def write(self, value: str) -> int:
        self.value += value
        return len(value)

    def flush(self) -> None:
        pass


def _seed_cli_memory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ProjectRegistry, str, Path]:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("demo", "scope", repo)
    registry.initialize_memory(project.project_id)
    snapshot = registry.memory_snapshot(project.project_id)
    registry.compare_and_swap_memory(
        project.project_id,
        expected_digest=snapshot.digest,
        updates={"state.md": "# State\nautomatic\n"},
        provenance={"session_id": "s", "seq_start": 1, "seq_end": 1},
    )
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(repo)
    return registry, project.project_id, repo


def test_cli_memory_accept_refuses_non_interactive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    registry, project_id, _ = _seed_cli_memory(tmp_path, monkeypatch)
    assert main(["project", "memory", "accept", "state.md"]) != 0
    error = capsys.readouterr().err
    assert "interactive terminal" in error
    assert registry.memory_log(project_id)[-1]["provenance"] != {"accepted_by": "user"}


def test_cli_memory_accept_refuses_inside_tool_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry, project_id, _ = _seed_cli_memory(tmp_path, monkeypatch)
    monkeypatch.setenv("ZETA_TOOL_SUBPROCESS", "1")
    stdin = _TTYInput("accept\n")
    stdout = _TTYOutput()
    monkeypatch.setattr("sys.stdin", stdin)
    monkeypatch.setattr("sys.stdout", stdout)
    assert main(["project", "memory", "accept", "state.md"]) != 0
    assert registry.memory_log(project_id)[-1]["provenance"] != {"accepted_by": "user"}


def test_cli_memory_accept_interactive_confirmation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry, project_id, _ = _seed_cli_memory(tmp_path, monkeypatch)
    stdin = _TTYInput("state.md\n")
    stdout = _TTYOutput()
    monkeypatch.setattr("sys.stdin", stdin)
    monkeypatch.setattr("sys.stdout", stdout)
    assert main(["project", "memory", "accept", "state.md"]) == 0
    assert registry.memory_log(project_id)[-1]["provenance"] == {"accepted_by": "user"}

    registry, project_id, _ = _seed_cli_memory(tmp_path / "wrong", monkeypatch)
    monkeypatch.setattr("sys.stdin", _TTYInput("nope\n"))
    monkeypatch.setattr("sys.stdout", _TTYOutput())
    assert main(["project", "memory", "accept", "state.md"]) != 0
    assert registry.memory_log(project_id)[-1]["provenance"] != {"accepted_by": "user"}


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
        assert all(
            verb in script
            for verb in ("url", "show-secret", "rotate-secret", "rotate-url")
        )
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
    mcp = _run_zsh_completion(script, b"zeta mcp ", tmp_path / "mcp")
    mcp_add = _run_zsh_completion(script, b"zeta mcp add --", tmp_path / "mcp-add")
    mcp_show = _run_zsh_completion(
        script, b"zeta mcp show --", tmp_path / "mcp-show"
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
    webhook = _run_zsh_completion(
        script, b"zeta automation webhook ", tmp_path / "webhook"
    )
    daemon = _run_zsh_completion(
        script, b"zeta --verbose automation daemon --", tmp_path / "daemon"
    )

    assert b"bad substitution" not in (
        top_level
        + session
        + provider
        + mcp
        + mcp_add
        + mcp_show
        + import_path
        + end_of_options_path
        + webhook
        + daemon
    )
    assert b"login" in top_level and b"completion" in top_level
    assert b"list" in session and b"rename" in session
    assert b"login" in provider and b"session" in provider
    assert b"add" in mcp and b"login" in mcp and b"logout" in mcp
    assert b"--scope" in mcp_add and b"--url" in mcp_add and b"--oauth" in mcp_add
    assert b"--scope" in mcp_show and b"--json" in mcp_show
    assert b"draft.json" in import_path
    assert b"-draft.json" in end_of_options_path
    assert all(
        verb in webhook
        for verb in (b"url", b"show-secret", b"rotate-secret", b"rotate-url")
    )
    assert all(
        option in daemon
        for option in (b"--webhook-host", b"--webhook-port", b"--allow-non-loopback")
    ), daemon


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
probe zeta mcp ''
probe zeta mcp add --
probe zeta mcp show --
probe zeta automation webhook ''
probe zeta --verbose automation daemon --
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
    assert "add" in lines[3] and "login" in lines[3] and "logout" in lines[3]
    assert "--scope" in lines[4] and "--url" in lines[4] and "--oauth" in lines[4]
    assert "--scope" in lines[5] and "--json" in lines[5]
    assert all(
        verb in lines[6]
        for verb in ("url", "show-secret", "rotate-secret", "rotate-url")
    )
    assert all(
        option in lines[7]
        for option in ("--webhook-host", "--webhook-port", "--allow-non-loopback")
    )


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


def _credential_job(tmp_path: Path, name: str, *, webhook: bool = True):
    from zeta.automations.models import parse_job

    trigger = {"kind": "webhook", "verify": "github"} if webhook else {
        "kind": "schedule", "cron": "0 9 * * *", "timezone": "UTC"
    }
    return parse_job(name, {
        "prompt": "test", "trigger": trigger, "servers": [], "allow": [],
        "deliver": "slack:U123", "provider": "codex", "model": "fake",
        "cwd": str(tmp_path),
    })


def test_cli_webhook_url_show_rotate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from datetime import UTC, datetime

    from zeta.automations.store import SQLiteStore

    monkeypatch.setenv("ZETA_HOME", str(tmp_path))
    with SQLiteStore(tmp_path) as store:
        for name, webhook in (("hook", True), ("plain", False), ("off", True)):
            state = store.draft(_credential_job(tmp_path, name, webhook=webhook))
            store.approve(name, state.revision, "U123", datetime.now(UTC))
        store.disable("off")
        before = store.webhook_credentials("hook")
    assert (tmp_path / "automations").stat().st_mode & 0o777 == 0o700

    assert main(["automation", "webhook", "url", "hook"]) == 0
    assert before.token in capsys.readouterr().out
    assert main(["automation", "webhook", "show-secret", "hook"]) == 0
    assert capsys.readouterr().out.strip() == before.secret.hex()
    assert main(["automation", "webhook", "rotate-secret", "hook"]) == 0
    capsys.readouterr()
    with SQLiteStore(tmp_path) as store:
        after_secret = store.webhook_credentials("hook")
    assert after_secret.secret != before.secret and after_secret.token == before.token
    assert main(["automation", "webhook", "rotate-url", "hook"]) == 0
    capsys.readouterr()
    with SQLiteStore(tmp_path) as store:
        after_url = store.webhook_credentials("hook")
    assert after_url.secret == after_secret.secret and after_url.token != after_secret.token

    assert main(["automation", "list"]) == 0
    listing = capsys.readouterr().out
    assert main(["automation", "show", "hook"]) == 0
    shown = capsys.readouterr().out
    for private in (after_url.secret.hex(), after_url.token):
        assert private not in listing and private not in shown
    for name in ("plain", "off"):
        for operation in ("url", "show-secret", "rotate-secret", "rotate-url"):
            assert main(["automation", "webhook", operation, name]) == 1
            capsys.readouterr()
