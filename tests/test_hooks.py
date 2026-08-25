from __future__ import annotations

import asyncio
import json
import shlex
import sys
import time
from pathlib import Path

import pytest

from zeta.core.hooks import Hook, HookConfigError, HookManager, load_hooks
from zeta.cli import build_parser
from zeta.tui.app import create_app
from zeta.tools import ToolRegistry
from zeta.types import ToolCall


def _python_command(source: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(source)}"


def _write_config(home: Path, entries: str) -> None:
    home.mkdir(exist_ok=True)
    (home / "hooks.toml").write_text(entries, encoding="utf-8")


def test_missing_hooks_are_silent(tmp_path: Path) -> None:
    assert load_hooks(tmp_path).hooks == ()


def test_malformed_hooks_fail_loudly(tmp_path: Path) -> None:
    (tmp_path / "hooks.toml").write_text("[[hook]\nevent =", encoding="utf-8")

    with pytest.raises(HookConfigError, match="invalid hook config"):
        load_hooks(tmp_path)


@pytest.mark.asyncio
async def test_pre_hook_allow_deny_and_error(tmp_path: Path) -> None:
    deny = _python_command("import sys; sys.stderr.write('not safe\\n'); sys.exit(2)")
    error = _python_command("import sys; sys.exit(1)")
    _write_config(
        tmp_path,
        f'[[hook]]\nevent = "pre_tool"\ncommand = {json.dumps(deny)}\n\n'
        f'[[hook]]\nevent = "pre_tool"\ncommand = {json.dumps(error)}\n',
    )
    notices: list[str] = []
    manager = load_hooks(tmp_path)
    manager.bind_session("session-1")
    manager.notice_sink = notices.append

    denied = await manager.pre_tool("exec", {"command": "rm"})

    assert denied == "not safe"
    assert notices == ["hook denied pre_tool: not safe"]

    manager = HookManager(
        load_hooks(tmp_path).hooks[1:],
        session_id="session-1",
        notice_sink=lambda _: notices.clear(),
    )
    assert await manager.pre_tool("exec", {}) is True
    assert notices == []


@pytest.mark.asyncio
async def test_pre_hook_deny_reason_reaches_tool_result(tmp_path: Path) -> None:
    deny = _python_command("import sys; sys.stderr.write('blocked by policy'); sys.exit(2)")
    _write_config(tmp_path, f'[[hook]]\nevent = "pre_tool"\ncommand = {json.dumps(deny)}\n')
    manager = load_hooks(tmp_path)
    manager.bind_session("session-1")
    registry = ToolRegistry(tmp_path, pre_execute_hook=manager.pre_tool, register_builtin=False)
    registry.register("exec", lambda arguments: "ran")

    result = await registry.execute(ToolCall("call-1", "exec", {}))

    assert result["isError"] is True
    assert result["content"][0]["text"] == "blocked by policy"


@pytest.mark.asyncio
async def test_nonblocking_hook_does_not_delay_turn(tmp_path: Path) -> None:
    marker = tmp_path / "marker"
    command = _python_command(
        f"import pathlib,time; time.sleep(.1); pathlib.Path({str(marker)!r}).write_text('done')"
    )
    _write_config(tmp_path, f'[[hook]]\nevent = "session_start"\ncommand = {json.dumps(command)}\n')
    manager = load_hooks(tmp_path)
    manager.bind_session("session-1")

    started = time.monotonic()
    manager.session_start()
    elapsed = time.monotonic() - started
    await manager.close()

    assert elapsed < 0.05
    assert marker.read_text(encoding="utf-8") == "done"


@pytest.mark.asyncio
async def test_tool_filter_and_event_payload(tmp_path: Path) -> None:
    marker = tmp_path / "events.jsonl"
    command = _python_command(
        f"import pathlib,sys; pathlib.Path({str(marker)!r}).open('a').write(sys.stdin.read())"
    )
    _write_config(
        tmp_path,
        f'[[hook]]\nevent = "pre_tool"\ncommand = {json.dumps(command)}\ntools = ["exec"]\n',
    )
    manager = load_hooks(tmp_path)
    manager.bind_session("session-1")

    assert await manager.pre_tool("read", {}) is True
    assert await manager.pre_tool("exec", {"command": "pwd"}) is True

    event = json.loads(marker.read_text(encoding="utf-8"))
    assert event == {
        "event": "pre_tool",
        "session_id": "session-1",
        "tool": "exec",
        "args": {"command": "pwd"},
    }


@pytest.mark.asyncio
async def test_timeout_kills_hook_process_group(tmp_path: Path) -> None:
    marker = tmp_path / "child-alive"
    child = f"import pathlib,time; time.sleep(.2); pathlib.Path({str(marker)!r}).write_text('alive')"
    parent = f"import subprocess,sys,time; subprocess.Popen([sys.executable, '-c', {child!r}]); time.sleep(5)"
    manager = HookManager((), session_id="session-1")
    result = await manager._run(
        Hook("session_start", _python_command(parent), timeout_seconds=0.05),
        {},
    )

    assert result.timed_out is True
    await asyncio.sleep(0.3)
    assert not marker.exists()


def test_status_lists_hooks_and_fake_default_is_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "zeta-home"
    hook = _python_command("pass")
    _write_config(home, f'[[hook]]\nevent = "stop"\ncommand = {json.dumps(hook)}\n')
    monkeypatch.setenv("ZETA_HOME", str(home))
    app = create_app(build_parser().parse_args(["--provider", "fake"]))
    assert "stop:" in "\n".join(app.slash_status().hooks)

    monkeypatch.delenv("ZETA_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "plain-home"))
    plain_app = create_app(build_parser().parse_args(["--provider", "fake"]))
    assert plain_app.slash_status().hooks == ()
