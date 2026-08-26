from __future__ import annotations

import asyncio
import json
import shlex
import sys
import time
from pathlib import Path

import pytest

from zeta.core.hooks import (
    HOOK_EVENT_PAYLOAD_LIMIT,
    HOOK_EVENT_STRING_LIMIT,
    Hook,
    HookConfigError,
    HookManager,
    _bound_event,
    load_hooks,
)
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


def test_empty_tools_filter_is_rejected_for_non_tool_events(tmp_path: Path) -> None:
    hook = _python_command("pass")
    _write_config(
        tmp_path,
        f'[[hook]]\nevent = "stop"\ncommand = {json.dumps(hook)}\ntools = []\n',
    )

    with pytest.raises(HookConfigError, match="tools only applies to tool events"):
        load_hooks(tmp_path)


def test_hook_event_mapping_keys_are_bounded_with_metadata() -> None:
    key = "k" * (HOOK_EVENT_STRING_LIMIT + 1)

    event = _bound_event({"args": {key: "value"}})

    bounded_key = next(iter(event["args"]))
    assert len(bounded_key) == HOOK_EVENT_STRING_LIMIT
    assert event["_truncated"] is True
    assert event["_truncations"][0]["path"] == "$.args[key 0]"


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
async def test_hook_output_is_bounded_and_sanitized(tmp_path: Path) -> None:
    command = _python_command(
        "import sys; sys.stdout.write('ignored' * 10000); "
        "sys.stderr.write('\\x1b[31m' + 'x' * 3000 + '\\x1b[0m\\x00'); sys.exit(2)"
    )
    _write_config(
        tmp_path,
        f'[[hook]]\nevent = "pre_tool"\ncommand = {json.dumps(command)}\n',
    )
    notices: list[str] = []
    manager = load_hooks(tmp_path)
    manager.notice_sink = notices.append

    reason = await manager.pre_tool("exec", {})

    assert isinstance(reason, str)
    assert len(reason) <= 2048
    assert reason.endswith("...[truncated]")
    assert chr(27) not in reason
    assert chr(0) not in reason
    assert notices[0].startswith("hook denied pre_tool: ")


@pytest.mark.asyncio
async def test_hook_event_payload_is_bounded_with_metadata(tmp_path: Path) -> None:
    marker = tmp_path / "event.json"
    command = _python_command(
        f"import pathlib,sys; pathlib.Path({str(marker)!r}).write_text(sys.stdin.read())"
    )
    _write_config(
        tmp_path,
        f'[[hook]]\nevent = "pre_tool"\ncommand = {json.dumps(command)}\n',
    )
    manager = load_hooks(tmp_path)

    assert await manager.pre_tool(
        "exec", {"command": "x" * 5000, "items": list(range(100))}
    ) is True

    event = json.loads(marker.read_text(encoding="utf-8"))
    assert len(event["args"]["command"]) == 4096
    assert event["args"]["items"] == list(range(64))
    assert event["_truncated"] is True
    assert {entry["path"] for entry in event["_truncations"]} == {
        "$.args.command",
        "$.args.items",
    }


@pytest.mark.asyncio
async def test_oversized_hook_payload_is_bounded_with_metadata(tmp_path: Path) -> None:
    marker = tmp_path / "event.json"
    command = _python_command(
        f"import pathlib,sys; pathlib.Path({str(marker)!r}).write_bytes(sys.stdin.buffer.read())"
    )
    _write_config(
        tmp_path,
        f'[[hook]]\nevent = "pre_tool"\ncommand = {json.dumps(command)}\n',
    )
    manager = load_hooks(tmp_path)
    args = {
        f"{index:02d}-{'k' * (HOOK_EVENT_STRING_LIMIT + 1)}": "v"
        * (HOOK_EVENT_STRING_LIMIT + 1)
        for index in range(64)
    }

    assert await manager.pre_tool("exec", args) is True

    payload = marker.read_bytes()
    event = json.loads(payload)
    assert len(payload) <= HOOK_EVENT_PAYLOAD_LIMIT
    assert event["_truncated"] is True
    assert {entry["kind"] for entry in event["_truncations"]} == {"payload"}


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


@pytest.mark.asyncio
async def test_cancellation_kills_hook_process_group(tmp_path: Path) -> None:
    marker = tmp_path / "child-canceled"
    child = f"import pathlib,time; time.sleep(.2); pathlib.Path({str(marker)!r}).write_text('alive')"
    parent = f"import subprocess,sys,time; subprocess.Popen([sys.executable, '-c', {child!r}]); time.sleep(5)"
    manager = HookManager((), session_id="session-1")
    task = asyncio.create_task(
        manager._run(Hook("session_start", _python_command(parent)), {})
    )

    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.3)

    assert not marker.exists()


@pytest.mark.asyncio
async def test_hook_subprocess_disables_nested_hook_loading(tmp_path: Path) -> None:
    marker = tmp_path / "nested-hooks"
    command = _python_command(
        "import pathlib; "
        "from zeta.core.hooks import load_hooks; "
        f"pathlib.Path({str(marker)!r}).write_text(str(len(load_hooks({str(tmp_path)!r}).hooks)))"
    )
    _write_config(
        tmp_path,
        f'[[hook]]\nevent = "session_start"\ncommand = {json.dumps(command)}\n',
    )
    manager = load_hooks(tmp_path)

    manager.session_start()
    await manager.close()

    assert marker.read_text(encoding="utf-8") == "0"


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
