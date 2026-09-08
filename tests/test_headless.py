from __future__ import annotations

import argparse
import io
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from zeta.cli import build_parser, main
from zeta.core.approval import ApprovalDecision, ApprovalPolicy, ApprovalRule
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.session import SessionManager, env_home
from zeta.core.store import ConversationStore
from zeta.headless import DENIAL_MARKER, drive_turn, run_headless
from zeta.loop import AgentLoop
from zeta.tools import ToolRegistry
from zeta.types import TextContent, ToolCall


async def _drive(
    loop: AgentLoop, prompt: str, output_format: str
) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = await drive_turn(
        loop,
        prompt,
        format=output_format,
        stdout=stdout,
        stderr=stderr,
    )
    return code, stdout.getvalue(), stderr.getvalue()


def test_print_mode_runs_session_hook_inside_async_activation(
    tmp_path: Path,
) -> None:
    home = tmp_path / "isolated-home"
    home.mkdir()
    hook = f"{shlex.quote(sys.executable)} -c {shlex.quote('pass')}"
    (home / "hooks.toml").write_text(
        f"[[hook]]\nevent = \"session_start\"\ncommand = {json.dumps(hook)}\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["ZETA_HOME"] = str(home)
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from zeta.cli import main; raise SystemExit(main())",
            "--provider",
            "fake",
            "-p",
            "hello",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("you said: hello")


async def test_text_mode_prints_final_message_and_exits_zero(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    backend = FakeBackend([ScriptedTurn(content=[TextContent("hello world")])])
    loop = AgentLoop(backend, store)

    code, out, err = await _drive(loop, "greet", "text")

    assert code == 0
    assert out == "hello world\n"
    assert err == ""


async def test_json_mode_streams_documented_lifecycle(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    backend = FakeBackend(
        [
            ScriptedTurn(
                content=[TextContent("done")],
                usage={"input_tokens": 3, "output_tokens": 4},
            )
        ]
    )
    loop = AgentLoop(backend, store)

    code, out, err = await _drive(loop, "hi", "json")

    assert code == 0
    assert err == ""
    events = [json.loads(line) for line in out.splitlines() if line]
    types = [event["type"] for event in events]
    assert types[0] == "turn_start"
    assert events[0]["prompt"] == "hi"
    assert "usage" in types
    usage_event = next(event for event in events if event["type"] == "usage")
    assert usage_event["usage"]["input_tokens"] == 3
    assert "turn_end" in types
    assert next(event for event in events if event["type"] == "turn_end")["tool_calls"] == 0
    assert types[-1] == "message"
    assert events[-1] == {"type": "message", "role": "assistant", "text": "done"}


async def test_json_mode_records_tool_call_and_result(tmp_path: Path) -> None:
    tool_call = ToolCall("call-1", "echo", {"value": "ok"})
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[tool_call]),
            ScriptedTurn(content=[TextContent("finished")]),
        ]
    )
    store = ConversationStore(tmp_path)
    registry = ToolRegistry(tmp_path, register_builtin=False)
    registry.register("echo", lambda arguments: f"got {arguments['value']}")
    loop = AgentLoop(backend, store, registry=registry)

    code, out, _err = await _drive(loop, "run", "json")

    assert code == 0
    events = [json.loads(line) for line in out.splitlines() if line]
    call_event = next(event for event in events if event["type"] == "tool_call")
    assert call_event == {
        "type": "tool_call",
        "id": "call-1",
        "name": "echo",
        "arguments": {"value": "ok"},
    }
    result_event = next(event for event in events if event["type"] == "tool_result")
    assert result_event["id"] == "call-1"
    assert result_event["name"] == "echo"
    assert result_event["is_error"] is False
    assert result_event["content"] == "got ok"


async def test_json_mode_bounds_large_tool_result(tmp_path: Path) -> None:
    call = ToolCall("call-1", "spam", {})
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[call]),
            ScriptedTurn(content=[TextContent("done")]),
        ]
    )
    store = ConversationStore(tmp_path)
    registry = ToolRegistry(tmp_path, register_builtin=False)
    big = "a" * 20_000
    registry.register("spam", lambda arguments: big)
    loop = AgentLoop(backend, store, registry=registry)

    code, out, _err = await _drive(loop, "go", "json")

    assert code == 0
    events = [json.loads(line) for line in out.splitlines() if line]
    result_event = next(event for event in events if event["type"] == "tool_result")
    assert "[truncated:" in result_event["content"]
    assert len(result_event["content"]) < len(big)


async def test_json_mode_tool_result_bound_is_byte_based(tmp_path: Path) -> None:
    from zeta.headless import TOOL_RESULT_MAX_BYTES

    call = ToolCall("call-1", "wide", {})
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[call]),
            ScriptedTurn(content=[TextContent("done")]),
        ]
    )
    store = ConversationStore(tmp_path)
    registry = ToolRegistry(tmp_path, register_builtin=False)
    # Four-byte UTF-8 emoji: char count is well under the limit, byte count is
    # well over it. Char-based bounds would silently ship the whole payload.
    big = "\U0001f600" * (TOOL_RESULT_MAX_BYTES // 2)
    registry.register("wide", lambda arguments: big)
    loop = AgentLoop(backend, store, registry=registry)

    code, out, _err = await _drive(loop, "go", "json")

    assert code == 0
    events = [json.loads(line) for line in out.splitlines() if line]
    result_event = next(event for event in events if event["type"] == "tool_result")
    assert "[truncated:" in result_event["content"]
    assert len(result_event["content"].encode("utf-8")) <= (
        TOOL_RESULT_MAX_BYTES + 64
    )


async def test_json_mode_bounds_large_tool_call_arguments(tmp_path: Path) -> None:
    from zeta.headless import TOOL_RESULT_MAX_BYTES

    big_arg = "z" * (TOOL_RESULT_MAX_BYTES * 2)
    call = ToolCall("call-1", "sink", {"payload": big_arg})
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[call]),
            ScriptedTurn(content=[TextContent("done")]),
        ]
    )
    store = ConversationStore(tmp_path)
    registry = ToolRegistry(tmp_path, register_builtin=False)
    registry.register("sink", lambda arguments: "ok")
    loop = AgentLoop(backend, store, registry=registry)

    code, out, _err = await _drive(loop, "go", "json")

    assert code == 0
    events = [json.loads(line) for line in out.splitlines() if line]
    call_event = next(event for event in events if event["type"] == "tool_call")
    arguments = call_event["arguments"]
    assert isinstance(arguments, str)
    assert "[truncated:" in arguments
    assert len(arguments.encode("utf-8")) <= (TOOL_RESULT_MAX_BYTES + 64)


async def test_headless_hook_rejection_does_not_show_yolo_hint(tmp_path: Path) -> None:
    call = ToolCall("call-1", "vetoed", {})
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[call]),
            ScriptedTurn(content=[TextContent("recovered")]),
        ]
    )
    store = ConversationStore(tmp_path)
    registry = ToolRegistry(tmp_path, register_builtin=False)
    registry.register("vetoed", lambda arguments: "must not run")
    registry.set_pre_execute_hook(lambda name, arguments: False)
    loop = AgentLoop(backend, store, registry=registry)

    code, out, err = await _drive(loop, "start", "text")

    assert code == 0
    assert "recovered" in out
    # Hook denial is distinct from approval-required denial. Headless must not
    # falsely suggest --yolo unlocks it — that only unlocks the approval path.
    assert "--yolo" not in err
    tool_results = [
        message.tool_result for message in store.messages() if message.tool_result
    ]
    assert tool_results[0].content == "tool execution denied by hook"


def test_headless_run_headless_hard_denies_always_ask_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Governance: headless must never block on ASK prompts.

    Even if a caller populates always_ask, the headless driver must strip it
    so the run terminates instead of polling for a UI answer that will never
    come.
    """

    from zeta.headless import run_headless

    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(["--provider", "fake", "-p", "hi"])

    import zeta.tui.app as tui_app

    original_create_app = tui_app.create_app
    captured: list[Any] = []

    def _wrapped_create_app(parsed: argparse.Namespace) -> tui_app.TUIApp:
        app = original_create_app(parsed)
        policy = app.approval_policy
        assert policy is not None
        policy.always_ask = frozenset({"anything"})
        captured.append(policy)
        return app

    monkeypatch.setattr(tui_app, "create_app", _wrapped_create_app)

    code = run_headless(args, args.prompt)
    capsys.readouterr()

    assert code == 0
    assert captured, "wrapped create_app should have been called"
    policy = captured[0]
    assert policy.always_ask == frozenset()
    assert policy.default is ApprovalDecision.DENY


def test_headless_respects_settings_yolo_without_cli_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """settings.toml yolo=true must reach the headless policy the same way
    it reaches the TUI: no --yolo on the command line, no --no-yolo, but the
    resolved default is ALLOW so headless does not clobber it to DENY.
    """

    from zeta.headless import run_headless

    home = tmp_path / "zeta-home"
    home.mkdir()
    monkeypatch.setenv("ZETA_HOME", str(home))
    (home / "settings.toml").write_text("yolo = true\n")

    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(["--provider", "fake", "-p", "hi"])
    assert args.yolo is None

    import zeta.tui.app as tui_app

    original_create_app = tui_app.create_app
    captured_policies: list[ApprovalPolicy] = []

    def _wrapped_create_app(parsed: argparse.Namespace) -> tui_app.TUIApp:
        app = original_create_app(parsed)
        assert app.approval_policy is not None
        captured_policies.append(app.approval_policy)
        return app

    monkeypatch.setattr(tui_app, "create_app", _wrapped_create_app)

    code = run_headless(args, args.prompt)
    capsys.readouterr()

    assert code == 0
    assert captured_policies, "wrapped create_app should have been called"
    policy = captured_policies[0]
    assert policy.default is ApprovalDecision.ALLOW


def test_headless_no_yolo_flag_beats_settings_yolo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--no-yolo on the CLI must override settings.toml yolo=true and force
    the headless policy back to DENY.
    """

    from zeta.headless import run_headless

    home = tmp_path / "zeta-home"
    home.mkdir()
    monkeypatch.setenv("ZETA_HOME", str(home))
    (home / "settings.toml").write_text("yolo = true\n")

    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(
        ["--provider", "fake", "--no-yolo", "-p", "hi"]
    )
    assert args.yolo is False

    import zeta.tui.app as tui_app

    original_create_app = tui_app.create_app
    captured_policies: list[ApprovalPolicy] = []

    def _wrapped_create_app(parsed: argparse.Namespace) -> tui_app.TUIApp:
        app = original_create_app(parsed)
        assert app.approval_policy is not None
        captured_policies.append(app.approval_policy)
        return app

    monkeypatch.setattr(tui_app, "create_app", _wrapped_create_app)

    code = run_headless(args, args.prompt)
    capsys.readouterr()

    assert code == 0
    assert captured_policies, "wrapped create_app should have been called"
    policy = captured_policies[0]
    assert policy.default is ApprovalDecision.DENY


def test_headless_hard_denies_argument_scoped_ask_rules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Regression (ZETA-86): a scoped ask rule from settings is neutralised too.

    The override assigns an empty rule set; if the rule representation ever
    stops flowing through that assignment, a ``bash(git push*)`` ask rule
    would survive and headless would poll for a UI answer that never comes.
    """

    from zeta.headless import run_headless

    home = tmp_path / "home"
    home.mkdir()
    (home / "settings.toml").write_text(
        '[approval]\nask = ["bash(git push*)"]\n', encoding="utf-8"
    )
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(["--provider", "fake", "-p", "hi"])

    import zeta.tui.app as tui_app

    original_create_app = tui_app.create_app
    captured_policies: list[ApprovalPolicy] = []

    def _wrapped_create_app(parsed: argparse.Namespace) -> tui_app.TUIApp:
        app = original_create_app(parsed)
        policy = app.approval_policy
        assert policy is not None
        assert policy.always_ask == {ApprovalRule("bash", "git push*")}
        assert policy.decide("bash", {"command": "git push"}) is ApprovalDecision.ASK
        captured_policies.append(policy)
        return app

    monkeypatch.setattr(tui_app, "create_app", _wrapped_create_app)

    code = run_headless(args, args.prompt)
    capsys.readouterr()

    assert code == 0
    policy = captured_policies[0]
    assert policy.always_ask == frozenset()
    assert policy.default is ApprovalDecision.DENY
    assert policy.decide("bash", {"command": "git push origin main"}) is ApprovalDecision.DENY
    assert policy.decide("bash", {"command": "git status"}) is ApprovalDecision.DENY


def test_headless_reports_dropped_scoped_rules_on_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from zeta.headless import run_headless

    home = tmp_path / "home"
    home.mkdir()
    (home / "settings.toml").write_text(
        '[approval]\nallow = ["todo(*)"]\n', encoding="utf-8"
    )
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(["--provider", "fake", "-p", "hi"])

    code = run_headless(args, args.prompt)
    captured = capsys.readouterr()

    assert code == 0
    assert "dropped rule 'todo(*)'" in captured.err
    assert "dropped rule" not in captured.out


async def test_headless_denies_ask_tool_and_writes_stderr_note(tmp_path: Path) -> None:
    call = ToolCall("call-1", "danger", {})
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[call]),
            ScriptedTurn(content=[TextContent("recovered")]),
        ]
    )
    store = ConversationStore(tmp_path)
    policy = ApprovalPolicy(default=ApprovalDecision.DENY, store=store)
    registry = ToolRegistry(tmp_path, register_builtin=False)
    registry.register("danger", lambda arguments: "must not run")
    loop = AgentLoop(
        backend, store, registry=registry, approval_policy=policy
    )

    code, out, err = await _drive(loop, "start", "text")

    assert code == 0
    assert "recovered" in out
    assert "denied tool call 'danger'" in err
    assert "--yolo" in err
    tool_results = [message.tool_result for message in store.messages() if message.tool_result]
    assert tool_results[0].content == DENIAL_MARKER


async def test_max_turns_error_returns_exit_one(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[ToolCall("call-1", "loop", {})]),
            ScriptedTurn(tool_calls=[ToolCall("call-2", "loop", {})]),
        ]
    )
    store = ConversationStore(tmp_path)
    registry = ToolRegistry(tmp_path, register_builtin=False)
    registry.register("loop", lambda arguments: "again")
    loop = AgentLoop(backend, store, registry=registry, max_turns=1)

    code, out, err = await _drive(loop, "start", "text")

    assert code == 1
    assert out == ""
    assert "max_turns" in err


async def test_json_mode_surfaces_error_event(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[ToolCall("call-1", "loop", {})]),
            ScriptedTurn(tool_calls=[ToolCall("call-2", "loop", {})]),
        ]
    )
    store = ConversationStore(tmp_path)
    registry = ToolRegistry(tmp_path, register_builtin=False)
    registry.register("loop", lambda arguments: "again")
    loop = AgentLoop(backend, store, registry=registry, max_turns=1)

    code, out, err = await _drive(loop, "start", "json")

    assert code == 1
    events = [json.loads(line) for line in out.splitlines() if line]
    error_events = [event for event in events if event["type"] == "error"]
    assert error_events and error_events[-1]["code"] == "max_turns"
    assert "max_turns" in err


def test_cli_headless_fake_provider_writes_final_text_to_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(["--provider", "fake", "-p", "hello"])
    assert args.prompt == "hello"

    code = run_headless(args, args.prompt)

    captured = capsys.readouterr()
    assert code == 0
    assert "you said: hello" in captured.out
    assert captured.err == ""
    # session persisted for this working directory
    cwd = str(tmp_path.resolve())
    sessions = [
        item
        for item in SessionManager(env_home()).list_sessions()
        if item.cwd == cwd
    ]
    assert sessions, "headless run should persist a session"


def test_cli_headless_json_mode_yields_valid_jsonl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(
        ["--provider", "fake", "-p", "greet", "--format", "json"]
    )

    code = run_headless(args, args.prompt)

    captured = capsys.readouterr()
    assert code == 0
    lines = [line for line in captured.out.splitlines() if line]
    events = [json.loads(line) for line in lines]
    assert events[0]["type"] == "turn_start"
    assert events[0]["prompt"] == "greet"
    assert events[-1]["type"] == "message"
    assert "you said: greet" in events[-1]["text"]


def test_cli_headless_resume_reuses_persisted_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    parser = build_parser()

    first = parser.parse_args(["--provider", "fake", "-p", "first turn"])
    assert run_headless(first, first.prompt) == 0
    capsys.readouterr()

    manager = SessionManager(env_home())
    cwd = str(tmp_path.resolve())
    sessions = [item for item in manager.list_sessions() if item.cwd == cwd]
    assert len(sessions) == 1
    session_id = sessions[0].session_id

    second = parser.parse_args(
        ["--provider", "fake", "--resume", session_id, "-p", "second turn"]
    )
    assert run_headless(second, second.prompt) == 0
    capsys.readouterr()

    resumed = ConversationStore(manager.sessions_dir, session_id=session_id)
    user_texts = [
        "".join(
            block.text for block in message.content if isinstance(block, TextContent)
        )
        for message in resumed.messages()
        if message.role.value == "user"
    ]
    assert "first turn" in user_texts
    assert "second turn" in user_texts


def test_cli_rejects_empty_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(["--provider", "fake", "-p", "   "])
    code = run_headless(args, args.prompt)
    captured = capsys.readouterr()
    assert code == 2
    assert "nonempty" in captured.err


def test_cli_format_requires_print_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        main(["--provider", "fake", "--format", "json"])
    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "--format requires --print" in captured.err
