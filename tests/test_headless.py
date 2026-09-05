from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from zeta.cli import build_parser, main
from zeta.core.approval import ApprovalDecision, ApprovalPolicy
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
