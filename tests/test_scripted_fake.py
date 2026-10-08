"""Scripted fake provider: script format, replay, and real approval wiring."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from tests.support.scripted_provider import (
    FAKE_SCRIPT_ENV,
    FakeScriptError,
    ScriptedFakeBackend,
    fake_script_from_env,
    load_fake_script,
    parse_fake_script,
)
from zeta.cli.main import build_parser
from zeta.protocol.types import (
    Message,
    MessageRole,
    StreamEventType,
    TextContent,
    ToolResult,
    ToolUseContent,
)
from zeta.runtime.headless import run_headless
from zeta.server import ZetaServer

FIXTURES = Path(__file__).parent / "fixtures" / "scripted-provider"
TIMEOUT = 5

BASH_SCRIPT = {
    "version": 1,
    "rules": [
        {
            "match": {"contains": "run"},
            "responses": [
                {
                    "steps": [
                        {"type": "text", "text": "Running."},
                        {
                            "type": "tool_call",
                            "name": "bash",
                            "arguments": {"command": "echo scripted-ok"},
                        },
                    ],
                    "usage": {"input_tokens": 7, "output_tokens": 3},
                },
                {"steps": [{"type": "text", "text": "All done."}]},
            ],
        },
        {
            "match": {"equals": "fail"},
            "responses": [
                {
                    "steps": [
                        {
                            "type": "error",
                            "code": "rate_limit_error",
                            "message": "slow down",
                            "status": 429,
                        }
                    ]
                }
            ],
        },
    ],
}


def _write(tmp_path: Path, script: object, name: str = "script.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(script), encoding="utf-8")
    return path


def _with_steps(steps: list[object]) -> dict[str, object]:
    return {"version": 1, "rules": [{"responses": [{"steps": steps}]}]}


@pytest.mark.parametrize(
    ("script", "message"),
    [
        ([], "<script>: must be an object"),
        ({"rules": []}, "missing field 'version'"),
        ({"version": 2, "rules": [{}]}, "<script>.version: must be 1"),
        ({"version": True, "rules": [{}]}, "<script>.version: must be 1"),
        ({"version": 1, "rules": []}, "<script>.rules: must be a non-empty list"),
        ({"version": 1, "rules": [], "extra": 1}, "unknown field 'extra'"),
        (
            {"version": 1, "rules": [{"responses": [{"steps": []}]}]},
            "rules[0].responses[0].steps: must be a non-empty list",
        ),
        (
            {"version": 1, "rules": [{"match": {}, "responses": []}]},
            "rules[0].match: set exactly one of",
        ),
        (
            {
                "version": 1,
                "rules": [{"match": {"contains": "a", "equals": "a"}, "responses": []}],
            },
            "rules[0].match: set exactly one of",
        ),
        (
            {"version": 1, "rules": [{"match": {"regex": "("}, "responses": []}]},
            "rules[0].match.regex:",
        ),
        (_with_steps([{"text": "x"}]), "steps[0]: must be an object with a 'type'"),
        (_with_steps([{"type": "sing"}]), "steps[0].type: must be one of"),
        (_with_steps([{"type": "text", "text": ""}]), "steps[0].text: must be a non-empty"),
        (
            _with_steps([{"type": "text", "text": "x", "chunk_size": 0}]),
            "steps[0].chunk_size: must be a positive integer",
        ),
        (
            _with_steps([{"type": "text", "text": "x", "delay": -1}]),
            "steps[0].delay: must be a non-negative number",
        ),
        (
            _with_steps([{"type": "text", "text": "x", "colour": "red"}]),
            "steps[0]: unknown field 'colour'",
        ),
        (
            _with_steps([{"type": "tool_call", "name": "bash", "arguments": []}]),
            "steps[0].arguments: must be an object",
        ),
        (_with_steps([{"type": "tool_call"}]), "steps[0]: missing field 'name'"),
        (
            _with_steps([{"type": "error", "code": "x", "status": 42}]),
            "steps[0].status: must be an HTTP status code",
        ),
        (
            _with_steps([{"type": "error", "code": "x"}, {"type": "text", "text": "y"}]),
            "an error step must be the last step",
        ),
        (
            {
                "version": 1,
                "rules": [
                    {
                        "responses": [
                            {
                                "steps": [{"type": "text", "text": "x"}],
                                "usage": {"input_tokens": -1},
                            }
                        ]
                    }
                ],
            },
            "usage.input_tokens: must be a non-negative integer",
        ),
    ],
)
def test_invalid_scripts_name_the_bad_field(script: object, message: str) -> None:
    with pytest.raises(FakeScriptError) as raised:
        parse_fake_script(script)
    assert message in str(raised.value)


def test_load_reports_missing_file_and_bad_json(tmp_path: Path) -> None:
    with pytest.raises(FakeScriptError, match="cannot read"):
        load_fake_script(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(FakeScriptError, match="invalid JSON"):
        load_fake_script(bad)


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("*.json")), ids=lambda path: path.name)
def test_scripted_provider_fixtures_are_valid(path: Path) -> None:
    assert load_fake_script(path).rules


def test_env_lookup_is_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(FAKE_SCRIPT_ENV, raising=False)
    assert fake_script_from_env() is None
    assert fake_script_from_env({FAKE_SCRIPT_ENV: ""}) is None
    path = _write(tmp_path, BASH_SCRIPT)
    assert fake_script_from_env({FAKE_SCRIPT_ENV: str(path)}) is not None


def test_hidden_serve_hook_uses_scripted_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZETA_TEST_SCRIPTED_PROVIDER", "1")
    server = ZetaServer(
        home=tmp_path, socket_path=tmp_path / "s.sock", provider="codex"
    )

    backend, model = server.runtime._build_backend("codex", None, tmp_path)

    assert model == "gpt-5.6-luna"
    assert type(backend).__name__ == "_TestScriptedBackend"


async def _events(backend: ScriptedFakeBackend, messages: list[Message]) -> list[Any]:
    return [event async for event in backend.complete(messages, [])]


def test_replay_is_deterministic_and_follows_the_conversation() -> None:
    backend = ScriptedFakeBackend(parse_fake_script(BASH_SCRIPT))
    user = Message(MessageRole.USER, [TextContent("please run it")])
    first = asyncio.run(_events(backend, [user]))
    assert asyncio.run(_events(backend, [user])) == first
    assert [event.delta for event in first if event.delta] == ["Running."]
    end = first[-1]
    assert end.type is StreamEventType.MESSAGE_END
    assert end.data == {
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 7, "output_tokens": 3},
    }
    call = next(block.tool_call for block in end.message.content if isinstance(block, ToolUseContent))
    assert call.name == "bash"
    assert call.arguments == {"command": "echo scripted-ok"}
    assert end.message.content == [TextContent("Running."), ToolUseContent(call)]

    history = [
        user,
        end.message,
        Message(MessageRole.TOOL_RESULT, tool_result=ToolResult(call.id, "ok")),
    ]
    second = asyncio.run(_events(backend, history))
    assert [event.delta for event in second if event.delta] == ["All done."]
    assert second[-1].data == {"stop_reason": "end_turn"}

    exhausted = asyncio.run(_events(backend, [*history, second[-1].message]))
    assert exhausted[-1].error.code == "fake_script_exhausted"

    unmatched = asyncio.run(
        _events(backend, [Message(MessageRole.USER, [TextContent("hello")])])
    )
    assert unmatched[-1].error.code == "fake_script_no_match"


def _tool_call_id(events: list[Any]) -> str:
    return next(
        block.tool_call.id
        for block in events[-1].message.content
        if isinstance(block, ToolUseContent)
    )


def _tool_script(*, call_id: str | None = None):
    step: dict[str, object] = {
        "type": "tool_call",
        "name": "bash",
        "arguments": {"command": "echo fresh"},
    }
    if call_id is not None:
        step["id"] = call_id
    return parse_fake_script(_with_steps([step]))


def test_explicit_script_id_unique_across_turns() -> None:
    backend = ScriptedFakeBackend(_tool_script(call_id="fixed"))
    first = asyncio.run(_events(backend, [Message(MessageRole.USER, [TextContent("one")])]))
    second = asyncio.run(_events(backend, [Message(MessageRole.USER, [TextContent("two")])]))
    first_id = _tool_call_id(first)
    second_id = _tool_call_id(second)
    assert first_id != second_id
    assert first_id.endswith("_fixed")
    assert second_id.endswith("_fixed")


def test_default_id_unique_after_history_reduction() -> None:
    backend = ScriptedFakeBackend(_tool_script())
    first = asyncio.run(_events(backend, [Message(MessageRole.USER, [TextContent("one")])]))
    second = asyncio.run(_events(backend, [Message(MessageRole.USER, [TextContent("two")])]))
    assert _tool_call_id(second) != _tool_call_id(first)


def test_same_request_replays_same_id() -> None:
    backend = ScriptedFakeBackend(_tool_script(call_id="readable"))
    request = [Message(MessageRole.USER, [TextContent("same")])]
    first = asyncio.run(_events(backend, request))
    replay = asyncio.run(_events(backend, request))
    assert _tool_call_id(first) == _tool_call_id(replay)


def test_text_and_thinking_stream_in_chunks() -> None:
    script = parse_fake_script(
        _with_steps(
            [
                {"type": "thinking", "text": "abcdef", "chunk_size": 4},
                {"type": "text", "text": "hello ", "chunk_size": 3},
                {"type": "text", "text": "world"},
            ]
        )
    )
    events = asyncio.run(
        _events(ScriptedFakeBackend(script), [Message(MessageRole.USER, [TextContent("x")])])
    )
    assert [event.content.text for event in events if event.content] == ["abcd", "ef"]
    assert [event.delta for event in events if event.delta] == ["hel", "lo ", "world"]
    assert [block.to_dict()["text"] for block in events[-1].message.content] == [
        "abcdef",
        "hello world",
    ]


def _socket_path(tmp_path: Path) -> Path:
    return Path("/tmp") / f"zeta-sf-{tmp_path.name}.sock"


async def _read(reader: asyncio.StreamReader) -> dict[str, Any]:
    line = await asyncio.wait_for(reader.readline(), TIMEOUT)
    assert line, "server closed the socket"
    return json.loads(line)


async def _request(reader, writer, request_id: int, method: str, params=None):
    payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}
    writer.write((json.dumps(payload) + "\n").encode())
    await asyncio.wait_for(writer.drain(), TIMEOUT)
    while True:
        frame = await _read(reader)
        if frame.get("id") == request_id:
            return frame


async def _event(reader: asyncio.StreamReader, name: str) -> dict[str, Any]:
    while True:
        frame = await _read(reader)
        if frame.get("params", {}).get("event") == name:
            return frame["params"]


async def _serve_bash_turn(tmp_path: Path, decision: str) -> dict[str, Any]:
    script = parse_fake_script(BASH_SCRIPT)
    server = ZetaServer(
        home=tmp_path / "home",
        cwd=tmp_path,
        socket_path=_socket_path(tmp_path),
        provider="codex",
        backend_factory=lambda _provider, model, _home: (
            ScriptedFakeBackend(script, model=model or "gpt-5.6-luna"),
            model or "gpt-5.6-luna",
        ),
    )
    await asyncio.wait_for(server.start(), TIMEOUT)
    reader, writer = await asyncio.wait_for(
        asyncio.open_unix_connection(str(server.socket_path)), TIMEOUT
    )
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.0"})
        await _request(reader, writer, 2, "new_session", {"provider": "codex"})
        await _request(reader, writer, 3, "send", {"text": "run the check"})
        approval = await _event(reader, "approval_request")
        request_id = approval["request_id"]
        assert request_id.startswith("fake_")
        assert approval["tool_call"]["name"] == "bash"
        reply = await _request(reader, writer, 4, decision, {"request_id": request_id})
        assert reply["result"]["decision"] == decision
        end = await _event(reader, "tool_end")
        message = await _event(reader, "assistant_message")
        assert message["message"]["content"] == [{"type": "text", "text": "All done."}]
        await _event(reader, "turn_end")
        return end["tool_result"]
    finally:
        writer.close()
        await asyncio.wait_for(writer.wait_closed(), TIMEOUT)
        await asyncio.wait_for(server.close(), TIMEOUT)


@pytest.mark.asyncio
async def test_serve_scripted_bash_asks_approval_then_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(FAKE_SCRIPT_ENV, str(_write(tmp_path, BASH_SCRIPT)))
    result = await _serve_bash_turn(tmp_path, "approve")
    assert result["is_error"] is False
    assert "scripted-ok" in json.dumps(result)


@pytest.mark.asyncio
async def test_serve_scripted_bash_deny_returns_error_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(FAKE_SCRIPT_ENV, str(_write(tmp_path, BASH_SCRIPT)))
    result = await _serve_bash_turn(tmp_path, "deny")
    assert result["is_error"] is True
    assert "scripted-ok" not in json.dumps(result)


@pytest.mark.asyncio
async def test_serve_scripted_provider_error_reaches_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(FAKE_SCRIPT_ENV, str(_write(tmp_path, BASH_SCRIPT)))
    monkeypatch.setattr(
        "zeta.providers.retry_policy.retry_wait_seconds", lambda *_args: 0.0
    )
    script = parse_fake_script(BASH_SCRIPT)
    server = ZetaServer(
        home=tmp_path / "home",
        cwd=tmp_path,
        socket_path=_socket_path(tmp_path),
        provider="codex",
        backend_factory=lambda _provider, model, _home: (
            ScriptedFakeBackend(script, model=model or "gpt-5.6-luna"),
            model or "gpt-5.6-luna",
        ),
    )
    await asyncio.wait_for(server.start(), TIMEOUT)
    reader, writer = await asyncio.wait_for(
        asyncio.open_unix_connection(str(server.socket_path)), TIMEOUT
    )
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.0"})
        await _request(reader, writer, 2, "new_session", {"provider": "codex"})
        await _request(reader, writer, 3, "send", {"text": " fail "})
        error = await _event(reader, "error")
        assert error["error"]["code"] == "rate_limit_error"
        assert error["error"]["message"] == "slow down"
    finally:
        writer.close()
        await asyncio.wait_for(writer.wait_closed(), TIMEOUT)
        await asyncio.wait_for(server.close(), TIMEOUT)


def test_headless_scripted_tool_respects_allowlist_and_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    script = parse_fake_script(BASH_SCRIPT)

    def build_scripted(_provider: str, model: str | None, **_kwargs: object):
        selected = model or "gpt-5.6-luna"
        return ScriptedFakeBackend(script, model=selected), selected

    monkeypatch.setattr("zeta.tui.app.build_backend", build_scripted)

    def run(*flags: str) -> list[dict[str, Any]]:
        args = build_parser().parse_args(
            ["--provider", "codex", "--no-session", "--format", "json", *flags, "-p", "run"]
        )
        assert run_headless(args, args.prompt) == 0
        out = capsys.readouterr().out
        return [json.loads(line) for line in out.splitlines() if line]

    denied = run()
    result = next(event for event in denied if event["type"] == "tool_result")
    assert result["is_error"] is True and "scripted-ok" not in result["content"]
    assert denied[-1] == {"type": "message", "role": "assistant", "text": "All done."}

    allowed = run("--yolo")
    result = next(event for event in allowed if event["type"] == "tool_result")
    assert result["is_error"] is False and "scripted-ok" in result["content"]

    restricted = run("--yolo", "--tools", "read")
    result = next(event for event in restricted if event["type"] == "tool_result")
    assert result["is_error"] is True and "scripted-ok" not in result["content"]
