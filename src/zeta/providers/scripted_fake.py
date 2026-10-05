"""Scripted offline provider for ``--provider fake`` with ``ZETA_FAKE_SCRIPT``.

The script is a JSON file. It maps user messages to a list of model responses.
Tool calls in a response go through the normal agent loop, so the tool
registry, the approval policy, and the allowlist apply exactly as they do for a
network provider. See ``docs/fake-provider.md`` for the format.

The backend is stateless: it finds the last user message in the request and
counts the assistant messages after it to select the next response. Replays of
the same conversation therefore produce the same events.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..protocol.types import (
    CompletionBackend,
    ContentBlock,
    ErrorInfo,
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolUseContent,
)

FAKE_SCRIPT_ENV = "ZETA_FAKE_SCRIPT"
SCRIPT_VERSION = 1
MAX_SCRIPT_BYTES = 4 * 1024 * 1024


class FakeScriptError(ValueError):
    """The fake-provider script is missing or does not match the format."""


@dataclass(frozen=True, slots=True)
class TextStep:
    text: str
    chunk_size: int | None = None
    delay: float = 0.0


@dataclass(frozen=True, slots=True)
class ThinkingStep:
    text: str
    chunk_size: int | None = None
    delay: float = 0.0


@dataclass(frozen=True, slots=True)
class ToolCallStep:
    name: str
    arguments: dict[str, Any]
    call_id: str | None = None
    delay: float = 0.0


@dataclass(frozen=True, slots=True)
class ErrorStep:
    code: str
    message: str
    status: int | None = None
    delay: float = 0.0


Step = TextStep | ThinkingStep | ToolCallStep | ErrorStep


@dataclass(frozen=True, slots=True)
class ScriptResponse:
    steps: tuple[Step, ...]
    usage: dict[str, int] | None = None
    stop_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ScriptRule:
    match_kind: str | None
    match_value: str | None
    pattern: re.Pattern[str] | None
    responses: tuple[ScriptResponse, ...]

    def matches(self, prompt: str) -> bool:
        if self.match_kind is None:
            return True
        if self.match_kind == "equals":
            return prompt.strip() == self.match_value
        if self.match_kind == "contains":
            assert self.match_value is not None
            return self.match_value in prompt
        assert self.pattern is not None
        return self.pattern.search(prompt) is not None


@dataclass(frozen=True, slots=True)
class FakeScript:
    rules: tuple[ScriptRule, ...]
    source: str = "<script>"


def fake_script_from_env(environ: Mapping[str, str] | None = None) -> FakeScript | None:
    """Load the script named by ``ZETA_FAKE_SCRIPT``; ``None`` when unset."""

    value = (os.environ if environ is None else environ).get(FAKE_SCRIPT_ENV, "")
    if not value:
        return None
    return load_fake_script(value)


def load_fake_script(path: str | Path) -> FakeScript:
    location = Path(path).expanduser()
    try:
        with location.open("rb") as handle:
            raw = handle.read(MAX_SCRIPT_BYTES + 1)
    except OSError as exc:
        raise FakeScriptError(
            f"{FAKE_SCRIPT_ENV}: cannot read {location}: {exc.strerror or exc}"
        ) from exc
    if len(raw) > MAX_SCRIPT_BYTES:
        raise FakeScriptError(f"{location}: script is larger than {MAX_SCRIPT_BYTES} bytes")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FakeScriptError(f"{location}: invalid JSON: {exc}") from exc
    return parse_fake_script(data, source=str(location))


def parse_fake_script(data: object, *, source: str = "<script>") -> FakeScript:
    root = _object(data, source, required={"version", "rules"})
    if root["version"] != SCRIPT_VERSION or type(root["version"]) is not int:
        raise FakeScriptError(f"{source}.version: must be {SCRIPT_VERSION}")
    rules = _list(root["rules"], f"{source}.rules")
    return FakeScript(
        tuple(_rule(rule, f"{source}.rules[{index}]") for index, rule in enumerate(rules)),
        source,
    )


def _rule(value: object, where: str) -> ScriptRule:
    rule = _object(value, where, required={"responses"}, optional={"match"})
    kind: str | None = None
    text: str | None = None
    pattern: re.Pattern[str] | None = None
    if "match" in rule:
        match = _object(rule["match"], f"{where}.match", optional={"equals", "contains", "regex"})
        if len(match) != 1:
            raise FakeScriptError(
                f"{where}.match: set exactly one of 'equals', 'contains', 'regex'"
            )
        ((kind, raw),) = match.items()
        text = _string(raw, f"{where}.match.{kind}")
        if kind == "regex":
            try:
                pattern = re.compile(text)
            except re.error as exc:
                raise FakeScriptError(f"{where}.match.regex: {exc}") from exc
    responses = _list(rule["responses"], f"{where}.responses")
    return ScriptRule(
        kind,
        text,
        pattern,
        tuple(
            _response(response, f"{where}.responses[{index}]")
            for index, response in enumerate(responses)
        ),
    )


def _response(value: object, where: str) -> ScriptResponse:
    response = _object(value, where, required={"steps"}, optional={"usage", "stop_reason"})
    steps = tuple(
        _step(step, f"{where}.steps[{index}]")
        for index, step in enumerate(_list(response["steps"], f"{where}.steps"))
    )
    for step in steps[:-1]:
        if isinstance(step, ErrorStep):
            raise FakeScriptError(f"{where}.steps: an error step must be the last step")
    usage: dict[str, int] | None = None
    if "usage" in response:
        usage = {}
        for key, count in _object(response["usage"], f"{where}.usage").items():
            if type(count) is not int or count < 0:
                raise FakeScriptError(f"{where}.usage.{key}: must be a non-negative integer")
            usage[key] = count
    stop_reason = (
        _string(response["stop_reason"], f"{where}.stop_reason")
        if "stop_reason" in response
        else None
    )
    return ScriptResponse(steps, usage, stop_reason)


def _step(value: object, where: str) -> Step:
    if not isinstance(value, dict) or "type" not in value:
        raise FakeScriptError(f"{where}: must be an object with a 'type' field")
    kind = value["type"]
    if kind in {"text", "thinking"}:
        step = _object(
            value, where, required={"type", "text"}, optional={"chunk_size", "delay"}
        )
        chunk_size = step.get("chunk_size")
        if chunk_size is not None and (type(chunk_size) is not int or chunk_size < 1):
            raise FakeScriptError(f"{where}.chunk_size: must be a positive integer")
        step_type = TextStep if kind == "text" else ThinkingStep
        return step_type(_string(step["text"], f"{where}.text"), chunk_size, _delay(step, where))
    if kind == "tool_call":
        step = _object(
            value, where, required={"type", "name"}, optional={"arguments", "id", "delay"}
        )
        arguments = _object(step.get("arguments", {}), f"{where}.arguments")
        call_id = _string(step["id"], f"{where}.id") if "id" in step else None
        return ToolCallStep(
            _string(step["name"], f"{where}.name"), arguments, call_id, _delay(step, where)
        )
    if kind == "error":
        step = _object(
            value, where, required={"type", "code"}, optional={"message", "status", "delay"}
        )
        status = step.get("status")
        if status is not None and (type(status) is not int or not 100 <= status <= 599):
            raise FakeScriptError(f"{where}.status: must be an HTTP status code")
        code = _string(step["code"], f"{where}.code")
        message = (
            _string(step["message"], f"{where}.message")
            if "message" in step
            else f"scripted provider error: {code}"
        )
        return ErrorStep(code, message, status, _delay(step, where))
    raise FakeScriptError(
        f"{where}.type: must be one of 'text', 'thinking', 'tool_call', 'error'"
    )


def _object(
    value: object,
    where: str,
    *,
    required: set[str] | None = None,
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FakeScriptError(f"{where}: must be an object")
    if required is None and optional is None:
        return value
    missing = sorted((required or set()) - value.keys())
    if missing:
        raise FakeScriptError(f"{where}: missing field '{missing[0]}'")
    unknown = sorted(value.keys() - (required or set()) - (optional or set()))
    if unknown:
        raise FakeScriptError(f"{where}: unknown field '{unknown[0]}'")
    return value


def _list(value: object, where: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise FakeScriptError(f"{where}: must be a non-empty list")
    return value


def _string(value: object, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise FakeScriptError(f"{where}: must be a non-empty string")
    return value


def _delay(step: Mapping[str, Any], where: str) -> float:
    delay = step.get("delay", 0)
    if (
        isinstance(delay, bool)
        or not isinstance(delay, int | float)
        or not math.isfinite(delay)
        or delay < 0
    ):
        raise FakeScriptError(f"{where}.delay: must be a non-negative number of seconds")
    return float(delay)


class ScriptedFakeBackend(CompletionBackend):
    """Play a ``FakeScript`` as provider stream events."""

    def __init__(self, script: FakeScript, *, model: str = "offline") -> None:
        self.script = script
        self.model = model

    async def complete(
        self, messages: Sequence[Message], tool_schemas: Sequence[dict[str, Any]]
    ) -> AsyncIterator[StreamEvent]:
        del tool_schemas
        prompt, response_index = _position(messages)
        request_digest = _request_digest(messages)
        yield StreamEvent(StreamEventType.MESSAGE_START)
        rule = next((rule for rule in self.script.rules if rule.matches(prompt)), None)
        if rule is None:
            yield _error("fake_script_no_match", f"no script rule matches: {prompt[:200]!r}")
            return
        if response_index >= len(rule.responses):
            yield _error(
                "fake_script_exhausted",
                f"script rule has {len(rule.responses)} responses; "
                f"the model was called {response_index + 1} times",
            )
            return
        response = rule.responses[response_index]
        blocks: list[ContentBlock] = []
        calls = 0
        for step_index, step in enumerate(response.steps):
            if isinstance(step, ErrorStep):
                await _sleep(step.delay)
                yield _error(step.code, step.message, step.status)
                return
            if isinstance(step, ToolCallStep):
                await _sleep(step.delay)
                call = ToolCall(
                    _call_id(request_digest, response_index, step_index, step.call_id),
                    step.name,
                    json.loads(json.dumps(step.arguments)),
                )
                calls += 1
                blocks.append(ToolUseContent(call))
                yield StreamEvent(StreamEventType.MESSAGE_UPDATE, tool_call=call)
                continue
            size = step.chunk_size or len(step.text)
            for start in range(0, len(step.text), size):
                await _sleep(step.delay)
                chunk = step.text[start : start + size]
                if isinstance(step, TextStep):
                    yield StreamEvent(StreamEventType.MESSAGE_UPDATE, delta=chunk)
                else:
                    yield StreamEvent(
                        StreamEventType.MESSAGE_UPDATE, content=ThinkingContent(chunk)
                    )
            _append_text(blocks, step)
        data: dict[str, Any] = {
            "stop_reason": response.stop_reason or ("tool_use" if calls else "end_turn")
        }
        if response.usage is not None:
            data["usage"] = dict(response.usage)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, blocks),
            data=data,
        )


def _position(messages: Sequence[Message]) -> tuple[str, int]:
    """Return the last user text and the number of replies after it."""

    prompt = ""
    replies = 0
    for message in messages:
        if message.role is MessageRole.USER:
            text = "".join(
                block.text for block in message.content if isinstance(block, TextContent)
            )
            if text:
                prompt = text
                replies = 0
        elif message.role is MessageRole.ASSISTANT:
            replies += 1
    return prompt, replies


def _request_digest(messages: Sequence[Message]) -> str:
    """Return a stable identity for the complete provider request."""

    serialized = json.dumps(
        [message.to_dict() for message in messages],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:12]


def _call_id(digest: str, response_index: int, step_index: int, suffix: str | None) -> str:
    base = f"fake_{digest}_{response_index}_{step_index}"
    return f"{base}_{suffix}" if suffix is not None else base


def _append_text(blocks: list[ContentBlock], step: TextStep | ThinkingStep) -> None:
    last = blocks[-1] if blocks else None
    if isinstance(step, TextStep):
        if isinstance(last, TextContent):
            blocks[-1] = TextContent(last.text + step.text)
        else:
            blocks.append(TextContent(step.text))
    elif isinstance(last, ThinkingContent):
        blocks[-1] = ThinkingContent(last.text + step.text)
    else:
        blocks.append(ThinkingContent(step.text))


def _error(code: str, message: str, status: int | None = None) -> StreamEvent:
    return StreamEvent(
        StreamEventType.ERROR, error=ErrorInfo(code, message, status_code=status)
    )


async def _sleep(delay: float) -> None:
    if delay:
        await asyncio.sleep(delay)


__all__ = [
    "FAKE_SCRIPT_ENV",
    "FakeScript",
    "FakeScriptError",
    "ScriptedFakeBackend",
    "fake_script_from_env",
    "load_fake_script",
    "parse_fake_script",
]
