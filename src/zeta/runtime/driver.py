"""Drive a normal agent turn without a frontend."""
from __future__ import annotations

import json
from typing import IO, TYPE_CHECKING, Any

from ..types import Message, StreamEventType, assistant_text

if TYPE_CHECKING:
    from ..loop import AgentLoop

TOOL_RESULT_MAX_BYTES = 8_000
DENIAL_MARKER = "tool execution denied"


def _bounded(value: str, limit: int = TOOL_RESULT_MAX_BYTES) -> str:
    data = value.encode("utf-8")
    if len(data) <= limit:
        return value
    kept = data[:limit].decode("utf-8", errors="ignore")
    dropped = len(data) - limit
    return kept + f"... [truncated: {dropped} bytes]"


def _bounded_arguments(
    arguments: dict[str, object],
    limit: int = TOOL_RESULT_MAX_BYTES,
) -> dict[str, object] | str:
    serialized = json.dumps(arguments, separators=(",", ":"), ensure_ascii=False)
    if len(serialized.encode("utf-8")) <= limit:
        return arguments
    return _bounded(serialized, limit)


def _emit_jsonl(stream: IO[str], event: dict[str, Any]) -> None:
    stream.write(json.dumps(event, separators=(",", ":"), ensure_ascii=False))
    stream.write("\n")
    stream.flush()


async def drive_turn(
    loop: AgentLoop,
    prompt: str,
    *,
    format: str,
    stdout: IO[str],
    stderr: IO[str],
    denial_hint: str = "headless approval-required tools are denied; pass --yolo to allow",
) -> int:
    """Drain one turn against ``loop`` and write results to the given streams.

    Returns 0 on success, nonzero when the loop reports a terminal error or
    produces no final assistant text.
    """

    final_message: Message | None = None
    error_code: str | None = None
    error_message: str | None = None

    if format == "json":
        _emit_jsonl(stdout, {"type": "turn_start", "prompt": prompt})

    async for event in loop.run_turn(prompt):
        if event.type is StreamEventType.MESSAGE_END:
            usage = event.data.get("usage") if isinstance(event.data, dict) else None
            if format == "json" and isinstance(usage, dict) and usage:
                _emit_jsonl(stdout, {"type": "usage", "usage": dict(usage)})
        elif event.type is StreamEventType.TOOL_EXECUTION_START:
            if event.tool_call is not None and format == "json":
                _emit_jsonl(
                    stdout,
                    {
                        "type": "tool_call",
                        "id": event.tool_call.id,
                        "name": event.tool_call.name,
                        "arguments": _bounded_arguments(event.tool_call.arguments),
                    },
                )
        elif event.type is StreamEventType.TOOL_EXECUTION_END:
            result = event.tool_result
            if result is None:
                continue
            name = event.tool_call.name if event.tool_call is not None else ""
            if format == "json":
                _emit_jsonl(
                    stdout,
                    {
                        "type": "tool_result",
                        "id": result.tool_call_id,
                        "name": name,
                        "is_error": bool(result.is_error),
                        "content": _bounded(result.content),
                    },
                )
            if result.is_error and result.content == DENIAL_MARKER:
                stderr.write(
                    f"zeta: denied tool call {name!r} "
                    f"({denial_hint})\n"
                )
                stderr.flush()
        elif event.type is StreamEventType.TURN_END:
            tool_calls = 0
            if isinstance(event.data, dict):
                data_calls = event.data.get("tool_calls")
                if isinstance(data_calls, int):
                    tool_calls = data_calls
            if tool_calls == 0 and event.message is not None:
                final_message = event.message
            if format == "json":
                _emit_jsonl(stdout, {"type": "turn_end", "tool_calls": tool_calls})
        elif event.type is StreamEventType.RETRY:
            data = event.data if isinstance(event.data, dict) else {}
            text = data.get("text")
            display = text if isinstance(text, str) and text else "retrying"
            if format == "json":
                payload: dict[str, Any] = {"type": "retry", "text": display}
                for key in ("retry", "delay", "is_stall"):
                    if key in data:
                        payload[key] = data[key]
                _emit_jsonl(stdout, payload)
            else:
                stderr.write(f"zeta: {display}\n")
                stderr.flush()
        elif event.type is StreamEventType.ERROR and event.error is not None:
            error_code = event.error.code
            error_message = event.error.message
            if format == "json":
                _emit_jsonl(
                    stdout,
                    {
                        "type": "error",
                        "code": error_code,
                        "message": error_message,
                    },
                )

    if error_message is not None:
        stderr.write(f"zeta: {error_code}: {error_message}\n")
        stderr.flush()
        return 1
    if final_message is None:
        stderr.write("zeta: no final assistant message produced\n")
        stderr.flush()
        return 1

    final_text = assistant_text(final_message)
    if format == "text":
        stdout.write(final_text)
        if not final_text.endswith("\n"):
            stdout.write("\n")
        stdout.flush()
    else:
        _emit_jsonl(
            stdout,
            {"type": "message", "role": "assistant", "text": final_text},
        )
    return 0

