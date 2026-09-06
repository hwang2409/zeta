"""One-shot headless driver: run a single turn without the TUI.

The CLI dispatches here when ``-p``/``--print`` is set. The driver reuses the
loop that ``zeta.tui.app.create_app`` builds, then drains one ``run_turn`` and
writes results to stdout as plain text or as a JSONL event stream.

JSONL event schema (``--format json``), one JSON object per line:

- ``{"type": "turn_start", "prompt": <str>}``
- ``{"type": "tool_call", "id": <str>, "name": <str>,
     "arguments": <object|str>}`` — ``arguments`` is the original object when it
  serializes within ``TOOL_RESULT_MAX_BYTES``; otherwise a truncated JSON string
  with a ``... [truncated: N bytes]`` suffix.
- ``{"type": "tool_result", "id": <str>, "name": <str>, "is_error": <bool>,
     "content": <str>}`` — ``content`` is trimmed the same way once it exceeds
  ``TOOL_RESULT_MAX_BYTES``.
- ``{"type": "usage", "usage": <object>}``
- ``{"type": "retry", "text": <str>, "retry": <int>, "delay": <float>,
     "is_stall": <bool>}`` — emitted for provider retries (pre-stream and
  stall). ``is_stall`` is present when the retry follows a mid-stream stall.
  In text mode the same text is written to stderr instead.
- ``{"type": "turn_end", "tool_calls": <int>}``
- ``{"type": "error", "code": <str>, "message": <str>}`` — terminates the turn;
  no ``message`` event follows.
- ``{"type": "message", "role": "assistant", "text": <str>}`` — emitted once on
  success as the final line; absent when an ``error`` fires.

Both truncation limits are enforced against UTF-8 byte length, not character
count.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import IO, TYPE_CHECKING, Any

from .core.approval import ApprovalDecision
from .core.session import SessionError
from .types import (
    Message,
    StreamEventType,
    assistant_text,
)

if TYPE_CHECKING:
    from .loop import AgentLoop

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
                    "(headless approval-required tools are denied; "
                    "pass --yolo to allow)\n"
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


def run_headless(args: argparse.Namespace, prompt: str) -> int:
    """Build the loop from CLI args, then drive one headless turn."""

    if not prompt.strip():
        print("zeta: prompt must be a nonempty string", file=sys.stderr)
        return 2

    from .tui.app import create_app

    try:
        app = create_app(args)
    except SessionError as exc:
        print(f"zeta: {exc}", file=sys.stderr)
        return 1

    if app.ephemeral_root is not None:
        print("zeta: ephemeral session — nothing will be persisted", file=sys.stderr)
    loop = app.loop
    policy = app.approval_policy
    if policy is not None and policy.default is not ApprovalDecision.ALLOW:
        # Headless has no UI to answer ASK prompts, so both the policy default
        # AND any always_ask entries must fall through to a hard DENY. When
        # yolo was resolved to True (via --yolo or settings.toml), create_app
        # already set the default to ALLOW; leave it alone in that case.
        policy.default = ApprovalDecision.DENY
        policy.always_ask = frozenset()

    # Detach the TUI sinks the create_app path wired up; without a running
    # prompt_toolkit app they call into ``get_app()`` and raise.
    loop.set_background_event_sink(None)
    loop.set_mcp_notice_sink(None)
    loop.set_mcp_prompt_refresh(None)

    async def _run() -> int:
        loop.session_start()
        try:
            return await drive_turn(
                loop,
                prompt,
                format=args.format,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
        finally:
            await loop.close()

    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        print("zeta: aborted", file=sys.stderr)
        return 130
    finally:
        if app.ephemeral_root is not None:
            import shutil

            shutil.rmtree(app.ephemeral_root, ignore_errors=True)


__all__ = ["DENIAL_MARKER", "TOOL_RESULT_MAX_BYTES", "drive_turn", "run_headless"]
