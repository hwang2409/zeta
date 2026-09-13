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
from typing import IO, Any

from .core.approval import ApprovalDecision
from .core.session import SessionError
from .runtime.driver import (
    DENIAL_MARKER,
    TOOL_RESULT_MAX_BYTES,
    drive_turn,
)


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

    async def _run() -> int:
        try:
            if app.ephemeral_root is not None:
                print("zeta: ephemeral session — nothing will be persisted", file=sys.stderr)
            loop = app.loop
            policy = app.approval_policy
            if policy is not None:
                for notice in policy.notices:
                    print(f"zeta: {notice}", file=sys.stderr)
            if policy is not None and policy.default is not ApprovalDecision.ALLOW:
                # Headless has no UI to answer ASK prompts, so both the policy default
                # AND any always_ask entries must fall through to a hard DENY. When
                # yolo was resolved to True (via --yolo or settings.toml), create_app
                # already set the default to ALLOW; leave it alone in that case. The
                # assignment goes through the rule-set setter, so it clears
                # argument-scoped ask rules (ZETA-86) as well as bare ones.
                policy.default = ApprovalDecision.DENY
                policy.always_ask = frozenset()

            # Detach the TUI sinks the create_app path wired up; without a running
            # prompt_toolkit app they call into ``get_app()`` and raise.
            loop.set_background_event_sink(None)
            loop.set_mcp_notice_sink(None)
            loop.set_mcp_prompt_refresh(None)

            await loop.activate()
            return await drive_turn(
                loop,
                prompt,
                format=args.format,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
        finally:
            await app.close()

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
