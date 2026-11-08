"""Codex tool-argument JSON parsing."""

from __future__ import annotations

import json
from typing import Any

from ..codex import CodexStreamError


def _parse_partial_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_complete_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError as exc:
        raise CodexStreamError("Codex tool arguments are invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise CodexStreamError("Codex tool arguments are not an object")
    return parsed
