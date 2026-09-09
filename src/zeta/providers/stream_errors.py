"""Decode structured stream errors independently of optional display text."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .auth import error_body_excerpt


@dataclass(frozen=True)
class StreamErrorDetail:
    code: str | None
    status_code: int | None
    retry_reason: str | None
    message: str


def decode_stream_error(
    detail: Mapping[str, Any], *, include_message: bool = False
) -> StreamErrorDetail:
    # Capture every machine field before inspecting optional human text.
    code = None
    for key in ("code", "type"):
        value = detail.get(key)
        if type(value) is str and value:
            code = value
            break
    status = detail.get("status_code")
    status_code = status if type(status) is int else None
    reason = detail.get("type")
    retry_reason = (
        reason if type(reason) is str and reason in {"overloaded_error", "rate_limit_error"}
        else None
    )

    message = "stream error"
    if include_message:
        raw_message = detail.get("message")
        if type(raw_message) is str:
            try:
                text = error_body_excerpt(raw_message.encode())
                text.encode()  # JSON message text can contain escaped lone surrogates.
                message = text or message
            except (UnicodeError, ValueError, RecursionError):
                pass
    if code:
        message = f"{code}: {message}"
    return StreamErrorDetail(code, status_code, retry_reason, message)
