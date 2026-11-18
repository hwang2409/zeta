"""Chat-span metadata from the one inference owner, and the closed ``zeta.error.code`` map."""

from __future__ import annotations

import asyncio
import random
from types import SimpleNamespace

import pytest

from zeta.codex import CodexHTTPError, CodexStreamError
from zeta.protocol.types import ErrorInfo, StreamEvent, StreamEventType
from zeta.providers.anthropic_errors import AnthropicHTTPError, AnthropicStreamError
from zeta.providers.completion import ProviderStallError
from zeta.providers.ollama import OllamaError
from zeta.runtime.loop import _completion
from zeta.runtime.loop._completion import InferenceObservation
from zeta.telemetry import SpanHandle
from zeta.telemetry.errors import ERROR_CODES, error_code

SECRET = "SECRET-PROMPT-TEXT"


@pytest.mark.parametrize(("error", "code"), [
    pytest.param(CodexStreamError(SECRET, code="invalid_request_error", retry_reason="invalid_request_error"), "invalid_request", id="codex invalid_request_error"),
    pytest.param(CodexStreamError(SECRET, code="rate_limit_exceeded"), "rate_limit", id="codex rate_limit_exceeded"),
    pytest.param(CodexStreamError(SECRET, code="server_is_overloaded", retry_reason="overloaded_error"), "overloaded", id="codex overloaded"),
    pytest.param(CodexStreamError(SECRET, code="context_length_exceeded"), "context_length", id="codex context length"),
    pytest.param(CodexHTTPError(SECRET, status_code=401), "auth", id="codex 401"),
    pytest.param(AnthropicStreamError("prompt is too long: " + SECRET, code="invalid_request_error"), "context_length", id="anthropic prompt too long"),
    pytest.param(AnthropicStreamError(SECRET, code="overloaded_error", retry_reason="overloaded_error"), "overloaded", id="anthropic overloaded"),
    pytest.param(AnthropicHTTPError(SECRET, status_code=429), "rate_limit", id="anthropic 429"),
    pytest.param(AnthropicHTTPError(SECRET, status_code=503), "server_error", id="anthropic 503"),
    pytest.param(OllamaError(SECRET, status_code=404), "model_not_found", id="ollama 404"),
    pytest.param(ProviderStallError("stream", 900.0), "timeout", id="provider stall"),
    pytest.param(ErrorInfo("max_turns", SECRET), "turn_limit", id="ErrorInfo max_turns"),
    pytest.param(ErrorInfo("backend_error", SECRET, status_code=529), "overloaded", id="ErrorInfo 529"),
    pytest.param(asyncio.CancelledError(), "cancelled", id="cancelled"),
    pytest.param(asyncio.CancelledError, "cancelled", id="cancelled class"),
    pytest.param(TimeoutError(SECRET), "timeout", id="timeout"),
    pytest.param(ConnectionResetError(SECRET), "network", id="connection reset"),
    # the message names a known code, but no machine field does: never read the message
    pytest.param(RuntimeError("invalid_request_error: rate_limit_error " + SECRET), "unknown", id="message only"),
    pytest.param(SimpleNamespace(code=SECRET, retry_reason=SECRET), "unknown", id="free-text fields"),
    pytest.param(None, "unknown", id="no error"),
])
def test_error_code_is_read_from_machine_fields_only(error: object, code: str) -> None:
    assert error_code(error) == code
    assert code in ERROR_CODES


# message words that a message-reading mapper would act on: known codes, secrets, free text.
# "prompt is too long" is left out: the anthropic provider maps it to a different machine code.
_WORDS = [*ERROR_CODES, "invalid_request_error", "rate_limit_error", "overloaded_error", "429", "timeout",
          SECRET, "sk-ant-api03-AAAA", "Bearer eyJhbGciOi", "password=hunter2", "", " ", "\n", "é", "😀"]


@pytest.mark.parametrize("make", [
    pytest.param(lambda m: CodexStreamError(m, code="invalid_request_error"), id="codex code"),
    pytest.param(lambda m: CodexStreamError(m, code="not_a_known_code", retry_reason="nor_this"), id="unknown classifier code"),
    pytest.param(lambda m: AnthropicStreamError(m, code="overloaded_error", retry_reason="overloaded_error"), id="anthropic code"),
    pytest.param(lambda m: AnthropicHTTPError(m, status_code=418), id="unmapped status"),
    pytest.param(lambda m: OllamaError(m, status_code=404), id="ollama status"),
    pytest.param(lambda m: ErrorInfo("backend_error", m), id="ErrorInfo unknown code"),
    pytest.param(lambda m: RuntimeError(m), id="no machine fields"),
])
def test_error_code_ignores_message_text(make) -> None:
    rng = random.Random(1445)
    base = make("")
    fields = lambda e: (getattr(e, "code", None), getattr(e, "retry_reason", None), getattr(e, "status_code", None))
    for _ in range(200):
        error = make(" ".join(rng.choice(_WORDS) for _ in range(rng.randint(1, 8))))
        assert fields(error) == fields(base)
        assert error_code(error) == error_code(base), f"message changed the code: {error!s}"
        assert error_code(error) in ERROR_CODES


class _Recorder(SpanHandle):
    def __init__(self) -> None:
        super().__init__()
        self.calls: dict[str, dict] = {}

    def set_usage(self, **values) -> None:
        self.calls["usage"] = values

    def set_response(self, **values) -> None:
        self.calls["response"] = values

    def fail(self, error, *, code=None) -> None:
        self.calls["fail"] = {"error": error, "code": code}


def _observe(monkeypatch: pytest.MonkeyPatch, events: list[tuple[int, StreamEvent]]) -> dict[str, dict]:
    """Run ``events`` through one observation; each pairs the monotonic clock (ns) with an event."""
    now = [1_000]
    monkeypatch.setattr(_completion, "time", SimpleNamespace(monotonic_ns=lambda: now[0]))
    span = _Recorder()
    with InferenceObservation() as inference:
        inference.start(span)
        for at, event in events:
            now[0] = at
            inference.observe(event)
    return span.calls


def test_ttft_is_set_once_at_the_first_model_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    retry = StreamEvent(StreamEventType.RETRY, data={"reason": "overloaded_error", "attempt": 1, "delay": 0.5})
    calls = _observe(monkeypatch, [
        (1_100, retry),  # a retry notice is not model progress
        (1_250, StreamEvent(StreamEventType.MESSAGE_START, data={"model": "qwen3:4b"})),
        (1_900, StreamEvent(StreamEventType.MESSAGE_UPDATE, delta="hi")),
        (2_000, StreamEvent(StreamEventType.MESSAGE_END, data={"usage": {"input_tokens": 7, "output_tokens": 2}})),
    ])

    assert calls["response"]["ttft_ns"] == 250
    assert calls["response"]["retry_count"] == 1
    # the provider reported no cache counts: absent, never a guessed 0
    assert calls["usage"] == {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9, "cache_read_tokens": None, "cache_write_tokens": None}


def test_no_model_progress_leaves_ttft_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _observe(monkeypatch, [(1_500, StreamEvent(StreamEventType.ERROR, error=ErrorInfo("max_tokens", SECRET)))])
    assert calls["response"]["ttft_ns"] is None
    assert calls["response"]["retry_count"] == 0


def test_reported_cache_counts_sum_across_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    failed = {"input_tokens": 1, "cache_read_input_tokens": 10, "output_tokens": 0}
    final = {"input_tokens": 2, "cache_read_input_tokens": 20, "cache_creation_input_tokens": 5, "output_tokens": 3}
    calls = _observe(monkeypatch, [
        (1_100, StreamEvent(StreamEventType.RETRY, data={"reason": "stream_error", "usage": failed})),
        (1_200, StreamEvent(StreamEventType.MESSAGE_END, data={"usage": final})),
    ])
    assert calls["usage"] == {"input_tokens": 38, "output_tokens": 3, "total_tokens": 41, "cache_read_tokens": 30, "cache_write_tokens": 5}
