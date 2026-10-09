"""Deterministic provider-input calibration for context estimates."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .core.store import ConversationEntry

_ALPHA = 0.5
_MIN_RATIO = 1.0
_MAX_RATIO = 2.0
_PROVIDER_SIZE = re.compile(r"([0-9][0-9,]*)\s+tokens?", re.IGNORECASE)


class ContextCalibration:
    """Smooth provider measurements and persist a bounded session ratio."""

    def __init__(self, entries: Sequence[ConversationEntry]) -> None:
        self.ratio = 1.0
        self.has_measurement = False
        self.provider_token_total: int | None = None
        for entry in entries:
            if entry.type != "message":
                continue
            metadata = entry.data.get("message", {}).get("metadata", {})
            persisted = metadata.get("context_calibration_ratio")
            if type(persisted) in {int, float}:
                self.ratio = min(_MAX_RATIO, max(_MIN_RATIO, float(persisted)))
                self.has_measurement = True
                continue
            usage = metadata.get("provider_usage")
            estimate = metadata.get("context_estimate_tokens")
            if isinstance(usage, Mapping) and type(estimate) is int:
                actual = self.provider_input_tokens(usage)
                if actual is not None:
                    self.observe(actual, estimate)

    @staticmethod
    def provider_input_tokens(usage: Mapping[str, Any]) -> int | None:
        values = [
            usage.get("input_tokens", usage.get("prompt_tokens")),
            usage.get("cache_read_input_tokens"),
            usage.get("cache_creation_input_tokens"),
        ]
        known = [value for value in values if type(value) is int and value >= 0]
        return sum(known) if known else None

    def observe(self, actual: int, estimated: int) -> None:
        if actual <= 0 or estimated <= 0:
            return
        observed = min(_MAX_RATIO, max(_MIN_RATIO, actual / estimated))
        self.ratio = (
            observed
            if not self.has_measurement
            else _ALPHA * observed + (1 - _ALPHA) * self.ratio
        )
        self.has_measurement = True

    def observe_usage(self, usage: Mapping[str, Any], estimated: int | None) -> None:
        actual = self.provider_input_tokens(usage)
        if actual is not None and estimated is not None:
            self.observe(actual, estimated)

    def observe_overflow(self, message: str, estimated: int | None) -> int | None:
        match = _PROVIDER_SIZE.search(message)
        if match is None:
            return None
        actual = int(match.group(1).replace(",", ""))
        if estimated is not None:
            self.observe(actual, estimated)
        self.provider_token_total = actual
        return actual

    def calibrated_tokens(self, estimated: int) -> int:
        calibrated = math.ceil(estimated * self.ratio)
        if self.provider_token_total is None:
            return calibrated
        return max(calibrated, self.provider_token_total)

    def reset_provider_total(self) -> None:
        self.provider_token_total = None

    def completion_metadata(
        self, usage: Mapping[str, Any], estimated: int | None
    ) -> dict[str, Any]:
        persisted_usage = {
            key: value
            for key in (
                "input_tokens",
                "prompt_tokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
            )
            if type(value := usage.get(key)) is int and value >= 0
        }
        metadata: dict[str, Any] = {"context_calibration_ratio": self.ratio}
        if persisted_usage:
            metadata["provider_usage"] = persisted_usage
        if estimated is not None:
            metadata["context_estimate_tokens"] = estimated
        return metadata
