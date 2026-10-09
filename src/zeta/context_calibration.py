"""Deterministic provider-input calibration for context estimates."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .core.store import ConversationEntry

_ALPHA = 0.5
_MIN_RATIO = 1.0
_MAX_RATIO = 2.0


@dataclass(frozen=True, slots=True)
class ProviderContextBudgets:
    """Provider-space limits derived from one exact request measurement."""

    trigger: int
    normal_target: int
    emergency_target: int


class ContextCalibration:
    """Smooth exact provider-request measurements into bounded context budgets."""

    def __init__(
        self,
        entries: Sequence[ConversationEntry],
        *,
        provider_limit: int,
        safety_margin: float,
        normal_target_ratio: float,
        emergency_target_ratio: float,
    ) -> None:
        self._provider_limit = provider_limit
        self._safety_margin = safety_margin
        self._normal_target_ratio = normal_target_ratio
        self._emergency_target_ratio = emergency_target_ratio
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
                    self._observe(estimate, actual)

    @staticmethod
    def provider_input_tokens(usage: Mapping[str, Any]) -> int | None:
        values = [
            usage.get("input_tokens", usage.get("prompt_tokens")),
            usage.get("cache_read_input_tokens"),
            usage.get("cache_creation_input_tokens"),
        ]
        known = [value for value in values if type(value) is int and value >= 0]
        return sum(known) if known else None

    def calibrate(
        self,
        estimated_prompt_tokens: int,
        provider_prompt_tokens: int,
    ) -> ProviderContextBudgets:
        """Record one exact request pair and return all provider-space budgets."""

        self._observe(estimated_prompt_tokens, provider_prompt_tokens)
        self.provider_token_total = provider_prompt_tokens
        return self.provider_budgets

    @property
    def provider_budgets(self) -> ProviderContextBudgets:
        return ProviderContextBudgets(
            trigger=max(
                1, int(self._provider_limit * (1 - self._safety_margin))
            ),
            normal_target=max(
                1, int(self._provider_limit * self._normal_target_ratio)
            ),
            emergency_target=max(
                1, int(self._provider_limit * self._emergency_target_ratio)
            ),
        )

    def _observe(self, estimated: int, actual: int) -> None:
        if actual <= 0 or estimated <= 0:
            return
        observed = min(_MAX_RATIO, max(_MIN_RATIO, actual / estimated))
        self.ratio = (
            observed
            if not self.has_measurement
            else _ALPHA * observed + (1 - _ALPHA) * self.ratio
        )
        self.has_measurement = True

    def calibrated_tokens(self, estimated: int) -> int:
        calibrated = math.ceil(estimated * self.ratio)
        if self.provider_token_total is None:
            return calibrated
        return max(calibrated, self.provider_token_total)

    def estimator_budget(self, provider_budget: int) -> int:
        return max(1, math.floor(provider_budget / self.ratio))

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
