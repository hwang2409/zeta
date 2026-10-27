"""Token estimation and calibration for one session's provider requests."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ...context_calibration import ContextCalibration
from ...protocol.types import Message, MessageRole, TextContent


class ContextRequestBudget:
    """Own calibrated request estimates and all provider-derived limits."""

    def __init__(
        self,
        entries: Sequence[object],
        *,
        provider_limit: int,
        safety_margin: float,
        normal_target_ratio: float,
        emergency_target_ratio: float,
        token_counter: Callable[[Message], int],
    ) -> None:
        self.calibration = ContextCalibration(
            entries,
            provider_limit=provider_limit,
            safety_margin=safety_margin,
            normal_target_ratio=normal_target_ratio,
            emergency_target_ratio=emergency_target_ratio,
        )
        self.provider_budgets = self.calibration.provider_budgets
        self.provider_attempt_estimate: int | None = None
        self._token_counter = token_counter

    @property
    def ratio(self) -> float:
        return self.calibration.ratio

    @property
    def trigger(self) -> int:
        return self.provider_budgets.trigger

    def estimator_budget(self, provider_limit: int) -> int:
        return self.calibration.estimator_budget(provider_limit)

    def update_provider_limit(self, provider_limit: int) -> None:
        self.calibration.update_provider_limit(provider_limit)
        self.provider_budgets = self.calibration.provider_budgets

    def count_messages(self, messages: Sequence[Message]) -> int:
        return sum(self._token_counter(message) for message in messages)

    def estimate_request(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[Mapping[str, Any]],
    ) -> int:
        estimate = self.count_messages(messages)
        if not tool_schemas:
            return estimate
        tools = Message(
            MessageRole.SYSTEM,
            [
                TextContent(
                    json.dumps(
                        {"tools": list(tool_schemas)},
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                )
            ],
        )
        return estimate + self._token_counter(tools)

    def estimate_tool_schemas(
        self, tool_schemas: Sequence[Mapping[str, Any]]
    ) -> int:
        return 0 if not tool_schemas else self.estimate_request((), tool_schemas)

    def notification_budget(self, provider_limit: int) -> int:
        normal_target = self.calibration.estimator_budget(
            self.provider_budgets.normal_target
        )
        return min(
            8_192,
            max(512, int(0.02 * normal_target)),
            max(0, self.estimator_budget(provider_limit) - normal_target),
        )

    def begin_attempt(
        self, messages: Sequence[Message], estimated_tokens: int | None
    ) -> None:
        self.provider_attempt_estimate = (
            self.count_messages(messages)
            if estimated_tokens is None
            else estimated_tokens
        )

    def record_overflow(self, provider_prompt_tokens: int | None) -> None:
        estimate = self.provider_attempt_estimate
        if provider_prompt_tokens is not None and estimate is not None:
            self.provider_budgets = self.calibration.calibrate(
                estimate, provider_prompt_tokens
            )

    def completion_metadata(self, usage: Mapping[str, Any]) -> dict[str, Any]:
        return self.calibration.completion_metadata(
            usage, self.provider_attempt_estimate
        )

    def record_usage(
        self,
        usage: Mapping[str, Any],
        fallback_estimate: int | None,
    ) -> None:
        provider_tokens = self.calibration.provider_input_tokens(usage)
        estimate = (
            fallback_estimate
            if self.provider_attempt_estimate is None
            else self.provider_attempt_estimate
        )
        if provider_tokens is not None and estimate is not None:
            self.provider_budgets = self.calibration.calibrate(
                estimate, provider_tokens
            )

    def reset_provider_total(self) -> None:
        self.calibration.reset_provider_total()

    def calibrated_tokens(self, estimator_tokens: int) -> int:
        return self.calibration.calibrated_tokens(estimator_tokens)

    def target_budget(self, *, emergency: bool) -> int:
        target = (
            self.provider_budgets.emergency_target
            if emergency
            else self.provider_budgets.normal_target
        )
        return self.calibration.estimator_budget(target)
