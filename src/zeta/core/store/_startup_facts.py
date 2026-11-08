"""Compact immutable facts read during runtime startup."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..checkpoints import ConversationEntry


@dataclass(frozen=True, slots=True)
class ContextCalibrationRecord:
    """One validated provider-estimate observation from a message."""

    ratio: float | None = None
    estimate: int | None = None
    actual: int | None = None


class StartupFactsMixin:
    """Expose compact immutable facts without transcript snapshots."""

    def sent_message_receipts(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._sent_message_receipts)

    def context_calibration_records(self) -> tuple[ContextCalibrationRecord, ...]:
        return tuple(
            record
            for entry in self._active_branch()
            if (record := context_calibration_record(entry)) is not None
        )


def context_calibration_record(
    entry: ConversationEntry,
) -> ContextCalibrationRecord | None:
    """Return the immutable calibration values read from one active message."""

    if entry.type != "message":
        return None
    metadata = entry.data.get("message", {}).get("metadata", {})
    if not isinstance(metadata, Mapping):
        return None
    persisted = metadata.get("context_calibration_ratio")
    if type(persisted) in {int, float}:
        return ContextCalibrationRecord(ratio=float(persisted))
    usage = metadata.get("provider_usage")
    estimate = metadata.get("context_estimate_tokens")
    if not isinstance(usage, Mapping) or type(estimate) is not int:
        return None
    actual = _provider_input_tokens(usage)
    if actual is None:
        return None
    return ContextCalibrationRecord(estimate=estimate, actual=actual)


def _provider_input_tokens(usage: Mapping[str, Any]) -> int | None:
    values = [
        usage.get("input_tokens", usage.get("prompt_tokens")),
        usage.get("cache_read_input_tokens"),
        usage.get("cache_creation_input_tokens"),
    ]
    known = [value for value in values if type(value) is int and value >= 0]
    return sum(known) if known else None
