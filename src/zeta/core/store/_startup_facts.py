"""compatibility imports for startup row facts."""

from ._rows import (
    ContextCalibrationRecord,
    context_calibration_record,
    sent_message_receipts,
)


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


__all__ = [
    "ContextCalibrationRecord",
    "StartupFactsMixin",
    "context_calibration_record",
    "sent_message_receipts",
]
