"""Typed transcript-load refusals raised by the store and served to clients."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ..checkpoints import ConversationIntegrityError

if TYPE_CHECKING:
    from . import TranscriptSourceFacts

TranscriptFailureKind = Literal[
    "torn_tail",
    "invalid_row",
    "broken_links",
    "unsupported_format",
    "unrecoverable_header",
]


class ObserverSnapshotChanged(ConversationIntegrityError):
    """The observer could not validate one authoritative source view."""


class TranscriptLoadError(ConversationIntegrityError):
    """A typed transcript refusal that never exposes persisted content or paths."""

    def __init__(
        self,
        *,
        kind: TranscriptFailureKind,
        session_id: str,
        row_number: int | None,
        byte_offset: int | None,
        valid_prefix_end: int | None,
        reason_code: str,
        repairable: bool,
        repair_available: bool,
        source_device: int,
        source_inode: int,
        source_size: int,
        prefix_digest: str,
        child_run_id: str | None = None,
        source_facts: TranscriptSourceFacts | None = None,
    ) -> None:
        self.kind = kind
        self.session_id = session_id
        self.row_number = row_number
        self.byte_offset = byte_offset
        self.valid_prefix_end = valid_prefix_end
        self.reason_code = reason_code
        self.repairable = repairable
        self.repair_available = repair_available
        self.source_device = source_device
        self.source_inode = source_inode
        self.source_size = source_size
        self.prefix_digest = prefix_digest
        self.child_run_id = child_run_id
        self._source_facts = source_facts
        location = f" at row {row_number}" if row_number is not None else ""
        detail = {
            "invalid_payload": "invalid payload for conversation entry",
            "missing_parent": "missing prior parent",
            "duplicate_id": "duplicate conversation id",
            "orphaned_root": "orphaned root",
            "invalid_sequence": "non-monotonic conversation sequence",
            "unlinked_approval_resolution": "approval resolution is not linked",
            "header_session_mismatch": "conversation header session id mismatch",
            "invalid_json": (
                "terminated invalid conversation row"
                if kind == "invalid_row"
                else kind.replace("_", " ")
            ),
        }.get(reason_code)
        label = detail or kind.replace("_", " ")
        super().__init__(f"transcript {label}{location} (session {session_id})")

    def protocol_data(self) -> dict[str, object]:
        """Return the bounded public diagnosis used by every serve error path."""
        return {
            "code": "corrupt_transcript",
            "kind": self.kind,
            "row": self.row_number,
            "repairable": self.repairable,
            "session_id": self.session_id,
            "actions": (
                [
                    {
                        "method": "repair_transcript",
                        "session_id": self.session_id,
                        **(
                            {"child_run_id": self.child_run_id}
                            if self.child_run_id is not None
                            else {}
                        ),
                        "requires_confirmation": True,
                    }
                ]
                if self.repair_available
                else []
            ),
        }
