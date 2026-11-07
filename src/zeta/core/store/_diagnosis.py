"""Bounded corruption diagnoses for conversation transcripts."""

from __future__ import annotations

from typing import Literal

from ..checkpoints import ConversationIntegrityError

TranscriptFailureKind = Literal[
    "torn_tail",
    "invalid_row",
    "broken_links",
    "unsupported_format",
    "unrecoverable_header",
]


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
        source_device: int,
        source_inode: int,
        source_size: int,
        prefix_digest: str,
    ) -> None:
        self.kind = kind
        self.session_id = session_id
        self.row_number = row_number
        self.byte_offset = byte_offset
        self.valid_prefix_end = valid_prefix_end
        self.reason_code = reason_code
        self.repairable = repairable
        self.source_device = source_device
        self.source_inode = source_inode
        self.source_size = source_size
        self.prefix_digest = prefix_digest
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
            # Diagnosis-only PR: no repair request exists yet.
            "actions": [],
        }
