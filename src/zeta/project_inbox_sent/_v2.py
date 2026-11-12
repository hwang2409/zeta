"""Strict schema-v2 sent-ledger shapes: durable sender intent.

A v2 record carries the full recipient envelope so a crash between the durable
sender intent and the recipient publish recovers to exactly one delivered
message. Large bodies live in a sender-owned spill file until publication
converts the reference to the ordinary recipient body reference.

Every decoder here rejects unknown keys through the stored-format registry, so
an older reader never silently drops a field a newer writer added.
"""

from __future__ import annotations

import re
from typing import Any

from ..storage_format import reject_unknown_keys, require_closed_variant

V2_SCHEMA_VERSION = 2
# Sender-owned spill directory inside the sender session directory.
SENDER_SPILL_DIR = "project-inbox-sent-spill"

_ID = re.compile(r"[0-9a-f]{32}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _hex_id(value: object, field: str) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ValueError(f"invalid {field}")
    return value


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{field} must be a valid string")
    return value


def sender_spill_name(message_id: str) -> str:
    """The exact spill path for one message's sender-held body."""
    return f"{SENDER_SPILL_DIR}/{_hex_id(message_id, 'message id')}.txt"


def recipient_body_name(message_id: str) -> str:
    """The exact recipient body reference for one message."""
    return f"bodies/{_hex_id(message_id, 'message id')}.txt"


def decode_recipient_body_ref(value: object, message_id: str | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("invalid recipient body reference")
    reject_unknown_keys(value, "sent_message_ledger.recipient_body_ref")
    file = value.get("file")
    if not isinstance(file, str) or not file:
        raise ValueError("invalid recipient body reference")
    if message_id is not None and file != recipient_body_name(message_id):
        raise ValueError("recipient body reference does not match its message id")
    return {"file": file}


def decode_sender_spill_ref(value: object, message_id: str | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("invalid sender spill reference")
    reject_unknown_keys(value, "sent_message_ledger.sender_spill_ref")
    file = value.get("file")
    length = value.get("length")
    sha256 = value.get("sha256")
    if not isinstance(file, str) or not file:
        raise ValueError("invalid sender spill reference")
    if type(length) is not int or length < 0:
        raise ValueError("sender spill length must be a non-negative integer")
    if not isinstance(sha256, str) or _SHA256.fullmatch(sha256) is None:
        raise ValueError("sender spill sha256 must be 64 lowercase hex characters")
    if message_id is not None and file != sender_spill_name(message_id):
        raise ValueError("sender spill reference does not match its message id")
    return {"file": file, "length": length, "sha256": sha256}


def decode_from(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("invalid sender identity")
    reject_unknown_keys(value, "sent_message_ledger.from")
    return {
        "project": _nonempty(value.get("project"), "from.project"),
        "session": _hex_id(value.get("session"), "from.session"),
    }


def decode_body(value: object, message_id: str | None = None) -> str | dict[str, Any]:
    """Resolve the three durable body forms: inline, recipient ref, spill ref."""
    if isinstance(value, str):
        if "\x00" in value:
            raise TypeError("invalid envelope body")
        return value
    if isinstance(value, dict):
        if "sha256" in value or "length" in value:
            return decode_sender_spill_ref(value, message_id)
        return decode_recipient_body_ref(value, message_id)
    raise ValueError("invalid envelope body")


def decode_envelope(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("invalid sender envelope")
    reject_unknown_keys(value, "sent_message_ledger.envelope")
    require_closed_variant("inbox.schema_version", value.get("schema_version"))
    message_id = _hex_id(value.get("id"), "envelope id")
    require_closed_variant("inbox.kind", value.get("kind"))
    envelope: dict[str, Any] = {
        "schema_version": value["schema_version"],
        "id": message_id,
        "from": decode_from(value.get("from")),
        "to_project": _nonempty(value.get("to_project"), "to_project"),
        "to_session": _opt_id(value.get("to_session"), "to_session"),
        "kind": value["kind"],
        "title": _nonempty(value.get("title"), "title"),
        "body": decode_body(value.get("body"), message_id),
        "in_reply_to": _opt_id(value.get("in_reply_to"), "in_reply_to"),
        "created_at": _nonempty(value.get("created_at"), "created_at"),
    }
    # At the durable-intent epoch (8) the delegation feature (7) is always on, so
    # a sender record always carries an immutable origin; the recipient publish
    # re-validates it.
    origin = value.get("origin")
    if origin not in {"agent", "user", "local"}:
        raise ValueError("invalid envelope origin")
    envelope["origin"] = origin
    return envelope


def decode_record(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("invalid v2 sent record")
    reject_unknown_keys(value, "sent_message_ledger.v2_record")
    message_id = _hex_id(value.get("id"), "message id")
    reported = value.get("reported")
    if not isinstance(reported, list) or len(set(reported)) != len(reported):
        raise ValueError("invalid v2 sent record")
    for status in reported:
        require_closed_variant("sent_message_ledger.reported_status", status)
    record: dict[str, Any] = {
        "id": message_id,
        "created_at": _nonempty(value.get("created_at"), "created_at"),
        "target_project": _nonempty(value.get("target_project"), "target_project"),
        "publication": require_closed_variant(
            "sent_message_ledger.publication", value.get("publication")
        ),
        "reported": list(reported),
    }
    if "envelope" in value:
        envelope = decode_envelope(value["envelope"])
        if envelope["id"] != message_id:
            raise ValueError("v2 record envelope id mismatch")
        if envelope["to_project"] != record["target_project"]:
            raise ValueError("v2 record target mismatch")
        record["envelope"] = envelope
    elif record["publication"] != "published":
        # A record without an envelope is only a lazily converted v1 row, which
        # is already published; a pending record must retain its envelope.
        raise ValueError("pending v2 record is missing its envelope")
    return record


def decode_root(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("invalid v2 sent ledger")
    reject_unknown_keys(value, "sent_message_ledger.v2_root")
    if value.get("schema_version") != V2_SCHEMA_VERSION:
        raise ValueError("v2 sent ledger schema_version must be 2")
    records = value.get("records")
    if not isinstance(records, list):
        raise TypeError("invalid v2 sent ledger")
    decoded = [decode_record(record) for record in records]
    if len({record["id"] for record in decoded}) != len(decoded):
        raise ValueError("duplicate v2 sent record id")
    return {"schema_version": V2_SCHEMA_VERSION, "records": decoded}


def _opt_id(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _hex_id(value, field)
