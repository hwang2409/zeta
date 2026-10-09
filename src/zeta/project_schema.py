"""Dependency-free project record and legacy-memory schema.

This module is also shipped verbatim with the SSH destination publisher, where
Zeta may not be installed. The installed sender or local destination performs
full format-2 decoding; the dependency-free remote publisher checks structure
and hashes so it publishes exactly the bytes that the sender validated.
"""

from __future__ import annotations

import datetime as dt
import os
import re
from pathlib import Path

SCHEMA_VERSION = 1
ID_PREFIX = "p_"
ID_HEX_LENGTH = 32
MAX_NAME_LENGTH = 128
MAX_SCOPE_LENGTH = 4096
MAX_RECORD_SIZE = 10_000_000
MAX_MEMORY_FILE_SIZE = 128 * 1024
PROJECT_ID_PATTERN = re.compile(r"p_[0-9a-f]{32}\Z")
_PROJECT_FIELDS = {
    "schema_version",
    "project_id",
    "name",
    "scope",
    "created_at",
    "updated_at",
    "canonical_integration_root",
}


def validate_project_record(
    value: dict[str, object], expected_id: str | None = None
) -> tuple[str, str, str, str, str, str | None]:
    """Validate one complete registry record using only the standard library."""

    if (
        not _PROJECT_FIELDS.issubset(value)
        or not set(value).issubset(_PROJECT_FIELDS | {"lanes"})
        or value.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("unknown or invalid project schema")
    project_id = validate_project_id(value["project_id"])
    if expected_id is not None and project_id != expected_id:
        raise ValueError("project ID does not match its path")
    name = validate_project_text(value["name"], "name", MAX_NAME_LENGTH)
    scope = validate_project_text(value["scope"], "scope", MAX_SCOPE_LENGTH)
    created = _validate_timestamp(value["created_at"], "created_at")
    updated = _validate_timestamp(value["updated_at"], "updated_at")
    root = validate_project_root(value["canonical_integration_root"])
    return project_id, name, scope, created, updated, root


def validate_project_id(value: object) -> str:
    if not isinstance(value, str) or not PROJECT_ID_PATTERN.fullmatch(value):
        raise ValueError("invalid project_id")
    return value


def validate_project_text(value: object, field: str, maximum: int) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(
            f"{field} must be a non-empty string of at most {maximum} characters"
        )
    if "\x00" in value or any(ord(character) < 32 for character in value):
        raise ValueError(f"{field} contains a control character")
    return value


def validate_project_root(value: object) -> str | None:
    if value is None:
        return None
    root = validate_project_text(value, "canonical_integration_root", MAX_SCOPE_LENGTH)
    path = Path(root)
    if not path.is_absolute() or ".." in path.parts or os.path.normpath(root) != root:
        raise ValueError(
            "canonical_integration_root must be an absolute normalized path"
        )
    return str(path.expanduser().resolve(strict=False))


def decode_legacy_memory(payload: bytes, name: str) -> str:
    """Apply the registry's size and UTF-8 rules to one legacy memory file."""

    if len(payload) > MAX_MEMORY_FILE_SIZE:
        raise ValueError(f"project memory file {name} is too large")
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"project memory file {name} is unreadable") from exc


def _validate_timestamp(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"invalid {field}")
    try:
        parsed = dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=dt.timezone.utc  # noqa: UP017 - remote Python can be 3.9
        )
    except ValueError as exc:
        raise ValueError(f"invalid {field}") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%S.%fZ") != value:
        raise ValueError(f"invalid {field}")
    return value
