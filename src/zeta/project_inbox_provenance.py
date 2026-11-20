"""Provenance projection for project inbox records."""

from __future__ import annotations

from typing import Any

from .project_registry import ProjectRegistry
from .storage_format import StorageFeature, feature_enabled


def delegation_enabled(registry: ProjectRegistry) -> bool:
    return feature_enabled(
        registry.observed_storage_epoch, StorageFeature.INBOX_DELEGATION
    )


def is_trusted(record: dict[str, Any]) -> bool:
    """The one trust rule for every inbox surface: only provenance grants work.

    Below the delegation epoch no record carries an authority, so every message
    is information there: local origin cannot tell a user from another agent.
    """
    return record.get("authority") == "trusted_assignment"


def decorate(
    registry: ProjectRegistry,
    project_id: str,
    record: dict[str, Any],
    delegations: set[tuple[str, str]],
) -> dict[str, Any]:
    if not delegation_enabled(registry):
        return dict(record)
    author = "user" if record.get("origin") == "user" else "agent"
    sender = record.get("from")
    source = sender.get("project") if isinstance(sender, dict) else None
    delegated = author == "agent" and isinstance(source, str) and (
        source, record.get("kind")
    ) in delegations
    result = dict(record)
    result["author"] = author
    result["authority"] = (
        "trusted_assignment" if author == "user" or delegated else "information"
    )
    return result
