"""Shared format dispatch and atomic publication for project-memory versions."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from zeta.core.session_files import atomic_publish_file
from zeta.project_errors import ProjectRegistryError, UnsupportedMemoryFormatError

CURRENT_POINTER = "memory-current.json"
UNSUPPORTED_FORMAT_2 = (
    "project memory uses format 2; this operation requires the entry-memory interface"
)

@dataclass(frozen=True, slots=True)
class PublicationContext:
    """History facts supplied to one format payload adapter."""

    version: str
    old_history: tuple[str, ...]
    retained_history: tuple[str, ...]
    dropped_history: tuple[str, ...]
    old_manifests: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class PreparedVersion[T]:
    """Format-owned payloads and manifest fields for one publication."""

    snapshot_payloads: Mapping[str, bytes]
    before_payloads: Mapping[str, bytes]
    manifest_fields: Mapping[str, object]
    value: T
    scalar_payload: bool = False


@dataclass(frozen=True, slots=True)
class PublishedVersion[T]:
    """The format result plus identifiers assigned by the shared publisher."""

    value: T
    snapshot: object
    version: str


class PayloadAdapter[T](Protocol):
    """Prepare format-specific payloads and manifest fields."""

    def prepare(self, context: PublicationContext) -> PreparedVersion[T]: ...


def stored_memory_format(
    directory_fd: int,
    *,
    pointer_reader: Callable[[int], dict[str, object] | None],
    version_handles: Callable[[int, bool], tuple[int, int, int]],
    manifest_reader: Callable[[int, str], dict[str, object]],
) -> int:
    """Return the authoritative format without interpreting its payload."""
    pointer = pointer_reader(directory_fd)
    if pointer is None:
        return 1
    root_fd, blobs_fd, versions_fd = version_handles(directory_fd, False)
    try:
        manifest = manifest_reader(versions_fd, str(pointer["current"]))
    finally:
        os.close(versions_fd)
        os.close(blobs_fd)
        os.close(root_fd)
    value = manifest.get("format", 1)
    if type(value) is not int or value < 1:
        raise ProjectRegistryError("project memory format is malformed")
    return value


def require_memory_format(
    directory_fd: int,
    expected: int,
    *,
    pointer_reader: Callable[[int], dict[str, object] | None],
    version_handles: Callable[[int, bool], tuple[int, int, int]],
    manifest_reader: Callable[[int, str], dict[str, object]],
) -> None:
    """Fail before a caller reads or writes a payload of another format."""
    actual = stored_memory_format(
        directory_fd,
        pointer_reader=pointer_reader,
        version_handles=version_handles,
        manifest_reader=manifest_reader,
    )
    if actual == expected:
        return
    if actual == 2 and expected == 1:
        raise UnsupportedMemoryFormatError(UNSUPPORTED_FORMAT_2)
    raise UnsupportedMemoryFormatError(
        f"project memory uses format {actual}, which this Zeta version does not support"
    )


def _publish_payloads(
    blobs_fd: int, payloads: Mapping[str, bytes], *, scalar: bool
) -> object:
    digests: dict[str, str] = {}
    for name, payload in payloads.items():
        digest = hashlib.sha256(payload).hexdigest()
        digests[name] = digest
        try:
            os.stat(digest, dir_fd=blobs_fd, follow_symlinks=False)
        except FileNotFoundError:
            atomic_publish_file(blobs_fd, digest, payload)
    if scalar:
        if len(digests) != 1:
            raise AssertionError("scalar project-memory payload must contain one blob")
        return next(iter(digests.values()))
    return digests


def publish_version[T](
    directory_fd: int,
    *,
    adapter: PayloadAdapter[T],
    retention_limit: int,
    reset_history: bool,
    pointer_reader: Callable[[int], dict[str, object] | None],
    version_handles: Callable[[int, bool], tuple[int, int, int]],
    manifest_reader: Callable[[int, str], dict[str, object]],
    transaction_step: Callable[[str], None],
    prune_versions: Callable[[int, int, set[str]], None],
) -> PublishedVersion[T]:
    """Publish one immutable version through the format-independent protocol."""
    root_fd, blobs_fd, versions_fd = version_handles(directory_fd, True)
    try:
        pointer = pointer_reader(directory_fd)
        old_history = (
            [] if reset_history or pointer is None else list(pointer["history"])
        )
        old_manifests = [manifest_reader(versions_fd, item) for item in old_history]
        version = uuid.uuid4().hex
        retained = [*old_history, version][-retention_limit:]
        dropped = old_history[: max(0, len(old_history) + 1 - retention_limit)]
        prepared = adapter.prepare(
            PublicationContext(
                version,
                tuple(old_history),
                tuple(retained),
                tuple(dropped),
                tuple(old_manifests),
            )
        )
        snapshot = _publish_payloads(
            blobs_fd, prepared.snapshot_payloads, scalar=prepared.scalar_payload
        )
        before_snapshot = _publish_payloads(
            blobs_fd, prepared.before_payloads, scalar=prepared.scalar_payload
        )
        os.fsync(blobs_fd)
        transaction_step("snapshot")

        manifest: dict[str, object] = {
            "version": version,
            **prepared.manifest_fields,
            "snapshot": snapshot,
            "before_snapshot": before_snapshot,
        }
        atomic_publish_file(
            versions_fd,
            f"{version}.json",
            json.dumps(manifest, sort_keys=True).encode(),
        )
        os.fsync(versions_fd)
        transaction_step("manifest")
        atomic_publish_file(
            directory_fd,
            CURRENT_POINTER,
            json.dumps(
                {"current": version, "history": retained}, sort_keys=True
            ).encode(),
            sync_directory=True,
        )
        transaction_step("publish")
        prune_versions(blobs_fd, versions_fd, set(retained))
        return PublishedVersion(prepared.value, snapshot, version)
    finally:
        os.close(versions_fd)
        os.close(blobs_fd)
        os.close(root_fd)
