"""Native zeta extensions: in-process Python loaded per session.

Slice 1 provides the loader, tool registration, and the author test kit. An
extension is a directory with a ``zeta-extension.toml`` manifest and a
synchronous ``activate(zx)`` entry function that declares tools on the
:class:`~zeta.extensions.api.ExtensionAPI`.
"""

from __future__ import annotations

from .api import EXTENSION_API_VERSION, ExtensionAPI, ExtensionError
from .discovery import apply_extensions, trust_project_extensions
from .host import (
    ExtensionHost,
    ExtensionStatus,
    LoadedExtension,
    PendingExtension,
)
from .trust import (
    ExtensionTrustError,
    fingerprint,
    remove,
    trust,
    untrust,
    update,
)
from .manifest import (
    ExtensionManifest,
    ExtensionManifestError,
    parse_manifest,
)

__all__ = [
    "EXTENSION_API_VERSION",
    "ExtensionAPI",
    "ExtensionError",
    "ExtensionTrustError",
    "fingerprint",
    "ExtensionHost",
    "ExtensionManifest",
    "ExtensionManifestError",
    "ExtensionStatus",
    "LoadedExtension",
    "PendingExtension",
    "apply_extensions",
    "parse_manifest",
    "trust_project_extensions",
    "remove",
    "trust",
    "untrust",
    "update",
]
