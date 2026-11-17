"""Parse and strictly validate a ``zeta-extension.toml`` manifest.

The manifest is untrusted outside input, so parsing is a trust boundary: a
typo or an unknown key is a loud error, not a silent drop. This module owns a
local strict validator and deliberately does NOT use
``storage_format.reject_unknown_keys`` -- that helper only accepts shapes that
are in the format-gate list, so calling it would force a ``WRITER_EPOCH`` bump
(see the extensions design, epoch impact). The manifest is a config file, not a
stored session payload, so it is validated here instead.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass

MANIFEST_FILENAME = "zeta-extension.toml"

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
# Keys implemented by the current extension loader.
_EXTENSION_KEYS = frozenset({"name", "version", "api_version", "entry", "description"})


class ExtensionManifestError(ValueError):
    """A ``zeta-extension.toml`` manifest is malformed or invalid."""


@dataclass(frozen=True, slots=True)
class ExtensionManifest:
    """A validated ``zeta-extension.toml``."""

    name: str
    version: str
    api_version: int
    entry_module: str
    entry_function: str
    description: str


def parse_manifest(text: str) -> ExtensionManifest:
    """Parse and validate manifest ``text``; raise on any problem."""

    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ExtensionManifestError(f"invalid TOML: {exc}") from exc
    unknown_top = set(data) - {"extension"}
    if unknown_top:
        raise ExtensionManifestError(
            f"unknown top-level table(s): {', '.join(sorted(unknown_top))}; "
            "the manifest has one [extension] table"
        )
    table = data.get("extension")
    if not isinstance(table, dict):
        raise ExtensionManifestError("missing [extension] table")
    unknown = set(table) - _EXTENSION_KEYS
    if unknown:
        raise ExtensionManifestError(
            f"unknown key(s) in [extension]: {', '.join(sorted(unknown))}"
        )

    name = _require_str(table, "name")
    if not _NAME_RE.match(name):
        raise ExtensionManifestError(
            f"name {name!r} must match {_NAME_RE.pattern} "
            "(lowercase letters, digits, hyphens)"
        )
    version = _require_str(table, "version")
    api_version = table.get("api_version")
    if type(api_version) is not int or api_version < 1:
        raise ExtensionManifestError("api_version must be an integer >= 1")
    entry_module, entry_function = _parse_entry(table.get("entry"))
    description = table.get("description", "")
    if not isinstance(description, str):
        raise ExtensionManifestError("description must be a string")
    return ExtensionManifest(
        name=name,
        version=version,
        api_version=api_version,
        entry_module=entry_module,
        entry_function=entry_function,
        description=description,
    )


def _require_str(table: dict[str, object], key: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise ExtensionManifestError(f"{key} must be a nonempty string")
    return value


def _parse_entry(raw: object) -> tuple[str, str]:
    if not isinstance(raw, str) or not raw:
        raise ExtensionManifestError(
            "entry must be a nonempty string 'module' or 'module:function'"
        )
    module, _, function = raw.partition(":")
    function = function or "activate"
    if not module.isidentifier():
        raise ExtensionManifestError(f"entry module {module!r} is not a valid identifier")
    if not function.isidentifier():
        raise ExtensionManifestError(
            f"entry function {function!r} is not a valid identifier"
        )
    return module, function


__all__ = [
    "MANIFEST_FILENAME",
    "ExtensionManifest",
    "ExtensionManifestError",
    "parse_manifest",
]
