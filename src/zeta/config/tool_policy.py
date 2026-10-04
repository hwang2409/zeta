"""Name-based tool capability policy."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase

_GLOB_CHARACTERS = frozenset("*?[")


def validate_tool_patterns(
    value: Sequence[str] | None,
    *,
    field: str,
) -> tuple[str, ...] | None:
    """Return normalized patterns while preserving order."""

    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{field} must be a list of patterns")
    patterns: list[str] = []
    for item in value:
        if type(item) is not str or not item.strip():
            raise ValueError(f"{field} must contain nonempty strings")
        pattern = item.strip()
        if "," in pattern:
            raise ValueError(f"{field} patterns must not contain commas")
        if pattern not in patterns:
            patterns.append(pattern)
    return tuple(patterns)


def parse_tool_patterns(value: str | None, *, field: str) -> tuple[str, ...] | None:
    """Parse one comma-separated CLI option."""

    if value is None:
        return None
    return validate_tool_patterns(value.split(","), field=field)


def is_exact_tool_name(pattern: str) -> bool:
    return not any(character in pattern for character in _GLOB_CHARACTERS)


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    """Allow names that match the allowlist and do not match the denylist."""

    allow: tuple[str, ...] | None = None
    deny: tuple[str, ...] = ()

    @classmethod
    def create(
        cls,
        allow: Sequence[str] | None = None,
        deny: Sequence[str] = (),
    ) -> ToolPolicy:
        normalized_allow = validate_tool_patterns(allow, field="tools")
        normalized_deny = validate_tool_patterns(deny, field="disallowed_tools")
        assert normalized_deny is not None
        return cls(normalized_allow, normalized_deny)

    def allows(self, name: str) -> bool:
        allowed = self.allow is None or any(
            fnmatchcase(name, pattern) for pattern in self.allow
        )
        return allowed and not any(fnmatchcase(name, pattern) for pattern in self.deny)

    @property
    def required_exact_names(self) -> tuple[str, ...]:
        if self.allow is None:
            return ()
        return tuple(pattern for pattern in self.allow if is_exact_tool_name(pattern))
