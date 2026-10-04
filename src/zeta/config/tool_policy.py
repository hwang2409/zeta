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
    allow_layers: tuple[tuple[str, ...], ...] = ()

    @classmethod
    def create(
        cls,
        allow: Sequence[str] | None = None,
        deny: Sequence[str] = (),
        *,
        allow_layers: Sequence[Sequence[str]] = (),
    ) -> ToolPolicy:
        normalized_allow = validate_tool_patterns(allow, field="tools")
        normalized_deny = validate_tool_patterns(deny, field="disallowed_tools")
        normalized_layers = tuple(
            dict.fromkeys(
                validate_tool_patterns(layer, field="tools") or ()
                for layer in allow_layers
            )
        )
        assert normalized_deny is not None
        if not normalized_layers and normalized_allow is not None:
            normalized_layers = (normalized_allow,)
        return cls(normalized_allow, normalized_deny, normalized_layers)

    def narrowed_by(self, upper_bound: ToolPolicy) -> ToolPolicy:
        """Intersect allowlists and union denylists without losing empty layers."""

        layers = tuple(dict.fromkeys((*self.allow_layers, *upper_bound.allow_layers)))
        allow = upper_bound.allow if upper_bound.allow is not None else self.allow
        deny = tuple(dict.fromkeys((*self.deny, *upper_bound.deny)))
        return ToolPolicy.create(allow, deny, allow_layers=layers)

    def allows(self, name: str) -> bool:
        allowed = all(
            any(fnmatchcase(name, pattern) for pattern in layer)
            for layer in self.allow_layers
        )
        return allowed and not any(fnmatchcase(name, pattern) for pattern in self.deny)

    def allows_mcp_server(self, server: str) -> bool:
        """Return whether a server namespace can contain an allowed tool."""

        prefix = f"{server}__"
        if not all(
            any(_pattern_may_match_namespace(pattern, prefix) for pattern in layer)
            for layer in self.allow_layers
        ):
            return False

        exact_candidate_layers = tuple(
            tuple(
                pattern
                for pattern in layer
                if is_exact_tool_name(pattern) and pattern.startswith(prefix)
            )
            for layer in self.allow_layers
            if all(is_exact_tool_name(pattern) for pattern in layer)
        )
        if exact_candidate_layers:
            candidates = set(exact_candidate_layers[0])
            for layer in exact_candidate_layers[1:]:
                candidates.intersection_update(layer)
            candidates = {
                candidate
                for candidate in candidates
                if all(
                    any(fnmatchcase(candidate, pattern) for pattern in layer)
                    for layer in self.allow_layers
                )
                and not any(fnmatchcase(candidate, pattern) for pattern in self.deny)
            }
            return bool(candidates)

        namespace_denied = any(
            pattern.endswith("*") and fnmatchcase(prefix, pattern)
            for pattern in self.deny
        )
        return not namespace_denied

    @property
    def exact_allow_names(self) -> tuple[str, ...]:
        """Return exact names requested by the policy's allowlist layers."""

        return tuple(
            dict.fromkeys(
                pattern
                for layer in self.allow_layers
                for pattern in layer
                if is_exact_tool_name(pattern)
            )
        )

    @property
    def restricted(self) -> bool:
        """Return whether this policy limits any tool capability."""

        return bool(self.allow_layers or self.deny)

    @property
    def required_exact_names(self) -> tuple[str, ...]:
        return tuple(name for name in self.exact_allow_names if self.allows(name))


def _pattern_may_match_namespace(pattern: str, prefix: str) -> bool:
    first_glob = min(
        (pattern.find(character) for character in _GLOB_CHARACTERS if character in pattern),
        default=-1,
    )
    if first_glob < 0:
        return pattern.startswith(prefix)
    fixed_prefix = pattern[:first_glob]
    if "__" not in fixed_prefix:
        return True
    return prefix.startswith(fixed_prefix) or fixed_prefix.startswith(prefix)
