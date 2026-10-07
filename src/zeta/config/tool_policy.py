"""Action-aware tool capability policy."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any, Protocol

_GLOB_CHARACTERS = frozenset("*?[")


class ResolvedToolCapability(Protocol):
    tool: str
    action: str | None


@dataclass(frozen=True, slots=True)
class ToolSelector:
    """A tool-name pattern, optionally narrowed to one exact action."""

    name: str
    action: str | None = None

    def __str__(self) -> str:
        if self.action is None:
            return self.name
        return f"{self.name}({self.action})"

    @property
    def exact(self) -> bool:
        return not any(character in self.name for character in _GLOB_CHARACTERS)

    def matches(self, name: str, action: str | None) -> bool:
        return fnmatchcase(name, self.name) and (
            self.action is None or self.action == action
        )


def parse_tool_selector(text: str) -> ToolSelector:
    """Parse ``name`` or ``name(action)`` capability syntax."""

    if type(text) is not str or not text:
        raise ValueError(f"invalid tool selector {text!r}: expected a nonempty string")
    name = text
    action: str | None = None
    if "(" in text or ")" in text:
        if not text.endswith(")") or text.count("(") != 1 or text.count(")") != 1:
            raise ValueError(
                f"invalid tool selector {text!r}: expected 'name' or 'name(action)'"
            )
        name, action = text[:-1].split("(", 1)
        if (
            not action
            or any(character.isspace() for character in action)
            or any(character in action for character in "|*?[")
        ):
            raise ValueError(
                f"invalid tool selector {text!r}: action must be one exact value"
            )
    if not name or any(character.isspace() for character in name) or ")" in name:
        raise ValueError(
            f"invalid tool selector {text!r}: tool name must be nonempty with no whitespace"
        )
    return ToolSelector(name, action)


def validate_tool_patterns(
    value: Sequence[str] | None,
    *,
    field: str,
) -> tuple[str, ...] | None:
    """Return normalized selectors while preserving order."""

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
        normalized = str(parse_tool_selector(pattern))
        if normalized not in patterns:
            patterns.append(normalized)
    return tuple(patterns)


def parse_tool_patterns(value: str | None, *, field: str) -> tuple[str, ...] | None:
    """Parse one comma-separated CLI option."""

    if value is None:
        return None
    return validate_tool_patterns(value.split(","), field=field)


def is_exact_tool_name(pattern: str) -> bool:
    return parse_tool_selector(pattern).exact


def _selectors(patterns: Sequence[str]) -> tuple[ToolSelector, ...]:
    return tuple(parse_tool_selector(pattern) for pattern in patterns)


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    """Allow capabilities that pass every allow layer and no deny selector."""

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

    def _allows_capability(self, name: str, action: str | None) -> bool:
        allowed = all(
            any(selector.matches(name, action) for selector in _selectors(layer))
            for layer in self.allow_layers
        )
        return allowed and not any(
            selector.matches(name, action) for selector in _selectors(self.deny)
        )

    def allows(self, name: str) -> bool:
        """Compatibility spelling for a tool with no action metadata."""

        return self._allows_capability(name, None)

    def allows_tool(self, name: str, actions: Sequence[str] | None = None) -> bool:
        """Return whether at least one capability of a tool remains visible."""

        if actions is None:
            return self._allows_capability(name, None)
        return any(self._allows_capability(name, action) for action in actions)

    def allows_call(self, capability: ResolvedToolCapability) -> bool:
        """Authorize one registry-resolved call capability."""

        return self._allows_capability(capability.tool, capability.action)

    def filter_schema(self, schema: Mapping[str, Any]) -> dict[str, Any] | None:
        """Copy a schema and narrow its root ``action.enum`` when present."""

        name = schema.get("name")
        if not isinstance(name, str):
            return None
        parameters = schema.get("parameters")
        action_values: list[str] | None = None
        if isinstance(parameters, Mapping):
            properties = parameters.get("properties")
            if isinstance(properties, Mapping):
                action_schema = properties.get("action")
                if isinstance(action_schema, Mapping):
                    enum = action_schema.get("enum")
                    if isinstance(enum, list) and all(
                        isinstance(value, str) for value in enum
                    ):
                        action_values = enum
        if action_values is None:
            return copy.deepcopy(schema) if self.allows_tool(name) else None
        allowed = [
            action
            for action in action_values
            if self._allows_capability(name, action)
        ]
        if not allowed:
            return None
        filtered = copy.deepcopy(schema)
        filtered["parameters"]["properties"]["action"]["enum"] = allowed
        return filtered

    def allows_mcp_server(self, server: str) -> bool:
        """Return whether a server namespace can contain an allowed tool."""

        prefix = f"{server}__"
        layers = tuple(tuple(_selectors(layer)) for layer in self.allow_layers)
        if not all(
            any(
                selector.action is None
                and _pattern_may_match_namespace(selector.name, prefix)
                for selector in layer
            )
            for layer in layers
        ):
            return False

        exact_candidate_layers = tuple(
            tuple(
                selector.name
                for selector in layer
                if selector.action is None
                and selector.exact
                and selector.name.startswith(prefix)
            )
            for layer in layers
            if all(selector.exact for selector in layer)
        )
        if exact_candidate_layers:
            candidates = set(exact_candidate_layers[0])
            for layer in exact_candidate_layers[1:]:
                candidates.intersection_update(layer)
            return any(self._allows_capability(candidate, None) for candidate in candidates)

        namespace_denied = any(
            selector.action is None
            and selector.name.endswith("*")
            and fnmatchcase(prefix, selector.name)
            for selector in _selectors(self.deny)
        )
        return not namespace_denied

    @property
    def exact_allow_names(self) -> tuple[str, ...]:
        """Return exact capability selectors requested by allowlist layers."""

        return tuple(
            dict.fromkeys(
                str(selector)
                for layer in self.allow_layers
                for selector in _selectors(layer)
                if selector.exact
            )
        )

    @property
    def restricted(self) -> bool:
        return bool(self.allow_layers or self.deny)

    @property
    def required_exact_names(self) -> tuple[str, ...]:
        return tuple(
            selector
            for selector in self.exact_allow_names
            if self._selector_survives(selector)
        )

    def _selector_survives(self, text: str) -> bool:
        selector = parse_tool_selector(text)
        if selector.action is None:
            return self.allows(selector.name)
        return self._allows_capability(selector.name, selector.action)


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
