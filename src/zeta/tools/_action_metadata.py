"""Registry-owned action validation and authorization metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any


class ApprovalBinding(StrEnum):
    """Stable object captured when a scoped child approval is granted."""

    NONE = "none"
    PATH = "path"
    CWD = "cwd"


@dataclass(frozen=True, slots=True)
class ResolvedCapability:
    """Authorization facts resolved from one registry tool call."""

    tool: str
    action: str | None
    requires_approval: bool
    subject_field: str | None
    subject_value: object
    binding: ApprovalBinding
    capability_class: str | None


@dataclass(frozen=True, slots=True)
class ToolAction:
    """Validation and authorization facts for one model-facing action."""

    required_fields: frozenset[str]
    allowed_fields: frozenset[str]
    requires_approval: bool
    capability_class: str
    approval_subject: str | None = None
    binding: ApprovalBinding = ApprovalBinding.NONE

    def __post_init__(self) -> None:
        object.__setattr__(self, "required_fields", frozenset(self.required_fields))
        object.__setattr__(self, "allowed_fields", frozenset(self.allowed_fields))
        fields = (*self.required_fields, *self.allowed_fields)
        if not self.capability_class or any(
            not isinstance(field, str) or not field for field in fields
        ):
            raise ValueError("action fields and capability class must be nonempty")


class UnknownToolAction(ValueError):
    """The call did not select an action declared by its tool."""


class InvalidActionArguments(ValueError):
    """The call mixed fields from incompatible actions."""


def normalize_actions(
    name: str,
    schema: Mapping[str, Any],
    actions: Mapping[str, ToolAction] | None,
) -> Mapping[str, ToolAction] | None:
    """Validate one action table and generate its provider action enum."""

    if actions is None:
        return None
    if not actions:
        raise ValueError(f"action metadata for tool {name!r} must not be empty")
    properties = schema.get("properties")
    action_schema = (
        properties.get("action") if isinstance(properties, Mapping) else None
    )
    if not isinstance(action_schema, dict) or action_schema.get("type") != "string":
        raise ValueError(f"action tool {name!r} must declare a string action property")
    for action in actions:
        if not action or any(character.isspace() for character in action):
            raise ValueError(f"invalid action name {action!r} for tool {name!r}")
    action_schema["enum"] = list(actions)
    if schema.get("required") != ["action"]:
        raise ValueError(
            f"action tool {name!r} must require only action in its provider schema"
        )
    field_names = (
        frozenset(properties) if isinstance(properties, Mapping) else frozenset()
    )
    for action, metadata in actions.items():
        _validate_action(name, action, metadata, field_names)
    return MappingProxyType(dict(actions))


def resolve_action(
    definition: Any, arguments: Mapping[str, object]
) -> tuple[str | None, ToolAction | None]:
    """Resolve and validate the action selected by one tool call."""

    action, metadata = select_action(definition, arguments)
    if metadata is None:
        return action, metadata
    supplied = frozenset(arguments)
    missing = metadata.required_fields - supplied
    disallowed = supplied - metadata.allowed_fields
    problems: list[str] = []
    if missing:
        problems.append(f"requires {', '.join(sorted(missing))}")
    if disallowed:
        problems.append(f"does not allow {', '.join(sorted(disallowed))}")
    if problems:
        raise InvalidActionArguments(
            f"{definition.name} action={action} {'; '.join(problems)}"
        )
    return action, metadata


def select_action(
    definition: Any, arguments: Mapping[str, object]
) -> tuple[str | None, ToolAction | None]:
    """Resolve action authorization metadata without validating action fields."""

    if definition.actions is None:
        return None, None
    action = arguments.get("action")
    if not isinstance(action, str) or action not in definition.actions:
        expected = ", ".join(definition.actions)
        detail = (
            f"unknown action {action!r}"
            if isinstance(action, str)
            else "action must be a string"
        )
        raise UnknownToolAction(
            f"{definition.name}: {detail}; expected one of: {expected}"
        )
    return action, definition.actions[action]


def resolve_capability(
    definition: Any, arguments: Mapping[str, object]
) -> ResolvedCapability:
    """Resolve authorization facts from one registered definition and call."""

    action, metadata = select_action(definition, arguments)
    subject_field = (
        definition.approval_subject if metadata is None else metadata.approval_subject
    )
    if definition.approval_subject_resolver is not None:
        subject_value = definition.approval_subject_resolver(arguments)
    else:
        subject_value = (
            None if subject_field is None else arguments.get(subject_field)
        )
    binding = ApprovalBinding.NONE
    if metadata is not None:
        binding = metadata.binding
    elif subject_field == "path":
        binding = ApprovalBinding.PATH
    elif subject_field == "command":
        binding = ApprovalBinding.CWD
    return ResolvedCapability(
        tool=definition.name,
        action=action,
        requires_approval=(
            definition.requires_approval
            if metadata is None
            else metadata.requires_approval
        ),
        subject_field=subject_field,
        subject_value=subject_value,
        binding=binding,
        capability_class=None if metadata is None else metadata.capability_class,
    )


def _validate_action(
    tool: str,
    action: str,
    metadata: ToolAction,
    field_names: frozenset[str],
) -> None:
    if not isinstance(metadata, ToolAction):
        raise TypeError(f"action {action!r} for tool {tool!r} must be ToolAction")
    if "action" not in metadata.allowed_fields:
        raise ValueError(f"action {action!r} must allow the action field")
    if not metadata.required_fields <= metadata.allowed_fields:
        raise ValueError(f"action {action!r} requires fields it does not allow")
    if not metadata.allowed_fields <= field_names:
        raise ValueError(f"action {action!r} allows fields absent from the schema")
    if metadata.approval_subject is not None and (
        metadata.approval_subject not in metadata.allowed_fields
        or metadata.approval_subject not in field_names
    ):
        raise ValueError(
            f"approval subject for {tool}({action}) must be an allowed field"
        )
    if metadata.binding is ApprovalBinding.PATH and metadata.approval_subject != "path":
        raise ValueError(f"path binding for {tool}({action}) requires path subject")
    if metadata.binding is ApprovalBinding.CWD and metadata.approval_subject != "command":
        raise ValueError(f"cwd binding for {tool}({action}) requires command subject")
