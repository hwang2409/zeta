"""Validation for tool results and parameter schemas."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from typing import Any

from ..protocol.types import StructuredToolResult, validate_tool_content_block

MAX_STRUCTURED_CONTENT_DEPTH = 32


def validate_tool_result(result: object) -> StructuredToolResult:
    """Validate one complete MCP-compatible structured tool result."""

    if type(result) is not dict:
        raise ValueError("expected a structured result object")
    if any(type(key) is not str for key in result):
        raise ValueError("top-level keys must be strings")
    expected_keys = {"content", "isError", "structuredContent"}
    result_keys = set(result)
    if "content_blocks" in result_keys:
        raise ValueError("legacy content_blocks is not allowed")
    missing_keys = expected_keys - result_keys
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"missing top-level keys: {missing}")
    extra_keys = result_keys - expected_keys - {"isCanceled"}
    if extra_keys:
        extra = ", ".join(sorted(extra_keys))
        raise ValueError(f"unexpected top-level keys: {extra}")

    content = result["content"]
    if type(content) is not list:
        raise ValueError("content must be an array")
    is_error = result["isError"]
    if type(is_error) is not bool:
        raise ValueError("isError must be a boolean")
    is_canceled = result.get("isCanceled", False)
    if type(is_canceled) is not bool:
        raise ValueError("isCanceled must be a boolean")
    structured_content = result["structuredContent"]
    if structured_content is not None:
        if type(structured_content) is not dict:
            raise ValueError("structuredContent must be an object or null")
        _validate_structured_content(structured_content)

    normalized_content = [
        validate_tool_content_block(index, block) for index, block in enumerate(content)
    ]
    return {**result, "content": normalized_content}


def _validate_structured_content(value: object) -> None:
    pending: list[tuple[object, int, bool]] = [(value, 0, False)]
    active: set[int] = set()
    while pending:
        current, depth, leaving = pending.pop()
        if leaving:
            active.remove(id(current))
            continue
        if depth > MAX_STRUCTURED_CONTENT_DEPTH:
            raise ValueError(
                f"structuredContent depth > {MAX_STRUCTURED_CONTENT_DEPTH}"
            )
        if current is None or type(current) in {str, int, bool}:
            continue
        if type(current) is float:
            if not math.isfinite(current):
                raise ValueError("structuredContent must contain finite numbers")
            continue
        if type(current) not in {list, dict}:
            raise ValueError("structuredContent must contain JSON values")
        current_id = id(current)
        if current_id in active:
            raise ValueError("cyclic structuredContent")
        active.add(current_id)
        if type(current) is list:
            pending.append((current, depth, True))
            pending.extend((item, depth + 1, False) for item in reversed(current))
            continue
        items = list(current.items())
        pending.append((current, depth, True))
        for key, item in reversed(items):
            if type(key) is not str:
                raise ValueError("structuredContent object keys must be strings")
            pending.append((item, depth + 1, False))


def _normalize_schema(
    schema: Mapping[str, Any] | None,
    *,
    validate_definition: bool = True,
) -> dict[str, Any]:
    if schema is None:
        return {"type": "object", "properties": {}}
    if not isinstance(schema, Mapping):
        raise TypeError("tool parameter schema must be an object")
    try:
        normalized = copy.deepcopy(dict(schema))
    except Exception as exc:
        raise ValueError("schema must contain JSON data") from exc
    if validate_definition:
        _validate_schema_definition(normalized, "schema")
    _validate_json_data(normalized, "schema")
    try:
        json.dumps(normalized, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("schema must contain JSON data") from exc
    return normalized


def _validate_json_data(value: Any, path: str) -> None:
    if value is None or type(value) in {bool, float, int, str}:
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json_data(item, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"schema must contain JSON data at {path}")
            _validate_json_data(item, f"{path}.{key}")
        return
    raise ValueError(f"schema must contain JSON data at {path}")


def _validate_arguments(arguments: object, schema: Mapping[str, Any]) -> dict[str, Any]:
    if type(arguments) is not dict:
        raise ValueError("arguments must be an object")
    _validate_finite_numbers(arguments, "arguments")
    _validate_schema(arguments, schema, "arguments")
    return dict(arguments)


def _coerce_arguments(arguments: object) -> dict[str, Any]:
    if type(arguments) is not dict:
        raise ValueError("arguments must be an object")
    _validate_finite_numbers(arguments, "arguments")
    return dict(arguments)


def _validate_finite_numbers(value: Any, path: str) -> None:
    if type(value) is float and not math.isfinite(value):
        raise ValueError(f"{path} must contain only finite numbers")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _validate_finite_numbers(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_finite_numbers(child, f"{path}[{index}]")


def _validate_schema(value: Any, schema: Mapping[str, Any], path: str) -> None:
    expected_type = schema.get("type")
    if expected_type is not None and not _matches_type(value, expected_type):
        raise ValueError(f"{path} must be {expected_type}")
    if type(value) is float and not math.isfinite(value):
        raise ValueError(f"{path} must be finite")
    if "const" in schema and not _schema_equal(value, schema["const"]):
        raise ValueError(f"{path} must equal the declared constant")
    if "enum" in schema and not any(
        _schema_equal(value, option) for option in schema["enum"]
    ):
        raise ValueError(f"{path} is not an allowed value")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise ValueError(f"{path} is too short")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ValueError(f"{path} is too long")
    if type(value) in {int, float} and type(value) is not bool:
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{path} is below the minimum")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            raise ValueError(f"{path} is not above the exclusive minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{path} is above the maximum")
    if isinstance(value, Mapping):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                raise ValueError(f"{path}.{key} is required")
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                raise ValueError(
                    f"{path} has unexpected properties: {', '.join(extra)}"
                )
        for key, child_schema in properties.items():
            if key in value:
                _validate_schema(value[key], child_schema, f"{path}.{key}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise ValueError(f"{path} has too few items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ValueError(f"{path} has too many items")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                _validate_schema(item, item_schema, f"{path}[{index}]")


def _matches_type(value: Any, expected: object) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return type(value) is bool
    if expected == "integer":
        return type(value) is int
    if expected == "number":
        return type(value) in {int, float}
    if expected == "null":
        return value is None
    return False


def _schema_equal(left: Any, right: Any) -> bool:
    if type(left) is bool or type(right) is bool:
        return type(left) is type(right) and left == right
    if type(left) in {int, float} and type(right) in {int, float}:
        return left == right
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _schema_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _schema_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return type(left) is type(right) and left == right


_SCHEMA_KEYS = {
    "description",
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "minLength",
    "maxLength",
    "minimum",
    "exclusiveMinimum",
    "maximum",
    "minItems",
    "maxItems",
    "enum",
    "const",
}
_SCHEMA_TYPES = {"array", "boolean", "integer", "null", "number", "object", "string"}


def _validate_schema_definition(schema: Mapping[str, Any], path: str) -> None:
    unsupported = set(schema) - _SCHEMA_KEYS
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"unsupported schema keywords at {path}: {names}")
    expected_type = schema.get("type")
    if expected_type is not None and (
        type(expected_type) is not str or expected_type not in _SCHEMA_TYPES
    ):
        raise ValueError(f"unsupported schema type at {path}")
    description = schema.get("description")
    if description is not None and type(description) is not str:
        raise ValueError(f"schema description must be a string at {path}")
    properties = schema.get("properties")
    if properties is not None:
        if not isinstance(properties, Mapping):
            raise ValueError(f"schema properties must be an object at {path}")
        for name, child in properties.items():
            if type(name) is not str or not isinstance(child, Mapping):
                raise ValueError(f"invalid schema property at {path}")
            _validate_schema_definition(child, f"{path}.{name}")
    required = schema.get("required")
    if required is not None and (
        type(required) is not list or any(type(name) is not str for name in required)
    ):
        raise ValueError(f"schema required must be a string array at {path}")
    additional = schema.get("additionalProperties")
    if additional is not None and type(additional) is not bool:
        raise ValueError(f"schema additionalProperties must be boolean at {path}")
    items = schema.get("items")
    if items is not None:
        if not isinstance(items, Mapping):
            raise ValueError(f"schema items must be an object at {path}")
        _validate_schema_definition(items, f"{path}.items")
    enum = schema.get("enum")
    if enum is not None and type(enum) is not list:
        raise ValueError(f"schema enum must be an array at {path}")
    for key in (
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
    ):
        value = schema.get(key)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"schema {key} must be a nonnegative integer at {path}")
    for key in ("minimum", "exclusiveMinimum", "maximum"):
        value = schema.get(key)
        if value is not None and (
            type(value) not in {int, float} or type(value) is bool
        ):
            raise ValueError(f"schema {key} must be numeric at {path}")
    if expected_type == "object" and schema.get("items") is not None:
        raise ValueError(f"schema items is not valid for an object at {path}")
    if expected_type != "object" and schema.get("properties") is not None:
        raise ValueError(f"schema properties is only valid for an object at {path}")
    if expected_type != "array" and schema.get("items") is not None:
        raise ValueError(f"schema items is only valid for an array at {path}")
