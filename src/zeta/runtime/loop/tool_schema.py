"""Provider-visible tool schema canonicalization."""

from __future__ import annotations

from collections.abc import Sequence

from ...protocol.types import ToolSchema


def canonical_tool_schemas(schemas: Sequence[ToolSchema]) -> list[ToolSchema]:
    """Validate, deduplicate, and order schemas at the provider boundary."""

    names: list[str] = []
    for schema in schemas:
        name = schema.get("name")
        if type(name) is not str or not name:
            raise ValueError("provider-visible tool schema name must be a non-empty string")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("duplicate provider-visible tool schema name")
    return sorted(schemas, key=lambda schema: schema["name"])
