"""The built-in session todo-list tool."""

from __future__ import annotations

from typing import TypedDict

from ..core.todo import TODO_STATUSES, TodoItem, parse_todo_items, todo_counts
from ..types import StructuredContentValue, StructuredToolResult
from .registry import ToolRegistry, _success_result, text_block


class TodoArguments(TypedDict, total=False):
    items: list[TodoItem]


def _todo_result(items: list[TodoItem]) -> StructuredToolResult:
    counts = todo_counts(items)
    content = (
        "todo list: "
        f"{counts['pending']} pending, "
        f"{counts['in_progress']} in progress, "
        f"{counts['completed']} completed, "
        f"{counts['canceled']} canceled"
    )
    structured: dict[str, StructuredContentValue] = {
        "items": items,
        "counts": counts,
    }
    return _success_result(text_block(content), structured_content=structured)


def _todo_error(reason: str) -> StructuredToolResult:
    return {
        "content": [text_block(reason)],
        "isError": True,
        "structuredContent": {
            "error": {
                "kind": "invalid_arguments",
                "message": reason,
                "hint": (
                    "reread the todo schema and retry with a valid items array"
                ),
            },
        },
    }


async def _todo(
    registry: ToolRegistry,
    arguments: TodoArguments,
) -> StructuredToolResult:
    store = registry.todo_store
    unexpected = sorted(set(arguments) - {"items"})
    if unexpected:
        return _todo_error(
            "todo arguments contain only items; unexpected properties: "
            + ", ".join(unexpected)
        )
    if "items" not in arguments:
        return _todo_result(store.todo_items())
    try:
        items = parse_todo_items(arguments["items"])
    except ValueError as exc:
        return _todo_error(str(exc))
    store.set_todo_items(items)
    return _todo_result(items)


def register(registry: ToolRegistry) -> None:
    registry.register_session_tool(
        "todo",
        _todo,
        description=(
            "Read the current todo list when items is omitted. "
            "Write the full todo list by providing items. "
            "Use for multi-step tasks: mark one item in_progress before starting "
            "it and completed immediately after finishing it."
        ),
        parameters={
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": list(TODO_STATUSES),
                            },
                        },
                        "required": ["content", "status"],
                        "additionalProperties": False,
                    },
                },
            },
            "additionalProperties": False,
        },
        validate_arguments=False,
        requires_approval=False,
    )
