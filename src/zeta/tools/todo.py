"""The built-in session todo-list tool."""

from __future__ import annotations

from typing import Literal, TypedDict

from ..core.todo import TodoItem, parse_todo_items, todo_counts
from ..types import StructuredContentValue, StructuredToolResult
from .registry import ToolRegistry, _success_result, text_block


class TodoArguments(TypedDict, total=False):
    action: Literal["read"]
    items: list[TodoItem]


def _todo_result(items: list[TodoItem]) -> StructuredToolResult:
    counts = todo_counts(items)
    content = (
        "todo list: "
        f"{counts['pending']} pending, "
        f"{counts['in_progress']} in progress, "
        f"{counts['completed']} completed"
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
        "structuredContent": {"error": reason},
    }


async def _todo(
    registry: ToolRegistry,
    arguments: TodoArguments,
) -> StructuredToolResult:
    store = registry.session_store
    if arguments.get("action") == "read":
        if "items" in arguments:
            return _todo_error("todo read action cannot include items")
        return _todo_result(store.todo_items())
    if "items" not in arguments:
        return _todo_result(store.todo_items())
    try:
        items = parse_todo_items(arguments["items"])
    except ValueError as exc:
        return _todo_error(str(exc))
    store.set_todo_items(items)
    return _todo_result(items)


def register(registry: ToolRegistry) -> None:
    registry.register(
        "todo",
        lambda arguments: _todo(registry, arguments),
        description=(
            "Use for multi-step tasks: mark one item in_progress before starting "
            "it and completed immediately after finishing it."
        ),
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["read"]},
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "status": {"type": "string"},
                        },
                        "required": ["content", "status"],
                        "additionalProperties": False,
                    },
                },
            },
            "additionalProperties": False,
        },
        requires_approval=False,
    )
