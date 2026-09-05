"""Typed todo-list state and boundary validation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypedDict

TodoStatus = Literal["pending", "in_progress", "completed", "canceled"]


class TodoItem(TypedDict):
    content: str
    status: TodoStatus


TODO_STATUSES: tuple[TodoStatus, ...] = (
    "pending",
    "in_progress",
    "completed",
    "canceled",
)
MAX_TODO_ITEMS = 50
MAX_TODO_CONTENT_LENGTH = 500


def parse_todo_items(value: object) -> list[TodoItem]:
    """Validate JSON-backed todo items and return a detached normalized list."""

    if type(value) is not list:
        raise ValueError("todo items must be an array")
    if len(value) > MAX_TODO_ITEMS:
        raise ValueError(f"todo list cannot contain more than {MAX_TODO_ITEMS} items")

    normalized: list[TodoItem] = []
    for index, item in enumerate(value):
        if type(item) is not dict:
            raise ValueError(f"todo item {index} must be an object")
        if set(item) != {"content", "status"}:
            raise ValueError(f"todo item {index} must contain only content and status")
        content = item.get("content")
        if type(content) is not str or not content.strip():
            raise ValueError(f"todo item {index} content must be nonempty")
        if len(content) > MAX_TODO_CONTENT_LENGTH:
            raise ValueError(
                f"todo item {index} content cannot exceed "
                f"{MAX_TODO_CONTENT_LENGTH} characters"
            )
        status = item.get("status")
        if type(status) is not str:
            raise ValueError(f"todo item {index} status must be a string")
        if status not in TODO_STATUSES:
            raise ValueError(
                f"todo item {index} status must be one of: {', '.join(TODO_STATUSES)}"
            )
        normalized.append({"content": content, "status": status})

    return normalized


def todo_counts(items: Sequence[TodoItem]) -> dict[TodoStatus, int]:
    counts: dict[TodoStatus, int] = {
        "pending": 0,
        "in_progress": 0,
        "completed": 0,
        "canceled": 0,
    }
    for item in items:
        counts[item["status"]] += 1
    return counts


def todo_count_tuple(items: Sequence[TodoItem]) -> tuple[int, int, int, int]:
    counts = todo_counts(items)
    return tuple(counts[status] for status in TODO_STATUSES)
