"""List the current project's open attention queue."""

from __future__ import annotations

import asyncio
from typing import Literal, TypedDict

from ...attention_decisions import compact_age, open_project_attention
from ...attention_records import Broken
from ...protocol.types import StructuredToolResult
from .._results import _success_result, text_block
from ..registry import ToolRegistry


class ListAttentionArguments(TypedDict, total=False):
    offset: int
    list: Literal["action", "fyi"]


async def _list_attention(
    registry: ToolRegistry, arguments: ListAttentionArguments
) -> StructuredToolResult:
    if registry.project_id is None:
        raise ValueError("list_attention requires a project")
    offset = arguments.get("offset", 0)
    view = arguments.get("list", "action")
    if view not in ("action", "fyi"):
        raise ValueError("list must be action or fyi")
    page = await asyncio.to_thread(
        open_project_attention,
        registry.session_store.session_dir.parent.parent,
        registry.project_id,
        offset=offset,
        fyi=view == "fyi",
    )
    items: list[dict[str, object]] = []
    broken_total = page.total - page.action_required_total - page.info_total
    lines = [f"{page.action_required_total} require action · {page.info_total} FYI · {broken_total} broken"]
    for item in page.items:
        record = item.record
        if isinstance(record, Broken):
            items.append(
                {
                    "id": record.id,
                    "kind": "broken",
                    "title": f"broken attention record ({record.reason})",
                    "age": "unknown age",
                    "created_at": None,
                    "owning_session": item.session_id,
                    "requires_action": False,
                    "broken": True,
                    "path": record.path,
                }
            )
            lines.append(f"{record.id} [broken: {record.reason}] ({item.session_id})")
            continue
        age = compact_age(record.created_at)
        items.append(
            {
                "id": record.id,
                "kind": record.kind,
                "requires_action": record.requires_action,
                "title": record.title,
                "age": age,
                "created_at": record.created_at,
                "owning_session": item.session_id,
            }
        )
        label = "action required" if record.requires_action else "FYI"
        lines.append(f"{record.id} [{label}, {record.kind}, {age}] {record.title} ({item.session_id})")
    if not items:
        lines.append(f"No items on this {view.upper()} page.")
    structured = {
        "items": items,
        "list": view,
        "total": page.total,
        "action_required_total": page.action_required_total,
        "info_total": page.info_total,
    }
    if page.next_offset:
        structured["next_offset"] = page.next_offset
    return _success_result(text_block("\n".join(lines)), structured_content=structured)


def register(registry: ToolRegistry) -> None:
    registry.register_session_tool(
        "list_attention",
        _list_attention,
        binding="shared",
        description=(
            "List this project's open attention, including dead predecessor sessions. By default, items "
            "contains the action list (decision and deferred), plus broken-record diagnostics. Use "
            "list=fyi to read answer and decided notes, which do not require a decision. Reading does "
            "not acknowledge FYI notes; the user can dismiss them via /decisions. action_required_total "
            "and info_total count their groups across all pages; total also includes broken records. "
            "Continue with next_offset using the same list."
        ),
        parameters={
            "type": "object",
            "properties": {
                "offset": {"type": "integer", "minimum": 0},
                "list": {"type": "string", "enum": ["action", "fyi"]},
            },
            "required": [],
            "additionalProperties": False,
        },
        requires_approval=False,
    )


__all__ = ["register"]
