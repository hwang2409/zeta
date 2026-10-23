"""Transfer unfinished descendants to their surviving agent owner."""

from __future__ import annotations

from typing import Protocol

from ..core.store import ConversationStore
from ..protocol.types import ToolCall


class _BackgroundOwner(Protocol):
    def adopt(
        self,
        instance_id: str,
        parent_store: ConversationStore,
        parent_instance_id: str | None = None,
    ) -> None: ...


def adopt_agent_children(
    child_store: ConversationStore,
    parent_store: ConversationStore,
    *,
    background_owner: _BackgroundOwner | None = None,
    parent_instance_id: str | None = None,
) -> None:
    """Move unfinished descendants into the surviving parent session."""

    for marker_key, marker in child_store.agent_children().items():
        tool_call = ToolCall.from_dict(marker["tool_call"])
        parent_store.register_agent_child(
            tool_call,
            child_session_path=marker["child_session_path"],
            description=marker["description"],
            agent_type=marker.get("agent_type"),
            background=marker.get("background", False),
            child_instance_id=marker.get("child_instance_id"),
            accepts_follow_ups=marker.get("accepts_follow_ups", False) is True,
            explicit_finish=marker.get("explicit_finish", False) is True,
            turns_used=(
                marker.get("turns_used", 0)
                if type(marker.get("turns_used", 0)) is int
                else 0
            ),
        )
        child_instance_id = marker.get("child_instance_id")
        if background_owner is not None and type(child_instance_id) is str:
            background_owner.adopt(child_instance_id, parent_store, parent_instance_id)
        child_store.finish_agent_child(marker_key)
