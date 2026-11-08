"""Transfer unfinished descendants to their surviving agent owner."""

from __future__ import annotations

from typing import Protocol

from ..core.agent_state import _agent_child_registration_values
from ..core.store import ConversationStore


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
        tool_call, registration_values = _agent_child_registration_values(marker)
        parent_store.register_agent_child(tool_call, **registration_values)
        child_instance_id = marker.get("child_instance_id")
        if background_owner is not None and type(child_instance_id) is str:
            background_owner.adopt(child_instance_id, parent_store, parent_instance_id)
        child_store.finish_agent_child(marker_key)
