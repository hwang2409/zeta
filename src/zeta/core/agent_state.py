"""Durable agent-lifecycle markers for the session state file."""

from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ..types import ToolCall
from .checkpoints import ConversationIntegrityError, _now

_AGENT_STATES = {"running", "completed", "canceled", "failed"}


def _agent_type_metadata(agent_type: str | None) -> dict[str, str]:
    return {} if agent_type is None else {"agent_type": agent_type}


def _parse_agent_state(value: dict[str, Any], state_path: Path) -> dict[str, Any]:
    """Validate and return the agent-lifecycle fields of a session state dict."""

    agent_counter = value.get("agent_counter", 0)
    if type(agent_counter) is not int or agent_counter < 0:
        raise ConversationIntegrityError(
            f"session state agent counter is invalid: {state_path}"
        )
    agent_children = value.get("agent_children", {})
    if type(agent_children) is not dict:
        raise ConversationIntegrityError(
            f"session state child markers are invalid: {state_path}"
        )
    for call_id, marker in agent_children.items():
        if (
            type(call_id) is not str
            or not call_id
            or type(marker) is not dict
            or type(marker.get("tool_call")) is not dict
            or type(marker.get("child_session_path")) is not str
            or not marker["child_session_path"]
            or type(marker.get("description")) is not str
            or not marker["description"]
            or (
                "agent_type" in marker
                and (
                    type(marker["agent_type"]) is not str
                    or not marker["agent_type"]
                )
            )
            or (
                "turns_used" in marker
                and (
                    type(marker["turns_used"]) is not int
                    or marker["turns_used"] < 0
                )
            )
            or (
                "background" in marker
                and type(marker["background"]) is not bool
            )
            or (
                "child_instance_id" in marker
                and (
                    type(marker["child_instance_id"]) is not str
                    or not marker["child_instance_id"]
                )
            )
        ):
            raise ConversationIntegrityError(
                f"session state child marker is invalid: {state_path}"
            )
        try:
            ToolCall.from_dict(marker["tool_call"])
        except ValueError as exc:
            raise ConversationIntegrityError(
                f"session state child tool call is invalid: {state_path}"
            ) from exc
    agent_parent = value.get("agent_parent")
    if agent_parent is not None and (
        type(agent_parent) is not dict
        or type(agent_parent.get("tool_call_id")) is not str
        or not agent_parent["tool_call_id"]
        or (
            "agent_type" in agent_parent
            and (
                type(agent_parent["agent_type"]) is not str
                or not agent_parent["agent_type"]
            )
        )
        or (
            "status" in agent_parent
            and agent_parent["status"] != "finished"
        )
    ):
        raise ConversationIntegrityError(
            f"session state parent marker is invalid: {state_path}"
        )
    agent_canceled = value.get("agent_canceled")
    if agent_canceled is not None and (
        type(agent_canceled) is not dict
        or type(agent_canceled.get("tool_call_id")) is not str
        or not agent_canceled["tool_call_id"]
        or agent_canceled.get("content") != "tool execution canceled"
        or (
            "agent_type" in agent_canceled
            and (
                type(agent_canceled["agent_type"]) is not str
                or not agent_canceled["agent_type"]
            )
        )
    ):
        raise ConversationIntegrityError(
            f"session state canceled marker is invalid: {state_path}"
        )
    return {
        "agent_counter": agent_counter,
        "agent_children": copy.deepcopy(agent_children),
        "agent_parent": copy.deepcopy(agent_parent),
        "agent_canceled": copy.deepcopy(agent_canceled),
    }


def _apply_agent_state(
    state: dict[str, Any],
    *,
    agent_counter: int,
    agent_children: dict[str, dict[str, Any]],
    agent_parent: dict[str, Any] | None,
    agent_canceled: dict[str, Any] | None,
) -> None:
    """Add the nonempty agent-lifecycle fields into a session state dict."""

    if agent_counter:
        state["agent_counter"] = agent_counter
    if agent_children:
        state["agent_children"] = copy.deepcopy(agent_children)
    if agent_parent is not None:
        state["agent_parent"] = copy.deepcopy(agent_parent)
    if agent_canceled is not None:
        state["agent_canceled"] = copy.deepcopy(agent_canceled)


class AgentStateMixin:
    """Public agent-lifecycle methods layered onto ConversationStore."""

    def allocate_agent_index(self) -> int:
        """Allocate the next durable child-agent directory number."""

        with self._append_lock():
            self._load()
            self._load_session_state()
            agents_root = self.session_dir / "agents"
            agents_root.mkdir(parents=True, exist_ok=True)
            candidate = self._agent_counter + 1
            while (agents_root / str(candidate)).exists():
                candidate += 1
            self._agent_counter = candidate
            self._write_session_state(self.bash_cwd, self._todo_items)
            return candidate

    def register_agent_child(
        self,
        tool_call: ToolCall,
        *,
        child_session_path: str,
        description: str,
        agent_type: str | None = None,
        background: bool = False,
        child_instance_id: str | None = None,
    ) -> None:
        """Persist a running child marker before the child starts."""

        if not child_session_path or not description or agent_type == "":
            raise ValueError("child marker fields must be nonempty")
        with self._append_lock():
            self._load()
            self._load_session_state()
            marker = {
                "tool_call": tool_call.to_dict(),
                "child_session_path": child_session_path,
                "description": description,
                "turns_used": 0,
            }
            if background:
                marker["background"] = True
            if child_instance_id is not None:
                marker["child_instance_id"] = child_instance_id
            marker.update(_agent_type_metadata(agent_type))
            marker_key = child_instance_id or tool_call.id
            self._agent_children[marker_key] = marker
            self._write_session_state(self.bash_cwd, self._todo_items)

    def agent_children(self) -> dict[str, dict[str, Any]]:
        """Return durable markers for children that did not finish."""

        return copy.deepcopy(self._agent_children)

    def update_agent_child_turns(self, marker_key: str, turns_used: int) -> None:
        """Persist completed turns for a child marker."""

        if type(turns_used) is not int or turns_used < 0:
            raise ValueError("child turns must be a nonnegative integer")
        with self._append_lock():
            self._load()
            self._load_session_state()
            marker = self._agent_children.get(marker_key)
            if marker is None:
                return
            marker["turns_used"] = turns_used
            self._write_session_state(self.bash_cwd, self._todo_items)

    def finish_agent_child(self, marker_key: str) -> None:
        """Remove a child marker after its parent result is durable."""

        with self._append_lock():
            self._load()
            self._load_session_state()
            if marker_key not in self._agent_children:
                return
            self._agent_children.pop(marker_key)
            self._write_session_state(self.bash_cwd, self._todo_items)

    def mark_agent_parent(
        self,
        parent_tool_call_id: str,
        *,
        agent_type: str | None = None,
    ) -> None:
        """Persist the parent call id in a child session before execution."""

        if not parent_tool_call_id or agent_type == "":
            raise ValueError("parent tool call id must be nonempty")
        with self._append_lock():
            self._load()
            self._load_session_state()
            self._agent_parent = {"tool_call_id": parent_tool_call_id}
            self._agent_parent.update(_agent_type_metadata(agent_type))
            self._agent_canceled = None
            self._write_session_state(self.bash_cwd, self._todo_items)

    def finish_agent_parent(self) -> None:
        """Remove the child marker after child execution finishes."""

        with self._append_lock():
            self._load()
            self._load_session_state()
            if self._agent_parent is None:
                return
            if "agent_type" in self._agent_parent:
                self._agent_parent["status"] = "finished"
            else:
                self._agent_parent = None
            self._write_session_state(self.bash_cwd, self._todo_items)

    def mark_agent_canceled(self, parent_tool_call_id: str) -> None:
        """Persist a child cancellation after its task has stopped."""

        if not parent_tool_call_id:
            raise ValueError("parent tool call id must be nonempty")
        with self._append_lock():
            self._load()
            self._load_session_state()
            agent_type = self.agent_type()
            self._agent_parent = None
            self._agent_canceled = {
                "tool_call_id": parent_tool_call_id,
                "content": "tool execution canceled",
            }
            self._agent_canceled.update(_agent_type_metadata(agent_type))
            if self._agent_lifecycle is not None:
                self._agent_lifecycle["state"] = "canceled"
                self._agent_lifecycle["finished_at"] = _now()
                self._agent_lifecycle["final_result"] = "tool execution canceled"
            self._write_session_state(self.bash_cwd, self._todo_items)
            if self._agent_lifecycle is not None:
                self._write_agent_lifecycle()

    def agent_type(self) -> str | None:
        """Return the child type from its agent marker."""

        marker = self._agent_parent or self._agent_canceled
        if marker is None:
            return None
        agent_type = marker.get("agent_type")
        return agent_type if type(agent_type) is str and agent_type else None

    def agent_canceled(self) -> dict[str, Any] | None:
        """Return the durable cancellation marker, if one exists."""

        return copy.deepcopy(self._agent_canceled)

    def agent_lifecycle(self) -> dict[str, Any] | None:
        """Return a detached snapshot of this child lifecycle."""

        return copy.deepcopy(self._agent_lifecycle)

    def agent_handle(self) -> str | None:
        """Return this child's stable handle, if it has one."""

        lifecycle = self._agent_lifecycle
        handle = lifecycle.get("handle") if lifecycle is not None else None
        return handle if type(handle) is str else None

    def start_agent_lifecycle(
        self,
        *,
        handle: str,
        started_at: str,
        tree_budget: int,
        depth: int,
        agent_type: str,
        description: str,
    ) -> None:
        """Persist the metadata used by the agent status tool."""

        if (
            not handle
            or not started_at
            or type(tree_budget) is not int
            or tree_budget < 1
            or type(depth) is not int
            or depth < 1
            or not agent_type
            or not description
        ):
            raise ValueError("invalid agent lifecycle metadata")
        with self._append_lock():
            self._load()
            self._load_session_state()
            self._agent_lifecycle = {
                "handle": handle,
                "state": "running",
                "started_at": started_at,
                "finished_at": None,
                "turns_used": 0,
                "tree_budget": tree_budget,
                "current_step": "starting",
                "depth": depth,
                "agent_type": agent_type,
                "description": description,
            }
            self._write_agent_lifecycle()

    def update_agent_lifecycle(
        self,
        *,
        current_step: str | None = None,
        turns_used: int | None = None,
    ) -> None:
        """Persist a live child's latest step and turn count."""

        if current_step is not None and not current_step:
            raise ValueError("agent lifecycle step must be nonempty")
        if turns_used is not None and (type(turns_used) is not int or turns_used < 0):
            raise ValueError("agent lifecycle turns must be nonnegative")
        with self._append_lock():
            self._load()
            self._load_session_state()
            if self._agent_lifecycle is None:
                return
            if current_step is not None:
                self._agent_lifecycle["current_step"] = current_step
            if turns_used is not None:
                self._agent_lifecycle["turns_used"] = turns_used
            self._write_agent_lifecycle()

    def finish_agent_lifecycle(
        self,
        state: str,
        *,
        final_result: str,
        turns_used: int | None = None,
        finished_at: str | None = None,
    ) -> None:
        """Persist a terminal child state and its final result text."""

        if state not in _AGENT_STATES - {"running"} or not final_result:
            raise ValueError("invalid terminal agent lifecycle")
        if turns_used is not None and (type(turns_used) is not int or turns_used < 0):
            raise ValueError("agent lifecycle turns must be nonnegative")
        with self._append_lock():
            self._load()
            self._load_session_state()
            if self._agent_lifecycle is None:
                return
            self._agent_lifecycle["state"] = state
            self._agent_lifecycle["finished_at"] = finished_at or _now()
            self._agent_lifecycle["final_result"] = final_result
            if turns_used is not None:
                self._agent_lifecycle["turns_used"] = turns_used
            self._write_agent_lifecycle()

    def _write_agent_lifecycle(self) -> None:
        """Atomically write lifecycle data without changing session state bytes."""

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.session_dir,
            prefix=".agent_lifecycle.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            try:
                json.dump(self._agent_lifecycle, temporary, separators=(",", ":"))
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
                os.replace(temporary_path, self.agent_lifecycle_path)
            finally:
                temporary_path.unlink(missing_ok=True)
