"""Child-marker writes keep the session_state.json bytes and the marker reads of base."""

from __future__ import annotations

import hashlib
import json

from zeta.core.store import ConversationStore
from zeta.protocol.types import ToolCall

# sha256 of the normalized trace below, recorded on base ad8a89068 (before markers were
# copied on write). A different value means the stored bytes or the marker reads changed.
_BASE_TRACE_SHA256 = "baaa90f9bedbf240c695d1abe4e36f95912434f581aef1bbc4c974fc7c315f94"


def _resume(cwd: str) -> dict[str, object]:
    return {
        "account_pins": "pending",
        "background": True,
        "cwd": cwd,
        "depth": 1,
        "description": "resumable child",
        "effective_max_turns": None,
        "explicit_finish": False,
        "follow_up": False,
        "model": "gpt-5.6-sol",
        "preset": {
            "accepts_follow_ups": False,
            "allow_delegation": False,
            "explicit_finish": False,
            "max_child_depth": None,
            "max_turns": None,
            "name": "general",
            "preamble": "",
            "prompt_suffix": "",
            "source": "builtin",
            "tool_names": None,
        },
        "provider": "codex",
        "requested_max_turns": None,
        "requested_model": None,
        "selected_accounts": {},
    }


def _register(store: ConversationStore, index: int, **kwargs: object) -> None:
    store.register_agent_child(
        ToolCall(f"call-{index}", "agent", {"prompt": f"task {index}"}),
        child_session_path=f"agents/{index}",
        description=f"child {index}",
        **kwargs,
    )


def test_marker_sequence_writes_base_bytes_and_reads_base_markers(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    cwd = str(project)
    trace: list[str] = []
    owner = ConversationStore(tmp_path / "sessions", cwd=project)
    other = None
    try:

        def record(step: str, store: ConversationStore) -> None:
            raw = (owner.session_dir / "session_state.json").read_text()
            trace.append(f"{step}\n{raw}{json.dumps(store.agent_children(), sort_keys=True)}\n")

        for _ in range(4):
            owner.allocate_agent_index()
        record("allocate", owner)
        _register(owner, 1, background=True, child_instance_id="run-1", resume=_resume(cwd))
        _register(owner, 2, background=True, child_instance_id="run-2", agent_type="worker")
        _register(owner, 3, accepts_follow_ups=True, explicit_finish=True, turns_used=1)
        record("register", owner)
        owner.select_child_accounts("run-1", {"codex": "codex:me"})
        record("select", owner)
        owner.update_agent_child_turns("call-3", 4)
        record("turns in memory", owner)
        owner.persist_agent_child_turns("run-1", 2)
        record("turns persisted", owner)
        owner.set_todo_items([{"content": "wait for children", "status": "in_progress"}])
        record("todo", owner)

        # A second store object of the same session writes; the first must see it.
        other = ConversationStore(tmp_path / "sessions", session_id=owner.session_id)
        _register(other, 4, background=True, child_instance_id="run-4")
        other.finish_agent_child("run-2")
        record("other writer", other)
        owner.finish_agent_child("call-3")
        record("finish after other writer", owner)
        _register(owner, 5)
        record("register after other writer", owner)

        reopened = ConversationStore(tmp_path / "sessions", session_id=owner.session_id)
        try:
            assert reopened.agent_children() == owner.agent_children()
            trace.append(json.dumps(reopened.agent_children(), sort_keys=True))
        finally:
            reopened.close()
    finally:
        if other is not None:
            other.close()
        owner.close()

    normalized = "".join(trace).replace(cwd, "<project>")
    assert set(owner.agent_children()) == {"run-1", "run-4", "call-5"}
    assert owner.agent_children()["run-1"]["turns_used"] == 2
    assert owner.agent_children()["run-1"]["resume"]["account_pins"] == "selected"
    assert hashlib.sha256(normalized.encode()).hexdigest() == _BASE_TRACE_SHA256, normalized


def test_returned_markers_do_not_alias_the_store(tmp_path) -> None:
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    try:
        _register(store, 1, background=True, child_instance_id="run-1")
        returned = store.agent_children()
        returned["run-1"]["description"] = "corrupted"
        returned["run-1"]["tool_call"]["arguments"]["prompt"] = "corrupted"
        returned.pop("run-1")
        _register(store, 2)

        marker = store.agent_children()["run-1"]
        assert marker["description"] == "child 1"
        assert marker["tool_call"]["arguments"] == {"prompt": "task 1"}
        state = json.loads((store.session_dir / "session_state.json").read_text())
        assert state["agent_children"]["run-1"]["description"] == "child 1"
    finally:
        store.close()
