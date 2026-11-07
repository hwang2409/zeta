import asyncio
import hashlib
import json
import os
import subprocess
import sys
import threading
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.support.notification_fixtures import notification_payloads
from zeta.core import agent_state
from zeta.core.store import (
    MAX_AGENT_NOTIFICATION_TEXT,
    ConversationEntry,
    ConversationIntegrityError,
    ConversationStore,
)
from zeta.core.store._step_delivery import PromptInput
from zeta.core.store._validation import (
    NOTIFICATION_KINDS,
    NOTIFICATION_VALIDATORS,
    validate_notification_data,
)
from zeta.protocol.types import (
    Message,
    MessageOrigin,
    MessageRole,
    TextContent,
    ToolCall,
    ToolUseContent,
    with_message_origin,
)


def message(role: MessageRole, text: str) -> Message:
    message = Message(role, [TextContent(text)])
    if role is MessageRole.USER:
        return with_message_origin(message, MessageOrigin.USER)
    return message


def test_every_notification_kind_has_a_validator() -> None:
    payloads = notification_payloads()

    assert set(NOTIFICATION_VALIDATORS) == set(NOTIFICATION_KINDS) == set(payloads)
    for kind in NOTIFICATION_KINDS:
        validate_notification_data(payloads[kind])


def test_persistent_detached_writer_retains_home_lease(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store._prepare_durable_writer_snapshot()
    writer = store._new_persistent_durable_writer()
    try:
        assert writer._storage_lease._entry is store._storage_lease._entry
    finally:
        writer.close()
        store.close()


def test_store_rejects_unknown_notification_kind_on_write(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)

    with pytest.raises(ValueError, match="unknown notification kind"):
        store._append_row(
            "notification",
            {"kind": "future_kind", "text": "future notification"},
        )


def test_store_reads_persisted_unknown_notification_kind_without_crash(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    store.append_task_notification(
        task_id="task-1",
        command="command finished",
        exit_code=0,
        output_tail="done",
    )
    persisted = store.path.read_text(encoding="utf-8").replace(
        '"kind":"task_exited"',
        '"kind":"future_kind"',
    )
    store.path.write_text(persisted, encoding="utf-8")

    reopened = ConversationStore(tmp_path, session_id=store.session_id)

    assert reopened.agent_notifications(pending_only=False)[0].data["kind"] == "future_kind"


def test_task_notification_rejects_non_string_output_tail(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)

    with pytest.raises(ConversationIntegrityError) as raised:
        store._append_row(
            "notification",
            {
                "origin": "notification",
                "kind": "task_exited",
                "task_id": "task-1",
                "headline": "command finished",
                "exit_code": 0,
                "output_tail": 7,
            },
        )

    assert str(raised.value.__cause__) == "invalid task notification"


def test_non_indexed_store_avoids_receipt_prewrite_fstat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path)
    real_fstat = os.fstat
    write_fstats = 0

    def count_fstat(fd: int) -> os.stat_result:
        nonlocal write_fstats
        frame = sys._getframe(1)
        if frame.f_code.co_name == "_write_bytes":
            write_fstats += 1
        return real_fstat(fd)

    monkeypatch.setattr(os, "fstat", count_fstat)
    store.append_message(message(MessageRole.USER, "one"))

    assert write_fstats == 1


def test_non_indexed_store_does_not_retain_append_receipts(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)

    for index in range(1_000):
        store.append_message(message(MessageRole.USER, f"message {index}"))

    assert store.take_persisted_appends() == ()
    assert store._persisted_appends == []


INVALID_KILLED_TASK_FIELDS = [
    {"killed_task_ids": "not-a-list"},
    {"killed_task_ids": [""]},
    {"killed_task_ids": ["x" * 65]},
    {"killed_task_ids": [f"task-{index}" for index in range(65)]},
    {"killed_task_count": -1},
    {"killed_task_ids": ["a", "b"], "killed_task_count": 1},
    {"killed_task_ids_truncated": "yes"},
]


def _notification_kwargs(**fields: object) -> dict[str, object]:
    values: dict[str, object] = {
        "killed_task_ids": ["a"],
        "killed_task_count": 1,
        "killed_task_ids_truncated": False,
    }
    values.update(fields)
    return values


@pytest.mark.parametrize("fields", INVALID_KILLED_TASK_FIELDS)
def test_append_rejects_invalid_killed_task_fields(
    tmp_path: Path, fields: dict[str, object]
) -> None:
    store = ConversationStore(tmp_path)
    with pytest.raises(ValueError, match="killed task fields"):
        store.append_agent_notification(
            "parent:1",
            child_session_path="/child",
            description="child",
            status="completed",
            text="done",
            **_notification_kwargs(**fields),
        )


def test_append_agent_notification_rejects_oversized_text_without_writing(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    before = store.path.read_bytes() if store.path.exists() else b""

    with pytest.raises(ValueError, match="invalid agent notification"):
        store.append_agent_notification(
            "parent:1",
            child_session_path="/child",
            description="child",
            status="completed",
            text="x" * (MAX_AGENT_NOTIFICATION_TEXT + 1),
        )

    assert (store.path.read_bytes() if store.path.exists() else b"") == before


@pytest.mark.parametrize(
    "invalid_data",
    [
        {
            "kind": "agent_completion",
            "child_instance_id": "",
            "child_session_path": "/child",
            "description": "child",
            "status": "completed",
            "text": "done",
        },
        {
            "kind": "agent_completion",
            "child_instance_id": "parent:1",
            "child_session_path": "/child",
            "description": "child",
            "status": "unknown",
            "text": "done",
        },
        {
            "kind": "agent_completion",
            "child_instance_id": "parent:1",
            "child_session_path": "/child",
            "description": "child",
            "status": "completed",
            "text": "x" * (MAX_AGENT_NOTIFICATION_TEXT + 1),
        },
        {
            "kind": "agent_completion",
            "child_instance_id": "parent:1",
            "child_session_path": "/child",
            "description": "child",
            "status": "completed",
            "text": "done",
            "stats": {"turns_used": -1},
        },
    ],
)
def test_notification_append_uses_loader_payload_validation(
    tmp_path: Path, invalid_data: dict[str, object]
) -> None:
    store = ConversationStore(tmp_path)
    entry = ConversationEntry(
        seq=1,
        id="candidate",
        parent_id=None,
        lane="main",
        type="notification",
        data=invalid_data,
    )
    with pytest.raises(ConversationIntegrityError):
        store._validate_entry_payload(entry)
    before = store.path.read_bytes() if store.path.exists() else b""

    with pytest.raises(ConversationIntegrityError):
        store._append_row("notification", invalid_data)

    assert (store.path.read_bytes() if store.path.exists() else b"") == before


def test_append_accepts_legacy_notification_without_killed_task_fields(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    store.append_agent_notification(
        "parent:1",
        child_session_path="/child",
        description="child",
        status="completed",
        text="done",
    )
    assert ConversationStore(
        tmp_path, session_id=store.session_id
    ).agent_notifications()



def test_load_accepts_legacy_notifications_without_origin(tmp_path: Path) -> None:
    """Rows written before notifications recorded ``origin`` still load."""

    store = ConversationStore(tmp_path)
    store.append_agent_notification(
        "parent:1",
        child_session_path="/child",
        description="child",
        status="completed",
        text="done",
    )
    store.append_task_notification(task_id="task-1", command="sleep 1", exit_code=0)
    store.close()
    rows = [json.loads(line) for line in store.path.read_text().splitlines()]
    stripped = 0
    for row in rows:
        if row.get("type") == "notification" and "origin" in row["data"]:
            del row["data"]["origin"]
            stripped += 1
    assert stripped == 2
    store.path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    reopened = ConversationStore(tmp_path, session_id=store.session_id)

    assert reopened.agent_notifications()
    kinds = {
        entry.data.get("kind")
        for entry in reopened.entries
        if entry.type == "notification"
    }
    assert "task_exited" in kinds

@pytest.mark.parametrize(
    "field_value",
    [
        ("killed_task_ids", "not-a-list"),
        ("killed_task_ids", [""]),
        ("killed_task_ids", ["x" * 65]),
        ("killed_task_ids", [f"task-{index}" for index in range(65)]),
        ("killed_task_count", -1),
        ("killed_task_count", 0),
        ("killed_task_ids_truncated", "yes"),
    ],
)
def test_load_rejects_corrupt_killed_task_fields(
    tmp_path: Path, field_value: tuple[str, object]
) -> None:
    store = ConversationStore(tmp_path)
    store.append_agent_notification(
        "parent:1",
        child_session_path="/child",
        description="child",
        status="completed",
        text="done",
        **_notification_kwargs(),
    )
    rows = [json.loads(line) for line in store.path.read_text().splitlines()]
    rows[-1]["data"][field_value[0]] = field_value[1]
    store.path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(
        ConversationIntegrityError,
        match="invalid payload for conversation entry",
    ) as raised:
        ConversationStore(tmp_path, session_id=store.session_id)
    assert raised.value.__cause__ is not None
    assert str(raised.value.__cause__) == "invalid agent notification"


@pytest.mark.parametrize("value", ["yes", 1, False, {"x": 1}, None])
def test_load_rejects_malformed_display_only(tmp_path: Path, value: object) -> None:
    store = ConversationStore(tmp_path)
    store.append_agent_notification(
        "parent:1",
        child_session_path="/child",
        description="child",
        status="completed",
        text="done",
        display_only=True,
    )
    rows = [json.loads(line) for line in store.path.read_text().splitlines()]
    rows[-1]["data"]["display_only"] = value
    store.path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(
        ConversationIntegrityError,
        match="invalid payload for conversation entry",
    ) as raised:
        ConversationStore(tmp_path, session_id=store.session_id)
    assert str(raised.value.__cause__) == "invalid agent notification"


@pytest.mark.parametrize("value", ["yes", 1, None])
def test_append_rejects_non_boolean_display_only(tmp_path: Path, value: object) -> None:
    store = ConversationStore(tmp_path)
    with pytest.raises(ValueError, match="invalid agent notification"):
        store.append_agent_notification(
            "parent:1",
            child_session_path="/child",
            description="child",
            status="completed",
            text="done",
            display_only=value,  # type: ignore[arg-type]
        )


def _start_agent_lifecycle(store: ConversationStore) -> None:
    store.start_agent_lifecycle(
        handle="parent:1",
        started_at="2026-09-30T00:00:00+00:00",
        depth=1,
        agent_type="agent",
        description="child",
    )


def test_live_lifecycle_updates_do_not_write_and_finish_persists_final_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path)
    _start_agent_lifecycle(store)
    writes: list[str] = []
    write_session_file = agent_state.write_session_file

    def record_write(directory_fd: int, name: str, value: bytes) -> None:
        writes.append(name)
        write_session_file(directory_fd, name, value)

    monkeypatch.setattr(agent_state, "write_session_file", record_write)

    store.update_agent_lifecycle(current_step="turn 2: running")
    store.update_agent_lifecycle(turns_used=2)
    store.update_agent_lifecycle(tool_calls=3)

    assert writes == []
    durable_running = json.loads(
        store.agent_lifecycle_path.read_text(encoding="utf-8")
    )
    assert durable_running["current_step"] == "starting"
    assert durable_running["turns_used"] == 0
    assert durable_running["tool_calls"] == 0

    store.finish_agent_lifecycle("completed", final_result="done")

    assert writes == ["agent_lifecycle.json"]
    durable_finished = json.loads(
        store.agent_lifecycle_path.read_text(encoding="utf-8")
    )
    assert durable_finished["state"] == "completed"
    assert durable_finished["turns_used"] == 2
    assert durable_finished["tool_calls"] == 3


def test_child_turn_progress_does_not_write_parent_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path)
    call = ToolCall("agent-1", "agent", {"prompt": "work"})
    store.register_agent_child(
        call,
        child_session_path=str(tmp_path / "agents" / "1"),
        description="work",
    )

    def fail_write(*args: object, **kwargs: object) -> None:
        raise AssertionError("progress must not write session_state.json")

    monkeypatch.setattr(store, "_write_session_state", fail_write)

    store.update_agent_child_turns(call.id, 4)

    assert store.agent_children()[call.id]["turns_used"] == 4


def test_child_transitions_preserve_sibling_live_turns(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    first = ToolCall("agent-1", "agent", {"prompt": "first"})
    second = ToolCall("agent-2", "agent", {"prompt": "second"})
    store.register_agent_child(
        first,
        child_session_path=str(tmp_path / "agents" / "1"),
        description="first",
    )
    store.update_agent_child_turns(first.id, 3)

    store.register_agent_child(
        second,
        child_session_path=str(tmp_path / "agents" / "2"),
        description="second",
    )

    assert store.agent_children()[first.id]["turns_used"] == 3
    durable = json.loads(store.state_path.read_text(encoding="utf-8"))
    assert durable["agent_children"][first.id]["turns_used"] == 3

    store.update_agent_child_turns(first.id, 4)
    store.finish_agent_child(second.id)

    assert store.agent_children()[first.id]["turns_used"] == 4
    durable = json.loads(store.state_path.read_text(encoding="utf-8"))
    assert durable["agent_children"][first.id]["turns_used"] == 4


def test_reopened_running_lifecycle_uses_last_durable_transition(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    _start_agent_lifecycle(store)
    store.update_agent_lifecycle(
        current_step="turn 4: running", turns_used=4, tool_calls=2
    )

    reopened = ConversationStore(tmp_path, session_id=store.session_id)

    lifecycle = reopened.agent_lifecycle()
    assert lifecycle is not None
    assert lifecycle["state"] == "running"
    assert lifecycle["current_step"] == "starting"
    assert lifecycle["turns_used"] == 0
    assert lifecycle["tool_calls"] == 0


def test_cancel_transition_persists_live_counters(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.mark_agent_parent("call-1", agent_type="general")
    _start_agent_lifecycle(store)
    store.update_agent_lifecycle(turns_used=2, tool_calls=3)

    store.mark_agent_canceled("call-1")

    reopened = ConversationStore(tmp_path, session_id=store.session_id)
    lifecycle = reopened.agent_lifecycle()
    assert lifecycle is not None
    assert lifecycle["state"] == "canceled"
    assert lifecycle["turns_used"] == 2
    assert lifecycle["tool_calls"] == 3
    assert reopened.agent_canceled() is not None


@pytest.mark.parametrize("state", ["failed", "canceled"])
def test_terminal_lifecycle_transitions_persist_live_counters(
    tmp_path: Path, state: str
) -> None:
    store = ConversationStore(tmp_path)
    _start_agent_lifecycle(store)
    store.update_agent_lifecycle(turns_used=2, tool_calls=3)

    store.finish_agent_lifecycle(state, final_result=state)

    reopened = ConversationStore(tmp_path, session_id=store.session_id)
    lifecycle = reopened.agent_lifecycle()
    assert lifecycle is not None
    assert lifecycle["state"] == state
    assert lifecycle["turns_used"] == 2
    assert lifecycle["tool_calls"] == 3


def test_lifecycle_accepts_legacy_record_without_killed_task_fields(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    _start_agent_lifecycle(store)
    store.finish_agent_lifecycle("completed", final_result="done")

    lifecycle = ConversationStore(
        tmp_path, session_id=store.session_id
    ).agent_lifecycle()

    assert lifecycle is not None
    assert lifecycle["final_result"] == "done"
    assert not any(field in lifecycle for field in _notification_kwargs())


@pytest.mark.parametrize("marker", [True, False])
def test_lifecycle_load_preserves_boolean_receipt_marker(
    tmp_path: Path, marker: bool
) -> None:
    store = ConversationStore(tmp_path)
    _start_agent_lifecycle(store)
    store.finish_agent_lifecycle("completed", final_result="done")
    lifecycle = json.loads(store.agent_lifecycle_path.read_text(encoding="utf-8"))
    lifecycle["final_result_is_receipt"] = marker
    store.agent_lifecycle_path.write_text(json.dumps(lifecycle), encoding="utf-8")

    loaded = ConversationStore(tmp_path, session_id=store.session_id).agent_lifecycle()

    assert loaded is not None
    assert loaded["final_result_is_receipt"] is marker


def test_lifecycle_load_accepts_omitted_receipt_marker(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    _start_agent_lifecycle(store)
    store.finish_agent_lifecycle("completed", final_result="done")

    loaded = ConversationStore(tmp_path, session_id=store.session_id).agent_lifecycle()

    assert loaded is not None
    assert "final_result_is_receipt" not in loaded


@pytest.mark.parametrize("marker", [1, "true", {}, None])
def test_lifecycle_load_drops_invalid_receipt_marker(
    tmp_path: Path, marker: object
) -> None:
    store = ConversationStore(tmp_path)
    _start_agent_lifecycle(store)
    store.finish_agent_lifecycle("completed", final_result="done")
    lifecycle = json.loads(store.agent_lifecycle_path.read_text(encoding="utf-8"))
    lifecycle["final_result_is_receipt"] = marker
    store.agent_lifecycle_path.write_text(json.dumps(lifecycle), encoding="utf-8")

    with pytest.warns(RuntimeWarning, match="invalid receipt marker"):
        reopened = ConversationStore(tmp_path, session_id=store.session_id)

    loaded = reopened.agent_lifecycle()
    assert loaded is not None
    assert "final_result_is_receipt" not in loaded
    assert loaded["final_result"] == "done"


def test_lifecycle_write_rejects_non_boolean_receipt_marker(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    _start_agent_lifecycle(store)
    store.finish_agent_lifecycle("completed", final_result="done")

    with pytest.raises(ValueError, match="receipt marker"):
        store.update_agent_lifecycle_result("updated", canonical_receipt=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("fields", INVALID_KILLED_TASK_FIELDS)
def test_finish_lifecycle_rejects_invalid_killed_task_fields(
    tmp_path: Path, fields: dict[str, object]
) -> None:
    store = ConversationStore(tmp_path)
    _start_agent_lifecycle(store)

    with pytest.raises(ValueError, match="killed task fields"):
        store.finish_agent_lifecycle("completed", final_result="done", **fields)


@pytest.mark.parametrize("fields", INVALID_KILLED_TASK_FIELDS)
def test_update_lifecycle_result_rejects_invalid_killed_task_fields(
    tmp_path: Path, fields: dict[str, object]
) -> None:
    store = ConversationStore(tmp_path)
    _start_agent_lifecycle(store)
    store.finish_agent_lifecycle("completed", final_result="done")

    with pytest.raises(ValueError, match="killed task fields"):
        store.update_agent_lifecycle_result("updated", **fields)


@pytest.mark.parametrize("fields", INVALID_KILLED_TASK_FIELDS)
def test_lifecycle_load_drops_invalid_killed_task_fields(
    tmp_path: Path, fields: dict[str, object]
) -> None:
    store = ConversationStore(tmp_path)
    _start_agent_lifecycle(store)
    store.finish_agent_lifecycle(
        "completed", final_result="done", **_notification_kwargs()
    )
    lifecycle = json.loads(store.agent_lifecycle_path.read_text(encoding="utf-8"))
    lifecycle.update(fields)
    store.agent_lifecycle_path.write_text(json.dumps(lifecycle), encoding="utf-8")

    with pytest.warns(RuntimeWarning, match="dropping it"):
        reopened = ConversationStore(tmp_path, session_id=store.session_id)

    loaded = reopened.agent_lifecycle()
    assert loaded is not None
    assert loaded["final_result"] == "done"
    assert not any(field in loaded for field in _notification_kwargs())


def test_append_replay_round_trip_and_parent_links(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="session-1", cwd="/work")
    first = store.append_message(with_message_origin(message(MessageRole.USER, "hello"), MessageOrigin.USER))
    second = store.append_message(message(MessageRole.ASSISTANT, "hi"))

    reopened = ConversationStore(tmp_path, session_id="session-1")

    assert [entry.id for entry in reopened.replay()] == [first.id, second.id]
    assert reopened.active_branch_head_sequence() == second.seq
    assert second.parent_id == first.id
    assert [item.content[0].text for item in reopened.messages()] == ["hello", "hi"]
    assert reopened.session_id == "session-1"
    assert reopened.cwd == "/work"


def test_tui_notification_presentation_is_durable_but_not_consumption(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path, session_id="presented")
    notification = store.append_agent_notification(
        child_instance_id="child-1",
        child_session_path="/tmp/child-1",
        description="background macro",
        status="canceled",
        text="canceled on session shutdown",
    )

    assert not store.is_agent_notification_presented_to_tui(notification.id)
    store.mark_agent_notification_presented_to_tui(notification.id)
    assert store.is_agent_notification_presented_to_tui(notification.id)
    assert [entry.id for entry in store.agent_notifications()] == [notification.id]

    reopened = ConversationStore(tmp_path, session_id="presented")
    assert reopened.is_agent_notification_presented_to_tui(notification.id)
    assert [entry.id for entry in reopened.agent_notifications()] == [notification.id]
    reopened.acknowledge_agent_notification(notification.id)
    assert reopened.agent_notifications() == []


def test_installed_hook_emits_after_durable_resident_single_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path, session_id="installed-single")
    write_returned = False
    observed: list[tuple[ConversationEntry, ...] | str] = []
    original_write_bytes = store._write_bytes

    def tracked_write_bytes(data: bytes) -> None:
        nonlocal write_returned
        original_write_bytes(data)
        write_returned = True

    def installed(delta: tuple[ConversationEntry, ...] | str) -> None:
        assert write_returned
        assert delta != "reset"
        assert delta[-1].id in store._entry_ids
        assert store._entries[-1].id == delta[-1].id
        durable = json.loads(store.path.read_bytes().splitlines()[-1])
        assert (durable["seq"], durable["id"]) == (delta[-1].seq, delta[-1].id)
        observed.append(delta)

    monkeypatch.setattr(store, "_write_bytes", tracked_write_bytes)
    store.on_entries_installed = installed

    entry = store.append_message(message(MessageRole.ASSISTANT, "installed"))

    assert [[(row.seq, row.id) for row in delta] for delta in observed] == [
        [(entry.seq, entry.id)]
    ]


def test_installed_hook_failure_does_not_fail_committed_append(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = ConversationStore(tmp_path, session_id="installed-failure")

    def fail(_delta: tuple[ConversationEntry, ...] | str) -> None:
        raise RuntimeError("observer failed")

    store.on_entries_installed = fail

    entry = store.append_message(message(MessageRole.ASSISTANT, "committed"))
    reopened = ConversationStore(tmp_path, session_id=store.session_id)

    assert store._entries[-1].id == entry.id
    assert reopened._entries[-1] == entry
    observed: list[tuple[ConversationEntry, ...] | str] = []
    store.on_entries_installed = observed.append
    later = store.append_message(message(MessageRole.ASSISTANT, "later"))

    assert observed == [(later,)]
    assert [record.getMessage() for record in caplog.records].count(
        "installed-row callback failed"
    ) == 1


def test_installed_hook_receives_owned_delta(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="installed-owned")

    def mutate(delta: tuple[ConversationEntry, ...] | str) -> None:
        assert delta != "reset"
        delta[0].data["message"]["content"][0]["text"] = "corrupted"

    store.on_entries_installed = mutate

    entry = store.append_message(message(MessageRole.ASSISTANT, "original"))
    reopened = ConversationStore(tmp_path, session_id=store.session_id)

    assert entry.data["message"]["content"][0]["text"] == "original"
    assert store._entries[-1].data["message"]["content"][0]["text"] == "original"
    assert reopened._entries[-1].data["message"]["content"][0]["text"] == "original"


def test_installed_hook_runs_after_flock_and_can_reenter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path, session_id="installed-reentrant")
    gate = threading.Lock()
    real_flock = __import__("fcntl").flock

    def gated_flock(fd: int, operation: int) -> None:
        if operation & __import__("fcntl").LOCK_UN:
            real_flock(fd, operation)
            gate.release()
            return
        assert gate.acquire(blocking=False), "callback re-entered before flock release"
        real_flock(fd, operation)

    monkeypatch.setattr("zeta.core.store._store.fcntl.flock", gated_flock)
    observed: list[str] = []

    def reenter(_delta: tuple[ConversationEntry, ...] | str) -> None:
        store.on_entries_installed = None
        observed.append(store.append_message(message(MessageRole.ASSISTANT, "inner")).id)

    store.on_entries_installed = reenter

    outer = store.append_message(message(MessageRole.ASSISTANT, "outer"))

    assert outer.id != observed[0]
    monkeypatch.undo()
    store.close()
    assert [row.data["message"]["content"][0]["text"] for row in store._entries] == [
        "outer",
        "inner",
    ]


def test_flock_install_paths_allow_reentrant_callback_append(tmp_path: Path) -> None:
    def run_reentrant(
        store: ConversationStore, action: Callable[[], object]
    ) -> None:
        observed: list[tuple[ConversationEntry, ...] | str] = []

        def installed(delta: tuple[ConversationEntry, ...] | str) -> None:
            observed.append(delta)
            store.on_entries_installed = None
            store.append_message(message(MessageRole.ASSISTANT, "inner"))

        store.on_entries_installed = installed
        action()
        assert len(observed) == 1
        assert store.messages()[-1].content[0].text == "inner"

    appended = ConversationStore(tmp_path, session_id="installed-path-append")
    run_reentrant(
        appended,
        lambda: appended.append_message(message(MessageRole.ASSISTANT, "outer")),
    )

    reloaded = ConversationStore(tmp_path, session_id="installed-path-reload")
    foreign = ConversationStore(tmp_path, session_id=reloaded.session_id)
    foreign.append_message(message(MessageRole.ASSISTANT, "foreign"))
    run_reentrant(reloaded, reloaded.refresh)

    forked = ConversationStore(tmp_path, session_id="installed-path-fork")
    checkpoint = forked.append_checkpoint("base")
    forked.append_message(message(MessageRole.ASSISTANT, "discarded"))
    run_reentrant(forked, lambda: forked.append_fork(str(checkpoint.seq)))

    switched = ConversationStore(tmp_path, session_id="installed-path-switch")
    checkpoint = switched.append_checkpoint("base")
    switched.append_message(message(MessageRole.ASSISTANT, "discarded"))
    foreign = ConversationStore(tmp_path, session_id=switched.session_id)
    foreign.append_fork(str(checkpoint.seq))
    run_reentrant(switched, switched.refresh)

    repaired = ConversationStore(tmp_path, session_id="installed-path-repair")
    repaired.append_message(message(MessageRole.ASSISTANT, "kept"))
    with repaired.path.open("ab") as handle:
        handle.write(b'{"seq":3,"id":"torn"')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        run_reentrant(
            repaired,
            lambda: repaired.append_message(message(MessageRole.ASSISTANT, "after")),
        )


def test_installed_hook_reentrant_append_emits_in_log_order(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="installed-reentrant-order")
    observed: list[str] = []

    def reenter(delta: tuple[ConversationEntry, ...] | str) -> None:
        assert delta != "reset"
        text = delta[0].data["message"]["content"][0]["text"]
        if text == "outer":
            store.append_message(message(MessageRole.ASSISTANT, "inner"))
        observed.append(text)

    store.on_entries_installed = reenter

    store.append_message(message(MessageRole.ASSISTANT, "outer"))

    durable = [
        row.data["message"]["content"][0]["text"] for row in store._entries
    ]
    assert observed == durable == ["outer", "inner"]


def test_installed_hook_serializes_two_threads_in_durable_order(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path, session_id="installed-two-threads")
    first_entered = threading.Event()
    release_first = threading.Event()
    second_returned = threading.Event()
    observed: list[int] = []
    active_callbacks = 0
    callbacks_lock = threading.Lock()

    def installed(delta: tuple[ConversationEntry, ...] | str) -> None:
        nonlocal active_callbacks
        assert delta != "reset"
        with callbacks_lock:
            active_callbacks += 1
            assert active_callbacks == 1
            observed.append(delta[0].seq)
        if delta[0].seq == 1:
            first_entered.set()
            assert release_first.wait(timeout=5)
        with callbacks_lock:
            active_callbacks -= 1

    def append(text: str, returned: threading.Event | None = None) -> None:
        store.append_message(message(MessageRole.ASSISTANT, text))
        if returned is not None:
            returned.set()

    store.on_entries_installed = installed
    owner = threading.Thread(target=append, args=("one",))
    owner.start()
    assert first_entered.wait(timeout=5)

    worker = threading.Thread(target=append, args=("two", second_returned))
    worker.start()
    assert second_returned.wait(timeout=5)
    assert observed == [1]

    release_first.set()
    owner.join(timeout=5)
    worker.join(timeout=5)

    assert not owner.is_alive()
    assert not worker.is_alive()
    assert observed == [1, 2]


@pytest.mark.asyncio
async def test_async_appends_encode_only_new_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path, session_id="installed-append-cost")
    store.append_many([("warning", {"message": str(index)}) for index in range(400)])
    await store._to_thread_durable(store._warm_step_delivery_writer)
    original_encode = store._encoded_entry_line
    encoded_rows = 0

    def tracked_encode(entry: ConversationEntry) -> bytes:
        nonlocal encoded_rows
        encoded_rows += 1
        return original_encode(entry)

    monkeypatch.setattr(
        ConversationStore, "_encoded_entry_line", staticmethod(tracked_encode)
    )
    for index in range(25):
        await store._to_thread_durable(
            store._commit_delivery_input,
            PromptInput(f"prompt-{index}", MessageOrigin.AGENT_SEND, None),
        )

    assert encoded_rows == 25


def test_installed_hook_emits_batch_in_log_order(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="installed-batch")
    observed: list[tuple[ConversationEntry, ...] | str] = []
    store.on_entries_installed = observed.append

    entries = store.append_many(
        [("warning", {"message": "one"}), ("warning", {"message": "two"})]
    )

    assert observed == [tuple(store._entries)]
    assert [(row.seq, row.id) for row in observed[0]] == [
        (row.seq, row.id) for row in entries
    ]
    assert type(store._entry_prefix_digest) is bytes
    assert len(store._entry_prefix_digest) == 32


@pytest.mark.asyncio
async def test_installed_hook_async_failure_keeps_receipt_bookkeeping(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path, session_id="installed-async-failure")
    receipts_accepted = False
    original_accept = store._accept_persisted_appends

    def accept(receipts: object) -> None:
        nonlocal receipts_accepted
        original_accept(receipts)
        receipts_accepted = True

    def fail(_delta: tuple[ConversationEntry, ...] | str) -> None:
        assert receipts_accepted
        raise RuntimeError("observer failed")

    monkeypatch.setattr(store, "_accept_persisted_appends", accept)
    store.on_entries_installed = fail

    entry = await store.append_message_async(message(MessageRole.ASSISTANT, "committed"))
    reopened = ConversationStore(tmp_path, session_id=store.session_id)

    assert receipts_accepted
    assert store._entries[-1].id == entry.id
    assert reopened._entries[-1] == entry
    assert [record.getMessage() for record in caplog.records].count(
        "installed-row callback failed"
    ) == 1


@pytest.mark.asyncio
async def test_installed_hook_async_writer_publishes_on_owner_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path, session_id="installed-async")
    owner_thread = threading.get_ident()
    copy_threads: list[int] = []
    observed: list[tuple[int, tuple[ConversationEntry, ...] | str]] = []
    original_copy = store._owned_installed_delta

    def tracked_copy(
        delta: tuple[ConversationEntry, ...] | str,
    ) -> tuple[ConversationEntry, ...] | str:
        copy_threads.append(threading.get_ident())
        return original_copy(delta)

    def installed(delta: tuple[ConversationEntry, ...] | str) -> None:
        observed.append((threading.get_ident(), delta))
        assert delta != "reset"
        delta[0].data["message"]["content"][0]["text"] = "corrupted"

    monkeypatch.setattr(type(store), "_owned_installed_delta", staticmethod(tracked_copy))
    store.on_entries_installed = installed

    entry = await store.append_message_async(
        message(MessageRole.ASSISTANT, "worker")
    )

    assert copy_threads
    assert owner_thread not in copy_threads
    assert observed[0][0] == owner_thread
    assert observed[0][1][0].id == entry.id
    assert store._entries[-1].data["message"]["content"][0]["text"] == "worker"


def test_installed_hook_incremental_reload_emits_foreign_suffix_once(
    tmp_path: Path,
) -> None:
    resident = ConversationStore(tmp_path, session_id="installed-incremental")
    foreign = ConversationStore(tmp_path, session_id=resident.session_id)
    observed: list[tuple[ConversationEntry, ...] | str] = []
    resident.on_entries_installed = observed.append

    first = foreign.append_message(message(MessageRole.ASSISTANT, "one"))
    second = foreign.append_message(message(MessageRole.ASSISTANT, "two"))
    resident.refresh()
    resident.refresh()

    assert [[row.id for row in delta] for delta in observed] == [
        [first.id, second.id]
    ]


def test_installed_hook_full_prefix_reload_emits_only_suffix(tmp_path: Path) -> None:
    resident = ConversationStore(tmp_path, session_id="installed-full-prefix")
    resident.append_message(message(MessageRole.ASSISTANT, "old"))
    foreign = ConversationStore(tmp_path, session_id=resident.session_id)
    appended = foreign.append_message(message(MessageRole.ASSISTANT, "new"))
    replacement = resident.path.with_suffix(".replacement")
    replacement.write_bytes(resident.path.read_bytes())
    replacement.replace(resident.path)
    observed: list[tuple[ConversationEntry, ...] | str] = []
    resident.on_entries_installed = observed.append

    resident.refresh()

    assert [[row.id for row in delta] for delta in observed] == [[appended.id]]


def test_installed_hook_unchanged_full_reload_emits_nothing(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="installed-unchanged")
    store.append_message(message(MessageRole.ASSISTANT, "same"))
    replacement = store.path.with_suffix(".replacement")
    replacement.write_bytes(store.path.read_bytes())
    replacement.replace(store.path)
    observed: list[tuple[ConversationEntry, ...] | str] = []
    store.on_entries_installed = observed.append

    store.refresh()

    assert observed == []


def test_installed_hook_changed_content_with_same_identity_emits_reset(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path, session_id="installed-content-reset")
    original = store.append_message(message(MessageRole.ASSISTANT, "before"))
    lines = store.path.read_bytes().splitlines(keepends=True)
    replacement_row = json.loads(lines[1])
    replacement_row["data"]["message"]["content"][0]["text"] = "after"
    assert replacement_row["seq"] == original.seq
    assert replacement_row["id"] == original.id
    replacement = store.path.with_suffix(".replacement")
    replacement.write_bytes(
        lines[0] + json.dumps(replacement_row, separators=(",", ":")).encode() + b"\n"
    )
    replacement.replace(store.path)
    observed: list[tuple[ConversationEntry, ...] | str] = []
    store.on_entries_installed = observed.append

    store.refresh()

    assert observed == ["reset"]
    assert store._entries[0].data["message"]["content"][0]["text"] == "after"


def test_installed_hook_non_prefix_reload_emits_reset(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="installed-reset")
    original = store.append_message(message(MessageRole.ASSISTANT, "old"))
    final = store.append_message(message(MessageRole.ASSISTANT, "same tip"))
    rows = store.path.read_text(encoding="utf-8").splitlines()
    replacement_first = json.loads(rows[1])
    replacement_final = json.loads(rows[2])
    replacement_first["id"] = "replacement-id"
    replacement_final["parent_id"] = "replacement-id"
    assert replacement_first["id"] != original.id
    assert replacement_final["id"] == final.id
    replacement = store.path.with_suffix(".replacement")
    replacement.write_text(
        "\n".join(
            [
                rows[0],
                json.dumps(replacement_first, separators=(",", ":")),
                json.dumps(replacement_final, separators=(",", ":")),
                "",
            ]
        ),
        encoding="utf-8",
    )
    replacement.replace(store.path)
    observed: list[tuple[ConversationEntry, ...] | str] = []
    store.on_entries_installed = observed.append

    store.refresh()

    assert observed == ["reset"]
    assert store._entries[0].id == "replacement-id"
    assert store._entries[-1].id == final.id


def test_installed_hook_fork_keeps_log_order_with_backward_parent(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path, session_id="installed-fork")
    checkpoint = store.append_checkpoint("base")
    store.append_message(message(MessageRole.ASSISTANT, "discarded"))
    observed: list[tuple[ConversationEntry, ...] | str] = []
    store.on_entries_installed = observed.append

    fork = store.append_fork(str(checkpoint.seq))

    assert observed == [(store._entries[-1],)]
    assert fork.seq > checkpoint.seq
    assert fork.parent_id == checkpoint.id


def test_installed_hook_does_not_access_resident_branch(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="installed-constant-time")
    entry = store.append_message(message(MessageRole.ASSISTANT, "row"))
    observed: list[tuple[ConversationEntry, ...] | str] = []
    store.on_entries_installed = observed.append

    class BranchAccessGuard(list[ConversationEntry]):
        def __iter__(self):
            raise AssertionError("installed hook walked the resident branch")

        def __len__(self):
            raise AssertionError("installed hook measured the resident branch")

        def __getitem__(self, key: object):
            raise AssertionError("installed hook indexed the resident branch")

    store._entries = BranchAccessGuard(store._entries)
    store._publish_entries_installed((entry,))

    assert observed == [(entry,)]


def test_close_clears_installed_hook(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="installed-close")
    store.on_entries_installed = lambda _delta: None

    store.close()

    assert store.on_entries_installed is None


@pytest.mark.asyncio
async def test_async_append_runs_off_loop_and_is_durable_on_return(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path, session_id="async-writer")
    event_loop_thread = threading.get_ident()
    append_threads: list[int] = []
    real_fsync = os.fsync

    def observed_fsync(fd: int) -> None:
        append_threads.append(threading.get_ident())
        real_fsync(fd)

    with patch("zeta.core.store._log.os.fsync", side_effect=observed_fsync):
        entry = await store.append_message_async(with_message_origin(message(MessageRole.USER, "durable"), MessageOrigin.USER))

    assert append_threads and append_threads[0] != event_loop_thread
    reopened = ConversationStore(tmp_path, session_id="async-writer")
    assert reopened.entries[-1].id == entry.id


@pytest.mark.asyncio
@pytest.mark.parametrize("foreign_change", ["branch", "replacement"])
async def test_async_append_never_loads_transcript_on_event_loop_after_foreign_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    foreign_change: str,
) -> None:
    store = ConversationStore(tmp_path, session_id="off-loop-publish")
    checkpoint = store.append_checkpoint("base")
    store.append_message(message(MessageRole.ASSISTANT, "discarded"))
    owner_thread = threading.get_ident()
    load_threads: list[int] = []
    original_load_full = ConversationStore._load_full

    def observed_load_full(candidate: ConversationStore) -> None:
        load_threads.append(threading.get_ident())
        original_load_full(candidate)

    monkeypatch.setattr(ConversationStore, "_load_full", observed_load_full)
    if foreign_change == "branch":
        foreign = ConversationStore(tmp_path, session_id=store.session_id)
        foreign.append_fork(str(checkpoint.seq))
        foreign.close()
    else:
        replacement = store.path.with_suffix(".replacement")
        replacement.write_bytes(store.path.read_bytes().splitlines(keepends=True)[0])
        replacement.replace(store.path)

    load_threads.clear()
    await store.append_message_async(message(MessageRole.ASSISTANT, "kept"))

    assert load_threads
    assert owner_thread not in load_threads
    fresh = ConversationStore(tmp_path, session_id=store.session_id)
    assert store.entries == fresh.entries


@pytest.mark.asyncio
async def test_async_publish_loads_large_branch_off_loop_with_bounded_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ConversationStore(tmp_path, session_id="large-off-loop-publish")
    notification_data = {
        "kind": "agent_completion",
        "child_instance_id": "root:1",
        "child_session_path": "/child",
        "description": "child",
        "status": "completed",
        "text": "done",
    }
    rows = [("warning", {"message": f"row {index}"}) for index in range(49_996)]
    rows.extend(
        [
            ("message", {"message": message(MessageRole.ASSISTANT, "history").to_dict()}),
            ("notification", notification_data),
            ("pending_prompt", {"text": "follow up", "origin": "user"}),
            (
                "checkpoint",
                {
                    "label": "large-base",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "todo_dismissed": False,
                },
            ),
        ]
    )
    store.append_many(rows)
    foreign = ConversationStore(tmp_path, session_id=store.session_id)
    foreign.append_message(message(MessageRole.ASSISTANT, "discarded"))
    foreign.append_fork("large-base")
    foreign.close()

    owner_thread = threading.get_ident()
    load_threads: list[int] = []
    active_rows_by_thread: dict[int, int] = {}
    install_threads: list[int] = []
    loop_entry_accesses = 0
    observe_loop_access = True

    class ObservedEntries(list[ConversationEntry]):
        def _record_loop_access(self, rows: int) -> None:
            nonlocal loop_entry_accesses
            if observe_loop_access and threading.get_ident() == owner_thread:
                loop_entry_accesses += rows

        def __iter__(self):  # type: ignore[no-untyped-def]
            self._record_loop_access(len(self))
            return super().__iter__()

        def __reversed__(self):  # type: ignore[no-untyped-def]
            self._record_loop_access(len(self))
            return super().__reversed__()

        def __getitem__(self, key):  # type: ignore[no-untyped-def]
            self._record_loop_access(len(self) if isinstance(key, slice) else 1)
            return super().__getitem__(key)

        def copy(self):  # type: ignore[no-untyped-def]
            self._record_loop_access(len(self))
            return super().copy()

    original_load_full = ConversationStore._load_full

    def observed_load_full(candidate: ConversationStore) -> None:
        load_threads.append(threading.get_ident())
        original_load_full(candidate)

    monkeypatch.setattr(ConversationStore, "_load_full", observed_load_full)

    original_record_active_entry = ConversationStore._record_active_entry

    def observed_record_active_entry(
        candidate: ConversationStore, entry: ConversationEntry
    ) -> None:
        thread = threading.get_ident()
        active_rows_by_thread[thread] = active_rows_by_thread.get(thread, 0) + 1
        original_record_active_entry(candidate, entry)

    monkeypatch.setattr(
        ConversationStore, "_record_active_entry", observed_record_active_entry
    )
    original_detach = ConversationStore._detach_log_state

    def observed_detach(candidate: ConversationStore):
        state = original_detach(candidate)
        if threading.get_ident() != owner_thread:
            state = replace(state, entries=ObservedEntries(state.entries))
        return state

    monkeypatch.setattr(ConversationStore, "_detach_log_state", observed_detach)
    original_install = ConversationStore._install_log_state

    def observed_install(candidate: ConversationStore, state: object):
        install_threads.append(threading.get_ident())
        return original_install(candidate, state)

    monkeypatch.setattr(ConversationStore, "_install_log_state", observed_install)
    # Guard the outgoing resident branch too: the loop detaches it during install.
    store._entries = ObservedEntries(store._entries)

    await store.append_message_async(message(MessageRole.ASSISTANT, "kept"))
    observe_loop_access = False

    assert load_threads
    assert all(thread != owner_thread for thread in load_threads)
    assert active_rows_by_thread.get(owner_thread, 0) == 0
    assert sum(active_rows_by_thread.values()) >= len(rows)
    assert install_threads == [owner_thread]
    assert loop_entry_accesses == 0
    fresh = ConversationStore(tmp_path, session_id=store.session_id)
    assert store.entries == fresh.entries
    assert store.messages() == fresh.messages()
    assert store.agent_notifications() == fresh.agent_notifications()
    assert (
        store._active_pending_prompt_entries
        == fresh._active_pending_prompt_entries
    )


@pytest.mark.asyncio
async def test_async_publish_matches_fresh_load_when_owner_writes_after_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ConversationStore(tmp_path, session_id="publish-generation")
    snapshot_ready = threading.Event()
    release_snapshot = threading.Event()
    owner_thread = threading.get_ident()
    original_detach = ConversationStore._detach_log_state
    blocked = False

    def block_first_worker_snapshot(candidate: ConversationStore):
        nonlocal blocked
        state = original_detach(candidate)
        if not blocked and threading.get_ident() != owner_thread:
            blocked = True
            snapshot_ready.set()
            assert release_snapshot.wait(timeout=5)
        return state

    monkeypatch.setattr(
        ConversationStore, "_detach_log_state", block_first_worker_snapshot
    )
    write = asyncio.create_task(
        store.append_message_async(message(MessageRole.ASSISTANT, "worker"))
    )
    assert await asyncio.to_thread(snapshot_ready.wait, 2)
    store.append_message(message(MessageRole.ASSISTANT, "owner"))
    release_snapshot.set()
    await write

    fresh = ConversationStore(tmp_path, session_id=store.session_id)
    assert store._detach_log_state() == fresh._detach_log_state()
    assert [item.content[0].text for item in store.messages()] == ["worker", "owner"]


@pytest.mark.asyncio
async def test_async_publish_matches_fresh_load_when_owner_writes_before_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ConversationStore(tmp_path, session_id="owner-before-commit")
    worker_ready = threading.Event()
    release_worker = threading.Event()
    original_write = ConversationStore._run_durable_write

    def blocked_write(candidate: ConversationStore, *args: object):
        worker_ready.set()
        assert release_worker.wait(timeout=5)
        return original_write(candidate, *args)

    monkeypatch.setattr(ConversationStore, "_run_durable_write", blocked_write)
    write = asyncio.create_task(
        store.append_message_async(message(MessageRole.ASSISTANT, "worker"))
    )
    assert await asyncio.to_thread(worker_ready.wait, 2)
    store.append_message(message(MessageRole.ASSISTANT, "owner"))
    release_worker.set()
    await write

    fresh = ConversationStore(tmp_path, session_id=store.session_id)
    assert store._detach_log_state() == fresh._detach_log_state()
    assert [item.content[0].text for item in store.messages()] == ["owner", "worker"]


@pytest.mark.asyncio
async def test_async_publish_matches_fresh_load_after_each_owner_worker_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ConversationStore(tmp_path, session_id="many-owner-worker-races")
    worker_ready = threading.Event()
    release_worker = threading.Event()
    original_write = ConversationStore._run_durable_write
    block_before_commit = True

    def blocked_write(candidate: ConversationStore, *args: object):
        if block_before_commit:
            worker_ready.set()
            assert release_worker.wait(timeout=5)
            return original_write(candidate, *args)
        outcome = original_write(candidate, *args)
        worker_ready.set()
        assert release_worker.wait(timeout=5)
        return outcome

    monkeypatch.setattr(ConversationStore, "_run_durable_write", blocked_write)
    for index in range(20):
        block_before_commit = index % 2 == 0
        worker_ready.clear()
        release_worker.clear()
        write = asyncio.create_task(
            store.append_message_async(
                message(MessageRole.ASSISTANT, f"worker {index}")
            )
        )
        assert await asyncio.to_thread(worker_ready.wait, 2)
        store.append_message(message(MessageRole.ASSISTANT, f"owner {index}"))
        release_worker.set()
        await write

        fresh = ConversationStore(tmp_path, session_id=store.session_id)
        assert store._detach_log_state() == fresh._detach_log_state()


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_change", ["bash_cwd", "todos", "agent", "lifecycle"])
async def test_durable_publish_preserves_newer_owner_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner_change: str,
) -> None:
    store = ConversationStore(tmp_path, session_id=f"durable-{owner_change}")
    commit_ready = threading.Event()
    release_commit = threading.Event()
    original_write = ConversationStore._run_durable_write

    def blocked_write(candidate: ConversationStore, *args: object):
        outcome = original_write(candidate, *args)
        commit_ready.set()
        assert release_commit.wait(timeout=5)
        return outcome

    monkeypatch.setattr(ConversationStore, "_run_durable_write", blocked_write)
    write = asyncio.create_task(
        store.append_message_async(message(MessageRole.ASSISTANT, "worker"))
    )
    assert await asyncio.to_thread(commit_ready.wait, 2)

    if owner_change == "bash_cwd":
        store.set_bash_cwd("/new")
    elif owner_change == "todos":
        store.set_todo_items([{"content": "new", "status": "pending"}])
    elif owner_change == "agent":
        store.mark_agent_parent("owner-parent", agent_type="worker")
    else:
        store.start_agent_lifecycle(
            handle="root:1",
            started_at="2026-01-01T00:00:00+00:00",
            depth=1,
            agent_type="worker",
            description="owner lifecycle",
        )

    release_commit.set()
    await write

    fresh = ConversationStore(tmp_path, session_id=store.session_id)
    assert store._detach_log_state() == fresh._detach_log_state()
    assert store.bash_cwd == fresh.bash_cwd
    assert store.todo_items() == fresh.todo_items()
    assert store.todo_dismissed == fresh.todo_dismissed
    assert store._agent_counter == fresh._agent_counter
    assert store.agent_children() == fresh.agent_children()
    assert store.agent_canceled() == fresh.agent_canceled()
    assert store._agent_parent == fresh._agent_parent
    assert store.agent_lifecycle() == fresh.agent_lifecycle()


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_change", ["bash_cwd", "todos", "agent", "lifecycle"])
async def test_off_loop_refresh_preserves_newer_owner_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner_change: str,
) -> None:
    store = ConversationStore(tmp_path, session_id=f"refresh-{owner_change}")
    snapshot_ready = threading.Event()
    release_snapshot = threading.Event()
    original_load = ConversationStore._load_detached_store_state

    def blocked_load(candidate: ConversationStore):
        state = original_load(candidate)
        snapshot_ready.set()
        assert release_snapshot.wait(timeout=5)
        return state

    monkeypatch.setattr(ConversationStore, "_load_detached_store_state", blocked_load)
    refresh = asyncio.create_task(store._refresh_off_loop())
    assert await asyncio.to_thread(snapshot_ready.wait, 2)

    if owner_change == "bash_cwd":
        store.set_bash_cwd("/new")
    elif owner_change == "todos":
        store.set_todo_items([{"content": "new", "status": "pending"}])
    elif owner_change == "agent":
        store.mark_agent_parent("owner-parent", agent_type="worker")
    else:
        store.start_agent_lifecycle(
            handle="root:1",
            started_at="2026-01-01T00:00:00+00:00",
            depth=1,
            agent_type="worker",
            description="owner lifecycle",
        )

    release_snapshot.set()
    await refresh

    fresh = ConversationStore(tmp_path, session_id=store.session_id)
    assert store._detach_log_state() == fresh._detach_log_state()
    assert store.bash_cwd == fresh.bash_cwd
    assert store.todo_items() == fresh.todo_items()
    assert store.todo_dismissed == fresh.todo_dismissed
    assert store._agent_counter == fresh._agent_counter
    assert store.agent_children() == fresh.agent_children()
    assert store.agent_canceled() == fresh.agent_canceled()
    assert store._agent_parent == fresh._agent_parent
    assert store.agent_lifecycle() == fresh.agent_lifecycle()


@pytest.mark.asyncio
async def test_off_loop_refresh_is_bounded_under_owner_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ConversationStore(tmp_path, session_id="bounded-refresh")
    original_load = ConversationStore._load_detached_store_state
    loop = asyncio.get_running_loop()
    loads = 0

    def load_during_owner_write(candidate: ConversationStore):
        nonlocal loads
        loads += 1
        if loads > 3:
            raise AssertionError("publication retried without a bound")
        state = original_load(candidate)
        changed = threading.Event()
        loop.call_soon_threadsafe(
            lambda: (store.set_bash_cwd(f"/owner-{loads}"), changed.set())
        )
        assert changed.wait(timeout=2)
        return state

    monkeypatch.setattr(
        ConversationStore, "_load_detached_store_state", load_during_owner_write
    )
    await asyncio.wait_for(store._refresh_off_loop(), timeout=2)

    assert loads == 1
    assert store.bash_cwd == "/owner-1"


@pytest.mark.asyncio
async def test_async_publication_cannot_be_starved_by_owner_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ConversationStore(tmp_path, session_id="bounded-durable-publication")
    snapshot_ready = threading.Event()
    release_snapshot = threading.Event()
    original_write = ConversationStore._run_durable_write

    def blocked_write(candidate: ConversationStore, *args: object):
        outcome = original_write(candidate, *args)
        snapshot_ready.set()
        assert release_snapshot.wait(timeout=5)
        return outcome

    monkeypatch.setattr(ConversationStore, "_run_durable_write", blocked_write)
    loop = asyncio.get_running_loop()
    refreshes = 0

    def load_during_owner_write(candidate: ConversationStore):
        nonlocal refreshes
        refreshes += 1
        if refreshes > 3:
            raise AssertionError("durable publication retried without a bound")
        reader = ConversationStore(
            candidate.root_dir,
            session_id=candidate.session_id,
            _must_exist=True,
        )
        try:
            state = reader._detach_log_state()
        finally:
            reader.close()
        changed = threading.Event()
        loop.call_soon_threadsafe(
            lambda: (
                store.append_message(
                    message(MessageRole.ASSISTANT, f"stream {refreshes}")
                ),
                changed.set(),
            )
        )
        assert changed.wait(timeout=2)
        return state

    monkeypatch.setattr(
        ConversationStore,
        "_load_detached_log_state",
        load_during_owner_write,
        raising=False,
    )
    write = asyncio.create_task(
        store.append_message_async(message(MessageRole.ASSISTANT, "worker"))
    )
    assert await asyncio.to_thread(snapshot_ready.wait, 2)
    for index in range(20):
        store.append_message(message(MessageRole.ASSISTANT, f"owner {index}"))
    release_snapshot.set()
    await asyncio.wait_for(write, timeout=2)

    fresh = ConversationStore(tmp_path, session_id=store.session_id)
    assert refreshes == 1
    assert store.entries == fresh.entries
    assert [item.content[0].text for item in store.messages()] == [
        "worker",
        *(f"owner {index}" for index in range(20)),
        "stream 1",
    ]


@pytest.mark.asyncio
async def test_off_loop_refresh_cancellation_is_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ConversationStore(tmp_path, session_id="cancel-refresh")
    started = threading.Event()
    release = threading.Event()
    original_load = ConversationStore._load_detached_store_state

    def blocked_load(candidate: ConversationStore):
        started.set()
        assert release.wait(timeout=5)
        return original_load(candidate)

    monkeypatch.setattr(ConversationStore, "_load_detached_store_state", blocked_load)
    refresh = asyncio.create_task(store._refresh_off_loop())
    assert await asyncio.to_thread(started.wait, 2)
    refresh.cancel()
    await asyncio.sleep(0.05)
    try:
        assert refresh.done()
        with pytest.raises(asyncio.CancelledError):
            await refresh
    finally:
        release.set()
        await asyncio.gather(refresh, return_exceptions=True)


@pytest.mark.asyncio
async def test_async_writes_reuse_one_detached_store_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path, session_id="persistent-writer")
    open_count = 0
    original_init = ConversationStore.__init__

    def count_open(instance: ConversationStore, *args: object, **kwargs: object) -> None:
        nonlocal open_count
        if kwargs.get("_must_exist"):
            open_count += 1
        original_init(instance, *args, **kwargs)

    monkeypatch.setattr(ConversationStore, "__init__", count_open)

    await asyncio.gather(
        store._to_thread_durable(store.append_pending_prompt, "follow up"),
        store._to_thread_durable(
            store.append_agent_notification,
            "root:1",
            child_session_path="/child",
            description="child",
            status="completed",
            text="done",
        ),
    )

    assert open_count == 1


@pytest.mark.asyncio
async def test_async_write_failure_discards_writer_before_next_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path, session_id="failed-writer")
    open_count = 0
    original_open = store._new_persistent_durable_writer

    def count_open() -> ConversationStore:
        nonlocal open_count
        open_count += 1
        return original_open()

    monkeypatch.setattr(store, "_new_persistent_durable_writer", count_open)
    real_fsync = os.fsync
    failed = True

    def fail_once(fd: int) -> None:
        nonlocal failed
        if failed:
            failed = False
            raise OSError("injected fsync failure")
        real_fsync(fd)

    with patch("zeta.core.store._log.os.fsync", side_effect=fail_once):
        with pytest.raises(OSError, match="injected fsync failure"):
            await store._to_thread_durable(store.append_pending_prompt, "failed")
        await store._to_thread_durable(store.append_pending_prompt, "succeeds")

    assert open_count == 2
    assert [entry.data["text"] for entry in store._entries] == ["failed", "succeeds"]


@pytest.mark.asyncio
async def test_mixed_async_writes_keep_fifo_order(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="persistent-fifo")

    await asyncio.gather(
        store.append_message_async(
            with_message_origin(message(MessageRole.USER, "first"), MessageOrigin.USER)
        ),
        store._to_thread_durable(store.append_pending_prompt, "second"),
        store.append_message_async(
            with_message_origin(message(MessageRole.USER, "third"), MessageOrigin.USER)
        ),
    )

    assert [entry.seq for entry in store._entries] == [1, 2, 3]


@pytest.mark.asyncio
async def test_async_write_cancelled_before_worker_start_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path, session_id="cancel-before-writer")
    started = asyncio.Event()

    def block_start(*args: object, **kwargs: object) -> asyncio.Future[object]:
        started.set()
        raise AssertionError("worker started")

    monkeypatch.setattr(store, "_start_durable_write", block_start)
    task = asyncio.create_task(store._to_thread_durable(store.append_pending_prompt, "nope"))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not started.is_set()
    assert store._entries == []


@pytest.mark.asyncio
async def test_async_write_cancelled_after_worker_start_is_durable(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path, session_id="cancel-after-writer")
    fsync_started = threading.Event()
    release_fsync = threading.Event()

    def blocked_fsync(_fd: int) -> None:
        fsync_started.set()
        assert release_fsync.wait(timeout=5)

    with patch("zeta.core.store._log.os.fsync", side_effect=blocked_fsync):
        task = asyncio.create_task(store._to_thread_durable(store.append_pending_prompt, "durable"))
        assert await asyncio.to_thread(fsync_started.wait, 2)
        task.cancel()
        release_fsync.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert store._entries


@pytest.mark.asyncio
async def test_close_drains_and_releases_persistent_writer(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="close-writer")
    await store._to_thread_durable(store.append_pending_prompt, "durable")

    store.close()

    assert store._closed


@pytest.mark.asyncio
async def test_async_publish_matches_fresh_load_after_interleaved_appends(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path, session_id="interleaved-publish")
    foreign = ConversationStore(tmp_path, session_id=store.session_id)

    for index in range(20):
        foreign.append_message(message(MessageRole.ASSISTANT, f"foreign {index}"))
        await store.append_message_async(message(MessageRole.ASSISTANT, f"own {index}"))

    await store._to_thread_durable(store.append_pending_prompt, "follow up")
    await store._to_thread_durable(
        store.append_agent_notification,
        "root:1",
        child_session_path="/child",
        description="child",
        status="completed",
        text="done",
    )

    fresh = ConversationStore(tmp_path, session_id=store.session_id)
    assert store.entries == fresh.entries
    assert store.messages() == fresh.messages()
    assert store.agent_notifications() == fresh.agent_notifications()
    assert (
        store._active_pending_prompt_entries
        == fresh._active_pending_prompt_entries
    )


@pytest.mark.asyncio
async def test_async_batch_is_never_partially_visible_in_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path, session_id="atomic-delivery")
    notification = store.append_agent_notification(
        child_instance_id="child-1",
        child_session_path="/tmp/child-1",
        description="child",
        status="completed",
        text="done",
    )
    install_blocked = threading.Event()
    release_install = threading.Event()
    visible_during_install: set[str] = set()

    def block_before_install() -> None:
        install_blocked.set()
        assert release_install.wait(timeout=5)

    def observe_while_blocked() -> None:
        assert install_blocked.wait(timeout=2)
        visible_during_install.update(entry.type for entry in store.entries)
        release_install.set()

    monkeypatch.setattr(store, "_before_incremental_state_install", block_before_install)
    observer = threading.Thread(target=observe_while_blocked)
    observer.start()
    await store.record_agent_notification_delivery_async(
        notification.id, presented=True, acknowledged=True
    )
    observer.join(timeout=2)

    markers = {"notification_tui_presented", "notification_ack"}
    assert not (visible_during_install & markers) or markers <= visible_during_install
    assert markers <= {entry.type for entry in store.entries}


@pytest.mark.asyncio
async def test_async_append_does_not_mutate_resident_store_off_loop(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path, session_id="loop-owned-state")
    event_loop_thread = threading.get_ident()
    mutation_threads: list[int] = []

    class ObservedEntries(list[ConversationEntry]):
        def append(self, entry: ConversationEntry) -> None:
            mutation_threads.append(threading.get_ident())
            super().append(entry)

        def extend(self, entries: list[ConversationEntry]) -> None:
            mutation_threads.append(threading.get_ident())
            super().extend(entries)

    store._entries = ObservedEntries(store._entries)
    store._before_incremental_state_install = lambda: mutation_threads.append(
        threading.get_ident()
    )
    await store.append_message_async(with_message_origin(message(MessageRole.USER, "loop-owned"), MessageOrigin.USER))

    assert mutation_threads
    assert set(mutation_threads) == {event_loop_thread}


def test_notification_delivery_batch_matches_separate_row_order(
    tmp_path: Path,
) -> None:
    separate = ConversationStore(tmp_path, session_id="separate-delivery")
    separate_notification = separate.append_agent_notification(
        child_instance_id="child-1",
        child_session_path="/tmp/child-1",
        description="child",
        status="completed",
        text="done",
    )
    separate.mark_agent_notification_presented_to_tui(separate_notification.id)
    separate.acknowledge_agent_notification(separate_notification.id)

    batched = ConversationStore(tmp_path, session_id="batched-delivery")
    batched_notification = batched.append_agent_notification(
        child_instance_id="child-1",
        child_session_path="/tmp/child-1",
        description="child",
        status="completed",
        text="done",
    )
    with patch("zeta.core.store._log.os.fsync") as fsync:
        batched.record_agent_notification_delivery(
            batched_notification.id, presented=True, acknowledged=True
        )
    fsync.assert_called_once()

    assert (
        [entry.type for entry in separate.entries]
        == [entry.type for entry in batched.entries]
        == ["notification", "notification_tui_presented", "notification_ack"]
    )
    assert separate.entries[0].data == batched.entries[0].data
    assert [
        {**entry.data, "notification_id": "notification"}
        for entry in separate.entries[1:]
    ] == [
        {**entry.data, "notification_id": "notification"}
        for entry in batched.entries[1:]
    ]


def test_legacy_notification_defaults_to_unpresented_in_tui(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    notification = store.append_agent_notification(
        child_instance_id="legacy-child",
        child_session_path="/tmp/legacy-child",
        description="legacy notification",
        status="completed",
        text="done",
    )

    assert not store.is_agent_notification_presented_to_tui(notification.id)


def test_bash_cwd_serializes_in_separate_state_file(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, cwd=tmp_path, bash_cwd="/tmp")

    store.set_bash_cwd("/var/tmp")
    reopened = ConversationStore(tmp_path, session_id=store.session_id)

    assert reopened.bash_cwd == "/var/tmp"
    state = json.loads(reopened.state_path.read_text(encoding="utf-8"))
    assert state == {"bash_cwd": "/var/tmp"}


def test_bash_cwd_state_write_failure_preserves_conversation(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(with_message_origin(message(MessageRole.USER, "kept"), MessageOrigin.USER))
    before = store.path.read_bytes()

    with patch(
        "zeta.core.store._store.os.replace",
        side_effect=OSError("injected replace failure"),
    ):
        with pytest.raises(OSError, match="injected replace failure"):
            store.set_bash_cwd("/tmp")

    assert store.path.read_bytes() == before
    assert store.bash_cwd == str(Path.cwd())


@pytest.mark.parametrize(
    ("damage", "kind", "reason_code"),
    [
        ("parse", "invalid_row", "invalid_json"),
        ("payload", "invalid_row", "invalid_payload"),
        ("links", "broken_links", "missing_parent"),
        ("approval", "broken_links", "unlinked_approval_resolution"),
    ],
)
def test_corrupt_transcript_diagnosis_localizes_first_bad_row(
    tmp_path: Path, damage: str, kind: str, reason_code: str
) -> None:
    store = ConversationStore(tmp_path, session_id=f"diagnose-{damage}")
    store.append_message(message(MessageRole.USER, "first"))
    store.append_message(message(MessageRole.ASSISTANT, "second"))
    path = store.path
    lines = path.read_bytes().splitlines(keepends=True)
    expected_offset = len(lines[0]) + len(lines[1])
    row = json.loads(lines[2])
    if damage == "parse":
        lines[2] = b'{"private":"must-not-leak"\n'
    elif damage == "payload":
        row["data"] = {"private": "must-not-leak"}
        lines[2] = json.dumps(row).encode() + b"\n"
    elif damage == "links":
        row["parent_id"] = "must-not-leak"
        lines[2] = json.dumps(row).encode() + b"\n"
    else:
        row["type"] = "approval_resolution"
        row["data"] = {"request_id": "must-not-leak", "decision": "deny"}
        lines[2] = json.dumps(row).encode() + b"\n"
    damaged = b"".join(lines)
    path.write_bytes(damaged)

    with pytest.raises(ConversationIntegrityError) as raised:
        ConversationStore(tmp_path, session_id=store.session_id, _read_only=True)

    error = raised.value
    expected_label = {
        "parse": "terminated invalid conversation row",
        "payload": "invalid payload for conversation entry",
        "links": "missing prior parent",
        "approval": "approval resolution is not linked",
    }[damage]
    assert str(error) == f"transcript {expected_label} at row 3 (session diagnose-{damage})"
    assert (error.kind, error.row_number, error.byte_offset) == (
        kind,
        3,
        expected_offset,
    )
    assert error.valid_prefix_end == expected_offset
    assert error.reason_code == reason_code
    stat = path.stat()
    assert (error.source_device, error.source_inode, error.source_size) == (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
    )
    assert error.prefix_digest == hashlib.sha256(damaged[:expected_offset]).hexdigest()
    assert "must-not-leak" not in str(error)
    assert str(path) not in str(error)
    assert path.read_bytes() == damaged


@pytest.mark.parametrize("damage", ["future_schema", "future_entry"])
def test_transcript_compatibility_classification_precedes_corruption(
    tmp_path: Path, damage: str
) -> None:
    store = ConversationStore(tmp_path, session_id=f"diagnose-{damage}")
    store.append_message(message(MessageRole.USER, "first"))
    lines = store.path.read_bytes().splitlines(keepends=True)
    row_index = 0 if damage == "future_schema" else 1
    row = json.loads(lines[row_index])
    if damage == "future_schema":
        row["data"]["schema"] = "zeta.conversation.v999"
    else:
        row["type"] = "future_entry"
        row["data"] = {"message": "not validated for a future type"}
    lines[row_index] = json.dumps(row).encode() + b"\n"
    lines.append(b'{"later":"corruption"\n')
    store.path.write_bytes(b"".join(lines))

    with pytest.raises(ConversationIntegrityError) as raised:
        ConversationStore(tmp_path, session_id=store.session_id, _read_only=True)

    assert str(raised.value) == (
        f"transcript unsupported format at row {row_index + 1} "
        f"(session diagnose-{damage})"
    )
    assert raised.value.kind == "unsupported_format"
    assert raised.value.row_number == row_index + 1
    assert raised.value.repairable is False


def test_torn_tail_is_dropped_with_warning_entry(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(with_message_origin(message(MessageRole.USER, "kept"), MessageOrigin.USER))
    with store.path.open("ab") as handle:
        handle.write(b'{"seq": 3, "id": "torn"')

    with pytest.warns(RuntimeWarning, match="torn"):
        reopened = ConversationStore(tmp_path, session_id=store.session_id)

    assert [item.content[0].text for item in reopened.messages()] == ["kept"]
    assert reopened.entries[-1].type == "warning"
    warning_id = reopened.entries[-1].id
    warning_seq = reopened.entries[-1].seq
    assert reopened.path.read_bytes().endswith(b"\n")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reopened_again = ConversationStore(tmp_path, session_id=reopened.session_id)

    assert caught == []
    assert reopened_again.entries[-1].id == warning_id
    assert reopened_again.entries[-1].seq == warning_seq
    assert reopened_again.entries[-1].parent_id == reopened.entries[-1].parent_id
    assert reopened_again.path.read_bytes() == reopened.path.read_bytes()


def test_terminated_invalid_tail_is_not_repaired(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    with store.path.open("ab") as handle:
        handle.write(b'{"invalid": }\n')

    with pytest.raises(
        ConversationIntegrityError, match="terminated invalid conversation row at row 2"
    ):
        ConversationStore(tmp_path, session_id=store.session_id)


@pytest.mark.parametrize(
    ("field", "value"),
    [("seq", "1"), ("id", ""), ("parent_id", 0), ("lane", "side")],
)
def test_invalid_entry_fields_are_rejected(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(with_message_origin(message(MessageRole.USER, "bad"), MessageOrigin.USER))
    rows = store.path.read_text().splitlines()
    row = json.loads(rows[-1])
    row[field] = value
    rows[-1] = json.dumps(row, separators=(",", ":"), sort_keys=True)
    store.path.write_text("\n".join(rows) + "\n")

    with pytest.raises(ConversationIntegrityError):
        ConversationStore(tmp_path, session_id=store.session_id)


@pytest.mark.parametrize("tail", [b"\xff\n", b"\xff"])
def test_non_utf8_tail_uses_termination_rule(tmp_path: Path, tail: bytes) -> None:
    store = ConversationStore(tmp_path)
    with store.path.open("ab") as handle:
        handle.write(tail)

    if tail.endswith(b"\n"):
        with pytest.raises(ConversationIntegrityError, match="terminated"):
            ConversationStore(tmp_path, session_id=store.session_id)
    else:
        with pytest.warns(RuntimeWarning, match="torn"):
            reopened = ConversationStore(tmp_path, session_id=store.session_id)
        assert reopened.entries[-1].type == "warning"


def test_huge_integer_row_is_rejected_with_typed_error(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    with store.path.open("ab") as handle:
        handle.write(b'{"seq":' + b"9" * 5000 + b"}\n")

    with pytest.raises(ConversationIntegrityError):
        ConversationStore(tmp_path, session_id=store.session_id)


def test_append_and_replay_use_data_snapshots(tmp_path: Path) -> None:
    arguments = {"nested": {"value": 1}}
    call = ToolCall("call-1", "tool", arguments)
    store = ConversationStore(
        tmp_path,
        session_id="snapshot",
    )
    entry = store.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(call)]))

    arguments["nested"]["value"] = 2
    entry.data["message"]["content"][0]["tool_call"]["arguments"]["nested"]["value"] = 4
    replay = store.replay()
    replay[0].data["message"]["content"][0]["tool_call"]["arguments"]["nested"][
        "value"
    ] = 3

    fresh = store.replay()
    assert (
        fresh[0].data["message"]["content"][0]["tool_call"]["arguments"]["nested"][
            "value"
        ]
        == 1
    )


def test_missing_nested_message_field_is_rejected_on_load(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(with_message_origin(message(MessageRole.USER, "hello"), MessageOrigin.USER))
    rows = store.path.read_text().splitlines()
    row = json.loads(rows[-1])
    del row["data"]["message"]["content"][0]["text"]
    rows[-1] = json.dumps(row, separators=(",", ":"), sort_keys=True)
    store.path.write_text("\n".join(rows) + "\n")

    with pytest.raises(ConversationIntegrityError):
        ConversationStore(tmp_path, session_id=store.session_id)


def test_bad_nested_role_is_rejected_on_load(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(with_message_origin(message(MessageRole.USER, "hello"), MessageOrigin.USER))
    rows = store.path.read_text().splitlines()
    row = json.loads(rows[-1])
    row["data"]["message"]["role"] = "not-a-role"
    rows[-1] = json.dumps(row, separators=(",", ":"), sort_keys=True)
    store.path.write_text("\n".join(rows) + "\n")

    with pytest.raises(ConversationIntegrityError):
        ConversationStore(tmp_path, session_id=store.session_id)


def test_numeric_nested_text_is_rejected_on_load(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(with_message_origin(message(MessageRole.USER, "hello"), MessageOrigin.USER))
    rows = store.path.read_text().splitlines()
    row = json.loads(rows[-1])
    row["data"]["message"]["content"][0]["text"] = 42
    rows[-1] = json.dumps(row, separators=(",", ":"), sort_keys=True)
    store.path.write_text("\n".join(rows) + "\n")

    with pytest.raises(ConversationIntegrityError):
        ConversationStore(tmp_path, session_id=store.session_id)


def test_string_compaction_sequence_is_rejected_on_load(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_compaction_marker("summary", 1, 2)
    rows = store.path.read_text().splitlines()
    row = json.loads(rows[-1])
    row["data"]["source_seq_start"] = "1"
    rows[-1] = json.dumps(row, separators=(",", ":"), sort_keys=True)
    store.path.write_text("\n".join(rows) + "\n")

    with pytest.raises(ConversationIntegrityError):
        ConversationStore(tmp_path, session_id=store.session_id)


def test_compaction_marker_persists(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    marker = store.append_compaction_marker("summary", 1, 4)
    reopened = ConversationStore(tmp_path, session_id=store.session_id)

    assert reopened.replay()[-1].id == marker.id
    assert reopened.replay()[-1].data == {
        "summary": "summary",
        "source_seq_start": 1,
        "source_seq_end": 4,
        "replaces": [],
    }


def test_append_fsyncs_before_return(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    with patch("zeta.core.store._store.os.fsync") as fsync:
        store.append_message(with_message_origin(message(MessageRole.USER, "hello"), MessageOrigin.USER))

    fsync.assert_called_once()


def test_asyncio_run_shutdown_drains_blocked_append(tmp_path: Path) -> None:
    script = """
import asyncio
import os
import sys
import threading
from pathlib import Path
from unittest.mock import patch

from zeta.core.store import ConversationStore
from zeta.protocol.types import Message, MessageOrigin, MessageRole, TextContent, with_message_origin

root = Path(sys.argv[1])
store = ConversationStore(root, session_id="shutdown-drain")
fsync_started = threading.Event()
release_fsync = threading.Event()
real_fsync = os.fsync


def blocked_fsync(fd: int) -> None:
    fsync_started.set()
    if not release_fsync.wait(timeout=5):
        raise RuntimeError("timed out waiting to release fsync")
    real_fsync(fd)


async def main() -> None:
    with patch("zeta.core.store._log.os.fsync", side_effect=blocked_fsync):
        asyncio.create_task(
            store.append_message_async(
                with_message_origin(
                    Message(MessageRole.USER, [TextContent("durable")]),
                    MessageOrigin.USER,
                )
            )
        )
        while not fsync_started.is_set():
            await asyncio.sleep(0.001)
        threading.Timer(0.05, release_fsync.set).start()


asyncio.run(main())
"""

    # The timeout is a hang guard, not a speed limit: it also covers interpreter
    # start and imports on a slow runner. The durable row below is the check.
    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        check=True,
        timeout=60,
    )

    reopened = ConversationStore(tmp_path, session_id="shutdown-drain")
    assert [item.content[0].text for item in reopened.messages()] == ["durable"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_count", [2, 5])
async def test_async_append_defers_repeated_cancellation_until_fsync(
    tmp_path: Path, cancel_count: int
) -> None:
    store = ConversationStore(tmp_path)
    fsync_started = threading.Event()
    release_fsync = threading.Event()

    def blocked_fsync(_fd: int) -> None:
        fsync_started.set()
        assert release_fsync.wait(timeout=5)

    with patch("zeta.core.store._log.os.fsync", side_effect=blocked_fsync):
        append = asyncio.create_task(
            store.append_message_async(with_message_origin(message(MessageRole.USER, "durable"), MessageOrigin.USER))
        )
        assert await asyncio.to_thread(fsync_started.wait, 2)
        for _ in range(cancel_count):
            append.cancel()
            await asyncio.sleep(0)
        assert not append.done()
        release_fsync.set()
        with pytest.raises(asyncio.CancelledError):
            await append

    assert [item.content[0].text for item in store.messages()] == ["durable"]


@pytest.mark.asyncio
async def test_close_waits_for_in_flight_async_append(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    fsync_started = threading.Event()
    release_fsync = threading.Event()
    close_finished = threading.Event()

    def blocked_fsync(_fd: int) -> None:
        fsync_started.set()
        assert release_fsync.wait(timeout=5)

    def close_store() -> None:
        store.close()
        close_finished.set()

    with patch("zeta.core.store._log.os.fsync", side_effect=blocked_fsync):
        append = asyncio.create_task(
            store.append_message_async(with_message_origin(message(MessageRole.USER, "durable"), MessageOrigin.USER))
        )
        assert await asyncio.to_thread(fsync_started.wait, 2)
        close_thread = threading.Thread(target=close_store)
        close_thread.start()
        try:
            assert not close_finished.wait(timeout=0.1)
            release_fsync.set()
            await append
        finally:
            release_fsync.set()
            close_thread.join(timeout=2)

    assert close_finished.is_set()


@pytest.mark.asyncio
async def test_close_on_event_loop_drains_worker_without_deadlock(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="loop-close")
    fsync_started = threading.Event()
    release_fsync = threading.Event()

    def blocked_fsync(_fd: int) -> None:
        fsync_started.set()
        assert release_fsync.wait(timeout=5)

    with patch("zeta.core.store._log.os.fsync", side_effect=blocked_fsync):
        append = asyncio.create_task(
            store.append_message_async(with_message_origin(message(MessageRole.USER, "durable"), MessageOrigin.USER))
        )
        assert await asyncio.to_thread(fsync_started.wait, 2)
        release = threading.Timer(0.1, release_fsync.set)
        release.start()
        try:
            store.close()
            await append
        finally:
            release_fsync.set()
            release.join(timeout=2)

    reopened = ConversationStore(tmp_path, session_id=store.session_id)
    assert [item.content[0].text for item in reopened.messages()] == ["durable"]


def test_sessions_are_isolated(tmp_path: Path) -> None:
    first = ConversationStore(tmp_path)
    second = ConversationStore(tmp_path)
    first.append_message(with_message_origin(message(MessageRole.USER, "one"), MessageOrigin.USER))
    second.append_message(with_message_origin(message(MessageRole.USER, "two"), MessageOrigin.USER))

    assert first.path != second.path
    assert [item.content[0].text for item in first.messages()] == ["one"]
    assert [item.content[0].text for item in second.messages()] == ["two"]


def test_concurrent_constructors_write_one_header(tmp_path: Path) -> None:
    def construct(_: int) -> ConversationStore:
        return ConversationStore(tmp_path, session_id="race")

    with ThreadPoolExecutor(max_workers=8) as executor:
        stores = list(executor.map(construct, range(8)))

    rows = stores[0].path.read_text().splitlines()
    assert [json.loads(row)["type"] for row in rows] == ["header"]


def test_concurrent_constructors_repair_once(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="repair-race")
    store.append_message(with_message_origin(message(MessageRole.USER, "kept"), MessageOrigin.USER))
    with store.path.open("ab") as handle:
        handle.write(b'{"seq": 3, "id": "torn"')

    def construct(_: int) -> ConversationStore:
        return ConversationStore(tmp_path, session_id="repair-race")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with ThreadPoolExecutor(max_workers=8) as executor:
            stores = list(executor.map(construct, range(8)))

    rows = [json.loads(row) for row in stores[0].path.read_text().splitlines()]
    assert [row["type"] for row in rows] == ["header", "message", "warning"]
    assert all(len(store.entries) == 2 for store in stores)


def test_invalid_parent_is_rejected_on_append(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)

    with pytest.raises(ConversationIntegrityError, match="missing prior parent"):
        store.append_message(with_message_origin(message(MessageRole.USER, "bad"), MessageOrigin.USER), parent_id="missing")


def test_empty_session_id_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConversationIntegrityError, match="safe path"):
        ConversationStore(tmp_path, session_id="")


def test_invalid_sequence_and_parent_are_rejected_on_load(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(with_message_origin(message(MessageRole.USER, "kept"), MessageOrigin.USER))
    rows = store.path.read_text().splitlines()
    rows[-1] = (
        rows[-1]
        .replace('"seq":1', '"seq":3')
        .replace('"parent_id":null', '"parent_id":"missing"')
    )
    store.path.write_text("\n".join(rows) + "\n")

    with pytest.raises(ConversationIntegrityError):
        ConversationStore(tmp_path, session_id=store.session_id)


def test_self_parent_is_rejected_on_load(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(with_message_origin(message(MessageRole.USER, "self"), MessageOrigin.USER))
    rows = store.path.read_text().splitlines()
    row = json.loads(rows[-1])
    row["parent_id"] = row["id"]
    rows[-1] = json.dumps(row, separators=(",", ":"), sort_keys=True)
    store.path.write_text("\n".join(rows) + "\n")

    with pytest.raises(ConversationIntegrityError, match="prior parent"):
        ConversationStore(tmp_path, session_id=store.session_id)


def test_forward_parent_is_rejected_on_load(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(with_message_origin(message(MessageRole.USER, "first"), MessageOrigin.USER))
    store.append_message(with_message_origin(message(MessageRole.USER, "second"), MessageOrigin.USER))
    rows = store.path.read_text().splitlines()
    first_row = json.loads(rows[-2])
    second_row = json.loads(rows[-1])
    first_row["parent_id"] = second_row["id"]
    rows[-2] = json.dumps(first_row, separators=(",", ":"), sort_keys=True)
    store.path.write_text("\n".join(rows) + "\n")

    with pytest.raises(ConversationIntegrityError, match="prior parent"):
        ConversationStore(tmp_path, session_id=store.session_id)


def test_replay_detects_a_parent_cycle(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    first = store.append_message(with_message_origin(message(MessageRole.USER, "first"), MessageOrigin.USER))
    second = store.append_message(with_message_origin(message(MessageRole.USER, "second"), MessageOrigin.USER))
    store._entries[0] = replace(first, parent_id=second.id)

    with pytest.raises(ConversationIntegrityError, match="cycle"):
        store.replay()


def test_duplicate_ids_are_rejected_on_load(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    first = store.append_message(with_message_origin(message(MessageRole.USER, "one"), MessageOrigin.USER))
    store.append_message(with_message_origin(message(MessageRole.USER, "two"), MessageOrigin.USER))
    rows = store.path.read_text().splitlines()
    second_row = json.loads(rows[-1])
    second_row["id"] = first.id
    rows[-1] = json.dumps(second_row, separators=(",", ":"), sort_keys=True)
    store.path.write_text("\n".join(rows) + "\n")

    with pytest.raises(ConversationIntegrityError, match="duplicate"):
        ConversationStore(tmp_path, session_id=store.session_id)


def test_appends_do_not_reread_the_full_log(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="incremental")
    for index in range(200):
        store.append_message(with_message_origin(message(MessageRole.USER, f"seed-{index}"), MessageOrigin.USER))

    full_reloads = store._full_reload_count
    for index in range(20):
        store.append_message(with_message_origin(message(MessageRole.USER, f"new-{index}"), MessageOrigin.USER))

    assert store._full_reload_count == full_reloads


def test_live_store_tail_syncs_external_append_without_full_read(tmp_path: Path) -> None:
    first = ConversationStore(tmp_path, session_id="shared-tail")
    second = ConversationStore(tmp_path, session_id="shared-tail")
    second.append_message(with_message_origin(message(MessageRole.USER, "external"), MessageOrigin.USER))

    full_reloads = first._full_reload_count
    first.append_message(message(MessageRole.ASSISTANT, "local"))

    assert first._full_reload_count == full_reloads
    assert [item.content[0].text for item in first.messages()] == [
        "external",
        "local",
    ]


@pytest.mark.parametrize(
    ("entry_type", "data"),
    [
        ("message", {"message": {"role": "bogus", "content": []}}),
        ("fork", {}),
        ("approval_request", {"request_id": "request-1"}),
        (
            "approval_resolution",
            {"request_id": "", "decision": "allow"},
        ),
        (
            "notification",
            {
                "kind": "agent_completion",
                "child_instance_id": "",
                "child_session_path": "/child",
                "description": "child",
                "status": "completed",
                "text": "done",
            },
        ),
        ("notification_tui_presented", {"notification_id": 7}),
        ("notification_ack", {"notification_id": 7}),
        ("pending_prompt", {"text": ""}),
        ("pending_prompt_ack", {"prompt_id": 7}),
        (
            "compaction",
            {"summary": "", "source_seq_start": 1, "source_seq_end": 1},
        ),
    ],
)
def test_incremental_tail_payload_validation_matches_full_load(
    tmp_path: Path, entry_type: str, data: dict[str, object]
) -> None:
    store = ConversationStore(tmp_path, session_id=f"payload-{entry_type}")
    row = {
        "seq": 1,
        "id": f"raw-{entry_type}",
        "parent_id": None,
        "lane": "main",
        "type": entry_type,
        "data": data,
    }
    with store.path.open("ab") as handle:
        handle.write(json.dumps(row, separators=(",", ":")).encode() + b"\n")

    def accepted(load: object) -> bool:
        try:
            if callable(load):
                load()
            return True
        except ConversationIntegrityError:
            return False

    incremental_accepted = accepted(store.refresh)
    full_accepted = accepted(
        lambda: ConversationStore(tmp_path, session_id=store.session_id)
    )

    assert incremental_accepted == full_accepted
    assert not incremental_accepted


def test_approval_states_snapshot_uses_resident_indexes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "sessions"
    resident = ConversationStore(root, session_id="snapshot-race")
    foreign = ConversationStore(root, session_id="snapshot-race")
    call = ToolCall("call-1", "read", {"path": "/tmp/input"})
    foreign.append_approval_request(call.id, call)
    def fail_if_parsed(
        cls: type[ConversationEntry], value: dict[str, object]
    ) -> ConversationEntry:
        raise AssertionError(f"approval snapshot parsed {value!r}")

    monkeypatch.setattr(
        ConversationEntry, "from_dict", classmethod(fail_if_parsed)
    )
    resident._install_log_state(foreign._snapshot_log_state())
    assert resident.approval_states_snapshot() == {call.id: (call, None)}


def test_resident_store_mutation_off_owner_thread_raises(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.bind_owner_thread()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            store.append_message,
            Message(MessageRole.ASSISTANT, [TextContent("off owner")]),
        )
        with pytest.raises(
            RuntimeError, match="resident conversation store accessed off owner thread"
        ):
            future.result()


def test_approval_resolution_data_never_mutates_resident_store_off_owner(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    call = ToolCall("call-1", "read", {"path": "/tmp/input"})
    store.append_approval_request(call.id, call)
    store.bind_owner_thread()

    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(store.approval_resolution_data, call.id).result() is None


def test_pending_approvals_delivers_foreign_append_after_flock(tmp_path: Path) -> None:
    script = tmp_path / "pending_approval_reentry.py"
    script.write_text(
        '''
import json
import subprocess
import sys
from pathlib import Path

from zeta.core.store import ConversationStore
from zeta.protocol.types import Message, MessageRole, TextContent, ToolCall, ToolUseContent

root = Path(sys.argv[1])
session_id = "pending-approval-reentry"
resident = ConversationStore(root, session_id=session_id)
foreign = """
from pathlib import Path
import sys
from zeta.core.store import ConversationStore
from zeta.protocol.types import Message, MessageRole, ToolCall, ToolUseContent

store = ConversationStore(Path(sys.argv[1]), session_id=sys.argv[2])
call = ToolCall("call-1", "echo", {"value": "approved"})
store.append_message_with_approval_requests(
    Message(MessageRole.ASSISTANT, [ToolUseContent(call)]), [(call.id, call)]
)
"""
subprocess.run([sys.executable, "-c", foreign, str(root), session_id], check=True)
observed = []


def installed(delta):
    observed.append("reset" if delta == "reset" else [row.id for row in delta])
    resident.on_entries_installed = None
    resident.append_message(Message(MessageRole.ASSISTANT, [TextContent("inner")]))


resident.on_entries_installed = installed
pending = resident.pending_approvals()
print(json.dumps({
    "observed": observed,
    "pending": [item[0] for item in pending],
    "last_text": resident.messages()[-1].content[0].text,
}))
'''.strip()
    )

    completed = subprocess.run(
        [sys.executable, str(script), str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )

    result = json.loads(completed.stdout)
    assert result["pending"] == ["call-1"]
    assert len(result["observed"]) == 1
    assert len(result["observed"][0]) == 1
    assert result["last_text"] == "inner"


def test_live_store_tail_syncs_subprocess_append(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="shared-subprocess")
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "from zeta.core.store import ConversationStore; "
                "from zeta.protocol.types import Message, MessageOrigin, MessageRole, TextContent, with_message_origin; "
                "s=ConversationStore(Path(__import__('sys').argv[1]), "
                "session_id='shared-subprocess'); "
                "s.append_message(with_message_origin(Message(MessageRole.USER, "
                "[TextContent('external')]), MessageOrigin.USER))"
            ),
            str(tmp_path),
        ],
        check=True,
    )

    store.append_message(message(MessageRole.ASSISTANT, "local"))

    assert [item.content[0].text for item in store.messages()] == [
        "external",
        "local",
    ]


def test_live_store_separates_external_complete_unterminated_row(
    tmp_path: Path,
) -> None:
    first = ConversationStore(tmp_path, session_id="live-unterminated")
    second = ConversationStore(tmp_path, session_id="live-unterminated")
    second.append_message(with_message_origin(message(MessageRole.USER, "external"), MessageOrigin.USER))
    second.close()
    with first.path.open("r+b") as handle:
        handle.seek(-1, 2)
        handle.truncate()

    first.append_message(message(MessageRole.ASSISTANT, "after"))

    reopened = ConversationStore(tmp_path, session_id="live-unterminated")
    assert [item.content[0].text for item in reopened.messages()] == [
        "external",
        "after",
    ]


def test_live_store_repairs_external_torn_tail_before_append(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="live-torn")
    store.append_message(with_message_origin(message(MessageRole.USER, "kept"), MessageOrigin.USER))
    with store.path.open("ab") as handle:
        handle.write(b'{"seq":3,"id":"torn"')

    with pytest.warns(RuntimeWarning, match="dropped torn final"):
        store.append_message(message(MessageRole.ASSISTANT, "after"))

    assert [entry.type for entry in store.entries] == ["message", "warning", "message"]
    reopened = ConversationStore(tmp_path, session_id="live-torn")
    assert [entry.type for entry in reopened.entries] == [
        "message",
        "warning",
        "message",
    ]


def test_live_store_full_reloads_after_file_replacement(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="replaced")
    store.append_message(with_message_origin(message(MessageRole.USER, "kept"), MessageOrigin.USER))
    replacement = store.path.with_suffix(".replacement")
    replacement.write_bytes(store.path.read_bytes())
    store.append_message(with_message_origin(message(MessageRole.USER, "discarded"), MessageOrigin.USER))
    replacement.replace(store.path)

    reloads = store._full_reload_count
    store.append_message(message(MessageRole.ASSISTANT, "after"))

    assert store._full_reload_count == reloads + 1
    assert [item.content[0].text for item in store.messages()] == ["kept", "after"]


def test_live_store_reloads_same_size_rewrite_with_restored_mtime(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path, session_id="same-size-rewrite")
    store.append_message(with_message_origin(message(MessageRole.USER, "old"), MessageOrigin.USER))
    before = store.path.stat()
    original = store.path.read_bytes()
    rewritten = original.replace(b'"text":"old"', b'"text":"new"')
    assert len(rewritten) == len(original)
    store.path.write_bytes(rewritten)
    os.utime(store.path, ns=(before.st_atime_ns, before.st_mtime_ns))

    store.refresh()

    assert [item.content[0].text for item in store.messages()] == ["new"]


def test_live_store_full_reloads_growing_in_place_rewrite(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="growing-rewrite")
    store.append_message(with_message_origin(message(MessageRole.USER, "kept"), MessageOrigin.USER))
    original = store.path.read_bytes()
    store.path.write_bytes(original.replace(b"\n", b"\r\n"))
    fresh = ConversationStore(tmp_path, session_id=store.session_id)

    store.refresh()

    assert store.entries == fresh.entries
    assert [item.content[0].text for item in store.messages()] == ["kept"]


def test_live_store_bounded_fingerprint_does_not_detect_old_prefix_rewrite(
    tmp_path: Path,
) -> None:
    resident = ConversationStore(tmp_path, session_id="old-prefix-rewrite")
    resident.append_message(with_message_origin(message(MessageRole.USER, "old"), MessageOrigin.USER))
    resident.append_message(with_message_origin(message(MessageRole.USER, "x" * 70_000), MessageOrigin.USER))
    original = resident.path.read_bytes()
    rewritten = original.replace(b'"text":"old"', b'"text":"new"')
    assert len(rewritten) == len(original)
    resident.path.write_bytes(rewritten)

    external = ConversationStore(tmp_path, session_id=resident.session_id)
    external.append_message(message(MessageRole.ASSISTANT, "appended"))
    resident.refresh()
    reopened = ConversationStore(tmp_path, session_id=resident.session_id)

    assert [item.content[0].text for item in resident.messages()][::2] == [
        "old",
        "appended",
    ]
    assert [item.content[0].text for item in reopened.messages()][::2] == [
        "new",
        "appended",
    ]


def test_live_store_full_reloads_after_truncation(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="truncated")
    header = store.path.read_bytes()
    store.append_message(with_message_origin(message(MessageRole.USER, "discarded"), MessageOrigin.USER))
    store.path.write_bytes(header)

    reloads = store._full_reload_count
    store.append_message(with_message_origin(message(MessageRole.USER, "after"), MessageOrigin.USER))

    assert store._full_reload_count == reloads + 1
    assert [item.content[0].text for item in store.messages()] == ["after"]


def test_live_stores_reload_under_session_lock(tmp_path: Path) -> None:
    first = ConversationStore(tmp_path, session_id="shared")
    second = ConversationStore(tmp_path, session_id="shared")

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(first.append_message, message(MessageRole.USER, "one")),
            executor.submit(second.append_message, message(MessageRole.USER, "two")),
        ]
        [future.result() for future in futures]

    reopened = ConversationStore(tmp_path, session_id="shared")
    assert [entry.seq for entry in reopened.entries] == [1, 2]
    assert len({entry.id for entry in reopened.entries}) == 2


def test_duplicate_generated_id_is_rejected_on_append(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="shared")
    first = store.append_message(with_message_origin(message(MessageRole.USER, "one"), MessageOrigin.USER))

    with patch(
        "zeta.core.store._store.uuid.uuid4",
        return_value=SimpleNamespace(hex=first.id),
    ):
        with pytest.raises(ConversationIntegrityError, match="duplicate"):
            store.append_message(with_message_origin(message(MessageRole.USER, "two"), MessageOrigin.USER))


def test_session_id_path_and_header_mismatches_are_rejected(tmp_path: Path) -> None:
    for session_id in ("../escape", "/", "//"):
        with pytest.raises(ConversationIntegrityError, match="safe path"):
            ConversationStore(tmp_path, session_id=session_id)

    store = ConversationStore(tmp_path, session_id="expected")
    rows = store.path.read_text().splitlines()
    header = json.loads(rows[0])
    header["data"]["session_id"] = "other"
    rows[0] = json.dumps(header, separators=(",", ":"), sort_keys=True)
    store.path.write_text("\n".join(rows) + "\n")

    with pytest.raises(ConversationIntegrityError, match="mismatch"):
        ConversationStore(tmp_path, session_id="expected")


def test_later_root_is_rejected_on_load(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(with_message_origin(message(MessageRole.USER, "first"), MessageOrigin.USER))
    store.append_message(with_message_origin(message(MessageRole.USER, "second"), MessageOrigin.USER))
    rows = store.path.read_text().splitlines()
    row = json.loads(rows[-1])
    row["parent_id"] = None
    rows[-1] = json.dumps(row, separators=(",", ":"), sort_keys=True)
    store.path.write_text("\n".join(rows) + "\n")

    with pytest.raises(ConversationIntegrityError, match="orphaned root"):
        ConversationStore(tmp_path, session_id=store.session_id)


def test_append_message_with_approvals_copies_only_new_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path)
    for index in range(50):
        store.append_message(message(MessageRole.USER, f"message {index}"))
    copied: list[str] = []
    snapshot = ConversationStore._snapshot_entry

    def counting_snapshot(entry: ConversationEntry) -> ConversationEntry:
        copied.append(entry.type)
        return snapshot(entry)

    monkeypatch.setattr(
        ConversationStore, "_snapshot_entry", staticmethod(counting_snapshot)
    )

    appended = store.append_message_with_approval_requests(
        message(MessageRole.ASSISTANT, "done")
    )

    assert appended.data["message"]["content"][0]["text"] == "done"
    assert copied == ["message"]


def test_all_agent_notifications_copy_only_notification_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # agent_status reads every notification once per child on the event loop;
    # copying the whole branch there froze a long session for ~34 s.
    store = ConversationStore(tmp_path)
    for index in range(50):
        store.append_message(message(MessageRole.USER, f"message {index}"))
    store.append_agent_notification(
        "root:1",
        child_session_path="/sessions/root/agents/1",
        description="worker",
        status="completed",
        text="done",
    )
    copied: list[str] = []
    snapshot = ConversationStore._snapshot_entry

    def counting_snapshot(entry: ConversationEntry) -> ConversationEntry:
        copied.append(entry.type)
        return snapshot(entry)

    monkeypatch.setattr(
        ConversationStore, "_snapshot_entry", staticmethod(counting_snapshot)
    )

    notifications = store.agent_notifications(pending_only=False)

    assert [entry.data["child_instance_id"] for entry in notifications] == ["root:1"]
    assert copied == ["notification"]
