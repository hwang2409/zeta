import asyncio
import json
import subprocess
import sys
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from zeta.core.store import (
    MAX_AGENT_NOTIFICATION_TEXT,
    ConversationEntry,
    ConversationIntegrityError,
    ConversationStore,
)
from zeta.protocol.types import (
    Message,
    MessageRole,
    TextContent,
    ToolCall,
    ToolUseContent,
)


def message(role: MessageRole, text: str) -> Message:
    return Message(role, [TextContent(text)])


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


def _start_agent_lifecycle(store: ConversationStore) -> None:
    store.start_agent_lifecycle(
        handle="parent:1",
        started_at="2026-09-30T00:00:00+00:00",
        depth=1,
        agent_type="agent",
        description="child",
    )


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
    first = store.append_message(message(MessageRole.USER, "hello"))
    second = store.append_message(message(MessageRole.ASSISTANT, "hi"))

    reopened = ConversationStore(tmp_path, session_id="session-1")

    assert [entry.id for entry in reopened.replay()] == [first.id, second.id]
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


@pytest.mark.asyncio
async def test_async_append_runs_off_loop_and_is_durable_on_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path, session_id="async-writer")
    event_loop_thread = threading.get_ident()
    append_threads: list[int] = []
    original = store.append_message

    def observed_append(value: Message, *, parent_id: str | None = None):
        append_threads.append(threading.get_ident())
        return original(value, parent_id=parent_id)

    monkeypatch.setattr(store, "append_message", observed_append)
    entry = await store.append_message_async(message(MessageRole.USER, "durable"))

    assert append_threads and append_threads[0] != event_loop_thread
    reopened = ConversationStore(tmp_path, session_id="async-writer")
    assert reopened.entries[-1].id == entry.id


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
    store.append_message(message(MessageRole.USER, "kept"))
    before = store.path.read_bytes()

    with patch(
        "zeta.core.store._store.os.replace",
        side_effect=OSError("injected replace failure"),
    ):
        with pytest.raises(OSError, match="injected replace failure"):
            store.set_bash_cwd("/tmp")

    assert store.path.read_bytes() == before
    assert store.bash_cwd == str(Path.cwd())


def test_torn_tail_is_dropped_with_warning_entry(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(message(MessageRole.USER, "kept"))
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

    with pytest.raises(ConversationIntegrityError, match="terminated"):
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
    store.append_message(message(MessageRole.USER, "bad"))
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
    store.append_message(message(MessageRole.USER, "hello"))
    rows = store.path.read_text().splitlines()
    row = json.loads(rows[-1])
    del row["data"]["message"]["content"][0]["text"]
    rows[-1] = json.dumps(row, separators=(",", ":"), sort_keys=True)
    store.path.write_text("\n".join(rows) + "\n")

    with pytest.raises(ConversationIntegrityError):
        ConversationStore(tmp_path, session_id=store.session_id)


def test_bad_nested_role_is_rejected_on_load(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(message(MessageRole.USER, "hello"))
    rows = store.path.read_text().splitlines()
    row = json.loads(rows[-1])
    row["data"]["message"]["role"] = "not-a-role"
    rows[-1] = json.dumps(row, separators=(",", ":"), sort_keys=True)
    store.path.write_text("\n".join(rows) + "\n")

    with pytest.raises(ConversationIntegrityError):
        ConversationStore(tmp_path, session_id=store.session_id)


def test_numeric_nested_text_is_rejected_on_load(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(message(MessageRole.USER, "hello"))
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
        store.append_message(message(MessageRole.USER, "hello"))

    fsync.assert_called_once()


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
            store.append_message_async(message(MessageRole.USER, "durable"))
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
            store.append_message_async(message(MessageRole.USER, "durable"))
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


def test_sessions_are_isolated(tmp_path: Path) -> None:
    first = ConversationStore(tmp_path)
    second = ConversationStore(tmp_path)
    first.append_message(message(MessageRole.USER, "one"))
    second.append_message(message(MessageRole.USER, "two"))

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
    store.append_message(message(MessageRole.USER, "kept"))
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
        store.append_message(message(MessageRole.USER, "bad"), parent_id="missing")


def test_empty_session_id_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConversationIntegrityError, match="safe path"):
        ConversationStore(tmp_path, session_id="")


def test_invalid_sequence_and_parent_are_rejected_on_load(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(message(MessageRole.USER, "kept"))
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
    store.append_message(message(MessageRole.USER, "self"))
    rows = store.path.read_text().splitlines()
    row = json.loads(rows[-1])
    row["parent_id"] = row["id"]
    rows[-1] = json.dumps(row, separators=(",", ":"), sort_keys=True)
    store.path.write_text("\n".join(rows) + "\n")

    with pytest.raises(ConversationIntegrityError, match="prior parent"):
        ConversationStore(tmp_path, session_id=store.session_id)


def test_forward_parent_is_rejected_on_load(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(message(MessageRole.USER, "first"))
    store.append_message(message(MessageRole.USER, "second"))
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
    first = store.append_message(message(MessageRole.USER, "first"))
    second = store.append_message(message(MessageRole.USER, "second"))
    store._entries[0] = replace(first, parent_id=second.id)

    with pytest.raises(ConversationIntegrityError, match="cycle"):
        store.replay()


def test_duplicate_ids_are_rejected_on_load(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    first = store.append_message(message(MessageRole.USER, "one"))
    store.append_message(message(MessageRole.USER, "two"))
    rows = store.path.read_text().splitlines()
    second_row = json.loads(rows[-1])
    second_row["id"] = first.id
    rows[-1] = json.dumps(second_row, separators=(",", ":"), sort_keys=True)
    store.path.write_text("\n".join(rows) + "\n")

    with pytest.raises(ConversationIntegrityError, match="duplicate"):
        ConversationStore(tmp_path, session_id=store.session_id)


def test_appends_do_not_reread_the_full_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path, session_id="incremental")
    for index in range(200):
        store.append_message(message(MessageRole.USER, f"seed-{index}"))

    from zeta.core.store import _log as log_module

    original = log_module.read_session_file
    conversation_reads = 0

    def counted_read(directory_fd: int, name: str) -> bytes:
        nonlocal conversation_reads
        if name == "conversation.jsonl":
            conversation_reads += 1
        return original(directory_fd, name)

    monkeypatch.setattr(log_module, "read_session_file", counted_read)
    for index in range(20):
        store.append_message(message(MessageRole.USER, f"new-{index}"))

    assert conversation_reads <= 1


def test_live_store_tail_syncs_external_append_without_full_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = ConversationStore(tmp_path, session_id="shared-tail")
    second = ConversationStore(tmp_path, session_id="shared-tail")
    second.append_message(message(MessageRole.USER, "external"))

    from zeta.core.store import _log as log_module

    original = log_module.read_session_file
    conversation_reads = 0

    def counted_read(directory_fd: int, name: str) -> bytes:
        nonlocal conversation_reads
        if name == "conversation.jsonl":
            conversation_reads += 1
        return original(directory_fd, name)

    monkeypatch.setattr(log_module, "read_session_file", counted_read)
    first.append_message(message(MessageRole.ASSISTANT, "local"))

    assert conversation_reads == 0
    assert [item.content[0].text for item in first.messages()] == [
        "external",
        "local",
    ]


def test_live_store_tail_syncs_subprocess_append(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="shared-subprocess")
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "from zeta.core.store import ConversationStore; "
                "from zeta.protocol.types import Message, MessageRole, TextContent; "
                "s=ConversationStore(Path(__import__('sys').argv[1]), "
                "session_id='shared-subprocess'); "
                "s.append_message(Message(MessageRole.USER, [TextContent('external')]))"
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
    second.append_message(message(MessageRole.USER, "external"))
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
    store.append_message(message(MessageRole.USER, "kept"))
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


def test_live_store_full_reloads_after_file_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path, session_id="replaced")
    store.append_message(message(MessageRole.USER, "kept"))
    replacement = store.path.with_suffix(".replacement")
    replacement.write_bytes(store.path.read_bytes())
    store.append_message(message(MessageRole.USER, "discarded"))
    replacement.replace(store.path)

    from zeta.core.store import _log as log_module

    original = log_module.read_session_file
    conversation_reads = 0

    def counted_read(directory_fd: int, name: str) -> bytes:
        nonlocal conversation_reads
        if name == "conversation.jsonl":
            conversation_reads += 1
        return original(directory_fd, name)

    monkeypatch.setattr(log_module, "read_session_file", counted_read)
    store.append_message(message(MessageRole.ASSISTANT, "after"))

    assert conversation_reads == 1
    assert [item.content[0].text for item in store.messages()] == ["kept", "after"]


def test_live_store_full_reloads_after_truncation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path, session_id="truncated")
    header = store.path.read_bytes()
    store.append_message(message(MessageRole.USER, "discarded"))
    store.path.write_bytes(header)

    from zeta.core.store import _log as log_module

    original = log_module.read_session_file
    conversation_reads = 0

    def counted_read(directory_fd: int, name: str) -> bytes:
        nonlocal conversation_reads
        if name == "conversation.jsonl":
            conversation_reads += 1
        return original(directory_fd, name)

    monkeypatch.setattr(log_module, "read_session_file", counted_read)
    store.append_message(message(MessageRole.USER, "after"))

    assert conversation_reads == 1
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
    first = store.append_message(message(MessageRole.USER, "one"))

    with patch(
        "zeta.core.store._store.uuid.uuid4",
        return_value=SimpleNamespace(hex=first.id),
    ):
        with pytest.raises(ConversationIntegrityError, match="duplicate"):
            store.append_message(message(MessageRole.USER, "two"))


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
    store.append_message(message(MessageRole.USER, "first"))
    store.append_message(message(MessageRole.USER, "second"))
    rows = store.path.read_text().splitlines()
    row = json.loads(rows[-1])
    row["parent_id"] = None
    rows[-1] = json.dumps(row, separators=(",", ":"), sort_keys=True)
    store.path.write_text("\n".join(rows) + "\n")

    with pytest.raises(ConversationIntegrityError, match="orphaned root"):
        ConversationStore(tmp_path, session_id=store.session_id)
