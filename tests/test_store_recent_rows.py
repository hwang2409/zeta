"""Key-shaped recent row lookup through the public ConversationStore accessors."""

import json
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from zeta.core.store import ConversationStore
from zeta.core.store.recent_rows import (
    RECENT_ROW_WINDOW,
    ApprovalHistory,
    ApprovalRequest,
    ApprovalResolution,
    AssistantResponseAfter,
    DeliveredMessage,
    ToolResult,
    UserTurnMessageAfter,
)
from zeta.protocol.types import (
    ASSISTANT_RESPONSE_COMPLETED,
    ASSISTANT_RESPONSE_STATE,
    Message,
    MessageOrigin,
    MessageRole,
    TextContent,
    ToolCall,
    ToolUseContent,
    with_message_origin,
)
from zeta.protocol.types import ToolResult as WireToolResult


def _user(text: str) -> Message:
    return with_message_origin(
        Message(MessageRole.USER, [TextContent(text)]), MessageOrigin.USER
    )


def _assistant(text: str, *blocks: ToolUseContent, state: str | None = None) -> Message:
    metadata = {ASSISTANT_RESPONSE_STATE: state} if state else {}
    return Message(
        MessageRole.ASSISTANT, [TextContent(text), *blocks], metadata=metadata
    )


@pytest.fixture
def store(tmp_path: Path) -> Iterator[ConversationStore]:
    owned = ConversationStore(tmp_path, session_id="recent-rows")
    try:
        yield owned
    finally:
        owned.close()


def _history(event: str, wire_id: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "event": event,
        "request_id": wire_id,
        "tool_call": {"id": "call-d", "name": "bash", "arguments": {}},
    }
    if event == "approval_request":
        payload |= {"delegated": True, "agent_instance_id": "child-1"}
    else:
        payload["data"] = {"decision": "allow"}
    return payload


def test_tool_result_matches_only_the_result_row(store: ConversationStore) -> None:
    call = ToolCall("call-1", "bash", {"command": "true"})
    use = store.append_message(_assistant("run", ToolUseContent(call)))
    assert store.recent_row_seq(ToolResult("call-1")) is None
    result = store.append_message(
        Message(MessageRole.TOOL_RESULT, [], tool_result=WireToolResult("call-1", "ok"))
    )
    store.append_message(_user("later"))

    assert store.recent_row_seq(ToolResult("call-1")) == result.seq != use.seq
    assert store.row_id_at(result.seq) == result.id


def test_approval_request_and_resolution(store: ConversationStore) -> None:
    call = ToolCall("req-1", "bash", {"command": "rm x"})
    anchor = store.append_message_with_approval_requests(
        _assistant("ask", ToolUseContent(call)), [("req-1", call)]
    )
    assert store.recent_row_seq(ApprovalResolution("req-1")) is None
    resolution = store.append_approval_resolution("req-1", "allow")

    assert store.recent_row_seq(ApprovalRequest("req-1")) == anchor.seq
    assert store.recent_row_seq(ApprovalRequest("req-other")) is None
    assert store.recent_row_seq(ApprovalResolution("req-1")) == resolution.seq


def test_approval_history_event_disambiguates_shared_wire_id(
    store: ConversationStore,
) -> None:
    request = store.append_approval_history(_history("approval_request", "approval-a"))
    assert store.recent_row_seq(ApprovalHistory("approval_end", "approval-a")) is None
    end = store.append_approval_history(_history("approval_end", "approval-a"))

    assert (
        store.recent_row_seq(ApprovalHistory("approval_request", "approval-a"))
        == request.seq
    )
    assert store.recent_row_seq(ApprovalHistory("approval_end", "approval-a")) == end.seq


def test_delivered_message_matches_the_tagged_send_row(
    store: ConversationStore,
) -> None:
    store.append_client_delivery(
        "send-1", "send", "delivered",
        {"accepted": True, "session_id": store.session_id},
        message=_user("hello"),
    )
    tagged = store.entries[-1]

    assert tagged.type == "message"
    assert store.recent_row_seq(DeliveredMessage("send-1")) == tagged.seq
    assert store.recent_row_seq(DeliveredMessage("send-2")) is None


def test_after_keys_name_the_first_match_after_the_mark(
    store: ConversationStore,
) -> None:
    mark = store.append_message(_user("before")).seq
    store.append_message(_assistant("partial", state="failed"))
    first = store.append_message(_assistant("one", state=ASSISTANT_RESPONSE_COMPLETED))
    store.append_message(_assistant("two", state=ASSISTANT_RESPONSE_COMPLETED))
    user = store.append_message(_user("next"))

    assert store.recent_row_seq(AssistantResponseAfter(mark)) == first.seq
    assert store.recent_row_seq(UserTurnMessageAfter(mark)) == user.seq
    assert store.recent_row_seq(UserTurnMessageAfter(user.seq)) is None


def test_window_floor_hides_older_rows(store: ConversationStore) -> None:
    old = store.append_approval_history(_history("approval_end", "approval-old"))
    for index in range(RECENT_ROW_WINDOW):
        store.append_message(_user(f"filler {index}"))
    floor = store.recent_row_floor()

    assert floor == old.seq + 1 == len(store.entries) - RECENT_ROW_WINDOW + 1
    assert store.recent_row_seq(ApprovalHistory("approval_end", "approval-old")) is None
    assert store.row_id_at(old.seq) is None
    assert store.row_id_at(floor) == store.entries[floor - 1].id
    # Rows after the mark start below the floor, so the first match is unknown.
    assert store.recent_row_seq(UserTurnMessageAfter(old.seq - 1)) is None


def test_reset_generation_counts_replacements_not_appends(
    store: ConversationStore,
) -> None:
    first = store.append_message(_assistant("old"))
    final = store.append_message(_assistant("tip"))
    start = store.reset_generation()
    store.append_message(_user("more"))
    store.refresh()
    assert store.reset_generation() == start

    rows = store.path.read_text(encoding="utf-8").splitlines()
    replaced_first = json.loads(rows[1])
    replaced_final = json.loads(rows[2])
    replaced_first["id"] = "replacement-id"
    replaced_final["parent_id"] = "replacement-id"
    assert final.seq == replaced_final["seq"]
    replacement = store.path.with_suffix(".replacement")
    replacement.write_text(
        "\n".join(
            [rows[0], json.dumps(replaced_first), json.dumps(replaced_final), ""]
        ),
        encoding="utf-8",
    )
    replacement.replace(store.path)

    store.refresh()

    assert store.reset_generation() == start + 1
    assert store.row_id_at(first.seq) == "replacement-id" != first.id


def _replace_first_row_id(store: ConversationStore) -> None:
    rows = store.path.read_text(encoding="utf-8").splitlines()
    replaced_first = json.loads(rows[1])
    replaced_next = json.loads(rows[2])
    replaced_first["id"] = "replacement-id"
    replaced_next["parent_id"] = "replacement-id"
    replacement = store.path.with_suffix(".replacement")
    replacement.write_text(
        "\n".join(
            [rows[0], json.dumps(replaced_first), json.dumps(replaced_next), ""]
        ),
        encoding="utf-8",
    )
    replacement.replace(store.path)


async def test_async_commit_counts_an_external_replacement(
    store: ConversationStore,
) -> None:
    first = await store.append_message_async(_assistant("old"))
    await store.append_message_async(_assistant("tip"))
    start = store.reset_generation()

    _replace_first_row_id(store)
    await store.append_message_async(_user("after replacement"))

    assert store.reset_generation() == start + 1
    assert store.row_id_at(first.seq) == "replacement-id" != first.id


def test_missing_log_with_resident_rows_counts_a_reset(
    store: ConversationStore,
) -> None:
    first = store.append_message(_assistant("old"))
    store.append_message(_assistant("tip"))
    start = store.reset_generation()

    store.path.unlink()
    store.refresh()

    assert store.reset_generation() == start + 1
    assert store.row_id_at(first.seq) is None


def test_accessors_assert_the_owner_thread(store: ConversationStore) -> None:
    store.bind_owner_thread()
    errors: list[BaseException] = []

    def off_thread() -> None:
        for call in (
            lambda: store.recent_row_seq(ToolResult("call-x")),
            store.recent_row_floor,
            lambda: store.row_id_at(1),
            store.reset_generation,
        ):
            try:
                call()
            except RuntimeError as error:
                errors.append(error)

    worker = threading.Thread(target=off_thread)
    worker.start()
    worker.join(10)

    assert not worker.is_alive()
    assert len(errors) == 4
    assert all("off owner thread" in str(error) for error in errors)


def test_scan_reads_raw_rows_without_decoding(
    store: ConversationStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.append_approval_history(_history("approval_end", "approval-raw"))
    for index in range(20):
        store.append_message(_user(f"row {index}"))

    def refuse(*_args: object, **_kwargs: object) -> Message:
        raise AssertionError("recent row scan decoded a Message")

    # test-contract: seam proves the lookup reads raw entry.data dicts
    monkeypatch.setattr(Message, "from_dict", classmethod(refuse))
    monkeypatch.setattr(ConversationStore, "replay", refuse)
    # test-contract: seam proves the lookup never snapshots rows
    monkeypatch.setattr(ConversationStore, "_snapshot_entry", staticmethod(refuse))

    assert store.recent_row_seq(ApprovalHistory("approval_end", "approval-raw")) == 1
    assert store.recent_row_seq(ToolResult("missing")) is None
    assert store.recent_row_seq(UserTurnMessageAfter(0)) == 2


def test_torn_tail_repair_of_a_rewritten_log_publishes_one_reset(
    store: ConversationStore,
) -> None:
    first = store.append_message(_assistant("old"))
    store.append_message(_assistant("tip"))
    start = store.reset_generation()
    installed: list[object] = []
    store.on_entries_installed = installed.append

    _replace_first_row_id(store)
    with store.path.open("ab") as handle:
        handle.write(b'{"seq": 3, "id": "torn"')
    with pytest.warns(RuntimeWarning, match="dropped torn final"):
        store.refresh()

    assert installed == ["reset"]
    assert store.reset_generation() == start + 1
    assert store.row_id_at(first.seq) == "replacement-id" != first.id
    assert store.entries[-1].type == "warning"


def test_torn_tail_repair_of_an_extended_log_publishes_the_suffix(
    store: ConversationStore,
) -> None:
    store.append_message(_assistant("old"))
    store.append_message(_assistant("tip"))
    start = store.reset_generation()
    installed: list[object] = []
    store.on_entries_installed = installed.append

    with store.path.open("ab") as handle:
        handle.write(b'{"seq": 3, "id": "torn"')
    with pytest.warns(RuntimeWarning, match="dropped torn final"):
        store.refresh()

    assert [[entry.type for entry in delta] for delta in installed] == [["warning"]]
    assert installed[0][0].id == store.entries[-1].id
    assert store.reset_generation() == start
