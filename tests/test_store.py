import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from zeta.store import ConversationIntegrityError, ConversationStore
from zeta.types import Message, MessageRole, TextContent, ToolCall, ToolUseContent


def message(role: MessageRole, text: str) -> Message:
    return Message(role, [TextContent(text)])


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


def test_append_and_replay_use_data_snapshots(tmp_path: Path) -> None:
    arguments = {"nested": {"value": 1}}
    call = ToolCall("call-1", "tool", arguments)
    store = ConversationStore(
        tmp_path,
        session_id="snapshot",
    )
    entry = store.append_message(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)])
    )

    arguments["nested"]["value"] = 2
    entry.data["message"]["content"][0]["tool_call"]["arguments"]["nested"]["value"] = 4
    replay = store.replay()
    replay[0].data["message"]["content"][0]["tool_call"]["arguments"]["nested"]["value"] = 3

    fresh = store.replay()
    assert fresh[0].data["message"]["content"][0]["tool_call"]["arguments"]["nested"]["value"] == 1


def test_compaction_marker_persists(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    marker = store.append_compaction_marker("summary", 1, 4)
    reopened = ConversationStore(tmp_path, session_id=store.session_id)

    assert reopened.replay()[-1].id == marker.id
    assert reopened.replay()[-1].data == {
        "summary": "summary",
        "source_seq_start": 1,
        "source_seq_end": 4,
    }


def test_append_fsyncs_before_return(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    with patch("zeta.store.os.fsync") as fsync:
        store.append_message(message(MessageRole.USER, "hello"))

    fsync.assert_called_once()


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
    rows[-1] = rows[-1].replace('"seq":1', '"seq":3').replace('"parent_id":null', '"parent_id":"missing"')
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
        "zeta.store.uuid.uuid4",
        return_value=SimpleNamespace(hex=first.id),
    ):
        with pytest.raises(ConversationIntegrityError, match="duplicate"):
            store.append_message(message(MessageRole.USER, "two"))


def test_session_id_path_and_header_mismatches_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConversationIntegrityError, match="safe path"):
        ConversationStore(tmp_path, session_id="../escape")

    store = ConversationStore(tmp_path, session_id="expected")
    rows = store.path.read_text().splitlines()
    header = json.loads(rows[0])
    header["data"]["session_id"] = "other"
    rows[0] = json.dumps(header, separators=(",", ":"), sort_keys=True)
    store.path.write_text("\n".join(rows) + "\n")

    with pytest.raises(ConversationIntegrityError, match="mismatch"):
        ConversationStore(tmp_path, session_id="expected")
