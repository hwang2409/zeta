from pathlib import Path
from unittest.mock import patch

import pytest

from zeta.store import ConversationStore
from zeta.types import Message, MessageRole, TextContent


def message(role: MessageRole, text: str) -> Message:
    return Message(role, [TextContent(text)])


def test_append_replay_round_trip_and_parent_links(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="session-1", cwd="/work")
    first = store.append_message(message(MessageRole.USER, "hello"))
    second = store.append_message(message(MessageRole.ASSISTANT, "hi"))

    reopened = ConversationStore(tmp_path)

    assert [entry.id for entry in reopened.replay()] == [first.id, second.id]
    assert second.parent_id == first.id
    assert [item.content[0].text for item in reopened.messages()] == ["hello", "hi"]
    assert reopened.session_id == "session-1"
    assert reopened.cwd == "/work"


def test_torn_tail_is_dropped_with_warning_entry(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(message(MessageRole.USER, "kept"))
    store.path.open("ab").write(b'{"seq": 3, "id": "torn"')

    with pytest.warns(RuntimeWarning, match="torn"):
        reopened = ConversationStore(tmp_path)

    assert [item.content[0].text for item in reopened.messages()] == ["kept"]
    assert reopened.entries[-1].type == "warning"


def test_compaction_marker_persists(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    marker = store.append_compaction_marker("summary", 1, 4)
    reopened = ConversationStore(tmp_path)

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
