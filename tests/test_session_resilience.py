import os
from pathlib import Path

import pytest

from zeta.core.session import SessionError, SessionManager
from zeta.core.store import (
    ConversationIntegrityError,
    ConversationStore,
    TranscriptLoadError,
)
from zeta.protocol.types import (
    Message,
    MessageOrigin,
    MessageRole,
    TextContent,
    with_message_origin,
)


@pytest.mark.parametrize("damage", ["missing", "json", "encoding", "directory"])
def test_listing_skips_unreadable_metadata(tmp_path: Path, damage: str) -> None:
    manager = SessionManager(tmp_path)
    good = manager.create(provider="codex", model="offline")
    bad = manager.create(provider="codex", model="offline")
    path = bad.store.session_dir / "meta.json"
    path.unlink()
    if damage == "json":
        path.write_text("{")
    elif damage == "encoding":
        path.write_bytes(b"\xff")
    elif damage == "directory":
        path.mkdir()

    assert manager.list_sessions() == [good.metadata]


@pytest.mark.parametrize("damage", ["metadata", "conversation", "corrupt_conversation"])
def test_previews_skip_unreadable_sessions(tmp_path: Path, damage: str) -> None:
    manager = SessionManager(tmp_path)
    good = manager.create(provider="codex", model="offline")
    bad = manager.create(provider="codex", model="offline")
    if damage == "metadata":
        (bad.store.session_dir / "meta.json").write_text("{")
    elif damage == "conversation":
        bad.store.path.unlink()
    else:
        bad.store.path.write_text("broken\n")

    previews = manager.list_session_previews()
    expected = [good.metadata.session_id]
    if damage != "metadata":
        expected.insert(0, bad.metadata.session_id)
        assert previews[0].preview == ""
    assert [item.session_id for item in previews] == expected


@pytest.mark.parametrize("fail_write", [False, True])
def test_create_publishes_only_complete_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_write: bool
) -> None:
    manager = SessionManager(tmp_path)
    original = SessionManager._write
    observed = []

    def checked_write(self, metadata):
        # Inspect the directory itself: a tolerant listing could hide a partial create.
        observed.append(list(manager.sessions_dir.iterdir()))
        if fail_write:
            raise OSError("interrupted write")
        original(self, metadata)
        observed.append(list(manager.sessions_dir.iterdir()))

    monkeypatch.setattr(SessionManager, "_write", checked_write)
    if fail_write:
        with pytest.raises(OSError, match="interrupted write"):
            manager.create(provider="codex", model="offline")
        assert list(manager.sessions_dir.iterdir()) == []
    else:
        opened = manager.create(provider="codex", model="offline")
        assert manager.open(opened.metadata.session_id).metadata == opened.metadata
        assert opened.store.root_dir == manager.sessions_dir
        assert opened.store.path.is_file()
    assert observed and all(paths == [] for paths in observed)


def test_missing_metadata_error_distinguishes_absent_session(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    (manager.sessions_dir / "partial").mkdir(parents=True)
    with pytest.raises(SessionError, match="session partial has no meta.json"):
        manager.open("partial")
    with pytest.raises(SessionError, match="session absent was not found"):
        manager.open("absent")


def test_listing_skips_deeply_nested_metadata(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    good = manager.create(provider="codex", model="offline")
    bad = manager.create(provider="codex", model="offline")
    (bad.store.session_dir / "meta.json").write_text("[" * 10_000 + "]" * 10_000)

    with pytest.raises(SessionError, match="session metadata could not be read"):
        manager.open(bad.metadata.session_id)
    assert manager.list_sessions() == [good.metadata]


def test_preview_limit_counts_catalog_rows(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    manager.create(provider="codex", model="offline")
    bad = manager.create(provider="codex", model="offline")
    bad.metadata.updated_at = "2099-01-01T00:00:00+00:00"
    manager._write(bad.metadata)
    bad.store.path.write_text("broken\n")

    previews = manager.list_session_previews(limit=1)
    assert [item.session_id for item in previews] == [bad.metadata.session_id]
    assert previews[0].preview == ""
    assert manager.list_session_previews(limit=0) == []


def test_listing_skips_missing_conversation(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    good = manager.create(provider="codex", model="offline")
    bad = manager.create(provider="codex", model="offline")
    bad.store.path.unlink()

    assert {item.session_id for item in manager.list_sessions()} == {
        good.metadata.session_id,
        bad.metadata.session_id,
    }


def test_create_preserves_empty_directory_created_during_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = SessionManager(tmp_path)
    original = SessionManager._write
    collision = None
    inode = None

    def racing_write(self, metadata):
        nonlocal collision, inode
        original(self, metadata)
        if collision is None:
            # Another creator reserves this ID after the initial exists check.
            collision = manager.sessions_dir / metadata.session_id
            collision.mkdir()
            inode = collision.stat().st_ino

    monkeypatch.setattr(SessionManager, "_write", racing_write)
    opened = manager.create(provider="codex", model="offline")

    assert collision is not None
    assert collision.stat().st_ino == inode
    assert list(collision.iterdir()) == []
    assert opened.store.session_dir != collision
    assert manager.open(opened.metadata.session_id).metadata == opened.metadata


@pytest.mark.parametrize("filename", ["session_state.json", "agent_lifecycle.json"])
def test_previews_do_not_load_deeply_nested_state(tmp_path: Path, filename: str) -> None:
    manager = SessionManager(tmp_path)
    good = manager.create(provider="codex", model="offline")
    bad = manager.create(provider="codex", model="offline")
    (bad.store.session_dir / filename).write_text("[" * 10_000 + "]" * 10_000)

    with pytest.raises(ConversationIntegrityError, match="could not be read"):
        ConversationStore(manager.sessions_dir, session_id=bad.metadata.session_id)
    with pytest.raises(SessionError, match="could not be opened"):
        manager.open(bad.metadata.session_id)
    assert {
        item.session_id for item in manager.list_session_previews()
    } == {good.metadata.session_id, bad.metadata.session_id}


def test_create_preserves_directory_created_at_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = SessionManager(tmp_path)
    manager.sessions_dir.mkdir()
    root_inode = manager.sessions_dir.stat().st_ino
    original_mkdir = os.mkdir
    collision = None
    inode = None

    def racing_mkdir(path, mode=0o777, *, dir_fd=None):
        nonlocal collision, inode
        # Inject a non-cooperating creator immediately before the atomic claim.
        if (collision is None and dir_fd is not None
                and os.fstat(dir_fd).st_ino == root_inode):
            original_mkdir(path, mode, dir_fd=dir_fd)
            collision = manager.sessions_dir / path
            inode = collision.stat().st_ino
        return original_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "mkdir", racing_mkdir)
    opened = manager.create(provider="codex", model="offline")

    assert collision is not None
    assert collision.stat().st_ino == inode
    assert list(collision.iterdir()) == []
    assert opened.store.session_dir != collision
    assert manager.list_sessions() == [opened.metadata]


def test_interrupted_publication_is_not_discoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = SessionManager(tmp_path)
    original = os.replace
    interrupted = None

    def interrupted_replace(source, target, **kwargs):
        nonlocal interrupted
        if target == "meta.json":
            destination = os.fstat(kwargs["dst_dir_fd"])
            interrupted = next((path for path in manager.sessions_dir.iterdir()
                                if path.stat().st_ino == destination.st_ino), None)
            if interrupted is not None:
                assert (interrupted / "conversation.jsonl").is_file()
                assert manager.list_sessions() == []
                assert manager.list_session_previews() == []
                raise OSError("interrupted publication")
        return original(source, target, **kwargs)

    monkeypatch.setattr(os, "replace", interrupted_replace)
    with pytest.raises(OSError, match="interrupted publication"):
        manager.create(provider="codex", model="offline")

    assert interrupted is not None
    assert interrupted.is_dir()
    assert manager.list_sessions() == []
    with pytest.raises(SessionError, match="has no meta.json"):
        manager.open(interrupted.name)


def test_shared_session_json_depth_limit() -> None:
    import json

    from zeta.core.checkpoints import MAX_SESSION_JSON_DEPTH, load_session_json

    for opening, closing in (("[", "]"), ('{"nested":', "}")):
        allowed = (
            opening * MAX_SESSION_JSON_DEPTH
            + '"value"'
            + closing * MAX_SESSION_JSON_DEPTH
        )
        assert load_session_json(allowed.encode()) == json.loads(allowed)
        excessive = opening + allowed + closing
        with pytest.raises(ConversationIntegrityError, match="depth"):
            load_session_json(excessive.encode())
    # Brackets and escaped quotes inside strings do not contribute to depth.
    text = json.dumps({"text": '[\\"' * 500})
    assert load_session_json(text.encode()) == json.loads(text)
    for invalid in (b"{", b"\xff", b"[" * 10_000 + b"]" * 10_000):
        with pytest.raises(ConversationIntegrityError):
            load_session_json(invalid)


@pytest.mark.parametrize(
    "filename",
    ["meta.json", "session_state.json", "agent_lifecycle.json", "conversation.jsonl"],
)
def test_listings_skip_decoder_surviving_depth(tmp_path: Path, filename: str) -> None:
    import json

    from zeta.protocol.types import Message, MessageRole, TextContent

    manager = SessionManager(tmp_path)
    good = manager.create(provider="codex", model="offline")
    good.store.append_message(
        with_message_origin(Message(MessageRole.USER, [TextContent("healthy preview")]), MessageOrigin.USER)
    )
    bad = manager.create(provider="codex", model="offline")
    deep = '{"nested":' * 500 + "0" + "}" * 500
    if filename == "session_state.json":
        payload = (
            '{"bash_cwd":"/tmp","agent_parent":{"tool_call_id":"call","extra":'
            + deep
            + "}}"
        )
    elif filename == "conversation.jsonl":
        payload = (
            '{"seq":1,"id":"deep","parent_id":null,"lane":"main","type":"warning","data":{"message":"test","extra":'
            + deep
            + "}}"
        )
    elif filename == "meta.json":
        payload = json.dumps(bad.metadata.to_dict())[:-1] + ',"extra":' + deep + "}"
    else:
        payload = '{"extra":' + deep + "}"
    assert isinstance(
        json.loads(payload), dict
    )  # The decoder accepts the round-3 probes.
    path = bad.store.session_dir / filename
    if filename == "conversation.jsonl":
        payload = path.read_text() + payload + "\n"
    path.write_text(payload)

    listed = manager.list_sessions()
    if filename == "meta.json":
        assert [item.session_id for item in listed] == [good.metadata.session_id]
    else:
        assert {item.session_id for item in listed} == {
            good.metadata.session_id,
            bad.metadata.session_id,
        }
    previews = manager.list_session_previews(limit=1, sessions=[good.metadata])
    assert [(item.session_id, item.preview) for item in previews] == [
        (good.metadata.session_id, "healthy preview")
    ]
    error_type = TranscriptLoadError if filename == "conversation.jsonl" else SessionError
    with pytest.raises(error_type):
        manager.open(bad.metadata.session_id)
    assert path.read_text() == payload


def test_mixed_store_listings_validate_without_mutating(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    good = manager.create(provider="codex", model="offline")
    # Legacy sessions may lack optional state and lock files. Listing must not create them.
    good.store.state_path.unlink()
    good.store.lock_path.unlink()
    for filename, content in (
        ("meta.json", b"{"),
        ("conversation.jsonl", b"broken\n"),
        ("conversation.jsonl", b""),
        ("session_state.json", b"{"),
        ("session_state.json", b'{"bash_cwd":12}'),
        ("agent_lifecycle.json", b"["),
        ("agent_lifecycle.json", b"[]"),
        ("agent_lifecycle.json", b"\xff"),
    ):
        bad = manager.create(provider="codex", model="offline")
        (bad.store.session_dir / filename).write_bytes(content)
    torn = manager.create(provider="codex", model="offline")
    with torn.store.path.open("ab") as handle:
        handle.write(b'{"seq":')
    no_state = manager.create(provider="codex", model="offline")
    no_state.store.state_path.unlink()
    no_state.store.agent_lifecycle_path.write_text("[]")
    before = {p: p.read_bytes() for p in manager.sessions_dir.rglob("*") if p.is_file()}

    assert manager.list_sessions()
    assert [p.session_id for p in manager.list_session_previews(sessions=[good.metadata])] == [
        good.metadata.session_id
    ]
    assert {
        p: p.read_bytes() for p in manager.sessions_dir.rglob("*") if p.is_file()
    } == before


def test_list_sessions_reads_only_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = SessionManager(tmp_path)
    opened = manager.create(provider="codex", model="offline")

    def fail(*args, **kwargs):
        raise AssertionError("conversation store opened")

    monkeypatch.setattr("zeta.core.session.ConversationStore", fail)
    assert manager.list_sessions_read_only() == [opened.metadata]


def test_preview_reads_bounded_prefix(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    opened = manager.create(provider="codex", model="offline")
    store = opened.store
    store.append_checkpoint("root")
    store.append_message(with_message_origin(
        Message(MessageRole.USER, [TextContent("first log message")]),
        MessageOrigin.USER,
    ))
    store.append_message(Message(MessageRole.ASSISTANT, [TextContent("reply")]))
    store.append_fork("root")
    store.append_message(with_message_origin(
        Message(MessageRole.USER, [TextContent("active branch message")]),
        MessageOrigin.USER,
    ))
    assert store.messages()[0].content[0].text == "active branch message"
    with store.path.open("ab") as conversation:
        conversation.write(b"not json\n" * 5_000)

    previews = manager.list_session_previews(sessions=[opened.metadata])

    assert [item.preview for item in previews] == ["first log message"]


@pytest.mark.parametrize("shape", ["huge_row", "unterminated", "no_user_message"])
def test_preview_prefix_reader_stops_at_its_caps(tmp_path: Path, shape: str) -> None:
    import tracemalloc

    manager = SessionManager(tmp_path)
    opened = manager.create(provider="codex", model="offline")
    opened.store.append_message(Message(MessageRole.ASSISTANT, [TextContent("note")]))
    opened.store.append_message(with_message_origin(
        Message(MessageRole.USER, [TextContent("late user message")]),
        MessageOrigin.USER,
    ))
    opened.store.close()
    header, other_row, user_row = opened.store.path.read_bytes().splitlines(
        keepends=True
    )[:3]
    bodies = {
        # one 32 MiB row before the user message
        "huge_row": header + b"x" * (32 << 20) + b"\n" + user_row,
        # an unterminated log: no newline for 32 MiB
        "unterminated": header + b"y" * (32 << 20),
        # the user message only after 100,000 non-user rows: past the row cap
        "no_user_message": header + other_row * 100_000 + user_row,
    }
    opened.store.path.write_bytes(bodies[shape])

    tracemalloc.start()
    try:
        previews = manager.list_session_previews(sessions=[opened.metadata])
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert [item.preview for item in previews] == [""]
    assert peak < 2 << 20, f"preview read allocated {peak} bytes"


def test_child_lifecycle_card_reader_rejects_decoder_surviving_depth(
    tmp_path: Path,
) -> None:
    from zeta.tui.agent_card import _read_lifecycle

    path = tmp_path / "agent_lifecycle.json"
    path.write_text('{"extra":' + '{"nested":' * 500 + "0" + "}" * 501)
    assert _read_lifecycle(str(tmp_path)) == {}


@pytest.mark.parametrize("fault", [RecursionError, OSError, KeyError])
def test_listing_boundaries_propagate_programming_faults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault
) -> None:
    manager = SessionManager(tmp_path)
    manager.create(provider="codex", model="offline")

    def fail(*args, **kwargs):
        raise fault("programming fault")

    monkeypatch.setattr(manager, "read_metadata", fail)
    with pytest.raises(fault, match="programming fault"):
        manager.list_sessions_read_only()


@pytest.mark.parametrize(
    "row",
    [b"\xff\n", b"broken\n", b"[" * 65 + b"]" * 65 + b"\n"],
    ids=["binary", "malformed", "deep"],
)
def test_export_rejects_corrupt_rows_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, row: bytes
) -> None:
    import argparse
    from io import StringIO

    from zeta.cli import session as session_cli

    monkeypatch.setenv("ZETA_HOME", str(tmp_path))
    manager = SessionManager(tmp_path)
    opened = manager.create(provider="codex", model="offline")
    with opened.store.path.open("ab") as handle:
        handle.write(row)
    before = opened.store.path.read_bytes()
    with pytest.raises(SessionError, match="could not be exported"):
        manager.export(opened.metadata.session_id)
    out, err = StringIO(), StringIO()
    destination = tmp_path / "export.jsonl"
    args = argparse.Namespace(
        session_verb="export",
        session_id=opened.metadata.session_id,
        out=str(destination),
    )
    assert session_cli.run(args, stdout=out, stderr=err) == 1
    assert "could not be exported" in err.getvalue()
    assert out.getvalue() == ""
    assert not destination.exists()
    assert opened.store.path.read_bytes() == before
