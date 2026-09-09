from pathlib import Path

import pytest

from zeta.core.session import SessionError, SessionManager


@pytest.mark.parametrize("damage", ["missing", "json", "encoding", "directory"])
def test_listing_skips_unreadable_metadata(tmp_path: Path, damage: str) -> None:
    manager = SessionManager(tmp_path)
    good = manager.create(provider="fake", model="offline")
    bad = manager.create(provider="fake", model="offline")
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
    good = manager.create(provider="fake", model="offline")
    bad = manager.create(provider="fake", model="offline")
    if damage == "metadata":
        (bad.store.session_dir / "meta.json").write_text("{")
    elif damage == "conversation":
        bad.store.path.unlink()
    else:
        bad.store.path.write_text("broken\n")

    assert [item.session_id for item in manager.list_session_previews()] == [
        good.metadata.session_id
    ]


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
            manager.create(provider="fake", model="offline")
        assert list(manager.sessions_dir.iterdir()) == []
    else:
        opened = manager.create(provider="fake", model="offline")
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
    good = manager.create(provider="fake", model="offline")
    bad = manager.create(provider="fake", model="offline")
    (bad.store.session_dir / "meta.json").write_text("[" * 10_000 + "]" * 10_000)

    with pytest.raises(SessionError, match="session metadata could not be read"):
        manager.open(bad.metadata.session_id)
    assert manager.list_sessions() == [good.metadata]


def test_preview_limit_counts_healthy_sessions(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    good = manager.create(provider="fake", model="offline")
    bad = manager.create(provider="fake", model="offline")
    bad.metadata.updated_at = "2099-01-01T00:00:00+00:00"
    manager._write(bad.metadata)
    bad.store.path.write_text("broken\n")

    assert [item.session_id for item in manager.list_session_previews(limit=1)] == [
        good.metadata.session_id
    ]
    assert manager.list_session_previews(limit=0) == []


def test_listing_skips_missing_conversation(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    good = manager.create(provider="fake", model="offline")
    bad = manager.create(provider="fake", model="offline")
    bad.store.path.unlink()

    assert manager.list_sessions() == [good.metadata]


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
    opened = manager.create(provider="fake", model="offline")

    assert collision is not None
    assert collision.stat().st_ino == inode
    assert list(collision.iterdir()) == []
    assert opened.store.session_dir != collision
    assert manager.open(opened.metadata.session_id).metadata == opened.metadata
