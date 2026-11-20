from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.live_home import files_containing


def _race_first_read(
    monkeypatch: pytest.MonkeyPatch, target: Path, rebuilt: bytes | None
) -> None:
    """Unlink ``target`` after it was listed, as a sqlite index rebuild does.

    With ``rebuilt``, the file is written again right after the failed read.
    """
    original = Path.read_bytes
    raced = False

    def racing_read(self: Path) -> bytes:
        nonlocal raced
        if self != target or raced:
            return original(self)
        raced = True
        target.unlink()
        try:
            return original(self)
        finally:
            if rebuilt is not None:
                target.write_bytes(rebuilt)

    monkeypatch.setattr(Path, "read_bytes", racing_read)


def _home(tmp_path: Path) -> Path:
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    (tmp_path / "sessions" / "s1" / "conversation.jsonl").write_bytes(b"{}\n")
    (tmp_path / "sessions" / "s1" / "history-index.sqlite3").write_bytes(b"old")
    return tmp_path


def test_vanished_file_fails_bare_scan_but_not_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path)
    index = home / "sessions" / "s1" / "history-index.sqlite3"

    _race_first_read(monkeypatch, index, None)
    with pytest.raises(FileNotFoundError):
        assert all(
            b"secret" not in path.read_bytes()
            for path in home.rglob("*")
            if path.is_file()
        )

    index.write_bytes(b"old")
    _race_first_read(monkeypatch, index, None)
    assert files_containing(home, b"secret") == []


def test_file_replaced_mid_scan_is_still_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path)
    index = home / "sessions" / "s1" / "history-index.sqlite3"

    _race_first_read(monkeypatch, index, b"rebuilt with secret")

    assert files_containing(home, b"secret") == [index]
