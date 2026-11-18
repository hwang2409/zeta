"""``zeta.config.home_settings``: the one home ``settings.toml`` writer and its restart discovery."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from zeta.config import home_settings
from zeta.config.home_settings import HomeSettingsError
from zeta.config.settings import SettingsError

BUILTIN = "http://127.0.0.1:4323/v1/traces"
ON = [{"key": "telemetry.enabled", "value": True}]
OFF = [{"key": "telemetry.enabled", "value": False}]


def _revision(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _home(tmp_path: Path, body: bytes | None = None) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    if body is not None:
        (home / "settings.toml").write_bytes(body)
    return home


def _refused(code: str, call) -> HomeSettingsError:
    with pytest.raises(HomeSettingsError) as caught:
        call()
    assert caught.value.code == code
    return caught.value


@contextmanager
def _held(path: Path) -> Iterator[None]:
    """Hold an exclusive flock on ``path`` through its own open file description."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


@contextmanager
def _signal_contention(monkeypatch: pytest.MonkeyPatch) -> Iterator[threading.Event]:
    """Set the event when a home-settings writer first finds the settings lock held."""
    contended = threading.Event()
    real = fcntl.flock

    def flock(fd: int, operation: int) -> None:
        try:
            real(fd, operation)
        except BlockingIOError:
            contended.set()
            raise

    monkeypatch.setattr(home_settings.fcntl, "flock", flock)
    monkeypatch.setattr(home_settings, "LOCK_SECONDS", 10.0)
    yield contended


def test_missing_file_projects_defaults_and_update_creates_it(tmp_path: Path) -> None:
    home = _home(tmp_path)

    got = home_settings.get(home)
    updated = home_settings.update(home, got["revision"], ON)

    assert got == {
        "revision": _revision(b""),
        "settings": {"telemetry": {"enabled": False, "source": "off", "editable": True}},
        "settings_malformed": False,
    }
    written = (home / "settings.toml").read_bytes()
    assert written == f'[otel]\nendpoint = "{BUILTIN}"\n'.encode()
    assert updated["revision"] == _revision(written)
    assert updated["settings"] == {"telemetry": {"enabled": True, "source": "builtin", "editable": True}}


@pytest.mark.parametrize("body", [b"model = [\n", b"model = \"\xff\"\n", b"[otel]\nendpoint = 5\n"])
def test_malformed_file_reads_as_unknown_and_is_never_written(tmp_path: Path, body: bytes) -> None:
    home = _home(tmp_path, body)

    got = home_settings.get(home)
    _refused("settings_malformed", lambda: home_settings.update(home, got["revision"], ON))

    assert got == {
        "revision": _revision(body),
        "settings": {"telemetry": {"source": "unknown", "editable": False}},
        "settings_malformed": True,
    }
    assert (home / "settings.toml").read_bytes() == body


@pytest.mark.parametrize("kind", ["symlink", "fifo", "hardlink"])
def test_unsafe_file_is_refused_and_never_followed_or_replaced(tmp_path: Path, kind: str) -> None:
    home = _home(tmp_path)
    target = tmp_path / "elsewhere.toml"
    target.write_bytes(b"# not zeta's\n")
    settings = home / "settings.toml"
    if kind == "symlink":
        settings.symlink_to(target)
    elif kind == "fifo":
        os.mkfifo(settings)
    else:
        os.link(target, settings)

    _refused("settings_unsafe", lambda: home_settings.get(home))
    _refused("settings_unsafe", lambda: home_settings.update(home, _revision(b"# not zeta's\n"), ON))

    assert target.read_bytes() == b"# not zeta's\n"
    assert settings.is_symlink() is (kind == "symlink")


def test_stale_revision_names_the_current_revision_and_writes_nothing(tmp_path: Path) -> None:
    home = _home(tmp_path, b'model = "a"\n')
    old = home_settings.get(home)["revision"]
    (home / "settings.toml").write_bytes(b'model = "b"\n')

    refused = _refused("stale_revision", lambda: home_settings.update(home, old, ON))

    assert refused.data == {"current_revision": _revision(b'model = "b"\n')}
    assert (home / "settings.toml").read_bytes() == b'model = "b"\n'


def test_a_held_lock_is_busy_after_its_deadline_and_writes_nothing(tmp_path: Path) -> None:
    home = _home(tmp_path)

    with _held(home / "settings.toml.lock"):
        refused = _refused("settings_busy", lambda: home_settings.update(home, _revision(b""), ON))

    assert refused.data == {"retryable": True}
    assert not (home / "settings.toml").exists()


def test_comments_and_unknown_keys_survive_both_directions(tmp_path: Path) -> None:
    original = b'# my settings\nmodel = "gpt-5.6-luna"  # pinned\nfuture_key = 1\n\n[future]\nx = "y"  # keep\n'
    home = _home(tmp_path, original)

    on = home_settings.update(home, _revision(original), ON)
    enabled = (home / "settings.toml").read_bytes()
    home_settings.update(home, on["revision"], OFF)

    assert enabled == original + f'\n[otel]\nendpoint = "{BUILTIN}"\n'.encode()
    assert (home / "settings.toml").read_bytes().rstrip(b"\n") == original.rstrip(b"\n")


def test_custom_endpoint_is_read_only_and_byte_identical(tmp_path: Path) -> None:
    original = b'[otel]\nendpoint = "http://127.0.0.1:9999/v1/traces"   # my collector\n'
    home = _home(tmp_path, original)

    got = home_settings.get(home)
    refusals = []
    for changes in (ON, OFF):
        try:
            home_settings.update(home, got["revision"], changes)
        except HomeSettingsError as exc:
            refusals.append(exc.code)
        assert (home / "settings.toml").read_bytes() == original

    assert refusals == ["custom_home_setting", "custom_home_setting"]
    assert got["settings"] == {"telemetry": {"enabled": True, "source": "custom", "editable": False}}


@pytest.mark.parametrize(
    ("revision", "changes"),
    [
        ("sha256:" + "0" * 63, ON),
        (None, ON),
        ("REV", [{"key": "telemetry.endpoint", "value": BUILTIN}]),
        ("REV", [{"key": "otel.endpoint", "value": BUILTIN}]),
        ("REV", [{"key": "yolo", "value": True}]),
        ("REV", [{"key": "telemetry.custom", "value": False}]),
        ("REV", [{"key": "telemetry.enabled", "value": 1}]),
        ("REV", [{"key": "telemetry.enabled", "value": "true"}]),
        ("REV", [{"key": "telemetry.enabled", "value": True, "path": "/tmp"}]),
        ("REV", ON + ON),
        ("REV", []),
        ("REV", {"telemetry.enabled": True}),
    ],
)
def test_unknown_or_malformed_changes_are_invalid_params(tmp_path: Path, revision, changes) -> None:
    home = _home(tmp_path)
    revision = _revision(b"") if revision == "REV" else revision

    _refused("invalid_params", lambda: home_settings.update(home, revision, changes))

    assert not (home / "settings.toml").exists()


def test_candidate_is_validated_before_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = _home(tmp_path, b'model = "a"\n')
    real = home_settings.validate_home_settings

    def validate(raw: bytes, home: Path):
        if BUILTIN.encode() in raw:
            raise SettingsError("candidate rejected")
        return real(raw, home)

    monkeypatch.setattr(home_settings, "validate_home_settings", validate)

    _refused("settings_malformed", lambda: home_settings.update(home, _revision(b'model = "a"\n'), ON))

    assert (home / "settings.toml").read_bytes() == b'model = "a"\n'
    assert sorted(os.listdir(home)) == ["settings.toml", "settings.toml.lock"]


def test_publication_is_private_and_synced_with_its_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = _home(tmp_path, b"")
    os.chmod(home / "settings.toml", 0o644)
    synced: list[str] = []
    real = os.fsync

    def fsync(fd: int) -> None:
        mode = os.fstat(fd).st_mode
        synced.append("directory" if stat.S_ISDIR(mode) else "file")
        real(fd)

    monkeypatch.setattr(os, "fsync", fsync)

    home_settings.update(home, _revision(b""), ON)

    assert stat.S_IMODE(os.stat(home / "settings.toml").st_mode) == 0o600
    assert synced == ["file", "directory"]
    assert sorted(os.listdir(home)) == ["settings.toml", "settings.toml.lock"]


def test_simultaneous_updates_from_one_revision_apply_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = _home(tmp_path)
    results: dict[str, object] = {}
    with _signal_contention(monkeypatch) as contended:
        first_inside = threading.Event()
        real_write = home_settings.write_session_file

        def write(*args, **kwargs) -> None:
            first_inside.set()
            assert contended.wait(10), "the second writer never waited for the settings lock"
            real_write(*args, **kwargs)

        monkeypatch.setattr(home_settings, "write_session_file", write)

        def run(name: str, changes: list) -> None:
            try:
                results[name] = home_settings.update(home, _revision(b""), changes)
            except HomeSettingsError as exc:
                results[name] = exc

        first = threading.Thread(target=run, args=("first", ON))
        first.start()
        assert first_inside.wait(10), "the first writer never reached publication"
        second = threading.Thread(target=run, args=("second", OFF))
        second.start()
        first.join(10)
        second.join(10)

    assert not first.is_alive() and not second.is_alive()
    assert type(results["first"]) is dict
    written = (home / "settings.toml").read_bytes()
    assert isinstance(results["second"], HomeSettingsError)
    assert results["second"].code == "stale_revision"
    assert results["second"].data == {"current_revision": _revision(written)}
    assert written == f'[otel]\nendpoint = "{BUILTIN}"\n'.encode()


def test_hand_edit_before_the_locked_reread_is_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = _home(tmp_path, b'model = "a"\n')
    expected = _revision(b'model = "a"\n')
    outcome: list[object] = []
    with _signal_contention(monkeypatch) as contended:

        def run() -> None:
            try:
                outcome.append(home_settings.update(home, expected, ON))
            except HomeSettingsError as exc:
                outcome.append(exc)

        with _held(home / "settings.toml.lock"):
            writer = threading.Thread(target=run)
            writer.start()
            assert contended.wait(10), "the writer never waited for the settings lock"
            (home / "settings.toml").write_bytes(b'model = "hand"\n')
        writer.join(10)

    assert not writer.is_alive()
    assert isinstance(outcome[0], HomeSettingsError) and outcome[0].code == "stale_revision"
    assert (home / "settings.toml").read_bytes() == b'model = "hand"\n'


def _session(home: Path, name: str, lease: dict | None = None) -> Path:
    directory = home / "sessions" / name
    directory.mkdir(parents=True)
    if lease is not None:
        (directory / "runtime.lease").write_text(json.dumps(lease))
    return directory


def test_restart_lists_every_lease_owner_without_a_cap_or_pid_trust(tmp_path: Path) -> None:
    home = _home(tmp_path)
    owners = sorted(f"{index:032x}" for index in range(25))
    for name in owners:
        _session(home, name, {"pid": 1, "surface": "tui"})
    # pid reuse: the hint names a live process (this one), but nobody holds the lease lock
    _session(home, "f" * 32, {"pid": os.getpid(), "surface": "serve"})
    _session(home, "e" * 32)
    (home / "sessions" / "stray.txt").write_text("not a session")
    held = [os.open(home / "sessions" / name / "runtime.lease", os.O_RDWR) for name in owners]
    try:
        for fd in held:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = home_settings.update(home, _revision(b""), ON)
    finally:
        for fd in held:
            os.close(fd)

    assert result["restart"] == {"serve": True, "sessions": owners, "complete": True}


def test_restart_scan_past_its_deadline_is_incomplete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = _home(tmp_path)
    _session(home, "a" * 32)
    monkeypatch.setattr(home_settings, "RESTART_SCAN_SECONDS", 0.0)

    result = home_settings.update(home, _revision(b""), ON)

    assert result["restart"] == {"serve": True, "sessions": [], "complete": False}


def test_restart_scan_with_an_unreadable_lease_is_incomplete(tmp_path: Path) -> None:
    home = _home(tmp_path)
    os.mkfifo(_session(home, "a" * 32) / "runtime.lease")

    result = home_settings.update(home, _revision(b""), ON)

    assert result["restart"] == {"serve": True, "sessions": [], "complete": False}
