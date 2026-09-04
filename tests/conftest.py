from __future__ import annotations

import json
import os
from collections.abc import Generator
from hashlib import sha256
from pathlib import Path

import httpx
import pytest
from rich.console import Console

LIVE_ZETA_HOME = Path.home() / ".zeta"
_TEST_SITE_PACKAGES = Path(__file__).parent


def _persistence_snapshot(live_home: Path) -> dict[Path, tuple[object, ...]]:
    """Capture marker, inode, metadata, and content for a live-home tree."""

    root = live_home.expanduser().resolve()
    if not root.exists():
        return {}

    snapshot: dict[Path, tuple[object, ...]] = {}
    paths = (root, *root.rglob("*"))
    for path in paths:
        try:
            metadata = path.lstat()
            content = sha256(path.read_bytes()).digest() if path.is_file() else None
            link_target = path.readlink() if path.is_symlink() else None
        except (FileNotFoundError, OSError):
            continue
        snapshot[path.relative_to(root)] = (
            metadata.st_mode,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
            content,
            link_target,
        )
    return snapshot


def _audit_ledger_size(ledger: Path) -> int:
    try:
        return ledger.stat().st_size
    except FileNotFoundError:
        return 0


def _audit_records(ledger: Path, offset: int) -> list[dict[str, object]]:
    try:
        with ledger.open("rb") as stream:
            stream.seek(offset)
            data = stream.read()
    except FileNotFoundError:
        return []

    records: list[dict[str, object]] = []
    for line in data.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


class LiveHomeWriteGuard:
    """Detect live-home changes and attribute recorded descendant writes."""

    def __init__(self, live_home: Path, audit_dir: Path) -> None:
        self._original_live_home = live_home.expanduser().resolve()
        self.live_home = self._original_live_home
        self._original_live_home_env = os.environ.get("ZETA_TEST_LIVE_HOME")
        self._snapshot = _persistence_snapshot(self.live_home)
        self._original_snapshot = self._snapshot
        self._root_session_id = os.getsid(0)
        audit_dir.mkdir(parents=True, exist_ok=True)
        self._ledger = audit_dir / f"{os.getpid()}.jsonl"
        self._ledger.touch()
        self._ledger_offset = _audit_ledger_size(self._ledger)

    def _violations(self) -> set[Path]:
        """Fail closed when a changed path has no attributable writer record."""

        current = _persistence_snapshot(self.live_home)
        changed = {
            path
            for path in self._snapshot.keys() | current.keys()
            if self._snapshot.get(path) != current.get(path)
        }
        if not changed:
            return set()

        records = _audit_records(self._ledger, self._ledger_offset)
        recorded_by_path: dict[Path, set[int]] = {}
        for record in records:
            path_value = record.get("path")
            session_id = record.get("session_id")
            if not isinstance(path_value, str) or not isinstance(session_id, int):
                continue
            try:
                path = Path(path_value).expanduser().resolve()
                relative = path.relative_to(self.live_home)
            except (OSError, RuntimeError, ValueError):
                continue
            recorded_by_path.setdefault(relative, set()).add(session_id)

        violations: set[Path] = set()
        for path in changed:
            sessions = {
                session_id
                for recorded_path, recorded_sessions in recorded_by_path.items()
                if recorded_path == path or recorded_path.is_relative_to(path)
                for session_id in recorded_sessions
            }
            if not sessions or self._root_session_id in sessions:
                violations.add(self.live_home / path)
        return violations

    def assert_clean(self) -> None:
        violations = self._violations()
        assert not violations, f"tests wrote to live zeta home: {sorted(violations)}"

    def watch(self, live_home: Path) -> None:
        self.live_home = live_home.expanduser().resolve()
        os.environ["ZETA_TEST_LIVE_HOME"] = str(self.live_home)
        self._snapshot = _persistence_snapshot(self.live_home)
        self._ledger_offset = _audit_ledger_size(self._ledger)

    def reset(self) -> None:
        self.live_home = self._original_live_home
        if self._original_live_home_env is None:
            os.environ.pop("ZETA_TEST_LIVE_HOME", None)
        else:
            os.environ["ZETA_TEST_LIVE_HOME"] = self._original_live_home_env
        self._snapshot = self._original_snapshot
        self._ledger_offset = _audit_ledger_size(self._ledger)


@pytest.fixture(scope="session")
def live_home_write_guard(
    tmp_path_factory: pytest.TempPathFactory,
) -> LiveHomeWriteGuard:
    audit_dir = tmp_path_factory.mktemp("zeta-live-home-audit")
    return LiveHomeWriteGuard(LIVE_ZETA_HOME, audit_dir)


@pytest.fixture(scope="session", autouse=True)
def isolate_zeta_home(
    tmp_path_factory: pytest.TempPathFactory,
    live_home_write_guard: LiveHomeWriteGuard,
) -> Generator[None, None, None]:
    """Keep every test away from the developer's real zeta home."""

    isolated_home = tmp_path_factory.mktemp("zeta-home")
    fake_home = isolated_home.parent / "home"
    audit_path = str(_TEST_SITE_PACKAGES)
    inherited_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = (
        f"{audit_path}{os.pathsep}{inherited_pythonpath}"
        if inherited_pythonpath
        else audit_path
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("ZETA_HOME", str(isolated_home))
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setenv("ZETA_TEST_AUDIT_LEDGER", str(live_home_write_guard._ledger))
        monkeypatch.setenv("ZETA_TEST_LIVE_HOME", str(LIVE_ZETA_HOME))
        monkeypatch.setenv("PYTHONPATH", pythonpath)
        yield
        assert (
            Path(os.environ["ZETA_HOME"]).expanduser().resolve()
            == isolated_home.resolve()
        )
        live_home_write_guard.assert_clean()


@pytest.fixture(autouse=True)
def stable_terminal_defaults(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Generator[None, None, None]:
    """Keep Rich output stable unless a test explicitly probes the environment."""
    if request.node.get_closest_marker("environment_sensitive") is not None:
        yield
        return

    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLORTERM", "truecolor")
    assert os.environ["TERM"] == "xterm-256color"
    assert os.environ["COLORTERM"] == "truecolor"

    original_init = Console.__init__

    def init_with_stable_defaults(
        self: Console, *args: object, **kwargs: object
    ) -> None:
        if kwargs.get("force_terminal") is not False:
            kwargs.setdefault("force_terminal", True)
            kwargs.setdefault("color_system", "truecolor")
            kwargs.setdefault("no_color", False)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(Console, "__init__", init_with_stable_defaults)

    yield


@pytest.fixture(autouse=True)
def block_real_http_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked_sync(
        self: httpx.HTTPTransport, request: httpx.Request
    ) -> httpx.Response:
        del self, request
        raise AssertionError("real network connections are forbidden in tests")

    async def blocked_async(
        self: httpx.AsyncHTTPTransport, request: httpx.Request
    ) -> httpx.Response:
        del self, request
        raise AssertionError("real network connections are forbidden in tests")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked_sync)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", blocked_async)
