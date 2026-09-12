from __future__ import annotations

import json
import os
import stat
import uuid
from collections.abc import Generator
from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from runpy import run_path
from tempfile import TemporaryDirectory

import httpx
import pytest
from rich.console import Console

_TEST_SITE_PACKAGES = Path(__file__).parent

run_path(_TEST_SITE_PACKAGES / "sitecustomize.py")


def pytest_configure(config: pytest.Config) -> None:
    # Configure HOME before collection: module-level Path.home() values and
    # child processes must use the same watched home as the test fixtures.
    cleanup = ExitStack()
    config.add_cleanup(cleanup.close)
    fake_home = cleanup.enter_context(TemporaryDirectory(prefix="zeta-test-home-"))
    monkeypatch = cleanup.enter_context(pytest.MonkeyPatch.context())
    monkeypatch.setenv("HOME", fake_home)
    # Toolchain caches belong outside the watched home. test_package_app builds
    # the bundle in subprocesses that drop ~/.rustup into HOME, which the guard
    # then reports as an unattributable write and fails the whole session on.
    # The guard exists to catch zeta writing to a user's home, not to police a
    # rust toolchain's cache, so give those their own directory.
    toolchain = cleanup.enter_context(TemporaryDirectory(prefix="zeta-test-toolchain-"))
    monkeypatch.setenv("RUSTUP_HOME", str(Path(toolchain) / "rustup"))
    monkeypatch.setenv("CARGO_HOME", str(Path(toolchain) / "cargo"))
    audit_dir = cleanup.enter_context(TemporaryDirectory(prefix="zeta-test-audit-"))
    config.stash[_HOME_GUARD] = LiveHomeWriteGuard(Path(fake_home), Path(audit_dir))


def _path_snapshot(path: Path) -> tuple[object, ...] | None:
    try:
        metadata = path.lstat()
        content = sha256(path.read_bytes()).digest() if path.is_file() else None
        link_target = path.readlink() if path.is_symlink() else None
    except (FileNotFoundError, OSError):
        return None
    return (
        metadata.st_mode,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        content,
        link_target,
    )


def _persistence_snapshot(live_home: Path) -> dict[Path, tuple[object, ...]]:
    """Capture marker, inode, metadata, and content for a live-home tree."""

    root = live_home.expanduser().resolve()
    if not root.exists():
        return {}

    snapshot: dict[Path, tuple[object, ...]] = {}
    paths = (root, *root.rglob("*"))
    for path in paths:
        path_snapshot = _path_snapshot(path)
        if path_snapshot is not None:
            snapshot[path.relative_to(root)] = path_snapshot
    return snapshot


def _path_content(path: Path) -> bytes | None:
    try:
        return path.read_bytes() if path.is_file() else None
    except (FileNotFoundError, OSError):
        return None


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
    for line in data.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


@dataclass
class _ExternalDeclaration:
    path: Path
    kind: str
    snapshot: tuple[object, ...] | None
    content: bytes | None
    expected: bytes | None = None
    used: bool = False


class LiveHomeWriteGuard:
    """Detect live-home changes and attribute recorded descendant writes."""

    def __init__(self, live_home: Path, audit_dir: Path) -> None:
        self._original_live_home = live_home.expanduser().resolve()
        self.live_home = self._original_live_home
        self._original_live_home_env = os.environ.get("ZETA_TEST_LIVE_HOME")
        self._snapshot = _persistence_snapshot(self.live_home)
        self._original_snapshot = self._snapshot
        self._ownership_token = uuid.uuid4().hex
        audit_dir.mkdir(parents=True, exist_ok=True)
        self._ledger = audit_dir / f"{os.getpid()}.jsonl"
        self._ledger.touch()
        self._ledger_offset = _audit_ledger_size(self._ledger)
        self._external_declarations: list[_ExternalDeclaration] = []

    def _violations(self) -> set[Path]:
        """Fail closed when a changed path has no attributable mutation."""

        current = _persistence_snapshot(self.live_home)
        changed = {
            path
            for path in self._snapshot.keys() | current.keys()
            if self._snapshot.get(path) != current.get(path)
        }
        if not changed:
            return set()

        records = _audit_records(self._ledger, self._ledger_offset)
        owned_paths: set[Path] = set()
        for record in records:
            path_value = record.get("path")
            if not isinstance(path_value, str):
                continue
            try:
                path = Path(path_value).expanduser().resolve()
                relative = path.relative_to(self.live_home)
            except (OSError, RuntimeError, ValueError):
                continue
            if record.get("ownership_token") == self._ownership_token:
                owned_paths.add(relative)

        violations: set[Path] = set()
        tolerated_parents: set[Path] = set()
        deferred: list[Path] = []
        for path in changed:
            if any(
                recorded_path == path or recorded_path.is_relative_to(path)
                for recorded_path in owned_paths
            ):
                violations.add(self.live_home / path)
                continue
            if self._declared_delta_is_tolerated(path, current, tolerated_parents):
                continue
            deferred.append(path)
        for path in deferred:
            if path in tolerated_parents and self._only_directory_metadata_changed(
                path, current
            ):
                continue
            violations.add(self.live_home / path)
        return violations

    def _only_directory_metadata_changed(
        self, relative: Path, current: dict[Path, tuple[object, ...]]
    ) -> bool:
        """Tolerate only the timestamp/size churn a declared child creation implies."""

        before = self._snapshot.get(relative)
        after = current.get(relative)
        return (
            before is not None
            and after is not None
            and stat.S_ISDIR(after[0])
            and after[0] == before[0]
            and after[1] == before[1]
            and after[5] == before[5]
            and after[6] == before[6]
        )

    def _declared_delta_is_tolerated(
        self,
        relative: Path,
        current: dict[Path, tuple[object, ...]],
        tolerated_parents: set[Path],
    ) -> bool:
        path = (self.live_home / relative).resolve()
        appends = [
            declaration
            for declaration in self._external_declarations
            if not declaration.used
            and declaration.path == path
            and declaration.kind == "append"
        ]
        if appends:
            return self._declared_append_chain_matches(
                relative, path, current, appends, tolerated_parents
            )
        for declaration in self._external_declarations:
            if declaration.used or declaration.path != path:
                continue
            if self._snapshot.get(relative) != declaration.snapshot:
                continue
            after = current.get(relative)
            before = declaration.snapshot
            if declaration.kind == "write":
                if (
                    after is not None
                    and stat.S_ISREG(after[0])
                    and (before is None or after[5] != before[5])
                ):
                    declaration.used = True
                    return True
            elif declaration.kind == "mkdir":
                if before is None and after is not None and stat.S_ISDIR(after[0]):
                    declaration.used = True
                    return True
            elif declaration.kind in {"remove", "rmdir"}:
                if before is not None and after is None:
                    declaration.used = True
                    return True
            elif declaration.kind in {"rename", "replace"} and before != after:
                declaration.used = True
                return True
        return False

    def _declared_append_chain_matches(
        self,
        relative: Path,
        path: Path,
        current: dict[Path, tuple[object, ...]],
        appends: list[_ExternalDeclaration],
        tolerated_parents: set[Path],
    ) -> bool:
        """Match the final content against every declared append, in order."""

        base = appends[0]
        after = current.get(relative)
        if (
            after is None
            or not stat.S_ISREG(after[0])
            or self._snapshot.get(relative) != base.snapshot
            or any(declaration.expected is None for declaration in appends)
        ):
            return False
        after_content = _path_content(path)
        if after_content is None or after[2] != len(after_content):
            return False
        if base.snapshot is None:
            before_content = b""
        else:
            if base.content is None:
                return False
            if after[0] != base.snapshot[0] or after[1] != base.snapshot[1]:
                return False
            before_content = base.content
        expected_final = before_content + b"".join(
            declaration.expected or b"" for declaration in appends
        )
        if after_content != expected_final:
            return False
        for declaration in appends:
            declaration.used = True
        if base.snapshot is None:
            tolerated_parents.add(relative.parent)
        return True

    def assert_clean(self) -> None:
        violations = self._violations()
        assert not violations, f"tests wrote to live zeta home: {sorted(violations)}"

    def watch(self, live_home: Path) -> None:
        self.live_home = live_home.expanduser().resolve()
        os.environ["ZETA_TEST_LIVE_HOME"] = str(self.live_home)
        self._snapshot = _persistence_snapshot(self.live_home)
        self._ledger_offset = _audit_ledger_size(self._ledger)
        self._external_declarations.clear()

    def declare_external_mutation(
        self, path: Path, *, kind: str, expected: bytes | None = None
    ) -> None:
        normalized_path = path.expanduser().resolve()
        try:
            normalized_path.relative_to(self.live_home)
        except ValueError as exc:
            raise ValueError("external mutation must be inside the watched home") from exc
        if kind == "append" and expected is None:
            raise ValueError("append declarations must state the expected suffix")
        self._external_declarations.append(
            _ExternalDeclaration(
                path=normalized_path,
                kind=kind,
                snapshot=_path_snapshot(normalized_path),
                content=_path_content(normalized_path) if kind == "append" else None,
                expected=expected,
            )
        )

    def reset(self) -> None:
        self.live_home = self._original_live_home
        if self._original_live_home_env is None:
            os.environ.pop("ZETA_TEST_LIVE_HOME", None)
        else:
            os.environ["ZETA_TEST_LIVE_HOME"] = self._original_live_home_env
        self._snapshot = self._original_snapshot
        self._ledger_offset = _audit_ledger_size(self._ledger)
        self._external_declarations.clear()


_HOME_GUARD = pytest.StashKey[LiveHomeWriteGuard]()


@pytest.fixture(scope="session")
def live_home_write_guard(pytestconfig: pytest.Config) -> LiveHomeWriteGuard:
    return pytestconfig.stash[_HOME_GUARD]


@pytest.fixture(scope="session", autouse=True)
def isolate_zeta_home(
    tmp_path_factory: pytest.TempPathFactory,
    live_home_write_guard: LiveHomeWriteGuard,
) -> Generator[None, None, None]:
    """Keep every test away from the developer's real zeta home."""

    isolated_home = tmp_path_factory.mktemp("zeta-home")
    audit_path = str(_TEST_SITE_PACKAGES)
    inherited_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = (
        f"{audit_path}{os.pathsep}{inherited_pythonpath}"
        if inherited_pythonpath
        else audit_path
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("ZETA_HOME", str(isolated_home))
        monkeypatch.setenv("ZETA_TEST_AUDIT_LEDGER", str(live_home_write_guard._ledger))
        monkeypatch.setenv(
            "ZETA_TEST_LIVE_HOME", str(live_home_write_guard.live_home)
        )
        monkeypatch.setenv(
            "ZETA_TEST_OWNERSHIP_TOKEN", live_home_write_guard._ownership_token
        )
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
