from __future__ import annotations

import json
import os
import stat
import sys
import threading
import uuid
import weakref
from collections.abc import Generator
from contextlib import ExitStack
from dataclasses import dataclass
from functools import wraps
from hashlib import sha256
from pathlib import Path
from runpy import run_path
from statistics import median
from tempfile import TemporaryDirectory

import httpx
import pytest
from rich.console import Console

_TEST_SITE_PACKAGES = Path(__file__).parent.resolve()

run_path(_TEST_SITE_PACKAGES / "sitecustomize.py")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if item.path.resolve().is_relative_to(_TEST_SITE_PACKAGES):
            item.fixturenames.append("close_direct_test_runtime_owners")


def pytest_configure(config: pytest.Config) -> None:
    # Configure HOME before collection: module-level Path.home() values and
    # child processes must use the same watched home as the test fixtures.
    cleanup = ExitStack()
    config.add_cleanup(cleanup.close)
    fake_home = cleanup.enter_context(TemporaryDirectory(prefix="zeta-test-home-"))
    monkeypatch = cleanup.enter_context(pytest.MonkeyPatch.context())
    monkeypatch.setenv("HOME", fake_home)
    monkeypatch.setenv("ZETA_STALL_WATCHDOG", "0")
    # Toolchain caches (rustup/cargo) invoked by any subprocess belong outside
    # the watched home; the guard polices zeta's writes, not toolchain caches.
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


# These variables control the test harness itself and must survive environment
# isolation.  Keep this list explicit so adding a ZETA_* setting cannot silently
# change CI behavior.
_TEST_HARNESS_ENV_ALLOWLIST = frozenset()


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
        # Configuration must come from each test, never from the developer's
        # shell.  Tests that exercise an environment-controlled behavior can
        # set it explicitly with their own monkeypatch calls.
        for name in tuple(os.environ):
            if name.startswith("ZETA_") and name not in _TEST_HARNESS_ENV_ALLOWLIST:
                monkeypatch.delenv(name, raising=False)
        for name in (
            "ANTHROPIC_API_KEY",
            "CLAUDE_CONFIG_DIR",
            "CODEX_HOME",
            "WIKI_AGENT_RUNTIME_DIR",
        ):
            monkeypatch.delenv(name, raising=False)

        monkeypatch.setenv("ZETA_HOME", str(isolated_home))
        monkeypatch.setenv("ZETA_TEST_SCRIPTED_PROVIDER", "1")
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


_FD_GROWTH: list[tuple[int, str]] = []
_FD_BREACHES: list[str] = []
# Leave 25% of FD_SETSIZE free for pytest, xdist, and a test's active work.
_FD_HIGH_WATER = 768
_FD_MAX = -1
_FD_OPEN_MAX = 0
_FD_RETAINED_COUNTS: list[int] = []
_FD_CURRENT_NODE = "between-tests"
_FD_PEAK_MAX = -1
_FD_PEAK_OPEN = 0
_FD_PEAK_NODE = "none"
_FD_MONITOR_STOP: threading.Event | None = None
_FD_MONITOR: threading.Thread | None = None
_OS_LISTDIR = os.listdir
_OS_FSTAT = os.fstat


def _describe_fd(descriptor: int, directory: str) -> str:
    try:
        target = os.readlink(f"{directory}/{descriptor}")
    except OSError:
        target = "unknown"
    try:
        mode = os.fstat(descriptor).st_mode
    except OSError:
        return target
    kind = next(
        (
            name
            for predicate, name in (
                (stat.S_ISREG, "file"),
                (stat.S_ISDIR, "directory"),
                (stat.S_ISFIFO, "pipe"),
                (stat.S_ISSOCK, "socket"),
                (stat.S_ISCHR, "character-device"),
            )
            if predicate(mode)
        ),
        "other",
    )
    return f"{target} ({kind})"


def _open_file_descriptors() -> dict[int, str]:
    for directory in ("/proc/self/fd", "/dev/fd"):
        try:
            names = _OS_LISTDIR(directory)
        except OSError:
            continue
        descriptors = {}
        for name in names:
            try:
                descriptor = int(name)
                _OS_FSTAT(descriptor)
            except (OSError, ValueError):
                continue
            descriptors[descriptor] = _describe_fd(descriptor, directory)
        return descriptors
    return {}


def _fd_numbers() -> list[int]:
    try:
        return [int(name) for name in _OS_LISTDIR("/proc/self/fd")]
    except (OSError, ValueError):
        return []


def _monitor_fd_peak(stop: threading.Event) -> None:
    global _FD_PEAK_MAX, _FD_PEAK_OPEN, _FD_PEAK_NODE
    while not stop.wait(0.02):
        descriptors = _fd_numbers()
        highest = max(descriptors, default=-1)
        if highest > _FD_PEAK_MAX:
            _FD_PEAK_MAX = highest
            _FD_PEAK_OPEN = len(descriptors)
            _FD_PEAK_NODE = _FD_CURRENT_NODE


def pytest_sessionstart(session: pytest.Session) -> None:
    global _FD_MONITOR_STOP, _FD_MONITOR
    if sys.platform.startswith("linux") and hasattr(session.config, "workerinput"):
        _FD_MONITOR_STOP = threading.Event()
        _FD_MONITOR = threading.Thread(
            target=_monitor_fd_peak,
            args=(_FD_MONITOR_STOP,),
            name="pytest-fd-peak",
            daemon=True,
        )
        _FD_MONITOR.start()


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None):
    """Report retained fds and stop workers before they approach FD_SETSIZE."""
    del nextitem
    global _FD_CURRENT_NODE, _FD_MAX, _FD_OPEN_MAX
    # Stable pytest/xdist descriptors and session fixtures predate this snapshot.
    before = _open_file_descriptors()
    _FD_MAX = max((_FD_MAX, *before), default=-1)
    _FD_OPEN_MAX = max(_FD_OPEN_MAX, len(before))
    _FD_CURRENT_NODE = item.nodeid
    yield
    _FD_CURRENT_NODE = "between-tests"
    after = _open_file_descriptors()
    _FD_MAX = max((_FD_MAX, *after), default=-1)
    _FD_OPEN_MAX = max(_FD_OPEN_MAX, len(after))
    _FD_RETAINED_COUNTS.append(len(after))
    retained = sorted(after.keys() - before.keys())
    _FD_GROWTH.append((len(retained), item.nodeid))
    if retained:
        details = ", ".join(f"fd {fd}: {after[fd]}" for fd in retained)
        print(f"fd-retained nodeid={item.nodeid} {details}", flush=True)
    highest = max(after, default=-1)
    if highest >= _FD_HIGH_WATER:
        breach = (
            f"{item.nodeid} reached fd {highest}; retained descriptors: "
            f"{details if retained else 'none added by this test'}"
        )
        _FD_BREACHES.append(breach)
        item.session.shouldstop = breach


def _fd_summary() -> str:
    worker = os.environ.get("PYTEST_XDIST_WORKER", "master")
    top = sorted(_FD_GROWTH, reverse=True)[:5]
    offenders = "; ".join(f"{nodeid} (+{count})" for count, nodeid in top)
    breach = _FD_BREACHES[0] if _FD_BREACHES else "none"
    if _FD_RETAINED_COUNTS:
        quarter = max(1, len(_FD_RETAINED_COUNTS) // 4)
        retained = (
            f"start={_FD_RETAINED_COUNTS[0]} end={_FD_RETAINED_COUNTS[-1]} "
            f"max={max(_FD_RETAINED_COUNTS)} "
            f"q1_median={median(_FD_RETAINED_COUNTS[:quarter]):.1f} "
            f"q4_median={median(_FD_RETAINED_COUNTS[-quarter:]):.1f}"
        )
    else:
        retained = "start=0 end=0 max=0 q1_median=0.0 q4_median=0.0"
    return (
        f"fd-summary worker={worker} retained_open=[{retained}] "
        f"retained_max_fd={_FD_MAX} active_peak_fd={_FD_PEAK_MAX} "
        f"active_peak_open={_FD_PEAK_OPEN} active_peak_test={_FD_PEAK_NODE} "
        f"top_retained=[{offenders}] breach=[{breach}]"
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    del exitstatus
    if _FD_MONITOR_STOP is not None:
        _FD_MONITOR_STOP.set()
    if _FD_MONITOR is not None:
        _FD_MONITOR.join(timeout=1)
    summary = _fd_summary()
    if _FD_BREACHES:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
    worker_output = getattr(session.config, "workeroutput", None)
    if worker_output is not None:
        worker_output["fd_summary"] = summary
    elif _FD_MAX >= 0:
        print(summary, flush=True)


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error) -> None:
    del error
    summary = node.workeroutput.get("fd_summary")
    if summary is not None:
        print(summary, flush=True)


@pytest.fixture
async def close_direct_test_runtime_owners(monkeypatch: pytest.MonkeyPatch):
    """Close stores and registries constructed directly by repository tests."""
    from zeta.core.store import ConversationStore
    from zeta.tools import ToolRegistry

    stores: list[weakref.ReferenceType[ConversationStore]] = []
    registries: list[weakref.ReferenceType[ToolRegistry]] = []
    store_init = ConversationStore.__init__
    registry_init = ToolRegistry.__init__
    store_close = ConversationStore.close
    registry_close = ToolRegistry.close

    @wraps(store_init)
    def tracked_store_init(store, *args, **kwargs):
        store_init(store, *args, **kwargs)
        caller = Path(sys._getframe(1).f_code.co_filename).resolve()
        if caller.is_relative_to(_TEST_SITE_PACKAGES):
            stores.append(weakref.ref(store))

    @wraps(registry_init)
    def tracked_registry_init(registry, *args, **kwargs):
        registry_init(registry, *args, **kwargs)
        caller = Path(sys._getframe(1).f_code.co_filename).resolve()
        if caller.is_relative_to(_TEST_SITE_PACKAGES):
            registries.append(weakref.ref(registry))

    monkeypatch.setattr(ConversationStore, "__init__", tracked_store_init)
    monkeypatch.setattr(ToolRegistry, "__init__", tracked_registry_init)
    yield
    for reference in reversed(registries):
        registry = reference()
        if registry is not None and not registry._closed:
            await registry_close(registry)
    for reference in reversed(stores):
        store = reference()
        if store is not None and not store._closed:
            store_close(store)


@pytest.fixture
async def close_direct_test_agent_loops(monkeypatch: pytest.MonkeyPatch):
    """Close AgentLoops whose constructor caller is test code, never src/zeta."""
    from zeta.runtime.loop import AgentLoop
    from zeta.tools import ToolRegistry

    loops: list[AgentLoop] = []
    loop_init = AgentLoop.__init__
    loop_close = AgentLoop.close
    registry_close = ToolRegistry.close

    @wraps(loop_init)
    def tracked_init(loop, *args, **kwargs):
        loop_init(loop, *args, **kwargs)
        caller = Path(sys._getframe(1).f_code.co_filename).resolve()
        if caller.is_relative_to(_TEST_SITE_PACKAGES):
            loops.append(loop)

    monkeypatch.setattr(AgentLoop, "__init__", tracked_init)
    yield
    for loop in reversed(loops):
        if not loop._closed:
            await loop_close(loop)
        if not loop.tool_registry._closed:
            await registry_close(loop.tool_registry)
        if not loop.store._closed:
            loop.store.close()


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

@pytest.fixture(autouse=True)
def _inject_scripted_completion_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep provider-backed tests offline through the production backend seam."""
    try:
        from tests.support.tui_backend import FakeInteractiveBackend
    except ModuleNotFoundError:
        return
    from zeta.tui import app as tui_app

    def build_backend(provider: str, model: str | None, **_kwargs: object):
        selected = model or {
            "claude": "claude-sonnet-4-6",
            "codex": "gpt-5.6-luna",
            "ollama": "qwen3:4b",
        }[provider]
        return FakeInteractiveBackend(delay=0, model=selected), selected

    monkeypatch.setattr(tui_app, "build_backend", build_backend)
