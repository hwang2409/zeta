#!/usr/bin/env python3
"""Store, eviction, context, child-RPC, serve, and search probes on copied sessions.

Exploratory only (not a CI gate). Every probe calls a named production
function on a clonefile copy of a real session inside a per-run sandbox
(sandbox.py): the copy is pinned to a local provider stub, credential sources
are removed from the environment, a network guard refuses non-local hosts, and
the sandbox is deleted on exit unless --keep is given. Each probe runs three
separate passes so the numbers do not disturb each other:

1. a timed pass with no instrumentation (n samples, p50/p95, a load reading
   per sample);
2. an operation-count pass with patched os/json/model entry points;
3. a cProfile pass written to ~/.zeta/scratch/perf/profiles/.

Run one probe at a time under the shared lock, for example:

    lockf -k ~/.zeta/scratch/perf/bench.lock \
        uv run python scripts/bench/store_probes.py open --session phoebe
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import cProfile
import io
import json
import os
import pstats
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from sandbox import (
    STUB_MODEL,
    STUB_PROVIDER,
    Sandbox,
    load1,
    pin_stub_provider,
    summary,
    timed,
)

import zeta
from zeta.core import checkpoints
from zeta.core.checkpoints import ConversationEntry
from zeta.protocol.types import (
    Message,
    MessageOrigin,
    MessageRole,
    TextContent,
    with_message_origin,
)
from zeta.storage_format import advance

SOURCES = {
    "phoebe": Path.home() / ".zeta/sessions/7897c1aed2e5444caba907f462306b10",
    "orchestrator": Path.home() / ".zeta/sessions/86e3d775a40b42ceaa76ca571e27ff1b",
}
PERF = Path.home() / ".zeta/scratch/perf"
PROFILE_DIR = PERF / "profiles"
RESULTS = PERF / "store"
WORKTREE = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- utilities


def load() -> list[float]:
    return [round(value, 2) for value in os.getloadavg()]


def sample(fn: Callable[[], Any], n: int) -> tuple[dict[str, Any], Any]:
    samples: list[tuple[float, float]] = []
    value = None
    for _ in range(n):
        value, one = timed(fn)
        samples.append(one)
    return summary(samples), value


def profile(name: str, fn: Callable[[], Any]) -> dict[str, Any]:
    """Profile one call; return the .prof path and the top-15 cumulative rows."""
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    path = PROFILE_DIR / f"store-{name}.prof"
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        fn()
    finally:
        profiler.disable()
    profiler.dump_stats(path)
    text = io.StringIO()
    stats = pstats.Stats(profiler, stream=text)
    stats.sort_stats("cumulative").print_stats(15)
    (PROFILE_DIR / f"store-{name}.txt").write_text(text.getvalue())
    rows = []
    for (file, line, func), (_cc, nc, tt, ct, _callers) in sorted(
        stats.stats.items(), key=lambda item: item[1][3], reverse=True
    ):
        if "zeta" not in file:
            continue
        rows.append(
            f"{ct * 1000:9.1f}ms cum {tt * 1000:8.1f}ms self n={nc} {_short(file)}:{line} {func}"
        )
        if len(rows) == 15:
            break
    return {"profile": str(path), "top15_zeta_cumulative": rows}


def _short(file: str) -> str:
    marker = "/src/zeta/"
    return "src/zeta/" + file.split(marker, 1)[1] if marker in file else file


# ------------------------------------------------------- operation counting


class Ops:
    """Count file, decode, and model operations while active.

    - opens: every open audit event (os.open, io.open, os.fdopen with a path)
    - read_open_bytes: file size at open for read-only opens (upper bound of
      bytes read; exact when the whole file is read, as the store does)
    - os_read_bytes / pread_bytes: exact bytes from raw fd reads
    - json_loads / json_bytes: json.loads calls and input bytes
    - entry_from_dict / message_from_dict: model decodes
    - deepcopy: top-level copy.deepcopy calls
    - active_branch / active_branch_rows: branch walks and rows walked
    - snapshot_log_state_rows: rows copied by ConversationStore._snapshot_log_state
    - validate_entries_rows: rows walked by ConversationStore._validate_entries
    - load_full: full transcript reloads; fsync; sleep (cooperative yields)
    """

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.files: Counter[str] = Counter()
        self.active = False
        self._local = threading.local()
        self._undo: list[Callable[[], None]] = []
        sys.addaudithook(self._audit)

    def _audit(self, event: str, args: tuple[Any, ...]) -> None:
        if self.active and event == "open":
            self.counts["opens"] += 1
            path = args[0]
            if isinstance(path, (str, bytes, os.PathLike)):
                self.files[os.path.basename(os.fsdecode(path))] += 1

    def _patch(self, owner: Any, name: str, wrapper: Callable[[Any], Any]) -> None:
        original = getattr(owner, name)
        setattr(owner, name, wrapper(original))
        self._undo.append(lambda: setattr(owner, name, original))

    def _patch_everywhere(self, original: Any, replacement: Any) -> None:
        for module in list(sys.modules.values()):
            if module is None or not getattr(module, "__name__", "").startswith("zeta"):
                continue
            for attr, value in list(vars(module).items()):
                if value is original:
                    setattr(module, attr, replacement)
                    self._undo.append(lambda m=module, a=attr: setattr(m, a, original))

    def _size_of_fd(self, fd: int, flags: int) -> None:
        if flags & (os.O_WRONLY | os.O_RDWR):
            return
        with contextlib.suppress(OSError):
            self.counts["read_open_bytes"] += os.fstat(fd).st_size

    def install(self) -> None:
        from zeta.core.store import ConversationStore

        counts = self.counts
        ops = self

        def wrap_open(original: Any) -> Any:
            def opener(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
                fd = original(path, flags, *args, **kwargs)
                if ops.active:
                    ops._size_of_fd(fd, flags)
                return fd

            return opener

        def wrap_io_open(original: Any) -> Any:
            def opener(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
                handle = original(file, mode, *args, **kwargs)
                if (
                    ops.active
                    and not isinstance(file, int)
                    and not any(c in mode for c in "wa+x")
                ):
                    with contextlib.suppress(OSError, ValueError, AttributeError):
                        counts["read_open_bytes"] += os.fstat(handle.fileno()).st_size
                return handle

            return opener

        def wrap_read(key: str) -> Callable[[Any], Any]:
            def wrapper(original: Any) -> Any:
                def read(*args: Any) -> bytes:
                    data = original(*args)
                    if ops.active:
                        counts[key] += len(data)
                        counts[key.replace("_bytes", "_calls")] += 1
                    return data

                return read

            return wrapper

        def wrap_count(key: str) -> Callable[[Any], Any]:
            def wrapper(original: Any) -> Any:
                def call(*args: Any, **kwargs: Any) -> Any:
                    if ops.active:
                        counts[key] += 1
                    return original(*args, **kwargs)

                return call

            return wrapper

        def wrap_sleep(original: Any) -> Any:
            def sleeper(seconds: float) -> None:
                if ops.active:
                    counts["sleep_calls"] += 1
                    counts["sleep_requested_us"] += int(seconds * 1e6)
                original(seconds)

            return sleeper

        def wrap_loads(original: Any) -> Any:
            def loads(source: Any, *args: Any, **kwargs: Any) -> Any:
                if ops.active:
                    counts["json_loads"] += 1
                    counts["json_bytes"] += len(source)
                return original(source, *args, **kwargs)

            return loads

        def wrap_deepcopy(original: Any) -> Any:
            def deepcopy(value: Any, memo: Any = None, _nil: Any = []) -> Any:  # noqa: B006
                depth = getattr(ops._local, "depth", 0)
                if ops.active and depth == 0:
                    counts["deepcopy"] += 1
                ops._local.depth = depth + 1
                try:
                    return original(value, memo)
                finally:
                    ops._local.depth = depth

            return deepcopy

        self._patch(os, "open", wrap_open)
        self._patch(io, "open", wrap_io_open)
        import builtins

        self._patch(builtins, "open", wrap_io_open)
        self._patch(os, "read", wrap_read("os_read_bytes"))
        self._patch(os, "pread", wrap_read("pread_bytes"))
        self._patch(os, "fsync", wrap_count("fsync"))
        self._patch(time, "sleep", wrap_sleep)
        self._patch(json, "loads", wrap_loads)
        self._patch(copy, "deepcopy", wrap_deepcopy)

        for cls, key in (
            (ConversationEntry, "entry_from_dict"),
            (Message, "message_from_dict"),
        ):
            original = cls.__dict__["from_dict"].__func__

            def classmethod_wrapper(original: Any = original, key: str = key) -> Any:
                def from_dict(klass: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
                    if ops.active:
                        counts[key] += 1
                    return original(klass, value, *args, **kwargs)

                return classmethod(from_dict)

            cls.from_dict = classmethod_wrapper()
            self._undo.append(
                lambda cls=cls, original=original: setattr(
                    cls, "from_dict", classmethod(original)
                )
            )

        original_branch = checkpoints.active_branch

        def active_branch(entries: Any) -> Any:
            result = original_branch(entries)
            if ops.active:
                counts["active_branch"] += 1
                counts["active_branch_rows"] += len(result)
            return result

        self._patch_everywhere(original_branch, active_branch)

        def method_rows(name: str, key: str) -> None:
            original = getattr(ConversationStore, name)

            def method(store: Any, *args: Any, **kwargs: Any) -> Any:
                if ops.active:
                    counts[key] += len(store._entries)
                    counts[key.replace("_rows", "_calls")] += 1
                return original(store, *args, **kwargs)

            setattr(ConversationStore, name, method)
            self._undo.append(lambda: setattr(ConversationStore, name, original))

        method_rows("_snapshot_log_state", "snapshot_log_state_rows")
        method_rows("_validate_entries", "validate_entries_rows")
        method_rows("_load_full", "load_full_rows_before")

    def uninstall(self) -> None:
        while self._undo:
            self._undo.pop()()

    @contextlib.contextmanager
    def measure(self) -> Iterator[dict[str, Any]]:
        result: dict[str, Any] = {}
        self.counts.clear()
        self.files.clear()
        self.install()
        self.active = True
        try:
            yield result
        finally:
            self.active = False
            self.uninstall()
            result.update(dict(self.counts))
            result["top_files_opened"] = dict(self.files.most_common(6))


OPS = Ops()


def count_ops(fn: Callable[[], Any]) -> dict[str, Any]:
    with OPS.measure() as result:
        fn()
    return result


# ---------------------------------------------------------- copied homes


def _meta(session: Path) -> dict[str, Any]:
    return json.loads((session / "meta.json").read_text())


BOX: Sandbox | None = None  # the per-run sandbox, set by main()


def _home_path(name: str, fraction: float) -> Path:
    assert BOX is not None, "probes run inside main()'s sandbox"
    return BOX.root / f"{name}-{round(fraction * 100)}"


def reset_home(name: str, fraction: float = 1.0) -> None:
    """Drop a grown copy so the next probe in this run re-clones it."""
    shutil.rmtree(_home_path(name, fraction), ignore_errors=True)


def make_home(name: str, fraction: float = 1.0) -> tuple[Path, str]:
    """Return a sandbox home holding a clonefile copy (cut at a row boundary)."""
    from zeta.core.session import SessionManager

    home = _home_path(name, fraction)
    ready = home / ".probe-ready"
    assert BOX is not None
    BOX.use_home(home)
    if ready.exists():
        return home, ready.read_text()
    (home / "sessions").mkdir(parents=True)
    (home / "projects").mkdir(mode=0o700)
    os.chmod(home, 0o700)
    advance(home, expected_epoch=0)
    if name == "fresh":
        opened = SessionManager(home).create(
            provider=STUB_PROVIDER, model=STUB_MODEL, cwd=WORKTREE, auto_project=False
        )
        opened.store.close()
        sid = opened.metadata.session_id
    else:
        sid = _clone_session(SOURCES[name], home, fraction)
    BOX.check_stubbed(home)
    ready.write_text(sid)
    return home, sid


def _clone_session(source: Path, home: Path, fraction: float) -> str:
    target = home / "sessions" / source.name
    subprocess.run(["cp", "-cR", str(source), str(target)], check=True)
    # Metadata comes from the copy, never the live source.
    project_id = pin_stub_provider(target).get("project_id")
    if project_id:
        project = Path.home() / ".zeta/projects" / project_id
        subprocess.run(
            ["cp", "-cR", str(project), str(home / "projects" / project_id)],
            check=True,
        )
    conversation = target / "conversation.jsonl"
    raw = conversation.read_bytes()
    raw = raw[: raw.rfind(b"\n") + 1]  # the live copy may end mid-row
    lines = raw.splitlines(keepends=True)
    keep = max(2, int(len(lines) * fraction))
    conversation.write_bytes(b"".join(lines[:keep]))
    for path in target.glob("*.sqlite*"):
        path.unlink()
    # Child receipts store absolute paths under the real home; point them
    # at the copy so agent_status/agent_output resolve the copied children.
    for path in (conversation, target / "session_state.json"):
        if path.exists():
            path.write_bytes(
                path.read_bytes().replace(str(source).encode(), str(target).encode())
            )
    return source.name


def scale(home: Path, sid: str) -> dict[str, Any]:
    session = home / "sessions" / sid
    conversation = session / "conversation.jsonl"
    with conversation.open("rb") as handle:
        rows = sum(1 for _ in handle)
    agents = session / "agents"
    return {
        "rows": rows,
        "bytes": conversation.stat().st_size,
        "child_dirs": len(list(agents.iterdir())) if agents.is_dir() else 0,
    }


def user_message(text: str) -> Message:
    return with_message_origin(
        Message(MessageRole.USER, [TextContent(text)]), MessageOrigin.USER
    )


def fractions_for(args: argparse.Namespace) -> list[float]:
    return [float(value) for value in args.fractions.split(",")]


# ------------------------------------------------------------- 1. open/replay


def probe_open(args: argparse.Namespace) -> dict[str, Any]:
    """SessionManager.open(sid) (writable resume), store.replay(), store.messages()."""

    out: dict[str, Any] = {
        "functions": [
            "zeta.core.session.SessionManager.open (writable, fresh manager each sample)",
            "ConversationStore.replay",
            "ConversationStore.messages",
        ]
    }
    for fraction in fractions_for(args):
        out[str(fraction)] = _open_row(args, fraction)
    return out


def _open_row(args: argparse.Namespace, fraction: float) -> dict[str, Any]:
    from zeta.core.session import SessionManager

    home, sid = make_home(args.session, fraction)
    row: dict[str, Any] = {"scale": scale(home, sid)}

    def open_close() -> None:
        opened = SessionManager(home).open(sid)
        opened.store.close()

    row["open"], _ = sample(open_close, args.n)
    row["open_ops"] = count_ops(open_close)
    opened = SessionManager(home).open(sid)
    store = opened.store
    try:
        row["replay"], entries = sample(store.replay, args.n)
        row["replay_rows"] = len(entries)
        row["replay_ops"] = count_ops(store.replay)
        row["messages"], messages = sample(store.messages, args.n)
        row["messages_count"] = len(messages)
        row["messages_ops"] = count_ops(store.messages)
        if fraction == 1.0:
            row["replay_profile"] = profile(f"replay-{args.session}", store.replay)
    finally:
        store.close()
    if fraction == 1.0:
        row["open_profile"] = profile(f"open-{args.session}", open_close)
    return row


# ------------------------------------------------------------- 2. append cost


def probe_append(args: argparse.Namespace) -> dict[str, Any]:
    """append_message (sync), append_message_async (production durable path), append_many."""

    out: dict[str, Any] = {
        "functions": [
            "ConversationStore.append_message (sync owner path)",
            "ConversationStore.append_message_async (runtime durable-writer path)",
            "ConversationStore.append_many (batch path used by child questions, pending prompts)",
        ]
    }
    for fraction in fractions_for(args):
        out[str(fraction)] = _append_row(args, fraction)
    # Undo growth so later probes see the original copy.
    for fraction in fractions_for(args):
        reset_home(args.session, fraction)
    return out


def _append_row(args: argparse.Namespace, fraction: float) -> dict[str, Any]:
    from zeta.core.session import SessionManager

    home, sid = make_home(args.session, fraction)
    row: dict[str, Any] = {"scale": scale(home, sid)}
    opened = SessionManager(home).open(sid)
    store = opened.store
    try:
        counter = iter(range(10**9))
        row["append_sync"], _ = sample(
            lambda: store.append_message(user_message(f"probe {next(counter)}")), args.n
        )
        row["append_sync_ops"] = count_ops(
            lambda: store.append_message(user_message("probe ops"))
        )
        row["append_many"], _ = sample(
            lambda: store.append_many(
                [("warning", {"message": f"probe {next(counter)}"})]
            ),
            args.n,
        )
        row["append_many_ops"] = count_ops(
            lambda: store.append_many([("warning", {"message": "probe ops"})])
        )

        async def run_async() -> dict[str, Any]:
            store.bind_owner_thread()
            store._prepare_durable_writer_snapshot()
            # First write builds the persistent durable writer (one full load).
            first_started = time.perf_counter_ns()
            with OPS.measure() as first_ops:
                await store.append_message_async(user_message("probe first async"))
            first_ms = (time.perf_counter_ns() - first_started) / 1e6
            times = []
            for index in range(args.n):
                started = time.perf_counter_ns()
                await store.append_message_async(user_message(f"probe async {index}"))
                times.append(((time.perf_counter_ns() - started) / 1e6, load1()))
            with OPS.measure() as steady_ops:
                await store.append_message_async(user_message("probe async ops"))
            return {
                "append_async_first_ms": round(first_ms, 3),
                "append_async_first_ops": first_ops,
                "append_async": summary(times),
                "append_async_ops": steady_ops,
            }

        row.update(asyncio.run(run_async()))
        store._owner_thread_id = None
        if fraction == 1.0:
            row["append_sync_profile"] = profile(
                f"append-sync-{args.session}",
                lambda: [
                    store.append_message(user_message(f"profile {i}"))
                    for i in range(20)
                ],
            )
    finally:
        store.close()
    return row


PROBES: dict[str, Callable[[argparse.Namespace], dict[str, Any]]] = {
    "open": probe_open,
    "append": probe_append,
}


# Probes that need a copied session's children or project; `fresh` has neither.
NEEDS_REAL_SESSION = {"children", "search"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", choices=list(PROBES))
    parser.add_argument("--session", choices=[*SOURCES, "fresh"], default="phoebe")
    parser.add_argument("--fractions", default="0.1,0.5,1.0")
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("--events", type=int, default=8)
    parser.add_argument("--keep", action="store_true", help="keep the sandbox copies")
    args = parser.parse_args()
    if args.session == "fresh" and args.probe in NEEDS_REAL_SESSION:
        parser.error(f"probe {args.probe!r} needs a copied session, not 'fresh'")
    if args.session == "fresh":
        args.fractions = "1.0"
    return args


def main() -> None:
    global BOX
    sys.modules.setdefault("store_probes", sys.modules[__name__])
    from store_probes_more import MORE_PROBES

    PROBES.update(MORE_PROBES)
    args = parse_args()
    result = {
        "probe": args.probe,
        "session": args.session,
        "zeta_file": zeta.__file__,
        "started_load": load(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with Sandbox("sandbox-store", keep=args.keep) as BOX:
        result.update(PROBES[args.probe](args))
    result["finished_load"] = load()
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{args.probe}-{args.session}.json"
    path.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
