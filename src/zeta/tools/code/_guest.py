"""Run the pinned Javy guest. No default WASI or process file descriptors."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import struct
import threading
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import Future
from functools import lru_cache
from importlib import import_module, metadata, resources
from typing import Any

VERSION = "50.0.1"
SHA256 = "35a481386b3bc7e6725b00e06dddeda4ea4e9e5c86330cd1f666a1d0bc0fde8b"
WASI = "wasi_snapshot_preview1"
IMPORTS = {
    "random_get": "ii", "environ_get": "ii", "environ_sizes_get": "ii",
    "clock_time_get": "iIi", "fd_close": "i", "fd_fdstat_get": "ii",
    "fd_read": "iiii", "fd_seek": "iIii", "fd_write": "iiii", "proc_exit": "i",
}
FRAME = 64 * 1024
OUTPUT_INPUT_LIMIT = 1024 * 1024
MEMORY = 32 * 1024**2
FUEL = 60 * 10**9
MAX_CALLS = 50
GuestCall = tuple[str, dict[str, Any], int]
Dispatch = Callable[[str, dict[str, Any], int], Awaitable[dict[str, object]]]


class GuestError(ValueError):
    pass


@lru_cache(maxsize=1)
def load_guest() -> tuple[Any, bytes, str]:
    """Read static assets once without compiling on the owning event loop."""
    try:
        wasm = import_module("wasmtime")
        version = metadata.version("wasmtime")
    except ImportError as exc:
        raise GuestError("code mode needs the optional runtime: uv pip install 'zeta[code]'") from exc
    if version != VERSION:
        raise GuestError(f"code mode requires wasmtime=={VERSION}")
    files = resources.files(__package__)
    artifact = files.joinpath("javy-plugin-v9.1.0.wasm").read_bytes()
    if hashlib.sha256(artifact).hexdigest() != SHA256:
        raise GuestError("Javy artifact hash mismatch")
    return wasm, artifact, files.joinpath("bridge.js").read_text(encoding="utf-8")


def _check_imports(module: Any) -> None:
    actual = [
        (imp.module, imp.name, tuple(map(str, imp.type.params)), tuple(map(str, imp.type.results)))
        for imp in module.imports
    ]
    expected = [
        (WASI, name, tuple("i32" if c == "i" else "i64" for c in params),
         () if name == "proc_exit" else ("i32",))
        for name, params in IMPORTS.items()
    ]
    if actual != expected:
        raise GuestError("Javy import ABI mismatch")


def _json_loads(data: str | bytes) -> Any:
    value = json.loads(data)
    remaining = 32768
    stack = [(0, value)]
    while stack:
        depth, item = stack.pop()
        remaining -= 1
        if depth > 64 or remaining < 0:
            raise GuestError("JSON exceeds depth or node limit")
        if isinstance(item, float) and not math.isfinite(item):
            raise GuestError("JSON numbers must be finite")
        if isinstance(item, (dict, list)):
            children = item.values() if isinstance(item, dict) else item
            stack.extend((depth + 1, child) for child in children)
    return value


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), allow_nan=False, ensure_ascii=False).encode(errors="backslashreplace")


class _Bridge:
    def __init__(self, request: Callable[[list[GuestCall]], bytes]) -> None:
        self.request = request
        self.calls = 0
        self.incoming = memoryview(b'{"event-loop":true,"simd-json-builtins":false}')
        self.frames = bytearray()
        self.output = bytearray()
        self.emitted = False
        self.running = False
        self.error: Exception | None = None

    def frame(self, payload: bytes) -> None:
        message = _json_loads(payload)
        if not isinstance(message, dict):
            raise GuestError("invalid bridge message")
        if message.get("op") == "output" and set(message) == {"op", "text", "end"}:
            text, end = message["text"], message["end"]
            if self.emitted or not isinstance(text, str) or type(end) is not bool:
                raise GuestError("invalid output message")
            self.output.extend(text.encode())
            if len(self.output) > OUTPUT_INPUT_LIMIT:
                raise GuestError("output exceeds 1 MiB")
            self.emitted = end
            return
        calls = message.get("calls")
        if (message.get("op") != "call" or set(message) != {"op", "calls"}
                or not isinstance(calls, list) or not calls or self.incoming):
            raise GuestError("invalid call frame")
        if self.calls + len(calls) > MAX_CALLS:
            raise GuestError("code mode allows at most 50 tool calls")
        batch: list[GuestCall] = []
        for index, call in enumerate(calls, self.calls + 1):
            if (not isinstance(call, dict) or set(call) != {"tool", "args"}
                    or not isinstance(call["tool"], str) or len(call["tool"]) > 128
                    or not isinstance(call["args"], dict)):
                raise GuestError("invalid nested call")
            batch.append((call["tool"], call["args"], index))
        self.calls += len(batch)
        data = self.request(batch)
        self.incoming = memoryview(struct.pack("<I", len(data)) + data)

    def io(self, caller: Any, fd: int, table: int, num: int, out: int, *, reading: bool) -> int:
        if fd != (0 if reading else 1) or (not reading and not self.running):
            return 8  # EBADF
        if self.error is not None:
            return 29  # EIO
        mem = caller.get("memory")
        length = mem.data_len(caller)

        def span(ptr: int, size: int) -> None:
            if not 0 <= ptr <= ptr + size <= length:
                raise GuestError("bad guest memory range")

        try:
            if not 0 <= num <= 8:
                raise GuestError("too many iovecs")
            span(table, num * 8)
            span(out, 4)
            count = 0
            for ptr, size in struct.iter_unpack("<II", mem.read(caller, table, table + num * 8)):
                span(ptr, size)
                if reading:
                    size = min(size, len(self.incoming))
                    if size:
                        mem.write(caller, self.incoming[:size], ptr)
                        self.incoming = self.incoming[size:]
                else:
                    if size > FRAME - len(self.frames):
                        raise GuestError("request frame too large")
                    self.frames.extend(mem.read(caller, ptr, ptr + size))
                count += size
            if not reading and b"\n" in self.frames:
                if self.frames.count(b"\n") != 1 or self.frames[-1] != 10:
                    raise GuestError("invalid frame boundary")
                self.frame(self.frames[:-1])
                self.frames.clear()
            mem.write(caller, struct.pack("<I", count), out)
        except Exception as exc:  # noqa: BLE001 - never leak Python exceptions through Wasmtime's global trampoline
            self.error = exc
            return 29
        return 0


def _invoke(wasm: Any, engine: Any, module: Any, source: bytes, bridge: _Bridge, stop: threading.Event) -> bytes:
    with wasm.Store(engine) as store, wasm.Linker(engine) as linker:
        store.set_epoch_deadline(1)
        if stop.is_set():
            raise GuestError("code mode stopped")
        store.set_fuel(FUEL)
        store.set_limits(memory_size=MEMORY, memories=1, instances=2)
        # A native Wasm trap avoids wasmtime-py's process-global callback exception slot.
        with wasm.Module(engine, '(module (func (export "exit") (param i32) unreachable))') as trap:
            exit_function = wasm.Instance(store, trap, []).exports(store)["exit"]
            linker.define(store, WASI, "proc_exit", exit_function)
        for imp in module.imports:
            if imp.name == "proc_exit":
                continue
            if imp.name in ("fd_read", "fd_write"):
                def callback(caller, fd, table, num, out, reading=imp.name == "fd_read"):
                    return bridge.io(caller, fd, table, num, out, reading=reading)
                linker.define_func(WASI, imp.name, imp.type, callback, access_caller=True)
            else:
                linker.define_func(WASI, imp.name, imp.type, lambda *_: 76)  # ENOTCAPABLE
        api = linker.instantiate(store, module).exports(store)
        mem = api["memory"]
        api["initialize-runtime"](store)
        if bridge.incoming:
            raise GuestError("Javy did not read its configuration")
        ptr = api["cabi_realloc"](store, 0, 0, 1, len(source))
        mem.write(store, source, ptr)
        ret = api["compile-src"](store, ptr, len(source))
        tag, ptr, length = struct.unpack("<III", mem.read(store, ret, ret + 12))
        if tag:
            raise GuestError(mem.read(store, ptr, ptr + min(length, 4096)).decode(errors="replace"))
        bridge.running = True
        try:
            api["invoke"](store, ptr, length, 0, 0, 0)
        finally:
            if bridge.error is not None:
                raise GuestError(str(bridge.error)) from bridge.error
        if bridge.frames or bridge.incoming or not bridge.emitted:
            raise GuestError(
                "script must finish with one output(value) and no pending frame.\n"
                "Use direct calls for one tool or when the shell already filters results.\n"
                "No filesystem, network, clocks, timers or persistent state in the sandbox.\n"
                "Nested approvals still apply; catch ToolDenied or ToolError.\n"
                "Use Promise.all for independent calls; up to 8 run at once."
            )
        _json_loads(bridge.output)
        return bytes(bridge.output)


async def run_guest(
    script: str, names: Sequence[str], dispatch: Dispatch, *, wall: float = 600, active: float = 60,
) -> bytes:
    """Own the invocation, including worker join and draining the current call batch."""
    source = script.encode()
    if len(source) > FRAME:
        raise GuestError("script exceeds 64 KiB")
    loop = asyncio.get_running_loop()
    done = loop.create_future()
    stop = threading.Event()
    engine = None
    engine_lock = threading.Lock()
    task = None
    pending = None
    timer = None
    started = loop.time()
    active_timeout = f"code mode exceeded {active:g} seconds of active JavaScript"

    def interrupt(message: str) -> None:
        if stop.is_set():
            return
        stop.set()
        with engine_lock:
            if engine is not None:
                engine.increment_epoch()
        if pending is not None:
            pending.cancel()
        if task is not None and not task.done():
            task.cancel()
        if not done.done():
            timeout_reason[0] = message

    timeout_reason = [""]

    def resume_timer() -> None:
        nonlocal started, timer
        started = loop.time()
        timer = loop.call_later(active, interrupt, active_timeout)

    async def run_batch(batch: list[GuestCall]) -> bytes:
        tasks = [loop.create_task(dispatch(*call)) for call in batch]
        replies = asyncio.gather(*tasks)
        try:
            results = await asyncio.shield(replies)
        except BaseException:
            for child in tasks:
                child.cancel()
            await asyncio.gather(replies, *tasks, return_exceptions=True)
            raise

        encoded = []
        for (name, _, index), result in zip(batch, results, strict=True):
            fallback = _json_bytes({"ok": False, "error": {"name": "ToolError", "tool": name,
                                   "callId": index, "content": "inner result exceeds bridge limits",
                                   "structuredContent": None, "canceled": False}})
            try:
                data = _json_bytes(result)
            except (ValueError, TypeError, RecursionError):
                data = fallback
            encoded.append((data, fallback))
        # Keep small siblings when replies would exceed the shared frame budget.
        limit = FRAME
        if 1 + sum(len(data) + 1 for data, _ in encoded) > FRAME:
            limit = (FRAME - 1) // len(batch) - 1
        return b"[" + b",".join(data if len(data) <= limit else fallback for data, fallback in encoded) + b"]"

    def start_batch(batch: list[GuestCall], reply: Future[bytes]) -> None:
        nonlocal task, pending, active
        if stop.is_set():
            reply.cancel()
            return
        pending = reply
        if timer is not None:
            timer.cancel()
        active -= loop.time() - started
        if active <= 0:
            interrupt(active_timeout)
            return
        task = loop.create_task(run_batch(batch))

        def transfer(completed: asyncio.Task[bytes]) -> None:
            if completed.cancelled():
                reply.cancel()
                return
            error = completed.exception()
            if not reply.done():
                resume_timer()
                if error is None:
                    reply.set_result(completed.result())
                else:
                    reply.set_exception(error)

        task.add_done_callback(transfer)

    def request(batch: list[GuestCall]) -> bytes:
        if stop.is_set():
            raise GuestError("code mode stopped")
        reply: Future[bytes] = Future()
        loop.call_soon_threadsafe(start_batch, batch, reply)
        return reply.result()

    def worker() -> None:
        nonlocal engine
        try:
            wasm, artifact, bootstrap = load_guest()
            config = wasm.Config()
            config.epoch_interruption = True
            config.consume_fuel = True
            with wasm.Engine(config) as owned_engine, wasm.Module(owned_engine, artifact) as module:
                _check_imports(module)
                with engine_lock:
                    engine = owned_engine
                try:
                    if stop.is_set():
                        raise GuestError("code mode stopped")
                    prefix = f"((()=>{{}}).constructor)({json.dumps('return (' + bootstrap + ')')})()({json.dumps(names)});\n"
                    value = _invoke(wasm, owned_engine, module, prefix.encode() + source, _Bridge(request), stop)
                finally:
                    with engine_lock:
                        engine = None
        except Exception as exc:  # noqa: BLE001 - hand errors to the loop after all native state has closed
            loop.call_soon_threadsafe(done.set_result, (None, exc))
        else:
            loop.call_soon_threadsafe(done.set_result, (value, None))

    thread = threading.Thread(target=worker, name="zeta-code-guest")
    resume_timer()
    thread.start()
    try:
        async with asyncio.timeout(wall):
            value, error = await asyncio.shield(done)
        if timeout_reason[0]:
            raise GuestError(timeout_reason[0]) from error
        if error is not None:
            raise GuestError(str(error)) from error
        return value
    except (asyncio.CancelledError, TimeoutError):
        interrupt("code mode wall deadline exceeded")
        raise
    finally:
        if timer is not None:
            timer.cancel()
        cleanup = asyncio.gather(done, *([task] if task is not None else []), return_exceptions=True)
        canceled = False
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                canceled = True
                interrupt("code mode stopped")
        thread.join()  # done is posted after native resources close, just before exit.
        if canceled:
            raise asyncio.CancelledError
