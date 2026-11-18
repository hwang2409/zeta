from __future__ import annotations

import asyncio
import json
import threading

import pytest

from tests.support.owned import await_or_fail
from zeta.tools.code._guest import GuestError, run_guest


async def test_guest_runs_dependent_calls_without_ambient_io():
    entered = []

    async def dispatch(name, args, index):
        entered.append((name, args, index))
        return {"ok": True, "value": args["x"]}

    result = await await_or_fail(run_guest('''
        const Javy = 'local';
        const a = await tools.read({x: 1});
        const b = await tools.read({x: a + 1});
        output([b, typeof globalThis.Javy, typeof Date, typeof fetch, typeof Math.random]);
        ''', ["read"], dispatch), what="Javy dependent calls")
    assert json.loads(result) == [2, "undefined", "undefined", "undefined", "undefined"]
    assert entered == [("read", {"x": 1}, 1), ("read", {"x": 2}, 2)]


async def test_cancel_drains_host_and_guest():
    both_entered = asyncio.Event()
    blocked = asyncio.Event()
    entered = []
    canceled = []

    async def dispatch(name, args, index):
        entered.append(index)
        if len(entered) == 2:
            both_entered.set()
        try:
            await await_or_fail(blocked.wait(), what="pending nested tool must be canceled")
        except asyncio.CancelledError:
            canceled.append(index)
            raise

    task = asyncio.create_task(run_guest('await Promise.all([tools.read({}), tools.read({})]);', ["read"], dispatch))
    try:
        await await_or_fail(both_entered.wait(), what="guest dispatches both calls")
    finally:
        task.cancel()
        try:
            with pytest.raises(asyncio.CancelledError):
                await await_or_fail(task, timeout=20, what="canceled guest is joined")
        finally:
            blocked.set()
    assert sorted(canceled) == [1, 2]
    assert not any(t.name == "zeta-code-guest" for t in threading.enumerate())


async def test_wall_deadline_joins_guest():
    with pytest.raises(TimeoutError):
        await await_or_fail(run_guest("while(true){}", [], None, wall=0.1), what="guest hard wall")
    assert not any(t.name == "zeta-code-guest" for t in threading.enumerate())


async def test_guest_memory_and_missing_output():
    result = await await_or_fail(run_guest('''
        let denied = false;
        try { new ArrayBuffer(48 * 1024 * 1024); } catch { denied = true; }
        output(denied);
        ''', [], None), what="guest memory cap")
    assert result == b"true"
    with pytest.raises(GuestError, match=r"finish with one output"):
        await await_or_fail(run_guest("42;", [], None), what="guest requires explicit output")


async def test_missing_extra_does_not_prevent_loading_tool(monkeypatch):
    from zeta.tools.code import _guest

    def missing(name):
        assert name == "wasmtime"
        raise ModuleNotFoundError(name)

    _guest.load_guest.cache_clear()
    monkeypatch.setattr(_guest, "import_module", missing)
    with pytest.raises(GuestError, match=r"uv pip install 'zeta\[code\]'"):
        await await_or_fail(run_guest("output(1);", [], None), what="missing optional runtime")
