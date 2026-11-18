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
    entered = asyncio.Event()
    drained = asyncio.Event()

    async def dispatch(*_):
        entered.set()
        try:
            await await_or_fail(asyncio.Event().wait(), what="pending nested tool must be canceled")
        finally:
            drained.set()

    task = asyncio.create_task(run_guest('await tools.read({});', ["read"], dispatch))
    try:
        await await_or_fail(entered.wait(), what="guest dispatches call")
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await await_or_fail(task, what="canceled guest is joined")
    assert drained.is_set()
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
