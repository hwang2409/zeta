from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path

import pytest

from tests.support.owner_counts import (
    BASELINE_END,
    BASELINE_START,
    OwnerCounts,
    refresh_baseline,
)


def _row(action):
    recorder = OwnerCounts()
    recorder.node_start("self")
    action(recorder)
    recorder.node_finish("self")
    return recorder.rows()[0]


def test_explicit_close_has_exact_row() -> None:
    def action(recorder):
        owner = object()
        recorder.constructed(owner, "Explicit")
        recorder.closed(owner)

    assert _row(action) == ("self", "Explicit", 0)


def test_intentionally_open_has_exact_row() -> None:
    assert _row(lambda recorder: recorder.constructed(object(), "Open")) == (
        "self",
        "Open",
        1,
    )


def test_thread_open_close_has_exact_row() -> None:
    def action(recorder):
        def worker():
            owner = object()
            recorder.constructed(owner, "Thread")
            recorder.closed(owner)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(10)
        assert not thread.is_alive()

    assert _row(action) == ("self", "Thread", 0)


def test_async_close_has_exact_row() -> None:
    async def run(recorder):
        owner = object()
        recorder.constructed(owner, "Async")
        recorder.closed(owner)

    assert _row(lambda recorder: asyncio.run(run(recorder))) == (
        "self",
        "Async",
        0,
    )


def _run_store_test(
    pytester: pytest.Pytester, *, baseline_count: int, close: bool
) -> pytest.RunResult:
    pytester.makeconftest(
        f'''
from pathlib import Path
import pytest
from tests.support.owner_counts import OwnerCounts, gate_failures

recorder = OwnerCounts()
baseline = Path("baseline.txt")
baseline.write_text("test_owner.py::test_owner::ConversationStore={baseline_count}\\n")


def pytest_sessionstart(session):
    recorder.install()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    del nextitem
    recorder.node_start(item.nodeid)
    yield
    recorder.node_finish(item.nodeid)


def pytest_sessionfinish(session, exitstatus):
    del exitstatus
    recorder.uninstall()
    failures = gate_failures(recorder.rows(), baseline)
    for failure in failures:
        print(failure, flush=True)
    if failures:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
'''
    )
    close_line = "store.close()" if close else ""
    pytester.makepyfile(
        test_owner=f'''
from zeta.core.store import ConversationStore


def test_owner(tmp_path):
    store = ConversationStore(tmp_path / "sessions")
    {close_line}
'''
    )
    root = str(Path(__file__).resolve().parents[1])
    old_pythonpath = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = (
        root if not old_pythonpath else os.pathsep.join((root, old_pythonpath))
    )
    try:
        return pytester.runpytest_subprocess("-q")
    finally:
        if old_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = old_pythonpath


def test_new_conversation_store_leak_fails_with_named_fix(
    pytester: pytest.Pytester,
) -> None:
    result = _run_store_test(pytester, baseline_count=0, close=False)

    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.stdout.fnmatch_lines(
        [
            (
                "*owner leak ratchet: test_owner.py::test_owner left 1 "
                "ConversationStore owner(s) unclosed (baseline 0); "
                "close the owner with store.close() or use the conftest owner helper*"
            )
        ]
    )


def test_closed_conversation_store_passes_gate(pytester: pytest.Pytester) -> None:
    result = _run_store_test(pytester, baseline_count=0, close=True)

    assert result.ret == pytest.ExitCode.OK
    result.assert_outcomes(passed=1)


def test_lower_count_passes_and_refresh_lowers_baseline(
    pytester: pytest.Pytester,
) -> None:
    result = _run_store_test(pytester, baseline_count=2, close=False)
    assert result.ret == pytest.ExitCode.OK

    baseline = pytester.path / "baseline.txt"
    observed = pytester.path / "observed.log"
    observed.write_text(
        f"{BASELINE_START}\n"
        "test_owner.py::test_owner::ConversationStore=1\n"
        f"{BASELINE_END}\n"
    )
    assert refresh_baseline(baseline, observed) == 1
    assert baseline.read_text() == (
        "test_owner.py::test_owner::ConversationStore=1\n"
    )
