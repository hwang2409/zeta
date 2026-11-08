from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from tests.support.owner_counts import OwnerCounts, baseline_growth, merge_rows


class _Owner:
    pass


def _row(action):
    recorder = OwnerCounts()
    recorder.node_start("self")
    action(recorder)
    recorder.node_finish("self")
    return recorder.rows()[0]


def test_explicit_close_has_exact_row() -> None:
    def action(recorder):
        owner = _Owner()
        recorder.constructed(owner, "Explicit")
        recorder.closed(owner)

    assert _row(action) == ("self", "Explicit", 0)


def test_intentionally_open_has_exact_row() -> None:
    assert _row(lambda recorder: recorder.constructed(_Owner(), "Open")) == (
        "self",
        "Open",
        1,
    )


def test_thread_open_close_has_exact_row() -> None:
    def action(recorder):
        def worker():
            owner = _Owner()
            recorder.constructed(owner, "Thread")
            recorder.closed(owner)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(10)
        assert not thread.is_alive()

    assert _row(action) == ("self", "Thread", 0)


def test_async_close_has_exact_row() -> None:
    async def run(recorder):
        owner = _Owner()
        recorder.constructed(owner, "Async")
        recorder.closed(owner)

    assert _row(lambda recorder: asyncio.run(run(recorder))) == (
        "self",
        "Async",
        0,
    )


def test_worker_rows_merge_independent_of_worker_order() -> None:
    first = [("a", "AgentLoop", 1)]
    second = [("b", "ConversationStore", 0)]

    assert merge_rows([first, second]) == merge_rows([second, first])
    with pytest.raises(ValueError, match="duplicate owner-count row: a::AgentLoop"):
        merge_rows([first, first])


def test_baseline_may_only_shrink() -> None:
    base = "a.py::test_a::AgentLoop=2\nb.py::test_b::ConversationStore=1\n"

    assert baseline_growth(base, "a.py::test_a::AgentLoop=1\n") == []
    assert baseline_growth(base, base + "c.py::test_c::ZetaServer=1\n") == [
        (
            "owner leak baseline: c.py::test_c::ZetaServer=1 (base has no row); "
            "do not add or raise baseline rows: close the owner in the test "
            "(await server.close())"
        )
    ]
    assert baseline_growth(base, "a.py::test_a::AgentLoop=3\n") == [
        (
            "owner leak baseline: a.py::test_a::AgentLoop=3 (base 2); "
            "do not add or raise baseline rows: close the owner in the test "
            "(await loop.close())"
        )
    ]


def test_baseline_guard_rejects_a_bad_ref_and_passes_a_deleted_baseline(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "scripts").mkdir()
    shutil.copy(root / "scripts/check_owner_leak_baseline.py", tmp_path / "scripts")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/support").symlink_to(root / "tests/support")
    baseline = tmp_path / "tests/owner_leak_ratchet.txt"
    baseline.write_text("tests/test_a.py::test_a::AgentLoop=1\n")

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            args, cwd=tmp_path, capture_output=True, text=True, check=False, timeout=60
        )

    git = ("git", "-c", "user.name=t", "-c", "user.email=t@t")
    assert run(*git, "init", "-q").returncode == 0
    assert run(*git, "add", "scripts", "tests/owner_leak_ratchet.txt").returncode == 0
    assert run(*git, "commit", "-qm", "base").returncode == 0
    guard = (sys.executable, "scripts/check_owner_leak_baseline.py")

    bad_ref = run(*guard, "no-such-ref")
    assert bad_ref.returncode == 2, bad_ref.stdout + bad_ref.stderr
    baseline.unlink()
    deleted = run(*guard, "HEAD")
    assert deleted.returncode == 0, deleted.stdout + deleted.stderr


def _run_store_test(
    pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
    *,
    baseline_count: int,
    body: str = "",
) -> pytest.RunResult:
    # A test with no baseline row is allowed 0, so write no row for 0.
    row = (
        f"test_owner.py::test_owner::ConversationStore={baseline_count}\\n"
        if baseline_count
        else ""
    )
    pytester.makeconftest(
        f'''
from pathlib import Path
import pytest
from tests.support.owner_counts import OwnerCounts, gate_failures

recorder = OwnerCounts()
baseline = Path("baseline.txt")
baseline.write_text("{row}")


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
    pytester.makepyfile(
        test_owner=f'''
from tests.support.owner_counts import dropped_for_gc
from zeta.core.store import ConversationStore


def test_owner(tmp_path):
    store = ConversationStore(tmp_path / "sessions")
    {body}
'''
    )
    root = str(Path(__file__).resolve().parents[1])
    old_pythonpath = os.environ.get("PYTHONPATH")
    monkeypatch.setenv(
        "PYTHONPATH",
        root if not old_pythonpath else os.pathsep.join((root, old_pythonpath)),
    )
    return pytester.runpytest_subprocess("-q")


def test_new_conversation_store_leak_fails_with_named_fix(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _run_store_test(pytester, monkeypatch, baseline_count=0)

    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.stdout.fnmatch_lines(
        [
            (
                "*owner leak ratchet: test_owner.py::test_owner left 1 "
                "ConversationStore owner(s) unclosed (baseline 0); "
                "close the owner with store.close(); "
                "do not add rows to tests/owner_leak_ratchet.txt*"
            )
        ]
    )


def test_owner_dropped_for_gc_is_not_counted_but_a_second_leak_is(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _run_store_test(
        pytester,
        monkeypatch,
        baseline_count=0,
        body=(
            'dropped_for_gc(store, "finalizer test")\n'
            '    ConversationStore(tmp_path / "other")'
        ),
    )

    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.stdout.fnmatch_lines(
        [
            (
                "*owner leak ratchet: test_owner.py::test_owner left 1 "
                "ConversationStore owner(s) unclosed (baseline 0)*"
                "dropped_for_gc(owner, reason)*"
            )
        ]
    )


def test_closed_conversation_store_passes_gate(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _run_store_test(pytester, monkeypatch, baseline_count=0, body="store.close()")

    assert result.ret == pytest.ExitCode.OK
    result.assert_outcomes(passed=1)


def test_lower_count_passes_and_refresh_lowers_baseline(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _run_store_test(pytester, monkeypatch, baseline_count=2)
    assert result.ret == pytest.ExitCode.OK

    baseline = pytester.path / "baseline.txt"
    baseline.write_text(
        "test_owner.py::test_owner::ConversationStore=2\n"
        "test_other.py::test_other::AgentLoop=1\n"
    )
    observed = pytester.path / "observed.txt"
    observed.write_text(
        "test_owner.py::test_owner::ConversationStore=1\n"
        "test_other.py::test_other::AgentLoop=3\n"
        "test_new.py::test_new::ToolRegistry=4\n"
    )
    script = Path(__file__).resolve().parents[1] / "scripts/refresh_owner_leak_baseline.py"
    subprocess.run(
        [sys.executable, str(script), str(observed), "--baseline", str(baseline)],
        check=True,
        timeout=60,
    )
    assert baseline.read_text() == (
        "test_other.py::test_other::AgentLoop=1\n"
        "test_owner.py::test_owner::ConversationStore=1\n"
    )


def test_repository_conftest_counts_owners_the_safety_net_closes_in_a_task(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the real conftest with xdist: the safety net closes ZetaServer through
    await_or_fail (a new task), and that close must not count as a test close."""
    root = Path(__file__).resolve().parents[1]
    shutil.copy(root / "pyproject.toml", pytester.path / "pyproject.toml")
    (pytester.path / "src").symlink_to(root / "src")
    tests = pytester.path / "tests"
    tests.mkdir()
    for name in ("__init__.py", "conftest.py", "sitecustomize.py"):
        shutil.copy(root / "tests" / name, tests / name)
    (tests / "support").symlink_to(root / "tests" / "support")
    (tests / "owner_leak_ratchet.txt").write_text("")
    (tests / "test_server_owner.py").write_text(
        """
from tests.support.owned import await_or_fail
from zeta.server import ZetaServer


async def test_leaked(tmp_path):
    ZetaServer(home=tmp_path, port=0, provider="codex")


async def test_closed(tmp_path):
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    await await_or_fail(server.close(), what="ZetaServer.close")
"""
    )
    observed = pytester.path / "observed.txt"

    monkeypatch.setenv("ZETA_OWNER_COUNTS", str(observed))

    result = pytester.runpytest_subprocess(
        "-q", "-n", "2", "-p", "no:cacheprovider", "tests/test_server_owner.py"
    )

    assert observed.read_text() == (
        "tests/test_server_owner.py::test_closed::ZetaServer=0\n"
        "tests/test_server_owner.py::test_leaked::ZetaServer=1\n"
    )
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.stdout.fnmatch_lines(
        [
            (
                "*owner leak ratchet: tests/test_server_owner.py::test_leaked left 1 "
                "ZetaServer owner(s) unclosed (baseline 0); close the owner with "
                "await server.close(); do not add rows to tests/owner_leak_ratchet.txt*"
            )
        ]
    )
    result.assert_outcomes(passed=2)
