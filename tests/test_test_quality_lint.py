from pathlib import Path

import pytest

from scripts import lint_test_quality as quality
from scripts.lint_test_quality import lint_diff


def _diff(path: str, source: str, added_lines: set[int] | None = None) -> str:
    lines = source.splitlines()
    selected = added_lines or set(range(1, len(lines) + 1))
    hunks = []
    for line_number in sorted(selected):
        hunks.extend(
            [
                f"@@ -{line_number},0 +{line_number},1 @@",
                f"+{lines[line_number - 1]}",
            ]
        )
    return "\n".join(
        [
            f"diff --git a/{path} b/{path}",
            f"--- a/{path}",
            f"+++ b/{path}",
            *hunks,
        ]
    )


def _lint(
    tmp_path: Path,
    source: str,
    added_lines: set[int] | None = None,
    diff: str | None = None,
    path: str = "tests/test_example.py",
):
    target = tmp_path / path
    target.parent.mkdir(parents=True)
    target.write_text(source)
    return lint_diff(diff or _diff(path, source, added_lines), tmp_path)


@pytest.mark.parametrize(
    "statement",
    [
        "time.sleep(0.01)",
        "await asyncio.sleep(1)",
        "assert time.perf_counter() - started < 5.0",
        "assert elapsed_seconds <= 5",
    ],
)
def test_time_rule_names_the_readiness_or_clock_fix(
    tmp_path: Path, statement: str
) -> None:
    source = f"async def test_wait():\n    {statement}\n"

    findings = _lint(tmp_path, source)

    assert [(finding.code, finding.line) for finding in findings] == [("TQ001", 2)]
    assert "wait on a readiness event or inject a clock" in findings[0].message
    assert "timeouts are hang guards only" in findings[0].message


def test_time_rule_allows_zero_yields_and_reasoned_product_latency(
    tmp_path: Path,
) -> None:
    source = """async def test_latency():
    await asyncio.sleep(0)
    # test-contract: latency benchmark measures the real retry delay
    await asyncio.sleep(0.1)
    # test-contract: latency benchmark measures the real retry budget
    assert elapsed <= 0.2
"""

    assert _lint(tmp_path, source) == []


def test_time_rule_rejects_bare_latency_marker(tmp_path: Path) -> None:
    source = """async def test_latency():
    # test-contract: latency
    await asyncio.sleep(0.1)
"""

    assert [finding.code for finding in _lint(tmp_path, source)] == ["TQ001"]


@pytest.mark.parametrize(
    ("timeout", "expected"),
    [(1, ["TQ001"]), (3, ["TQ001"]), (9, ["TQ001"]), (10, [])],
)
def test_time_rule_rejects_short_wait_for_hang_guards(
    tmp_path: Path, timeout: int, expected: list[str]
) -> None:
    source = f"async def test_wait():\n    await asyncio.wait_for(ready.wait(), timeout={timeout})\n"

    findings = _lint(tmp_path, source)

    assert [finding.code for finding in findings] == expected
    if findings:
        assert findings[0].message == (
            "use a generous hang guard (>= 10 s) and wait on the event; use an "
            "injected clock for timeout behavior"
        )


def test_time_rule_ignores_sleep_inside_child_process_script(tmp_path: Path) -> None:
    source = """def test_child():
    subprocess.run([sys.executable, "-c", "import time; time.sleep(30)"])
"""

    assert _lint(tmp_path, source) == []


@pytest.mark.parametrize(
    ("statement", "line"),
    [
        ("await ready.wait()", 2),
        ("await task", 2),
        ("await owner.task", 2),
        ("await asyncio.gather(first, second)", 2),
    ],
)
def test_await_deadline_rule_rejects_unbounded_waits(
    tmp_path: Path, statement: str, line: int
) -> None:
    source = f"async def test_wait():\n    {statement}\n"

    findings = _lint(tmp_path, source)

    assert [(finding.code, finding.line) for finding in findings] == [("TQ004", line)]
    assert findings[0].message == (
        "await without a deadline: use `await await_or_fail(x, what=...)` "
        "(tests/support/owned.py), or asyncio.wait_for(x, 10)"
    )


def test_await_deadline_rule_catches_pr_765_original_bare_task_await(
    tmp_path: Path,
) -> None:
    source = """async def test_wait_returns_terminal_state_after_kill_cancels_monitor():
    waiter = asyncio.create_task(registry.wait(task_id))
    status = await waiter
"""

    findings = _lint(tmp_path, source)

    assert [(finding.code, finding.line) for finding in findings] == [("TQ004", 3)]


def test_await_deadline_rule_catches_both_pr_772_notification_gates(
    tmp_path: Path,
) -> None:
    source = """async def force_parent_tool_end_before_approval_end(event):
    if event == "tool_end":
        await parent_tool_end_release.wait()
    elif event == "approval_end":
        await approval_end_release.wait()
"""

    findings = _lint(tmp_path, source)

    assert [(finding.code, finding.line) for finding in findings] == [
        ("TQ004", 3),
        ("TQ004", 5),
    ]


def test_deadline_rule_checks_the_whole_changed_function(tmp_path: Path) -> None:
    source = """async def test_wait():
    await ready.wait()
    assert state == "new"
"""

    findings = _lint(tmp_path, source, added_lines={3})

    assert [(finding.code, finding.line) for finding in findings] == [("TQ004", 2)]


def test_support_deadline_exemption_applies_only_to_bounded_await(
    tmp_path: Path,
) -> None:
    source = """async def helper():
    await dangerous.wait()
    await asyncio.wait_for(other.wait(), 10)
"""

    findings = _lint(tmp_path, source, path="tests/support/example.py")

    assert [(finding.code, finding.line) for finding in findings] == [("TQ004", 2)]


@pytest.mark.parametrize(
    ("wait_and_gather", "expected_lines"),
    [
        (
            """done, pending = await asyncio.wait([task], timeout=10)
    await asyncio.gather(*done, dangerous.wait())""",
            [3],
        ),
        (
            """done, pending = await asyncio.wait([task], timeout=None)
    await asyncio.gather(*done)""",
            [2, 3],
        ),
    ],
)
def test_support_gather_requires_reasoned_marker(
    tmp_path: Path, wait_and_gather: str, expected_lines: list[int]
) -> None:
    source = f"async def helper(task):\n    {wait_and_gather}\n"

    findings = _lint(tmp_path, source, path="tests/support/example.py")

    assert [(finding.code, finding.line) for finding in findings] == [
        ("TQ004", line) for line in expected_lines
    ]


@pytest.mark.parametrize("target", ["thread", "process"])
def test_await_deadline_rule_rejects_unbounded_thread_join(
    tmp_path: Path, target: str
) -> None:
    source = f"def test_worker():\n    {target}.join()\n"

    findings = _lint(tmp_path, source)

    assert [(finding.code, finding.line) for finding in findings] == [("TQ004", 2)]
    assert findings[0].message == "blocking wait without a deadline: pass timeout=10"


@pytest.mark.parametrize(
    "call",
    ["worker.join(10)", "worker.join(timeout=10)", '", ".join(parts)'],
)
def test_await_deadline_rule_accepts_bounded_or_string_join(
    tmp_path: Path, call: str
) -> None:
    source = f"def test_worker():\n    {call}\n"

    assert _lint(tmp_path, source) == []


@pytest.mark.parametrize(
    "statement",
    [
        "ready = event.wait()",
        "assert ready.wait()",
        "return ready.wait()",
        "consume(event.wait())",
        "status = process.wait()",
        "joined = worker.join()",
    ],
)
def test_await_deadline_rule_rejects_sync_wait_used_as_value(
    tmp_path: Path, statement: str
) -> None:
    source = f"def test_worker():\n    {statement}\n"

    findings = _lint(tmp_path, source)

    assert [(finding.code, finding.line) for finding in findings] == [("TQ004", 2)]
    assert findings[0].message == "blocking wait without a deadline: pass timeout=10"


@pytest.mark.parametrize("call", ["ready.wait(10)", "ready.wait(timeout=10)"])
def test_await_deadline_rule_accepts_bounded_sync_event_wait(
    tmp_path: Path, call: str
) -> None:
    source = f"def test_worker():\n    {call}\n"

    assert _lint(tmp_path, source) == []


def test_await_deadline_rule_catches_pr_774_blocking_pipe_receive(
    tmp_path: Path,
) -> None:
    source = """async def test_other_process():
    await asyncio.to_thread(parent_pipe.recv)
"""

    findings = _lint(tmp_path, source)

    assert [(finding.code, finding.line) for finding in findings] == [("TQ004", 2)]


@pytest.mark.parametrize(
    "target",
    [
        "socket.recv",
        "socket.recv_into",
        "stream.read",
        "stream.readline",
        "listener.accept",
        "queue.get",
    ],
)
def test_await_deadline_rule_rejects_blocking_to_thread(
    tmp_path: Path, target: str
) -> None:
    source = f"async def test_worker():\n    await asyncio.to_thread({target})\n"

    findings = _lint(tmp_path, source)

    assert [(finding.code, finding.line) for finding in findings] == [("TQ004", 2)]


@pytest.mark.parametrize(
    "statement",
    [
        "await asyncio.wait_for(asyncio.to_thread(pipe.recv), 10)",
        "await await_or_fail(asyncio.to_thread(stream.read), what='stream read')",
        "async with asyncio.timeout(10):\n        await asyncio.to_thread(listener.accept)",
    ],
)
def test_await_deadline_rule_accepts_bounded_blocking_to_thread(
    tmp_path: Path, statement: str
) -> None:
    source = f"async def test_worker():\n    {statement}\n"

    assert _lint(tmp_path, source) == []


def test_await_deadline_rule_accepts_supported_deadlines(tmp_path: Path) -> None:
    source = """async def test_waits():
    await asyncio.wait_for(ready.wait(), timeout=10)
    async with asyncio.timeout(10):
        await task
    await await_or_fail(task, what="worker")
"""

    assert _lint(tmp_path, source) == []


def test_await_deadline_rule_accepts_reasoned_marker(tmp_path: Path) -> None:
    source = """async def test_waits():
    # test-contract: seam no public event exposes worker readiness
    await worker._ready
"""

    assert _lint(tmp_path, source) == []


def test_short_wait_for_requires_latency_marker(tmp_path: Path) -> None:
    source = """async def test_waits():
    await asyncio.wait_for(task, timeout=1)
    # test-contract: latency verifies a real one-second product deadline
    await asyncio.wait_for(task, timeout=1)
"""

    findings = _lint(tmp_path, source)

    assert [(finding.code, finding.line) for finding in findings] == [("TQ001", 2)]


def test_digest_rule_rejects_real_action_policy_payload_hash(tmp_path: Path) -> None:
    source = """def test_existing_tool_payloads_match_pre_action_policy_snapshots():
    snapshots = {"schemas": existing_schemas, "codex": payload}
    expected = {"schemas": "839776b9", "codex": "2db72314"}
    assert {
        name: hashlib.sha256(
            json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
        for name, value in snapshots.items()
    } == expected
"""

    findings = _lint(tmp_path, source)

    assert [(finding.code, finding.line) for finding in findings] == [("TQ002", 4)]
    assert findings[0].message == "assert semantic fields, not a hash"


@pytest.mark.parametrize(
    ("digest", "expected_line"),
    [
        (
            """assert {
        "schema": hashlib.sha256(json.dumps(payload).encode()).hexdigest()
    } == expected""",
            4,
        ),
        (
            '''serialized = json.dumps(payload).encode()
    assert hashlib.sha256(serialized).hexdigest() == "839776b9"''',
            5,
        ),
    ],
)
def test_digest_rule_rejects_a_literal_full_payload_hash(
    tmp_path: Path, digest: str, expected_line: int
) -> None:
    source = f"""def test_payload():
    payload = {{"name": "tool"}}
    expected = {{"schema": "839776b9"}}
    {digest}
"""

    findings = _lint(tmp_path, source)

    assert [(finding.code, finding.line) for finding in findings] == [
        ("TQ002", expected_line)
    ]


@pytest.mark.parametrize(
    "statement",
    [
        'monkeypatch.setattr(app, "_close_runtime", close)',
        'monkeypatch.setattr("zeta.tui.app._run_tui", run)',
        'with patch("zeta.core.store._store.os._private"):\n        pass',
        "mock.assert_called_once_with(1)",
        "assert mock.call_count == 1",
        "assert mock.mock_calls == [call(1), call(2)]",
    ],
)
def test_internal_contract_rule_names_the_public_interface_fix(
    tmp_path: Path, statement: str
) -> None:
    source = f"def test_wiring():\n    {statement}\n"

    findings = _lint(tmp_path, source)

    assert [(finding.code, finding.line) for finding in findings] == [("TQ003", 2)]
    assert "seam <reason>" in findings[0].message


@pytest.mark.parametrize("contract", ["complexity", "durability", "seam"])
def test_internal_contract_rule_accepts_reasoned_markers(
    tmp_path: Path, contract: str
) -> None:
    source = f"""def test_contract():
    # test-contract: {contract} no public event exposes durable publication
    assert mock.call_count == 1
"""

    assert _lint(tmp_path, source) == []


def test_internal_contract_rule_accepts_real_private_fault_seam(tmp_path: Path) -> None:
    source = """def test_refresh_failure():
    # test-contract: seam no public hook injects a refresh disk failure
    monkeypatch.setattr(index_module, "_refresh_sources", fail_once)
"""

    assert _lint(tmp_path, source) == []


def test_internal_contract_rule_rejects_bare_seam_marker(tmp_path: Path) -> None:
    source = """def test_refresh_failure():
    # test-contract: seam
    monkeypatch.setattr(index_module, "_refresh_sources", fail_once)
"""

    findings = _lint(tmp_path, source)

    assert [finding.code for finding in findings] == ["TQ003"]
    assert "seam <reason>" in findings[0].message


def test_rule_checks_only_added_test_lines(tmp_path: Path) -> None:
    source = """def test_existing_and_new():
    time.sleep(0.1)
    assert result == "ok"
"""

    assert _lint(tmp_path, source, {3}) == []


def test_rule_ignores_pure_reindent(tmp_path: Path) -> None:
    source = """def test_existing():
    if enabled:
        time.sleep(1)
"""
    diff = """diff --git a/tests/test_example.py b/tests/test_example.py
--- a/tests/test_example.py
+++ b/tests/test_example.py
@@ -2,1 +3,1 @@
-    time.sleep(1)
+        time.sleep(1)
"""

    assert _lint(tmp_path, source, diff=diff) == []


def test_rule_ignores_pure_move(tmp_path: Path) -> None:
    source = """def test_existing():
    time.sleep(1)
"""
    diff = """diff --git a/tests/test_example.py b/tests/test_example.py
--- a/tests/test_example.py
+++ b/tests/test_example.py
@@ -1,1 +1,2 @@
-    time.sleep(1)
 def test_existing():
+    time.sleep(1)
"""

    assert _lint(tmp_path, source, diff=diff) == []


def test_move_pairing_keeps_a_genuinely_new_duplicate_violation(tmp_path: Path) -> None:
    source = """def test_existing():
    time.sleep(1)
    time.sleep(1)
"""
    diff = """diff --git a/tests/test_example.py b/tests/test_example.py
--- a/tests/test_example.py
+++ b/tests/test_example.py
@@ -1,1 +1,3 @@
-    time.sleep(1)
 def test_existing():
+    time.sleep(1)
+    time.sleep(1)
"""

    findings = _lint(tmp_path, source, diff=diff)

    assert [(finding.code, finding.line) for finding in findings] == [("TQ001", 3)]


def test_deleted_file_does_not_hide_an_added_violation(tmp_path: Path) -> None:
    source = """def test_new_wait():
    time.sleep(1)
"""
    diff = """diff --git a/tests/test_a.py b/tests/test_a.py
--- a/tests/test_a.py
+++ b/tests/test_a.py
@@ -1,1 +1,2 @@
 def test_new_wait():
+    time.sleep(1)
diff --git a/tests/test_b.py b/tests/test_b.py
--- a/tests/test_b.py
+++ /dev/null
@@ -1,2 +0,0 @@
-def test_old_wait():
-    time.sleep(1)
"""

    target = tmp_path / "tests/test_a.py"
    target.parent.mkdir()
    target.write_text(source)

    findings = lint_diff(diff, tmp_path)

    assert [(finding.code, finding.path, finding.line) for finding in findings] == [
        ("TQ001", Path("tests/test_a.py"), 2)
    ]


def test_renamed_file_pairs_old_and_new_lines(tmp_path: Path) -> None:
    source = """def test_existing():
    time.sleep(1)
"""
    diff = """diff --git a/tests/test_old.py b/tests/test_example.py
similarity index 88%
rename from tests/test_old.py
rename to tests/test_example.py
--- a/tests/test_old.py
+++ b/tests/test_example.py
@@ -1,2 +1,2 @@
 def test_existing():
-    time.sleep(1)
+        time.sleep(1)
"""

    assert _lint(tmp_path, source, diff=diff) == []


def test_all_mode_reads_function_ratchet(tmp_path: Path) -> None:
    source = """async def test_existing_wait():
    await task

async def test_new_wait():
    await other_task
"""
    target = tmp_path / "tests/test_example.py"
    target.parent.mkdir()
    target.write_text(source)
    ratchet = tmp_path / quality.RATCHET_PATH
    ratchet.write_text("tests/test_example.py::test_existing_wait\n")

    findings = quality.lint_all(tmp_path)

    assert [(finding.code, finding.line) for finding in findings] == [("TQ004", 5)]


def test_write_ratchet_command_records_each_function_once(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    source = """async def test_waits():
    await first
    await second
"""
    target = tmp_path / "tests/test_example.py"
    target.parent.mkdir()
    target.write_text(source)
    monkeypatch.chdir(tmp_path)

    assert quality.main(["--write-ratchet"]) == 0

    assert (tmp_path / quality.RATCHET_PATH).read_text() == (
        "tests/test_example.py::test_waits\n"
    )
    assert capsys.readouterr().out == (
        "wrote 1 keys from 2 findings to tests/test_quality_ratchet.txt\n"
    )


def test_support_helper_can_mark_its_bounded_drain(tmp_path: Path) -> None:
    path = "tests/support/waits.py"
    source = """async def drain(task):
    done, pending = await asyncio.wait([task], timeout=10)
    # test-contract: seam no public task-drain seam exposes only completed tasks
    await asyncio.gather(*done)
"""
    target = tmp_path / path
    target.parent.mkdir(parents=True)
    target.write_text(source)

    assert lint_diff(_diff(path, source), tmp_path) == []


def test_select_base_fetches_and_uses_pull_request_merge_base(monkeypatch) -> None:
    runs = []
    checks = []
    monkeypatch.setattr(
        "scripts.lint_test_quality.subprocess.run",
        lambda command, check: runs.append((command, check)),
    )
    monkeypatch.setattr(
        "scripts.lint_test_quality.subprocess.check_output",
        lambda command, text: checks.append((command, text)) or "merge-base\n",
    )

    assert quality.select_base("pull_request", "main") == "merge-base"
    assert runs == [(["git", "fetch", "--no-tags", "origin", "main"], True)]
    assert checks == [(["git", "merge-base", "HEAD", "FETCH_HEAD"], True)]


def test_select_base_uses_first_parent_for_push(monkeypatch) -> None:
    commands = []
    monkeypatch.setattr(
        "scripts.lint_test_quality.subprocess.check_output",
        lambda command, text: commands.append((command, text)) or "parent\n",
    )

    assert quality.select_base("push") == "parent"
    assert commands == [(["git", "rev-parse", "HEAD^"], True)]


def test_ci_runs_quality_lint_for_pull_requests_and_main_pushes() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text()
    step = workflow.split("- name: Reject new low-signal test patterns", 1)[1]
    step = step.split("\n  browser:", 1)[0]

    assert "if: github.event_name == 'pull_request'" not in step
    assert "GITHUB_EVENT_NAME: ${{ github.event_name }}" in step
    assert '--base "$GITHUB_EVENT_NAME" "$GITHUB_BASE_REF"' in step


def test_non_test_python_files_are_ignored(tmp_path: Path) -> None:
    path = "src/zeta/wait.py"
    target = tmp_path / path
    target.parent.mkdir(parents=True)
    source = "time.sleep(0.1)\n"
    target.write_text(source)

    assert lint_diff(_diff(path, source), tmp_path) == []
