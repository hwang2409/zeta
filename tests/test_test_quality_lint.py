import difflib
import subprocess
from pathlib import Path

import pytest
import yaml

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


def _replacement_diff(path: str, before: str, after: str, *, context: int = 3) -> str:
    body = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        n=context,
    )
    return f"diff --git a/{path} b/{path}\n" + "".join(body)


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


def test_transport_marker_does_not_excuse_a_missing_deadline(tmp_path: Path) -> None:
    source = "async def test_wait():\n    await ready.wait()  # test-contract: transport x\n"

    findings = _lint(tmp_path, source)

    assert [(finding.code, finding.line) for finding in findings] == [("TQ004", 2)]


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


@pytest.mark.parametrize("path", ["tests/test_example.py", "tests/support/helper.py"])
def test_await_deadline_rule_rejects_sync_wait_in_shared_helper(
    tmp_path: Path, path: str
) -> None:
    source = "def helper():\n    ready.wait()\n"

    findings = _lint(tmp_path, source, path=path)

    assert [(finding.code, finding.line) for finding in findings] == [("TQ004", 2)]


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


def test_await_deadline_rule_rejects_shielded_task_wait(tmp_path: Path) -> None:
    source = "async def test_worker():\n    await asyncio.shield(task)\n"

    findings = _lint(tmp_path, source)

    assert [(finding.code, finding.line) for finding in findings] == [("TQ004", 2)]


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
    "target",
    ["lambda: pipe.recv()", "functools.partial(stream.read)"],
)
def test_await_deadline_rule_rejects_wrapped_blocking_to_thread(
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


def test_tq005_rejects_production_bootstraps_in_conftest(tmp_path: Path) -> None:
    source = """import os

os.environ[\"ZETA_HOME\"] = \"/tmp/home\"
os.environ[\"ZETA_TEST_SCRIPTED_PROVIDER\"] = \"1\"
"""

    findings = _lint(tmp_path, source, path="tests/conftest.py")

    assert [(finding.code, finding.line) for finding in findings] == [
        ("TQ005", 3),
        ("TQ005", 4),
    ]


def test_tq005_reasoned_bootstrap_marker_suppresses_finding(tmp_path: Path) -> None:
    source = """import os
# test-contract: bootstrap session-wide isolated home is required by the harness
os.environ["ZETA_HOME"] = "/tmp/home"
"""

    assert _lint(tmp_path, source, path="tests/conftest.py") == []


def test_tq005_bare_or_wrong_marker_does_not_suppress(tmp_path: Path) -> None:
    for index, marker in enumerate(
        ("# test-contract: bootstrap", "# test-contract: latency unrelated")
    ):
        source = f"""import os
{marker}
os.environ["ZETA_HOME"] = "/tmp/home"
"""
        case = tmp_path / f"case{index}"
        case.mkdir()
        findings = _lint(case, source, path="tests/conftest.py")
        assert [finding.code for finding in findings] == ["TQ005"]


def test_marker_inline_comment_does_not_suppress_next_statement(tmp_path: Path) -> None:
    source = """async def test_wait():
    await asyncio.sleep(0.1)  # test-contract: latency real retry delay
    await asyncio.sleep(0.2)
"""

    findings = _lint(tmp_path, source)

    assert [(finding.code, finding.line) for finding in findings] == [("TQ001", 3)]


def test_marker_comment_line_applies_only_to_next_statement(tmp_path: Path) -> None:
    source = """async def test_wait():
    # test-contract: latency real retry delay
    await asyncio.sleep(0.1)
    await asyncio.sleep(0.2)
"""

    findings = _lint(tmp_path, source)

    assert [(finding.code, finding.line) for finding in findings] == [("TQ001", 4)]


def test_marker_on_multiline_statement_closing_line_applies_to_that_line() -> None:
    lines = [
        "call(",
        "    value,",
        "    option,",
        ")  # test-contract: seam real public seam is unavailable",
    ]

    assert quality._has_marker(lines, 4, {"seam"})


def test_tq005_inline_marker_does_not_suppress_next_statement(tmp_path: Path) -> None:
    source = """import os
os.environ["ZETA_HOME"] = "/tmp/home"  # test-contract: bootstrap session home
os.environ["ZETA_TEST_BOOTSTRAP_STORAGE_FORMAT"] = "1"
"""

    findings = _lint(tmp_path, source, path="tests/conftest.py")

    assert [(finding.code, finding.line) for finding in findings] == [("TQ005", 3)]


def test_tq005_comment_marker_applies_only_to_next_statement(tmp_path: Path) -> None:
    source = """import os
# test-contract: bootstrap session home
os.environ["ZETA_HOME"] = "/tmp/home"
os.environ["ZETA_TEST_BOOTSTRAP_STORAGE_FORMAT"] = "1"
"""

    findings = _lint(tmp_path, source, path="tests/conftest.py")

    assert [(finding.code, finding.line) for finding in findings] == [("TQ005", 4)]


def test_tq005_marker_on_multiline_statement_first_line_suppresses(
    tmp_path: Path,
) -> None:
    source = """import os
os.environ[  # test-contract: bootstrap session home
    "ZETA_HOME"
] = "/tmp/home"
"""

    assert _lint(tmp_path, source, path="tests/conftest.py") == []


def test_tq005_rejects_global_zeta_bootstraps_in_conftest(tmp_path: Path) -> None:
    source = 'import os\nos.environ["ZETA_TEST_BOOTSTRAP_STORAGE_FORMAT"] = "1"\n'

    findings = _lint(tmp_path, source, path="tests/conftest.py")

    assert [(finding.code, finding.line) for finding in findings] == [("TQ005", 2)]
    assert findings[0].message == quality.GLOBAL_BOOTSTRAP_ERROR.format(
        name="ZETA_TEST_BOOTSTRAP_STORAGE_FORMAT"
    )


@pytest.mark.parametrize(
    "source",
    [
        'def test_local(monkeypatch):\n    monkeypatch.setenv("ZETA_LOCAL", "1")\n',
        'def helper():\n    os.environ["ZETA_LOCAL"] = "1"\n',
    ],
)
def test_tq005_allows_local_environment_setup(tmp_path: Path, source: str) -> None:
    assert _lint(tmp_path, source) == []


@pytest.mark.parametrize(
    "source",
    [
        """import os

environ = os.environ
environ[\"ZETA_ALIAS\"] = \"1\"
""",
        """from os import environ
environ[\"ZETA_IMPORTED\"] = \"1\"
""",
        """import os
os.environ |= {\"ZETA_AUGASSIGN\": \"1\"}
""",
        """from os import environ
from pytest import fixture

@fixture(scope=\"session\")
def session():
    environ[\"ZETA_IMPORTED_FIXTURE\"] = \"1\"
""",
    ],
)
def test_tq005_rejects_environment_syntax_variants(tmp_path: Path, source: str) -> None:
    findings = _lint(tmp_path, source, path="tests/conftest.py")

    assert [finding.code for finding in findings] == ["TQ005"]


def test_tq005_reports_dynamic_environment_expression(tmp_path: Path) -> None:
    source = 'import os\nname = input_name\nos.environ[name] = "1"\n'

    findings = _lint(tmp_path, source, path="tests/conftest.py")

    assert len(findings) == 1
    assert "global test bootstrap name hides" in findings[0].message


def test_tq005_rejects_session_fixture_and_hook(tmp_path: Path) -> None:
    source = """@pytest.fixture(scope=\"session\")
def session(monkeypatch):
    monkeypatch.setenv(\"ZETA_SESSION\", \"1\")

def pytest_configure(config):
    os.putenv(\"ZETA_CONFIG\", \"1\")
"""

    findings = _lint(tmp_path, source, path="tests/conftest.py")

    assert [(finding.code, finding.line) for finding in findings] == [
        ("TQ005", 3),
        ("TQ005", 6),
    ]


def test_tq005_reports_expression_names_and_update_keys(tmp_path: Path) -> None:
    source = """import os
name = \"ZETA_DYNAMIC\"
os.environ.setdefault(name, \"1\")
os.environ.update({\"ZETA_ONE\": \"1\", \"NOPE\": \"1\"})
MonkeyPatch().setenv(\"ZETA_PATCH\", \"1\")
"""

    findings = _lint(tmp_path, source, path="tests/sitecustomize.py")

    assert [(finding.code, finding.line, finding.message) for finding in findings] == [
        (
            "TQ005",
            3,
            quality.GLOBAL_BOOTSTRAP_ERROR.format(name="name"),
        ),
        (
            "TQ005",
            4,
            quality.GLOBAL_BOOTSTRAP_ERROR.format(name="ZETA_ONE"),
        ),
        (
            "TQ005",
            5,
            quality.GLOBAL_BOOTSTRAP_ERROR.format(name="ZETA_PATCH"),
        ),
    ]


@pytest.mark.parametrize("layout", ["one-line", "split"])
@pytest.mark.parametrize("marked", [False, True])
def test_private_patch_rule_is_layout_independent(
    tmp_path: Path, layout: str, marked: bool
) -> None:
    path = "tests/test_example.py"
    one_line = '    monkeypatch.setattr(target, "_private", replacement)\n'
    if layout == "one-line":
        call = one_line
        if marked:
            call = (
                "    # test-contract: seam no public replacement hook exists\n" + call
            )
    else:
        marker = (
            "  # test-contract: seam no public replacement hook exists"
            if marked
            else ""
        )
        call = (
            "    monkeypatch.setattr(\n"
            '        target, "_private", replacement\n'
            f"    ){marker}\n"
        )
    source = "def test_case(monkeypatch):\n" + call
    new_diff = _replacement_diff(path, "def test_case(monkeypatch):\n", source)
    reflow_diff = _replacement_diff(
        path, "def test_case(monkeypatch):\n" + one_line, source
    )

    new_findings = _lint(tmp_path, source, diff=new_diff)
    assert [finding.code for finding in new_findings] == ([] if marked else ["TQ003"])
    assert lint_diff(reflow_diff, tmp_path) == []


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (
            'unused = 1\n\ndef test_case(monkeypatch):\n    monkeypatch.setattr(target, "_private", replacement)\n',
            'def test_case(monkeypatch):\n    monkeypatch.setattr(\n        target, "_private", replacement\n    )\n',
        ),
        (
            'def test_case(monkeypatch):\n    monkeypatch.setattr(target, "_private", replacement)\n    assert one\n    assert two\n\nunused = 1\n',
            'def test_case(monkeypatch):\n    monkeypatch.setattr(\n        target, "_private", replacement\n    )\n    assert one\n    assert two\n',
        ),
        (
            'first = 1\n\ndef test_case(monkeypatch):\n    monkeypatch.setattr(target, "_private", replacement)\n\nlast = 1\n',
            'def test_case(monkeypatch):\n    monkeypatch.setattr(\n        target, "_private", replacement\n    )\n',
        ),
    ],
    ids=["deletion-at-line-one", "deletion-at-eof", "multiple-deletion-hunks"],
)
def test_private_patch_reflow_survives_zero_count_destination_hunks(
    tmp_path: Path, before: str, after: str
) -> None:
    diff = _replacement_diff("tests/test_example.py", before, after, context=0)

    assert _lint(tmp_path, after, diff=diff) == []


def test_new_file_has_an_exactly_empty_base_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = 'def test_case(monkeypatch):\n    monkeypatch.setattr(target, "_private", replacement)\n'
    seen_base_sources = []
    real_lint_source = quality.lint_source

    def capture_base_source(path, current, added, base_source=None):
        seen_base_sources.append(base_source)
        return real_lint_source(path, current, added, base_source)

    # test-contract: seam lint_diff does not expose its reconstructed base source
    monkeypatch.setattr(quality, "lint_source", capture_base_source)

    diff = """diff --git a/tests/test_example.py b/tests/test_example.py
new file mode 100644
--- /dev/null
+++ b/tests/test_example.py
@@ -0,0 +1,2 @@
+def test_case(monkeypatch):
+    monkeypatch.setattr(target, "_private", replacement)
"""

    findings = _lint(tmp_path, source, diff=diff)

    assert findings == [
        quality.Finding(
            Path("tests/test_example.py"),
            2,
            5,
            "TQ003",
            quality.CONTRACT_ERROR,
        )
    ]
    assert seen_base_sources == [""]


def test_rule_checks_only_added_test_lines(tmp_path: Path) -> None:
    source = """def test_existing_and_new():
    time.sleep(0.1)
    assert result == "ok"
"""

    assert _lint(tmp_path, source, {3}) == []


def test_src_zeta_keeps_existing_test_rules(tmp_path: Path) -> None:
    source = "async def test_existing():\n    time.sleep(1)\n"

    findings = _lint(tmp_path, source, path="src/zeta/example/tests/test_example.py")

    assert [(finding.code, finding.line) for finding in findings] == [("TQ001", 2)]


@pytest.mark.parametrize(
    ("source", "method"),
    [
        ("async def check(value: Path):\n    value.{method}()\n", method)
        for method in ("exists", "is_dir", "stat", "open", "mkdir")
    ]
    + [
        ("class Tool:\n    x: Path\n\n    async def check(self):\n        self.x.{method}()\n", method)
        for method in ("exists", "is_dir", "stat", "open", "mkdir")
    ]
    + [
        ("class Tool:\n    def __init__(self):\n        self.x = Path('x')\n\n    async def check(self):\n        self.x.{method}()\n", method)
        for method in ("exists", "is_dir", "stat", "open", "mkdir")
    ]
    + [
        ("async def check(root):\n    (root / 'x').{method}()\n", method)
        for method in ("exists", "is_dir", "stat", "open", "mkdir")
    ]
    + [
        ("async def check():\n    (Path('root') / 'x').{method}()\n", method)
        for method in ("exists", "is_dir", "stat", "open", "mkdir")
    ],
)
def test_tq007_path_receiver_shapes(tmp_path: Path, source: str, method: str) -> None:
    rendered = "from pathlib import Path\n\n" + source.format(method=method)
    findings = _lint(tmp_path, rendered, path="src/zeta/example.py")
    call_line = next(
        number for number, line in enumerate(rendered.splitlines(), 1) if f".{method}()" in line
    )

    assert [(finding.code, finding.line) for finding in findings] == [("TQ007", call_line)]


@pytest.mark.parametrize("call", [
    "asyncio.to_thread(Path('x').exists())",
    "asyncio.to_thread(open('x'))",
    "loop.run_in_executor(None, open('x'))",
])
def test_tq007_scans_invoked_offload_arguments(tmp_path: Path, call: str) -> None:
    source = f"from pathlib import Path\n\nasync def check():\n    await {call}\n"

    findings = _lint(tmp_path, source, path="src/zeta/example.py")

    assert [(finding.code, finding.line) for finding in findings] == [("TQ007", 4)]


def test_tq007_allows_uncalled_offload_receivers(tmp_path: Path) -> None:
    source = """from pathlib import Path

async def check(path):
    await asyncio.to_thread(path.exists)
    await asyncio.to_thread(open, path)
    await loop.run_in_executor(None, open, path)
"""

    assert _lint(tmp_path, source, path="src/zeta/example.py") == []


def test_tq007_flags_filesystem_calls_in_async_functions(tmp_path: Path) -> None:
    source = """from pathlib import Path

async def guard():
    Path("cwd").is_dir()
    open("log", "w")
    os.stat("cwd")
"""

    findings = _lint(tmp_path, source, path="src/zeta/example.py")

    assert [(finding.code, finding.line) for finding in findings] == [
        ("TQ007", 4),
        ("TQ007", 5),
        ("TQ007", 6),
    ]
    assert "asyncio.to_thread" in findings[0].message
    assert "guard" in findings[0].message


def test_tq007_skips_nested_sync_and_off_loop_arguments(tmp_path: Path) -> None:
    source = """from pathlib import Path

async def outer():
    def sync():
        Path("old").is_dir()
    await asyncio.to_thread(Path("safe").is_dir)
    mapping.exists()
"""

    assert _lint(tmp_path, source, path="src/zeta/example.py") == []


def test_tq007_requires_a_reasoned_marker(tmp_path: Path) -> None:
    source = """from pathlib import Path

async def guarded():
    # async-blocking: reviewed because the path is already off-loop
    Path("safe").is_dir()

async def unguarded():
    # async-blocking:
    Path("unsafe").is_dir()
"""

    findings = _lint(tmp_path, source, path="src/zeta/example.py")

    assert [(finding.code, finding.line) for finding in findings] == [("TQ007", 9)]


def test_tq007_diff_mode_ignores_untouched_old_hit(tmp_path: Path) -> None:
    source = """from pathlib import Path

async def old():
    Path("old").is_dir()

async def new():
    return 1
"""

    assert _lint(tmp_path, source, added_lines={7}, path="src/zeta/example.py") == []


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


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=path, text=True).strip()


def test_workflow_dispatch_lints_all_commits_since_main(
    tmp_path: Path, monkeypatch
) -> None:
    origin = tmp_path / "origin.git"
    repo_path = tmp_path / "repo"
    subprocess.run(["git", "init", "--bare", "-q", str(origin)], check=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo_path)], check=True)
    subprocess.run(["git", "config", "user.email", "t@e"], cwd=repo_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo_path, check=True)
    (repo_path / "baseline.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
    subprocess.run(["git", "commit", "-qm", "main"], cwd=repo_path, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", str(origin)], cwd=repo_path, check=True
    )
    subprocess.run(["git", "push", "-q", "origin", "main"], cwd=repo_path, check=True)
    subprocess.run(["git", "checkout", "-qb", "feature"], cwd=repo_path, check=True)
    (repo_path / "tests").mkdir()
    (repo_path / "tests/test_bad.py").write_text("def test_bad():\n    time.sleep(1)\n")
    (repo_path / "tests/test_second.py").write_text("def test_second(): pass\n")
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
    subprocess.run(["git", "commit", "-qm", "feature first"], cwd=repo_path, check=True)
    (repo_path / "README").write_text("second commit\n")
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "feature second"], cwd=repo_path, check=True
    )

    monkeypatch.chdir(repo_path)
    base = quality.select_base("workflow_dispatch")
    diff = subprocess.check_output(
        ["git", "diff", "--unified=0", base, "HEAD", "--", "*.py"],
        cwd=repo_path,
        text=True,
    )
    findings = quality.lint_diff(diff, repo_path)
    assert [(finding.code, finding.path) for finding in findings] == [
        ("TQ001", Path("tests/test_bad.py"))
    ]


def test_workflow_dispatch_on_main_has_empty_diff(tmp_path: Path, monkeypatch) -> None:
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    subprocess.run(["git", "config", "user.email", "t@e"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "empty.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "main"], cwd=tmp_path, check=True)
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-q", str(origin)], check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", str(origin)], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "push", "-q", "origin", "main"], cwd=tmp_path, check=True)
    monkeypatch.chdir(tmp_path)
    assert quality.select_base("workflow_dispatch") == _git(
        tmp_path, "rev-parse", "HEAD"
    )


def test_ci_events_are_supported_by_quality_lint_base_selection(
    tmp_path: Path, monkeypatch
) -> None:
    workflow = yaml.safe_load(Path(".github/workflows/ci.yml").read_text())
    events = workflow.get("on", workflow.get(True, {}))
    assert events
    results = []
    for event in events:
        repo_path = tmp_path / event
        repo_path.mkdir()
        subprocess.run(["git", "init", "-q", "-b", "main", str(repo_path)], check=True)
        subprocess.run(
            ["git", "config", "user.email", "t@e"], cwd=repo_path, check=True
        )
        subprocess.run(["git", "config", "user.name", "t"], cwd=repo_path, check=True)
        (repo_path / "file").write_text("x\n")
        subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
        subprocess.run(["git", "commit", "-qm", "main"], cwd=repo_path, check=True)
        (repo_path / "second").write_text("x\n")
        subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
        subprocess.run(["git", "commit", "-qm", "second"], cwd=repo_path, check=True)
        origin = repo_path.with_suffix(".git")
        subprocess.run(["git", "init", "--bare", "-q", str(origin)], check=True)
        subprocess.run(
            ["git", "remote", "add", "origin", str(origin)], cwd=repo_path, check=True
        )
        subprocess.run(
            ["git", "push", "-q", "origin", "main"], cwd=repo_path, check=True
        )
        monkeypatch.chdir(repo_path)
        try:
            base = quality.select_base(event, "main")
        except (ValueError, subprocess.CalledProcessError) as error:
            base = f"error: {error}"
        results.append((event, base))
    failures = [
        (event, base) for event, base in results if str(base).startswith("error:")
    ]
    assert failures == [], results


def test_ci_runs_quality_lint_for_pull_requests_and_main_pushes() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text()
    step = workflow.split("- name: Reject new low-signal test patterns", 1)[1]
    step = step.split("\n  browser:", 1)[0]

    assert "if: github.event_name == 'pull_request'" not in step
    assert "GITHUB_EVENT_NAME: ${{ github.event_name }}" in step
    assert '--base "$GITHUB_EVENT_NAME" "$GITHUB_BASE_REF"' in step


def test_merge_resolution_passes_pull_request_body() -> None:
    workflow = Path(".github/workflows/merge-resolution.yml").read_text()

    assert "types: [opened, synchronize, reopened, edited]" in workflow
    assert "name: merge resolution" in workflow
    assert "if: github.event_name" not in workflow
    assert "group: merge-resolution-${{ github.event.pull_request.number || github.run_id }}" in workflow
    assert "PR_BODY: ${{ github.event.pull_request.body }}" not in workflow
    script = Path("scripts/ci_merge_resolution.sh").read_text()
    assert 'gh pr view "$PR_NUMBER" --json body -q .body' in script
    assert 'args+=(--pr-body-file "$body_file")' in script
    assert '"${args[@]}"' in script


def test_non_test_python_files_are_ignored(tmp_path: Path) -> None:
    path = "src/zeta/wait.py"
    target = tmp_path / path
    target.parent.mkdir(parents=True)
    source = "time.sleep(0.1)\n"
    target.write_text(source)

    assert lint_diff(_diff(path, source), tmp_path) == []


@pytest.mark.parametrize(
    ("path", "source", "codes"),
    [
        ("tests/test_x.py", "async def test_x():\n    await x\n", {"TQ004"}),
        (
            "tests/conftest.py",
            'import os\nos.environ["ZETA_HOME"] = "x"\n',
            {"TQ005"},
        ),
        (
            "sitecustomize.py",
            'import os\nos.environ["ZETA_HOME"] = "x"\n',
            {"TQ005"},
        ),
        (
            "conftest.py",
            'import os\nos.environ["ZETA_HOME"] = "x"\n',
            {"TQ005"},
        ),
        (
            "src/zeta/x.py",
            (
                'from pathlib import Path\nimport asyncio\n'
                'async def x():\n    await x\n'
                '    await asyncio.to_thread(Path("x").exists())\n'
            ),
            {"TQ007"},
        ),
        (
            "src/zeta/tests/test_y.py",
            (
                'from pathlib import Path\nimport asyncio\n'
                'async def test_y():\n    await x\n'
                '    await asyncio.to_thread(Path("x").exists())\n'
            ),
            {"TQ004", "TQ007"},
        ),
        ("scripts/x.py", "async def x():\n    await x\n", set()),
    ],
)
def test_lint_diff_applies_rule_families_by_path(
    tmp_path: Path, path: str, source: str, codes: set[str]
) -> None:
    target = tmp_path / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source)

    assert {finding.code for finding in lint_diff(_diff(path, source), tmp_path)} == codes


# The pre-migration helper behind CI run 42502453427: it stopped at turn_end while
# the deny response (id 4) was still on its way.
_PAST_RAW_SERVE_READ = """
import asyncio
import json


async def _read_raw_until(reader, frames, predicate):
    while True:
        frame = await asyncio.wait_for(reader.readline(), 10)
        assert frame
        frames.append(frame)
        decoded = json.loads(frame)
        if predicate(decoded):
            return decoded
"""


@pytest.mark.parametrize(
    ("path", "source", "flagged"),
    [
        ("tests/test_server_observers.py", _PAST_RAW_SERVE_READ, True),
        (
            "tests/test_x.py",
            "async def f(self):\n    return await self.replay_reader.readline()\n",
            True,
        ),
        # Websocket and HTTP framing reads are not newline JSON-RPC frames.
        (
            "tests/test_x.py",
            (
                "async def f(replay_reader):\n"
                "    return await replay_reader.readexactly(4)\n"
            ),
            False,
        ),
        (
            "tests/test_x.py",
            (
                "async def f(reader):\n"
                "    # test-contract: transport fake HTTP server request line\n"
                "    return await reader.readline()\n"
            ),
            False,
        ),
        (
            "tests/test_x.py",
            (
                "async def f(reader):\n"
                "    # test-contract: seam wrong kind for a reader\n"
                "    return await reader.readline()\n"
            ),
            True,
        ),
        ("tests/support/server_client.py", _PAST_RAW_SERVE_READ, False),
        (
            "tests/test_x.py",
            "async def f(process):\n    return await process.stdout.readline()\n",
            False,
        ),
    ],
)
def test_raw_serve_socket_reads_must_use_the_shared_client(
    tmp_path: Path, path: str, source: str, flagged: bool
) -> None:
    findings = [
        finding
        for finding in _lint(tmp_path, source, path=path)
        if finding.code == "TQ008"
    ]

    assert bool(findings) is flagged
    assert all(
        "use tests/support/server_client.py read helpers" in finding.message
        and "'# test-contract: transport <reason>'" in finding.message
        for finding in findings
    )
