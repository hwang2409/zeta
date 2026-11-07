from pathlib import Path

import pytest

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


def _lint(tmp_path: Path, source: str, added_lines: set[int] | None = None):
    path = "tests/test_example.py"
    target = tmp_path / path
    target.parent.mkdir()
    target.write_text(source)
    return lint_diff(_diff(path, source, added_lines), tmp_path)


@pytest.mark.parametrize(
    "statement",
    [
        "time.sleep(0.01)",
        "await asyncio.sleep(1)",
        "assert time.monotonic() - started < 0.2",
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


def test_time_rule_allows_zero_yields_and_marked_product_latency(
    tmp_path: Path,
) -> None:
    source = """async def test_latency():
    await asyncio.sleep(0)
    # test-contract: latency
    await asyncio.sleep(0.1)
"""

    assert _lint(tmp_path, source) == []


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
    assert findings[0].message == "assert semantic fields, not a hash"


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
    assert (
        "assert through <public interface> or mark the contract" in findings[0].message
    )


@pytest.mark.parametrize("contract", ["complexity", "durability"])
def test_internal_contract_rule_accepts_narrow_markers(
    tmp_path: Path, contract: str
) -> None:
    source = f"""def test_contract():
    # test-contract: {contract}
    assert mock.call_count == 1
"""

    assert _lint(tmp_path, source) == []


def test_rule_checks_only_added_test_lines(tmp_path: Path) -> None:
    source = """def test_existing_and_new():
    time.sleep(0.1)
    assert result == "ok"
"""

    assert _lint(tmp_path, source, {3}) == []


def test_non_test_python_files_are_ignored(tmp_path: Path) -> None:
    path = "src/zeta/wait.py"
    target = tmp_path / path
    target.parent.mkdir(parents=True)
    source = "time.sleep(0.1)\n"
    target.write_text(source)

    assert lint_diff(_diff(path, source), tmp_path) == []
