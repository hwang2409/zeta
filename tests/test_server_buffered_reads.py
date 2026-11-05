from __future__ import annotations

import ast
import asyncio
import json
import subprocess
from collections import deque
from pathlib import Path

import pytest

from tests.support.server_client import read as buffered_read
from tests.support.server_client import request

_READ_METHODS = {"readline", "readuntil"}


def _unbuffered_reader_reads(source: str, path: str) -> list[str]:
    tree = ast.parse(source, filename=path)
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            continue
        request_readers = {
            argument.id
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id in {"_request", "_buffered_request"}
            for argument in call.args[:1]
            if isinstance(argument, ast.Name)
        }
        direct_readers = {
            expression.func.value.id
            for expression in ast.walk(node)
            if isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Attribute)
            and expression.func.attr in _READ_METHODS
            and isinstance(expression.func.value, ast.Name)
        }
        if node.name == "_frames_until_event" or request_readers & direct_readers:
            violations.append(f"{path}:{node.lineno}:{node.name}")
    return violations


def test_server_tests_do_not_mix_raw_reader_reads_with_buffered_requests() -> None:
    root = Path(__file__).parent
    violations = [
        violation
        for path in sorted(root.glob("test_server*.py"))
        for violation in _unbuffered_reader_reads(path.read_text(), str(path))
    ]
    assert not violations, "use _read(reader); raw reader reads mixed with _request: " + ", ".join(
        violations
    )


class _Reader:
    def __init__(self, *frames: dict[str, object]) -> None:
        self.lines = deque(json.dumps(frame).encode() + b"\n" for frame in frames)

    async def readline(self) -> bytes:
        await asyncio.sleep(0)
        return self.lines.popleft() if self.lines else b""


class _Writer:
    def write(self, _: bytes) -> None:
        pass

    async def drain(self) -> None:
        pass


@pytest.mark.asyncio
async def test_buffered_read_regression_rejects_old_direct_read() -> None:
    reader = _Reader(
        {"params": {"event": "tool_end"}},
        {"params": {"event": "approval_end"}},
        {"id": 1, "result": {}},
    )
    await request(reader, _Writer(), 1, "deny")

    with pytest.raises(AssertionError, match="server closed"):
        while True:
            line = await reader.readline()
            assert line, "server closed the socket"
            if json.loads(line).get("params", {}).get("event") == "tool_end":
                break

    assert (await buffered_read(reader))["params"]["event"] == "tool_end"


def test_buffered_read_guard_catches_pre_647_mistake() -> None:
    result = subprocess.run(
        ["git", "show", "eecf85c9^:tests/test_server.py"],
        capture_output=True,
        check=False,
        text=True,
        cwd=Path(__file__).parents[1],
    )
    source = result.stdout or '''
async def test_parent_mode_resolves_live_delegated_approvals_independently(reader):
    await _request(reader, writer, 4, "status")
    frame = await asyncio.wait_for(reader.readline(), TIMEOUT)
'''
    violations = _unbuffered_reader_reads(source, "eecf85c9^:tests/test_server.py")
    assert any(
        "test_parent_mode_resolves_live_delegated_approvals_independently" in violation
        for violation in violations
    )
