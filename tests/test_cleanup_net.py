from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROBE = """
import pytest
from pathlib import Path
from zeta.core.store import ConversationStore
from tests.support.owned import owned_resources

@pytest.mark.asyncio
async def test_case(tmp_path):
    MODE = {mode!r}
    if MODE == 'cleanup-error':
        def fail_close(self):
            raise RuntimeError('cleanup failed')
        ConversationStore.close = fail_close
    elif MODE == 'leak-marker':
        original_close = ConversationStore.close
        def mark_close(self):
            original_close(self)
            Path(__import__('os').environ['CLEANUP_MARKER']).write_text('closed')
        ConversationStore.close = mark_close
    store = ConversationStore(tmp_path / 'sessions')
    if MODE == 'owned':
        async with owned_resources() as owner:
            owner.store(store)
    elif MODE == 'body-failure':
        pytest.fail('body failed')
"""


def _run_probe(tmp_path: Path, mode: str) -> subprocess.CompletedProcess[str]:
    probe = Path(__file__).with_name("_cleanup_net_probe.py")
    probe.write_text(PROBE.format(mode=mode))
    try:
        return subprocess.run(
            [sys.executable, "-m", "pytest", str(probe), "-vv", "-o", "asyncio_debug=true"],
            cwd=Path(__file__).parents[1],
            env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src"), "CLEANUP_MARKER": str(tmp_path / "closed")},
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        probe.unlink(missing_ok=True)


def test_open_resource_fails_with_owner_and_creation_site(tmp_path: Path) -> None:
    result = _run_probe(tmp_path, "leak")
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "ConversationStore created at" in output
    assert "owned_resources().store(store)" in output


def test_leak_is_closed_before_teardown_failure(tmp_path: Path) -> None:
    result = _run_probe(tmp_path, "leak-marker")
    assert result.returncode != 0
    assert (tmp_path / "closed").read_text() == "closed"


def test_owned_resource_passes(tmp_path: Path) -> None:
    result = _run_probe(tmp_path, "owned")
    assert result.returncode == 0, result.stdout + result.stderr


def test_body_failure_keeps_original_failure_and_cleanup_note(tmp_path: Path) -> None:
    result = _run_probe(tmp_path, "body-failure")
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "body failed" in output
    assert "ConversationStore created at" in output


def test_cleanup_error_is_reported(tmp_path: Path) -> None:
    result = _run_probe(tmp_path, "cleanup-error")
    assert result.returncode != 0
    assert "cleanup failed" in result.stdout + result.stderr


def test_ratchet_entry_suppresses_known_leak(tmp_path: Path) -> None:
    ratchet = Path(__file__).with_name("test_cleanup_net_ratchet.txt")
    ratchet.write_text("tests/_cleanup_net_probe.py::test_case\n")
    try:
        result = _run_probe(tmp_path, "leak")
    finally:
        ratchet.unlink(missing_ok=True)
    assert result.returncode == 0, result.stdout + result.stderr
