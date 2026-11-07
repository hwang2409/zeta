from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
RUNNER = ROOT / "scripts" / "test-pr.sh"


def test_repository_pytest_rejects_external_import(tmp_path: Path) -> None:
    external_src = tmp_path / "src"
    shutil.copytree(ROOT / "src", external_src)
    external = external_src / "zeta"
    probe = ROOT / "tests" / "test_import_guard_probe.py"
    probe.write_text("def test_probe():\n    assert True\n")
    try:
        result = subprocess.run(
            ["uv", "run", "pytest", "-q", str(probe)],
            cwd=ROOT,
            env={
                **os.environ,
                "PYTHONPATH": str(external.parent),
                "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
            },
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        probe.unlink()
    assert result.returncode == 2
    assert "wrong zeta import:" in result.stdout


def test_runner_refuses_wrong_import(tmp_path: Path) -> None:
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        "#!/bin/sh\nprintf '%s\\n' /outside/site-packages/zeta/__init__.py\n"
    )
    fake_uv.chmod(0o755)
    result = subprocess.run(
        [str(RUNNER), "--record", str(tmp_path / "record.txt")],
        cwd=ROOT,
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert (
        "refusing: zeta imports from /outside/site-packages/zeta/__init__.py"
        in result.stderr
    )


def test_runner_records_explicit_test(tmp_path: Path) -> None:
    test_file = tmp_path / "test_probe.py"
    test_file.write_text("def test_probe():\n    assert True\n")
    record = tmp_path / "record.txt"
    result = subprocess.run(
        [str(RUNNER), "--record", str(record), str(test_file)],
        cwd=ROOT,
        env={**os.environ, "UV_CACHE_DIR": str(tmp_path / "uv-cache")},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    contents = record.read_text()
    assert "head sha:" in contents
    assert "dirty-tree digest:" in contents
    assert "zeta.__file__: " in contents
    assert "uv run pytest -q" in contents
    assert "exit code: 0" in contents
    assert "1 passed" in contents
