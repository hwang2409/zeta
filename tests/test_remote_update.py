from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from zeta.remote.update import _remote_script


def git(cwd: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def repo_pair(tmp_path: Path) -> tuple[Path, Path, str, str]:
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init", "--quiet", "-b", "main")
    git(source, "config", "user.email", "test@example.com")
    git(source, "config", "user.name", "test")
    (source / "pyproject.toml").write_text("[project]\nname='zeta'\n")
    git(source, "add", ".")
    git(source, "commit", "--quiet", "-m", "base")
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "clone", "--quiet", "--bare", str(source), str(origin)], check=True)
    remote = tmp_path / "remote"
    subprocess.run(["git", "clone", "--quiet", str(origin), str(remote)], check=True)
    target_source = source
    (target_source / "README").write_text("new\n")
    git(target_source, "add", "README")
    git(target_source, "commit", "--quiet", "-m", "target")
    target = git(target_source, "rev-parse", "HEAD")
    subprocess.run(["git", "--git-dir", str(origin), "fetch", "--quiet", str(source), "main"], check=True)
    subprocess.run(["git", "--git-dir", str(origin), "update-ref", "refs/heads/main", target], check=True)
    git(remote, "remote", "set-url", "origin", str(origin))
    return origin, remote, git(remote, "rev-parse", "HEAD"), target


def run_script(remote: Path, target: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    tool = tmp_path / "tool"
    (tool / "bin").mkdir(parents=True, exist_ok=True)
    uv = tmp_path / ".local/bin/uv"
    uv.parent.mkdir(parents=True, exist_ok=True)
    uv.write_text(f"#!/bin/sh\nif [ \"$1\" = tool ] && [ \"$2\" = dir ]; then echo {tool}; else exit 1; fi\n")
    uv.chmod(0o755)
    python = tool / "bin/python"
    if not python.exists():
        python.symlink_to(sys.executable)
    init = remote / "src/zeta"
    init.mkdir(parents=True, exist_ok=True)
    (init / "__init__.py").write_text("")
    env = {**os.environ, "HOME": str(tmp_path), "PATH": "/usr/bin:/bin", "PYTHONPATH": str(remote / "src")}
    return subprocess.run(
        ["sh", "-c", _remote_script(target, "fake-host")],
        cwd=remote,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_clean_checkout_moves_to_target(tmp_path: Path) -> None:
    _, remote, old, target = repo_pair(tmp_path)
    result = run_script(remote, target, tmp_path)
    assert result.returncode == 0, result.stderr
    assert target in result.stdout
    assert git(remote, "rev-parse", "HEAD") == target
    assert old in result.stdout


def test_dirty_checkout_is_unchanged(tmp_path: Path) -> None:
    _, remote, old, target = repo_pair(tmp_path)
    (remote / "pyproject.toml").write_text("dirty\n")
    result = run_script(remote, target, tmp_path)
    assert result.returncode != 0
    assert "not clean" in result.stderr
    assert git(remote, "rev-parse", "HEAD") == old


def test_backwards_update_is_refused(tmp_path: Path) -> None:
    _, remote, old, target = repo_pair(tmp_path)
    assert run_script(remote, target, tmp_path).returncode == 0
    result = run_script(remote, old, tmp_path)
    assert result.returncode != 0
    assert "backwards" in result.stderr
    assert git(remote, "rev-parse", "HEAD") == target


def test_host_starting_with_dash_is_rejected() -> None:
    from zeta.remote.update import update_remote

    with pytest.raises(ValueError, match="invalid HOST"):
        update_remote("-bad")
