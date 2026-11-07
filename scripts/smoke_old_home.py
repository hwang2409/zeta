#!/usr/bin/env python3
"""Smoke an installed zeta against a home written by pre-#685 zeta."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HINT = re.compile(r"zeta storage-format advance --expected-epoch 0 --target-epoch \d+")


def digest_session_files(home: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((home / "sessions").rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(home).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def run(binary: Path, home: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {
        "HOME": str(home / "user"),
        "ZETA_HOME": str(home / "zeta"),
        "PATH": os.environ["PATH"],
    }
    return subprocess.run(
        [str(binary), *args], env=env, text=True, capture_output=True, check=False
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    fixture = root / "tests" / "fixtures" / "fresh_old_home"
    with tempfile.TemporaryDirectory(prefix="zeta-old-home-") as temp:
        temp_root = Path(temp)
        home = temp_root / "home"
        shutil.copytree(fixture, home / "zeta")
        (home / "user").mkdir()
        venv = temp_root / "venv"
        wheel_argument = Path(sys.argv[1]) if len(sys.argv) == 2 else None
        if wheel_argument is None:
            subprocess.run(
                ["uv", "build", "--wheel", "--out-dir", str(temp_root / "dist")],
                cwd=root,
                check=True,
            )
            wheel = next((temp_root / "dist").glob("zeta-*.whl"))
        else:
            wheel = wheel_argument.resolve()
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(venv / "bin" / "python"),
                str(wheel),
            ],
            check=True,
        )
        binary = venv / "bin" / "zeta"

        before = digest_session_files(home / "zeta")
        refused = run(binary, home, "session", "list")
        if refused.returncode == 0:
            raise AssertionError("old home unexpectedly started before format advance")
        hint = HINT.search(refused.stderr)
        if not hint:
            raise AssertionError(
                f"missing complete advance hint in stderr:\n{refused.stderr}"
            )
        if digest_session_files(home / "zeta") != before:
            raise AssertionError("refused startup changed session files")

        command = hint.group(0).split()
        advanced = run(binary, home, *command[1:])
        if advanced.returncode != 0:
            raise AssertionError(
                f"format advance failed:\n{advanced.stdout}\n{advanced.stderr}"
            )
        started = run(binary, home, "session", "list")
        if started.returncode != 0:
            raise AssertionError(
                f"second startup failed:\n{started.stdout}\n{started.stderr}"
            )
        session_id = next(
            path.name
            for path in (home / "zeta" / "sessions").iterdir()
            if path.is_dir()
        )
        if session_id[:8] not in started.stdout:
            raise AssertionError(
                f"session {session_id} missing from listing:\n{started.stdout}"
            )
        print(f"passed: session={session_id}")


if __name__ == "__main__":
    main()
