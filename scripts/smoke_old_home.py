#!/usr/bin/env python3
"""Smoke an installed zeta against a home written by pre-#685 zeta."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HINT = re.compile(r"zeta storage-format advance --expected-epoch 0 --target-epoch \d+")


def snapshot_home(home: Path) -> dict[str, tuple[str, bytes | None]]:
    snapshot: dict[str, tuple[str, bytes | None]] = {}
    for path in sorted(home.rglob("*")):
        relative = path.relative_to(home).as_posix()
        if path.is_dir():
            snapshot[relative] = ("directory", None)
        elif path.is_file():
            snapshot[relative] = ("file", path.read_bytes())
        else:
            snapshot[relative] = ("other", None)
    return snapshot


def report_home_changes(
    before: dict[str, tuple[str, bytes | None]],
    after: dict[str, tuple[str, bytes | None]],
) -> list[str]:
    changes = []
    for path in sorted(set(before) | set(after)):
        if path not in before:
            changes.append(f"added: {path}")
        elif path not in after:
            changes.append(f"removed: {path}")
        elif before[path] != after[path]:
            changes.append(f"changed: {path}")
    return changes


def run(binary: Path, home: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {
        "HOME": str(home / "user"),
        "ZETA_HOME": str(home / "zeta"),
        "PATH": os.environ["PATH"],
    }
    return subprocess.run(
        [str(binary), *args], env=env, text=True, capture_output=True, check=False
    )


LOGIN = """
import os
from pathlib import Path
from zeta.accounts import AccountCredentials, AccountId, AccountStore
from zeta.oauth import OAuthTokens
from zeta.providers.anthropic import AnthropicCredentialDriver
AccountCredentials(
    AccountStore(Path(os.environ["ZETA_HOME"])),
    AccountId("anthropic", "default"),
    AnthropicCredentialDriver(),
).save(OAuthTokens("access", "refresh", 9_999_999_999))
"""


def start_fresh(binary: Path, home: Path, label: str) -> None:
    started = run(binary, home, "session", "list")
    if started.returncode != 0:
        raise AssertionError(
            f"{label}: first start failed:\n{started.stdout}\n{started.stderr}"
        )
    if not (home / "zeta" / "storage-format.json").is_file():
        raise AssertionError(f"{label}: first start wrote no stored-format manifest")
    print(f"passed: {label}")


def smoke_fresh_homes(binary: Path, python: Path, root: Path) -> None:
    """A new user's home must start with no manual storage-format step."""
    empty = root / "empty"
    (empty / "user").mkdir(parents=True)
    (empty / "zeta").mkdir()
    start_fresh(binary, empty, "empty home")

    missing = root / "missing"
    (missing / "user").mkdir(parents=True)
    start_fresh(binary, missing, "missing home")

    concurrent = root / "concurrent"
    (concurrent / "user").mkdir(parents=True)
    env = {
        "HOME": str(concurrent / "user"),
        "ZETA_HOME": str(concurrent / "zeta"),
        "PATH": os.environ["PATH"],
    }
    starts = [
        subprocess.Popen(
            [str(binary), "session", "list"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(4)
    ]
    for started in starts:
        stdout, stderr = started.communicate(timeout=120)
        if started.returncode != 0:
            raise AssertionError(
                f"concurrent first start failed:\n{stdout}\n{stderr}"
            )
    print("passed: concurrent first starts")

    login = root / "login"
    (login / "user").mkdir(parents=True)
    logged_in = subprocess.run(
        [str(python), "-c", LOGIN],
        env={"HOME": str(login / "user"), "ZETA_HOME": str(login / "zeta")},
        text=True,
        capture_output=True,
        check=False,
    )
    if logged_in.returncode != 0:
        raise AssertionError(f"login through zeta's store failed:\n{logged_in.stderr}")
    start_fresh(binary, login, "login-only home")


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
        requirements = temp_root / "requirements.txt"
        subprocess.run(
            [
                "uv",
                "export",
                "--frozen",
                "--no-dev",
                "--no-emit-project",
                "--no-hashes",
                "-o",
                str(requirements),
            ],
            cwd=root,
            check=True,
        )
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(venv / "bin" / "python"),
                "-r",
                str(requirements),
            ],
            check=True,
        )
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(venv / "bin" / "python"),
                "--no-deps",
                str(wheel),
            ],
            check=True,
        )
        imported = subprocess.run(
            [str(venv / "bin" / "python"), "-c", "import zeta; print(zeta.__file__)"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        if not Path(imported).resolve().is_relative_to(venv.resolve()):
            raise AssertionError(
                f"zeta imported outside virtual environment: {imported}"
            )
        binary = venv / "bin" / "zeta"
        smoke_fresh_homes(binary, venv / "bin" / "python", temp_root / "fresh")

        zeta_home = home / "zeta"
        before = snapshot_home(zeta_home)
        refused = run(binary, home, "session", "list")
        if refused.returncode == 0:
            raise AssertionError("old home unexpectedly started before format advance")
        hint = HINT.search(refused.stderr)
        if not hint:
            raise AssertionError(
                f"missing complete advance hint in stderr:\n{refused.stderr}"
            )
        changes = report_home_changes(before, snapshot_home(zeta_home))
        if changes == ["added: .storage-format.lock"]:
            pass
        elif changes:
            raise AssertionError(
                "refused startup changed the copied home:\n" + "\n".join(changes)
            )
        else:
            raise AssertionError(
                "refused startup did not create the expected .storage-format.lock"
            )

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
    raise SystemExit(main())
