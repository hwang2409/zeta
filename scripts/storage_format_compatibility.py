#!/usr/bin/env python3
"""Run the stored-format probe against the selected first-parent predecessor."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path

from zeta.storage_format import StorageFormatError, compare_registry_dumps

PROBE_PATHS = ("src/zeta/storage_format.py", "tests/compatibility/storage_probe.py")
ROUNDTRIP = Path(__file__).resolve().parents[1] / "tests/compatibility/storage_roundtrip.py"
CLASSIFIER_PREFIXES = ("src/zeta/", "tests/compatibility/", ".github/workflows/", "scripts/")
CLASSIFIER_FILES = {"pyproject.toml", "uv.lock"}

def _git(*args: str, check: bool = True) -> str:
    return subprocess.check_output(("git", *args), text=True, stderr=subprocess.STDOUT).strip()

def qualifying(commit: str) -> bool:
    """Classify first-parent changes, including merge commits."""
    parents = _git("rev-list", "--parents", "-n", "1", commit).split()
    if len(parents) == 1:
        changed = _git("diff-tree", "--root", "--no-commit-id", "--name-only", "-r", commit).splitlines()
    else:
        changed = _git("diff", "--name-only", f"{commit}^1", commit).splitlines()
    return any(path in CLASSIFIER_FILES or path.startswith(CLASSIFIER_PREFIXES) for path in changed)

def has_probe(commit: str) -> bool:
    return all(subprocess.run(("git", "cat-file", "-e", f"{commit}:{path}"), capture_output=True, check=False).returncode == 0 for path in PROBE_PATHS)

def select_predecessor(*, event: str, head: str, base: str) -> tuple[str, list[str]]:
    anchor = base if event == "pull_request" else _git("rev-parse", f"{head}^")
    commits = _git("rev-list", "--first-parent", anchor).splitlines()
    predecessor = next((commit for commit in commits if qualifying(commit)), None)
    if predecessor is None:
        raise RuntimeError(f"no qualifying predecessor at or before {anchor}")
    return predecessor, commits[commits.index(predecessor):]

def _digest_tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name != ".storage-format.lock"
        and ".storage-format-shape-locks" not in path.parts
    }


def _assert_mutation_delta(before: dict[str, str], after: dict[str, str], allowed: Iterable[str]) -> None:
    changed = {path for path in set(before) | set(after) if before.get(path) != after.get(path)}
    unexpected = changed - set(allowed)
    if unexpected:
        raise RuntimeError(f"old-writer mutation changed undeclared paths: {sorted(unexpected)!r}")


def _run_roundtrips(source: Path, predecessor_epoch: int) -> None:
    with tempfile.TemporaryDirectory(prefix="storage-format-roundtrip-") as temporary, tempfile.TemporaryDirectory(prefix="storage-format-uv-cache-") as cache:
        home = Path(temporary)
        environment = {**os.environ, "UV_CACHE_DIR": cache}
        (home / "storage-format.json").write_text(json.dumps({"writer_epoch": predecessor_epoch}) + "\n")
        (home / "meta.json").write_text(json.dumps({"owned": "before", "future": {"keep": True}}) + "\n")
        (home / ".storage-format.lock").touch()
        high = home / "higher"
        shutil.copytree(home, high)
        (high / "storage-format.json").write_text(json.dumps({"writer_epoch": predecessor_epoch + 1}) + "\n")
        before_high = _digest_tree(high)
        result = subprocess.run(
            ("uv", "run", "--frozen", "python", str(ROUNDTRIP), "--home", str(high), "--mode", "write"),
            cwd=source, env={**environment, "PYTHONPATH": str(source / "src")},
            text=True, capture_output=True, check=False,
        )
        if result.returncode == 0:
            raise RuntimeError("old writer accepted a higher-epoch home")
        if _digest_tree(high) != before_high:
            raise RuntimeError("old writer mutated a higher-epoch home")
        before = _digest_tree(home)
        result = subprocess.run(
            ("uv", "run", "--frozen", "python", str(ROUNDTRIP), "--home", str(home), "--mode", "write"),
            cwd=source, env={**environment, "PYTHONPATH": str(source / "src")},
            text=True, capture_output=True, check=False,
        )
        if result.returncode:
            raise RuntimeError(f"old-writer round trip failed: {result.stderr or result.stdout}")
        _assert_mutation_delta(before, _digest_tree(home), ("meta.json",))
        candidate = subprocess.run(
            ("uv", "run", "--frozen", "python", str(ROUNDTRIP), "--home", str(home), "--mode", "reopen"),
            cwd=ROUNDTRIP.parents[1],
            text=True, capture_output=True,
            env={**environment, "PYTHONPATH": str(ROUNDTRIP.parents[1] / "src")},
            check=False,
        )
        if candidate.returncode:
            raise RuntimeError(f"candidate could not reopen old-writer data: {candidate.stderr or candidate.stdout}")
    print("::notice::selected predecessor old-writer round trips passed")


def _probe_source(source: Path, output: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source / "src")
    result = subprocess.run(("uv", "run", "--frozen", "python", "tests/compatibility/storage_probe.py"), cwd=source, env=env, text=True, capture_output=True, check=False)
    if result.returncode or not result.stdout.strip():
        raise RuntimeError(f"predecessor probe failed: {result.stderr or result.stdout}")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"predecessor probe emitted malformed json: {exc}") from exc
    output.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    event = os.environ.get("GITHUB_EVENT_NAME", "push")
    head = os.environ.get("GITHUB_SHA", "HEAD")
    base = os.environ.get("GITHUB_BASE_SHA") or _git("rev-parse", "origin/main")
    predecessor, earlier = select_predecessor(event=event, head=head, base=base)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "storage-format-predecessor-sha").write_text(predecessor + "\n")
    _git("archive", "--format=tar", predecessor, "-o", str(output / "predecessor.tar"))
    with tempfile.TemporaryDirectory(prefix="storage-format-lock-") as lock_dir:
        lock_path = Path(lock_dir) / "uv.lock"
        subprocess.run(("git", "show", f"{predecessor}:uv.lock"), check=True, stdout=lock_path.open("wb"))
        shutil.copy2(lock_path, output / "uv.lock")
    candidate = output / "storage-format-candidate.json"
    env = os.environ.copy()
    result = subprocess.run(("uv", "run", "--frozen", "python", "tests/compatibility/storage_probe.py"), text=True, capture_output=True, env=env, check=False)
    if result.returncode:
        raise RuntimeError(f"candidate probe failed: {result.stderr or result.stdout}")
    try:
        candidate.write_text(json.dumps(json.loads(result.stdout), sort_keys=True, separators=(",", ":")) + "\n")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"candidate probe emitted malformed json: {exc}") from exc
    if not has_probe(predecessor):
        if any(has_probe(commit) for commit in earlier[1:] if qualifying(commit)):
            raise RuntimeError(f"selected predecessor {predecessor} is missing the compatibility probe")
        print(f"::notice::bootstrap predecessor {predecessor}: skipped registry comparison and old-writer round trips")
        return 0
    with tempfile.TemporaryDirectory(prefix="storage-format-predecessor-") as temporary:
        source = Path(temporary)
        subprocess.run(("tar", "xf", str(output / "predecessor.tar"), "-C", str(source)), check=True)
        _probe_source(source, output / "storage-format-predecessor.json")
    predecessor_registry = json.loads((output / "storage-format-predecessor.json").read_text())
    compare_registry_dumps(predecessor_registry, json.loads(candidate.read_text()))
    _run_roundtrips(source, predecessor_registry["writer_epoch"])
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (StorageFormatError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
