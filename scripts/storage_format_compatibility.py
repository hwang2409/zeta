#!/usr/bin/env python3
"""Run the stored-format probe against the selected first-parent predecessor."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from zeta.storage_format import StorageFormatError, compare_registry_dumps

PROBE_PATHS = ("src/zeta/storage_format.py", "tests/compatibility/storage_probe.py")
CLASSIFIER_PREFIXES = ("src/zeta/", "tests/compatibility/", ".github/workflows/", "scripts/")
CLASSIFIER_FILES = {"pyproject.toml", "uv.lock"}

def _git(*args: str) -> str:
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
        print(f"::notice::bootstrap predecessor {predecessor}: registry comparison starts with this predecessor")
        return 0
    with tempfile.TemporaryDirectory(prefix="storage-format-predecessor-") as temporary:
        source = Path(temporary)
        subprocess.run(("tar", "xf", str(output / "predecessor.tar"), "-C", str(source)), check=True)
        _probe_source(source, output / "storage-format-predecessor.json")
    compare_registry_dumps(json.loads((output / "storage-format-predecessor.json").read_text()), json.loads(candidate.read_text()))
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (StorageFormatError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
