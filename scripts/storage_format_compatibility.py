#!/usr/bin/env python3
"""Select a first-parent predecessor and run the candidate registry probe."""
from __future__ import annotations
import os, subprocess
from pathlib import Path

def qualifying(commit: str) -> bool:
    changed = subprocess.check_output(["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit], text=True)
    return any(p == "pyproject.toml" or p == "uv.lock" or p.startswith("src/zeta/") for p in changed.splitlines())

def has_probe(commit: str) -> bool:
    return all(subprocess.run(["git", "cat-file", "-e", f"{commit}:{path}"], capture_output=True).returncode == 0 for path in ("src/zeta/storage_format.py", "tests/compatibility/storage_probe.py"))

def main() -> int:
    event = os.environ.get("GITHUB_EVENT_NAME", "push")
    head = os.environ.get("GITHUB_SHA", "HEAD")
    base = os.environ.get("GITHUB_BASE_SHA") or "origin/main"
    anchor = base if event == "pull_request" else subprocess.check_output(["git", "rev-parse", f"{head}^"], text=True).strip()
    commits = subprocess.check_output(["git", "rev-list", "--first-parent", anchor], text=True).splitlines()
    predecessor = next((c for c in commits if qualifying(c)), None)
    if predecessor is None: return 0
    print(f"selected predecessor {predecessor}")
    if not has_probe(predecessor): print(f"::notice::bootstrap predecessor {predecessor}: skipped registry comparison and old-writer round trips")
    return 0
if __name__ == "__main__": raise SystemExit(main())
