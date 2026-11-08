#!/usr/bin/env python3
"""Fail when tests/owner_leak_ratchet.txt gains a row or a higher count than BASE."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.support.owner_counts import baseline_growth

BASELINE = "tests/owner_leak_ratchet.txt"


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_owner_leak_baseline.py <base-ref>", file=sys.stderr)
        return 2
    ref = sys.argv[1]
    if _git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").returncode != 0:
        print(f"base ref {ref!r} is not a commit", file=sys.stderr)
        return 2
    if not _git("ls-tree", "--name-only", ref, "--", BASELINE).stdout.strip():
        print(f"{BASELINE} is not on the base; nothing to compare")
        return 0
    base = _git("show", f"{ref}:{BASELINE}")
    if base.returncode != 0:
        print(base.stderr, file=sys.stderr)
        return 2
    head = ROOT / BASELINE
    # A deleted baseline removes every row, and removing rows is allowed.
    head_text = head.read_text(encoding="utf-8") if head.exists() else ""
    failures = baseline_growth(base.stdout, head_text)
    for failure in failures:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
