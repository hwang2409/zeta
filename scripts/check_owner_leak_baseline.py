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


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_owner_leak_baseline.py <base-ref>", file=sys.stderr)
        return 2
    base = subprocess.run(
        ["git", "show", f"{sys.argv[1]}:{BASELINE}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if base.returncode != 0:
        print(f"{BASELINE} is not on the base; nothing to compare")
        return 0
    failures = baseline_growth(base.stdout, (ROOT / BASELINE).read_text(encoding="utf-8"))
    for failure in failures:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
