#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.support.owner_counts import refresh_baseline


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "lower tests/owner_leak_ratchet.txt to observed unclosed-owner counts; "
            "never raises a count"
        )
    )
    parser.add_argument(
        "observed",
        type=Path,
        help="owner-counts.txt from a CI artifact or ZETA_OWNER_COUNTS=<path> pytest",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("tests/owner_leak_ratchet.txt"),
    )
    args = parser.parse_args()
    count = refresh_baseline(args.baseline, args.observed)
    print(f"wrote {count} owner-leak baseline rows to {args.baseline}")


if __name__ == "__main__":
    main()
