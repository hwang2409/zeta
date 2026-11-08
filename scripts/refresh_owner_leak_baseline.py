#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from tests.support.owner_counts import refresh_baseline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="lower the owner-leak baseline from a recorded pytest log"
    )
    parser.add_argument("observed", type=Path)
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
