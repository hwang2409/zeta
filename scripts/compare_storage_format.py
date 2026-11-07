#!/usr/bin/env python3
"""Compare two storage-format probe dumps in isolated environments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from zeta.storage_format import StorageFormatError, compare_registry_dumps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("predecessor", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    try:
        compare_registry_dumps(json.loads(args.predecessor.read_text()), json.loads(args.candidate.read_text()))
    except (StorageFormatError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
