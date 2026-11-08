#!/usr/bin/env python3
"""Emit the release's deterministic stored-format compatibility registry."""

from __future__ import annotations

import json
import sys

from zeta.storage_format import registry_dump


def main() -> int:
    json.dump(registry_dump(), sys.stdout, sort_keys=True, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
