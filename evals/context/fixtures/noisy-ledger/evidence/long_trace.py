"""Generate a deterministic, realistic long replay trace without storing it."""

from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--part", type=int, choices=(1, 2), required=True)
args = parser.parse_args()
for index in range(4_000):
    partition = (index * 19 + args.part * 5) % 113
    offset = args.part * 1_000_000 + index
    print(
        f"2026-02-{args.part + 12:02d}T08:{index % 60:02d}:{(index * 11) % 60:02d}Z "
        f"consumer={index % 37:02d} partition={partition:03d} offset={offset:07d} "
        f"parcel=P-{(index * 23) % 100003:06d} event=scan attempt=1 "
        "checkpoint=committed dedupe=hit validation=ok outcome=accepted lag_ms=14"
    )
