"""Generate a deterministic, realistic long incident trace without storing it."""

from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--part", type=int, choices=(1, 2), required=True)
args = parser.parse_args()
for index in range(4_000):
    shard = (index * 17 + args.part * 11) % 97
    latency = 8 + (index * 13) % 211
    print(
        f"2026-01-{args.part + 10:02d}T12:{index % 60:02d}:{(index * 7) % 60:02d}Z "
        f"worker={index % 41:02d} shard={shard:02d} job=task-{args.part}-{index:06d} "
        f"event=poll queue_depth={1200 + index % 503} latency_ms={latency} "
        "cache=warm lock=held retry=0 outcome=deferred checkpoint=stable"
    )
