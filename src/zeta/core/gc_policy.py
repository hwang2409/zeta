"""Garbage-collector policy for long-lived interactive sessions."""

from __future__ import annotations

import gc


def freeze_long_lived_heap() -> None:
    """Move the fully built session graph out of future cyclic GC scans.

    Resume creates a large, mostly immutable object graph.  Collecting once
    before freezing avoids retaining dead bootstrap objects, while freezing
    keeps later generation-2 collections focused on objects created by turns.
    Callers own idempotency because a rebuilt session may need another freeze.
    """

    gc.collect()
    gc.freeze()
