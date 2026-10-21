"""Canonicalization of persisted reasoning-evicted message views."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ..protocol.types import Message
from .receipt_constructors import (
    RECEIPT_KIND_METADATA,
    _canonicalize_reasoning_hybrid,
)


def canonicalize(
    records: Sequence[tuple[int, Message]],
    *,
    allows: Callable[[int], bool],
) -> tuple[list[tuple[int, Message]], set[int]]:
    """Replace trusted persisted reasoning hybrids with receipt views."""
    output = list(records)
    regenerated: set[int] = set()
    for index, (seq, message) in enumerate(records):
        if not allows(seq) or message.metadata.get(RECEIPT_KIND_METADATA):
            continue
        replacement = _canonicalize_reasoning_hybrid(message, seq)
        if replacement is not None:
            output[index] = (seq, replacement)
            regenerated.add(seq)
    return output, regenerated
