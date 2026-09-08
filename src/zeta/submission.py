"""Immutable values submitted by the composer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Submission:
    """One immutable submission identity and its captured composer state."""

    id: int
    text: str
    draft_revision: int = 0
    attachment_paths: tuple[Path, ...] = ()
    attachment_tokens: tuple[tuple[str, Path], ...] = ()
    next_image_token: int = 1
    steer: bool = True


__all__ = ["Submission"]
