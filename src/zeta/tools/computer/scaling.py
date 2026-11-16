"""Model-coordinate to backend-coordinate mapping.

Slice 1 passes coordinates through unchanged. Slice 3 replaces this with real
display scaling and origin offset, using the last ``ScreenFrame``'s ``scale``
and origin. Every coordinate conversion goes through here so that change is one
place.
"""

from __future__ import annotations


def to_backend_point(x: int, y: int) -> tuple[int, int]:
    return (x, y)
