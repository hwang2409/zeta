"""Model-coordinate to backend-coordinate mapping.

Every screenshot is downscaled to a model-friendly size (the WXGA box below)
so the model always works in one small coordinate space, whatever the display
resolution. The backend captures physical pixels; the model sees the downscaled
image. ``ScreenGeometry`` records the one frame's geometry at the last
screenshot, and the pure functions here map between the two spaces.

The display point space is the global CoreGraphics space the backend acts in:
``frame`` pixels are ``scale`` pixels per point, offset by the display origin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Anthropic's XGA/WXGA guidance: keep screenshots at or below WXGA (1280x800).
WXGA_WIDTH = 1280
WXGA_HEIGHT = 800


class OutOfBounds(ValueError):
    """A model coordinate fell outside the downscaled image."""


def downscale_cap(frame_width: int, frame_height: int) -> int:
    """Longest-side pixel cap that fits the frame in the WXGA box, no upscaling.

    Feeding this to the image policy's ``max_dimension`` downscales any display
    so both sides land inside 1280x800 while keeping the aspect ratio.
    """

    factor = min(WXGA_WIDTH / frame_width, WXGA_HEIGHT / frame_height, 1.0)
    return max(1, round(max(frame_width, frame_height) * factor))


@dataclass(frozen=True, slots=True)
class ScreenGeometry:
    """Geometry of the one frame the model is looking at.

    ``frame_*`` are the captured physical pixels; ``image_*`` are the downscaled
    pixels the model sees; ``scale`` is physical pixels per point; ``origin_*``
    is the display origin in global points.
    """

    frame_width: int
    frame_height: int
    image_width: int
    image_height: int
    scale: float
    origin_x: int
    origin_y: int


def to_backend_point(geometry: ScreenGeometry, x: int, y: int) -> tuple[int, int]:
    """Map a model image coordinate to a global display point.

    Rejects coordinates outside the downscaled image; clamps the result to the
    display bounds so edge rounding never lands off-screen.
    """

    if not (0 <= x < geometry.image_width and 0 <= y < geometry.image_height):
        raise OutOfBounds(
            f"coordinate ({x}, {y}) is outside the "
            f"{geometry.image_width}x{geometry.image_height} screenshot"
        )
    point_x = geometry.origin_x + (x * geometry.frame_width / geometry.image_width) / geometry.scale
    point_y = geometry.origin_y + (y * geometry.frame_height / geometry.image_height) / geometry.scale
    max_x = geometry.origin_x + geometry.frame_width / geometry.scale
    max_y = geometry.origin_y + geometry.frame_height / geometry.scale
    # Display bounds are half-open. Clamp after rounding to the largest integer
    # point strictly below the exclusive edge, including negative origins.
    last_x = math.ceil(max_x) - 1
    last_y = math.ceil(max_y) - 1
    return (
        min(max(round(point_x), geometry.origin_x), last_x),
        min(max(round(point_y), geometry.origin_y), last_y),
    )


def to_model_point(geometry: ScreenGeometry, x: int, y: int) -> tuple[int, int]:
    """Map a global display point back to a model image coordinate.

    Used for ``cursor_position``. Clamps to the image bounds.
    """

    model_x = (x - geometry.origin_x) * geometry.scale * geometry.image_width / geometry.frame_width
    model_y = (y - geometry.origin_y) * geometry.scale * geometry.image_height / geometry.frame_height
    return (
        round(min(max(model_x, 0), geometry.image_width)),
        round(min(max(model_y, 0), geometry.image_height)),
    )
