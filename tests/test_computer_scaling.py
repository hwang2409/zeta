"""Pure coordinate-mapping tests for the computer tool (slice 3)."""

from __future__ import annotations

import pytest

from zeta.tools.computer.scaling import (
    OutOfBounds,
    ScreenGeometry,
    downscale_cap,
    to_backend_point,
    to_model_point,
)


@pytest.mark.parametrize(
    "frame, expected_cap",
    [
        ((2560, 1600), 1280),  # 16:10 retina -> halve
        ((1920, 1080), 1280),  # 16:9 -> width-bound
        ((800, 1280), 800),  # portrait -> height-bound
        ((2048, 1536), 1067),  # 4:3 retina -> height-bound
        ((1024, 768), 1024),  # XGA fits -> no upscale
        ((1280, 800), 1280),  # WXGA exact -> identity
        ((640, 480), 640),  # smaller than box -> no upscale
    ],
)
def test_downscale_cap(frame: tuple[int, int], expected_cap: int) -> None:
    assert downscale_cap(*frame) == expected_cap


# geometry, model (x, y), expected backend point
_BACKEND_CASES = [
    # 1x, no downscale, no offset -> identity
    (ScreenGeometry(1000, 800, 1000, 800, 1.0, 0, 0), (500, 400), (500, 400)),
    # 2x retina downscaled 2560->1280: physical/image ratio and scale cancel
    (ScreenGeometry(2560, 1600, 1280, 800, 2.0, 0, 0), (640, 400), (640, 400)),
    # same, with a non-zero display origin
    (ScreenGeometry(2560, 1600, 1280, 800, 2.0, 100, 50), (640, 400), (740, 450)),
    # 1x downscaled 1920->1280: 1.5x back out
    (ScreenGeometry(1920, 1080, 1280, 720, 1.0, 0, 0), (640, 360), (960, 540)),
    # odd image size forces rounding: 10 * 100/33 = 30.30 -> 30
    (ScreenGeometry(100, 100, 33, 33, 1.0, 0, 0), (10, 10), (30, 30)),
    # top-left corner maps to the origin
    (ScreenGeometry(2560, 1600, 1280, 800, 2.0, 100, 50), (0, 0), (100, 50)),
    # last valid pixel stays inside the half-open display bounds
    (ScreenGeometry(2560, 1600, 1280, 800, 2.0, 100, 50), (1279, 799), (1379, 849)),
    # negative-origin display keeps its right and bottom edges exclusive
    (ScreenGeometry(1280, 800, 1280, 800, 1.0, -1280, 0), (1279, 799), (-1, 799)),
    # non-integer scale rounds, then clamps to the last integer point in bounds
    (ScreenGeometry(1000, 800, 1000, 800, 1.25, -100, -20), (999, 799), (699, 619)),
]


@pytest.mark.parametrize("geometry, model, expected", _BACKEND_CASES)
def test_to_backend_point(
    geometry: ScreenGeometry, model: tuple[int, int], expected: tuple[int, int]
) -> None:
    assert to_backend_point(geometry, *model) == expected


@pytest.mark.parametrize("x, y", [(-1, 0), (0, -1), (1280, 0), (0, 800)])
def test_to_backend_point_rejects_out_of_bounds(x: int, y: int) -> None:
    geometry = ScreenGeometry(2560, 1600, 1280, 800, 2.0, 0, 0)
    with pytest.raises(OutOfBounds):
        to_backend_point(geometry, x, y)


def test_to_model_point_inverts_origin_and_scale() -> None:
    geometry = ScreenGeometry(2560, 1600, 1280, 800, 2.0, 100, 50)
    assert to_model_point(geometry, 740, 450) == (640, 400)
    assert to_model_point(geometry, 100, 50) == (0, 0)


def test_to_model_point_clamps_to_image_bounds() -> None:
    geometry = ScreenGeometry(2560, 1600, 1280, 800, 2.0, 100, 50)
    # a point past the bottom-right of the display clamps to the image size
    assert to_model_point(geometry, 9999, 9999) == (1280, 800)
    assert to_model_point(geometry, 0, 0) == (0, 0)  # above-left clamps to zero


@pytest.mark.parametrize(
    "geometry, model",
    [
        (ScreenGeometry(2560, 1600, 1280, 800, 2.0, 100, 50), (640, 400)),
        (ScreenGeometry(1920, 1080, 1280, 720, 1.0, 0, 0), (640, 360)),
        (ScreenGeometry(1000, 800, 1000, 800, 1.0, 0, 0), (500, 400)),
    ],
)
def test_round_trip(geometry: ScreenGeometry, model: tuple[int, int]) -> None:
    # evenly-dividing cases round-trip exactly; odd ratios can differ by one px
    assert to_model_point(geometry, *to_backend_point(geometry, *model)) == model
