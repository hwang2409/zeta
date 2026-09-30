"""Time-sliced background materialization for transcript history."""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PrewarmAssembly:
    """Incrementally assembled paint and location data."""

    lines: list[list[tuple[str, str]]] = field(default_factory=list)
    raw_locations: list[tuple[str, Any, int]] = field(default_factory=list)
    locations: list[tuple[Any, int]] = field(default_factory=list)
    _pending_lines: list[list[tuple[str, str]]] = field(default_factory=list)
    _has_line_content: bool = False
    _has_location_content: bool = False

    def add_lines(self, lines: list[list[tuple[str, str]]]) -> None:
        """Add lines while incrementally trimming leading/trailing blanks."""

        for line in lines:
            if not self._has_line_content:
                if not "".join(fragment[1] for fragment in line).strip():
                    continue
                self._has_line_content = True
            if line:
                self.lines.extend(self._pending_lines)
                self._pending_lines.clear()
                self.lines.append(line)
            else:
                self._pending_lines.append(line)

    def add_locations(self, locations: list[tuple[str, Any, int]]) -> None:
        """Add location rows with the same incremental blank trimming."""

        for location in locations:
            if not self._has_location_content:
                if not location[0].strip():
                    continue
                self._has_location_content = True
            self.raw_locations.append(location)
            self.locations.append((location[1], location[2]))


def unit_within_limit(unit: Any, search_active: bool, max_chars: int) -> bool:
    """Return whether a unit's readily available source is prewarm-safe."""

    if unit is None:
        return True
    value = unit.value
    if hasattr(value, "search_renderable"):
        value = value.search_renderable if search_active else value.renderable
    source = getattr(value, "plain", None)
    if not isinstance(source, str):
        source = getattr(value, "markup", None)
    return not isinstance(source, str) or len(source) <= max_chars


def add_cached_unit(
    assembly: PrewarmAssembly,
    unit: Any,
    line_cache: dict[int, tuple[str, list[list[tuple[str, str]]]]],
    location_cache: dict[int, tuple[str, list[str], list[int]]],
) -> None:
    """Incrementally add one unit's cached paint and location rows."""

    if unit is None:
        assembly.add_lines([[]])
        assembly.add_locations([("", None, 0)])
        return
    assembly.add_lines(line_cache[unit.key][1])
    _, plain_lines, offsets = location_cache[unit.key]
    assembly.add_locations(
        [(line, unit, offset) for line, offset in zip(plain_lines, offsets)]
    )


def finish_assembly(
    assembly: PrewarmAssembly,
    *,
    limit_lines: Callable[..., list[Any]],
    dim_style: str,
    max_lines: int | None,
) -> tuple[list[list[tuple[str, str]]], list[tuple[Any, int]]]:
    """Apply final line limits without rebuilding the default location map."""

    parsed = limit_lines(
        assembly.lines or [[]],
        first_line=lambda line: "".join(fragment[1] for fragment in line),
        marker=lambda marker_text: [(dim_style, marker_text)],
    )
    locations = assembly.locations
    if max_lines is not None:
        raw_locations = limit_lines(
            assembly.raw_locations,
            first_line=lambda line: line[0],
            marker=lambda marker_text: (marker_text, None, 0),
        )
        locations = [(unit, offset) for _, unit, offset in raw_locations]
    return parsed, locations


def remember_bounded(
    cache: OrderedDict[Any, Any], key: Any, value: Any, *, max_size: int = 3
) -> None:
    """Insert and retain only the newest bounded cache entries."""

    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_size:
        cache.popitem(last=False)


class TranscriptPrewarm:
    """Run transcript prewarming in callbacks bounded by time and unit count.

    The clock is checked between logical units. A callback therefore consumes at
    most the configured budget plus the cost of one unit already in progress.
    Callers should reject predictably oversized units before rendering them.
    """

    def __init__(
        self,
        *,
        scheduler: Callable[[Callable[[], None]], object] | None,
        clock: Callable[[], float] | None = None,
        time_budget: float = 0.008,
        chunk_size: int = 16,
    ) -> None:
        self.scheduler = scheduler
        self.clock = clock or time.monotonic
        self.time_budget = max(0.000_001, time_budget)
        self.chunk_size = max(1, chunk_size)
        self.key: tuple[int, int, int] | None = None
        self.phase = "idle"
        self._closed = False
        self._generation = 0
        self._handles: set[object] = set()
        self._width = 0
        self._count = 0
        self._index = -1
        self._skipped = False
        self._assembly = PrewarmAssembly()
        self._render: Callable[[int, int], None] = lambda _index, _width: None
        self._should_render: Callable[[int], bool] = lambda _index: True
        self._assemble: Callable[[int, int, PrewarmAssembly], None] = (
            lambda _index, _width, _assembly: None
        )
        self._complete: Callable[[int, PrewarmAssembly], None] = (
            lambda _width, _assembly: None
        )

    def cancel(self) -> None:
        """Cancel the active run and any callback handles it scheduled."""

        self._generation += 1
        self.key = None
        self.phase = "idle"
        self._index = -1
        for handle in self._handles:
            cancel = getattr(handle, "cancel", None)
            if cancel is not None:
                cancel()
        self._handles.clear()

    def close(self) -> None:
        """Permanently stop this prewarmer."""

        self._closed = True
        self.cancel()

    def start(
        self,
        *,
        key: tuple[int, int, int],
        width: int,
        count: int,
        render: Callable[[int, int], None],
        should_render: Callable[[int], bool],
        assemble: Callable[[int, int, PrewarmAssembly], None],
        complete: Callable[[int, PrewarmAssembly], None],
    ) -> None:
        """Start a fresh render then incremental assembly run."""

        if self._closed or self.key == key:
            return
        self.cancel()
        self.key = key
        self.phase = "render"
        self._width = width
        self._count = count
        self._index = count - 1
        self._skipped = False
        self._assembly = PrewarmAssembly()
        self._render = render
        self._should_render = should_render
        self._assemble = assemble
        self._complete = complete
        if not self._schedule():
            self.cancel()

    def _schedule(self) -> bool:
        generation = self._generation
        handle: object | None = None

        def callback() -> None:
            if handle is not None:
                self._handles.discard(handle)
            if self._closed or generation != self._generation:
                return
            self._step()

        if self.scheduler is not None:
            handle = self.scheduler(callback)
        else:
            try:
                handle = asyncio.get_running_loop().call_soon(callback)
            except RuntimeError:
                return False
        if handle is not None and hasattr(handle, "cancel"):
            self._handles.add(handle)
        return True

    def _step(self) -> None:
        if self.phase == "finalize":
            self._complete(self._width, self._assembly)
            self.key = None
            self.phase = "idle"
            return

        deadline = self.clock() + self.time_budget
        processed = 0
        while processed < self.chunk_size:
            if self.phase == "render":
                if self._index < 0:
                    if self._skipped:
                        self.key = None
                        self.phase = "idle"
                        return
                    self.phase = "assemble"
                    self._index = 0
                    break
                index = self._index
                self._index -= 1
                if self._should_render(index):
                    self._render(index, self._width)
                else:
                    self._skipped = True
            elif self.phase == "assemble":
                if self._index >= self._count:
                    self.phase = "finalize"
                    break
                self._assemble(self._index, self._width, self._assembly)
                self._index += 1
            processed += 1
            if self.clock() >= deadline:
                break

        if not self._schedule():
            self.cancel()
