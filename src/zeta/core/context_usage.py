"""Session-level provider usage accounting."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


class SessionUsage:
    """Accumulate provider usage without changing context calibration."""

    def __init__(
        self,
        *,
        sink: Callable[[Mapping[str, Any]], None] | None = None,
        on_growth: Callable[[int], None] | None = None,
    ) -> None:
        self.total = 0
        self.cache_read = 0
        self.cache_creation = 0
        self.uncached_input = 0
        self.output = 0
        self._sink = sink
        self._on_growth = on_growth

    def record(self, usage: Mapping[str, Any]) -> None:
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
        cache_read = usage.get("cache_read_input_tokens")
        cache_creation = usage.get("cache_creation_input_tokens")
        for value, attribute in (
            (input_tokens, "uncached_input"),
            (output_tokens, "output"),
            (cache_read, "cache_read"),
            (cache_creation, "cache_creation"),
        ):
            if type(value) is int and value >= 0:
                setattr(self, attribute, getattr(self, attribute) + value)
        total = usage.get("total_tokens")
        if type(total) is not int:
            known = [
                value
                for value in (input_tokens, output_tokens, cache_read, cache_creation)
                if type(value) is int and value >= 0
            ]
            total = sum(known) if known else None
        if type(total) is int and total >= 0:
            self.total += total
        if self._sink is not None:
            self._sink(usage)
        if self._on_growth is not None:
            self._on_growth(self.total)
