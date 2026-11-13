from __future__ import annotations

import multiprocessing
import threading
from pathlib import Path

import pytest

from zeta.core.session_execution import SessionInUseError
from zeta.session_retention.handshake import (
    RetentionYieldError,
    open_with_retention_yield,
    planner_active,
)


def _hold_active(home: str, ready, release) -> None:
    with planner_active(Path(home)):
        ready.set()
        release.wait(10)


def test_open_yields_to_a_planner_in_another_process(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    process = context.Process(target=_hold_active, args=(str(tmp_path), ready, release))
    process.start()
    try:
        assert ready.wait(10)
        ticks = iter((1.0, 3.0))
        with pytest.raises(RetentionYieldError, match="did not yield"):
            open_with_retention_yield(
                tmp_path,
                lambda: (_ for _ in ()).throw(SessionInUseError("busy")),
                clock=lambda: next(ticks),
                wait=lambda _: None,
            )
        release.set()
        process.join(10)
        assert process.exitcode == 0
    finally:
        release.set()
        if process.is_alive():
            process.terminate()
            process.join(10)
    assert not process.is_alive()


def test_open_without_planner_does_not_wait(tmp_path: Path) -> None:
    waited = threading.Event()
    result = open_with_retention_yield(
        tmp_path, lambda: "opened", wait=lambda _: waited.set()
    )
    assert result == "opened"
    assert not waited.is_set()
