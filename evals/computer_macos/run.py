"""Bounded phase-0 and pilot orchestration; no provider calls are made here."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

from .fixture import trial_state

_INPUT_ACTIONS = {"move", "click", "mouse_down", "mouse_up", "drag", "type", "key", "scroll"}


def prepare_home(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    home = (run_dir / "zeta-home").resolve()
    real = Path.home() / ".zeta"
    if home == real or home.exists() and home.is_symlink():
        raise RuntimeError("refusing a real or symlinked ZETA_HOME")
    home.mkdir(parents=True, exist_ok=True)
    settings = home / "settings.toml"
    if not settings.exists():
        settings.write_text('[computer]\nenabled = true\ndefault_app_access = "allow"\n')
    os.environ["ZETA_HOME"] = str(home)
    os.environ["ZETA_ACCOUNT_HOME"] = str(real)
    return home


def _event(run_dir: Path, action: str, started: int, ended: int, **fields: Any) -> dict[str, Any]:
    return {"monotonic_ns": started, "ended_ns": ended, "action": action, "trial": 0, "turn": 0, "call": 0, "action_id": len(fields), **fields}


async def phase0(run_dir: Path, backend: Any | None = None) -> dict[str, Any]:
    prepare_home(run_dir)
    if backend is None:
        if platform.system() != "Darwin":
            raise RuntimeError("phase 0 requires macOS")
        from zeta.tools.computer.mac import make_mac_backend
        backend = make_mac_backend()
        if backend is None:
            raise RuntimeError("macOS helper is not built")
    events: list[dict[str, Any]] = []
    try:
        cold = time.perf_counter_ns()
        await backend.screenshot()
        cold_end = time.perf_counter_ns()
        for _ in range(20):
            started = time.perf_counter_ns()
            frame = await backend.screenshot()
            ended = time.perf_counter_ns()
            events.append(_event(run_dir, "screenshot", started, ended, width=frame.width, height=frame.height, scale=frame.scale, origin=[frame.origin_x, frame.origin_y], image_bytes=len(frame.image)))
        for _ in range(100):
            started = time.perf_counter_ns()
            x, y = await backend.cursor_position()
            ended = time.perf_counter_ns()
            events.append(_event(run_dir, "cursor_position", started, ended, x=x, y=y))
    finally:
        close = getattr(backend, "close", None)
        if close:
            close()
    payload = {"kind": "phase0", "platform": platform.platform(), "cold_start_ns": cold_end - cold, "events": events, "input_events": [e for e in events if e["action"] in _INPUT_ACTIONS]}
    (run_dir / "events.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(e, sort_keys=True) for e in events) + "\n")
    (run_dir / "manifest.json").write_text(json.dumps({"phase": "phase0", "python": sys.version, "head": os.environ.get("GIT_HEAD", "unknown")}, indent=2) + "\n")
    return payload


def pilot(run_dir: Path, trials: int) -> None:
    if trials != 3:
        raise ValueError("the first pilot is exactly three trials")
    prepare_home(run_dir)
    state_dir = run_dir / "fixture"
    for trial in range(1, trials + 1):
        trial_state(state_dir, trial)
    (run_dir / "pilot.json").write_text(json.dumps({"trials": trials, "provider_calls": 0, "status": "not_started"}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("phase0", "pilot"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    if args.mode == "phase0":
        asyncio.run(phase0(args.run_dir))
    else:
        pilot(args.run_dir, args.trials)


if __name__ == "__main__":
    main()
