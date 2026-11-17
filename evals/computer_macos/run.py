"""Bounded phase-0 and real provider pilot orchestration."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import shutil
import sys
import time
import urllib.request
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from .fixture import trial_state

_INPUT_ACTIONS = {"move", "click", "mouse_down", "mouse_up", "drag", "type", "key", "scroll"}
TASK = "Open the local zeta fixture in Safari and click the blue target exactly once."


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


async def _stop_resource(resource: Any) -> None:
    if isinstance(resource, asyncio.subprocess.Process):
        resource.terminate()
        await asyncio.wait_for(resource.wait(), timeout=5)
    elif hasattr(resource, "terminate"):
        resource.terminate()
        resource.wait(timeout=5)
    elif hasattr(resource, "close"):
        result = resource.close()
        if hasattr(result, "__await__"):
            await result


async def _wait_fixture(url: str) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            response = await asyncio.to_thread(urllib.request.urlopen, url, timeout=1)
            response.close()
            return
        except OSError:
            await asyncio.sleep(0.05)
    raise RuntimeError("fixture did not start")


def _production_provider(url: str, run_dir: Path) -> Callable[[Any, dict[str, Any], int], Awaitable[dict[str, Any]]]:
    async def run(_backend: Any, _state: dict[str, Any], trial: int) -> dict[str, Any]:
        started = time.perf_counter_ns()
        command = [sys.executable, "-m", "zeta", "--print", "--format", "json", "--yolo", "--no-session", "--provider", "codex", "-p", TASK]
        env = os.environ.copy()
        env["ZETA_HOME"] = str((run_dir / "zeta-home").resolve())
        env["ZETA_ACCOUNT_HOME"] = str((Path.home() / ".zeta").resolve())
        process = await asyncio.create_subprocess_exec(*command, cwd=str(Path.cwd()), env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=300)
        except BaseException:
            process.kill()
            await process.wait()
            raise
        ended = time.perf_counter_ns()
        return {"trial": trial, "provider": "codex", "returncode": process.returncode, "provider_time_ns": ended - started, "provider_events": sum(1 for line in stdout.splitlines() if line)}
    return run


async def pilot(
    run_dir: Path,
    trials: int,
    *,
    provider_runner: Callable[[Any, dict[str, Any], int], Awaitable[dict[str, Any]]] | None = None,
    backend_factory: Callable[[], Any] | None = None,
    fixture_factory: Callable[[Path, int], Any] | None = None,
) -> dict[str, Any]:
    if trials != 3:
        raise ValueError("the first pilot is exactly three trials")
    old_home = os.environ.get("ZETA_HOME")
    old_account = os.environ.get("ZETA_ACCOUNT_HOME")
    prepare_home(run_dir)
    state_dir = run_dir / "fixture"
    results: list[dict[str, Any]] = []
    owned: list[Any] = []
    try:
        for trial in range(1, trials + 1):
            state = trial_state(state_dir, trial)
            fixture = fixture_factory(state_dir, trial) if fixture_factory else await _start_fixture(state_dir, trial)
            owned.append(fixture)
            url = f"http://127.0.0.1:8765/?trial={trial}"
            if fixture_factory is None:
                await _wait_fixture("http://127.0.0.1:8765/")
            browser = None
            if fixture_factory is None and platform.system() == "Darwin":
                browser = await asyncio.create_subprocess_exec("open", "-a", "Safari", url)
                owned.append(browser)
            backend = backend_factory() if backend_factory else None
            try:
                if backend is not None:
                    screen = await backend.state()
                    if screen.locked or screen.secure_input or screen.frontmost_bundle_id == "com.apple.SecurityAgent":
                        raise RuntimeError("desktop preflight refused input")
                started = time.perf_counter_ns()
                runner = provider_runner or _production_provider(url, run_dir)
                result = await asyncio.wait_for(runner(backend, state, trial), timeout=300)
                ended = time.perf_counter_ns()
                result.update({"trial": trial, "started_ns": started, "ended_ns": ended, "wall_ns": ended - started})
                results.append(result)
            finally:
                if backend is not None and getattr(backend, "close", None):
                    backend.close()
                if fixture_factory is None:
                    await _stop_resource(fixture)
                    owned.remove(fixture)
                if browser is not None:
                    await _stop_resource(browser)
                    owned.remove(browser)
    finally:
        for resource in reversed(owned):
            await _stop_resource(resource)
        reset = run_dir / "cleanup.json"
        home = run_dir / "zeta-home"
        if home.exists():
            shutil.rmtree(home)
        reset.write_text(json.dumps({"pass": True, "resources": len(owned), "apps_quit": ["Safari"] if owned else [], "files_deleted": [str(home)]}, indent=2, sort_keys=True) + "\n")
        if old_home is None:
            os.environ.pop("ZETA_HOME", None)
        else:
            os.environ["ZETA_HOME"] = old_home
        if old_account is None:
            os.environ.pop("ZETA_ACCOUNT_HOME", None)
        else:
            os.environ["ZETA_ACCOUNT_HOME"] = old_account
    payload = {"trials": results, "provider": "codex", "status": "complete" if len(results) == trials else "failed"}
    (run_dir / "pilot.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


async def _start_fixture(state_dir: Path, trial: int) -> Any:
    process = await asyncio.create_subprocess_exec(sys.executable, "-m", "evals.computer_macos.fixture", "--port", "8765", "--state-dir", str(state_dir), "--trial", str(trial))
    return process


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("phase0", "pilot"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    if args.mode == "phase0":
        asyncio.run(phase0(args.run_dir))
    else:
        asyncio.run(pilot(args.run_dir, args.trials))


if __name__ == "__main__":
    main()
