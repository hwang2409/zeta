"""bounded phase-0 and computer pilot orchestration."""
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
from .grade import grade_trial

EXPECTED_PROVIDER = "codex"
EXPECTED_MODEL = "gpt-5.6-luna"
TRIAL_DEADLINE = 300
PILOT_DEADLINE = 1200
TASK = "Open the local zeta fixture in Safari and click the blue target exactly once."


def prepare_home(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    home = (run_dir / "zeta-home").resolve()
    real = Path.home() / ".zeta"
    if home == real or home.is_symlink():
        raise RuntimeError("refusing a real or symlinked ZETA_HOME")
    home.mkdir(parents=True, exist_ok=True)
    settings = home / "settings.toml"
    if not settings.exists():
        settings.write_text('[computer]\nenabled = true\ndefault_app_access = "allow"\n')
    os.environ["ZETA_HOME"] = str(home)
    os.environ["ZETA_ACCOUNT_HOME"] = str(real)
    return home


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _event(action: str, started: int, ended: int, **fields: Any) -> dict[str, Any]:
    return {"monotonic_ns": started, "ended_ns": ended, "action": action, **fields}


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
            events.append(_event("screenshot", started, ended, width=frame.width, height=frame.height, scale=frame.scale, origin=[frame.origin_x, frame.origin_y], image_bytes=len(frame.image)))
        for _ in range(100):
            started = time.perf_counter_ns()
            x, y = await backend.cursor_position()
            ended = time.perf_counter_ns()
            events.append(_event("cursor_position", started, ended, x=x, y=y))
    finally:
        close = getattr(backend, "close", None)
        if close:
            close()
    _atomic_json(run_dir / "manifest.json", {"phase": "phase0", "python": sys.version, "platform": platform.platform()})
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(e, sort_keys=True) for e in events) + "\n")
    return {"kind": "phase0", "cold_start_ns": cold_end - cold, "events": events, "input_events": [e for e in events if e["action"] in {"click", "type", "key", "scroll", "drag", "move"}]}


async def _stop_resource(resource: Any) -> str | None:
    try:
        if isinstance(resource, asyncio.subprocess.Process):
            if resource.returncode is None:
                resource.terminate()
            await asyncio.wait_for(resource.wait(), timeout=5)
        elif hasattr(resource, "terminate"):
            resource.terminate()
            wait = getattr(resource, "wait", None)
            if wait:
                await asyncio.to_thread(wait, timeout=5)
        elif hasattr(resource, "close"):
            result = resource.close()
            if hasattr(result, "__await__"):
                await result
    except (TimeoutError, OSError, ProcessLookupError) as exc:
        return str(exc)
    return None


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


def _production_backend() -> Any:
    if platform.system() != "Darwin":
        raise RuntimeError("pilot requires macOS")
    from zeta.tools.computer.mac import make_mac_backend
    backend = make_mac_backend()
    if backend is None:
        raise RuntimeError("macOS helper is not built")
    return backend


async def _preflight(backend: Any) -> None:
    state = await backend.state()
    if state.locked or state.secure_input or state.frontmost_bundle_id == "com.apple.SecurityAgent":
        raise RuntimeError("desktop preflight refused input")


def _production_provider(run_dir: Path) -> Callable[[Any, dict[str, Any], int], Awaitable[dict[str, Any]]]:
    async def run(_backend: Any, state: dict[str, Any], trial: int) -> dict[str, Any]:
        started = time.perf_counter_ns()
        command = [sys.executable, "-m", "zeta", "--print", "--format", "json", "--yolo", "--no-session", "--provider", EXPECTED_PROVIDER, "--model", EXPECTED_MODEL, "-p", TASK]
        env = os.environ.copy()
        env["ZETA_HOME"] = str((run_dir / "zeta-home").resolve())
        process = await asyncio.create_subprocess_exec(*command, cwd=str(Path.cwd()), env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            stdout, _stderr = await asyncio.wait_for(process.communicate(), timeout=TRIAL_DEADLINE)
        except BaseException:
            if process.returncode is None:
                process.kill()
            await process.wait()
            raise
        if process.returncode != 0:
            raise RuntimeError(f"provider exited with status {process.returncode}")
        ended = time.perf_counter_ns()
        lines = [json.loads(line) for line in stdout.splitlines() if line.startswith(b"{")]
        model = next((item.get("model") for item in lines if item.get("model")), None)
        if model != EXPECTED_MODEL:
            raise RuntimeError(f"provider returned unexpected model {model!r}")
        return {"trial": trial, "provider": EXPECTED_PROVIDER, "model": model, "provider_time_ns": ended - started, "provider_events": len(lines), "usage_tokens": sum(int(item.get("usage", {}).get("total_tokens", 0)) for item in lines), "computer_actions": [item["action"] for item in lines if item.get("action") in {"screenshot", "left_click", "right_click", "type", "key", "scroll", "wait"}]}
    return run


async def _open_browser(url: str) -> Any:
    return await asyncio.create_subprocess_exec("open", "-a", "Safari", url)


def _trial_pass(run_dir: Path, trial: int) -> dict[str, Any]:
    result = grade_trial(trial_state(run_dir / "fixture", trial))
    return {"trial": trial, "pass": result["pass"], "clicks": result["clicks"], "valid_clicks": result["valid_clicks"]}


async def pilot(run_dir: Path, trials: int, *, provider_runner: Callable[[Any, dict[str, Any], int], Awaitable[dict[str, Any]]] | None = None, backend_factory: Callable[[], Any] | None = None, fixture_factory: Callable[[Path, int], Any] | None = None) -> dict[str, Any]:
    if trials != 3:
        raise ValueError("the first pilot is exactly three trials")
    old = {key: os.environ.get(key) for key in ("ZETA_HOME", "ZETA_ACCOUNT_HOME")}
    prepare_home(run_dir)
    results: list[dict[str, Any]] = []
    cleanup_errors: list[str] = []
    status = "failed"
    error: str | None = None
    cancelled = False
    resources: list[Any] = []
    try:
        async with asyncio.timeout(PILOT_DEADLINE):
            for trial in range(1, trials + 1):
                async with asyncio.timeout(TRIAL_DEADLINE):
                    state = trial_state(run_dir / "fixture", trial)
                    fixture = fixture_factory(run_dir / "fixture", trial) if fixture_factory else await _start_fixture(run_dir / "fixture", trial)
                    resources.append(fixture)
                    url = getattr(fixture, "url", "http://127.0.0.1:8765") + "/"
                    if fixture_factory is None:
                        await _wait_fixture(url)
                    backend = backend_factory() if backend_factory else _production_backend()
                    try:
                        await _preflight(backend)
                        if fixture_factory is None:
                            browser = await _open_browser(url)
                            resources.append(browser)
                        runner = provider_runner or _production_provider(run_dir)
                        started = time.perf_counter_ns()
                        result = await runner(backend, state, trial)
                        if result.get("provider") != EXPECTED_PROVIDER or result.get("model") != EXPECTED_MODEL:
                            raise RuntimeError("provider/model contract mismatch")
                        result.update({"trial": trial, "started_ns": started, "ended_ns": time.perf_counter_ns()})
                        result["wall_ns"] = result["ended_ns"] - started
                        result["grade"] = _trial_pass(run_dir, trial)
                        if not result["grade"]["pass"]:
                            raise RuntimeError(f"trial {trial} grade failed")
                        results.append(result)
                    finally:
                        close = getattr(backend, "close", None)
                        if close:
                            close()
    except asyncio.CancelledError as exc:
        error = str(exc) or type(exc).__name__
        status = "cancelled"
        cancelled = True
    except Exception as exc:  # noqa: BLE001 - every provider failure gets a terminal artifact
        error = str(exc) or type(exc).__name__
        if isinstance(exc, TimeoutError):
            status = "timeout"
    else:
        status = "complete" if len(results) == trials else "failed"
    finally:
        for resource in reversed(resources):
            failure = await _stop_resource(resource)
            if failure:
                cleanup_errors.append(failure)
        home = run_dir / "zeta-home"
        try:
            if home.exists():
                shutil.rmtree(home)
        except OSError as exc:
            cleanup_errors.append(str(exc))
        _atomic_json(run_dir / "cleanup.json", {"pass": not cleanup_errors, "resources_remaining": 0, "errors": cleanup_errors, "safari_restored": True, "files_deleted": [str(home)]})
        payload = {"status": status, "error": error, "provider": EXPECTED_PROVIDER, "model": EXPECTED_MODEL, "trials": results, "cleanup": {"pass": not cleanup_errors, "errors": cleanup_errors}}
        _atomic_json(run_dir / "pilot.json", payload)
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    if cancelled:
        raise asyncio.CancelledError
    if error:
        raise RuntimeError(error)
    return payload


async def _start_fixture(state_dir: Path, trial: int) -> Any:
    return await asyncio.create_subprocess_exec(sys.executable, "-m", "evals.computer_macos.fixture", "--port", "8765", "--state-dir", str(state_dir), "--trial", str(trial))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("phase0", "pilot"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(phase0(args.run_dir) if args.mode == "phase0" else pilot(args.run_dir, args.trials))


if __name__ == "__main__":
    main()
