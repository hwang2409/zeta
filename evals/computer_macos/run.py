"""bounded phase-0 and computer-use evaluation orchestration."""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from .fixture import trial_state
from .grade import grade_trial

EXPECTED_PROVIDER = "codex"
EXPECTED_MODEL = "gpt-5.6-luna"
TRIAL_DEADLINE = 300
PILOT_DEADLINE = 1200
TASK = "Open the local zeta fixture in Safari and click the blue target exactly once."

# Public contract used by scripted lifecycle tests; cleanup never changes the primary state.
TERMINAL_STATE_TABLE = {
    "success": {"grade": True, "cleanup": True},
    "provider_failure": {"grade": False, "cleanup": True},
    "provider_nonzero": {"grade": False, "cleanup": True},
    "cancelled": {"grade": False, "cleanup": True},
    "trial_timeout": {"grade": False, "cleanup": True},
    "pilot_timeout": {"grade": False, "cleanup": True},
    "fixture_partial_startup": {"grade": False, "cleanup": True},
    "browser_partial_startup": {"grade": False, "cleanup": True},
    "restoration_failure": {"grade": False, "cleanup": False},
    "cleanup_failure": {"grade": False, "cleanup": False},
    "restart_next_trial": {"grade": True, "cleanup": True},
    "duplicate_input": {"grade": False, "cleanup": True},
}


class BrowserAdapter(Protocol):
    async def snapshot(self) -> dict[str, Any]: ...
    async def open_owned(self, url: str) -> Any: ...
    async def cleanup_owned(self, before: dict[str, Any], owned: Any = None) -> dict[str, Any]: ...


class ProviderAdapter(Protocol):
    provider: str
    model: str


class TrialFixture(Protocol):
    url: str
    async def close_and_verify(self) -> bool: ...


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
    return home


@contextlib.contextmanager
def _home_environment(home: Path):
    previous = {key: os.environ.get(key) for key in ("ZETA_HOME", "ZETA_ACCOUNT_HOME")}
    os.environ["ZETA_HOME"] = str(home)
    os.environ["ZETA_ACCOUNT_HOME"] = str(Path.home() / ".zeta")
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _event_metrics(events: list[Any]) -> dict[str, Any]:
    actions: list[str] = []
    usage = 0
    for event in events:
        call = getattr(event, "tool_call", None)
        if call is not None and call.name == "computer":
            action = call.arguments.get("action")
            if isinstance(action, str):
                actions.append(action)
        data = getattr(event, "data", {})
        if isinstance(data, dict):
            value = data.get("usage", {}).get("total_tokens", 0)
            if isinstance(value, int):
                usage += value
    return {"computer_actions": actions, "usage_tokens": usage, "provider_events": len(events)}


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
            events.append({"monotonic_ns": started, "ended_ns": ended, "action": "screenshot", "width": frame.width, "height": frame.height, "scale": frame.scale, "origin": [frame.origin_x, frame.origin_y], "image_bytes": len(frame.image)})
        for _ in range(100):
            started = time.perf_counter_ns()
            x, y = await backend.cursor_position()
            ended = time.perf_counter_ns()
            events.append({"monotonic_ns": started, "ended_ns": ended, "action": "cursor_position", "x": x, "y": y})
    finally:
        close = getattr(backend, "close", None)
        if close:
            result = close()
            if hasattr(result, "__await__"):
                await result
    _atomic_json(run_dir / "manifest.json", {"phase": "phase0", "python": sys.version, "platform": platform.platform()})
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(e, sort_keys=True) for e in events) + "\n")
    return {"kind": "phase0", "cold_start_ns": cold_end - cold, "events": events, "input_events": []}


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


async def _preflight(backend: Any) -> None:
    state = await backend.state()
    if state.locked or state.secure_input or state.frontmost_bundle_id == "com.apple.SecurityAgent":
        raise RuntimeError("desktop preflight refused input")


class MacSafari:
    """Safari ownership adapter; every claim is made by a post-operation observation."""

    def __init__(self) -> None:
        self._before_tabs: set[tuple[str, str]] = set()

    def _script(self, source: str) -> str:
        return subprocess.check_output(["osascript", "-e", source], text=True)

    async def snapshot(self) -> dict[str, Any]:
        output = await asyncio.to_thread(self._script, """
        tell application \"System Events\" to set front to bundle identifier of first application process whose frontmost is true
        set output to front & return
        tell application \"Safari\"
          if (count of windows) > 0 then set output to output & (id of front window as text) & tab & (index of current tab of front window as text) & tab & (URL of current tab of front window) & return
          set output to output & return
          repeat with w in windows
            repeat with t in tabs of w
              set output to output & (id of w as text) & tab & (id of t as text) & tab & (URL of t) & return
            end repeat
          end repeat
        end tell
        return output
        """)
        lines = output.splitlines()
        tabs = []
        selected = lines[1].split("\t", 1) if len(lines) > 1 else []
        for line in lines[2:]:
            parts = line.split("\t", 2)
            if len(parts) == 3:
                tabs.append({"wid": parts[0], "tid": parts[1], "u": parts[2]})
        snapshot = {"frontmost": lines[0] if lines else None, "selected": selected, "tabs": tabs}
        self._before_tabs = {(row["wid"], row["tid"]) for row in tabs}
        return snapshot

    async def open_owned(self, url: str) -> dict[str, str]:
        script = f'tell application "Safari" to make new document with properties {{URL:"{url}"}}'
        await asyncio.to_thread(subprocess.check_call, ["osascript", "-e", script])
        after = await self.snapshot()
        candidates = [row for row in after.get("tabs", []) if (row["wid"], row["tid"]) not in self._before_tabs]
        return candidates[-1] if candidates else {"wid": "", "tid": ""}

    async def cleanup_owned(self, before: dict[str, Any], owned: Any = None) -> dict[str, Any]:
        after = await self.snapshot()
        identity = (owned or {}).get("wid"), (owned or {}).get("tid")
        row = next((item for item in after.get("tabs", []) if (item.get("wid"), item.get("tid")) == identity), None)
        if row is not None:
            script = f'tell application "Safari" to close tab id {row["tid"]} of window id {row["wid"]}'
            await asyncio.to_thread(subprocess.check_call, ["osascript", "-e", script])
        selected = before.get("selected", [])
        if len(selected) >= 2:
            script = f'tell application "Safari" to set current tab of window id {selected[0]} to tab {selected[1]}'
            await asyncio.to_thread(subprocess.check_call, ["osascript", "-e", script])
            if len(selected) >= 3:
                script = f'tell application "Safari" to set URL of current tab of window id {selected[0]} to {json.dumps(selected[2])}'
                await asyncio.to_thread(subprocess.check_call, ["osascript", "-e", script])
        frontmost = before.get("frontmost")
        if isinstance(frontmost, str) and frontmost != "com.apple.Safari":
            script = f'tell application "System Events" to set frontmost of first application process whose bundle identifier is "{frontmost}" to true'
            await asyncio.to_thread(subprocess.check_call, ["osascript", "-e", script])
        observed = await self.snapshot()
        remaining = {(item["wid"], item["tid"]) for item in observed.get("tabs", [])}
        return {"owned_browser_state_removed": identity not in remaining, "prior_state_restored": observed == before}


async def _close_backend(backend: Any) -> bool:
    close = getattr(backend, "close", None)
    if close is None:
        return True
    result = close()
    if hasattr(result, "__await__"):
        await result
    return bool(getattr(backend, "closed", True))


async def _run_zeta_loop(provider: ProviderAdapter, backend: Any, run_dir: Path) -> dict[str, Any]:
    from zeta.config.settings import ComputerUseConfig
    from zeta.core.store import ConversationStore
    from zeta.protocol.types import MessageOrigin, StreamEventType
    from zeta.runtime.loop import AgentLoop
    from zeta.skills import SkillCatalog
    from zeta.tools import ToolRegistry

    session = ConversationStore(run_dir / "session", cwd=run_dir)
    registry = ToolRegistry(run_dir, session_store=session, skill_catalog=SkillCatalog.empty(), computer=ComputerUseConfig(enabled=True, default_app_access="allow"), required_tool_names=("computer",))
    registry.computer_backend = backend
    loop = AgentLoop(provider, session, registry=registry, skill_catalog=SkillCatalog.empty(), skip_mcp_mount=True, max_turns=8)
    await loop.activate()
    events: list[Any] = []
    try:
        async for event in loop.run_turn(TASK, origin=MessageOrigin.USER):
            events.append(event)
            if event.type is StreamEventType.ERROR:
                raise RuntimeError(event.error.message if event.error else "provider failed")
    finally:
        await loop.close()
        await registry.close()
        session.close()
    return _event_metrics(events)


async def _bounded(awaitable: Any, seconds: float) -> Any:
    """cleanup must not extend the owning trial or pilot deadline."""
    return await asyncio.wait_for(asyncio.shield(awaitable), timeout=seconds)


def _safe_error(exc: BaseException) -> str:
    return type(exc).__name__


async def run_trial(
    run_dir: Path, trial: int, provider: ProviderAdapter, backend: Any,
    *, fixture_factory: Callable[[Path, int], Any] | None = None,
    browser: BrowserAdapter | None = None,
) -> dict[str, Any]:
    """Own one trial. The returned record is the sole cleanup evidence."""
    fixture = None
    before = None
    owned = None
    cleanup: dict[str, Any] = {
        "fixture_dead": False, "backend_closed": False,
        "owned_browser_state_removed": browser is None, "prior_state_restored": browser is None,
    }
    terminal = {"trial": trial, "status": "failed", "error": None, "cleanup": cleanup}
    fixture_dir = run_dir / "fixture"
    state_path = fixture_dir / f"trial-{trial}.json"
    state_path.unlink(missing_ok=True)
    try:
        state = trial_state(fixture_dir, trial)
        fixture = fixture_factory(fixture_dir, trial) if fixture_factory else await _start_fixture(fixture_dir, trial)
        url = getattr(fixture, "url", "http://127.0.0.1:8765") + "/"
        if fixture_factory is None:
            await _wait_fixture(url)
        if getattr(backend, "url", None) is None:
            backend.url = url.rstrip("/")
        if not getattr(backend, "nonce", None):
            backend.nonce = state["nonce"]
        await _preflight(backend)
        if browser is not None:
            before = await browser.snapshot()
            owned = await browser.open_owned(url)
        if provider.provider != EXPECTED_PROVIDER or provider.model != EXPECTED_MODEL:
            raise RuntimeError("provider/model contract mismatch")
        with _home_environment(run_dir / "zeta-home"):
            metrics = await _run_zeta_loop(provider, backend, run_dir / f"trial-{trial}")
        grade = grade_trial(trial_state(fixture_dir, trial))
        terminal.update({"provider": provider.provider, "model": provider.model, "metrics": metrics,
                         "grade": {"trial": trial, **grade}, "status": "complete" if grade["pass"] else "failed"})
        if not grade["pass"]:
            terminal["error"] = "grade_failed"
    except asyncio.CancelledError:
        terminal["status"], terminal["error"] = "cancelled", "CancelledError"
        raise
    except TimeoutError:
        terminal["status"], terminal["error"] = "timeout", "TimeoutError"
        raise
    except BaseException as exc:
        terminal["error"] = _safe_error(exc)
        raise
    finally:
        async def close_one(name: str, action: Any) -> None:
            try:
                observed = await _bounded(action(), 10)
                if name == "backend":
                    cleanup["backend_closed"] = bool(observed)
                elif name == "fixture":
                    cleanup["fixture_dead"] = bool(observed)
                elif isinstance(observed, dict):
                    cleanup.update({key: bool(value) for key, value in observed.items() if key in cleanup})
            except BaseException as exc:  # noqa: BLE001 - cleanup continues after cancellation
                cleanup[f"{name}_error"] = _safe_error(exc)
        async def finish_cleanup(action: Any) -> None:
            task = asyncio.create_task(action())
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                # finish every cleanup independently, then let the original cancellation win.
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    await task
        if browser is not None and before is not None:
            async def cleanup_browser() -> Any:
                try:
                    return await browser.cleanup_owned(before, owned)
                except TypeError as exc:
                    if "positional" not in str(exc) and "argument" not in str(exc):
                        raise
                    return await browser.cleanup_owned(before)
            await finish_cleanup(lambda: close_one("browser", cleanup_browser))
        await finish_cleanup(lambda: close_one("backend", lambda: _close_backend(backend)))
        if fixture is not None:
            await finish_cleanup(lambda: close_one("fixture", lambda: _close_fixture(fixture)))
        terminal["cleanup"] = cleanup
        if terminal["status"] == "complete" and not all(cleanup.get(key, False) for key in ("fixture_dead", "backend_closed", "owned_browser_state_removed", "prior_state_restored")):
            terminal["status"], terminal["error"] = "failed", "cleanup_failed"
        _atomic_json(run_dir / f"trial-{trial}.json", terminal)
    return terminal


async def _close_fixture(fixture: Any) -> bool:
    close = getattr(fixture, "close_and_verify", None)
    if close is not None:
        return bool(await close())
    if fixture.returncode is None:
        fixture.terminate()
    try:
        await asyncio.wait_for(fixture.wait(), 10)
    except TimeoutError:
        fixture.kill()
        await asyncio.wait_for(fixture.wait(), 10)
    return fixture.returncode is not None


async def pilot(
    run_dir: Path, trials: int = 3, *,
    provider_factory: Callable[[int], ProviderAdapter] | None = None,
    backend_factory: Callable[[int], Any] | None = None,
    fixture_factory: Callable[[Path, int], Any] | None = None,
    browser_factory: Callable[[], BrowserAdapter] | None = None,
    timeout_factory: Callable[[float], Any] = asyncio.timeout,
) -> dict[str, Any]:
    if trials != 3:
        raise ValueError("the pilot is exactly three trials")
    results: list[dict[str, Any]] = []
    status, error = "failed", None
    try:
        prepare_home(run_dir)
        async with timeout_factory(PILOT_DEADLINE):
            for trial in range(1, 4):
                provider = provider_factory(trial) if provider_factory else _production_provider(run_dir)
                backend = backend_factory(trial) if backend_factory else _production_backend()
                browser = browser_factory() if browser_factory else (MacSafari() if platform.system() == "Darwin" else None)
                try:
                    async with timeout_factory(TRIAL_DEADLINE):
                        record = await run_trial(run_dir, trial, provider, backend, fixture_factory=fixture_factory, browser=browser)
                except BaseException:
                    artifact = run_dir / f"trial-{trial}.json"
                    if artifact.exists():
                        results.append(json.loads(artifact.read_text()))
                    raise
                results.append(record)
                if record["status"] != "complete":
                    raise RuntimeError(record.get("error") or "trial failed")
        status = "complete"
    except asyncio.CancelledError:
        status, error = "cancelled", "CancelledError"
        raise
    except TimeoutError:
        status, error = "timeout", "TimeoutError"
        raise
    except BaseException as exc:
        error = _safe_error(exc)
        raise
    finally:
        cleanup = {"fixture_dead": all(r.get("cleanup", {}).get("fixture_dead", False) for r in results),
                   "backend_closed": all(r.get("cleanup", {}).get("backend_closed", False) for r in results),
                   "owned_browser_state_removed": all(r.get("cleanup", {}).get("owned_browser_state_removed", False) for r in results),
                   "prior_state_restored": all(r.get("cleanup", {}).get("prior_state_restored", False) for r in results),
                   "disposable_home_removed": False}
        try:
            home = run_dir / "zeta-home"
            if home.exists():
                shutil.rmtree(home)
            cleanup["disposable_home_removed"] = not home.exists()
        except OSError as exc:
            cleanup["home_error"] = _safe_error(exc)
        if status == "complete" and not cleanup["disposable_home_removed"]:
            status, error = "failed", "cleanup_failed"
        payload = {"status": status, "error": error, "provider": EXPECTED_PROVIDER, "model": EXPECTED_MODEL,
                   "trials": results, "cleanup": cleanup}
        _atomic_json(run_dir / "cleanup.json", cleanup)
        _atomic_json(run_dir / "pilot.json", payload)
    return payload

def _production_backend() -> Any:
    if platform.system() != "Darwin":
        raise RuntimeError("pilot requires macOS")
    from zeta.tools.computer.mac import make_mac_backend
    backend = make_mac_backend()
    if backend is None:
        raise RuntimeError("macOS helper is not built")
    return backend


def _production_provider(run_dir: Path) -> ProviderAdapter:
    from zeta.providers.factory import build_backend
    backend, model = build_backend(EXPECTED_PROVIDER, EXPECTED_MODEL, home=run_dir / "zeta-home", require_credentials=True)
    if backend.provider != EXPECTED_PROVIDER or model != EXPECTED_MODEL:
        raise RuntimeError(f"provider/model selection mismatch: {backend.provider}/{model}")
    return backend


async def _start_fixture(state_dir: Path, trial: int) -> Any:
    return await asyncio.create_subprocess_exec(sys.executable, "-m", "evals.computer_macos.fixture", "--port", "8765", "--state-dir", str(state_dir), "--trial", str(trial))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("phase0", "pilot"))
    parser.add_argument("--run-dir", type=Path, required=True)
    if "pilot" in sys.argv:
        parser.add_argument("--trials", type=int, choices=(3,), default=3)
    args = parser.parse_args()
    asyncio.run(phase0(args.run_dir) if args.mode == "phase0" else pilot(args.run_dir, trials=args.trials))


if __name__ == "__main__":
    main()
