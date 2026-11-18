"""bounded phase-0 and computer-use evaluation orchestration."""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import json
import os
import platform
import shutil
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


@dataclasses.dataclass(frozen=True)
class SafariOwnership:
    window_id: str
    tab_id: str
    prior_frontmost_bundle: str


class BrowserAdapter(Protocol):
    async def open_owned(self, url: str) -> SafariOwnership: ...
    async def cleanup_owned(self, ownership: SafariOwnership) -> dict[str, Any]: ...


CleanupEvidence = dict[str, Any]


def _empty_cleanup() -> CleanupEvidence:
    return {
        "fixture_dead": False,
        "backend_closed": False,
        "owned_browser_state_removed": False,
        "prior_state_restored": False,
        "focus_restore_safe": False,
        "error": None,
    }


class ProviderAdapter(Protocol):
    provider: str
    model: str


class TrialFixture(Protocol):
    url: str
    async def close_and_verify(self) -> bool: ...


@contextlib.contextmanager
def prepare_home(run_dir: Path):
    old = {key: os.environ.get(key) for key in ("ZETA_HOME", "ZETA_ACCOUNT_HOME")}
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
    try:
        yield home
    finally:
        for key, value in old.items():
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
    with prepare_home(run_dir):
        return await _phase0(run_dir, backend)


async def _phase0(run_dir: Path, backend: Any | None = None) -> dict[str, Any]:
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
    """Stateless Safari ownership adapter."""

    @staticmethod
    async def _osascript(source: str) -> str:
        process = await asyncio.create_subprocess_exec("osascript", "-e", source, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            stdout, stderr = await process.communicate()
        except BaseException:
            if process.returncode is None:
                process.kill()
            await process.wait()
            raise
        if process.returncode:
            raise RuntimeError(stderr.decode(errors="replace").strip() or f"osascript exited {process.returncode}")
        return stdout.decode()

    async def open_owned(self, url: str) -> SafariOwnership:
        escaped = url.replace('\\', '\\\\').replace('"', '\\"')
        output = await self._osascript(f"""tell application \"System Events\" to set prior to bundle identifier of first application process whose frontmost is true
 tell application \"Safari\"
   set ownedTab to make new document with properties {{URL:\"{escaped}\"}}
   set ownedWindow to window of ownedTab
   return prior & tab & (id of ownedWindow as text) & tab & (id of ownedTab as text)
 end tell""")
        parts = output.strip().split("\t")
        if len(parts) != 3 or not all(parts):
            raise RuntimeError("Safari did not return an ownership token")
        return SafariOwnership(parts[1], parts[2], parts[0])

    async def cleanup_owned(self, ownership: SafariOwnership) -> dict[str, Any]:
        bundle = ownership.prior_frontmost_bundle.replace('"', '\\\\"')
        output = await self._osascript(f"""set removed to false
set restored to false
set safariGlobalFront to false
set exactFront to false
tell application "Safari"
 set ownedWindow to window id {ownership.window_id}
 set tabsInWindow to tabs of ownedWindow
 if (count of tabsInWindow) is 1 and (id of item 1 of tabsInWindow as text) is "{ownership.tab_id}" then
   close ownedWindow
   set removed to true
 else
   repeat with candidate in tabsInWindow
     if (id of candidate as text) is "{ownership.tab_id}" then
       close candidate
       set removed to true
       exit repeat
     end if
   end repeat
 end if
end tell
tell application "System Events"
 set safariGlobalFront to frontmost of first application process whose bundle identifier is "com.apple.Safari"
end tell
tell application "Safari"
 set exactFront to ((id of front window as text) is "{ownership.window_id}")
end tell
if safariGlobalFront and exactFront then
 tell application "System Events" to set frontmost of first application process whose bundle identifier is "{bundle}" to true
 set restored to true
end if
return (removed as text) & tab & (restored as text) & tab & (safariGlobalFront as text) & tab & (exactFront as text)""")
        parts = output.strip().split("\t")
        if len(parts) != 4:
            raise RuntimeError("Safari cleanup did not return evidence")
        return {
            "owned_browser_state_removed": parts[0].lower() == "true",
            "prior_state_restored": parts[1].lower() == "true",
            "focus_restore_safe": True,
        }


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


async def _bounded(awaitable: Any, timeout: float) -> Any:
    task = asyncio.create_task(awaitable)
    try:
        return await asyncio.wait_for(task, timeout)
    except BaseException:
        if not task.done():
            task.cancel()
        try:
            await task
        except (Exception, asyncio.CancelledError):  # noqa: BLE001, S110 - drain before propagating
            pass
        raise


async def _cleanup_trial(
    backend: Any,
    fixture: Any | None,
    browser: BrowserAdapter | None,
    ownership: SafariOwnership | None,
) -> CleanupEvidence:
    evidence = _empty_cleanup()
    steps: list[tuple[str, Callable[[], Any]]] = []
    if browser is not None and ownership is not None:
        steps.append(("browser", lambda: browser.cleanup_owned(ownership)))
    steps.append(("backend", lambda: _close_backend(backend)))
    if fixture is not None:
        steps.append(("fixture", lambda: _close_fixture(fixture)))
    for name, action in steps:
        try:
            observed = action()
            if hasattr(observed, "__await__"):
                observed = await _bounded(observed, 10)
            if name == "backend":
                evidence["backend_closed"] = bool(observed)
            elif name == "fixture":
                evidence["fixture_dead"] = bool(observed)
            elif isinstance(observed, dict):
                for key in ("owned_browser_state_removed", "prior_state_restored", "focus_restore_safe"):
                    if key in observed:
                        evidence[key] = bool(observed[key])
        except BaseException as exc:  # noqa: BLE001 - cleanup continues after cancellation/failure
            evidence["error"] = str(exc) or type(exc).__name__
    return evidence


async def run_trial(
    run_dir: Path,
    trial: int,
    provider: ProviderAdapter,
    backend: Any,
    *,
    fixture_factory: Callable[[Path, int], Any] | None = None,
    browser: BrowserAdapter | None = None,
) -> dict[str, Any]:
    """Own one complete trial, including all resources and verified cleanup."""
    fixture: Any | None = None
    ownership: SafariOwnership | None = None
    result: dict[str, Any] | None = None
    failure: BaseException | None = None
    try:
        state = trial_state(run_dir / "fixture", trial)
        fixture = fixture_factory(run_dir / "fixture", trial) if fixture_factory else await _start_fixture(run_dir / "fixture", trial)
        url = getattr(fixture, "url", "http://127.0.0.1:8765") + "/"
        if fixture_factory is None:
            await _wait_fixture(url)
        if getattr(backend, "url", None) is None:
            backend.url = url.rstrip("/")
        if not getattr(backend, "nonce", None):
            backend.nonce = state["nonce"]
        await _preflight(backend)
        if browser is not None:
            ownership = await browser.open_owned(url)
        if provider.provider != EXPECTED_PROVIDER or provider.model != EXPECTED_MODEL:
            raise RuntimeError(f"provider/model contract mismatch: {provider.provider}/{provider.model}")
        metrics = await _run_zeta_loop(provider, backend, run_dir)
        grade = grade_trial(trial_state(run_dir / "fixture", trial))
        if not grade["pass"]:
            raise RuntimeError(f"trial {trial} grade failed")
        result = {"trial": trial, "provider": provider.provider, "model": provider.model, "metrics": metrics, "grade": {"trial": trial, "pass": grade["pass"], "clicks": grade["clicks"], "valid_clicks": grade["valid_clicks"]}, "status": "complete"}
    except BaseException as exc:
        failure = exc
    cleanup = await _cleanup_trial(backend, fixture, browser, ownership)
    if result is not None:
        if not all(cleanup[key] for key in ("fixture_dead", "backend_closed", "owned_browser_state_removed", "focus_restore_safe")):
            failure = RuntimeError(f"trial cleanup failed: {cleanup}")
            setattr(failure, "cleanup", cleanup)
            raise failure
        result["cleanup"] = cleanup
        return result
    if isinstance(failure, asyncio.CancelledError):
        failure = TimeoutError("trial cleanup cancelled")
    setattr(failure, "cleanup", cleanup)
    raise failure


async def _close_fixture(fixture: Any) -> bool:
    close = getattr(fixture, "close_and_verify", None)
    if close is not None:
        result = close()
        if hasattr(result, "__await__"):
            result = await result
        return bool(result)
    process = fixture
    if process.returncode is None:
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), 5)
        except TimeoutError:
            process.kill()
            await process.wait()
    return process.returncode is not None


async def pilot(
    run_dir: Path,
    trials: int = 3,
    *,
    provider_factory: Callable[[int], ProviderAdapter] | None = None,
    backend_factory: Callable[[int], Any] | None = None,
    fixture_factory: Callable[[Path, int], Any] | None = None,
    browser_factory: Callable[[], BrowserAdapter] | None = None,
    timeout_factory: Callable[[float], Any] = asyncio.timeout,
) -> dict[str, Any]:
    if trials != 3:
        raise ValueError("the pilot is exactly three trials")
    with prepare_home(run_dir):
        return await _pilot(run_dir, trials, provider_factory=provider_factory, backend_factory=backend_factory, fixture_factory=fixture_factory, browser_factory=browser_factory, timeout_factory=timeout_factory)


async def _pilot(
    run_dir: Path,
    trials: int,
    *,
    provider_factory: Callable[[int], ProviderAdapter] | None,
    backend_factory: Callable[[int], Any] | None,
    fixture_factory: Callable[[Path, int], Any] | None,
    browser_factory: Callable[[], BrowserAdapter] | None,
    timeout_factory: Callable[[float], Any],
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    trial_cleanup: list[CleanupEvidence] = []
    status = "failed"
    error: str | None = None
    try:
        async with timeout_factory(PILOT_DEADLINE):
            for trial in range(1, 4):
                provider = provider_factory(trial) if provider_factory else _production_provider(run_dir)
                backend = backend_factory(trial) if backend_factory else _production_backend()
                browser = browser_factory() if browser_factory else (MacSafari() if platform.system() == "Darwin" else None)
                async with timeout_factory(TRIAL_DEADLINE):
                    results.append(await run_trial(run_dir, trial, provider, backend, fixture_factory=fixture_factory, browser=browser))
                    trial_cleanup.append(results[-1]["cleanup"])
        status = "complete"
    except asyncio.CancelledError as exc:
        status, error = "cancelled", type(exc).__name__
        cleanup = getattr(exc, "cleanup", None)
        if cleanup is not None:
            trial_cleanup.append(cleanup)
        raise
    except TimeoutError as exc:
        status, error = "timeout", type(exc).__name__
        cleanup = getattr(exc, "cleanup", None)
        if cleanup is not None:
            trial_cleanup.append(cleanup)
        raise
    except Exception as exc:
        error = str(exc) or type(exc).__name__
        cleanup = getattr(exc, "cleanup", None)
        if cleanup is not None:
            trial_cleanup.append(cleanup)
        raise
    finally:
        cleanup = {
            "fixture_dead": all(item["fixture_dead"] for item in trial_cleanup),
            "backend_closed": all(item["backend_closed"] for item in trial_cleanup),
            "owned_browser_state_removed": all(item["owned_browser_state_removed"] for item in trial_cleanup),
            "prior_state_restored": all(item["prior_state_restored"] for item in trial_cleanup),
            "focus_restore_safe": all(item["focus_restore_safe"] for item in trial_cleanup),
            "errors": [item["error"] for item in trial_cleanup if item["error"]],
            "disposable_home_removed": False,
        }
        if not trial_cleanup:
            cleanup.update({"fixture_dead": False, "backend_closed": False, "owned_browser_state_removed": False, "prior_state_restored": False, "focus_restore_safe": False})
        home = run_dir / "zeta-home"
        try:
            if home.exists():
                shutil.rmtree(home)
            cleanup["disposable_home_removed"] = not home.exists()
        except OSError as exc:
            cleanup["home_error"] = str(exc)
        if not all(value for key, value in cleanup.items() if key in {"disposable_home_removed"}):
            status = "failed"
        payload = {"status": status, "error": error, "provider": EXPECTED_PROVIDER, "model": EXPECTED_MODEL, "trials": results if status == "complete" else [], "cleanup": cleanup}
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
    args = parser.parse_args()
    asyncio.run(phase0(args.run_dir) if args.mode == "phase0" else pilot(args.run_dir))


if __name__ == "__main__":
    main()
