"""MacBackend translation of the helper protocol, and the tool's per-action error messages."""

from __future__ import annotations

import asyncio
import dataclasses
import sys
import threading
from pathlib import Path

import pytest

from zeta.config.settings import ComputerUseConfig, load_settings, resolve
from zeta.core.project_context import ProjectContext
from zeta.core.session import SessionManager
from zeta.core.slash import UsageTracker, build_slash_status
from zeta.protocol.types import ToolCall
from zeta.runtime.cleanup import close_session
from zeta.runtime.composition import compose_runtime
from zeta.runtime.status import capture_slash_status
from zeta.server.runtime import ServerRuntime
from zeta.server.slash_commands import run_command
from zeta.skills import SkillCatalog
from zeta.skills.agent_catalog import AgentCatalog
from zeta.tools import ToolRegistry
from zeta.tools.computer import mac as mac_module
from zeta.tools.computer.backend import Button, ComputerError, ScreenFrame, ScreenState
from zeta.tools.computer.mac import MacBackend
from zeta.tools.computer.scaling import ScreenGeometry
from zeta.tui.app import _status_card_lines

FAKE = Path(__file__).parent / "fixtures" / "computer_fake_helper.py"


def make_backend() -> MacBackend:
    return MacBackend([sys.executable, str(FAKE)])


@pytest.mark.asyncio
async def test_repeated_cancellation_drains_one_cleanup_and_restarts() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.release = threading.Event()
            self.started = 0
            self.cancelled = 0
            self.closed = 0

        def start(self) -> None:
            self.started += 1

        def request(self, _op: str, **_args: object) -> dict:
            self.release.wait(10)
            return {}

        def cancel(self) -> None:
            self.cancelled += 1
            self.release.set()

        def close(self) -> None:
            self.closed += 1

    client = FakeClient()
    backend = MacBackend([])
    backend._client = client  # type: ignore[assignment]
    call = asyncio.create_task(backend._call("move", x=1, y=2))
    await asyncio.sleep(0)
    call.cancel()
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(call, 10)
    assert (client.cancelled, client.closed, backend._started) == (1, 1, False)

    await asyncio.wait_for(backend._call("move", x=1, y=2), 10)
    assert client.started == 2


def _compose(tmp_path: Path, computer: ComputerUseConfig):
    config = resolve(
        load_settings(home=tmp_path / "home").settings,
        cli_provider="codex",
        cli_model="offline",
        cli_yolo=None,
        cli_token_budget=None,
    )
    return compose_runtime(
        home=tmp_path / "home",
        cwd=tmp_path,
        manager=SessionManager(tmp_path / "home"),
        config=dataclasses.replace(config, computer=computer),
        provider="codex",
        model="offline",
        project_context=ProjectContext("system", ()),
        backend_builder=lambda *_args, **_kwargs: (object(), "offline"),
        skill_catalog=SkillCatalog.empty(),
        agent_catalog=AgentCatalog.empty(),
    )


@pytest.mark.asyncio
async def test_composition_closes_helper_on_session_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = make_backend()
    monkeypatch.setattr(mac_module, "make_mac_backend", lambda _config=None: backend)
    runtime = await asyncio.to_thread(
        _compose, tmp_path, ComputerUseConfig(enabled=True, default_app_access="allow")
    )
    try:
        await backend.state()
        assert backend._client._proc is not None
    finally:
        await close_session(runtime.loop)
    assert backend._client._proc is None


@pytest.mark.asyncio
async def test_status_names_empty_allow_list_with_working_settings_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mac_module, "make_mac_backend", lambda _config=None: None)
    runtime = await asyncio.to_thread(_compose, tmp_path, ComputerUseConfig(enabled=True))
    try:
        loop = runtime.loop
        snapshot = capture_slash_status(
            loop,
            session_id=runtime.opened.metadata.session_id,
            provider="codex",
            model="offline",
            context_files=(),
            vim_mode=False,
            pending_approvals=(),
            usage_tracker=UsageTracker(loop.context_assembler, provider="codex"),
            project_context_health=None,
        )
        tui_text = "\n".join(
            "".join(text for _style, text in line)
            for line in _status_card_lines(build_slash_status(snapshot))
        )
        serve = ServerRuntime.borrowed(
            home=tmp_path / "home",
            opened=runtime.opened,
            loop=loop,
            policy=runtime.policy,
            project_dir=runtime.project_dir,
            repo_root=runtime.repo_root,
        )
        result = await run_command(serve, "/status")
    finally:
        await close_session(runtime.loop)
    assert result["kind"] == "output"
    for text in (tui_text, result["text"]):
        assert "computer: enabled, allowed app count=0; the allow-list is empty" in text
        assert '[computer.macos.bundle_ids]\n"com.apple.Safari" = "allow"' in text


@pytest.mark.asyncio
async def test_screenshot_decodes_frame() -> None:
    backend = make_backend()
    try:
        frame = await backend.screenshot()
    finally:
        backend.close()
    assert isinstance(frame, ScreenFrame)
    assert frame.media_type == "image/png"
    assert frame.width == 1 and frame.height == 1 and frame.scale == 2.0
    assert frame.image.startswith(b"\x89PNG")


@pytest.mark.asyncio
async def test_state_and_cursor_round_trip() -> None:
    backend = make_backend()
    try:
        state = await backend.state()
        cursor = await backend.cursor_position()
    finally:
        backend.close()
    assert state == ScreenState(False, False, "dev.zeta.fake")
    assert cursor == (12, 34)


@pytest.mark.asyncio
async def test_helper_error_maps_to_computer_error() -> None:
    backend = make_backend()
    try:
        with pytest.raises(ComputerError) as excinfo:
            await backend.screenshot(99)  # fixture sentinel -> secure_input error
    finally:
        backend.close()
    assert excinfo.value.code == "secure_input"


class _RefusingBackend:
    """A backend whose click is refused, to check the tool's per-action message."""

    async def screenshot(self, display: int | None = None) -> ScreenFrame:
        raise AssertionError("not called")

    async def click(self, x: int, y: int, button: Button, count: int) -> None:
        raise ComputerError("locked", "screen is locked")

    async def state(self) -> ScreenState:
        return ScreenState(False, False, "com.example.app")


@pytest.mark.asyncio
async def test_tool_reports_action_specific_refusal(tmp_path: Path) -> None:
    registry = ToolRegistry(
        tmp_path,
        skill_catalog=SkillCatalog.empty(),
        computer=ComputerUseConfig(enabled=True, default_app_access="allow"),
    )
    registry.computer_backend = _RefusingBackend()  # type: ignore[assignment]
    registry.computer_geometry = ScreenGeometry(100, 100, 100, 100, 1.0, 0, 0)
    try:
        result = await registry.execute(
            ToolCall("c", "computer", {"action": "left_click", "coordinate": [1, 2]})
        )
    finally:
        await registry.close()
    assert result["isError"] is True
    text = result["content"][0]["text"]
    assert "left_click failed" in text
    assert "screen is locked" in text
