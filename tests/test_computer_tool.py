"""Tests for the opt-in ``computer`` tool and its backend seam (slice 1)."""

from __future__ import annotations

import asyncio
import base64
import json
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from textwrap import dedent

import pytest

from tests.support.owned import await_or_fail
from zeta.agent.tool_results import validated_tool_result
from zeta.config.settings import (
    SETTINGS_FILENAME,
    ComputerUseConfig,
    computer_allow_guidance,
    load_settings,
    resolve,
)
from zeta.core.store import ConversationStore
from zeta.protocol.types import Message, MessageRole, ToolCall
from zeta.providers.anthropic import build_messages_payload
from zeta.providers.codex import build_responses_payload
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry
from zeta.tools import computer as computer_module
from zeta.tools.computer.backend import ComputerError, ScreenState
from zeta.tools.computer.fake import FakeBackend
from zeta.tools.computer.scaling import ScreenGeometry


@asynccontextmanager
async def _computer_registry(
    tmp_path: Path, *, enabled: bool = True, backend: FakeBackend | None = None
):
    registry = ToolRegistry(
        tmp_path, skill_catalog=SkillCatalog.empty(), computer=ComputerUseConfig(enabled=enabled, default_app_access="allow")
    )
    if backend is not None:
        registry.computer_backend = backend
    try:
        yield registry
    finally:
        await registry.close()


def _ops(backend: FakeBackend) -> list[tuple]:
    return [action for action in backend.actions if action[0] != "state"]


_ACTION_CASES = [
    ({"action": "screenshot"}, [("state",), ("screenshot", None)]),
    ({"action": "left_click", "coordinate": [10, 20]}, [("state",), ("click", 10, 20, "left", 1), ("state",), ("screenshot", None)]),
    ({"action": "right_click", "coordinate": [1, 2]}, [("state",), ("click", 1, 2, "right", 1), ("state",), ("screenshot", None)]),
    ({"action": "middle_click", "coordinate": [2, 3]}, [("state",), ("click", 2, 3, "middle", 1), ("state",), ("screenshot", None)]),
    ({"action": "double_click", "coordinate": [3, 4]}, [("state",), ("click", 3, 4, "left", 2), ("state",), ("screenshot", None)]),
    ({"action": "triple_click", "coordinate": [5, 6]}, [("state",), ("click", 5, 6, "left", 3), ("state",), ("screenshot", None)]),
    ({"action": "mouse_move", "coordinate": [7, 8]}, [("state",), ("move", 7, 8), ("state",), ("screenshot", None)]),
    ({"action": "left_click_drag", "start_coordinate": [0, 0], "coordinate": [4, 5]}, [("state",), ("drag", [(0, 0), (4, 5)]), ("state",), ("screenshot", None)]),
    ({"action": "left_mouse_down", "coordinate": [9, 1]}, [("state",), ("mouse_down", 9, 1, "left"), ("state",), ("screenshot", None)]),
    ({"action": "left_mouse_up", "coordinate": [2, 3]}, [("state",), ("mouse_up", 2, 3, "left"), ("state",), ("screenshot", None)]),
    ({"action": "type", "text": "hi"}, [("state",), ("type_text", "hi"), ("state",), ("screenshot", None)]),
    ({"action": "key", "text": "cmd+s"}, [("state",), ("key", "cmd+s"), ("state",), ("screenshot", None)]),
    ({"action": "scroll", "coordinate": [1, 1], "scroll_direction": "down", "scroll_amount": 3}, [("state",), ("scroll", 1, 1, 0, 3), ("state",), ("screenshot", None)]),
    ({"action": "cursor_position"}, [("state",), ("cursor_position",)]),
    ({"action": "wait"}, [("state",), ("state",), ("screenshot", None)]),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("arguments", "expected_actions"), _ACTION_CASES, ids=[row[0]["action"] for row in _ACTION_CASES])
async def test_computer_action_acceptance_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arguments: dict, expected_actions: list[tuple]) -> None:
    original_sleep = asyncio.sleep

    async def yielding_sleep(_delay: float) -> None:
        await original_sleep(0)

    monkeypatch.setattr(computer_module.asyncio, "sleep", yielding_sleep)
    backend = FakeBackend(cursor=(42, 7))
    config = ComputerUseConfig(enabled=True, bundle_ids={"com.example.app": "allow"})
    async with _computer_registry(tmp_path, backend=backend) as registry:
        registry.computer_config = config
        registry.computer_geometry = ScreenGeometry(100, 100, 100, 100, 1, 0, 0)
        result = await registry.execute(ToolCall("case", "computer", arguments))

    action = arguments["action"]
    has_screenshot = action != "cursor_position"
    assert result["isError"] is False
    assert result["structuredContent"]["computer"] == {
        "action": action,
        "control": "idle",
        "kill_switch": "ctrl+opt+cmd+esc",
        "frontmost_bundle_id": "com.example.app",
        "has_screenshot": has_screenshot,
    }
    assert backend.actions == expected_actions
    assert Counter(op[0] for op in backend.actions) == Counter(op[0] for op in expected_actions)
    assert any(block["type"] == "image" for block in result["content"]) is has_screenshot
    if action == "cursor_position":
        assert result["structuredContent"] == {
            "computer": result["structuredContent"]["computer"],
            "x": 42,
            "y": 7,
        }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "arguments", "frontmost", "control", "is_error", "expected_actions", "message"),
    [
        ("no_backend", {"action": "screenshot"}, None, "idle", True, [], "no computer backend"),
        ("absent_bundle", {"action": "screenshot"}, None, "denied", True, [("state",)], "not allowed"),
        ("empty_bundle", {"action": "screenshot"}, "", "denied", True, [("state",)], "not allowed"),
        ("unknown_bundle", {"action": "screenshot"}, "com.unknown.app", "denied", True, [("state",)], "not allowed"),
        ("no_prefix", {"action": "screenshot"}, "com.example.app.beta", "denied", True, [("state",)], "not allowed"),
        ("locked", {"action": "screenshot"}, "com.example.app", "idle", True, [("state",)], "locked"),
        ("secure", {"action": "screenshot"}, "com.example.app", "idle", True, [("state",)], "secure input"),
        ("killed", {"action": "left_click", "coordinate": [1, 2]}, "com.example.app", "killed", True, [("state",), ("click", 1, 2, "left", 1)], "kill switch"),
        ("backend_error", {"action": "left_click", "coordinate": [1, 2]}, "com.example.app", "idle", True, [("state",), ("click", 1, 2, "left", 1)], "input failed"),
        ("focus_change", {"action": "left_click", "coordinate": [1, 2]}, "com.denied.app", "denied", False, [("state",), ("click", 1, 2, "left", 1), ("state",)], "screenshot failed"),
        ("capture_failure", {"action": "left_click", "coordinate": [1, 2]}, "com.changed.app", "idle", False, [("state",), ("click", 1, 2, "left", 1), ("state",), ("screenshot", None)], "done; screenshot failed"),
        ("optional_coordinate", {"action": "left_click"}, "com.example.app", "idle", False, [("state",), ("cursor_position",), ("click", 0, 0, "left", 1), ("state",), ("screenshot", None)], "screenshot"),
    ],
)
async def test_computer_exit_acceptance_table(tmp_path: Path, case: str, arguments: dict, frontmost: str | None, control: str, is_error: bool, expected_actions: list[tuple], message: str) -> None:
    class CaseBackend(FakeBackend):
        async def state(self) -> ScreenState:
            self.actions.append(("state",))
            if case in {"focus_change", "capture_failure"} and any(op[0] == "click" for op in self.actions):
                self.frontmost_bundle_id = "com.denied.app" if case == "focus_change" else "com.changed.app"
            return ScreenState(self.locked, self.secure_input, self.frontmost_bundle_id)

        async def click(self, x: int, y: int, button: str, count: int) -> None:
            self.actions.append(("click", x, y, button, count))
            if case == "killed":
                raise ComputerError("killed", "kill switch engaged")
            if case == "backend_error":
                raise ComputerError("internal", "input failed")

        async def screenshot(self, display: int | None = None):
            if case == "capture_failure":
                self.actions.append(("screenshot", display))
                raise ComputerError("not_permitted", "screen recording denied")
            return await super().screenshot(display)

    initial_bundle = "com.example.app" if case in {"focus_change", "capture_failure"} else frontmost
    backend = None if case == "no_backend" else CaseBackend(frontmost_bundle_id=initial_bundle, locked=case == "locked", secure_input=case == "secure")
    config = ComputerUseConfig(enabled=True, bundle_ids={"com.example.app": "allow", "com.changed.app": "allow"})
    async with _computer_registry(tmp_path, backend=backend) as registry:
        registry.computer_config = config
        registry.computer_geometry = ScreenGeometry(100, 100, 100, 100, 1, 0, 0)
        result = await registry.execute(ToolCall(case, "computer", arguments))

    action = arguments["action"] if isinstance(arguments["action"], str) else "action"
    assert result["isError"] is is_error
    assert result["structuredContent"]["computer"] == {
        "action": action,
        "control": control,
        "kill_switch": "ctrl+opt+cmd+esc",
        "frontmost_bundle_id": frontmost,
        "has_screenshot": bool(expected_actions) and not (case == "capture_failure") and expected_actions[-1][0] == "screenshot",
    }
    actual_actions = [] if backend is None else backend.actions
    assert actual_actions == expected_actions
    assert Counter(op[0] for op in actual_actions) == Counter(op[0] for op in expected_actions)
    assert message in result["content"][0]["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {"action": "bogus"},
        {"action": 7},
        {"action": "screenshot", "coordinate": "not-an-array"},
        {"action": "screenshot", "scroll_amount": 0},
    ],
)
async def test_schema_rejects_irrelevant_malformed_fields_before_desktop(
    tmp_path: Path, arguments: dict
) -> None:
    backend = FakeBackend()
    async with _computer_registry(tmp_path, backend=backend) as registry:
        result = await registry.execute(ToolCall("c", "computer", arguments))

    assert result["isError"] is True
    assert result["structuredContent"]["error"]["kind"] == "invalid_arguments"
    assert result["structuredContent"]["computer"] == {
        "action": arguments["action"] if isinstance(arguments["action"], str) else "action",
        "control": "idle",
        "kill_switch": "ctrl+opt+cmd+esc",
        "frontmost_bundle_id": None,
        "has_screenshot": False,
    }
    assert "invalid arguments:" in result["content"][0]["text"]
    assert backend.actions == []


@pytest.mark.asyncio
async def test_non_computer_registry_validation_error_is_byte_identical(
    tmp_path: Path,
) -> None:
    async def unused(_arguments: dict) -> dict:
        raise AssertionError("invalid arguments reached the handler")

    async with _computer_registry(tmp_path) as registry:
        registry.register(
            "schema_test",
            unused,
            parameters={
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        )
        result = await registry.execute(
            ToolCall("schema", "schema_test", {"value": "wrong"})
        )

    message = "invalid arguments: arguments.value must be integer"
    expected = {
        "content": [
            {
                "type": "text",
                "text": message,
                "truncated": False,
                "full_size": len(message.encode()),
            }
        ],
        "isError": True,
        "structuredContent": {
            "error": {
                "kind": "invalid_arguments",
                "hint": "reread the tool schema and retry with correct arguments",
                "message": message,
                "tool": "schema_test",
            }
        },
    }
    assert json.dumps(result, sort_keys=True, separators=(",", ":")).encode() == json.dumps(
        expected, sort_keys=True, separators=(",", ":")
    ).encode()


@pytest.mark.asyncio
async def test_wait_yields_before_capture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def controlled_sleep(_delay: float) -> None:
        entered.set()
        await await_or_fail(release.wait(), what="computer wait release")

    monkeypatch.setattr(computer_module.asyncio, "sleep", controlled_sleep)
    backend = FakeBackend()
    async with _computer_registry(tmp_path, backend=backend) as registry:
        task = asyncio.create_task(registry.execute(ToolCall("wait", "computer", {"action": "wait"})))
        try:
            await await_or_fail(entered.wait(), what="computer wait to yield")
            assert backend.actions == [("state",)]
            release.set()
            result = await await_or_fail(task, what="computer wait result")
        finally:
            release.set()
            if not task.done():
                task.cancel()
                await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 10)
    assert result["isError"] is False
    assert backend.actions == [("state",), ("state",), ("screenshot", None)]


@pytest.mark.asyncio
async def test_schema_actions_have_one_reachable_handler(tmp_path: Path) -> None:
    async with _computer_registry(tmp_path, backend=FakeBackend()) as registry:
        schema = next(schema for schema in registry.schemas if schema["name"] == "computer")
    schema_actions = schema["parameters"]["properties"]["action"]["enum"]
    accepted_actions = [arguments["action"] for arguments, _expected in _ACTION_CASES]
    assert schema_actions == list(computer_module._ACTION_HANDLERS)
    assert len(schema_actions) == len(set(schema_actions))
    assert set(accepted_actions) == set(schema_actions)
    assert all(callable(spec.handler) for spec in computer_module._ACTION_HANDLERS.values())


def _tool_names(registry: ToolRegistry) -> set[str]:
    return {schema["name"] for schema in registry.schemas}


@pytest.mark.asyncio
async def test_native_action_semantics_are_bounded_and_ordered(tmp_path: Path) -> None:
    backend = FakeBackend(cursor=(8, 9))
    async with _computer_registry(tmp_path, backend=backend) as registry:
        registry.computer_geometry = ScreenGeometry(100, 100, 100, 100, 1, 0, 0)
        repeat = await registry.execute(ToolCall("k", "computer", {
            "action": "key", "text": "a", "repeat": 2, "hold": True,
        }))
        drag = await registry.execute(ToolCall("d", "computer", {
            "action": "left_click_drag", "path": [[1, 2], [3, 4], [5, 6]],
        }))
        wait = await registry.execute(ToolCall("w", "computer", {
            "action": "wait", "duration": 11,
        }))
    assert repeat["isError"] is False
    assert drag["isError"] is False
    assert wait["isError"] is True
    assert ("key", "a", 2, True, False) in backend.actions
    assert ("drag", [(1, 2), (3, 4), (5, 6)]) in backend.actions
    assert "duration" in wait["content"][0]["text"]


@pytest.mark.asyncio
async def test_computer_tool_is_opt_in(tmp_path: Path) -> None:
    async with _computer_registry(tmp_path, enabled=False) as off:
        assert "computer" not in _tool_names(off)
    async with _computer_registry(tmp_path, enabled=True) as on:
        assert "computer" in _tool_names(on)


@pytest.mark.asyncio
async def test_cursor_position_edge_can_be_reused_for_click(tmp_path: Path) -> None:
    backend = FakeBackend(width=1280, height=800, cursor=(1280, 800))
    async with _computer_registry(tmp_path, backend=backend) as registry:
        await registry.execute(ToolCall("s", "computer", {"action": "screenshot"}))
        position = await registry.execute(
            ToolCall("p", "computer", {"action": "cursor_position"})
        )
        point = position["structuredContent"]
        result = await registry.execute(
            ToolCall(
                "c",
                "computer",
                {"action": "left_click", "coordinate": [point["x"], point["y"]]},
            )
        )

    assert position["isError"] is False
    assert {key: point[key] for key in ("x", "y")} == {"x": 1279, "y": 799}
    assert result["isError"] is False
    assert ("click", 1279, 799, "left", 1) in _ops(backend)


@pytest.mark.asyncio
async def test_child_registry_does_not_inherit_parent_screenshot(
    tmp_path: Path,
) -> None:
    backend = FakeBackend()
    (tmp_path / "parent").mkdir()
    parent = ToolRegistry(
        tmp_path / "parent",
        skill_catalog=SkillCatalog.empty(),
        computer=ComputerUseConfig(enabled=True, default_app_access="allow"),
    )
    child_store = ConversationStore(tmp_path / "child", cwd=tmp_path)
    child: ToolRegistry | None = None
    try:
        parent.computer_backend = backend
        await parent.execute(ToolCall("s", "computer", {"action": "screenshot"}))
        child = parent.clone_for_session(child_store)
        child_backend = FakeBackend()
        child.computer_backend = child_backend
        result = await child.execute(
            ToolCall("c", "computer", {"action": "left_click", "coordinate": [1, 2]})
        )
        assert result["isError"] is True
        assert "computer" in result["content"][0]["text"]
        assert _ops(child_backend) == []
    finally:
        if child is not None:
            await child.close()
        await parent.close()
        child_store.close()


@pytest.mark.asyncio
async def test_coordinate_action_before_screenshot_is_a_clear_error(
    tmp_path: Path,
) -> None:
    backend = FakeBackend()
    async with _computer_registry(tmp_path, backend=backend) as registry:
        result = await registry.execute(
            ToolCall("c", "computer", {"action": "left_click", "coordinate": [1, 2]})
        )

    assert result["isError"] is True
    assert "screenshot" in result["content"][0]["text"]
    assert _ops(backend) == []


@pytest.mark.asyncio
async def test_screenshot_reaches_provider_payloads_as_image(tmp_path: Path) -> None:
    async with _computer_registry(tmp_path, backend=FakeBackend()) as registry:
        raw = await registry.execute(
            ToolCall("shot", "computer", {"action": "screenshot"})
        )
    tool_result = validated_tool_result(raw, "shot")
    message = Message(MessageRole.TOOL_RESULT, tool_result=tool_result)

    anthropic = build_messages_payload(
        [message], [], model="claude-test", max_tokens=4096, thinking_budget=2048
    )
    content = anthropic["messages"][0]["content"][0]["content"]
    image = next(block for block in content if block["type"] == "image")
    assert image["source"]["type"] == "base64"
    assert base64.b64decode(image["source"]["data"]) == FakeBackend().image

    codex = build_responses_payload([message], [], model="codex-test")
    codex_image = codex["input"][1]["content"][0]
    assert codex_image["type"] == "input_image"
    assert codex_image["image_url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_click_maps_through_downscaled_retina_frame(tmp_path: Path) -> None:
    # A 2x retina display downscaled to the WXGA box; the physical/image ratio
    # (2.0) and the point scale (2.0) cancel, so model coords land at origin+xy.
    backend = FakeBackend(width=2560, height=1600, scale=2.0, origin_x=100, origin_y=50)
    async with _computer_registry(tmp_path, backend=backend) as registry:
        await registry.execute(ToolCall("s", "computer", {"action": "screenshot"}))
        geometry = registry.computer_geometry
        assert (geometry.image_width, geometry.image_height) == (1280, 800)
        result = await registry.execute(
            ToolCall("c", "computer", {"action": "left_click", "coordinate": [640, 400]})
        )

    assert result["isError"] is False
    assert ("click", 740, 450, "left", 1) in _ops(backend)


@pytest.mark.asyncio
async def test_out_of_bounds_coordinate_is_rejected(tmp_path: Path) -> None:
    backend = FakeBackend(width=1280, height=800)
    async with _computer_registry(tmp_path, backend=backend) as registry:
        await registry.execute(ToolCall("s", "computer", {"action": "screenshot"}))
        result = await registry.execute(
            ToolCall("c", "computer", {"action": "left_click", "coordinate": [2000, 10]})
        )

    assert result["isError"] is True
    assert "outside" in result["content"][0]["text"]
    assert all(action[0] != "click" for action in _ops(backend))


def _write_settings(base: Path, body: str) -> None:
    base.mkdir(parents=True, exist_ok=True)
    (base / SETTINGS_FILENAME).write_text(dedent(body).lstrip(), encoding="utf-8")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("table", "existing"),
    [
        ("", {}),
        ("[computer.macos.bundle_ids]\n", {}),
        ('[computer.macos.bundle_ids]\n"com.apple.finder" = "allow"\n', {"com.apple.finder": "allow"}),
    ],
    ids=["missing", "empty", "nonempty"],
)
async def test_denied_app_guidance_is_parseable_and_allows_app(
    tmp_path: Path, table: str, existing: dict[str, str]
) -> None:
    home = tmp_path / "home"
    _write_settings(home, "[computer]\nenabled = true\n" + table)
    initial = load_settings(home=home).settings.computer
    backend = FakeBackend(frontmost_bundle_id="com.apple.Safari")
    async with _computer_registry(tmp_path, backend=backend) as registry:
        registry.computer_config = initial
        denied = await registry.execute(ToolCall("c", "computer", {"action": "screenshot"}))
        text = denied["content"][0]["text"]
        assert text.endswith(
            computer_allow_guidance(
                "com.apple.Safari",
                has_existing_table=initial.bundle_ids_table_present,
            )
        )
        suggestion = text.split("settings.toml:\n", 1)[1]
        settings_path = home / SETTINGS_FILENAME
        settings_path.write_text(
            settings_path.read_text(encoding="utf-8") + suggestion + "\n",
            encoding="utf-8",
        )
        registry.computer_config = resolve(
            load_settings(home=home).settings,
            cli_provider="claude",
            cli_model=None,
            cli_yolo=None,
            cli_token_budget=None,
        ).computer
        assert registry.computer_config.bundle_ids == existing | {"com.apple.Safari": "allow"}
        allowed = await registry.execute(ToolCall("c", "computer", {"action": "screenshot"}))
    assert allowed["isError"] is False


def _resolved_computer_enabled(home: Path, project: Path | None = None) -> bool:
    settings = load_settings(home=home, project_dir=project).settings
    return resolve(
        settings,
        cli_provider="claude",
        cli_model=None,
        cli_yolo=None,
        cli_token_budget=None,
    ).computer_enabled


def test_computer_defaults_disabled(tmp_path: Path) -> None:
    assert _resolved_computer_enabled(tmp_path / "empty") is False


def test_home_can_enable_computer(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_settings(home, "[computer]\nenabled = true\n")
    assert _resolved_computer_enabled(home) is True


def test_project_cannot_enable_computer(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    _write_settings(home, 'provider = "claude"\n')
    _write_settings(project, "[computer]\nenabled = true\n")
    assert _resolved_computer_enabled(home, project) is False

@pytest.mark.asyncio
async def test_plan_1_missing_config_omits_schema_and_backend(tmp_path: Path) -> None:
    async with _computer_registry(tmp_path, enabled=False) as registry:
        assert "computer" not in _tool_names(registry)
        assert registry.computer_backend is None


def test_plan_2_project_settings_cannot_enable_or_broaden(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    _write_settings(home, """
        [computer]
        enabled = true
    """)
    _write_settings(project, """
        [computer]
        enabled = true
        default_app_access = "allow"
    """)
    settings = load_settings(home=home, project_dir=project).settings
    assert settings.computer.enabled is True
    assert settings.computer.default_app_access == "deny"


@pytest.mark.asyncio
async def test_plan_7_child_registry_has_no_computer_capability(tmp_path: Path) -> None:
    async with _computer_registry(tmp_path, backend=FakeBackend()) as registry:
        child_store = ConversationStore(tmp_path / "child")
        child = registry.clone_for_session(child_store)
        try:
            assert "computer" not in _tool_names(child)
        finally:
            await child.close()
            child_store.close()


def test_plan_8_settings_reject_disabled_safety_controls(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_settings(home, """
        [computer]
        enabled = true
        indicator = false
        kill_switch = false
    """)
    loaded = load_settings(home=home).settings
    assert loaded.computer.indicator is True
    assert loaded.computer.kill_switch is True
