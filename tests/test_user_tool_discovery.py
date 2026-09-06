"""ZETA-77 external tool discovery tests."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from zeta.core.approval import ApprovalDecision, ApprovalPolicy
from zeta.core.fake import FakeBackend
from zeta.core.slash import create_slash_registry
from zeta.core.store import ConversationStore
from zeta.loop import AgentLoop
from zeta.tools import ToolRegistry
from zeta.tools._user_discovery import (
    apply_external_tools,
    trust_project_tools,
)
from zeta.tui.app import TUIApp
from zeta.types import ToolCall


def _echo_source(name: str, description: str = "echo tool") -> str:
    return (
        "async def _handle(arguments, abort_signal):\n"
        "    message = arguments['message']\n"
        "    return {\n"
        "        'content': [{'type': 'text', 'text': message,\n"
        "                     'truncated': False,\n"
        "                     'full_size': len(message.encode('utf-8'))}],\n"
        "        'isError': False,\n"
        "        'structuredContent': {'message': message},\n"
        "    }\n"
        "\n"
        "def register(registry):\n"
        f"    registry.register({name!r}, _handle, description={description!r},\n"
        "        parameters={\n"
        "            'type': 'object',\n"
        "            'properties': {'message': {'type': 'string'}},\n"
        "            'required': ['message'],\n"
        "            'additionalProperties': False,\n"
        "        })\n"
    )


def _write_tool(directory: Path, filename: str, source: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(source, encoding="utf-8")
    return path


def _new_registry(tmp_path: Path) -> ToolRegistry:
    (tmp_path / "cwd").mkdir(exist_ok=True)
    return ToolRegistry(tmp_path / "cwd")


@pytest.mark.asyncio
async def test_user_tool_registers_and_executes(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_tool(home / "tools", "echo_user.py", _echo_source("user_echo"))
    registry = _new_registry(tmp_path)

    discovery = apply_external_tools(
        registry, home=home, project_dir=tmp_path / "project" / ".zeta"
    )

    assert discovery.warnings == ()
    assert discovery.pending_project_tools == ()
    assert "user_echo" in registry.registered_names
    result = await registry.execute(
        ToolCall("user-1", "user_echo", {"message": "hi"})
    )
    assert result["isError"] is False
    assert result["structuredContent"] == {"message": "hi"}


def test_user_tool_shadows_builtin_with_notice(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_tool(home / "tools", "override_read.py", _echo_source("read"))
    registry = _new_registry(tmp_path)

    discovery = apply_external_tools(
        registry, home=home, project_dir=tmp_path / "project" / ".zeta"
    )

    assert any("shadows built-in" in notice for notice in discovery.notices)
    assert any("read" in notice for notice in discovery.notices)


def test_malformed_user_tool_fails_open(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_tool(
        home / "tools", "broken.py", "raise RuntimeError('boom')\n"
    )
    _write_tool(home / "tools", "good.py", _echo_source("good_echo"))
    registry = _new_registry(tmp_path)

    discovery = apply_external_tools(
        registry, home=home, project_dir=None
    )

    assert "good_echo" in registry.registered_names
    assert any("broken.py" in notice for notice in discovery.notices)
    assert any("boom" in notice for notice in discovery.notices)


def test_user_tool_missing_register_is_reported(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_tool(home / "tools", "no_register.py", "value = 42\n")
    registry = _new_registry(tmp_path)

    discovery = apply_external_tools(
        registry, home=home, project_dir=None
    )

    assert any(
        "no callable register" in notice for notice in discovery.notices
    )


def test_underscore_and_init_modules_are_ignored(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_tool(
        home / "tools",
        "_helper.py",
        "raise AssertionError('helper must not be imported')\n",
    )
    _write_tool(
        home / "tools",
        "__init__.py",
        "raise AssertionError('init must not run')\n",
    )
    registry = _new_registry(tmp_path)

    discovery = apply_external_tools(
        registry, home=home, project_dir=None
    )

    assert discovery.notices == ()


def test_project_tools_are_pending_until_trust(tmp_path: Path) -> None:
    project = tmp_path / "project" / ".zeta"
    _write_tool(project / "tools", "proj_echo.py", _echo_source("proj_echo"))
    registry = _new_registry(tmp_path)

    discovery = apply_external_tools(
        registry, home=tmp_path / "home", project_dir=project
    )

    assert "proj_echo" not in registry.registered_names
    assert len(discovery.pending_project_tools) == 1
    assert discovery.pending_project_tools[0].module_stem == "proj_echo"
    assert any(
        "untrusted project tools present" in warning
        for warning in discovery.warnings
    )
    assert any(
        "proj_echo" in warning and "/tools trust" in warning
        for warning in discovery.warnings
    )


@pytest.mark.asyncio
async def test_project_tool_not_executable_pre_trust(tmp_path: Path) -> None:
    project = tmp_path / "project" / ".zeta"
    _write_tool(project / "tools", "proj_echo.py", _echo_source("proj_echo"))
    registry = _new_registry(tmp_path)
    apply_external_tools(
        registry, home=None, project_dir=project
    )

    result = await registry.execute(
        ToolCall("call-1", "proj_echo", {"message": "hi"})
    )

    assert result["isError"] is True
    assert result["structuredContent"]["error"]["kind"] == "unknown_tool"


@pytest.mark.asyncio
async def test_trust_registers_project_tools(tmp_path: Path) -> None:
    project = tmp_path / "project" / ".zeta"
    _write_tool(project / "tools", "proj_echo.py", _echo_source("proj_echo"))
    registry = _new_registry(tmp_path)
    discovery = apply_external_tools(
        registry, home=None, project_dir=project
    )
    assert "proj_echo" not in registry.registered_names

    notices, trusted = trust_project_tools(registry, discovery)

    assert notices == ()
    assert len(trusted) == 1
    assert "proj_echo" in registry.registered_names
    assert discovery.pending_project_tools == ()
    result = await registry.execute(
        ToolCall("call-2", "proj_echo", {"message": "trusted"})
    )
    assert result["isError"] is False
    assert result["structuredContent"] == {"message": "trusted"}


def test_project_tool_shadows_user_after_trust(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project" / ".zeta"
    _write_tool(home / "tools", "shared.py", _echo_source("shared"))
    _write_tool(project / "tools", "shared.py", _echo_source("shared"))
    registry = _new_registry(tmp_path)

    discovery = apply_external_tools(
        registry, home=home, project_dir=project
    )

    assert "shared" in registry.registered_names
    assert len(discovery.pending_project_tools) == 1

    notices, _trusted = trust_project_tools(registry, discovery)

    assert any(
        "shadows user tool" in notice and "shared" in notice
        for notice in notices
    )
    assert not any("shadows built-in" in notice for notice in notices)


def test_project_tool_cannot_replace_builtin(tmp_path: Path) -> None:
    project = tmp_path / "project" / ".zeta"
    _write_tool(project / "tools", "shadow.py", _echo_source("read"))
    registry = _new_registry(tmp_path)
    builtin_read = registry._tools["read"]
    discovery = apply_external_tools(
        registry, home=None, project_dir=project
    )

    notices, _trusted = trust_project_tools(registry, discovery)

    assert any(
        "REJECTED" in notice and "read" in notice and "shadow.py" in notice
        for notice in notices
    ), notices
    assert registry._tools["read"] is builtin_read
    assert not any("shadows built-in" in notice for notice in notices)


def test_project_tool_reject_still_registers_other_tools(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project" / ".zeta"
    _write_tool(
        project / "tools",
        "helpers.py",
        (
            "async def _run(arguments, abort_signal):\n"
            "    return {\n"
            "        'content': [{'type': 'text', 'text': 'ok',\n"
            "                     'truncated': False, 'full_size': 2}],\n"
            "        'isError': False,\n"
            "        'structuredContent': {},\n"
            "    }\n"
            "\n"
            "def register(registry):\n"
            "    registry.register('read', _run)\n"
            "    registry.register('project_only', _run,\n"
            "        parameters={'type': 'object', 'properties': {},\n"
            "                    'additionalProperties': False})\n"
        ),
    )
    registry = _new_registry(tmp_path)
    builtin_read = registry._tools["read"]

    discovery = apply_external_tools(
        registry, home=None, project_dir=project
    )
    notices, _trusted = trust_project_tools(registry, discovery)

    assert "project_only" in registry.registered_names
    assert registry._tools["read"] is builtin_read
    assert any(
        "REJECTED" in notice and "read" in notice for notice in notices
    )


def test_project_tool_reject_leaves_always_allow_policy_intact(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project" / ".zeta"
    _write_tool(project / "tools", "shadow.py", _echo_source("read"))
    store = ConversationStore(tmp_path / "sessions", session_id="s-77-reject")
    policy = ApprovalPolicy(
        store=store,
        default=ApprovalDecision.ASK,
        always_allow=("read",),
    )
    (tmp_path / "cwd").mkdir(exist_ok=True)
    registry = ToolRegistry(tmp_path / "cwd", approval_policy=policy)
    registry.bind_session_store(store)
    builtin_read = registry._tools["read"]

    discovery = apply_external_tools(
        registry, home=None, project_dir=project
    )
    notices, _trusted = trust_project_tools(registry, discovery)

    assert registry._tools["read"] is builtin_read
    assert any("REJECTED" in notice for notice in notices)


def test_failed_project_register_preserves_prior_user_tool(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project" / ".zeta"
    _write_tool(home / "tools", "shared.py", _echo_source("shared"))
    _write_tool(
        project / "tools",
        "bad.py",
        (
            "async def _proj(arguments, abort_signal):\n"
            "    return {\n"
            "        'content': [{'type': 'text', 'text': 'p',\n"
            "                     'truncated': False, 'full_size': 1}],\n"
            "        'isError': False,\n"
            "        'structuredContent': {},\n"
            "    }\n"
            "\n"
            "def register(registry):\n"
            "    registry.register('shared', _proj,\n"
            "        parameters={'type': 'object',\n"
            "                    'properties': {'message': {'type': 'string'}},\n"
            "                    'required': ['message'],\n"
            "                    'additionalProperties': False})\n"
            "    raise RuntimeError('boom')\n"
        ),
    )
    registry = _new_registry(tmp_path)
    discovery = apply_external_tools(
        registry, home=home, project_dir=project
    )
    user_shared = registry._tools["shared"]
    trust_notices, _ = trust_project_tools(registry, discovery)

    assert "shared" in registry.registered_names
    assert registry._tools["shared"] is user_shared
    assert any(
        "bad.py" in notice and "register() raised" in notice
        for notice in trust_notices
    )


def test_failed_project_register_after_rejected_builtin_preserves_state(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project" / ".zeta"
    _write_tool(
        project / "tools",
        "bad.py",
        (
            "async def _proj(arguments, abort_signal):\n"
            "    return {\n"
            "        'content': [{'type': 'text', 'text': 'p',\n"
            "                     'truncated': False, 'full_size': 1}],\n"
            "        'isError': False,\n"
            "        'structuredContent': {},\n"
            "    }\n"
            "\n"
            "def register(registry):\n"
            "    registry.register('read', _proj)\n"
            "    raise RuntimeError('boom')\n"
        ),
    )
    registry = _new_registry(tmp_path)
    builtin_read = registry._tools["read"]
    discovery = apply_external_tools(
        registry, home=None, project_dir=project
    )

    notices, _ = trust_project_tools(registry, discovery)

    assert registry._tools["read"] is builtin_read
    assert any("REJECTED" in notice and "read" in notice for notice in notices)
    assert any("bad.py" in notice and "register() raised" in notice for notice in notices)


def test_project_tool_shadowing_earlier_project_tool_notices(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project" / ".zeta"
    _write_tool(
        project / "tools", "a_first.py", _echo_source("dupe", description="a")
    )
    _write_tool(
        project / "tools", "b_second.py", _echo_source("dupe", description="b")
    )
    registry = _new_registry(tmp_path)
    discovery = apply_external_tools(
        registry, home=None, project_dir=project
    )

    notices, _trusted = trust_project_tools(registry, discovery)

    assert any(
        "shadows project tool" in notice
        and "dupe" in notice
        and "b_second" in notice
        for notice in notices
    ), notices


def test_malformed_project_tool_fails_open_at_trust(tmp_path: Path) -> None:
    project = tmp_path / "project" / ".zeta"
    _write_tool(project / "tools", "broken.py", "raise RuntimeError('boom')\n")
    _write_tool(project / "tools", "ok.py", _echo_source("ok_echo"))
    registry = _new_registry(tmp_path)
    discovery = apply_external_tools(
        registry, home=None, project_dir=project
    )
    assert len(discovery.pending_project_tools) == 2

    notices, _trusted = trust_project_tools(registry, discovery)

    assert "ok_echo" in registry.registered_names
    assert "broken" not in registry.registered_names
    assert any("broken.py" in notice for notice in notices)


def test_import_error_isolation_user_scope(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_tool(
        home / "tools", "a_break.py", "import _definitely_missing_pkg_\n"
    )
    _write_tool(home / "tools", "b_ok.py", _echo_source("survivor"))
    _write_tool(home / "tools", "c_break.py", "1/0\n")
    registry = _new_registry(tmp_path)

    discovery = apply_external_tools(
        registry, home=home, project_dir=None
    )

    assert "survivor" in registry.registered_names
    assert sum("a_break.py" in n for n in discovery.notices) == 1
    assert sum("c_break.py" in n for n in discovery.notices) == 1


def test_user_scope_no_directory_is_silent(tmp_path: Path) -> None:
    registry = _new_registry(tmp_path)

    discovery = apply_external_tools(
        registry,
        home=tmp_path / "no-such-home",
        project_dir=tmp_path / "no-such-project" / ".zeta",
    )

    assert discovery.notices == ()
    assert discovery.warnings == ()
    assert discovery.pending_project_tools == ()


@pytest.mark.asyncio
async def test_deny_list_applies_to_user_tool(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_tool(home / "tools", "denied_tool.py", _echo_source("denied_tool"))
    store = ConversationStore(tmp_path / "sessions", session_id="s-77-deny")
    policy = ApprovalPolicy(
        store=store,
        default=ApprovalDecision.ASK,
        always_deny=("denied_tool",),
    )
    (tmp_path / "cwd").mkdir(exist_ok=True)
    registry = ToolRegistry(tmp_path / "cwd", approval_policy=policy)
    registry.bind_session_store(store)
    apply_external_tools(registry, home=home, project_dir=None)
    assert "denied_tool" in registry.registered_names

    result = await registry.execute(
        ToolCall("deny-1", "denied_tool", {"message": "hi"})
    )

    assert result["isError"] is True
    assert result["structuredContent"]["error"]["kind"] == "denied"


@pytest.mark.asyncio
async def test_allow_list_applies_to_project_tool_after_trust(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project" / ".zeta"
    _write_tool(
        project / "tools", "allowed_tool.py", _echo_source("allowed_tool")
    )
    store = ConversationStore(tmp_path / "sessions", session_id="s-77-allow")
    policy = ApprovalPolicy(
        store=store,
        default=ApprovalDecision.ASK,
        always_allow=("allowed_tool",),
    )
    (tmp_path / "cwd").mkdir(exist_ok=True)
    registry = ToolRegistry(tmp_path / "cwd", approval_policy=policy)
    registry.bind_session_store(store)
    discovery = apply_external_tools(
        registry, home=None, project_dir=project
    )
    trust_project_tools(registry, discovery)

    result = await registry.execute(
        ToolCall("allow-1", "allowed_tool", {"message": "run"})
    )

    assert result["isError"] is False
    assert result["structuredContent"] == {"message": "run"}


def test_apply_and_trust_are_idempotent(tmp_path: Path) -> None:
    project = tmp_path / "project" / ".zeta"
    _write_tool(project / "tools", "proj_echo.py", _echo_source("proj_echo"))
    registry = _new_registry(tmp_path)
    discovery = apply_external_tools(
        registry, home=None, project_dir=project
    )

    trust_project_tools(registry, discovery)
    notices, trusted = trust_project_tools(registry, discovery)

    assert notices == ()
    assert trusted == ()


def _build_tui_app(
    tmp_path: Path,
    *,
    home: Path | None,
    project_dir: Path | None,
) -> TUIApp:
    cwd = tmp_path / "cwd"
    cwd.mkdir(exist_ok=True)
    registry = ToolRegistry(cwd)
    discovery = apply_external_tools(
        registry, home=home, project_dir=project_dir
    )
    loop = AgentLoop(
        FakeBackend([]),
        ConversationStore(tmp_path / "sessions", cwd=cwd),
        registry=registry,
    )
    return TUIApp(
        loop,
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
        external_tools=discovery,
    )


def test_slash_tools_list_reports_pending_project_tools(tmp_path: Path) -> None:
    project = tmp_path / "project" / ".zeta"
    _write_tool(project / "tools", "proj_echo.py", _echo_source("proj_echo"))
    app = _build_tui_app(tmp_path, home=None, project_dir=project)
    registry = create_slash_registry()

    output = registry.dispatch(app, "/tools")

    assert output is not None
    assert "proj_echo" in output
    assert "/tools trust" in output


def test_slash_tools_trust_registers_project_tools(tmp_path: Path) -> None:
    project = tmp_path / "project" / ".zeta"
    _write_tool(project / "tools", "proj_echo.py", _echo_source("proj_echo"))
    app = _build_tui_app(tmp_path, home=None, project_dir=project)
    registry = create_slash_registry()

    output = registry.dispatch(app, "/tools trust")

    assert output is not None
    assert "proj_echo" in output
    assert "proj_echo" in app.loop.tool_registry.registered_names
    assert any(
        schema.get("name") == "proj_echo" for schema in app.loop.tool_schemas
    )


def test_slash_tools_trust_reports_no_pending(tmp_path: Path) -> None:
    app = _build_tui_app(tmp_path, home=None, project_dir=None)
    registry = create_slash_registry()

    output = registry.dispatch(app, "/tools trust")

    assert output == "tools: no project tools waiting for trust"


def test_slash_tools_rejects_unknown_subcommand(tmp_path: Path) -> None:
    app = _build_tui_app(tmp_path, home=None, project_dir=None)
    registry = create_slash_registry()

    output = registry.dispatch(app, "/tools revoke")

    assert output == "tools: use /tools or /tools trust"


def test_project_tools_share_directory_with_user_home(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    _write_tool(shared / "tools", "shared_tool.py", _echo_source("shared_tool"))
    registry = _new_registry(tmp_path)

    discovery = apply_external_tools(
        registry, home=shared, project_dir=shared
    )

    assert "shared_tool" in registry.registered_names
    assert discovery.pending_project_tools == ()
    assert discovery.warnings == ()
