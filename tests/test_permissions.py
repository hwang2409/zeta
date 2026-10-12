from __future__ import annotations

import asyncio
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from tests.support.fake_backend import FakeBackend
from zeta.config.settings import load_settings, resolve
from zeta.core.approval import ApprovalDecision, ApprovalPolicy, ApprovalRule
from zeta.core.project_context import ProjectContext
from zeta.core.session import SessionManager
from zeta.core.store import ConversationStore
from zeta.protocol.types import ToolCall
from zeta.runtime.composition import compose_runtime
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.skills.agent_catalog import AgentCatalog
from zeta.tools.registry import ToolRegistry
from zeta.tui.app import TUIApp


@pytest.mark.parametrize("cli_yolo", [True, False], ids=["yolo", "no-yolo"])
async def test_permissions_shows_effective_policy_without_changing_it(
    tmp_path: Path,
    cli_yolo: bool,
) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    (project / ".zeta").mkdir(parents=True)
    settings_path = home / "settings.toml"
    settings_path.write_text(
        f"yolo = {str(not cli_yolo).lower()}\n"
        'tools = ["bash", "write", "read"]\ndisallowed_tools = ["read"]\n'
        '[approval]\nallow = ["bash(git status*)", "read", "todo(*)"]\n'
        'ask = ["bash(git push*)"]\ndeny = ["bash(rm *)"]\n',
        encoding="utf-8",
    )
    (project / ".zeta" / "settings.toml").write_text(
        'tools = ["bash", "read"]\n[approval]\nallow = ["write"]\n',
        encoding="utf-8",
    )
    # A custom executable command must not intercept this read-only built-in.
    (home / "commands").mkdir()
    (home / "commands" / "permissions.md").write_text(
        "---\nkind: exec\n---\necho should-not-run\n",
        encoding="utf-8",
    )
    config = resolve(
        load_settings(home=home, project_dir=project / ".zeta").settings,
        cli_provider="codex",
        cli_model=None,
        cli_yolo=cli_yolo,
        cli_token_budget=None,
        cli_auto_memory=False,
    )
    backend = FakeBackend([])
    composition = compose_runtime(
        home=home,
        cwd=project,
        manager=SessionManager(home),
        config=config,
        provider="codex",
        model="offline",
        project_context=ProjectContext("system", ()),
        backend_builder=lambda *_args, **_kwargs: (backend, "offline"),
        skill_catalog=SkillCatalog.empty(),
        agent_catalog=AgentCatalog.empty(),
        auto_project=False,
    )
    output = StringIO()
    app = TUIApp(
        composition.loop,
        provider="codex",
        model="offline",
        zeta_home=home,
        approval_policy=composition.policy,
        console=Console(file=output, width=160, force_terminal=False),
    )
    policy = composition.policy
    registry = app.loop.tool_registry
    try:
        settings_before = settings_path.read_bytes()
        rules_before = (policy.always_deny, policy.always_ask, policy.always_allow)
        entries_before = app.loop.store.replay()
        capabilities = [
            registry.resolve_call("bash", {"command": command})
            for command in ("rm file", "git push", "git status", "echo hello")
        ]
        expected = ["deny", "ask", "allow", "allow" if cli_yolo else "ask"]
        assert [policy.decide(call) for call in capabilities] == expected

        await app._handle_prompt_value("/permissions")
        assert "Permissions · current session" in output.getvalue()
        view = app._slash_commands.dispatch(app, "/permissions")
        assert f"Default: {expected[-1]}" in view
        assert "deny > ask > allow > default" in view
        assert "Deny rules:\n  bash(rm *)" in view
        assert "Ask rules:\n  bash(git push*)" in view
        assert (
            "Allow rules (including session grants):\n  bash(git status*)\n  read"
            in view
        )
        assert "Session grants (/always):\n  none" in view
        assert "Ignored approval rules:" in view and "todo(*)" in view
        assert str(settings_path) in view
        assert "bash, write, read" in view and "bash, read" in view
        assert "Deny: read" in view
        assert "a call must match every layer" in view
        assert "Project settings cannot set approval rules" in view
        assert not registry.tool_policy.allows("read")
        assert policy.always_allow == {
            ApprovalRule("bash", "git status*"),
            ApprovalRule("read"),
        }
        assert (
            policy.always_deny,
            policy.always_ask,
            policy.always_allow,
        ) == rules_before
        assert [policy.decide(call) for call in capabilities] == expected
        assert settings_path.read_bytes() == settings_before
        assert app.loop.store.replay() == entries_before
        assert not backend.calls

        commands = app._slash_commands
        assert commands.exec_command_for("/permissions") is None
        assert any(
            "shadows built-in /permissions" in notice for notice in commands.notices
        )
        help_text = commands.dispatch(app, "/help")
        assert (
            "/permissions" in help_text
            and "tool/action for the running session" in help_text
        )
        assert any(name == "permissions" for name, _, _ in commands.completion_entries)
        assert "takes no arguments" in commands.dispatch(app, "/permissions allow bash")
        # The next view uses the live policy, not cached settings or constructor state.
        policy.default = ApprovalDecision.DENY
        assert "Default: deny" in commands.dispatch(app, "/permissions")
    finally:
        await app.close()


async def test_permissions_tracks_always_scope_and_session_lifetime(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    policy = ApprovalPolicy(store=store, always_ask={"bash(git push*)"})
    backend = FakeBackend([])
    registry = ToolRegistry(
        tmp_path,
        skill_catalog=SkillCatalog.empty(),
        project_id="test-project",
        inbox_enabled=True,
    )
    app = TUIApp(
        AgentLoop(
            backend,
            store,
            tools=registry,
            approval_policy=policy,
            skill_catalog=SkillCatalog.empty(),
            skip_mcp_mount=True,
        ),
        provider="codex",
        model="offline",
        zeta_home=tmp_path / "home",
        approval_policy=policy,
        console=Console(file=StringIO(), width=160),
    )
    registry = app.loop.tool_registry
    pending: asyncio.Task | None = None
    try:
        for call in (
            ToolCall("shell", "bash", {"command": "echo first"}),
            ToolCall(
                "send",
                "inbox",
                {
                    "action": "send",
                    "project": "demo",
                    "kind": "question",
                    "title": "one",
                    "body": "first",
                },
            ),
        ):
            pending = asyncio.create_task(
                policy.authorize(
                    call,
                    asyncio.Event(),
                    persist_request=False,
                    capability=registry.resolve_call(call.name, call.arguments),
                )
            )
            # Let authorize publish its request before answering.
            await asyncio.sleep(0)
            requests_before = policy.pending_requests()
            assert len(requests_before) == 1
            view = app._slash_commands.dispatch(app, "/permissions")
            assert "with any arguments" in view
            assert policy.pending_requests() == requests_before
            await app._handle_prompt_value(f"/always {call.id}")
            assert await asyncio.wait_for(pending, timeout=5) is ApprovalDecision.ALLOW
            policy.forget_ephemeral(call.id)

        assert policy.session_allow == {
            ApprovalRule("bash"),
            ApprovalRule("inbox", action="send"),
        }
        view = app._slash_commands.dispatch(app, "/permissions")
        assert "Session grants (/always):\n  bash\n  inbox(send)" in view
        for name, arguments, expected in (
            ("bash", {"command": "echo different"}, "allow"),
            ("bash", {"command": "git push"}, "ask"),
            (
                "inbox",
                {
                    "action": "send",
                    "project": "different",
                    "kind": "question",
                    "title": "two",
                    "body": "second",
                },
                "allow",
            ),
            ("inbox", {"action": "claim", "id": "other"}, "ask"),
        ):
            assert policy.decide(registry.resolve_call(name, arguments)) == expected
        receipt = app.console.file.getvalue()
        assert "always allowed bash for this session" in receipt
        assert "always allowed inbox(send) for this session" in receipt
        assert not (tmp_path / "home" / "settings.toml").exists()
        assert not backend.calls
        # Reconstructing a policy does not restore an /always session grant.
        restarted = ApprovalPolicy(store=store)
        registry.set_approval_policy(restarted)
        view = app._slash_commands.dispatch(app, "/permissions")
        assert "Session grants (/always):\n  none" in view
        assert (
            restarted.decide(
                registry.resolve_call("bash", {"command": "echo different"})
            )
            is ApprovalDecision.ASK
        )
    finally:
        if pending is not None:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        await app.close()
