from __future__ import annotations

import json
from pathlib import Path

import pytest

from zeta.cli.main import build_parser
from zeta.config.settings import load_settings, resolve
from zeta.core.session import SessionMetadata
from zeta.protocol.types import Message, MessageRole, TextContent, ToolCall
from zeta.providers.anthropic import build_request_payload
from zeta.providers.codex import build_responses_payload
from zeta.providers.ollama import _tools as ollama_tools
from zeta.runtime.headless import run_headless
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry


def _registry(tmp_path: Path, **kwargs: object) -> ToolRegistry:
    tmp_path.mkdir(parents=True, exist_ok=True)
    registry = ToolRegistry(
        tmp_path,
        register_builtin=False,
        skill_catalog=SkillCatalog.empty(),
        **kwargs,
    )
    registry.register("read", lambda arguments: "read")
    registry.register("bash", lambda arguments: "bash")
    registry.register("computer__click", lambda arguments: "clicked")
    registry.register("computer__type", lambda arguments: "typed")
    return registry


def _messages() -> list[Message]:
    return [Message(MessageRole.USER, [TextContent("go")])]


def test_disallowed_tool_schemas_are_omitted_from_all_provider_payloads(
    tmp_path: Path,
) -> None:
    registry = _registry(
        tmp_path,
        tool_allow=("computer__*",),
        tool_deny=("computer__type",),
    )

    names = [schema["name"] for schema in registry.schemas]
    assert names == ["computer__click"]

    anthropic = build_request_payload(
        _messages(), registry.schemas, model="claude-test", max_tokens=2048, thinking_budget=1024
    )
    codex = build_responses_payload(_messages(), registry.schemas, model="codex-test")
    ollama = ollama_tools(registry.schemas)

    assert [tool["name"] for tool in anthropic["tools"]] == ["computer__click"]
    assert [tool["name"] for tool in codex["tools"]] == ["computer__click"]
    assert [tool["function"]["name"] for tool in ollama] == ["computer__click"]


@pytest.mark.asyncio
async def test_calling_disallowed_tool_returns_clear_policy_error(tmp_path: Path) -> None:
    registry = _registry(tmp_path, tool_allow=("computer__*",))

    result = await registry.execute(ToolCall("call-1", "bash", {}))

    assert result["isError"] is True
    assert "bash" in result["content"][0]["text"]
    assert "not allowed" in result["content"][0]["text"]


def test_glob_allow_and_deny_patterns_apply_to_builtins_and_mcp_names(
    tmp_path: Path,
) -> None:
    registry = _registry(
        tmp_path,
        tool_allow=("computer__*", "read"),
        tool_deny=("*type",),
    )

    assert registry.registered_names == frozenset({"read", "computer__click"})


def test_child_registry_can_narrow_but_cannot_widen_tool_policy(tmp_path: Path) -> None:
    from zeta.core.store import ConversationStore

    parent = _registry(tmp_path, tool_allow=("computer__*", "read"))
    child_store = ConversationStore(tmp_path / "child", cwd=tmp_path)
    try:
        child = parent.clone_for_session(
            child_store,
            exclude_names={"computer__type"},
        )
        assert child.registered_names == frozenset({"read", "computer__click"})
        assert "bash" not in child.registered_names
    finally:
        child_store.close()


def test_tool_policy_round_trips_in_session_metadata(tmp_path: Path) -> None:
    metadata = SessionMetadata.new(
        session_id="session-1",
        provider="fake",
        model="offline",
        cwd=str(tmp_path),
        retained_tail=8,
        compaction_budget=1000,
        tool_allow=("computer__*",),
        tool_deny=("computer__type",),
    )

    restored = SessionMetadata.from_dict(
        metadata.to_storage_dict(), path=tmp_path / "meta.json"
    )

    assert restored.tool_allow == ("computer__*",)
    assert restored.tool_deny == ("computer__type",)


@pytest.mark.asyncio
async def test_resume_keeps_persisted_tool_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.tui.app import create_app

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    first = create_app(
        build_parser().parse_args(
            [
                "--provider",
                "fake",
                "--tools",
                "computer__*",
                "--disallowed-tools",
                "computer__type",
            ]
        )
    )
    session_id = first.loop.store.session_id
    await first.close()
    (home / "settings.toml").write_text(
        'tools = ["bash"]\ndisallowed_tools = []\n', encoding="utf-8"
    )

    resumed = create_app(
        build_parser().parse_args(["--resume", session_id, "--provider", "fake"])
    )
    try:
        assert resumed.loop.tool_registry.tool_allow == ("computer__*",)
        assert resumed.loop.tool_registry.tool_deny == ("computer__type",)
        tools_status = resumed.slash_tools("")
        assert "allow: computer__*" in tools_status
        assert "deny: computer__type" in tools_status
    finally:
        await resumed.close()


def test_cli_and_layered_settings_resolve_tool_patterns(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    (home / "settings.toml").write_text(
        'tools = ["read", "computer__*"]\ndisallowed_tools = ["computer__type"]\n',
        encoding="utf-8",
    )
    loaded = load_settings(home=home, project_dir=project)
    config = resolve(
        loaded.settings,
        cli_provider="fake",
        cli_model=None,
        cli_yolo=None,
        cli_token_budget=None,
        cli_tools=None,
        cli_disallowed_tools=None,
    )
    assert config.tool_allow == ("read", "computer__*")
    assert config.tool_deny == ("computer__type",)

    args = build_parser().parse_args(
        ["--tools", "computer__*,read", "--disallowed-tools", "computer__type", "-p", "go"]
    )
    assert args.tools == "computer__*,read"
    assert args.disallowed_tools == "computer__type"


@pytest.mark.asyncio
async def test_mcp_start_failure_with_allowlist_does_not_advertise_builtins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.tui.app import create_app

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    (home / "mcp.json").write_text(
        json.dumps(
            {
                "servers": {
                    "computer": {
                        "transport": "stdio",
                        "command": "definitely-not-a-server",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        ["--provider", "fake", "--tools", "computer__*", "-p", "go"]
    )
    app = create_app(args)
    try:
        await app.loop.activate()
        await app.loop.ensure_mcp_servers()
        assert app.loop.tool_registry.schemas == []
    finally:
        await app.close()


def test_require_tools_exits_nonzero_when_exact_allowlisted_tool_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(
        [
            "--provider",
            "fake",
            "--tools",
            "computer__click",
            "--require-tools",
            "-p",
            "go",
        ]
    )

    assert run_headless(args, args.prompt) != 0
    assert "computer__click" in capsys.readouterr().err


def test_default_tool_policy_keeps_provider_request_bytes_identical(
    tmp_path: Path,
) -> None:
    baseline = _registry(tmp_path / "baseline")
    explicit_default = _registry(tmp_path / "default", tool_allow=None, tool_deny=())

    def payloads(registry: ToolRegistry) -> list[object]:
        return [
            build_request_payload(
                _messages(),
                registry.schemas,
                model="claude-test",
                max_tokens=2048,
                thinking_budget=1024,
            ),
            build_responses_payload(
                _messages(), registry.schemas, model="codex-test"
            ),
            ollama_tools(registry.schemas),
        ]

    encode = lambda value: json.dumps(
        value, separators=(",", ":"), sort_keys=True
    ).encode()
    assert encode(payloads(explicit_default)) == encode(payloads(baseline))
