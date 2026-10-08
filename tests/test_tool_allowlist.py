from __future__ import annotations

import asyncio
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
    registry.register("remote__click", lambda arguments: "clicked")
    registry.register("remote__type", lambda arguments: "typed")
    return registry


def _messages() -> list[Message]:
    return [Message(MessageRole.USER, [TextContent("go")])]


def test_disallowed_tool_schemas_are_omitted_from_all_provider_payloads(
    tmp_path: Path,
) -> None:
    registry = _registry(
        tmp_path,
        tool_allow=("remote__*",),
        tool_deny=("remote__type",),
    )

    names = [schema["name"] for schema in registry.schemas]
    assert names == ["remote__click"]

    anthropic = build_request_payload(
        _messages(), registry.schemas, model="claude-test", max_tokens=2048, thinking_budget=1024
    )
    codex = build_responses_payload(_messages(), registry.schemas, model="codex-test")
    ollama = ollama_tools(registry.schemas)

    assert [tool["name"] for tool in anthropic["tools"]] == ["remote__click"]
    assert [tool["name"] for tool in codex["tools"]] == ["remote__click"]
    assert [tool["function"]["name"] for tool in ollama] == ["remote__click"]


@pytest.mark.asyncio
async def test_calling_disallowed_tool_returns_clear_policy_error(tmp_path: Path) -> None:
    registry = _registry(tmp_path, tool_allow=("remote__*",))

    result = await registry.execute(ToolCall("call-1", "bash", {}))

    assert result["isError"] is True
    assert "bash" in result["content"][0]["text"]
    assert "not allowed" in result["content"][0]["text"]


def test_glob_allow_and_deny_patterns_apply_to_builtins_and_mcp_names(
    tmp_path: Path,
) -> None:
    registry = _registry(
        tmp_path,
        tool_allow=("remote__*", "read"),
        tool_deny=("*type",),
    )

    assert registry.registered_names == frozenset({"read", "remote__click"})


def test_child_registry_can_narrow_but_cannot_widen_tool_policy(tmp_path: Path) -> None:
    from zeta.core.store import ConversationStore

    parent = _registry(tmp_path, tool_allow=("remote__*", "read"))
    child_store = ConversationStore(tmp_path / "child", cwd=tmp_path)
    try:
        child = parent.clone_for_session(
            child_store,
            exclude_names={"remote__type"},
        )
        assert child.registered_names == frozenset({"read", "remote__click"})
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
        tool_allow=("remote__*",),
        tool_deny=("remote__type",),
    )

    restored = SessionMetadata.from_dict(
        metadata.to_storage_dict(), path=tmp_path / "meta.json"
    )

    assert restored.tool_allow == ("remote__*",)
    assert restored.tool_deny == ("remote__type",)


@pytest.mark.asyncio
async def test_resume_intersects_and_persists_invocation_tool_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.tui.app import create_app

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    first = create_app(build_parser().parse_args(["--provider", "fake"]))
    session_id = first.loop.store.session_id
    await first.close()

    narrowed = create_app(
        build_parser().parse_args(
            [
                "--resume",
                session_id,
                "--provider",
                "fake",
                "--tools",
                "remote__*",
                "--disallowed-tools",
                "remote__type",
            ]
        )
    )
    try:
        assert "bash" not in narrowed.loop.tool_registry.registered_names
        assert not narrowed.loop.tool_registry.tool_is_allowed("remote__type")
        from zeta.core.session import SessionManager

        persisted = SessionManager(home).read_metadata(session_id)
        assert persisted.tool_allow_layers == (("remote__*",),)
        assert persisted.tool_deny == ("remote__type",)
    finally:
        await narrowed.close()

    resumed = create_app(
        build_parser().parse_args(["--resume", session_id, "--provider", "fake"])
    )
    try:
        assert "bash" not in resumed.loop.tool_registry.registered_names
        assert resumed.loop.tool_registry.tool_allow == ("remote__*",)
        assert resumed.loop.tool_registry.tool_deny == ("remote__type",)
    finally:
        await resumed.close()


@pytest.mark.asyncio
async def test_resume_intersects_persisted_policy_with_ambient_policy(
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
                "remote__*",
                "--disallowed-tools",
                "remote__type",
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
        assert resumed.loop.tool_registry.tool_policy.allow_layers == (
            ("remote__*",),
            ("bash",),
        )
        assert resumed.loop.tool_registry.registered_names == frozenset()
        assert resumed.loop.tool_registry.tool_deny == ("remote__type",)
        tools_status = resumed.slash_tools("")
        assert "allow: remote__* AND bash" in tools_status
        assert "deny: remote__type" in tools_status
    finally:
        await resumed.close()


@pytest.mark.asyncio
async def test_server_resume_applies_server_policy_as_upper_bound(tmp_path: Path) -> None:
    from zeta.server.runtime import ServerRuntime

    home = tmp_path / "home"
    first = ServerRuntime(home, cwd=tmp_path, provider="fake")
    try:
        metadata = await first.create_session()
        session_id = metadata.session_id
    finally:
        await first.close()

    restricted = ServerRuntime(
        home, cwd=tmp_path, provider="fake", tools="remote__*"
    )
    try:
        await restricted.resume_session(session_id)
        assert restricted.loop is not None
        assert "bash" not in restricted.loop.tool_registry.registered_names
        persisted = restricted.manager.read_metadata(session_id)
        assert persisted.tool_allow_layers == (("remote__*",),)
    finally:
        await restricted.close()


@pytest.mark.asyncio
async def test_server_require_tools_checks_effective_resume_policy(
    tmp_path: Path,
) -> None:
    from zeta.server.runtime import ServerRuntime

    home = tmp_path / "home"
    first = ServerRuntime(home, cwd=tmp_path, provider="fake", tools="read")
    try:
        metadata = await first.create_session()
        session_id = metadata.session_id
    finally:
        await first.close()

    restricted = ServerRuntime(
        home,
        cwd=tmp_path,
        provider="fake",
        tools="definitely_missing_tool",
        require_tools=True,
    )
    with pytest.raises(ValueError, match="definitely_missing_tool"):
        await restricted.resume_session(session_id)


@pytest.mark.asyncio
async def test_server_require_tools_keeps_persisted_exact_requirements(
    tmp_path: Path,
) -> None:
    from zeta.server.runtime import ServerRuntime

    home = tmp_path / "home"
    first = ServerRuntime(
        home, cwd=tmp_path, provider="fake", tools="definitely_missing_tool"
    )
    try:
        metadata = await first.create_session()
        session_id = metadata.session_id
    finally:
        await first.close()

    resumed = ServerRuntime(
        home,
        cwd=tmp_path,
        provider="fake",
        require_tools=True,
    )
    with pytest.raises(ValueError, match="definitely_missing_tool"):
        await resumed.resume_session(session_id)


def test_cli_require_tools_rejects_exact_name_removed_by_resume_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from zeta.tui.app import create_app

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    first = create_app(
        build_parser().parse_args(["--provider", "fake", "--tools", "read"])
    )
    session_id = first.loop.store.session_id
    asyncio.run(first.close())

    args = build_parser().parse_args(
        [
            "--resume",
            session_id,
            "--provider",
            "fake",
            "--tools",
            "definitely_missing_tool",
            "--require-tools",
            "-p",
            "go",
        ]
    )

    assert run_headless(args, args.prompt) != 0
    assert "definitely_missing_tool" in capsys.readouterr().err


@pytest.mark.parametrize("pattern", ["*", "*__click", "c*"])
def test_ambiguous_allow_pattern_keeps_mcp_server_eligible(pattern: str) -> None:
    from zeta.config.tool_policy import ToolPolicy

    assert ToolPolicy.create((pattern,)).allows_mcp_server("unrelated")


def test_mcp_server_must_satisfy_every_allow_layer() -> None:
    from zeta.config.tool_policy import ToolPolicy

    policy = ToolPolicy.create(
        allow=("*",), allow_layers=(("remote__*",), ("*",))
    )

    assert policy.allows_mcp_server("remote")
    assert not policy.allows_mcp_server("unrelated")


def test_mcp_server_rejects_disjoint_exact_allow_layers() -> None:
    from zeta.config.tool_policy import ToolPolicy

    policy = ToolPolicy.create(
        allow_layers=(("unrelated__echo",), ("unrelated__other",))
    )

    assert not policy.allows_mcp_server("unrelated")


def test_mcp_server_resolves_exact_candidates_against_glob_layers() -> None:
    from zeta.config.tool_policy import ToolPolicy

    compatible = ToolPolicy.create(
        allow_layers=(("unrelated__echo",), ("unrelated__e*",))
    )
    incompatible = ToolPolicy.create(
        allow_layers=(("unrelated__echo",), ("unrelated__other*",))
    )

    assert compatible.allows_mcp_server("unrelated")
    assert not incompatible.allows_mcp_server("unrelated")


def test_mcp_server_keeps_ambiguous_glob_only_intersection_eligible() -> None:
    from zeta.config.tool_policy import ToolPolicy

    policy = ToolPolicy.create(
        allow_layers=(("unrelated__e*",), ("unrelated__*o",))
    )

    assert policy.allows_mcp_server("unrelated")


def test_whole_namespace_deny_excludes_mcp_server() -> None:
    from zeta.config.tool_policy import ToolPolicy

    assert not ToolPolicy.create(deny=("unrelated__*",)).allows_mcp_server(
        "unrelated"
    )
    assert ToolPolicy.create(deny=("unrelated__click",)).allows_mcp_server(
        "unrelated"
    )


def test_policy_intersection_preserves_absent_and_empty_allowlists() -> None:
    from zeta.config.tool_policy import ToolPolicy

    unrestricted = ToolPolicy.create()
    empty = ToolPolicy.create(())

    narrowed = unrestricted.narrowed_by(empty)
    assert narrowed.allow == ()
    assert narrowed.allow_layers == ((),)
    assert not narrowed.allows("bash")

    still_empty = narrowed.narrowed_by(unrestricted)
    assert still_empty.allow == ()
    assert still_empty.allow_layers == ((),)


def test_policy_intersection_unions_denylists() -> None:
    from zeta.config.tool_policy import ToolPolicy

    persisted = ToolPolicy.create(None, ("bash",))
    invocation = ToolPolicy.create(None, ("write",))

    effective = persisted.narrowed_by(invocation)

    assert effective.deny == ("bash", "write")
    assert effective.restricted is True


def test_project_tool_policy_can_only_narrow_global_policy(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    (home / "settings.toml").write_text(
        'tools = ["remote__*"]\ndisallowed_tools = ["bash"]\n',
        encoding="utf-8",
    )
    (project / "settings.toml").write_text(
        'tools = ["*"]\ndisallowed_tools = []\n', encoding="utf-8"
    )

    loaded = load_settings(home=home, project_dir=project)
    config = resolve(
        loaded.settings,
        cli_provider=None,
        cli_model=None,
        cli_yolo=None,
        cli_token_budget=None,
    )
    registry = _registry(
        tmp_path / "registry",
        tool_allow=config.tool_allow,
        tool_allow_layers=config.tool_allow_layers,
        tool_deny=config.tool_deny,
    )

    assert registry.registered_names == frozenset(
        {"remote__click", "remote__type"}
    )
    assert config.tool_deny == ("bash",)
    assert any("cannot widen" in notice for notice in loaded.notices)


def test_empty_project_allowlist_allows_nothing(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    (home / "settings.toml").write_text(
        'tools = ["remote__*"]\n', encoding="utf-8"
    )
    (project / "settings.toml").write_text('tools = []\n', encoding="utf-8")

    loaded = load_settings(home=home, project_dir=project)
    config = resolve(
        loaded.settings,
        cli_provider=None,
        cli_model=None,
        cli_yolo=None,
        cli_token_budget=None,
    )
    registry = _registry(
        tmp_path / "registry",
        tool_allow=config.tool_allow,
        tool_allow_layers=config.tool_allow_layers,
        tool_deny=config.tool_deny,
    )

    assert registry.registered_names == frozenset()


@pytest.mark.parametrize(
    "body, field",
    [
        ('tools = "remote__*"\n', "tools"),
        ('tools = [""]\n', "tools"),
        ('disallowed_tools = "bash"\n', "disallowed_tools"),
        ('disallowed_tools = [""]\n', "disallowed_tools"),
    ],
)
def test_malformed_tool_policy_is_a_startup_error(
    tmp_path: Path, body: str, field: str
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / "settings.toml").write_text(body, encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        load_settings(home=home, project_dir=None)


def test_cli_and_layered_settings_resolve_tool_patterns(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    (home / "settings.toml").write_text(
        'tools = ["read", "remote__*"]\ndisallowed_tools = ["remote__type"]\n',
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
    assert config.tool_allow == ("read", "remote__*")
    assert config.tool_allow_layers == (("read", "remote__*"),)
    assert config.tool_deny == ("remote__type",)

    overridden = resolve(
        loaded.settings,
        cli_provider=None,
        cli_model=None,
        cli_yolo=None,
        cli_token_budget=None,
        cli_tools="bash",
        cli_disallowed_tools="read",
    )
    assert overridden.tool_allow_layers == (("bash",),)
    assert overridden.tool_deny == ("read",)

    args = build_parser().parse_args(
        ["--tools", "remote__*,read", "--disallowed-tools", "remote__type", "-p", "go"]
    )
    assert args.tools == "remote__*,read"
    assert args.disallowed_tools == "remote__type"


def test_only_global_settings_can_enable_hooks(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    (home / "settings.toml").write_text("allow_hooks = true\n", encoding="utf-8")
    (project / "settings.toml").write_text("allow_hooks = false\n", encoding="utf-8")

    loaded = load_settings(home=home, project_dir=project)
    config = resolve(
        loaded.settings,
        cli_provider=None,
        cli_model=None,
        cli_yolo=None,
        cli_token_budget=None,
    )

    assert config.allow_hooks is True
    assert any("allow_hooks" in warning for warning in loaded.warnings)


def test_only_global_settings_can_enable_external_tools(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    (home / "settings.toml").write_text(
        "allow_external_tools = true\n", encoding="utf-8"
    )
    (project / "settings.toml").write_text(
        "allow_external_tools = false\n", encoding="utf-8"
    )

    loaded = load_settings(home=home, project_dir=project)
    config = resolve(
        loaded.settings,
        cli_provider=None,
        cli_model=None,
        cli_yolo=None,
        cli_token_budget=None,
    )

    assert config.allow_external_tools is True
    assert any("allow_external_tools" in warning for warning in loaded.warnings)


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
                    "remote": {
                        "transport": "stdio",
                        "command": "definitely-not-a-server",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        ["--provider", "fake", "--tools", "remote__*", "-p", "go"]
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
            "remote__click",
            "--require-tools",
            "-p",
            "go",
        ]
    )

    assert run_headless(args, args.prompt) != 0
    assert "remote__click" in capsys.readouterr().err


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
