import asyncio
import subprocess
from io import StringIO
from pathlib import Path

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.data_structures import Size
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output.vt100 import Vt100_Output
from rich.console import Console

from zeta.core.approval import ApprovalDecision, ApprovalPolicy
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.slash import (
    COMMAND_FILE_SIZE_LIMIT,
    CustomCommand,
    SlashModelInput,
    create_slash_registry,
    load_custom_commands,
)
from zeta.core.store import ConversationStore
from zeta.loop import AgentLoop
from zeta.mcp import MCPPrompt, MCPPromptArgument
from zeta.tools import ToolRegistry
from zeta.tools.exec import MacroDisplay, run_exec_macro, run_inline_shell_batch
from zeta.tui.app import TUIApp
from zeta.tui.composer import (
    FullScreenPromptSession,
    SlashCompleter,
    build_key_bindings,
)
from zeta.tui.render import render_approval_card
from zeta.types import TextContent, ToolCall


def _write_command(directory: Path, name: str, content: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.md").write_text(content, encoding="utf-8")


def test_loads_home_and_project_commands_with_project_precedence(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    _write_command(home / "commands", "shared", "home $1 $ARGUMENTS")
    _write_command(home / "commands", "home-only", "home command")
    _write_command(
        project / ".zeta" / "commands",
        "shared",
        "---\ndescription: project command\nunknown: ignored\n---\nproject $1",
    )
    _write_command(project / ".zeta" / "commands", "project-only", "project command")

    registry = create_slash_registry(zeta_home=home, project_dir=project)

    assert registry.input_for_model("/shared first second") == "project first"
    assert registry.input_for_model("/home-only") == "home command"
    assert {command.name for command in registry.custom_commands} == {
        "home-only",
        "project-only",
        "shared",
    }
    assert str(project / ".zeta" / "commands" / "shared.md") in registry.help_text()


def test_substitution_uses_empty_missing_positions_and_keeps_raw_tail(
    tmp_path: Path,
) -> None:
    _write_command(
        tmp_path / "commands",
        "args",
        "one=$1 two=$2 nine=$9 raw=[$ARGUMENTS]",
    )

    registry = create_slash_registry(zeta_home=tmp_path, project_dir=tmp_path / "empty")

    assert registry.input_for_model("/args alpha  beta") == (
        "one=alpha two=beta nine= raw=[alpha  beta]"
    )
    assert registry.input_for_model("/args") == "one= two= nine= raw=[]"


def test_builtin_shadow_is_ignored_and_notices_include_bad_files(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    _write_command(home / "commands", "status", "do not replace status")
    _write_command(project / ".zeta" / "commands", "status", "also ignored")
    _write_command(project / ".zeta" / "commands", "broken", "---\nnot: [yaml")

    registry = create_slash_registry(zeta_home=home, project_dir=project)

    assert registry.input_for_model("/status") == "/status"
    assert any("status.md" in notice for notice in registry.notices)
    assert any("broken.md" in notice for notice in registry.notices)
    assert any("shadows built-in" in notice for notice in registry.warning_notices)


def test_completer_shows_description_and_source_badge(tmp_path: Path) -> None:
    _write_command(
        tmp_path / ".zeta" / "commands",
        "review",
        "---\ndescription: review the change\n---\nreview",
    )
    registry = create_slash_registry(zeta_home=tmp_path / "home", project_dir=tmp_path)
    completions = list(
        SlashCompleter(registry).get_completions(
            Document("/rev"), CompleteEvent(completion_requested=True)
        )
    )

    assert len(completions) == 1
    assert completions[0].text == "review"
    assert completions[0].display_meta[0][1] == "[project] review the change"


def test_completion_applies_to_a_real_buffer(tmp_path: Path) -> None:
    _write_command(tmp_path / ".zeta" / "commands", "review", "review")
    registry = create_slash_registry(zeta_home=tmp_path / "home", project_dir=tmp_path)
    buffer = Buffer(
        completer=SlashCompleter(registry),
        document=Document("/rev"),
    )
    completion = next(
        SlashCompleter(registry).get_completions(
            buffer.document, CompleteEvent(completion_requested=True)
        )
    )

    buffer.apply_completion(completion)

    assert buffer.text == "/review"


@pytest.mark.asyncio
async def test_mcp_prompt_completion_and_resolution() -> None:
    registry = create_slash_registry(
        zeta_home=Path("/does/not/exist"),
        project_dir=Path("/does/not/exist"),
    )
    registry.set_mcp_prompts(
        [
            (
                "server:review",
                "server",
                MCPPrompt(
                    "review",
                    "review code",
                    (MCPPromptArgument("topic", required=True),),
                ),
            )
        ]
    )

    class Session:
        async def slash_mcp_prompt(
            self, name: str, arguments: dict[str, str]
        ) -> str:
            assert name == "server:review"
            return f"resolved {arguments['topic']}"

    completions = list(
        SlashCompleter(registry).get_completions(
            Document("/server:rev"), CompleteEvent(completion_requested=True)
        )
    )
    result = await registry.dispatch_async(Session(), "/server:review tests")

    assert completions[0].display_meta[0][1] == "[server] review code"
    assert isinstance(result, SlashModelInput)
    assert result.text == "resolved tests"


@pytest.mark.asyncio
async def test_mcp_prompt_missing_required_argument_does_not_call_session() -> None:
    registry = create_slash_registry(
        zeta_home=Path("/does/not/exist"),
        project_dir=Path("/does/not/exist"),
    )
    registry.set_mcp_prompts(
        [
            (
                "server:review",
                "server",
                MCPPrompt(
                    "review",
                    "review code",
                    (MCPPromptArgument("topic", required=True),),
                ),
            )
        ]
    )

    class Session:
        async def slash_mcp_prompt(
            self, name: str, arguments: dict[str, str]
        ) -> str:
            raise AssertionError("missing argument made a remote call")

    result = await registry.dispatch_async(Session(), "/server:review")

    assert isinstance(result, str)
    assert "missing required argument" in result


def test_custom_colon_name_is_rejected_for_mcp_namespace(tmp_path: Path) -> None:
    _write_command(tmp_path / "commands", "user:prompt", "body")

    registry = create_slash_registry(
        zeta_home=tmp_path, project_dir=tmp_path / "project"
    )

    assert all(command.name != "user:prompt" for command in registry.custom_commands)
    assert any("colon names" in notice for notice in registry.notices)


async def test_completion_menu_arrows_do_not_navigate_history(tmp_path: Path) -> None:
    _write_command(tmp_path / ".zeta" / "commands", "rebase", "rebase")
    _write_command(tmp_path / ".zeta" / "commands", "review", "review")
    registry = create_slash_registry(zeta_home=tmp_path / "home", project_dir=tmp_path)
    output_text = StringIO()

    with create_pipe_input() as pipe:
        session = PromptSession(
            input=pipe,
            output=Vt100_Output(
                output_text,
                lambda: Size(rows=24, columns=80),
            ),
            key_bindings=build_key_bindings(
                on_interrupt=lambda: None,
                on_exit=lambda: None,
            ),
            completer=SlashCompleter(registry),
            multiline=True,
        )
        task = asyncio.create_task(session.prompt_async(" > "))
        await asyncio.sleep(0)
        pipe.send_text("/r\t")
        for _ in range(100):
            if session.app.current_buffer.complete_state is not None:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("completion menu did not open")

        pipe.send_text("\x1b[A")
        for _ in range(100):
            state = session.app.current_buffer.complete_state
            if state is not None and state.complete_index is not None:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("up arrow did not select a completion")

        assert session.app.current_buffer.text == "/review"
        session.app.exit()
        await task


def test_malformed_files_fail_open(tmp_path: Path) -> None:
    _write_command(tmp_path / "commands", "empty", "   ")
    _write_command(tmp_path / "commands", "unterminated", "---\ndescription: bad")

    result = load_custom_commands(home=tmp_path, project_dir=tmp_path / "project")

    assert result.commands == ()
    assert len(result.notices) == 2


def test_oversized_command_file_is_ignored(tmp_path: Path) -> None:
    _write_command(
        tmp_path / "commands",
        "large",
        "x" * (COMMAND_FILE_SIZE_LIMIT + 1),
    )

    result = load_custom_commands(home=tmp_path, project_dir=tmp_path / "project")

    assert result.commands == ()
    assert "byte limit" in result.notices[0]


def test_deep_yaml_command_fails_open(tmp_path: Path) -> None:
    nested_sequence = "[" * 500 + "]" * 500
    _write_command(
        tmp_path / "commands",
        "deep",
        f"---\nvalue: {nested_sequence}\n---\nbody",
    )

    result = load_custom_commands(home=tmp_path, project_dir=tmp_path / "project")

    assert result.commands == ()
    assert any("deep.md" in notice for notice in result.notices)


def test_tui_command_loading_uses_configured_home(
    tmp_path: Path, monkeypatch
) -> None:
    live_home = tmp_path / "live-home"
    _write_command(live_home / "commands", "leak", "must not load")
    fake_home = tmp_path / "fake-home"
    _write_command(fake_home / "commands", "configured", "must load")
    monkeypatch.setattr(Path, "home", lambda: live_home)

    app = TUIApp(
        AgentLoop(
            FakeBackend([]),
            ConversationStore(tmp_path / "sessions", cwd=tmp_path),
        ),
        provider="fake",
        model="offline",
        zeta_home=fake_home,
        console=Console(file=StringIO(), force_terminal=False),
    )

    assert all(command.name != "leak" for command in app._slash_commands.custom_commands)
    assert [command.name for command in app._slash_commands.custom_commands] == [
        "configured"
    ]


def test_tui_command_loading_skips_host_home_without_configured_home(
    tmp_path: Path, monkeypatch
) -> None:
    sentinel_home = tmp_path / "sentinel-home"
    sentinel_path = sentinel_home / ".zeta" / "commands" / "sentinel.md"
    _write_command(sentinel_home / ".zeta" / "commands", "sentinel", "must not load")
    reads: list[Path] = []
    original_read_text = Path.read_text

    def read_text(path: Path, *args, **kwargs) -> str:
        if path == sentinel_path:
            reads.append(path)
        return original_read_text(path, *args, **kwargs)

    monkeypatch.delenv("ZETA_HOME", raising=False)
    monkeypatch.setenv("HOME", str(sentinel_home))
    monkeypatch.setattr(Path, "home", lambda: sentinel_home)
    monkeypatch.setattr(Path, "read_text", read_text)

    app = TUIApp(
        AgentLoop(
            FakeBackend([]),
            ConversationStore(tmp_path / "sessions", cwd=tmp_path),
        ),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )

    assert all(
        command.name != "sentinel" for command in app._slash_commands.custom_commands
    )
    assert reads == []


def test_tui_command_loading_uses_repository_root_from_nested_cwd(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    nested = repo / "src" / "nested"
    nested.mkdir(parents=True)
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
    _write_command(repo / ".zeta" / "commands", "review", "review the change")

    app = TUIApp(
        AgentLoop(
            FakeBackend([]),
            ConversationStore(tmp_path / "sessions", cwd=nested),
        ),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )

    assert [command.name for command in app._slash_commands.custom_commands] == [
        "review"
    ]


async def test_transcript_search_cancels_completion_before_up_navigation(
    tmp_path: Path,
) -> None:
    _write_command(tmp_path / ".zeta" / "commands", "review", "review")
    output_text = StringIO()
    search_active = False

    def start_search() -> None:
        nonlocal search_active
        search_active = True

    def end_search() -> None:
        nonlocal search_active
        search_active = False

    with create_pipe_input() as pipe:
        session = FullScreenPromptSession(
            input=pipe,
            output=Vt100_Output(
                output_text,
                lambda: Size(rows=24, columns=80),
            ),
            key_bindings=build_key_bindings(
                on_interrupt=lambda: None,
                on_exit=lambda: None,
                on_search_start=start_search,
                search_active=lambda: search_active,
                on_search_input=lambda _value: None,
                on_search_next=lambda: None,
                on_search_end=end_search,
            ),
            completer=SlashCompleter(
                create_slash_registry(zeta_home=tmp_path / "home", project_dir=tmp_path)
            ),
            multiline=True,
        )
        task = asyncio.create_task(session.prompt_async(" > "))
        await asyncio.sleep(0.05)
        pipe.send_text("/r\t")
        for _ in range(100):
            if session.app.current_buffer.complete_state is not None:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("completion menu did not open")

        pipe.send_text("\x06")
        for _ in range(100):
            if search_active and session.app.current_buffer.complete_state is None:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("transcript search did not cancel completion")

        pipe.send_text("\r")
        await asyncio.sleep(0.05)
        pipe.send_text("\x1b")
        for _ in range(100):
            if not search_active:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("transcript search did not end")

        pipe.send_text("\x1b[A")
        await asyncio.sleep(0.05)
        assert session.app.current_buffer.text == "/r"
        session.app.exit()
        await task


async def test_custom_command_becomes_the_model_user_message(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    _write_command(home / "commands", "review", "Review $1")
    backend = FakeBackend(
        [ScriptedTurn(content=[TextContent("done")])]
    )
    app = TUIApp(
        AgentLoop(backend, ConversationStore(tmp_path / "sessions", cwd=tmp_path)),
        provider="fake",
        model="offline",
        zeta_home=home,
        console=Console(file=StringIO(), force_terminal=False),
    )

    await app._handle_prompt_value("/review changes")
    assert app._active_task is not None
    await app._active_task

    user_message = next(
        message
        for message in backend.calls[0][0]
        if message.role.value == "user"
    )
    assert user_message.content[0].text == "Review changes"


async def test_prompt_macro_resolves_inline_shell_and_template_attachments(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    notes = tmp_path / "notes.txt"
    image = tmp_path / "image.png"
    notes.write_text("template note", encoding="utf-8")
    image.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000d49444154789c6360606000000004000000a6f645"
            "0000000049454e44ae426082"
        )
    )
    _write_command(
        home / "commands",
        "inspect",
        "read !`printf shell-value` @./notes.txt @./image.png",
    )
    backend = FakeBackend([ScriptedTurn(content=[TextContent("done")])])
    app = TUIApp(
        AgentLoop(
            backend,
            ConversationStore(tmp_path / "sessions", cwd=tmp_path),
        ),
        provider="fake",
        model="offline",
        zeta_home=home,
        console=Console(file=StringIO(), force_terminal=True),
    )

    await app._handle_prompt_value("/inspect")
    await app._active_task

    user_message = next(
        message
        for message in backend.calls[0][0]
        if message.role.value == "user"
    )
    assert user_message.content[0].text == "read shell-value @./notes.txt @./image.png"
    assert any(
        isinstance(block, TextContent) and block.text.endswith("template note")
        for block in user_message.content
    )
    assert any(block.type.value == "image" for block in user_message.content)


async def test_inline_shell_approval_covers_the_complete_batch(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_command(
        home / "commands",
        "inspect",
        "values !`printf first` and !`printf second`",
    )
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    policy = ApprovalPolicy(store=store, default=ApprovalDecision.ASK)
    backend = FakeBackend([ScriptedTurn(content=[TextContent("done")])])
    output = StringIO()
    app = TUIApp(
        AgentLoop(backend, store, approval_policy=policy),
        provider="fake",
        model="offline",
        zeta_home=home,
        approval_policy=policy,
        console=Console(file=output, force_terminal=True, width=120),
    )

    task = asyncio.create_task(app._handle_prompt_value("/inspect"))
    for _ in range(100):
        if app.pending_approvals:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("inline shell approval did not appear")
    assert len(app.pending_approvals) == 1
    assert "printf first" in output.getvalue()
    assert "printf second" in output.getvalue()
    assert policy.approve(app.pending_approvals[0].key)
    await task
    await app._active_task

    user_message = next(
        message
        for message in backend.calls[0][0]
        if message.role.value == "user"
    )
    assert user_message.content[0].text == "values first and second"


async def test_inline_shell_failure_and_output_are_bounded(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_command(
        home / "commands",
        "inspect",
        "bad !`sh -c 'printf bad >&2; exit 4'` long !`printf '%020000d' 1`",
    )
    backend = FakeBackend([ScriptedTurn(content=[TextContent("done")])])
    app = TUIApp(
        AgentLoop(
            backend,
            ConversationStore(tmp_path / "sessions", cwd=tmp_path),
        ),
        provider="fake",
        model="offline",
        zeta_home=home,
        console=Console(file=StringIO(), force_terminal=True),
    )

    await app._handle_prompt_value("/inspect")
    await app._active_task
    user_message = next(
        message
        for message in backend.calls[0][0]
        if message.role.value == "user"
    )
    assert "[inline shell failed: exit 4]" in user_message.content[0].text
    assert len(user_message.content[0].text) < 10_000
    assert await run_inline_shell_batch(
        ToolRegistry(tmp_path),
        ("sleep 1",),
        lifecycle_sink=lambda _kind, _call: None,
        timeout=0.01,
    ) == ("[inline shell failed: timed out]",)


async def test_background_exec_macro_notifies_on_next_turn_and_cancels_on_exit(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    _write_command(
        home / "commands",
        "background",
        "---\nkind: exec\nbackground: true\n---\nprintf complete\n",
    )
    backend = FakeBackend([ScriptedTurn(content=[TextContent("done")])])
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    output = StringIO()
    app = TUIApp(
        AgentLoop(backend, store),
        provider="fake",
        model="offline",
        zeta_home=home,
        console=Console(file=output, force_terminal=True),
    )

    await app._handle_prompt_value("/background")
    assert "/background · running" in output.getvalue()
    await app.loop._background_owner.wait()
    assert store.agent_notifications()[0].data["status"] == "completed"

    await app._handle_prompt_value("continue")
    await app._active_task
    assert "background · /background · completed" in output.getvalue()

    slow_home = tmp_path / "slow-home"
    _write_command(
        slow_home / "commands",
        "slow",
        "---\nkind: exec\nbackground: true\n---\nsleep 30\n",
    )
    slow_store = ConversationStore(tmp_path / "slow-sessions", cwd=tmp_path)
    slow_app = TUIApp(
        AgentLoop(FakeBackend([]), slow_store),
        provider="fake",
        model="offline",
        zeta_home=slow_home,
        console=Console(file=StringIO(), force_terminal=True),
    )
    await slow_app._handle_prompt_value("/slow")
    await slow_app.loop.close()
    assert slow_store.agent_notifications()[0].data["status"] == "canceled"


def test_exec_kind_loads_and_unknown_kind_fails_open(tmp_path: Path) -> None:
    _write_command(
        tmp_path / "commands",
        "rebuild",
        "---\nkind: exec\n---\nprintf rebuild\n",
    )
    _write_command(
        tmp_path / "commands",
        "unknown",
        "---\nkind: mystery\n---\nprintf unknown\n",
    )

    result = load_custom_commands(home=tmp_path, project_dir=tmp_path / "project")

    assert [command.kind for command in result.commands] == ["exec"]
    assert any("unknown command kind" in notice for notice in result.notices)


def test_exec_macro_timeout_defaults_and_reads_frontmatter(tmp_path: Path) -> None:
    _write_command(tmp_path / "commands", "default", "---\nkind: exec\n---\necho default")
    _write_command(
        tmp_path / "commands",
        "custom",
        "---\nkind: exec\ntimeout: 12.5\n---\necho custom",
    )

    result = load_custom_commands(home=tmp_path, project_dir=tmp_path / "project")

    assert [(command.name, command.timeout) for command in result.commands] == [
        ("custom", 12.5),
        ("default", 300.0),
    ]


def test_prompt_timeout_metadata_is_ignored(tmp_path: Path) -> None:
    _write_command(
        tmp_path / "commands",
        "prompt",
        "---\ntimeout: not-a-number\n---\nuse the prompt",
    )
    _write_command(
        tmp_path / "commands",
        "exec",
        "---\nkind: exec\ntimeout: not-a-number\n---\necho exec",
    )

    result = load_custom_commands(home=tmp_path, project_dir=tmp_path / "project")

    assert [command.name for command in result.commands] == ["prompt"]
    assert result.commands[0].timeout == 300.0
    assert any("exec.md" in notice for notice in result.notices)


async def test_exec_macro_streams_receipt_writes_log_and_skips_provider(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    _write_command(
        home / "commands",
        "rebuild",
        "---\nkind: exec\n---\nprintf 'arg=%s all=%s\\n' $1 \"$ARGUMENTS\"\nprintf failure >&2\nexit 3\n",
    )
    backend = FakeBackend([])
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    output = StringIO()
    app = TUIApp(
        AgentLoop(backend, store),
        provider="fake",
        model="offline",
        zeta_home=home,
        console=Console(file=output, force_terminal=False),
    )

    await app._handle_prompt_value("/rebuild first second")

    log = next(store.session_dir.glob("macro-*.log"))
    assert log.read_text(encoding="utf-8") == "arg=first all=first second\nfailure"
    assert backend.calls == []
    assert "/rebuild · exit 3" in output.getvalue()
    assert "failure" not in output.getvalue()


async def test_exec_macro_approval_is_ephemeral_and_uses_substituted_script(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    _write_command(
        home / "commands",
        "deploy",
        "---\nkind: exec\n---\nprintf approved-$1\n",
    )
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    policy = ApprovalPolicy(store=store, default=ApprovalDecision.ASK)
    output = StringIO()
    app = TUIApp(
        AgentLoop(FakeBackend([]), store, approval_policy=policy),
        provider="fake",
        model="offline",
        zeta_home=home,
        approval_policy=policy,
        console=Console(file=output, force_terminal=False),
    )

    task = asyncio.create_task(app._handle_prompt_value("/deploy now"))
    for _ in range(100):
        if app.pending_approvals:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("macro approval did not appear")
    assert "command=printf approved-now" in output.getvalue()
    assert store.messages() == []
    assert policy.approve(app.pending_approvals[0].key)
    await task
    assert store.messages() == []


def test_exec_macro_approval_card_keeps_script_and_every_argv_value() -> None:
    trusted = MacroDisplay(
        "printf '%s\\n' \"$@\"",
        ("first-target", "second target", "$HOME"),
    )

    output = StringIO()
    Console(file=output, force_terminal=False, width=80).print(
        render_approval_card(
            "exec",
            {"command": "sh -c 'printf ...'"},
            trusted_display=trusted,
        )
    )

    card = output.getvalue()
    assert 'command=printf \'%s\\n\' "$@"' in card
    assert "[1] first-target" in card
    assert "[2] second target" in card
    assert "[3] $HOME" in card


def test_render_approval_card_ignores_provider_supplied_display_fields() -> None:
    """Provider-injected display fields must never override the real command."""

    arguments = {
        "command": "rm -rf /tmp/real-target",
        "display_command": "printf safe",
        "display_argv": ["harmless"],
    }

    output = StringIO()
    Console(file=output, force_terminal=False, width=120).print(
        render_approval_card("exec", arguments)
    )

    card = output.getvalue()
    assert "command=rm -rf /tmp/real-target" in card
    assert "printf safe" not in card
    assert "harmless" not in card
    assert "argv:" not in card


async def test_exec_macro_approval_card_shows_argv_via_trusted_display(
    tmp_path: Path,
) -> None:
    """The rendered card must show argv for the actual macro tool call."""

    home = tmp_path / "home"
    _write_command(
        home / "commands",
        "deploy",
        "---\nkind: exec\n---\nprintf '%s\\n' \"$@\"\n",
    )
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    policy = ApprovalPolicy(store=store, default=ApprovalDecision.ASK)
    output = StringIO()
    app = TUIApp(
        AgentLoop(FakeBackend([]), store, approval_policy=policy),
        provider="fake",
        model="offline",
        zeta_home=home,
        approval_policy=policy,
        console=Console(file=output, force_terminal=False, width=120),
    )

    task = asyncio.create_task(
        app._handle_prompt_value("/deploy staging release-42 $HOME")
    )
    for _ in range(100):
        if app.pending_approvals:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("macro approval did not appear")

    card = output.getvalue()
    assert "argv:" in card
    assert "[1] staging" in card
    assert "[2] release-42" in card
    assert "[3] $HOME" in card

    assert policy.approve(app.pending_approvals[0].key)
    await task


async def test_exec_macro_deny_renders_denied_receipt(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_command(home / "commands", "deploy", "---\nkind: exec\n---\nprintf deploy")
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    policy = ApprovalPolicy(store=store, default=ApprovalDecision.ASK)
    output = StringIO()
    app = TUIApp(
        AgentLoop(FakeBackend([]), store, approval_policy=policy),
        provider="fake",
        model="offline",
        zeta_home=home,
        approval_policy=policy,
        console=Console(file=output, force_terminal=False),
    )

    task = asyncio.create_task(app._handle_prompt_value("/deploy"))
    for _ in range(100):
        if app.pending_approvals:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("macro approval did not appear")

    assert policy.deny(app.pending_approvals[0].key)
    await task

    assert "/deploy · denied" in output.getvalue()
    assert "/deploy · failed" not in output.getvalue()
    assert store.messages() == []


async def test_exec_macro_abort_kills_process_and_renders_canceled_receipt(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    _write_command(home / "commands", "wait", "---\nkind: exec\n---\nsleep 30\n")
    output = StringIO()
    app = TUIApp(
        AgentLoop(
            FakeBackend([]), ConversationStore(tmp_path / "sessions", cwd=tmp_path)
        ),
        provider="fake",
        model="offline",
        zeta_home=home,
        console=Console(file=output, force_terminal=False),
    )

    task = asyncio.create_task(app._handle_prompt_value("/wait"))
    for _ in range(100):
        if app.active:
            break
        await asyncio.sleep(0.01)
    app.abort_active()
    await asyncio.wait_for(task, timeout=3)

    assert "/wait · canceled" in output.getvalue()
    log = next(app.loop.store.session_dir.glob("macro-*.log"))
    assert log.exists()
    assert "canceled · log " in output.getvalue()


async def test_exec_macro_passes_special_arguments_as_shell_argv(tmp_path: Path) -> None:
    command = CustomCommand(
        "args",
        "",
        "printf 'one=<%s>\\n' \"$1\"; printf 'all=<%s>\\n' \"$@\"; "
        "printf 'raw=<%s>\\n' \"$ARGUMENTS\"; "
        "printf 'ten=<%s> ten0=<%s>\\n' \"${10}\" \"$10\"",
        tmp_path / "args.md",
        "home",
        "exec",
    )
    registry = ToolRegistry(tmp_path)
    call = ToolCall(
        "macro-args",
        "exec",
        {
            "command": command.render_exec("one; '$HOME'\nline two"),
            "timeout": command.timeout,
        },
    )

    result = await run_exec_macro(
        registry,
        call,
        tmp_path / "args.log",
        stream_sink=lambda _event: None,
        lifecycle_sink=lambda _kind: None,
    )

    assert result.is_error is False
    assert "one=<one;>" in result.content
    assert "all=<'$HOME'>" in result.content
    assert "raw=<one; '$HOME'\nline two>" in result.content
    assert "ten=<> ten0=<one;0>" in result.content


async def test_macro_input_loop_keeps_processing_approval_input(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_command(home / "commands", "deploy", "---\nkind: exec\n---\nsleep 0.1")
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    policy = ApprovalPolicy(store=store, default=ApprovalDecision.ASK)
    app = TUIApp(
        AgentLoop(FakeBackend([]), store, approval_policy=policy),
        provider="fake",
        model="offline",
        zeta_home=home,
        approval_policy=policy,
        console=Console(file=StringIO(), force_terminal=False),
    )
    app._input_loop_active = True

    await app._handle_prompt_value("/deploy")

    assert app.active
    for _ in range(100):
        if app.pending_approvals:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("macro approval did not appear")
    key = app.pending_approvals[0].key
    await app._handle_prompt_value(f"approve {key}")
    await asyncio.wait_for(app._active_task, timeout=2)


async def test_macro_receipts_queue_until_the_next_provider_turn(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_command(home / "commands", "first", "---\nkind: exec\n---\nprintf first")
    _write_command(home / "commands", "second", "---\nkind: exec\n---\nprintf second")
    backend = FakeBackend([ScriptedTurn(content=[TextContent("done")])])
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    app = TUIApp(
        AgentLoop(backend, store),
        provider="fake",
        model="offline",
        zeta_home=home,
        console=Console(file=StringIO(), force_terminal=False),
    )

    await app._handle_prompt_value("/first")
    await app._handle_prompt_value("/second")
    await app._handle_prompt_value("continue")
    await asyncio.wait_for(app._active_task, timeout=2)

    user_message = next(message for message in backend.calls[0][0] if message.role.value == "user")
    assert user_message.content[0].text.startswith(
        "ran /first, exit 0\nran /second, exit 0\n\ncontinue"
    )


async def test_queued_prompt_consumes_macro_receipt_at_provider_start(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    _write_command(home / "commands", "slow", "---\nkind: exec\n---\nsleep 0.05")
    backend = FakeBackend([ScriptedTurn(content=[TextContent("done")])])
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    app = TUIApp(
        AgentLoop(backend, store),
        provider="fake",
        model="offline",
        zeta_home=home,
        console=Console(file=StringIO(), force_terminal=False),
    )
    app._input_loop_active = True

    macro_task = asyncio.create_task(app._handle_prompt_value("/slow"))
    for _ in range(100):
        if app.active:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("slow macro did not start")

    await app._handle_prompt_value("continue")
    assert len(app._queued) == 1
    await asyncio.wait_for(app._active_task, timeout=2)
    await macro_task
    app._active_task = None
    app._start_queued_turn()
    await app._active_task

    user_message = next(
        message
        for message in backend.calls[0][0]
        if message.role.value == "user"
    )
    assert user_message.content[0].text == "ran /slow, exit 0\n\ncontinue"


async def test_two_macros_queued_prompt_and_abort_keep_receipt_order(
    tmp_path: Path,
) -> None:
    """The provider must see receipts in insertion order: first, second, then abort receipt."""

    home = tmp_path / "home"
    _write_command(home / "commands", "first", "---\nkind: exec\n---\nprintf first")
    _write_command(home / "commands", "second", "---\nkind: exec\n---\nsleep 30\n")
    backend = FakeBackend([ScriptedTurn(content=[TextContent("done")])])
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    app = TUIApp(
        AgentLoop(backend, store),
        provider="fake",
        model="offline",
        zeta_home=home,
        console=Console(file=StringIO(), force_terminal=False),
    )
    app._input_loop_active = True

    await app._handle_prompt_value("/first")
    if app._active_task is not None:
        await asyncio.wait_for(app._active_task, timeout=3)
        app._active_task = None
    assert list(app._macro_receipts) == ["ran /first, exit 0"]

    await app._handle_prompt_value("/second")
    for _ in range(100):
        if app.active:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("second macro did not start")

    await app._handle_prompt_value("continue")
    assert len(app._queued) == 1

    macro_task = app._active_task
    app.abort_active()
    assert macro_task is not None
    await asyncio.wait_for(macro_task, timeout=3)
    assert list(app._macro_receipts) == [
        "ran /first, exit 0",
        "ran /second, canceled",
    ]

    app._active_task = None
    app._start_queued_turn()
    await asyncio.wait_for(app._active_task, timeout=3)

    user_message = next(
        message
        for message in backend.calls[0][0]
        if message.role.value == "user"
    )
    assert user_message.content[0].text == (
        "ran /first, exit 0\nran /second, canceled\n\ncontinue"
    )


async def test_macro_abort_does_not_cancel_background_agent(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_command(home / "commands", "wait", "---\nkind: exec\n---\nsleep 30")
    app = TUIApp(
        AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions", cwd=tmp_path)),
        provider="fake",
        model="offline",
        zeta_home=home,
        console=Console(file=StringIO(), force_terminal=False),
    )
    canceled = asyncio.Event()

    async def background() -> None:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            canceled.set()
            raise

    watcher = asyncio.create_task(background())
    app.loop._background_owner.register("background", watcher.cancel, watcher)
    macro_task = asyncio.create_task(app._handle_prompt_value("/wait"))
    for _ in range(100):
        if app.active:
            break
        await asyncio.sleep(0.01)

    app.abort_active()
    await asyncio.wait_for(macro_task, timeout=3)
    assert not canceled.is_set()
    watcher.cancel()
    await asyncio.gather(watcher, return_exceptions=True)
    app.loop._background_owner.unregister("background")


async def test_exec_macro_timeout_has_a_distinct_receipt_status(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_command(
        home / "commands",
        "short",
        "---\nkind: exec\ntimeout: 0.02\n---\nsleep 1",
    )
    output = StringIO()
    app = TUIApp(
        AgentLoop(
            FakeBackend([]), ConversationStore(tmp_path / "sessions", cwd=tmp_path)
        ),
        provider="fake",
        model="offline",
        zeta_home=home,
        console=Console(file=output, force_terminal=False),
    )

    await app._handle_prompt_value("/short")

    assert "/short · timeout" in output.getvalue()
    assert "/short · exit" not in output.getvalue()
