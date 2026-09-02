import asyncio
import subprocess
from io import StringIO
from pathlib import Path

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
    create_slash_registry,
    load_custom_commands,
)
from zeta.core.store import ConversationStore
from zeta.loop import AgentLoop
from zeta.tui.app import TUIApp
from zeta.tui.composer import (
    FullScreenPromptSession,
    SlashCompleter,
    build_key_bindings,
)
from zeta.types import TextContent


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
