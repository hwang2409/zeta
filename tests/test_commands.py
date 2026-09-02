from io import StringIO
from pathlib import Path

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from rich.console import Console

from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.slash import create_slash_registry, load_custom_commands
from zeta.core.store import ConversationStore
from zeta.loop import AgentLoop
from zeta.tui.app import TUIApp
from zeta.tui.composer import SlashCompleter
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
    assert completions[0].text == "iew"
    assert completions[0].display_meta[0][1] == "[project] review the change"


def test_malformed_files_fail_open(tmp_path: Path) -> None:
    _write_command(tmp_path / "commands", "empty", "   ")
    _write_command(tmp_path / "commands", "unterminated", "---\ndescription: bad")

    result = load_custom_commands(home=tmp_path, project_dir=tmp_path / "project")

    assert result.commands == ()
    assert len(result.notices) == 2


async def test_custom_command_becomes_the_model_user_message(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    _write_command(home / "commands", "review", "Review $1")
    monkeypatch.setenv("ZETA_HOME", str(home))
    backend = FakeBackend(
        [ScriptedTurn(content=[TextContent("done")])]
    )
    app = TUIApp(
        AgentLoop(backend, ConversationStore(tmp_path / "sessions", cwd=tmp_path)),
        provider="fake",
        model="offline",
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
