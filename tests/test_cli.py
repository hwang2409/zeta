from __future__ import annotations

from zeta.cli import build_parser, main
from zeta.completion import completion_script


def test_completion_parser_accepts_both_shells() -> None:
    parser = build_parser()

    assert parser.parse_args(["completion", "zsh"]).shell == "zsh"
    assert parser.parse_args(["completion", "bash"]).shell == "bash"


def test_completion_scripts_are_deterministic_and_cover_cli_surface() -> None:
    required = (
        "login",
        "serve",
        "session",
        "list",
        "rename",
        "delete",
        "export",
        "automation",
        "show",
        "approve",
        "disable",
        "import",
        "daemon",
        "completion",
        "--provider",
        "--model",
        "--continue",
        "--resume",
        "--no-session",
        "--force-provider",
        "--verbose",
        "--yolo",
        "--no-yolo",
        "--token-budget",
        "--max-turns",
        "--print",
        "--format",
        "--system-prompt",
        "--append-system-prompt",
    )

    for shell in ("zsh", "bash"):
        script = completion_script(shell)
        assert script == completion_script(shell)
        assert all(value in script for value in required)


def test_completion_command_prints_without_reading_runtime_state(capsys) -> None:
    assert main(["completion", "bash"]) == 0
    output = capsys.readouterr().out

    assert output.startswith("# install: zeta completion bash")
    assert "complete -F _zeta_completions zeta" in output
