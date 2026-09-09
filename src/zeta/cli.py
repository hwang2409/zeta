"""Command-line entry point for zeta."""

from __future__ import annotations

import argparse
import asyncio
import sys

from prompt_toolkit.patch_stdout import patch_stdout

from .core.login_flow import run_login
from .core.session import SessionError, env_home
from .providers.login import build_login_provider, pkce_values
from .tui.app import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="chat with the zeta harness")
    parser.add_argument(
        "--provider",
        choices=("fake", "claude", "codex"),
        help="completion provider",
    )
    parser.add_argument("--model", help="provider model override")
    session_group = parser.add_mutually_exclusive_group()
    session_group.add_argument(
        "--continue",
        "-c",
        dest="continue_session",
        action="store_true",
        help="resume the most recent session in this directory",
    )
    session_group.add_argument(
        "--resume",
        nargs="?",
        const="",
        help="resume a session by id, or choose one from the recent-session picker",
    )
    session_group.add_argument(
        "--no-session",
        dest="no_session",
        action="store_true",
        help="run one ephemeral session; nothing is written to the sessions store",
    )
    parser.add_argument(
        "--force-provider",
        action="store_true",
        help="allow provider or model overrides during resume",
    )
    parser.add_argument("--verbose", action="store_true", help="show raw stream events")
    parser.add_argument(
        "--yolo",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "auto-approve every tool call; --no-yolo forces prompts even "
            "when settings.toml enables yolo"
        ),
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=None,
        help="override the compaction/context token budget for this run",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="cap the assistant's tool-use loop turns per user message",
    )
    parser.add_argument(
        "-p",
        "--print",
        dest="prompt",
        metavar="PROMPT",
        default=None,
        help=(
            "run one turn without the TUI: print the final assistant text "
            "and exit; combine with --format json to emit a JSONL event stream"
        ),
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="headless output format (text or json); requires --print",
    )
    parser.add_argument(
        "--system-prompt",
        dest="system_prompt",
        metavar="TEXT-OR-@FILE",
        default=None,
        help=(
            "replace the built-in system prompt (identity and walked AGENTS.md); "
            "pass @path to load from a file. Overrides ~/.zeta/SYSTEM.md and "
            "drops any --append-system-prompt for this session"
        ),
    )
    parser.add_argument(
        "--append-system-prompt",
        dest="append_system_prompt",
        metavar="TEXT-OR-@FILE",
        default=None,
        help=(
            "append text after the default system prompt; pass @path to load "
            "from a file. Overrides ~/.zeta/APPEND_SYSTEM.md. Ignored when "
            "--system-prompt is set"
        ),
    )
    commands = parser.add_subparsers(dest="command")
    login_parser = commands.add_parser("login", help="log in to an OAuth provider")
    login_parser.add_argument(
        "--provider",
        choices=("anthropic", "codex"),
        default="anthropic",
        help="OAuth provider (default: anthropic)",
    )
    from .session_cli import add_subcommand as _add_session_subcommand

    _add_session_subcommand(commands)
    serve_parser = commands.add_parser(
        "serve", help="serve zeta to one local frontend client"
    )
    serve_parser.add_argument(
        "--socket", dest="socket_path", help="Unix socket path"
    )
    serve_parser.add_argument(
        "--port", type=int, help="listen on localhost TCP instead of a Unix socket"
    )
    serve_parser.add_argument("--provider", dest="serve_provider", choices=("fake", "claude", "codex"))
    serve_parser.add_argument("--model", dest="serve_model")
    serve_parser.add_argument("--cwd", help="working directory for new sessions")
    return parser


def _run_login(provider: str) -> str | None:
    return asyncio.run(run_login(build_login_provider(provider, env_home()), pkce_values))


def _cleanup_ephemeral(app: object) -> None:
    import shutil

    root = app.ephemeral_root
    if root is None:
        return
    shutil.rmtree(root, ignore_errors=True)


def _print_exit_hint(app: object) -> None:
    if app.ephemeral_root is not None:
        return
    print(f"resume with: zeta --resume {app.loop.store.session_id}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "login":
        try:
            handle = _run_login(args.provider)
        except KeyboardInterrupt:
            print("login cancelled", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"login failed: {exc}", file=sys.stderr)
            return 1
        print(f"logged in as {handle}" if handle else "ok")
        return 0
    if args.command == "session":
        from .session_cli import run as _run_session

        return _run_session(args)
    if args.command == "serve":
        from .server import ZetaServer, run_server

        server = ZetaServer(
            cwd=args.cwd,
            socket_path=args.socket_path,
            port=args.port,
            provider=args.serve_provider or args.provider,
            model=args.serve_model or args.model,
        )
        try:
            asyncio.run(run_server(server))
        except KeyboardInterrupt:
            return 130
        return 0
    if args.prompt is not None:
        from .headless import run_headless

        return run_headless(args, args.prompt)
    if args.format != "text":
        parser.error("--format requires --print")
    while True:
        try:
            app = create_app(args)
        except SessionError as exc:
            parser.error(str(exc))
        try:
            with patch_stdout(raw=True):
                asyncio.run(app.run())
            if app.new_session_requested:
                args.continue_session = False
                args.resume = None
                continue
            _print_exit_hint(app)
            return 0
        finally:
            _cleanup_ephemeral(app)


__all__ = ["build_parser", "main"]
