"""Command-line entry point for zeta."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from prompt_toolkit.patch_stdout import patch_stdout

from ..config.tool_policy import parse_tool_patterns
from ..core.commands.completion import completion_script
from ..core.login_flow import run_login
from ..core.session import SessionError, env_home
from ..models.catalog import PROVIDERS, REMOVED_PROVIDER_ERROR
from ..providers.login import build_login_provider, pkce_values
from ..tui.app import create_app


def _tool_patterns_arg(value: str) -> str:
    try:
        parse_tool_patterns(value, field="tool patterns")
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return value


class _ArgumentParser(argparse.ArgumentParser):
    """Parse the ``mcp add --`` command tail independently of argparse internals."""

    def parse_args(
        self,
        args: list[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        argv = list(sys.argv[1:] if args is None else args)
        server_command: list[str] | None = None
        option_actions = {
            option: action
            for action in self._actions
            for option in action.option_strings
        }
        index = 0
        while index < len(argv):
            token = argv[index]
            if token == "--" or not token.startswith("-"):
                break
            option, has_value = token.split("=", 1) if "=" in token else (token, False)
            action = option_actions.get(option)
            if action is None:
                break
            nargs = action.nargs
            if nargs in (None, 1):
                if not has_value:
                    index += 1
            elif nargs == 0:
                if has_value:
                    break
            elif nargs == "?":
                if (
                    not has_value
                    and index + 1 < len(argv)
                    and not argv[index + 1].startswith("-")
                ):
                    index += 1
            else:
                break
            index += 1
        mcp_index = (
            index
            if index + 1 < len(argv) and argv[index : index + 2] == ["mcp", "add"]
            else None
        )
        try:
            separator = (
                argv.index("--", mcp_index + 2) if mcp_index is not None else None
            )
        except ValueError:
            separator = None
        if separator is not None:
            server_command = argv[separator + 1 :]
            del argv[separator:]
        parsed = super().parse_args(argv, namespace)
        if server_command is not None:
            parsed.server_command = server_command
        return parsed


def _provider_name(value: str) -> str:
    if value == "fake":
        raise argparse.ArgumentTypeError(
            REMOVED_PROVIDER_ERROR
        )
    if value not in PROVIDERS:
        raise argparse.ArgumentTypeError(
            f"invalid provider {value!r}; choose claude, codex or ollama"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        description="chat with the zeta harness",
        epilog="install shell completion with: zeta completion zsh > ~/.zsh/completions/_zeta",
    )
    parser.add_argument(
        "--provider",
        type=_provider_name,
        choices=("claude", "codex", "ollama"),
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
        "--auto-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="automatically reconcile durable project memory (default: on)",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=None,
        help="override the compaction/context token budget for this run",
    )
    parser.add_argument(
        "--compaction",
        choices=("summary", "evict"),
        default=None,
        help=(
            "compaction mode (default: evict; summary keeps model-written "
            "compaction); with --resume/--continue it switches the stored mode"
        ),
    )
    parser.add_argument(
        "--tools",
        metavar="PATTERN,...",
        type=_tool_patterns_arg,
        help=(
            "trusted override: replace settings allowlists and advertise only "
            "matching tool names or name(action) selectors"
        ),
    )
    parser.add_argument(
        "--disallowed-tools",
        metavar="PATTERN,...",
        type=_tool_patterns_arg,
        help=(
            "trusted override: replace the settings denylist and omit matching "
            "tool names or name(action) selectors"
        ),
    )
    parser.add_argument(
        "--require-tools",
        action="store_true",
        help=(
            "fail headless startup when an exact --tools name or "
            "name(action) capability is unavailable"
        ),
    )
    parser.add_argument(
        "--allow-hooks",
        action="store_true",
        default=None,
        help=(
            "run trusted global command hooks even when a tool allowlist is active "
            "or the bash tool is denied"
        ),
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
            "replace the base system prompt (~/.zeta/AGENTS.md and walked AGENTS.md); "
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
    from .inbox import add_subcommand as _add_inbox_subcommand
    from .panel import add_subcommand as _add_panel_subcommand
    from .project import add_subcommand as _add_project_subcommand
    from .session import add_subcommand as _add_session_subcommand

    _add_session_subcommand(commands)
    _add_project_subcommand(commands)
    _add_inbox_subcommand(commands)
    _add_panel_subcommand(commands)

    from ..automations.cli import add_subcommand as _add_automation_subcommand

    _add_automation_subcommand(commands)
    mcp = commands.add_parser("mcp", help="manage MCP servers")
    mcp_commands = mcp.add_subparsers(dest="mcp_action", required=True)
    add = mcp_commands.add_parser("add")
    add.add_argument("name")
    add.add_argument("--scope", choices=("user", "project"), default="user")
    add.add_argument("--url")
    add.add_argument("--oauth", action="store_true")
    add.add_argument("--env", action="append", default=[])
    add.add_argument("--header", action="append", default=[])
    add.add_argument("server_command", nargs="*")
    listing = mcp_commands.add_parser("list")
    listing.add_argument(
        "--scope", choices=("user", "project", "effective"), default="effective"
    )
    listing.add_argument("--json", action="store_true")
    show = mcp_commands.add_parser("show")
    show.add_argument("name")
    show.add_argument(
        "--scope", choices=("user", "project", "effective"), default="effective"
    )
    show.add_argument("--json", action="store_true")
    for action in ("remove", "enable", "disable", "trust", "untrust"):
        sub = mcp_commands.add_parser(action)
        sub.add_argument("name")
        if action not in {"trust", "untrust"}:
            sub.add_argument("--scope", choices=("user", "project"), required=True)
    for action in ("test", "login", "logout"):
        sub = mcp_commands.add_parser(action)
        sub.add_argument("name")
        sub.add_argument(
            "--scope", choices=("user", "project", "effective"), default="effective"
        )
    serve_parser = commands.add_parser(
        "serve", help="serve zeta to one local frontend client"
    )
    serve_parser.add_argument("--socket", dest="socket_path", help="Unix socket path")
    serve_parser.add_argument(
        "--port", type=int, help="listen on localhost TCP instead of a Unix socket"
    )
    serve_parser.add_argument(
        "--provider",
        dest="serve_provider",
        type=_provider_name,
        choices=("claude", "codex", "ollama"),
    )
    serve_parser.add_argument("--model", dest="serve_model")
    serve_parser.add_argument("--cwd", help="working directory for new sessions")
    serve_parser.add_argument(
        "--auto-memory",
        dest="serve_auto_memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="automatically reconcile durable project memory (default: on)",
    )
    serve_parser.add_argument(
        "--tools",
        dest="serve_tools",
        metavar="PATTERN,...",
        type=_tool_patterns_arg,
        help=(
            "trusted override: replace settings allowlists with tool-name or "
            "name(action) selectors for served sessions"
        ),
    )
    serve_parser.add_argument(
        "--disallowed-tools",
        dest="serve_disallowed_tools",
        metavar="PATTERN,...",
        type=_tool_patterns_arg,
        help=(
            "trusted override: replace the settings denylist with tool-name or "
            "name(action) selectors for served sessions"
        ),
    )
    serve_parser.add_argument(
        "--require-tools", dest="serve_require_tools", action="store_true"
    )
    serve_parser.add_argument(
        "--allow-hooks",
        dest="serve_allow_hooks",
        action="store_true",
        default=None,
        help="run trusted global command hooks in restricted served sessions",
    )
    completion_parser = commands.add_parser(
        "completion",
        help="print a static shell completion script",
        description="print a static shell completion script",
        epilog="example: zeta completion zsh > ~/.zsh/completions/_zeta",
    )
    completion_parser.add_argument("shell", choices=("zsh", "bash"))
    return parser


def _run_login(provider: str) -> str | None:
    return asyncio.run(
        run_login(build_login_provider(provider, env_home()), pkce_values)
    )


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


async def _run_tui(app: object) -> None:
    await app.run()


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
    if args.command == "mcp":
        from ..core.project_context import discover_repo_root
        from ..mcp.management import MCPManagementError, MCPManagementService

        service = MCPManagementService(project_dir=discover_repo_root(os.getcwd()))
        try:
            if args.mcp_action == "add":
                server_command = list(args.server_command)
                if server_command[:1] == ["--"]:
                    server_command.pop(0)
                command = server_command[0] if server_command else None
                result = service.add(
                    args.name,
                    scope=args.scope,
                    command=command,
                    args=tuple(server_command[1:]),
                    url=args.url,
                    oauth=args.oauth,
                    env=dict(item.split("=", 1) for item in args.env),
                    headers=dict(item.split("=", 1) for item in args.header),
                )
                print(json.dumps(result.as_json(), sort_keys=True))
                return 0
            if args.mcp_action == "list":
                values = [item.as_json() for item in service.list(scope=args.scope)]
                output = (
                    json.dumps(values, sort_keys=True)
                    if args.json
                    else "\n".join(
                        f"{item['name']}\t{item['scope']}\t{item['status']}"
                        for item in values
                    )
                )
                print(output)
                return 0
            if args.mcp_action == "show":
                value = service.show(args.name, scope=args.scope).as_json()
                output = (
                    json.dumps(value, indent=2, sort_keys=True)
                    if args.json
                    else f"{value['name']}\t{value['scope']}\t{value['status']}"
                )
                print(output)
                return 0
            if args.mcp_action in {"enable", "disable"}:
                service.set_enabled(
                    args.name,
                    scope=args.scope,
                    enabled=args.mcp_action == "enable",
                )
                return 0
            if args.mcp_action == "remove":
                service.remove(args.name, scope=args.scope)
                return 0
            if args.mcp_action == "trust":
                service.trust(args.name)
                return 0
            if args.mcp_action == "untrust":
                service.untrust(args.name)
                return 0
            if args.mcp_action == "logout":
                service.logout(args.name, scope=args.scope)
                return 0
            result = (
                asyncio.run(service.test(args.name, scope=args.scope))
                if args.mcp_action == "test"
                else asyncio.run(service.login(args.name, scope=args.scope))
            )
            if result is not None:
                print(json.dumps(result, sort_keys=True))
            return 0
        except (MCPManagementError, OSError, ValueError, KeyError) as exc:
            print(f"mcp error: {exc}", file=sys.stderr)
            return 2
    if args.command == "automation":
        from ..automations.cli import run as run_automation

        return run_automation(args)
    if args.command == "session":
        from .session import run as _run_session

        return _run_session(args)
    if args.command == "project":
        from .project import run as _run_project

        return _run_project(args)
    if args.command == "inbox":
        from .inbox import run as _run_inbox

        return _run_inbox(args)
    if args.command == "panel":
        from .panel import run as _run_panel

        return _run_panel(args)
    if args.command == "serve":
        from ..server import ZetaServer, run_server

        try:
            server = ZetaServer(
                cwd=args.cwd,
                socket_path=args.socket_path,
                port=args.port,
                provider=args.serve_provider or args.provider,
                model=args.serve_model or args.model,
                compaction=args.compaction,
                tools=args.serve_tools if args.serve_tools is not None else args.tools,
                disallowed_tools=(
                    args.serve_disallowed_tools
                    if args.serve_disallowed_tools is not None
                    else args.disallowed_tools
                ),
                require_tools=args.require_tools or args.serve_require_tools,
                allow_hooks=(
                    args.serve_allow_hooks
                    if args.serve_allow_hooks is not None
                    else args.allow_hooks
                ),
                auto_memory=(
                    args.serve_auto_memory
                    if args.serve_auto_memory is not None
                    else args.auto_memory
                ),
                cli_yolo=args.yolo,
            )
        except (SessionError, ValueError) as exc:
            print(f"zeta serve: {exc}", file=sys.stderr)
            return 2
        try:
            asyncio.run(run_server(server))
        except KeyboardInterrupt:
            return 130
        return 0
    if args.command == "completion":
        print(completion_script(args.shell), end="")
        return 0
    if args.prompt is not None:
        from ..runtime.headless import run_headless

        return run_headless(args, args.prompt)
    if args.format != "text":
        parser.error("--format requires --print")
    while True:
        try:
            if args.continue_session or args.resume is not None:
                print("Loading session…", flush=True)
                app = asyncio.run(asyncio.to_thread(create_app, args))
            else:
                app = create_app(args)
        except SessionError as exc:
            parser.error(str(exc))
        try:
            with patch_stdout(raw=True):
                asyncio.run(_run_tui(app))
            if app.new_session_requested:
                args.continue_session = False
                args.resume = None
                continue
            _print_exit_hint(app)
            return 0
        finally:
            _cleanup_ephemeral(app)


__all__ = ["build_parser", "main"]
