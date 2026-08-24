"""Command-line entry point for zeta."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from prompt_toolkit.patch_stdout import patch_stdout

from .core.login_flow import LoginError, LoginProvider, run_login
from .core.session import SessionError, env_home
from .tui.app import create_app
from .providers.anthropic import (
    AnthropicCredentialStore,
    build_authorization_url as build_anthropic_authorization_url,
    exchange_authorization_code as exchange_anthropic_authorization_code,
)
from .providers.auth import OAuthTokens, build_pkce_parameters
from .providers.codex import (
    CodexCredentialStore,
    build_authorization_url as build_codex_authorization_url,
    exchange_authorization_code as exchange_codex_authorization_code,
    extract_account_id,
)


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
    session_group.add_argument("--resume", help="resume a session by id")
    parser.add_argument(
        "--force-provider",
        action="store_true",
        help="allow provider or model overrides during resume",
    )
    parser.add_argument("--verbose", action="store_true", help="show raw stream events")
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="auto-approve every tool call (skip approval prompts)",
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
    commands = parser.add_subparsers(dest="command")
    login_parser = commands.add_parser("login", help="log in to an OAuth provider")
    login_parser.add_argument(
        "--provider",
        choices=("anthropic", "codex"),
        default="anthropic",
        help="OAuth provider (default: anthropic)",
    )
    return parser


def _pkce_values() -> tuple[str, str, str]:
    parameters = build_pkce_parameters()
    return parameters.verifier, parameters.challenge, parameters.state


def _no_handle(tokens: OAuthTokens) -> str | None:
    del tokens
    return None


def _build_login_provider(provider: str) -> LoginProvider[OAuthTokens]:
    home = env_home()
    if provider == "anthropic":
        return LoginProvider(
            name="anthropic",
            build_authorization_url=build_anthropic_authorization_url,
            exchange_authorization_code=exchange_anthropic_authorization_code,
            credential_store=AnthropicCredentialStore(home / "anthropic-oauth.json"),
            token_handle=_no_handle,
        )
    if provider == "codex":
        return LoginProvider(
            name="codex",
            build_authorization_url=build_codex_authorization_url,
            exchange_authorization_code=exchange_codex_authorization_code,
            credential_store=CodexCredentialStore(home / "codex-oauth.json"),
            token_handle=lambda tokens: extract_account_id(tokens.access_token),
        )
    raise ValueError(f"unsupported login provider: {provider}")


def _run_login(provider: str) -> str | None:
    return asyncio.run(run_login(_build_login_provider(provider), _pkce_values))


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
    if args.force_provider and args.model is None:
        parser.error("--force-provider requires --model")
    try:
        app = create_app(args)
    except SessionError as exc:
        parser.error(str(exc))
    try:
        with patch_stdout(raw=True):
            asyncio.run(app.run())
    finally:
        try:
            os.write(sys.__stdout__.fileno(), b"\x1b[?1049l\x1b[?25h")
        except OSError:
            pass
    return 0


__all__ = ["build_parser", "main"]
