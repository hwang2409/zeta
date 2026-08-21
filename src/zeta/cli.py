"""Command-line entry point for zeta."""

from __future__ import annotations

import argparse
import asyncio

from prompt_toolkit.patch_stdout import patch_stdout

from .core.session import SessionError
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
    session_group.add_argument("--resume", help="resume a session by id")
    parser.add_argument(
        "--force-provider",
        action="store_true",
        help="allow provider or model overrides during resume",
    )
    parser.add_argument("--verbose", action="store_true", help="show raw stream events")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.force_provider and args.model is None:
        parser.error("--force-provider requires --model")
    try:
        app = create_app(args)
    except SessionError as exc:
        parser.error(str(exc))
    with patch_stdout(raw=True):
        asyncio.run(app.run())
    return 0


__all__ = ["build_parser", "main"]
