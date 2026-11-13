"""Retention-aware TUI opening."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable

from ..core.session import SessionError
from ..storage_format import StorageFormatError


def create_tui_app(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    tui_print: Callable[[str], None],
    has_console_message: Callable[[Exception], bool],
) -> object | None:
    from ..tui.bootstrap import create_app

    try:
        if args.continue_session or args.resume is not None:
            tui_print("Loading session…")
            return asyncio.run(asyncio.to_thread(create_app, args))
        return create_app(args)
    except SessionError as exc:
        parser.error(str(exc))
    except StorageFormatError:
        raise
    except ValueError as exc:
        tui_print(f"zeta: {exc}")
        return None
    except Exception as exc:
        if has_console_message(exc):
            raise
        from ..tui.loop_errors import log_tui_crash

        tui_print(f"zeta: crashed; details in {log_tui_crash(exc)}")
        return None
