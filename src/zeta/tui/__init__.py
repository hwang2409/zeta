"""Interactive terminal interface for zeta."""

from .app import TUIApp

__all__ = ["TUIApp", "main"]


def __getattr__(name: str) -> object:
    if name == "main":
        from ..cli import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
