"""Interactive terminal interface for zeta."""

__all__ = ["TUIApp", "main"]


def __getattr__(name: str) -> object:
    if name == "TUIApp":
        from .app import TUIApp

        return TUIApp
    if name == "main":
        from ..cli.main import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
