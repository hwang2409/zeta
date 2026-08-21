"""Compatibility imports for the agent loop."""

from __future__ import annotations

def __getattr__(name: str) -> object:
    if name == "AgentLoop":
        from ..loop import AgentLoop

        return AgentLoop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["AgentLoop"]
