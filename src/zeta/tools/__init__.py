"""Built-in and custom tool registration."""

from .registry import (
    AbortSignal,
    ToolAbortSignal,
    ToolDefinition,
    ToolHandler,
    ToolRegistry,
)


def register_default_tools(registry: ToolRegistry) -> None:
    from . import exec, list, read, write

    read.register(registry)
    list.register(registry)
    exec.register(registry)
    write.register(registry)


__all__ = [
    "AbortSignal",
    "ToolAbortSignal",
    "ToolDefinition",
    "ToolHandler",
    "ToolRegistry",
    "register_default_tools",
]
