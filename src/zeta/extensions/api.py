"""The public surface an extension's ``activate(zx)`` receives.

An extension author imports nothing from zeta internals: they write a
synchronous ``activate(zx: ExtensionAPI) -> None`` entry function and call
methods on ``zx``. The entry function is synchronous and only *records*
declarations (it does not run the session), mirroring the existing
``register(registry)`` contract for ``~/.zeta/tools`` tools.

Slice 1 exposes ``register_tool`` only. Later slices add commands, event
hooks, and UI primitives on this same object.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..runtime.execution import ToolHandler
    from .host import ExtensionHost, LoadingExtension

# Host capability version. Bump when the ExtensionAPI grows a surface. A
# manifest whose ``api_version`` is greater than this is refused at load.
EXTENSION_API_VERSION = 1


class ExtensionError(RuntimeError):
    """An extension used the API incorrectly."""


class ExtensionAPI:
    """Handed to one extension's ``activate``; bound to one session."""

    def __init__(self, host: ExtensionHost, extension: LoadingExtension) -> None:
        self._host = host
        self._extension = extension

    @property
    def name(self) -> str:
        return self._extension.manifest.name

    @property
    def scope(self) -> str:
        return self._extension.scope

    @property
    def api_version(self) -> int:
        return EXTENSION_API_VERSION

    @property
    def home(self) -> Path:
        return self._host.home

    @property
    def project_dir(self) -> Path | None:
        return self._host.project_dir

    def register_tool(
        self,
        name: str,
        handler: ToolHandler,
        *,
        description: str = "",
        parameters: Mapping[str, Any] | None = None,
        parallel_safe: bool = False,
        requires_approval: bool = True,
        actions: Mapping[str, Any] | None = None,
        binding: str = "loop",
    ) -> None:
        """Declare a tool. Committed to the session registry after activate.

        A thin pass-through to ``ToolRegistry.register``: same validation and
        approval contract. Nothing is registered until ``activate`` returns
        without raising, so a half-finished activate registers no tools.
        """

        if type(name) is not str or not name:
            raise ExtensionError("register_tool name must be a nonempty string")
        if not callable(handler):
            raise ExtensionError("register_tool handler must be callable")
        self._extension.declare_tool(
            name,
            handler,
            description=description,
            parameters=parameters,
            parallel_safe=parallel_safe,
            requires_approval=requires_approval,
            actions=actions,
            binding=binding,
        )


__all__ = ["EXTENSION_API_VERSION", "ExtensionAPI", "ExtensionError"]
