"""Per-session record of loaded extensions and their tool declarations.

An ``ExtensionHost`` is built once per top-level session (children do not load
extensions, matching ``~/.zeta/tools``). It stages each extension's tool
declarations while ``activate`` runs and commits them to the session's
``ToolRegistry`` only if activate returns without raising. Committing applies
the same precedence rule as external tools: a project-scope extension may never
replace a built-in tool (loud REJECTED notice; the built-in stays), and any
shadow of an existing tool is recorded as a notice so it is visible at session
start.

Every failure fails OPEN: a bad manifest, an import error, an activate that
raises, or a bad tool declaration becomes a notice and the extension is
skipped. One broken extension never breaks the session or stops the rest.
"""

from __future__ import annotations

import inspect
import itertools
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..tools._shared.user_discovery import PROJECT_SCOPE, USER_SCOPE

if TYPE_CHECKING:
    from ..runtime.execution import ToolHandler
    from ..tools.registry import ToolRegistry
    from .manifest import ExtensionManifest

from .api import ExtensionError
from .hooks import (
    CompositeToolHook,
    ExtensionHook,
    TaskOwner,
    _missing_task_owner,
    run_extension_hooks,
)


@dataclass(frozen=True, slots=True)
class _ToolDeclaration:
    name: str
    handler: ToolHandler
    kwargs: dict[str, Any]


class LoadingExtension:
    """One extension while its ``activate`` runs; collects declarations."""

    def __init__(self, manifest: ExtensionManifest, scope: str, directory: Path) -> None:
        self.manifest = manifest
        self.scope = scope
        self.directory = directory
        self._tools: list[_ToolDeclaration] = []
        self._hooks: list[ExtensionHook] = []

    def declare_tool(
        self,
        name: str,
        handler: ToolHandler,
        *,
        description: str,
        parameters: Any,
        parallel_safe: bool,
        requires_approval: bool,
        actions: Any,
        binding: str,
    ) -> None:
        kwargs: dict[str, Any] = {
            "description": description,
            "parallel_safe": parallel_safe,
            "requires_approval": requires_approval,
            "binding": binding,
        }
        if parameters is not None:
            kwargs["parameters"] = parameters
        if actions is not None:
            kwargs["actions"] = actions
        self._tools.append(_ToolDeclaration(name, handler, kwargs))

    def declare_hook(self, event: str, handler: Any) -> None:
        if not inspect.iscoroutinefunction(handler):
            raise ExtensionError(
                f"extension hook '{event}' must be an async function"
            )
        self._hooks.append(ExtensionHook(self.manifest.name, event, handler))

    @property
    def declarations(self) -> tuple[_ToolDeclaration, ...]:
        return tuple(self._tools)

    @property
    def hooks(self) -> tuple[ExtensionHook, ...]:
        return tuple(self._hooks)


@dataclass(frozen=True, slots=True)
class LoadedExtension:
    """An extension that activated and committed its declarations."""

    name: str
    version: str
    scope: str
    directory: Path
    tool_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PendingExtension:
    """A project-scope extension discovered but not yet trusted (not loaded)."""

    manifest: ExtensionManifest
    directory: Path


_NAMESPACE = "_zeta_extensions"
_LOAD_NUMBER = itertools.count()


class ExtensionHost:
    """Mutable per-session record of extension loading."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        home: Path,
        project_dir: Path | None,
        builtin_names: frozenset[str],
        hook_timeout_seconds: float = 5.0,
        task_owner: TaskOwner | None = None,
    ) -> None:
        self._registry = registry
        self.home = home
        self.project_dir = project_dir
        self._builtin_names = builtin_names
        self.hook_timeout_seconds = hook_timeout_seconds
        self._task_owner = task_owner or _missing_task_owner
        self.notice_sink: Any = None
        self._hooks: list[ExtensionHook] = []
        self.loaded: list[LoadedExtension] = []
        self.pending_project: list[PendingExtension] = []
        self.notices: list[str] = []
        self.warnings: list[str] = []
        self._packages: dict[str, Path] = {}
        self._closed = False

    def new_package(self, directory: Path, extension_name: str) -> tuple[str, types.ModuleType]:
        safe = re.sub(r"[^0-9A-Za-z_]", "_", extension_name)
        package = f"{_NAMESPACE}.{safe}_{next(_LOAD_NUMBER):x}"
        if _NAMESPACE not in sys.modules:
            namespace = types.ModuleType(_NAMESPACE)
            namespace.__path__ = []  # type: ignore[attr-defined]
            sys.modules[_NAMESPACE] = namespace
        module = types.ModuleType(package)
        module.__path__ = [str(directory)]  # type: ignore[attr-defined]
        module.__package__ = package
        sys.modules[package] = module
        self._packages[package] = directory
        return package, module

    def remove_package_for_directory(self, directory: Path) -> None:
        for package, package_directory in tuple(self._packages.items()):
            if package_directory == directory:
                self.remove_package(package)

    def remove_package(self, package: str) -> None:
        self._packages.pop(package, None)
        for name in tuple(sys.modules):
            if name == package or name.startswith(package + "."):
                del sys.modules[name]

    def close(self) -> None:
        """Release imported extension namespaces; safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        for package in tuple(self._packages):
            self.remove_package(package)

    def set_notice_sink(self, sink: Any) -> None:
        self.notice_sink = sink

    def _notice(self, message: str) -> None:
        self.notices.append(message)
        if self.notice_sink is not None:
            self.notice_sink(message)

    def pre_tool_hook(self, base: Any = None) -> CompositeToolHook:
        return CompositeToolHook(base, self._hooks, self._notice, self._task_owner)

    async def emit(self, event: str, *payload: Any) -> str | None:
        return await run_extension_hooks(
            self._hooks, event, payload, self._notice, self._task_owner
        )

    def commit(self, extension: LoadingExtension) -> None:
        """Register a successfully activated extension's tools."""

        display = _display(extension)
        added: list[str] = []
        for declaration in extension.declarations:
            name = declaration.name
            if extension.scope == PROJECT_SCOPE and name in self._builtin_names:
                self.notices.append(
                    f"extensions · REJECTED {extension.scope} extension tool "
                    f"'{name}' from {display}: cannot replace built-in tool"
                )
                continue
            shadow = self._shadow_phrase(name)
            try:
                self._registry.register(name, declaration.handler, **declaration.kwargs)
            except Exception as exc:  # noqa: BLE001 - one bad tool must not break the rest
                self.notices.append(
                    f"extensions · ignored tool '{name}' from {display}: {_short(exc)}"
                )
                continue
            added.append(name)
            if shadow is not None:
                self.notices.append(
                    f"extensions · {extension.scope} extension tool '{name}' from "
                    f"{display} shadows {shadow}"
                )
        for hook in extension.hooks:
            hook.timeout_seconds = self.hook_timeout_seconds
            self._hooks.append(hook)
        self.loaded.append(
            LoadedExtension(
                name=extension.manifest.name,
                version=extension.manifest.version,
                scope=extension.scope,
                directory=extension.directory,
                tool_names=tuple(added),
            )
        )

    def _shadow_phrase(self, name: str) -> str | None:
        if name in self._builtin_names:
            return "built-in"
        if name in self._registry.registered_names:
            return "an existing tool"
        return None


def _display(extension: LoadingExtension) -> str:
    try:
        home = Path.home()
    except (RuntimeError, OSError):
        return extension.manifest.name
    try:
        return f"~/{extension.directory.relative_to(home)}"
    except ValueError:
        return str(extension.directory)


def _short(exc: BaseException) -> str:
    text = str(exc) or exc.__class__.__name__
    text = text.replace("\n", " ")
    if len(text) > 200:
        text = text[:200] + "…"
    return text


__all__ = [
    "PROJECT_SCOPE",
    "USER_SCOPE",
    "ExtensionHost",
    "LoadedExtension",
    "LoadingExtension",
    "PendingExtension",
]
