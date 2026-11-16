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

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..tools._shared.user_discovery import PROJECT_SCOPE, USER_SCOPE

if TYPE_CHECKING:
    from ..runtime.execution import ToolHandler
    from ..tools.registry import ToolRegistry
    from .manifest import ExtensionManifest


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

    @property
    def declarations(self) -> tuple[_ToolDeclaration, ...]:
        return tuple(self._tools)


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


class ExtensionHost:
    """Mutable per-session record of extension loading."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        home: Path,
        project_dir: Path | None,
        builtin_names: frozenset[str],
    ) -> None:
        self._registry = registry
        self.home = home
        self.project_dir = project_dir
        self._builtin_names = builtin_names
        self.loaded: list[LoadedExtension] = []
        self.pending_project: list[PendingExtension] = []
        self.notices: list[str] = []
        self.warnings: list[str] = []

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
