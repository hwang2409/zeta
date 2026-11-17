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

import itertools
import os
import re
import stat
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


@dataclass(frozen=True, slots=True)
class _ToolDeclaration:
    name: str
    handler: ToolHandler
    kwargs: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _CommandDeclaration:
    name: str
    description: str
    handler: Any
    body: str | None
    kind: str


class LoadingExtension:
    """One extension while its ``activate`` runs; collects declarations."""

    def __init__(self, manifest: ExtensionManifest, scope: str, directory: Path) -> None:
        self.manifest = manifest
        self.scope = scope
        self.directory = directory
        self._tools: list[_ToolDeclaration] = []
        self._commands: list[_CommandDeclaration] = []

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

    def declare_command(
        self, name: str, description: str, handler: Any, body: str | None, kind: str
    ) -> None:
        self._commands.append(_CommandDeclaration(name, description, handler, body, kind))

    @property
    def declarations(self) -> tuple[_ToolDeclaration, ...]:
        return tuple(self._tools)

    @property
    def command_declarations(self) -> tuple[_CommandDeclaration, ...]:
        return tuple(self._commands)


@dataclass(frozen=True, slots=True)
class LoadedExtension:
    """An extension that activated and committed its declarations."""

    name: str
    version: str
    scope: str
    directory: Path
    tool_names: tuple[str, ...]
    command_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PendingExtension:
    """An extension discovered but not loaded in this session."""

    manifest: ExtensionManifest
    directory: Path
    scope: str


@dataclass(frozen=True, slots=True)
class ExtensionStatus:
    """Current public state of one discovered extension."""

    name: str
    version: str
    scope: str
    state: str
    tool_names: tuple[str, ...] = ()
    command_names: tuple[str, ...] = ()


_NAMESPACE = "_zeta_extensions"
_LOAD_NUMBER = itertools.count()
_DISABLED_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


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
        self._packages: dict[str, Path] = {}
        self._command_declarations: dict[str, tuple[_CommandDeclaration, ...]] = {}
        self._disabled: set[str] = set()
        self._settings_disabled: set[str] = set()
        self.disabled_inventory: list[PendingExtension] = []
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

    def register_module(self, package: str, name: str, module: types.ModuleType) -> None:
        """Register an imported entry module under a host-owned namespace."""
        if package not in self._packages:
            raise ValueError(f"unknown extension package: {package}")
        sys.modules[f"{package}.{name}"] = module

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
        self._command_declarations[extension.manifest.name] = extension.command_declarations
        self.loaded.append(
            LoadedExtension(
                name=extension.manifest.name,
                version=extension.manifest.version,
                scope=extension.scope,
                directory=extension.directory,
                tool_names=tuple(added),
                command_names=tuple(item.name for item in extension.command_declarations),
            )
        )

    @property
    def disabled(self) -> frozenset[str]:
        return frozenset(self._disabled)

    def load_disabled_markers(self) -> None:
        directory = self._disabled_directory(create=False)
        if directory is None:
            return
        for entry in directory.iterdir():
            if _DISABLED_NAME_RE.fullmatch(entry.name) is None:
                continue
            try:
                exists = self._disabled_marker(entry, "read")
            except OSError as exc:
                self.notices.append(f"extensions · cannot read disabled marker {entry.name}: {exc}")
                continue
            if exists:
                self._disabled.add(entry.name)

    def persist_disabled(self, name: str, enabled: bool) -> bool:
        """Create or remove one zeta-owned disabled marker."""
        if _DISABLED_NAME_RE.fullmatch(name) is None:
            raise ValueError(f"invalid extension name: {name!r}")
        if name in self._settings_disabled and enabled:
            return False
        directory = self._disabled_directory(create=True)
        exists = self._disabled_marker(directory / name, "remove" if enabled else "create")
        if exists:
            self._disabled.add(name)
        else:
            self._disabled.discard(name)
        return True

    def _disabled_marker(self, marker: Path, action: str) -> bool:
        if action == "create":
            try:
                fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                action = "read"
            else:
                os.close(fd)
                return True
        try:
            mode = marker.lstat().st_mode
        except FileNotFoundError:
            return False
        if not stat.S_ISREG(mode):
            raise OSError(f"disabled marker is not a file: {marker.name}")
        if action == "remove":
            try:
                marker.unlink()
            except FileNotFoundError:
                pass
            return False
        return True

    def statuses(self) -> tuple[ExtensionStatus, ...]:
        """Project current loading state into one record per discovered name."""
        records: list[ExtensionStatus] = []
        seen: set[str] = set()
        for item in self.loaded:
            if item.name not in seen:
                seen.add(item.name)
                records.append(self._loaded_status(item))
        for item in (*self.disabled_inventory, *self.pending_project):
            if item.manifest.name not in seen:
                seen.add(item.manifest.name)
                records.append(self._pending_status(item))
        return tuple(records)

    def status(self, name: str) -> ExtensionStatus | None:
        return next((record for record in self.statuses() if record.name == name), None)

    def _loaded_status(self, item: LoadedExtension) -> ExtensionStatus:
        return ExtensionStatus(
            name=item.name, version=item.version, scope=item.scope,
            state=self._state(item.name, loaded=True),
            tool_names=item.tool_names, command_names=item.command_names,
        )

    def _pending_status(self, item: PendingExtension) -> ExtensionStatus:
        return ExtensionStatus(
            name=item.manifest.name, version=item.manifest.version, scope=item.scope,
            state=self._state(
                item.manifest.name,
                loaded=False,
                disabled_inventory=item in self.disabled_inventory,
            ),
        )

    def _state(self, name: str, *, loaded: bool, disabled_inventory: bool = False) -> str:
        if name in self._settings_disabled:
            return "disabled_in_settings"
        if name in self._disabled:
            return "disabled_next_session"
        if disabled_inventory:
            return "enabled_next_session"
        return "loaded" if loaded else "pending"

    def _disabled_directory(self, *, create: bool) -> Path | None:
        directory = self.home / "extensions-disabled"
        try:
            mode = directory.lstat().st_mode
        except FileNotFoundError:
            if not create:
                return None
            directory.mkdir(parents=True, mode=0o700, exist_ok=True)
            mode = directory.lstat().st_mode
        if directory.is_symlink() or not stat.S_ISDIR(mode):
            raise OSError("extensions-disabled must be a real directory")
        os.chmod(directory, 0o700)
        return directory

    def loaded_declarations(self, name: str) -> tuple[_CommandDeclaration, ...]:
        return self._command_declarations.get(name, ())

    def register_commands(self, registry: Any) -> None:
        """Add activated extension commands to a shared slash registry."""
        from .commands import register_commands

        register_commands(registry, self)

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
    "ExtensionStatus",
    "LoadedExtension",
    "LoadingExtension",
    "PendingExtension",
]
