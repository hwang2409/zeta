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

import asyncio
import importlib.abc
import importlib.machinery
import importlib.util
import itertools
import re
import sys
import threading
import time
import types
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..protocol.types import StreamEvent, StreamEventType
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


@dataclass(slots=True)
class ExtensionStatus:
    name: str
    scope: str
    version: str
    enabled: bool
    trusted: bool
    loaded: bool = False
    error: str | None = None


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


class _SourceExtensionLoader(importlib.machinery.SourceFileLoader):
    """Compile extension source directly without bytecode caches."""

    def get_code(self, fullname: str) -> object:
        source = self.get_data(self.get_filename(fullname))
        return compile(source, self.get_filename(fullname), "exec", dont_inherit=True)


class _ExtensionFinder(importlib.abc.MetaPathFinder):
    def __init__(self, package: str, directory: Path) -> None:
        self.package = package
        self.directory = directory

    def find_spec(self, fullname: str, path: object = None, target: object = None):
        prefix = self.package + "."
        if not fullname.startswith(prefix):
            return None
        relative = fullname[len(prefix):].split(".")
        if any(not part.isidentifier() for part in relative):
            return None
        base = self.directory.joinpath(*relative)
        source = base / "__init__.py" if base.is_dir() else base.with_suffix(".py")
        if not source.is_file():
            return None
        loader = _SourceExtensionLoader(fullname, str(source))
        locations = [str(base)] if source.name == "__init__.py" else None
        return importlib.util.spec_from_file_location(
            fullname, source, loader=loader, submodule_search_locations=locations
        )


class ExtensionHost:
    """Mutable per-session record of extension loading."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        home: Path,
        project_dir: Path | None,
        builtin_names: frozenset[str],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._registry = registry
        self.home = home
        self.project_dir = project_dir
        self._builtin_names = builtin_names
        self.loaded: list[LoadedExtension] = []
        self.pending_project: list[PendingExtension] = []
        self.notices: list[str] = []
        self.warnings: list[str] = []
        self.statuses: dict[tuple[str, str], ExtensionStatus] = {}
        self._event_sink: Callable[[StreamEvent], None] | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self.notify_dropped = 0
        self._pending_notifications: deque[StreamEvent] = deque()
        self._pending_notification_drops: int = 0
        self._event_thread_id: int | None = None
        self._foreign_thread_warnings: set[str] = set()
        self._notification_clock = clock
        self._notification_buckets: dict[str, tuple[float, float, int]] = {}
        self._packages: dict[str, Path] = {}
        self._finders: dict[str, _ExtensionFinder] = {}
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
        finder = _ExtensionFinder(package, directory)
        self._finders[package] = finder
        sys.meta_path.insert(0, finder)
        return package, module

    def remove_package_for_directory(self, directory: Path) -> None:
        for package, package_directory in tuple(self._packages.items()):
            if package_directory == directory:
                self.remove_package(package)

    def remove_package(self, package: str) -> None:
        self._packages.pop(package, None)
        finder = self._finders.pop(package, None)
        if finder is not None and finder in sys.meta_path:
            sys.meta_path.remove(finder)
        for name in tuple(sys.modules):
            if name == package or name.startswith(package + "."):
                del sys.modules[name]

    def close(self) -> None:
        """Release imported extension namespaces; safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        self._event_sink = None
        self._event_loop = None
        self._pending_notifications.clear()
        self._pending_notification_drops = 0

        for package in tuple(self._packages):
            self.remove_package(package)

    def set_event_sink(
        self, sink: Callable[[StreamEvent], None] | None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._event_sink = sink
        if sink is None or loop is None:
            return
        if self._event_loop is None:
            self._event_loop = loop
            self._event_thread_id = threading.get_ident()
        if self._event_loop is not loop:
            return
        pending = tuple(self._pending_notifications)
        self._pending_notifications.clear()
        dropped = self._pending_notification_drops
        self._pending_notification_drops = 0
        for event in pending:
            loop.call_soon(self._send, event)
        if dropped:
            loop.call_soon(
                self._send,
                StreamEvent(
                    StreamEventType.AGENT_NOTIFICATION,
                    data={
                        "kind": "extension",
                        "extension": "extensions",
                        "text": f"{dropped} extension notices dropped",
                        "level": "warning",
                    },
                ),
            )

    def _send(self, event: StreamEvent) -> None:
        sink = self._event_sink
        if self._closed or sink is None:
            self.notify_dropped += 1
            return
        try:
            sink(event)
        except Exception:  # noqa: BLE001 - extension notices must not break a session
            self.notify_dropped += 1

    def notify(self, extension: str, text: str, level: str) -> None:
        if type(text) is not str or level not in {"info", "warning", "error"}:
            raise ValueError("invalid extension notification")
        if len(text) > 2000:
            text = text[:1997] + "..."
        event = StreamEvent(
            StreamEventType.AGENT_NOTIFICATION,
            data={"kind": "extension", "extension": extension, "text": text, "level": level},
        )

        if self._closed:
            self.notify_dropped += 1
            return
        if self._event_loop is not None and threading.get_ident() != self._event_thread_id:
            self.notify_dropped += 1
            if extension not in self._foreign_thread_warnings:
                self._foreign_thread_warnings.add(extension)
                self.warnings.append(
                    f"extension {extension} notification dropped from a foreign thread"
                )
            return
        now = self._notification_clock()
        last, tokens, dropped = self._notification_buckets.get(extension, (now, 20.0, 0))
        tokens = min(20.0, tokens + max(0.0, now - last) * 10.0)
        if tokens < 1.0:
            self._notification_buckets[extension] = (now, tokens, dropped + 1)
            self.notify_dropped += 1
            return
        tokens -= 1.0
        if dropped:
            event = StreamEvent(
                StreamEventType.AGENT_NOTIFICATION,
                data={
                    "kind": "extension", "extension": extension,
                    "text": f"{dropped} extension notices dropped", "level": "warning",
                },
            )
            dropped = 0
        self._notification_buckets[extension] = (now, tokens, dropped)
        if self._event_sink is None or self._event_loop is None:
            if len(self._pending_notifications) < 32:
                self._pending_notifications.append(event)
            else:
                self._pending_notification_drops += 1
                self.notify_dropped += 1
            return
        self._event_loop.call_soon(self._send, event)

    def add_status(self, manifest: ExtensionManifest, scope: str, *, enabled: bool) -> None:
        self.statuses[(manifest.name, scope)] = ExtensionStatus(
            manifest.name, scope, manifest.version, enabled, scope == USER_SCOPE
        )

    def mark_loaded(self, name: str, scope: str) -> None:
        status = self.statuses.get((name, scope))
        if status is not None:
            status.loaded = True
            status.error = None

    def mark_error(self, name: str, error: str, scope: str) -> None:
        status = self.statuses.get((name, scope))
        if status is not None:
            status.error = error

    def mark_trusted(self, name: str, scope: str) -> None:
        status = self.statuses.get((name, scope))
        if status is not None:
            status.trusted = True

    def status_rows(self) -> list[dict[str, object]]:
        return [{
            "name": item.name, "scope": item.scope, "version": item.version,
            "enabled": item.enabled, "trusted": item.trusted, "loaded": item.loaded,
            "error": item.error,
        } for item in self.statuses.values()]

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
        self.mark_loaded(extension.manifest.name, extension.scope)
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
    "ExtensionStatus",
    "LoadedExtension",
    "LoadingExtension",
    "PendingExtension",
]
