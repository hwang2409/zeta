"""Discover, import, and activate zeta extensions for one session.

An extension is a directory that contains a ``zeta-extension.toml`` manifest
and an entry module. Two scopes are scanned, mirroring ``~/.zeta/tools`` and
``<project>/.zeta/tools``:

* user -- ``<home>/extensions/*/``. Imported and activated directly in a
  session (full power, the user's own machine).
* project -- ``<project>/.zeta/extensions/*/``. Imported only when the
  installed digest matches its durable project-scope trust record.

Import runs synchronously at compose time, exactly like ``~/.zeta/tools`` and
``apply_external_tools`` today, so a blocking module import delays session
start (not the steady-state loop). This is acceptable for the same reason it is
for external tools; the per-handler timeout for hooks (a later slice) does not
cover load.

Every failure fails OPEN with a notice; one broken extension never stops the
rest or breaks the session.
"""

from __future__ import annotations

import importlib.util
import inspect
import time
from collections.abc import Callable
from pathlib import Path

from .api import EXTENSION_API_VERSION, ExtensionAPI
from .host import (
    PROJECT_SCOPE,
    USER_SCOPE,
    ExtensionHost,
    LoadingExtension,
    PendingExtension,
)
from .manifest import ExtensionManifest
from .store import ExtensionStore, InstalledExtension

EXTENSIONS_DIRNAME = "extensions"


def apply_extensions(
    registry: object,
    *,
    home: str | Path | None,
    project_dir: str | Path | None,
    builtin_names: frozenset[str],
    disabled: tuple[str, ...] = (),
    clock: Callable[[], float] = time.monotonic,
) -> ExtensionHost:
    """Load trusted canonical installed extensions through ``ExtensionStore``."""

    home_path = _as_path(home)
    project_path = _as_path(project_dir)
    host = ExtensionHost(
        registry,  # type: ignore[arg-type]
        home=home_path,
        project_dir=project_path,
        builtin_names=builtin_names,
        clock=clock,
    )
    host._settings_disabled = set(disabled)
    host._disabled = set(disabled)
    if home_path is None:
        return host
    try:
        host.load_disabled_markers()
    except OSError as exc:
        host.notices.append(f"extensions · cannot read disabled markers: {exc}")
    store = ExtensionStore(home=home_path, project_dir=project_path)
    for scope in (USER_SCOPE, PROJECT_SCOPE):
        if scope == PROJECT_SCOPE and project_path is None:
            continue
        result = store.discover(scope)
        for notice in result.notices:
            if notice.startswith("extension loading disabled"):
                host.notify("extensions", notice, "warning")
            else:
                host.notices.append(f"extensions · {notice}")
        for item in result.extensions:
            _apply_item(host, item)
    if host.pending_project:
        names = ", ".join(sorted(item.manifest.name for item in host.pending_project))
        host.warnings.append(
            "extensions · untrusted project extensions present: "
            f"{names}; trust them to enable them for this session"
        )
    return host


def _apply_item(host: ExtensionHost, item: InstalledExtension) -> None:
    manifest = item.manifest
    if manifest.name in host._disabled:
        host.disabled_inventory.append(PendingExtension(manifest, item.directory, item.scope))
        host.notices.append(f"extensions · {manifest.name} disabled by settings")
        return
    if manifest.api_version > EXTENSION_API_VERSION:
        host.notices.append(
            f"extensions · {manifest.name} ({_display(item.directory)}) built for "
            f"extension API v{manifest.api_version}; this zeta supports "
            f"v{EXTENSION_API_VERSION} — upgrade zeta"
        )
        return
    if item.scope == PROJECT_SCOPE and not item.trusted:
        host.pending_project.append(PendingExtension(manifest, item.directory, item.scope))
        return
    _load_one(host, item.directory, manifest, item.scope)


def _skip(
    host: ExtensionHost,
    manifest: ExtensionManifest,
    directory: Path,
    scope: str,
    disabled: frozenset[str],
) -> bool:
    if manifest.name in disabled:
        host.disabled_inventory.append(PendingExtension(manifest, directory, scope))
        host.notices.append(f"extensions · {manifest.name} disabled by settings")
        return True
    if manifest.api_version > EXTENSION_API_VERSION:
        host.notices.append(
            f"extensions · {manifest.name} ({_display(directory)}) built for "
            f"extension API v{manifest.api_version}; this zeta supports "
            f"v{EXTENSION_API_VERSION} — upgrade zeta"
        )
        return True
    return False


def _load_one(
    host: ExtensionHost,
    directory: Path,
    manifest: ExtensionManifest,
    scope: str,
) -> None:
    display = _display(directory)
    try:
        module = _import_entry(directory, manifest, host=host)
    except Exception as exc:  # noqa: BLE001 - loading must fail open
        error = f"import failed: {_short(exc)}"
        host.record_error(manifest.name, manifest.version, scope, error)
        host.notices.append(
            f"extensions · ignored {scope} extension {manifest.name} ({display}): {error}"
        )
        return
    entry = getattr(module, manifest.entry_function, None)
    if not callable(entry):
        host.remove_package_for_directory(directory)
        error = f"no callable {manifest.entry_function}(zx)"
        host.record_error(manifest.name, manifest.version, scope, error)
        host.notices.append(
            f"extensions · ignored {scope} extension {manifest.name} ({display}): {error}"
        )
        return
    loading = LoadingExtension(manifest, scope, directory)
    api = ExtensionAPI(host, loading)
    try:
        result = entry(api)
    except Exception as exc:  # noqa: BLE001 - a broken activate must not break the session
        host.remove_package_for_directory(directory)
        error = f"activate raised: {_short(exc)}"
        host.record_error(manifest.name, manifest.version, scope, error)
        host.notices.append(
            f"extensions · ignored {scope} extension {manifest.name} ({display}): {error}"
        )
        return
    if inspect.isawaitable(result):
        if inspect.iscoroutine(result):
            result.close()
        host.remove_package_for_directory(directory)
        error = "activate must be synchronous"
        host.record_error(manifest.name, manifest.version, scope, error)
        host.notices.append(
            f"extensions · ignored {scope} extension {manifest.name} ({display}): {error}"
        )
        return
    host.commit(loading)


def _import_entry(
    directory: Path, manifest: ExtensionManifest, *, host: ExtensionHost
) -> object:
    """Import the entry module as a package rooted at the extension dir.

    The extension dir is made importable only as this synthetic package's
    search location (``__path__``), never added to the global ``sys.path``, so
    two extensions with a same-named helper module do not collide.
    """

    pkg_name, _pkg = host.new_package(directory, manifest.name)
    entry_full = f"{pkg_name}.{manifest.entry_module}"
    single = directory / f"{manifest.entry_module}.py"
    package_init = directory / manifest.entry_module / "__init__.py"
    if single.is_file():
        entry_path = single
        search: list[str] | None = None
        entry_package = pkg_name
    elif package_init.is_file():
        entry_path = package_init
        search = [str(directory / manifest.entry_module)]
        entry_package = entry_full
    else:
        host.remove_package(pkg_name)
        raise ImportError(
            f"entry module '{manifest.entry_module}' not found in {directory}"
        )
    spec = importlib.util.spec_from_file_location(
        entry_full, entry_path, submodule_search_locations=search
    )
    if spec is None or spec.loader is None:
        host.remove_package(pkg_name)
        raise ImportError(f"could not build import spec for {entry_path}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = entry_package
    host.register_module(pkg_name, manifest.entry_module, module)
    try:
        spec.loader.exec_module(module)
    except BaseException:
        host.remove_package(pkg_name)
        raise
    return module


def _as_path(base: str | Path | None) -> Path | None:
    if base is None:
        return None
    try:
        return Path(base).expanduser().resolve(strict=False)
    except OSError:
        return None


def _display(directory: Path) -> str:
    try:
        home = Path.home()
    except (RuntimeError, OSError):
        return str(directory)
    try:
        return f"~/{directory.relative_to(home)}"
    except ValueError:
        return str(directory)


def _short(exc: BaseException) -> str:
    text = str(exc) or exc.__class__.__name__
    text = text.replace("\n", " ")
    if len(text) > 200:
        text = text[:200] + "…"
    return text


__all__ = [
    "EXTENSIONS_DIRNAME",
    "apply_extensions",
]
