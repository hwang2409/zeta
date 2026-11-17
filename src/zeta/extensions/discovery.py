"""Discover, import, and activate canonical installed extensions."""

from __future__ import annotations

import importlib.util
import inspect
import sys
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
    _SourceExtensionLoader,
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
    resolved_home = _as_path(home)
    resolved_project = _as_path(project_dir)
    host = ExtensionHost(
        registry,  # type: ignore[arg-type]
        home=resolved_home,
        project_dir=resolved_project,
        builtin_names=builtin_names,
        clock=clock,
    )
    if resolved_home is None:
        return host
    store = ExtensionStore(home=resolved_home, project_dir=resolved_project)
    disabled_set = frozenset(disabled)
    for scope in (USER_SCOPE, PROJECT_SCOPE):
        if scope == PROJECT_SCOPE and resolved_project is None:
            continue
        result = store.discover(scope)
        for notice in result.notices:
            host.notices.append(f"extensions · {notice}")
            host.notify("extensions", notice, "warning")
        pending: list[str] = []
        for item in result.extensions:
            _apply_item(host, item, disabled_set, pending)
        if pending:
            host.warnings.append(
                "extensions · untrusted project extensions present: "
                f"{', '.join(sorted(pending))}; trust them to enable them for this session"
            )
    return host


def _apply_item(
    host: ExtensionHost,
    item: InstalledExtension,
    disabled: frozenset[str],
    pending: list[str],
) -> None:
    manifest = item.manifest
    enabled = not _skip(host, manifest, item.directory, disabled)
    host.add_status(manifest, item.scope, enabled=enabled)
    if not enabled:
        return
    if item.scope == PROJECT_SCOPE and not item.trusted:
        host.pending_project.append(PendingExtension(item))
        pending.append(item.name)
        return
    if item.trusted:
        host.mark_trusted(item.name, item.scope)
    _load_one(host, item.directory, manifest, item.scope)


def _skip(
    host: ExtensionHost,
    manifest: ExtensionManifest,
    directory: Path,
    disabled: frozenset[str],
) -> bool:
    if manifest.name in disabled:
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


def _load_one(host: ExtensionHost, directory: Path, manifest: ExtensionManifest, scope: str) -> None:
    display = _display(directory)
    try:
        module = _import_entry(directory, manifest, host=host)
    except Exception as exc:  # noqa: BLE001 - loading must fail open
        error = f"import failed: {_short(exc)}"
        host.mark_error(manifest.name, error, scope)
        host.notices.append(f"extensions · ignored {scope} extension {manifest.name} ({display}): {error}")
        return
    entry = getattr(module, manifest.entry_function, None)
    if not callable(entry):
        host.remove_package_for_directory(directory)
        error = f"no callable {manifest.entry_function}(zx)"
        host.mark_error(manifest.name, error, scope)
        host.notices.append(f"extensions · ignored {scope} extension {manifest.name} ({display}): {error}")
        return
    loading = LoadingExtension(manifest, scope, directory)
    api = ExtensionAPI(host, loading)
    try:
        result = entry(api)
    except Exception as exc:  # noqa: BLE001 - a broken activate must not break the session
        host.remove_package_for_directory(directory)
        error = f"activate raised: {_short(exc)}"
        host.mark_error(manifest.name, error, scope)
        host.notices.append(f"extensions · ignored {scope} extension {manifest.name} ({display}): {error}")
        return
    if inspect.isawaitable(result):
        if inspect.iscoroutine(result):
            result.close()
        host.remove_package_for_directory(directory)
        error = "activate must be synchronous"
        host.mark_error(manifest.name, error, scope)
        host.notices.append(f"extensions · ignored {scope} extension {manifest.name} ({display}): {error}")
        return
    host.commit(loading)


def _import_entry(directory: Path, manifest: ExtensionManifest, *, host: ExtensionHost) -> object:
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
        raise ImportError(f"entry module '{manifest.entry_module}' not found in {directory}")
    spec = importlib.util.spec_from_file_location(
        entry_full,
        entry_path,
        loader=_SourceExtensionLoader(entry_full, str(entry_path)),
        submodule_search_locations=search,
    )
    if spec is None or spec.loader is None:
        host.remove_package(pkg_name)
        raise ImportError(f"could not build import spec for {entry_path}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = entry_package
    sys.modules[entry_full] = module
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
        return f"~/{directory.relative_to(Path.home())}"
    except (ValueError, RuntimeError, OSError):
        return str(directory)


def _short(exc: BaseException) -> str:
    text = (str(exc) or exc.__class__.__name__).replace("\n", " ")
    return text[:200] + "…" if len(text) > 200 else text


__all__ = ["EXTENSIONS_DIRNAME", "apply_extensions"]
