"""Discover, import, and activate zeta extensions for one session.

An extension is a directory that contains a ``zeta-extension.toml`` manifest
and an entry module. Two scopes are scanned, mirroring ``~/.zeta/tools`` and
``<project>/.zeta/tools``:

* user -- ``<home>/extensions/*/``. Imported and activated directly in a
  session (full power, the user's own machine).
* project -- ``<project>/.zeta/extensions/*/``. TRUST BOUNDARY: code from a
  cloned repository. Discovered and listed with a loud warning, but NOT
  imported or activated until the user trusts them with
  :func:`trust_project_extensions`. (A durable trust store and the ``/ext``
  trust command are later slices; this slice stages them in memory.)

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
import sys
from pathlib import Path

from .api import EXTENSION_API_VERSION, ExtensionAPI
from .host import (
    PROJECT_SCOPE,
    USER_SCOPE,
    ExtensionHost,
    LoadingExtension,
    PendingExtension,
)
from .manifest import (
    MANIFEST_FILENAME,
    ExtensionManifest,
    ExtensionManifestError,
    parse_manifest,
)

EXTENSIONS_DIRNAME = "extensions"


def apply_extensions(
    registry: object,
    *,
    home: str | Path | None,
    project_dir: str | Path | None,
    builtin_names: frozenset[str],
    disabled: tuple[str, ...] = (),
) -> ExtensionHost:
    """Load user extensions into ``registry`` and stage project extensions.

    ``home`` is the zeta home (``~/.zeta``); ``project_dir`` is the project
    root (``None`` disables the project scope). ``builtin_names`` is the set of
    tool names present before any external tool or extension loaded; a project
    extension may never register one of them.
    """

    host = ExtensionHost(
        registry,  # type: ignore[arg-type]
        home=_as_path(home),
        project_dir=_as_path(project_dir),
        builtin_names=builtin_names,
    )
    disabled_set = frozenset(disabled)
    user_dir = _scope_dir(home, "")
    project_scope_dir = _scope_dir(project_dir, ".zeta")
    for directory in _extension_dirs(user_dir):
        manifest = _manifest_or_notice(host, directory, USER_SCOPE)
        if manifest is None:
            continue
        enabled = not _skip(host, manifest, directory, disabled_set)
        host.add_status(manifest, USER_SCOPE, enabled=enabled)
        if not enabled:
            continue
        _load_one(host, directory, manifest, USER_SCOPE)
    if project_scope_dir is not None and project_scope_dir != user_dir:
        pending_names: list[str] = []
        for directory in _extension_dirs(project_scope_dir):
            manifest = _manifest_or_notice(host, directory, PROJECT_SCOPE)
            if manifest is None:
                continue
            enabled = not _skip(host, manifest, directory, disabled_set)
            host.add_status(manifest, PROJECT_SCOPE, enabled=enabled)
            if not enabled:
                continue
            host.pending_project.append(PendingExtension(manifest, directory))
            pending_names.append(manifest.name)
        if pending_names:
            host.warnings.append(
                "extensions · untrusted project extensions present: "
                f"{', '.join(sorted(pending_names))}; trust them to enable them "
                "for this session"
            )
    return host


def trust_project_extensions(host: ExtensionHost) -> None:
    """Import and activate the staged project extensions (the trust action).

    The pending list is cleared regardless of per-extension success, so a
    malformed one is reported once and stays dropped.
    """

    pending = list(host.pending_project)
    host.pending_project.clear()
    for item in pending:
        _load_one(host, item.directory, item.manifest, PROJECT_SCOPE)


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
        host.mark_error(manifest.name, error)
        host.notices.append(
            f"extensions · ignored {scope} extension {manifest.name} ({display}): {error}"
        )
        return
    entry = getattr(module, manifest.entry_function, None)
    if not callable(entry):
        host.remove_package_for_directory(directory)
        error = f"no callable {manifest.entry_function}(zx)"
        host.mark_error(manifest.name, error)
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
        host.mark_error(manifest.name, error)
        host.notices.append(
            f"extensions · ignored {scope} extension {manifest.name} ({display}): {error}"
        )
        return
    if inspect.isawaitable(result):
        if inspect.iscoroutine(result):
            result.close()
        host.remove_package_for_directory(directory)
        error = "activate must be synchronous"
        host.mark_error(manifest.name, error)
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
    sys.modules[entry_full] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        host.remove_package(pkg_name)
        raise
    return module


def _manifest_or_notice(
    host: ExtensionHost, directory: Path, scope: str
) -> ExtensionManifest | None:
    path = directory / MANIFEST_FILENAME
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        host.notices.append(
            f"extensions · ignored {scope} extension {_display(directory)}: "
            f"cannot read manifest: {_short(exc)}"
        )
        return None
    try:
        return parse_manifest(text)
    except ExtensionManifestError as exc:
        host.notices.append(
            f"extensions · ignored {scope} extension {_display(directory)}: "
            f"invalid manifest: {_short(exc)}"
        )
        return None


def _extension_dirs(base: Path | None) -> list[Path]:
    if base is None:
        return []
    found: list[Path] = []
    try:
        for path in sorted(base.iterdir()):
            if path.is_dir() and (path / MANIFEST_FILENAME).is_file():
                found.append(path)
    except OSError:
        return []
    return found


def _scope_dir(base: str | Path | None, nested: str) -> Path | None:
    resolved = _as_path(base)
    if resolved is None:
        return None
    candidate = resolved / nested / EXTENSIONS_DIRNAME if nested else resolved / EXTENSIONS_DIRNAME
    return candidate if candidate.is_dir() else None


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
    "trust_project_extensions",
]
