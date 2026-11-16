"""Validated paths for extension management."""

from __future__ import annotations

import re
import stat
from pathlib import Path


class ExtensionTrustError(ValueError):
    """An extension could not be installed or trusted."""


_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def _directory(path: Path, *, missing_ok: bool = False) -> Path:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        if missing_ok:
            return path
        raise ExtensionTrustError(f"extension directory is missing: {path}")
    except OSError as exc:
        raise ExtensionTrustError(f"cannot inspect extension directory: {path}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ExtensionTrustError(f"extension path is not a real directory: {path}")
    return path


def extension_path(
    home: str | Path,
    project_dir: str | Path | None,
    scope: str,
    name: str,
    *,
    create: bool = False,
) -> Path:
    """Return a checked management path for one extension."""
    if not _NAME_RE.fullmatch(name):
        raise ExtensionTrustError(f"invalid extension name: {name}")
    if scope == "user":
        home_path = Path(home).expanduser()
        if create and not home_path.exists():
            home_path.mkdir(parents=True, exist_ok=True)
        try:
            base = home_path.resolve(strict=True)
        except OSError as exc:
            raise ExtensionTrustError(f"cannot inspect extension home: {home}") from exc
        _directory(base)
        root = base / "extensions"
        if create:
            root.mkdir(parents=True, exist_ok=True)
        _directory(root)
    elif scope == "project" and project_dir is not None:
        base = _directory(Path(project_dir).expanduser())
        zeta = base / ".zeta"
        root = zeta / "extensions"
        if create:
            root.mkdir(parents=True, exist_ok=True)
        _directory(zeta)
        _directory(root)
    else:
        raise ExtensionTrustError("project scope requires a repository")
    return _directory(root / name, missing_ok=create)
