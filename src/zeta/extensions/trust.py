"""Installation and content trust for native extensions."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from .manifest import MANIFEST_FILENAME, ExtensionManifestError, parse_manifest

TRUST_FILENAME = "extensions-trust.json"
GIT_FILENAME = ".zeta-extension-git.json"


class ExtensionTrustError(ValueError):
    """An extension could not be installed or trusted."""


def fingerprint(directory: str | Path) -> str:
    root = Path(directory).resolve()
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _read(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ExtensionTrustError(f"invalid extension trust file: {path}") from exc
    if type(value) is not dict or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        raise ExtensionTrustError(f"invalid extension trust file: {path}")
    return value


def _write(path: Path, values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def trust_path(home: str | Path) -> Path:
    return Path(home).expanduser() / TRUST_FILENAME


def trust_key(scope: str, name: str) -> str:
    return f"{scope}:{name}"


def is_trusted(home: str | Path, scope: str, name: str, directory: str | Path) -> bool:
    values = _read(trust_path(home))
    return values.get(trust_key(scope, name)) == fingerprint(directory)


def set_trusted(home: str | Path, scope: str, name: str, directory: str | Path, trusted: bool) -> None:
    path = trust_path(home)
    values = _read(path)
    key = trust_key(scope, name)
    if trusted:
        values[key] = fingerprint(directory)
    else:
        values.pop(key, None)
    _write(path, values)


def _source_git(source: str) -> tuple[str, str] | None:
    if not source.startswith("git+"):
        return None
    value = source[4:]
    if "@" not in value:
        raise ExtensionTrustError("git source must include @ref")
    url, ref = value.rsplit("@", 1)
    if not url or not ref:
        raise ExtensionTrustError("git source must include a repository and ref")
    return url, ref


def _manifest(directory: Path):
    try:
        return parse_manifest((directory / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ExtensionManifestError) as exc:
        raise ExtensionTrustError(f"invalid extension manifest in {directory}: {exc}") from exc


def _files(directory: Path) -> list[str]:
    return [p.relative_to(directory).as_posix() for p in sorted(directory.rglob("*")) if p.is_file()]


def install(
    source: str,
    *,
    home: str | Path,
    project_dir: str | Path | None,
    scope: str,
    accept: bool = False,
    confirm: Callable[[str], str] = input,
) -> tuple[str, str]:
    """Review and install a path or git source, returning name and commit pin."""
    destination_root = Path(home) / "extensions" if scope == "user" else Path(project_dir or "") / ".zeta" / "extensions"
    if scope == "project" and project_dir is None:
        raise ExtensionTrustError("project scope requires a repository")
    destination_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="zeta-extension-stage-") as temporary:
        staged = Path(temporary) / "extension"
        git = _source_git(source)
        commit = ""
        if git:
            subprocess.run(["git", "clone", "--quiet", git[0], str(staged)], check=True, capture_output=True, text=True)
            subprocess.run(["git", "-C", str(staged), "checkout", "--quiet", git[1]], check=True, capture_output=True, text=True)
            commit = subprocess.run(["git", "-C", str(staged), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        else:
            source_path = Path(source).expanduser().resolve(strict=True)
            shutil.copytree(source_path, staged)
        manifest = _manifest(staged)
        print(f"name: {manifest.name}\nversion: {manifest.version}\napi_version: {manifest.api_version}\nentry: {manifest.entry_module}:{manifest.entry_function}\ndescription: {manifest.description}")
        print("warning: this runs arbitrary Python as you, with access to your files, credentials, and session history")
        print("files:\n" + "\n".join(_files(staged)))
        if not accept and confirm("install this extension? [y/N] ").strip().lower() != "y":
            raise ExtensionTrustError("installation cancelled")
        target = destination_root / manifest.name
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(staged), target)
        if commit:
            (target / GIT_FILENAME).write_text(json.dumps({"source": source, "commit": commit}, sort_keys=True) + "\n", encoding="utf-8")
        set_trusted(home, scope, manifest.name, target, True)
        return manifest.name, commit


def remove(*, home: str | Path, project_dir: str | Path | None, scope: str, name: str) -> None:
    root = Path(home) / "extensions" if scope == "user" else Path(project_dir or "") / ".zeta" / "extensions"
    target = root / name
    if not target.is_dir():
        raise ExtensionTrustError(f"extension not installed: {name}")
    shutil.rmtree(target)
    set_trusted(home, scope, name, target, False)


def update(name: str, *, home: str | Path, project_dir: str | Path | None, scope: str, accept: bool = False, confirm: Callable[[str], str] = input) -> tuple[str, str]:
    root = Path(home) / "extensions" if scope == "user" else Path(project_dir or "") / ".zeta" / "extensions"
    target = root / name
    try:
        source = json.loads((target / GIT_FILENAME).read_text(encoding="utf-8"))["source"]
    except (OSError, ValueError, KeyError) as exc:
        raise ExtensionTrustError(f"extension {name} is not git-installed") from exc
    return install(source, home=home, project_dir=project_dir, scope=scope, accept=accept, confirm=confirm)


__all__ = ["ExtensionTrustError", "fingerprint", "install", "is_trusted", "remove", "set_trusted", "trust_path", "update"]
