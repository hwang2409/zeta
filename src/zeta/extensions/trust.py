"""Installation and content trust for native extensions."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from ..core.session_files import atomic_publish_file
from .manifest import MANIFEST_FILENAME, ExtensionManifestError, parse_manifest

TRUST_FILENAME = "extensions-trust.json"
GIT_FILENAME = ".zeta-extension-git.json"
_MAX_FILES = 10_000
_MAX_TOTAL_BYTES = 64 * 1024 * 1024
_MAX_FILE_BYTES = 16 * 1024 * 1024
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class ExtensionTrustError(ValueError):
    """An extension could not be installed or trusted."""


def _regular_files(root: Path):
    pending = [root]
    count = 0
    while pending:
        current = pending.pop()
        try:
            entries = sorted(os.scandir(current), key=lambda entry: entry.name)
        except OSError as exc:
            raise ExtensionTrustError(f"cannot inspect extension tree: {current}") from exc
        for entry in entries:
            try:
                mode = entry.stat(follow_symlinks=False).st_mode
            except OSError as exc:
                raise ExtensionTrustError(f"cannot inspect extension file: {entry.path}") from exc
            if stat.S_ISLNK(mode):
                raise ExtensionTrustError(f"extension tree contains symlink: {entry.path}")
            if stat.S_ISDIR(mode):
                pending.append(Path(entry.path))
                continue
            if not stat.S_ISREG(mode):
                raise ExtensionTrustError(f"extension tree contains non-regular file: {entry.path}")
            count += 1
            if count > _MAX_FILES:
                raise ExtensionTrustError(f"extension tree exceeds {_MAX_FILES} files")
            yield Path(entry.path)


def _hash_file(path: Path, relative: bytes, digest: hashlib._Hash, total: int, max_total_bytes: int) -> int:
    digest.update(len(relative).to_bytes(8, "big"))
    digest.update(relative)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise ExtensionTrustError(f"cannot open extension file: {path}") from exc
    file_total = 0
    try:
        while chunk := os.read(fd, 1024 * 1024):
            file_total += len(chunk)
            total += len(chunk)
            if file_total > _MAX_FILE_BYTES or total > max_total_bytes:
                raise ExtensionTrustError(f"extension tree exceeds {max_total_bytes} byte limit")
            digest.update(chunk)
    finally:
        os.close(fd)
    return total


def fingerprint(directory: str | Path, *, max_total_bytes: int = _MAX_TOTAL_BYTES) -> str:
    root = Path(directory)
    try:
        if stat.S_ISLNK(root.lstat().st_mode):
            raise ExtensionTrustError(f"extension tree is a symlink: {root}")
        root = root.resolve(strict=True)
    except OSError as exc:
        raise ExtensionTrustError(f"cannot inspect extension tree: {root}") from exc
    digest = hashlib.sha256()
    total = 0
    for path in _regular_files(root):
        total = _hash_file(path, path.relative_to(root).as_posix().encode(), digest, total, max_total_bytes)
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
    payload = (json.dumps(values, indent=2, sort_keys=True) + "\n").encode()
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        atomic_publish_file(directory_fd, path.name, payload)
    finally:
        os.close(directory_fd)


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


def _root(home: str | Path, project_dir: str | Path | None, scope: str) -> Path:
    if scope == "project" and project_dir is None:
        raise ExtensionTrustError("project scope requires a repository")
    return (Path(home) / "extensions") if scope == "user" else Path(project_dir) / ".zeta" / "extensions"


def _resolve(root: Path, name: str) -> Path:
    if not _NAME_RE.fullmatch(name) or Path(name).is_absolute() or Path(name).name != name:
        raise ExtensionTrustError(f"invalid extension name: {name}")
    candidate = root / name
    try:
        try:
            if stat.S_ISLNK(candidate.lstat().st_mode):
                raise ExtensionTrustError(f"extension path is a symlink: {name}")
        except FileNotFoundError:
            pass
        root_real = root.resolve(strict=False)
        target = candidate.resolve(strict=False)
        target.relative_to(root_real)
    except ExtensionTrustError:
        raise
    except (OSError, ValueError) as exc:
        raise ExtensionTrustError(f"extension path escapes root: {name}") from exc
    return candidate


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


def _stage(source: str) -> tuple[Path, str, tempfile.TemporaryDirectory[str]]:
    temporary = tempfile.TemporaryDirectory(prefix="zeta-extension-stage-")
    staged = Path(temporary.name) / "extension"
    git = _source_git(source)
    commit = ""
    if git:
        subprocess.run(["git", "clone", "--quiet", git[0], str(staged)], check=True, capture_output=True, text=True)
        subprocess.run(["git", "-C", str(staged), "checkout", "--quiet", git[1]], check=True, capture_output=True, text=True)
        commit = subprocess.run(["git", "-C", str(staged), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    else:
        shutil.copytree(Path(source).expanduser().resolve(strict=True), staged, symlinks=True)
    return staged, commit, temporary


def _review(manifest, staged: Path, *, current: Path | None, action: str, accept: bool, confirm: Callable[[str], str]) -> None:
    print(f"name: {manifest.name}\nversion: {manifest.version}\napi_version: {manifest.api_version}\nentry: {manifest.entry_module}:{manifest.entry_function}\ndescription: {manifest.description}")
    print("warning: this runs arbitrary Python as you, with access to your files, credentials, and session history")
    files = _files(staged)
    if current is not None:
        old = set(_files(current))
        new = set(files)
        print("changed files:\n" + "\n".join(sorted(old ^ new | {p for p in old & new if (current / p).read_bytes() != (staged / p).read_bytes()})))
        print(f"diff summary: {len(new - old)} added, {len(old - new)} removed, {len(old & new) - sum((current / p).read_bytes() == (staged / p).read_bytes() for p in old & new)} changed")
        changelog = staged / "CHANGELOG.md"
        if changelog.is_file():
            print("changelog:\n" + changelog.read_text(encoding="utf-8"))
    else:
        print("files:\n" + "\n".join(files))
    if not accept and confirm(f"{action} this extension? [y/N] ").strip().lower() != "y":
        raise ExtensionTrustError(f"{action} cancelled")


def install(source: str, *, home: str | Path, project_dir: str | Path | None, scope: str, accept: bool = False, confirm: Callable[[str], str] = input) -> tuple[str, str]:
    root = _root(home, project_dir, scope)
    root.mkdir(parents=True, exist_ok=True)
    staged, commit, temporary = _stage(source)
    with temporary:
        fingerprint(staged)
        manifest = _manifest(staged)
        _review(manifest, staged, current=None, action="install", accept=accept, confirm=confirm)
        target = _resolve(root, manifest.name)
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(staged), target)
        if commit:
            (target / GIT_FILENAME).write_text(json.dumps({"source": source, "commit": commit}, sort_keys=True) + "\n", encoding="utf-8")
        set_trusted(home, scope, manifest.name, target, True)
        return manifest.name, commit


def remove(*, home: str | Path, project_dir: str | Path | None, scope: str, name: str) -> None:
    root = _root(home, project_dir, scope)
    target = _resolve(root, name)
    if not target.is_dir():
        raise ExtensionTrustError(f"extension not installed: {name}")
    shutil.rmtree(target)
    set_trusted(home, scope, name, target, False)


def untrust(*, home: str | Path, project_dir: str | Path | None, scope: str, name: str) -> None:
    target = _resolve(_root(home, project_dir, scope), name)
    if not target.is_dir():
        raise ExtensionTrustError(f"extension not installed: {name}")
    set_trusted(home, scope, name, target, False)


def trust(*, home: str | Path, project_dir: str | Path | None, scope: str, name: str, accept: bool = False, confirm: Callable[[str], str] = input) -> None:
    root = _root(home, project_dir, scope)
    target = _resolve(root, name)
    if not target.is_dir():
        raise ExtensionTrustError(f"extension not installed: {name}")
    fingerprint(target)
    manifest = _manifest(target)
    _review(manifest, target, current=None, action="trust", accept=accept, confirm=confirm)
    set_trusted(home, scope, name, target, True)


def update(name: str, *, home: str | Path, project_dir: str | Path | None, scope: str, accept: bool = False, confirm: Callable[[str], str] = input) -> tuple[str, str]:
    root = _root(home, project_dir, scope)
    target = _resolve(root, name)
    try:
        source = json.loads((target / GIT_FILENAME).read_text(encoding="utf-8"))["source"]
    except (OSError, ValueError, KeyError) as exc:
        raise ExtensionTrustError(f"extension {name} is not git-installed") from exc
    staged, commit, temporary = _stage(source)
    with temporary:
        fingerprint(staged)
        fingerprint(target)
        manifest = _manifest(staged)
        if manifest.name != name:
            raise ExtensionTrustError(
                f"update manifest name {manifest.name!r} does not match installed name {name!r}"
            )
        _review(manifest, staged, current=target, action="update", accept=accept, confirm=confirm)
        shutil.rmtree(target)
        shutil.move(str(staged), target)
        (target / GIT_FILENAME).write_text(json.dumps({"source": source, "commit": commit}, sort_keys=True) + "\n", encoding="utf-8")
        set_trusted(home, scope, manifest.name, target, True)
        return manifest.name, commit


__all__ = ["ExtensionTrustError", "fingerprint", "install", "is_trusted", "remove", "set_trusted", "trust", "trust_path", "untrust", "update"]
