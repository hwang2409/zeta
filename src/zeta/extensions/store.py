"""Private snapshot storage, trust, and recovery for native extensions."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from ..core.session_files import atomic_publish_file, require_off_loop_lock
from .manifest import (
    MANIFEST_FILENAME,
    ExtensionManifest,
    ExtensionManifestError,
    parse_manifest,
)

TRUST_FILENAME = "extensions-trust.json"
GIT_FILENAME = ".zeta-extension-git.json"
_STORE_NAME = ".zeta-extension-store"
_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_MAX_FILES = 10_000
_MAX_TOTAL_BYTES = 64 * 1024 * 1024
_MAX_FILE_BYTES = 16 * 1024 * 1024


class ExtensionTrustError(ValueError):
    """The extension tree, trust state, or update state is unsafe."""


@dataclass(frozen=True, slots=True)
class InstallResult:
    name: str
    commit: str
    digest: str


@dataclass(frozen=True, slots=True)
class InstalledExtension:
    scope: str
    name: str
    directory: Path
    manifest: ExtensionManifest
    digest: str
    trusted: bool


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    extensions: tuple[InstalledExtension, ...]
    notices: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Snapshot:
    operation_dir: Path
    tree: Path
    manifest: ExtensionManifest
    files: tuple[str, ...]
    digest: str
    display_source: str
    commit: str = ""


@dataclass(frozen=True, slots=True)
class _UpdateOwner:
    name: str
    new_digest: str
    old_digest: str
    old_trust: str | None
    version: int = 1


@dataclass(frozen=True, slots=True)
class _Recovery:
    directory: Path
    owner: _UpdateOwner | None
    decision: str


def _json_object(text: str, error: str) -> dict[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in items:
            if key in value:
                raise ExtensionTrustError(error)
            value[key] = item
        return value

    try:
        value = json.loads(text, object_pairs_hook=pairs)
    except (ValueError, TypeError) as exc:
        raise ExtensionTrustError(error) from exc
    if type(value) is not dict:
        raise ExtensionTrustError(error)
    return value


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError as exc:
        raise ExtensionTrustError(f"cannot sync extension directory: {path}") from exc


def _publish(path: Path, payload: bytes) -> None:
    try:
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            atomic_publish_file(fd, path.name, payload)
        finally:
            os.close(fd)
    except OSError as exc:
        raise ExtensionTrustError(f"cannot publish extension file: {path}") from exc


def _remove_tree_entries(path: Path, progress: Callable[[], None] | None) -> None:
    called = False
    for entry in sorted(os.scandir(path), key=lambda item: item.name):
        item = Path(entry.path)
        if stat.S_ISDIR(entry.stat(follow_symlinks=False).st_mode):
            _remove_tree(item, progress if not called else None)
        else:
            os.unlink(item)
            if progress is not None and not called:
                called = True
                progress()
        _fsync_directory(path)


def _remove_tree(path: Path, progress: Callable[[], None] | None = None) -> None:
    if not path.exists():
        return
    try:
        mode = path.lstat().st_mode
        if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
            raise ExtensionTrustError(f"unsafe extension cleanup path: {path}")
        _remove_tree_entries(path, progress)
        os.rmdir(path)
        _fsync_directory(path.parent)
    except ExtensionTrustError:
        raise
    except OSError as exc:
        raise ExtensionTrustError(f"cannot remove extension tree: {path}") from exc


class ExtensionStore:
    """Own extension acquisition, review, publication, trust, and discovery."""

    def __init__(self, *, home: Path, project_dir: Path | None) -> None:
        try:
            self.home = home.expanduser().resolve(strict=False)
            self.home.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(self.home, 0o700)
        except OSError as exc:
            raise ExtensionTrustError(f"invalid extension home: {home}") from exc
        self.project_dir = project_dir.expanduser().resolve(strict=False) if project_dir else None
        self._trust_path = self.home / TRUST_FILENAME
        self._lock_path = self.home / ".extensions.lock"

    def _checkpoint(self, name: str) -> None:
        return None

    def install(
        self,
        source: str,
        *,
        scope: str,
        accept: bool = False,
        confirm: Callable[[str], str] = input,
    ) -> InstallResult:
        root = self._root(scope, create=True)
        snapshot = self._acquire(source, root)
        try:
            self._review(snapshot, current=None, action="install", accept=accept, confirm=confirm)
            with self._locked():
                self._recover_or_raise(root)
                trust = self._read_trust()
                target = root / snapshot.manifest.name
                if target.exists():
                    current = self._snapshot_existing(target, root)
                    if current.digest != snapshot.digest:
                        raise ExtensionTrustError(
                            f"extension already installed with different content: {snapshot.manifest.name}"
                        )
                    self._change_trust(trust, scope, snapshot.manifest.name, snapshot.digest)
                    return InstallResult(snapshot.manifest.name, snapshot.commit, snapshot.digest)
                os.rename(snapshot.tree, target)
                _fsync_directory(root)
                self._checkpoint("install_target_published")
                try:
                    self._change_trust(trust, scope, snapshot.manifest.name, snapshot.digest)
                except ExtensionTrustError:
                    if self._read_trust().get(self._key(scope, snapshot.manifest.name)) == snapshot.digest:
                        return InstallResult(snapshot.manifest.name, snapshot.commit, snapshot.digest)
                    raise
                self._checkpoint("install_trust_published")
                return InstallResult(snapshot.manifest.name, snapshot.commit, snapshot.digest)
        finally:
            self._discard_operation(snapshot.operation_dir)

    def update(
        self,
        name: str,
        *,
        scope: str,
        accept: bool = False,
        confirm: Callable[[str], str] = input,
    ) -> InstallResult:
        self._validate_name(name)
        root = self._root(scope, create=True)
        with self._locked():
            self._recover_or_raise(root)
            target = self._target(root, name)
            metadata = self._git_metadata(target, name)
        snapshot = self._acquire(str(metadata["source"]), root)
        current: _Snapshot | None = None
        try:
            current = self._copy_installed_for_review(target, root)
            if snapshot.manifest.name != name:
                raise ExtensionTrustError(
                    f"manifest name {snapshot.manifest.name!r} does not match folder {name!r}"
                )
            with self._locked():
                self._recover_or_raise(root)
                trust = self._read_trust()
                current_locked = self._snapshot_existing(target, root)
                if current_locked.digest == snapshot.digest and trust.get(self._key(scope, name)) == snapshot.digest:
                    return InstallResult(name, snapshot.commit, snapshot.digest)
                if current_locked.digest != current.digest:
                    raise ExtensionTrustError(f"extension changed since review: {name}")
            self._review(snapshot, current=current, action="update", accept=accept, confirm=confirm)
            with self._locked():
                self._recover_or_raise(root)
                trust = self._read_trust()
                current_locked = self._snapshot_existing(target, root)
                if current_locked.digest == snapshot.digest and trust.get(self._key(scope, name)) == snapshot.digest:
                    return InstallResult(name, snapshot.commit, snapshot.digest)
                if current_locked.digest != current.digest:
                    raise ExtensionTrustError(f"extension changed since review: {name}")
                owner = _UpdateOwner(name, snapshot.digest, current.digest, trust.get(self._key(scope, name)))
                update_dir = self._publish_owner(snapshot, owner, root)
                backup = update_dir / "backup"
                os.rename(target, backup)
                _fsync_directory(root)
                _fsync_directory(update_dir)
                self._checkpoint("update_backup_published")
                os.rename(update_dir / "tree", target)
                _fsync_directory(update_dir)
                _fsync_directory(root)
                self._checkpoint("update_target_published")
                try:
                    self._change_trust(trust, scope, name, snapshot.digest)
                except ExtensionTrustError:
                    persisted = self._read_trust()
                    self._recover(root, persisted)
                    if persisted.get(self._key(scope, name)) == snapshot.digest:
                        return InstallResult(name, snapshot.commit, snapshot.digest)
                    raise
                self._checkpoint("update_trust_published")
                self._finish_committed(update_dir)
                return InstallResult(name, snapshot.commit, snapshot.digest)
        finally:
            if current is not None:
                self._discard_operation(current.operation_dir)
            self._discard_operation(snapshot.operation_dir)

    def trust(
        self,
        name: str,
        *,
        scope: str,
        accept: bool = False,
        confirm: Callable[[str], str] = input,
    ) -> None:
        self._validate_name(name)
        root = self._root(scope, create=False)
        target = self._target(root, name)
        snapshot = self._copy_installed_for_review(target, root)
        try:
            self._review(snapshot, current=None, action="trust", accept=accept, confirm=confirm)
            with self._locked():
                self._recover_or_raise(root)
                current = self._snapshot_existing(target, root)
                if current.digest != snapshot.digest:
                    raise ExtensionTrustError(f"extension changed since review: {name}")
                trust = self._read_trust()
                self._change_trust(trust, scope, name, snapshot.digest)
                self._checkpoint("trust_published")
        finally:
            self._discard_operation(snapshot.operation_dir)

    def untrust(self, name: str, *, scope: str) -> None:
        self._validate_name(name)
        root = self._root(scope, create=True)
        with self._locked():
            self._recover_or_raise(root)
            trust = self._read_trust()
            self._change_trust(trust, scope, name, None)
            self._checkpoint("untrust_published")

    def remove(self, name: str, *, scope: str) -> None:
        self._validate_name(name)
        root = self._root(scope, create=True)
        with self._locked():
            self._recover_or_raise(root)
            trust = self._read_trust()
            target = root / name
            trash: Path | None = None
            if target.exists():
                self._snapshot_existing(target, root)
                store = self._private_root(root)
                trash = store / f"remove-{uuid.uuid4().hex}"
                os.rename(target, trash)
                _fsync_directory(root)
                _fsync_directory(store)
                self._checkpoint("remove_hidden")
            else:
                self._remove_residue(root, name)
            self._change_trust(trust, scope, name, None)
            self._checkpoint("remove_trust_published")
            if trash is not None:
                _remove_tree(trash)
                self._checkpoint("remove_deleted")

    def discover(self, scope: str) -> DiscoveryResult:
        root = self._root(scope, create=True)
        with self._locked():
            trust, notices, blocked = self._discovery_trust(root)
            if blocked:
                return DiscoveryResult((), tuple(notices))
            try:
                self._recover(root, trust)
            except ExtensionTrustError as exc:
                if "extension update transaction" not in str(exc):
                    return DiscoveryResult((), (str(exc),))
                return DiscoveryResult(
                    (),
                    (f"extension loading disabled for {scope} scope: update recovery requires attention",),
                )
            return self._discover_extensions(root, scope, trust, notices)

    def _discovery_trust(self, root: Path) -> tuple[dict[str, str], list[str], bool]:
        try:
            return self._read_trust(), [], False
        except ExtensionTrustError as exc:
            notices = [str(exc)]
            return {}, notices, any(self._private_root(root).glob("update-*"))

    def _discover_extensions(
        self,
        root: Path,
        scope: str,
        trust: dict[str, str],
        notices: list[str],
    ) -> DiscoveryResult:
        extensions: list[InstalledExtension] = []
        try:
            entries = sorted(os.scandir(root), key=lambda item: item.name)
        except OSError:
            return DiscoveryResult((), (f"cannot inspect extension root: {root}",))
        for entry in entries:
            if entry.name == _STORE_NAME or not _NAME.fullmatch(entry.name):
                continue
            try:
                if not stat.S_ISDIR(entry.stat(follow_symlinks=False).st_mode):
                    raise ExtensionTrustError(f"extension tree is not a real directory: {entry.path}")
                snapshot = self._snapshot_existing(Path(entry.path), root)
                if snapshot.manifest.name != entry.name:
                    raise ExtensionTrustError(
                        f"manifest name {snapshot.manifest.name!r} does not match folder {entry.name!r}"
                    )
                extensions.append(
                    InstalledExtension(
                        scope,
                        entry.name,
                        Path(entry.path),
                        snapshot.manifest,
                        snapshot.digest,
                        trust.get(self._key(scope, entry.name)) == snapshot.digest,
                    )
                )
            except ExtensionTrustError as exc:
                notices.append(f"ignored {scope} extension {entry.name}: {exc}")
        return DiscoveryResult(tuple(extensions), tuple(notices))

    @contextmanager
    def _locked(self) -> Iterator[None]:
        require_off_loop_lock("extension store", ExtensionTrustError)
        flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK
        try:
            fd = os.open(self._lock_path, flags, 0o600)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ExtensionTrustError(f"invalid extension lock: {self._lock_path}")
            os.chmod(self._lock_path, 0o600)
            os.fsync(fd)
            _fsync_directory(self.home)
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError as exc:
            raise ExtensionTrustError(f"cannot lock extension store: {self._lock_path}") from exc
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _scope_root(self, scope: str) -> tuple[Path, Path]:
        if scope == "user":
            return self.home / "extensions", self.home
        if scope == "project":
            if self.project_dir is None:
                raise ExtensionTrustError("project scope requires a project directory")
            return self.project_dir / ".zeta" / "extensions", self.project_dir
        raise ExtensionTrustError(f"invalid extension scope: {scope}")

    def _validate_root_components(self, root: Path, boundary: Path) -> None:
        current = root
        while current != boundary:
            mode = current.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise ExtensionTrustError(f"extension path is not a real directory: {current}")
            current = current.parent

    def _root(self, scope: str, *, create: bool) -> Path:
        root, boundary = self._scope_root(scope)
        try:
            if create:
                root.mkdir(parents=True, exist_ok=True)
            if root.exists():
                self._validate_root_components(root, boundary)
            return root
        except OSError as exc:
            raise ExtensionTrustError(f"invalid extension root: {root}") from exc

    def _private_root(self, root: Path) -> Path:
        store = root / _STORE_NAME
        try:
            store.mkdir(mode=0o700, exist_ok=True)
            if stat.S_ISLNK(store.lstat().st_mode) or not store.is_dir():
                raise ExtensionTrustError(f"invalid extension private directory: {store}")
            os.chmod(store, 0o700)
            return store
        except OSError as exc:
            raise ExtensionTrustError(f"invalid extension private directory: {store}") from exc

    def _stage(self, root: Path) -> Path:
        store = self._private_root(root)
        operation = store / f"stage-{uuid.uuid4().hex}"
        try:
            operation.mkdir(mode=0o700)
            os.chmod(operation, 0o700)
            _fsync_directory(store)
            self._checkpoint("stage_created")
            return operation
        except OSError as exc:
            raise ExtensionTrustError(f"cannot create extension stage: {operation}") from exc
        except Exception:
            self._discard_operation(operation)
            raise

    def _acquire(self, source: str, root: Path) -> _Snapshot:
        operation = self._stage(root)
        tree = operation / "tree"
        commit = ""
        try:
            git = self._parse_git_source(source)
            if git is None:
                shutil.copytree(Path(source).expanduser(), tree, symlinks=True)
            else:
                subprocess.run(["git", "clone", "--quiet", git[0], str(tree)], check=True, capture_output=True, text=True)
                subprocess.run(["git", "-C", str(tree), "checkout", "--quiet", git[1]], check=True, capture_output=True, text=True)
                commit = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
                shutil.rmtree(tree / ".git")
            os.chmod(tree, 0o700)
            self._checkpoint("source_acquired")
            if git is not None:
                payload = (json.dumps({"commit": commit, "source": source}, sort_keys=True) + "\n").encode()
                _publish(tree / GIT_FILENAME, payload)
                self._checkpoint("git_metadata_published")
            snapshot = self._verify(operation, tree, source, commit)
            self._checkpoint("snapshot_verified")
            return snapshot
        except (ExtensionTrustError, subprocess.CalledProcessError):
            self._discard_operation(operation)
            raise
        except OSError as exc:
            self._discard_operation(operation)
            raise ExtensionTrustError(f"cannot acquire extension source: {source}") from exc
        except Exception:
            self._discard_operation(operation)
            raise
        except BaseException:
            raise

    def _copy_installed_for_review(self, target: Path, root: Path) -> _Snapshot:
        operation = self._stage(root)
        tree = operation / "tree"
        try:
            shutil.copytree(target, tree, symlinks=True)
            os.chmod(tree, 0o700)
            self._checkpoint("source_acquired")
            snapshot = self._verify(operation, tree, str(target), "")
            self._checkpoint("snapshot_verified")
            return snapshot
        except ExtensionTrustError:
            self._discard_operation(operation)
            raise
        except OSError as exc:
            self._discard_operation(operation)
            raise ExtensionTrustError(f"extension not installed: {target.name}") from exc
        except Exception:
            self._discard_operation(operation)
            raise

    def _snapshot_existing(self, target: Path, root: Path) -> _Snapshot:
        if target.parent != root:
            raise ExtensionTrustError(f"extension path escapes root: {target}")
        return self._verify(target.parent, target, str(target), "")

    def _verify(self, operation: Path, tree: Path, source: str, commit: str) -> _Snapshot:
        try:
            mode = tree.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise ExtensionTrustError(f"extension tree is not a real directory: {tree}")
            manifest_text = (tree / MANIFEST_FILENAME).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ExtensionTrustError(f"cannot read manifest in {tree}: {exc}") from exc
        try:
            manifest = parse_manifest(manifest_text)
        except ExtensionManifestError as exc:
            raise ExtensionTrustError(f"invalid extension manifest in {tree}: {exc}") from exc
        files, digest = self._inventory(tree)
        return _Snapshot(operation, tree, manifest, files, digest, source, commit)

    def _inventory(self, root: Path) -> tuple[tuple[str, ...], str]:
        files = self._inventory_paths(root)
        digest = hashlib.sha256()
        total = 0
        for relative in files:
            size, file_digest = self._hash_inventory_file(root / relative, _MAX_TOTAL_BYTES - total)
            total += size
            encoded = relative.encode()
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            digest.update(size.to_bytes(8, "big"))
            digest.update(file_digest)
        return tuple(files), digest.hexdigest()

    def _inventory_paths(self, root: Path) -> list[str]:
        files: list[str] = []
        pending = [root]
        while pending:
            directory = pending.pop()
            try:
                entries = sorted(os.scandir(directory), key=lambda item: item.name, reverse=True)
            except OSError as exc:
                raise ExtensionTrustError(f"cannot inspect extension tree: {directory}") from exc
            for entry in entries:
                mode = entry.stat(follow_symlinks=False).st_mode
                if stat.S_ISLNK(mode):
                    raise ExtensionTrustError(f"extension tree contains symlink: {entry.path}")
                if stat.S_ISDIR(mode):
                    pending.append(Path(entry.path))
                    continue
                if not stat.S_ISREG(mode):
                    raise ExtensionTrustError(f"extension tree contains non-regular file: {entry.path}")
                files.append(Path(entry.path).relative_to(root).as_posix())
        files.sort()
        if len(files) > _MAX_FILES:
            raise ExtensionTrustError(f"extension tree exceeds {_MAX_FILES} files")
        return files

    def _hash_inventory_file(self, path: Path, remaining: int) -> tuple[int, bytes]:
        digest = hashlib.sha256()
        size = 0
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                while chunk := os.read(fd, 1024 * 1024):
                    size += len(chunk)
                    if size > _MAX_FILE_BYTES or size > remaining:
                        raise ExtensionTrustError("extension tree exceeds byte limit")
                    digest.update(chunk)
            finally:
                os.close(fd)
        except OSError as exc:
            raise ExtensionTrustError(f"cannot open extension file: {path}") from exc
        return size, digest.digest()

    def _review(
        self,
        snapshot: _Snapshot,
        *,
        current: _Snapshot | None,
        action: str,
        accept: bool,
        confirm: Callable[[str], str],
    ) -> None:
        manifest = snapshot.manifest
        print(f"name: {manifest.name}\nversion: {manifest.version}\napi_version: {manifest.api_version}\nentry: {manifest.entry_module}:{manifest.entry_function}\ndescription: {manifest.description}")
        print("warning: this runs arbitrary Python as you, with access to your files, credentials, and session history")
        if current is None:
            print("files:\n" + "\n".join(snapshot.files))
        else:
            old = set(current.files)
            new = set(snapshot.files)
            changed = old ^ new
            changed.update(path for path in old & new if (current.tree / path).read_bytes() != (snapshot.tree / path).read_bytes())
            print("changed files:\n" + "\n".join(sorted(changed)))
            print(f"diff summary: {len(new - old)} added, {len(old - new)} removed, {len(changed - (old ^ new))} changed")
            changelog = snapshot.tree / "CHANGELOG.md"
            if changelog.is_file():
                print("changelog:\n" + changelog.read_text(encoding="utf-8"))
        if not accept and confirm(f"{action} this extension? [y/N] ").strip().lower() != "y":
            raise ExtensionTrustError(f"{action} cancelled")
        self._checkpoint("review_accepted")

    def _read_trust(self) -> dict[str, str]:
        if not self._trust_path.exists():
            return {}
        error = f"invalid extension trust file: {self._trust_path}"
        try:
            raw = _json_object(self._trust_path.read_text(encoding="utf-8"), error)
        except (OSError, UnicodeError) as exc:
            raise ExtensionTrustError(error) from exc
        if not all(type(key) is str and type(value) is str and _DIGEST.fullmatch(value) for key, value in raw.items()):
            raise ExtensionTrustError(error)
        return {str(key): str(value) for key, value in raw.items()}

    def _change_trust(self, trust: dict[str, str], scope: str, name: str, digest: str | None) -> None:
        key = self._key(scope, name)
        if digest is None:
            trust.pop(key, None)
        else:
            trust[key] = digest
        payload = (json.dumps(trust, indent=2, sort_keys=True) + "\n").encode()
        _publish(self._trust_path, payload)

    def _publish_owner(self, snapshot: _Snapshot, owner: _UpdateOwner, root: Path) -> Path:
        payload = (json.dumps({"name": owner.name, "new_digest": owner.new_digest, "old_digest": owner.old_digest, "old_trust": owner.old_trust, "version": 1}, indent=2, sort_keys=True) + "\n").encode()
        _publish(snapshot.operation_dir / "owner.json", payload)
        os.chmod(snapshot.operation_dir / "owner.json", 0o600)
        update = snapshot.operation_dir.with_name(snapshot.operation_dir.name.replace("stage-", "update-", 1))
        os.rename(snapshot.operation_dir, update)
        _fsync_directory(update.parent)
        self._checkpoint("update_owner_published")
        return update

    def _recover_or_raise(self, root: Path) -> None:
        self._recover(root, self._read_trust())

    def _recover(self, root: Path, trust: dict[str, str]) -> None:
        recoveries = self._classify_recovery(root, trust)
        for recovery in recoveries:
            if recovery.owner is None:
                os.rmdir(recovery.directory)
                _fsync_directory(recovery.directory.parent)
            elif recovery.decision == "commit":
                self._finish_committed(recovery.directory)
            else:
                self._finish_rollback(root, recovery.directory, recovery.owner)

    def _recovery_decision(
        self,
        owner: _UpdateOwner,
        *,
        target_digest: str | None,
        backup_digest: str | None,
        tree_digest: str | None,
        current_trust: str | None,
    ) -> str:
        committed = (
            target_digest == owner.new_digest
            and current_trust == owner.new_digest
            and tree_digest is None
        )
        rollback_in_progress = (
            current_trust == owner.old_trust
            and backup_digest == owner.old_digest
            and target_digest in {None, owner.new_digest}
            and tree_digest in {None, owner.new_digest}
        )
        rolled_back = (
            target_digest == owner.old_digest
            and current_trust == owner.old_trust
            and backup_digest is None
        )
        if committed:
            return "commit"
        if rollback_in_progress or rolled_back:
            return "rollback"
        raise ExtensionTrustError(f"ambiguous extension update transaction: {owner.name}")

    def _classify_update(
        self,
        root: Path,
        path: Path,
        trust: dict[str, str],
        names: set[str],
        error: ExtensionTrustError,
    ) -> _Recovery:
        if stat.S_ISLNK(path.lstat().st_mode) or not path.is_dir():
            raise error
        entries = {item.name for item in os.scandir(path)}
        if not entries:
            return _Recovery(path, None, "empty")
        if "owner.json" not in entries or not entries <= {"owner.json", "tree", "backup"}:
            raise error
        owner = self._read_owner(path / "owner.json", error)
        if owner.name in names:
            raise ExtensionTrustError(f"ambiguous extension update transaction: {owner.name}")
        names.add(owner.name)
        decision = self._recovery_decision(
            owner,
            target_digest=self._safe_digest(root / owner.name, root, owner.name),
            backup_digest=self._safe_digest(path / "backup", path, owner.name),
            tree_digest=self._safe_digest(path / "tree", path, owner.name),
            current_trust=trust.get(self._key(self._scope_for_root(root), owner.name)),
        )
        return _Recovery(path, owner, decision)

    def _classify_recovery(self, root: Path, trust: dict[str, str]) -> tuple[_Recovery, ...]:
        found: list[_Recovery] = []
        names: set[str] = set()
        for path in sorted(self._private_root(root).glob("update-*")):
            error = ExtensionTrustError(f"invalid extension update transaction: {path}")
            try:
                found.append(self._classify_update(root, path, trust, names, error))
            except ExtensionTrustError:
                raise
            except OSError as exc:
                raise error from exc
        return tuple(found)

    def _read_owner(self, path: Path, error: ExtensionTrustError) -> _UpdateOwner:
        try:
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise error
            raw = _json_object(path.read_text(encoding="utf-8"), str(error))
        except (OSError, UnicodeError) as exc:
            raise error from exc
        keys = {"name", "new_digest", "old_digest", "old_trust", "version"}
        if len(raw) != len(keys) or any(key not in keys for key in raw):
            raise error
        name, new, old, old_trust, version = (raw["name"], raw["new_digest"], raw["old_digest"], raw["old_trust"], raw["version"])
        if type(name) is not str or not _NAME.fullmatch(name) or type(new) is not str or not _DIGEST.fullmatch(new) or type(old) is not str or not _DIGEST.fullmatch(old) or (old_trust is not None and (type(old_trust) is not str or not _DIGEST.fullmatch(old_trust))) or type(version) is not int or version != 1:
            raise error
        return _UpdateOwner(name, new, old, old_trust)

    def _finish_committed(self, update: Path) -> None:
        backup = update / "backup"
        if backup.exists():
            _remove_tree(backup, lambda: self._checkpoint("update_backup_cleanup_progress"))
        _fsync_directory(update)
        self._checkpoint("update_backup_deleted")
        self._finish_owner(update)

    def _finish_rollback(self, root: Path, update: Path, owner: _UpdateOwner) -> None:
        target = root / owner.name
        backup = update / "backup"
        tree = update / "tree"
        if backup.exists():
            if target.exists():
                if tree.exists():
                    raise ExtensionTrustError(f"ambiguous extension update transaction: {owner.name}")
                os.rename(target, tree)
                _fsync_directory(root)
                _fsync_directory(update)
            os.rename(backup, target)
            _fsync_directory(update)
            _fsync_directory(root)
        if tree.exists():
            _remove_tree(tree, lambda: self._checkpoint("update_tree_cleanup_progress"))
        _fsync_directory(update)
        self._finish_owner(update)

    def _finish_owner(self, update: Path) -> None:
        owner = update / "owner.json"
        if (update / "backup").exists() or (update / "tree").exists():
            raise ExtensionTrustError(f"invalid extension update transaction: {update}")
        if owner.exists():
            owner.unlink()
            _fsync_directory(update)
            self._checkpoint("update_owner_deleted")
        os.rmdir(update)
        _fsync_directory(update.parent)

    def _safe_digest(self, path: Path, parent: Path, expected_name: str) -> str | None:
        if not path.exists():
            return None
        try:
            snapshot = self._snapshot_existing(path, parent)
            if snapshot.manifest.name != expected_name:
                return "invalid"
            return snapshot.digest
        except ExtensionTrustError:
            return "invalid"

    def _remove_residue(self, root: Path, name: str) -> None:
        store = self._private_root(root)
        for path in sorted(store.glob("remove-*")):
            try:
                snapshot = self._snapshot_existing(path, store)
            except ExtensionTrustError:
                continue
            if snapshot.manifest.name == name:
                _remove_tree(path)

    def _target(self, root: Path, name: str) -> Path:
        target = root / name
        if not target.exists():
            raise ExtensionTrustError(f"extension not installed: {name}")
        snapshot = self._snapshot_existing(target, root)
        if snapshot.manifest.name != name:
            raise ExtensionTrustError(f"manifest name {snapshot.manifest.name!r} does not match folder {name!r}")
        return target

    def _git_metadata(self, target: Path, name: str) -> dict[str, str]:
        path = target / GIT_FILENAME
        error = ExtensionTrustError(f"extension {name} is not git-installed")
        try:
            raw = _json_object(path.read_text(encoding="utf-8"), str(error))
        except (OSError, UnicodeError):
            raise error from None
        keys = {"source", "commit"}
        if len(raw) != len(keys) or any(key not in keys for key in raw):
            raise error
        if type(raw["source"]) is not str or type(raw["commit"]) is not str or not _COMMIT.fullmatch(raw["commit"]):
            raise error
        self._parse_git_source(str(raw["source"]), required=True)
        return {"source": str(raw["source"]), "commit": str(raw["commit"])}

    @staticmethod
    def _parse_git_source(source: str, *, required: bool = False) -> tuple[str, str] | None:
        if not source.startswith("git+"):
            if required:
                raise ExtensionTrustError("git source must include @ref")
            return None
        value = source[4:]
        if "@" not in value:
            raise ExtensionTrustError("git source must include @ref")
        url, ref = value.rsplit("@", 1)
        if not url or not ref:
            raise ExtensionTrustError("git source must include a repository and ref")
        return url, ref

    @staticmethod
    def _validate_name(name: str) -> None:
        if not _NAME.fullmatch(name):
            raise ExtensionTrustError(f"invalid extension name: {name}")

    @staticmethod
    def _key(scope: str, name: str) -> str:
        return f"{scope}:{name}"

    def _scope_for_root(self, root: Path) -> str:
        return "user" if root == self.home / "extensions" else "project"

    @staticmethod
    def _discard_operation(operation: Path) -> None:
        try:
            if operation.exists():
                shutil.rmtree(operation)
        except OSError:
            pass
