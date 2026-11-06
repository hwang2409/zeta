"""Home-wide stored-format gate and lossless JSON publication."""

from __future__ import annotations

import asyncio
import copy
import fcntl
import json
import os
import threading
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Self

WRITER_EPOCH: Final = 1
_MANIFEST = "storage-format.json"
_LOCK = ".storage-format.lock"


class StorageFormatError(ValueError):
    """Raised when a home cannot be accessed by this stored-format version."""


@dataclass(frozen=True, slots=True)
class VerificationReport:
    writer_epoch: int


@dataclass(frozen=True, slots=True)
class SetValue:
    path: tuple[str, ...]
    value: Any


@dataclass(frozen=True, slots=True)
class DeleteValue:
    path: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReplaceSubtree:
    """Replace a subtree that this format explicitly owns in full."""

    path: tuple[str, ...]
    value: Any


type Mutation = SetValue | DeleteValue | ReplaceSubtree
PublishBytes = Callable[[int, str, bytes], None]


@dataclass(slots=True)
class _RegistryEntry:
    fd: int
    references: int
    epoch: int


_REGISTRY: dict[Path, _RegistryEntry] = {}
_REGISTRY_LOCK = threading.Lock()


def _canonical_home(home: str | Path) -> Path:
    return Path(home).expanduser().resolve()


def _open_lock(home: Path) -> int:
    home.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(home / _LOCK, flags, 0o600)
    os.set_inheritable(fd, False)
    if os.get_inheritable(fd):
        os.close(fd)
        raise StorageFormatError("stored-format lock descriptor is inheritable")
    return fd


def _read_manifest(home: Path, *, absent_epoch: int | None = None) -> int:
    try:
        raw = (home / _MANIFEST).read_bytes()
    except FileNotFoundError:
        if absent_epoch is not None:
            return absent_epoch
        raise StorageFormatError(
            f"stored-format manifest is missing at {home}; run the offline format advance"
        ) from None
    try:
        decoder = json.JSONDecoder()
        text = raw.decode()
        value, end = decoder.raw_decode(text)
        if text[end:].strip() or type(value) is not dict or set(value) != {"writer_epoch"}:
            raise ValueError
        epoch = value["writer_epoch"]
        if type(epoch) is not int or epoch < 0:
            raise ValueError
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise StorageFormatError(f"stored-format manifest is invalid at {home}") from exc
    return epoch


def open_writer(home: str | Path, *, capability: int = WRITER_EPOCH) -> WriterLease:
    """Retain a process-shared home lease after checking writer compatibility."""

    canonical = _canonical_home(home)
    with _REGISTRY_LOCK:
        registered = _REGISTRY.get(canonical)
        if registered is not None:
            if registered.epoch > capability:
                raise StorageFormatError(
                    f"stored format {registered.epoch} is newer than writer capability {capability}"
                )
            registered.references += 1
            return WriterLease(canonical, registered)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise StorageFormatError(
                "stored-format lock acquisition must run off the event loop"
            )
        fd = _open_lock(canonical)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
            epoch = _read_manifest(canonical)
            if epoch > capability:
                raise StorageFormatError(
                    f"stored format {epoch} is newer than writer capability {capability}"
                )
            registered = _RegistryEntry(fd=fd, references=1, epoch=epoch)
            _REGISTRY[canonical] = registered
            return WriterLease(canonical, registered)
        except BaseException:
            os.close(fd)
            raise


def _publish_manifest(home: Path, epoch: int) -> None:
    temporary = home / f".{_MANIFEST}.{uuid.uuid4().hex}.tmp"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(f'{{"writer_epoch":{epoch}}}\n'.encode())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, home / _MANIFEST)
        directory_fd = os.open(home, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def advance(
    home: str | Path, *, expected_epoch: int, target_epoch: int = WRITER_EPOCH
) -> None:
    """Advance an idle home to a strictly newer supported writer epoch."""

    if target_epoch > WRITER_EPOCH or target_epoch <= expected_epoch:
        raise StorageFormatError("stored-format target must be newer and supported")
    canonical = _canonical_home(home)
    fd = _open_lock(canonical)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StorageFormatError("stored-format home has active writers") from exc
        observed = _read_manifest(canonical, absent_epoch=0)
        if observed != expected_epoch:
            raise StorageFormatError(
                f"stored-format epoch changed: expected {expected_epoch}, found {observed}"
            )
        _publish_manifest(canonical, target_epoch)
    finally:
        os.close(fd)


def verify(home: str | Path, *, capability: int = WRITER_EPOCH) -> VerificationReport:
    canonical = _canonical_home(home)
    epoch = _read_manifest(canonical)
    if epoch > capability:
        raise StorageFormatError(
            f"stored format {epoch} is newer than writer capability {capability}"
        )
    return VerificationReport(writer_epoch=epoch)


def _mapping_at(root: dict[str, Any], path: tuple[str, ...], *, create: bool) -> tuple[dict[str, Any], str]:
    if not path:
        raise StorageFormatError("a mutation path cannot be empty")
    parent = root
    for key in path[:-1]:
        child = parent.get(key)
        if child is None and create:
            child = {}
            parent[key] = child
        if type(child) is not dict:
            raise StorageFormatError(f"stored JSON path is not an object: {'.'.join(path)}")
        parent = child
    return parent, path[-1]


def _merge_mapping(existing: dict[str, Any], value: Mapping[str, Any]) -> None:
    for key, child in value.items():
        if type(key) is not str:
            raise StorageFormatError("stored JSON object keys must be strings")
        current = existing.get(key)
        if type(child) is dict and type(current) is dict:
            _merge_mapping(current, child)
        else:
            existing[key] = copy.deepcopy(child)


def apply_mutations(raw: object, mutations: Iterable[Mutation]) -> dict[str, Any]:
    """Apply exact declared deltas while retaining every unowned JSON key."""

    if type(raw) is not dict:
        raise StorageFormatError("stored JSON root must be an object")
    output = copy.deepcopy(raw)
    for mutation in mutations:
        parent, key = _mapping_at(output, mutation.path, create=not isinstance(mutation, DeleteValue))
        if isinstance(mutation, DeleteValue):
            parent.pop(key, None)
        elif isinstance(mutation, ReplaceSubtree):
            parent[key] = copy.deepcopy(mutation.value)
        else:
            current = parent.get(key)
            if type(mutation.value) is dict and type(current) is dict:
                _merge_mapping(current, mutation.value)
            else:
                parent[key] = copy.deepcopy(mutation.value)
    return output


def mapping_mutations(
    value: Mapping[str, Any],
    *,
    delete: Iterable[str] = (),
    broad: Iterable[str] = (),
) -> tuple[Mutation, ...]:
    """Describe a normalized mapping as leaf patches and explicit broad sequences."""

    broad_paths = frozenset(broad)
    mutations: list[Mutation] = [DeleteValue((key,)) for key in delete]
    for key, item in value.items():
        mutation_type = (
            ReplaceSubtree if isinstance(item, list) or key in broad_paths else SetValue
        )
        mutations.append(mutation_type((key,), item))
    return tuple(mutations)


class WriterLease:
    """A retained shared lock that is the sole authoritative publication seam."""

    __slots__ = ("_closed", "_entry", "home")

    def __init__(self, home: Path, entry: _RegistryEntry) -> None:
        self.home = home
        self._entry = entry
        self._closed = False

    def retain(self) -> WriterLease:
        if self._closed:
            raise StorageFormatError("stored-format lease is closed")
        with _REGISTRY_LOCK:
            self._entry.references += 1
        return WriterLease(self.home, self._entry)

    def publish_json(
        self,
        directory_fd: int,
        name: str,
        mutations: Iterable[Mutation],
        publish: PublishBytes,
        *,
        initial: object | None = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise StorageFormatError("stored-format lease is closed")
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        except FileNotFoundError:
            raw = {} if initial is None else initial
        else:
            try:
                with os.fdopen(fd, "rb") as stream:
                    raw = json.load(stream)
            except (OSError, ValueError) as exc:
                raise StorageFormatError(f"stored JSON is invalid: {name}") from exc
        output = apply_mutations(raw, mutations)
        encoded = (json.dumps(output, separators=(",", ":"), sort_keys=True) + "\n").encode()
        publish(directory_fd, name, encoded)
        return output

    def publish_records(
        self,
        directory_fd: int,
        name: str,
        records: Iterable[Mapping[str, Any]],
        *,
        identity: str,
        publish: PublishBytes,
    ) -> list[dict[str, Any]]:
        """Patch an object-record collection by identity and delete absent records."""

        if self._closed:
            raise StorageFormatError("stored-format lease is closed")
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        except FileNotFoundError:
            raw: object = []
        else:
            try:
                with os.fdopen(fd, "rb") as stream:
                    raw = json.load(stream)
            except (OSError, ValueError) as exc:
                raise StorageFormatError(f"stored JSON is invalid: {name}") from exc
        if type(raw) is not list or any(type(item) is not dict for item in raw):
            raise StorageFormatError(f"stored JSON record collection is invalid: {name}")
        existing = {
            item.get(identity): copy.deepcopy(item)
            for item in raw
            if type(item.get(identity)) is str
        }
        output: list[dict[str, Any]] = []
        seen: set[str] = set()
        for record in records:
            record_id = record.get(identity)
            if type(record_id) is not str or not record_id or record_id in seen:
                raise StorageFormatError(f"stored JSON record identity is invalid: {name}")
            seen.add(record_id)
            current = existing.get(record_id, {})
            _merge_mapping(current, record)
            output.append(current)
        encoded = (json.dumps(output, separators=(",", ":"), sort_keys=True) + "\n").encode()
        publish(directory_fd, name, encoded)
        return output

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with _REGISTRY_LOCK:
            self._entry.references -= 1
            if self._entry.references:
                return
            _REGISTRY.pop(self.home, None)
            fd, self._entry.fd = self._entry.fd, -1
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
