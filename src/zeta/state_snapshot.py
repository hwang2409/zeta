"""Small projection helpers for lock-free owner snapshots."""

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

BODY_BYTES = 64 * 1024


def stamps(paths: Iterable[Path]) -> tuple[object, ...]:
    result = []
    for path in paths:
        try:
            info = path.lstat()
        except FileNotFoundError:
            result.append((path, None))
        else:
            result.append((path, (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)))
    return tuple(result)


def timestamp_ns(text: str) -> int:
    value = datetime.fromisoformat(text)
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    delta = value - datetime(1970, 1, 1, tzinfo=UTC)
    return max(0, ((delta.days * 86400 + delta.seconds) * 10**6 + delta.microseconds) * 1000)


def clip(text: str, cap: int) -> tuple[str, bool]:
    data = text.encode()
    return data[:cap].decode(errors="ignore"), len(data) > cap
