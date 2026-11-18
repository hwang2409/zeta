"""Database paths, sizes, and retention limits shared by the receiver and serve."""

from pathlib import Path

RAW_DAYS = 14
ROLLUP_DAYS = 90
MAX_BYTES = 1 << 30


def database_path(home: Path) -> Path:
    return home / "telemetry" / "telemetry.sqlite3"


def used_bytes(home: Path) -> int:
    path = database_path(home)
    total = 0
    for suffix in ("", "-wal", "-shm"):
        try:
            total += path.with_name(path.name + suffix).stat().st_size
        except FileNotFoundError:
            pass
    return total
