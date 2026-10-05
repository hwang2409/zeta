"""The ``[computer]`` table of the global ``settings.toml``.

Only the global file can configure computer use; project settings cannot.

```toml
[computer]
backend = "local"      # desktop backend
cpus = 4               # sandbox VM size, applied when setup creates the VM
memory_gib = 6
disk_gib = 50
recording = true       # record frames and actions to the session directory
desktop_minutes = 60   # maximum lifetime of one desktop
```
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .backend import BACKENDS, DEFAULT_BACKEND


class ComputerSettingsError(ValueError):
    """The ``[computer]`` table is invalid."""


@dataclass(frozen=True, slots=True)
class ComputerSettings:
    backend: str = DEFAULT_BACKEND
    cpus: int = 4
    memory_gib: int = 6
    disk_gib: int = 50
    recording: bool = True
    desktop_minutes: int = 60


_INT_LIMITS = {
    "cpus": (1, 64),
    "memory_gib": (2, 256),
    "disk_gib": (20, 2048),
    "desktop_minutes": (1, 24 * 60),
}


def load_computer_settings(home: Path) -> ComputerSettings:
    """Read and validate ``[computer]`` from ``home/settings.toml``."""

    path = home / "settings.toml"
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ComputerSettings()
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ComputerSettingsError(f"{path}: {exc}") from exc
    table = data.get("computer", {})
    if not isinstance(table, dict):
        raise ComputerSettingsError("settings: computer must be a table")
    defaults = ComputerSettings()
    unknown = set(table) - set(ComputerSettings.__dataclass_fields__)
    if unknown:
        raise ComputerSettingsError(
            f"settings: unknown computer keys: {', '.join(sorted(unknown))}"
        )
    values: dict[str, object] = {}
    for key, (low, high) in _INT_LIMITS.items():
        value = table.get(key, getattr(defaults, key))
        if type(value) is not int or not low <= value <= high:
            raise ComputerSettingsError(f"settings: computer.{key} must be an integer {low}..{high}")
        values[key] = value
    recording = table.get("recording", defaults.recording)
    if type(recording) is not bool:
        raise ComputerSettingsError("settings: computer.recording must be true or false")
    backend = table.get("backend", defaults.backend)
    if backend not in BACKENDS:
        raise ComputerSettingsError(
            f"settings: computer.backend must be one of: {', '.join(sorted(BACKENDS))}"
        )
    return ComputerSettings(backend=backend, recording=recording, **values)


__all__ = ["ComputerSettings", "ComputerSettingsError", "load_computer_settings"]
