"""Durable per-user and per-project settings.

Two files, layered project-over-global:

    ~/.zeta/settings.toml           (global defaults)
    <project>/.zeta/settings.toml   (project override)

Callers pass the directory that directly contains ``settings.toml`` for
each layer — for the global layer that is ``env_home()`` (already
``~/.zeta``); for the project layer that is
``discover_repo_root(cwd) / ".zeta"``.

Merge rule: tables merge recursively; every other value (scalars and lists)
from the project file replaces the global file's value. So a project may set
one table entry (``[approval]\\nallow = [...]``) without restating unrelated
tables, but replacing a list is one atomic swap.

Trust boundary: the project layer may only contribute safe keys — provider,
model, theme, token_budget, keybindings. ``yolo`` and ``[approval]`` from the
project file are IGNORED with a loud startup warning ("project settings
cannot grant approvals; see docs"). Global settings retain full key access.
A future ``/trust`` mechanism may relax this per-repo, but until then a
hostile checkout cannot silently grant itself tool approvals.

Precedence: CLI flags override settings; settings override built-in defaults.
The ``yolo`` flag is tri-state — an explicit ``--yolo`` or ``--no-yolo`` wins
either way, while an omitted flag inherits the settings value.

Malformed files fail open with a dim notice at session start; a missing file
is silent.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

SETTINGS_FILENAME = "settings.toml"
_PROVIDER_CHOICES = frozenset({"fake", "claude", "codex"})
_TOP_KEYS = frozenset(
    {
        "provider",
        "model",
        "yolo",
        "token_budget",
        "theme",
        "approval",
        "keybindings",
        "stream_stall_seconds",
        "stream_stall_retries",
    }
)
_PROJECT_SAFE_KEYS = frozenset(
    {"provider", "model", "theme", "token_budget", "keybindings"}
)
_APPROVAL_KEYS = frozenset({"allow", "deny", "ask"})
_EMPTY_MAPPING: Mapping[str, Any] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated settings from the merged global+project files."""

    provider: str | None = None
    model: str | None = None
    yolo: bool | None = None
    token_budget: int | None = None
    theme: str | None = None
    approval_allow: tuple[str, ...] = ()
    approval_deny: tuple[str, ...] = ()
    approval_ask: tuple[str, ...] = ()
    keybindings: Mapping[str, Any] = field(default_factory=lambda: _EMPTY_MAPPING)
    stream_stall_seconds: int | None = None
    stream_stall_retries: int | None = None


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    """CLI-vs-settings-vs-default resolution for one session."""

    provider: str
    model: str | None
    yolo: bool
    token_budget: int | None
    theme: str | None
    approval_allow: tuple[str, ...]
    approval_deny: tuple[str, ...]
    approval_ask: tuple[str, ...]
    keybindings: Mapping[str, Any]
    stream_stall_seconds: int | None = None
    stream_stall_retries: int | None = None


@dataclass(frozen=True, slots=True)
class LoadedSettings:
    """The resolved settings, dim notices, and loud warnings for session start."""

    settings: Settings
    notices: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def load_settings(
    *,
    home: str | Path | None = None,
    project_dir: str | Path | None = None,
) -> LoadedSettings:
    """Read, deep-merge, and validate the layered settings files.

    ``home`` and ``project_dir`` are the directories that directly contain
    ``settings.toml`` (``~/.zeta`` and ``<repo>/.zeta`` in production).
    """

    notices: list[str] = []
    warnings: list[str] = []
    global_path = _settings_path(home)
    project_path = _settings_path(project_dir)
    global_data = _parse(global_path, notices)
    project_data = (
        _parse(project_path, notices) if project_path != global_path else {}
    )
    project_data = _strip_unsafe_project_keys(project_data, project_path, warnings)
    merged = _deep_merge(global_data, project_data)
    settings = _validate(merged, notices)
    return LoadedSettings(settings, tuple(notices), tuple(warnings))


def resolve(
    settings: Settings,
    *,
    cli_provider: str | None,
    cli_model: str | None,
    cli_yolo: bool | None,
    cli_token_budget: int | None,
    default_provider: str = "fake",
) -> ResolvedConfig:
    """Layer CLI flags over the loaded settings; CLI wins where set.

    ``cli_yolo`` is tri-state: ``True`` for an explicit ``--yolo``, ``False``
    for an explicit ``--no-yolo`` (both override settings), and ``None`` when
    the flag was omitted (settings value inherited).
    """

    provider = cli_provider or settings.provider or default_provider
    yolo = bool(settings.yolo) if cli_yolo is None else cli_yolo
    token_budget = (
        cli_token_budget if cli_token_budget is not None else settings.token_budget
    )
    return ResolvedConfig(
        provider=provider,
        model=cli_model or settings.model,
        yolo=yolo,
        token_budget=token_budget,
        theme=settings.theme,
        approval_allow=settings.approval_allow,
        approval_deny=settings.approval_deny,
        approval_ask=settings.approval_ask,
        keybindings=settings.keybindings,
        stream_stall_seconds=settings.stream_stall_seconds,
        stream_stall_retries=settings.stream_stall_retries,
    )


def _settings_path(base: str | Path | None) -> Path | None:
    if base is None:
        return None
    return Path(base).expanduser() / SETTINGS_FILENAME


def _display_path(path: Path) -> str:
    """Collapse ``$HOME`` prefixes to ``~/`` so notices do not leak layouts."""

    try:
        home = Path.home()
    except (RuntimeError, OSError):
        return str(path)
    try:
        relative = path.relative_to(home)
    except ValueError:
        return str(path)
    return f"~/{relative}"


def _parse(path: Path | None, notices: list[str]) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return {}
    except OSError as exc:
        notices.append(f"settings · could not read {_display_path(path)}: {exc}")
        return {}
    try:
        parsed = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        notices.append(f"settings · ignored {_display_path(path)}: {exc}")
        return {}
    if not isinstance(parsed, dict):
        notices.append(
            f"settings · ignored {_display_path(path)}: top-level is not a table"
        )
        return {}
    return parsed


def _strip_unsafe_project_keys(
    data: dict[str, Any],
    path: Path | None,
    warnings: list[str],
) -> dict[str, Any]:
    unsafe = sorted(data.keys() - _PROJECT_SAFE_KEYS)
    if not unsafe:
        return data
    where = _display_path(path) if path is not None else "project settings"
    warnings.append(
        "settings · project settings cannot grant approvals; "
        f"ignoring {', '.join(unsafe)} in {where} (see docs)"
    )
    return {key: value for key, value in data.items() if key not in unsafe}


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _validate(data: Mapping[str, Any], notices: list[str]) -> Settings:
    for key in data.keys() - _TOP_KEYS:
        notices.append(f"settings · ignored unknown key '{key}'")
    provider = _validated_choice(data, "provider", _PROVIDER_CHOICES, notices)
    model = _validated_string(data, "model", notices)
    theme = _validated_string(data, "theme", notices)
    yolo = _validated_bool(data, "yolo", notices)
    token_budget = _validated_positive_int(data, "token_budget", notices)
    stream_stall_seconds = _validated_positive_int(
        data, "stream_stall_seconds", notices
    )
    stream_stall_retries = _validated_nonnegative_int(
        data, "stream_stall_retries", notices
    )
    allow, deny, ask = _validated_approval(data, notices)
    keybindings = _validated_keybindings(data, notices)
    return Settings(
        provider=provider,
        model=model,
        yolo=yolo,
        token_budget=token_budget,
        theme=theme,
        approval_allow=allow,
        approval_deny=deny,
        approval_ask=ask,
        keybindings=keybindings,
        stream_stall_seconds=stream_stall_seconds,
        stream_stall_retries=stream_stall_retries,
    )


def _validated_string(
    data: Mapping[str, Any], key: str, notices: list[str]
) -> str | None:
    if key not in data:
        return None
    value = data[key]
    if type(value) is not str:
        notices.append(f"settings · ignored key '{key}': expected string")
        return None
    return value


def _validated_choice(
    data: Mapping[str, Any],
    key: str,
    choices: frozenset[str],
    notices: list[str],
) -> str | None:
    value = _validated_string(data, key, notices)
    if value is None:
        return None
    if value not in choices:
        notices.append(
            f"settings · ignored key '{key}': "
            f"{value!r} not in {sorted(choices)}"
        )
        return None
    return value


def _validated_bool(
    data: Mapping[str, Any], key: str, notices: list[str]
) -> bool | None:
    if key not in data:
        return None
    value = data[key]
    if type(value) is not bool:
        notices.append(f"settings · ignored key '{key}': expected boolean")
        return None
    return value


def _validated_positive_int(
    data: Mapping[str, Any], key: str, notices: list[str]
) -> int | None:
    if key not in data:
        return None
    value = data[key]
    if type(value) is not int or value <= 0:
        notices.append(f"settings · ignored key '{key}': expected positive integer")
        return None
    return value


def _validated_nonnegative_int(
    data: Mapping[str, Any], key: str, notices: list[str]
) -> int | None:
    if key not in data:
        return None
    value = data[key]
    if type(value) is not int or value < 0:
        notices.append(f"settings · ignored key '{key}': expected nonnegative integer")
        return None
    return value


def _validated_approval(
    data: Mapping[str, Any], notices: list[str]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if "approval" not in data:
        return (), (), ()
    table = data["approval"]
    if not isinstance(table, Mapping):
        notices.append("settings · ignored key 'approval': expected table")
        return (), (), ()
    for key in table.keys() - _APPROVAL_KEYS:
        notices.append(f"settings · ignored unknown key 'approval.{key}'")
    return (
        _validated_str_list("approval.allow", table.get("allow"), notices),
        _validated_str_list("approval.deny", table.get("deny"), notices),
        _validated_str_list("approval.ask", table.get("ask"), notices),
    )


def _validated_str_list(
    label: str,
    value: Any,
    notices: list[str],
) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        notices.append(f"settings · ignored key '{label}': expected list of strings")
        return ()
    entries: list[str] = []
    for item in value:
        if type(item) is not str:
            notices.append(
                f"settings · ignored key '{label}': every entry must be a string"
            )
            return ()
        entries.append(item)
    return tuple(entries)


def _validated_keybindings(
    data: Mapping[str, Any], notices: list[str]
) -> Mapping[str, Any]:
    if "keybindings" not in data:
        return _EMPTY_MAPPING
    value = data["keybindings"]
    if not isinstance(value, Mapping):
        notices.append("settings · ignored key 'keybindings': expected table")
        return _EMPTY_MAPPING
    entries: dict[str, str] = {}
    for name, binding in value.items():
        if type(binding) is not str:
            notices.append(
                f"settings · ignored key 'keybindings.{name}': expected string"
            )
            continue
        entries[name] = binding
    if not entries:
        return _EMPTY_MAPPING
    return MappingProxyType(entries)


__all__ = [
    "SETTINGS_FILENAME",
    "LoadedSettings",
    "ResolvedConfig",
    "Settings",
    "load_settings",
    "resolve",
]
