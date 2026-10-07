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
model, token_budget, compaction, workspace_snapshot_cap, tools, and
disallowed_tools. Project tool policy is cumulative: allowlists intersect and
denylists are combined. ``yolo``, ``allow_hooks``, ``allow_external_tools``,
``[approval]``, ``theme``, and ``[keybindings]`` from the project file are
IGNORED with a loud startup warning. Global settings retain full key access. A
future
``/trust`` mechanism may relax this per-repo, but until then a hostile
checkout cannot silently grant itself tool approvals, remap ``ctrl-c`` to
exfiltrate the composer, or hide the abort key.

Precedence: CLI flags override settings; settings override built-in defaults.
The ``yolo`` flag is tri-state — an explicit ``--yolo`` or ``--no-yolo`` wins
either way, while an omitted flag inherits the settings value.

Malformed TOML files stop startup because they can contain security policy; a
missing file is silent. Invalid tool policy entries also stop startup rather
than silently removing a restriction. Approval entries are ``tool`` or
``tool(pattern)`` (ZETA-86); an entry that does not parse is dropped with a
loud warning, since a rule the
user wrote that silently never applies would change what gets approved.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from fnmatch import fnmatchcase
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ..compaction import COMPACTION_MODES, DEFAULT_SESSION_COMPACTION
from ..core.approval import parse_approval_rule
from .tool_policy import parse_tool_patterns, validate_tool_patterns

SETTINGS_FILENAME = "settings.toml"


class SettingsError(ValueError):
    """A security-sensitive settings value is invalid."""


_PROVIDER_CHOICES = frozenset({"fake", "claude", "codex", "ollama"})
_TOP_KEYS = frozenset(
    {
        "provider",
        "model",
        "yolo",
        "token_budget",
        "compaction",
        "theme",
        "approval",
        "keybindings",
        "stream_stall_seconds",
        "stream_stall_retries",
        "workspace_snapshot_cap",
        "ollama_base_url",
        "auto_project",
        "memory",
        "inbox",
        "tools",
        "disallowed_tools",
        "allow_hooks",
        "allow_external_tools",
        # Parsed by zeta.remote_sync; global only.
        "remotes",
        # Validated by zeta.computer.settings; global only.
        "computer",
    }
)
_PROJECT_SAFE_KEYS = frozenset(
    {
        "provider",
        "model",
        "token_budget",
        "compaction",
        "workspace_snapshot_cap",
        "tools",
        "disallowed_tools",
    }
)
_APPROVAL_KEYS = frozenset({"allow", "deny", "ask"})
_INBOX_KEYS = frozenset({"enabled"})
_EMPTY_MAPPING: Mapping[str, Any] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated settings from the merged global+project files."""

    provider: str | None = None
    model: str | None = None
    yolo: bool | None = None
    token_budget: int | None = None
    compaction: str | None = None
    theme: str | None = None
    approval_allow: tuple[str, ...] = ()
    approval_deny: tuple[str, ...] = ()
    approval_ask: tuple[str, ...] = ()
    keybindings: Mapping[str, Any] = field(default_factory=lambda: _EMPTY_MAPPING)
    stream_stall_seconds: int | None = None
    stream_stall_retries: int | None = None
    workspace_snapshot_cap: int | None = None
    ollama_base_url: str | None = None
    auto_project: bool | None = None
    inbox_enabled: bool | None = None
    tool_allow: tuple[str, ...] | None = None
    tool_deny: tuple[str, ...] = ()
    tool_allow_layers: tuple[tuple[str, ...], ...] = ()
    allow_hooks: bool | None = None
    allow_external_tools: bool | None = None
    memory_auto: bool | None = None
    memory_model: str | None = None
    memory_token_threshold: int | None = None
    memory_idle_minutes: int | None = None


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
    compaction: str = DEFAULT_SESSION_COMPACTION
    compaction_pinned: bool = False
    stream_stall_seconds: int | None = None
    stream_stall_retries: int | None = None
    workspace_snapshot_cap: int | None = None
    auto_project: bool = True
    inbox_enabled: bool = True
    tool_allow: tuple[str, ...] | None = None
    tool_deny: tuple[str, ...] = ()
    tool_allow_layers: tuple[tuple[str, ...], ...] = ()
    allow_hooks: bool = False
    allow_external_tools: bool = False
    memory_auto: bool = True
    memory_model: str = "gpt-5.6-luna"
    memory_token_threshold: int = 50_000
    memory_idle_minutes: int = 10


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
    global_allow, global_deny = _validate_policy_layer(global_data, global_path)
    project_allow, project_deny = _validate_policy_layer(project_data, project_path)
    merged = _deep_merge(global_data, project_data)
    tool_allow, tool_allow_layers, tool_deny = _merge_tool_policy(
        global_data=global_data,
        project_data=project_data,
        global_allow=global_allow,
        project_allow=project_allow,
        global_deny=global_deny,
        project_deny=project_deny,
        notices=notices,
    )
    if tool_allow is None:
        merged.pop("tools", None)
    else:
        merged["tools"] = tool_allow
    merged["disallowed_tools"] = tool_deny
    settings = replace(
        _validate(merged, notices, warnings),
        tool_allow_layers=tool_allow_layers,
    )
    return LoadedSettings(settings, tuple(notices), tuple(warnings))


def resolve(
    settings: Settings,
    *,
    cli_provider: str | None,
    cli_model: str | None,
    cli_yolo: bool | None,
    cli_token_budget: int | None,
    cli_compaction: str | None = None,
    cli_tools: str | None = None,
    cli_disallowed_tools: str | None = None,
    cli_allow_hooks: bool | None = None,
    cli_auto_memory: bool | None = None,
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
    cli_allow = parse_tool_patterns(cli_tools, field="--tools")
    cli_deny = parse_tool_patterns(
        cli_disallowed_tools, field="--disallowed-tools"
    )
    return ResolvedConfig(
        provider=provider,
        model=cli_model or settings.model,
        yolo=yolo,
        token_budget=token_budget,
        compaction=cli_compaction or settings.compaction or DEFAULT_SESSION_COMPACTION,
        compaction_pinned=cli_compaction is not None or settings.compaction is not None,
        theme=settings.theme,
        approval_allow=settings.approval_allow,
        approval_deny=settings.approval_deny,
        approval_ask=settings.approval_ask,
        keybindings=settings.keybindings,
        stream_stall_seconds=settings.stream_stall_seconds,
        stream_stall_retries=settings.stream_stall_retries,
        workspace_snapshot_cap=settings.workspace_snapshot_cap,
        auto_project=settings.auto_project is not False,
        inbox_enabled=settings.inbox_enabled is not False,
        tool_allow=settings.tool_allow if cli_allow is None else cli_allow,
        tool_deny=settings.tool_deny if cli_deny is None else cli_deny,
        tool_allow_layers=(
            settings.tool_allow_layers if cli_allow is None else (cli_allow,)
        ),
        allow_hooks=(
            bool(settings.allow_hooks)
            if cli_allow_hooks is None
            else cli_allow_hooks
        ),
        allow_external_tools=bool(settings.allow_external_tools),
        memory_auto=(
            settings.memory_auto is not False
            if cli_auto_memory is None
            else cli_auto_memory
        ),
        memory_model=settings.memory_model or "gpt-5.6-luna",
        memory_token_threshold=settings.memory_token_threshold or 50_000,
        memory_idle_minutes=settings.memory_idle_minutes or 10,
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
        raise SettingsError(
            f"could not parse settings file {_display_path(path)}: {exc}"
        ) from exc
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
        "settings · project layer cannot grant approvals or remap "
        "keybindings/theme; "
        f"ignoring {', '.join(unsafe)} in {where} (see docs)"
    )
    return {key: value for key, value in data.items() if key not in unsafe}


def _validate_policy_layer(
    data: Mapping[str, Any], path: Path | None
) -> tuple[tuple[str, ...] | None, tuple[str, ...] | None]:
    """Validate one trust layer before policy composition."""

    where = _display_path(path) if path is not None else "settings"

    def patterns(key: str) -> tuple[str, ...] | None:
        if key not in data:
            return None
        raw = data[key]
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise SettingsError(
                f"invalid policy in {where}: {key} must be a list of patterns"
            )
        try:
            return validate_tool_patterns(raw, field=key)
        except (TypeError, ValueError) as exc:
            raise SettingsError(f"invalid policy in {where}: {exc}") from exc

    return patterns("tools"), patterns("disallowed_tools")


def _allow_pattern_is_narrower(pattern: str, parents: tuple[str, ...]) -> bool:
    """Return true only for simple, provable glob containment."""

    if not any(character in pattern for character in "*?["):
        return any(fnmatchcase(pattern, parent) for parent in parents)
    for parent in parents:
        if pattern == parent or parent == "*":
            return True
        if (
            pattern.endswith("*")
            and parent.endswith("*")
            and not any(character in pattern[:-1] for character in "?[")
            and not any(character in parent[:-1] for character in "?[")
            and pattern[:-1].startswith(parent[:-1])
        ):
            return True
    return False


def _merge_tool_policy(
    *,
    global_data: Mapping[str, Any],
    project_data: Mapping[str, Any],
    global_allow: tuple[str, ...] | None,
    project_allow: tuple[str, ...] | None,
    global_deny: tuple[str, ...] | None,
    project_deny: tuple[str, ...] | None,
    notices: list[str],
) -> tuple[
    tuple[str, ...] | None,
    tuple[tuple[str, ...], ...],
    tuple[str, ...],
]:
    """Compose untrusted project policy as an additional restriction."""

    allow_layers = tuple(
        layer for layer in (global_allow, project_allow) if layer is not None
    )
    effective_allow = allow_layers[-1] if allow_layers else None
    effective_deny = tuple(
        dict.fromkeys((*(global_deny or ()), *(project_deny or ())))
    )

    allow_widens = (
        "tools" in global_data
        and "tools" in project_data
        and bool(project_allow)
        and not all(
            _allow_pattern_is_narrower(pattern, global_allow or ())
            for pattern in project_allow
        )
    )
    deny_widens = (
        "disallowed_tools" in global_data
        and "disallowed_tools" in project_data
        and bool(global_deny)
        and not set(global_deny or ()).issubset(project_deny or ())
    )
    if allow_widens or deny_widens:
        notices.append(
            "settings · project tool policy cannot widen global policy; "
            "applying allowlists cumulatively and preserving global denials"
        )
    return effective_allow, allow_layers, effective_deny


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _validate(
    data: Mapping[str, Any], notices: list[str], warnings: list[str]
) -> Settings:
    for key in data.keys() - _TOP_KEYS:
        notices.append(f"settings · ignored unknown key '{key}'")
    provider = _validated_choice(data, "provider", _PROVIDER_CHOICES, notices)
    model = _validated_string(data, "model", notices)
    theme = _validated_string(data, "theme", notices)
    yolo = _validated_bool(data, "yolo", notices)
    token_budget = _validated_positive_int(data, "token_budget", notices)
    compaction = _validated_choice(
        data, "compaction", COMPACTION_MODES, notices
    )
    stream_stall_seconds = _validated_positive_int(
        data, "stream_stall_seconds", notices
    )
    stream_stall_retries = _validated_nonnegative_int(
        data, "stream_stall_retries", notices
    )
    ollama_base_url = _validated_string(data, "ollama_base_url", notices)
    workspace_snapshot_cap = _validated_positive_int(
        data, "workspace_snapshot_cap", notices
    )
    auto_project = _validated_bool(data, "auto_project", notices)
    inbox_enabled = _validated_inbox(data, notices)
    allow_hooks = _validated_bool(data, "allow_hooks", notices)
    allow_external_tools = _validated_bool(data, "allow_external_tools", notices)
    memory = data.get("memory", {})
    if not isinstance(memory, Mapping):
        notices.append("settings · ignored key 'memory': expected table")
        memory = {}
    for key in memory.keys() - {"auto", "model", "token_threshold", "idle_minutes"}:
        notices.append(f"settings · ignored unknown key 'memory.{key}'")
    memory_auto = _validated_bool(memory, "auto", notices)
    memory_model = _validated_string(memory, "model", notices)
    memory_token_threshold = _validated_positive_int(
        memory, "token_threshold", notices
    )
    memory_idle_minutes = _validated_positive_int(memory, "idle_minutes", notices)
    tool_allow = _validated_tool_patterns(data, "tools", notices, optional=True)
    tool_deny = _validated_tool_patterns(
        data, "disallowed_tools", notices, optional=False
    )
    allow, deny, ask = _validated_approval(data, notices, warnings)
    keybindings = _validated_keybindings(data, notices)
    return Settings(
        provider=provider,
        model=model,
        yolo=yolo,
        token_budget=token_budget,
        compaction=compaction,
        theme=theme,
        approval_allow=allow,
        approval_deny=deny,
        approval_ask=ask,
        keybindings=keybindings,
        stream_stall_seconds=stream_stall_seconds,
        stream_stall_retries=stream_stall_retries,
        workspace_snapshot_cap=workspace_snapshot_cap,
        ollama_base_url=ollama_base_url,
        auto_project=auto_project,
        inbox_enabled=inbox_enabled,
        tool_allow=tool_allow,
        tool_deny=tool_deny or (),
        tool_allow_layers=() if tool_allow is None else (tool_allow,),
        allow_hooks=allow_hooks,
        allow_external_tools=allow_external_tools,
        memory_auto=memory_auto,
        memory_model=memory_model,
        memory_token_threshold=memory_token_threshold,
        memory_idle_minutes=memory_idle_minutes,
    )


def _validated_inbox(
    data: Mapping[str, Any], notices: list[str]
) -> bool | None:
    if "inbox" not in data:
        return None
    table = data["inbox"]
    if not isinstance(table, Mapping):
        notices.append("settings · ignored key 'inbox': expected table")
        return None
    for key in table.keys() - _INBOX_KEYS:
        notices.append(f"settings · ignored unknown key 'inbox.{key}'")
    return _validated_bool(table, "enabled", notices)


def _validated_tool_patterns(
    data: Mapping[str, Any],
    key: str,
    notices: list[str],
    *,
    optional: bool,
) -> tuple[str, ...] | None:
    if key not in data:
        return None if optional else ()
    raw = data[key]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or any(
        type(item) is not str for item in raw
    ):
        _validated_str_list(key, raw, notices)
        return None if optional else ()
    try:
        return validate_tool_patterns(raw, field=key)
    except (TypeError, ValueError) as exc:
        notices.append(f"settings · ignored key '{key}': {exc}")
        return None if optional else ()


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
    data: Mapping[str, Any], notices: list[str], warnings: list[str]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if "approval" not in data:
        return (), (), ()
    table = data["approval"]
    if not isinstance(table, Mapping):
        notices.append("settings · ignored key 'approval': expected table")
        return (), (), ()
    for key in table.keys() - _APPROVAL_KEYS:
        notices.append(f"settings · ignored unknown key 'approval.{key}'")
    return tuple(
        _validated_rules(
            f"approval.{key}",
            _validated_str_list(f"approval.{key}", table.get(key), notices),
            warnings,
        )
        for key in ("allow", "deny", "ask")
    )


def _validated_rules(
    label: str,
    entries: tuple[str, ...],
    warnings: list[str],
) -> tuple[str, ...]:
    """Keep only entries that parse as ``tool`` or ``tool(pattern)``."""

    kept: list[str] = []
    for entry in entries:
        try:
            parse_approval_rule(entry)
        except ValueError as exc:
            warnings.append(f"settings · dropped {label} entry: {exc}")
            continue
        kept.append(entry)
    return tuple(kept)


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
