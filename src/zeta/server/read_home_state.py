"""Allowlisted home state for serve; all reads run in the bounded read pool."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from ..accounts.store import AccountStore
from ..config import home_settings
from ..core.session import account_home
from ..core.session_files import child_directory
from ..mcp.config import home_config_path, load_mcp_config
from ..read_page import InvalidParams, Key, TooLarge, page_result, resume
from ..telemetry.receiver.paths import used_bytes
from ..telemetry.receiver.store import MAX_BYTES, RAW_DAYS, ROLLUP_DAYS
from .protocol import MethodDescriptor
from .read_catalog import validated_state_read

_ORDERS = {
    "mcp": (Key("name", str), Key("transport", str)),
    "redacted_keys": (Key("name", str),),
    "accounts": (Key("provider", str), Key("account_id", str)),
    "warnings": (Key("text", str), Key("ordinal", int)),
}
STORAGE_ENTRY_LIMIT = 4096  # shared across the three trees; at most one stat per entry
_STORAGE_DEPTH = 16


def _generation(paths):
    result = []
    for path in paths:
        try:
            info = path.lstat()
        except FileNotFoundError:
            result.append(None)
        else:
            result.append((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns))
    return result


def _settings(home, collection):
    settings = home_settings.read_settings(home)
    telemetry = home_settings.telemetry_projection(settings)
    summary = {
        "model": settings.model,
        "provider": settings.provider,
        "approval_mode": None if settings.yolo is None else "allow" if settings.yolo else "ask",
        "max_depth": settings.agents_max_depth,
        "otel": {
            "enabled": telemetry["enabled"],
            "endpoint": {"off": None, "builtin": "builtin", "custom": "external"}[telemetry["source"]],
        },
    }
    if collection == "redacted_keys":
        return summary, [((name,), name) for name in settings.redacted_keys]
    config = load_mcp_config(home_config_path(home))
    rows = []
    for name, server in config.configured_servers.items():
        row = {"name": name, "transport": server.transport, "configured": name in config.servers and server.enabled}
        rows.append(((name, server.transport), row))
    return summary, rows


def _warning_rows(warnings):
    return [((text, ordinal), text) for ordinal, text in enumerate(sorted(warnings))]


def _storage(home: Path, accounts: Path, epoch: int):
    warnings = []
    remaining = STORAGE_ENTRY_LIMIT

    def scan(fd, depth, count_depth):
        nonlocal remaining
        count = size = 0
        with os.scandir(fd) as entries:
            for entry in entries:
                if not remaining:
                    return count, size, False
                remaining -= 1
                try:
                    info = entry.stat(follow_symlinks=False)
                    if stat.S_ISREG(info.st_mode):
                        size += info.st_size
                    elif stat.S_ISDIR(info.st_mode) and not entry.name.startswith("."):
                        count += int(depth == count_depth)
                        if depth >= _STORAGE_DEPTH:
                            return count, size, False
                        with child_directory(fd, entry.name) as child:
                            nested_count, nested_size, complete = scan(child, depth + 1, count_depth)
                        count += nested_count
                        size += nested_size
                        if not complete:
                            return count, size, False
                except FileNotFoundError:
                    return count, size, False
        return count, size, True

    result = {"writer_epoch": epoch, "telemetry": {"bytes": used_bytes(home)}}
    for area, base, count_depth in (("sessions", home, 1), ("projects", home, 1), ("accounts", accounts, 2)):
        try:
            root = os.open(base, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                with child_directory(root, area) as fd:
                    count, size, complete = scan(fd, 1, count_depth)
            finally:
                os.close(root)
        except FileNotFoundError:
            count, size, complete = 0, 0, True
        if not complete:
            warnings.append(f"{area}: partial storage scan")
        result[area] = {"count": count, "bytes": size}
    return result, _warning_rows(warnings)


def _page(method, params, fits, snapshot, *, collections=()):
    allowed = {"limit", "cursor"}
    collection = "warnings"
    if collections:
        allowed.add("collection")
        collection = params.get("collection")
        if not isinstance(collection, str) or collection not in collections:
            raise InvalidParams("invalid collection")
    limit = params.get("limit", 200)
    if params.keys() - allowed or type(limit) is not int or not 1 <= limit <= 200:
        raise InvalidParams("invalid page params")
    start = resume(params.get("cursor"), (method.wire, collection), (), _ORDERS[collection])
    summary, rows = snapshot(collection)
    items_key = "warnings"
    if collections:
        summary["collection"] = collection
        items_key = "items"
    rows.sort(key=lambda row: row[0])
    after = [(key, item) for key, item in rows if start.position is None or key > start.position]
    return page_result(
        [item for _, item in after[:limit]], lambda index: after[index][0], start,
        more=len(after) > limit, total=None, fits=fits,
        shape=lambda items, cursor: {**summary, items_key: items, "next_cursor": cursor},
    )


def _read(home, kind, paths, snapshot):
    return validated_state_read(home, kind, lambda: _generation((home / "storage-format.json", *paths)), snapshot)


def settings_page(home, params, fits):
    def snapshot(collection):
        paths = [home / "settings.toml"]
        if collection == "mcp":
            paths.append(home_config_path(home))
        return _read(home, "settings", paths, lambda _: _settings(home, collection))

    return _page(MethodDescriptor.STATE_SETTINGS, params, fits, snapshot, collections=("mcp", "redacted_keys"))


def accounts_page(home, params, fits):
    def snapshot(collection):
        store = AccountStore(account_home(home))
        accounts, warnings = _read(home, "accounts", (store.registry, store.accounts), lambda _: store.state_rows())
        rows = [((row["provider"], row["account_id"]), row) for row in accounts] if collection == "accounts" else _warning_rows(warnings)
        return {}, rows

    return _page(MethodDescriptor.STATE_ACCOUNTS, params, fits, snapshot, collections=("accounts", "warnings"))


def storage_page(home, params, fits):
    def snapshot(_collection):
        accounts = account_home(home)
        paths = (home / "sessions", home / "projects", accounts / "accounts")
        return _read(home, "storage", paths, lambda epoch: _storage(home, accounts, epoch))

    return _page(MethodDescriptor.STATE_STORAGE, params, fits, snapshot)


def retention_result(home, params, fits):
    if params:
        raise InvalidParams("invalid retention params")
    settings = _read(home, "settings", (home / "settings.toml",), lambda _: home_settings.read_settings(home))
    result = {
        "session_days": settings.retention_days, "telemetry_raw_days": RAW_DAYS,
        "telemetry_rollup_days": ROLLUP_DAYS, "telemetry_max_bytes": MAX_BYTES,
        # The receiver does not persist a prune timestamp.
        "last_telemetry_prune_ns": None,
    }
    if not fits(result):
        raise TooLarge("retention does not fit the frame")
    return result
