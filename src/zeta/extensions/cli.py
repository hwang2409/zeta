"""Command-line handlers for extension management."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from ..core.project_context import discover_repo_root
from .store import ExtensionStore, ExtensionTrustError


def handle(args: object, *, home: Path) -> int:
    store = ExtensionStore(home=home, project_dir=discover_repo_root(os.getcwd()))
    try:
        action = args.ext_action  # type: ignore[attr-defined]
        scope = args.scope  # type: ignore[attr-defined]
        if action == "install":
            result = store.install(args.source, scope=scope, accept=args.yes)  # type: ignore[attr-defined]
            print(f"installed {result.name}" + (f" at {result.commit}" if result.commit else ""))
        elif action == "update":
            result = store.update(args.name, scope=scope, accept=args.yes)  # type: ignore[attr-defined]
            print(f"updated {result.name}" + (f" at {result.commit}" if result.commit else ""))
        elif action == "remove":
            store.remove(args.name, scope=scope)  # type: ignore[attr-defined]
        elif action == "trust":
            store.trust(args.name, scope=scope, accept=args.yes)  # type: ignore[attr-defined]
        else:
            store.untrust(args.name, scope=scope)  # type: ignore[attr-defined]
        return 0
    except (ExtensionTrustError, subprocess.CalledProcessError) as exc:
        print(f"extension error: {exc}", file=sys.stderr)
        return 2
