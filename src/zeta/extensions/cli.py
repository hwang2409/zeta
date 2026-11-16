"""command-line handlers for extension management."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from ..core.project_context import discover_repo_root
from .trust import ExtensionTrustError, install, remove, trust, untrust, update


def handle(args: object, *, home: Path) -> int:
    project_dir = discover_repo_root(os.getcwd())
    try:
        action = args.ext_action  # type: ignore[attr-defined]
        scope = args.scope  # type: ignore[attr-defined]
        if action == "install":
            name, commit = install(args.source, home=home, project_dir=project_dir, scope=scope, accept=args.yes)  # type: ignore[attr-defined]
            print(f"installed {name}" + (f" at {commit}" if commit else ""))
        elif action == "update":
            name, commit = update(args.name, home=home, project_dir=project_dir, scope=scope, accept=args.yes)  # type: ignore[attr-defined]
            print(f"updated {name}" + (f" at {commit}" if commit else ""))
        elif action == "remove":
            remove(home=home, project_dir=project_dir, scope=scope, name=args.name)  # type: ignore[attr-defined]
        elif action == "trust":
            trust(home=home, project_dir=project_dir, scope=scope, name=args.name, accept=args.yes)  # type: ignore[attr-defined]
        else:
            untrust(home=home, project_dir=project_dir, scope=scope, name=args.name)  # type: ignore[attr-defined]
        return 0
    except (ExtensionTrustError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"extension error: {exc}", file=sys.stderr)
        return 2
