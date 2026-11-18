"""Reset and cleanup without touching personal files or credentials."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .run import prepare_home


def reset(run_dir: Path, verify: bool) -> dict:
    with prepare_home(run_dir):
        fixture = run_dir / "fixture"
        removed = []
        if fixture.exists():
            for path in fixture.glob("trial-*.json"):
                path.unlink()
                removed.append(str(path))
        home = run_dir / "zeta-home"
        if home.exists():
            shutil.rmtree(home)
            removed.append(str(home))
        result = {"pass": not home.exists(), "removed": removed, "apps_quit": [], "files_deleted": []}
        if verify and not result["pass"]:
            raise SystemExit("reset verification failed")
        (run_dir / "cleanup.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return result

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    reset(args.run_dir, args.verify)


if __name__ == "__main__":
    main()
