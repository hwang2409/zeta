"""Load the flat set of project instruction files for a zeta session."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
import subprocess

from ..prompts import load_identity


@dataclass(frozen=True, slots=True)
class ProjectContext:
    """The static prompt additions and files loaded for one session."""

    system_prompt: str
    files: tuple[Path, ...]


def discover_repo_root(cwd: str | Path | None = None) -> Path:
    """Resolve the git worktree root, or use cwd when it is not a repository."""

    directory = Path(cwd or Path.cwd()).expanduser().resolve()
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return directory
    root = result.stdout.strip()
    return Path(root).expanduser().resolve() if root else directory


def _present(path: Path) -> bool:
    try:
        path.stat()
    except FileNotFoundError:
        return False
    return True


def load_project_context(
    *,
    repo_root: str | Path | None = None,
    zeta_home: str | Path | None = None,
) -> ProjectContext:
    """Load zeta and repository instructions without walking directories."""

    root = Path(repo_root or Path.cwd()).expanduser().resolve()
    home = Path(zeta_home or Path.home() / ".zeta").expanduser().resolve()
    candidates = [home / "AGENTS.md"]
    repo_agents = root / "AGENTS.md"
    candidates.append(repo_agents if _present(repo_agents) else root / "CLAUDE.md")

    loaded: list[Path] = []
    sections = [load_identity()]
    for path in candidates:
        if not _present(path):
            continue
        content = path.read_text(encoding="utf-8")
        loaded.append(path)
        source = escape(str(path), quote=True)
        sections.append(
            f'<zeta-project-instructions source="{source}">\n'
            f"{escape(content, quote=True)}\n"
            "</zeta-project-instructions>"
        )

    return ProjectContext("\n\n".join(sections), tuple(loaded))


__all__ = ["ProjectContext", "discover_repo_root", "load_project_context"]
