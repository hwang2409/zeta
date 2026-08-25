"""Load the flat set of project instruction files for a zeta session."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..prompts import load_identity


@dataclass(frozen=True, slots=True)
class ProjectContext:
    """The static prompt additions and files loaded for one session."""

    system_prompt: str
    files: tuple[Path, ...]


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
        sections.append(f"Instructions from {path}:\n{content}")

    return ProjectContext("\n\n".join(sections), tuple(loaded))


__all__ = ["ProjectContext", "load_project_context"]
