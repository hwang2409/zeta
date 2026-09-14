"""Packaged prompt resources used by zeta."""

from __future__ import annotations

import os
from importlib.resources import files
from pathlib import Path

from ..skill_catalog import SkillCatalog, discover_session_skills


def load_identity(
    *,
    home: str | Path | None = None,
    project_dir: str | Path | None = None,
    catalog: SkillCatalog | None = None,
) -> str:
    """Load static identity with the skill index for one session."""

    identity = files(__package__).joinpath("identity.md").read_text(encoding="utf-8")
    session_catalog = catalog or discover_session_skills(
        home=home or os.environ.get("ZETA_HOME"),
        project_dir=project_dir or Path.cwd(),
    )
    return f"{identity.rstrip()}\n\n{session_catalog.index()}\n"


def load_skill(name: str, *, catalog: SkillCatalog | None = None) -> str:
    """Load one skill from a session catalog."""

    session_catalog = catalog or discover_session_skills(
        home=os.environ.get("ZETA_HOME"), project_dir=Path.cwd()
    )
    return session_catalog.load(name)


__all__ = ["load_identity", "load_skill"]
