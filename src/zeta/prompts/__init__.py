"""Packaged prompt resources used by zeta."""

from __future__ import annotations

from importlib.resources import files

from ..skill_catalog import SkillCatalog


def load_identity(
    *,
    catalog: SkillCatalog | None = None,
) -> str:
    """Load static identity with the skill index for one session."""

    identity = files(__package__).joinpath("identity.md").read_text(encoding="utf-8")
    session_catalog = catalog if catalog is not None else SkillCatalog(())
    return f"{identity.rstrip()}\n\n{session_catalog.index()}\n"


def load_skill(name: str, *, catalog: SkillCatalog | None = None) -> str:
    """Load one skill from a session catalog."""

    session_catalog = catalog if catalog is not None else SkillCatalog(())
    return session_catalog.load(name)


__all__ = ["load_identity", "load_skill"]
