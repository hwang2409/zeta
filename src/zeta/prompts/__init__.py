"""Packaged prompt resources used by zeta."""

from __future__ import annotations

from importlib.resources import files

from ..skills import SkillCatalog


def load_identity(
    *,
    catalog: SkillCatalog,
) -> str:
    """Load static identity with the skill index for one session."""

    identity = files(__package__).joinpath("identity.md").read_text(encoding="utf-8")
    return f"{identity.rstrip()}\n\n{catalog.index()}\n"


def load_skill(name: str, *, catalog: SkillCatalog) -> str:
    """Load one skill from a session catalog."""

    return catalog.load(name)


__all__ = ["load_identity", "load_skill"]
