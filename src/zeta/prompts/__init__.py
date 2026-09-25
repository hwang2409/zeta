"""Packaged prompt resources used by zeta."""

from __future__ import annotations

from importlib.resources import files

from ..skills import SkillCatalog


def load_identity(
    *,
    catalog: SkillCatalog,
    identity: str | None = None,
) -> str:
    """Load identity content with the skill index for one session."""

    content = load_packaged_identity() if identity is None else identity
    return f"{content.rstrip()}\n\n{catalog.index()}\n"


def load_packaged_identity() -> str:
    """Load the packaged identity without the session skill index."""

    return files(__package__).joinpath("identity.md").read_text(encoding="utf-8")


def load_skill(name: str, *, catalog: SkillCatalog) -> str:
    """Load one skill from a session catalog."""

    return catalog.load(name)


__all__ = ["load_identity", "load_packaged_identity", "load_skill"]
