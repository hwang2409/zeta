"""Packaged prompt resources used by zeta."""

from __future__ import annotations

from functools import cache
from importlib.resources import files

from ..skills import discover_packaged_skills


@cache
def load_identity() -> str:
    """Load the static identity and skill index once for the process."""

    identity = files(__package__).joinpath("identity.md").read_text(encoding="utf-8")
    return f"{identity.rstrip()}\n\n{discover_packaged_skills().index()}\n"


def load_skill(name: str) -> str:
    """Load one skill from the process-static packaged catalog."""

    return discover_packaged_skills().load(name)


__all__ = ["load_identity", "load_skill"]
