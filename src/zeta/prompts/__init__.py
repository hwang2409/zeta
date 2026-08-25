"""Packaged prompt resources used by zeta."""

from __future__ import annotations

from functools import cache
from importlib.resources import files


@cache
def load_identity() -> str:
    """Load the static zeta identity once for the process."""

    return files(__package__).joinpath("identity.md").read_text(encoding="utf-8")


__all__ = ["load_identity"]
