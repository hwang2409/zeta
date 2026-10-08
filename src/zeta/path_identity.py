"""Shared physical path identity rules."""

from __future__ import annotations

import os
from os import PathLike


def same_physical_path(left: str | PathLike[str], right: str | PathLike[str]) -> bool:
    """Return whether two paths identify the same filesystem object.

    Existing paths use filesystem identity, including symlink resolution. If
    either path does not exist, canonical path text provides stable comparison
    for callers that must classify a not-yet-created location.
    """

    try:
        return os.path.samefile(left, right)
    except (FileNotFoundError, NotADirectoryError):
        return os.path.realpath(left) == os.path.realpath(right)


__all__ = ["same_physical_path"]
