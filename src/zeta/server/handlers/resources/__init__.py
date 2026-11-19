"""Serve resource families, each with a local feature flag and method list."""

from . import child_control, home_settings

FAMILIES = (
    home_settings,
    child_control,
)
