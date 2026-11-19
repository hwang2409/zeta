"""Serve resource families, each with a local feature flag and method list."""

from . import agent_catalog, home_settings

FAMILIES = (
    home_settings,
    agent_catalog,
)
