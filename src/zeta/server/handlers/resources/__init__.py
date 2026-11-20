"""Serve resource families, each with a local feature flag and method list."""

from . import account_list, agent_catalog, child_control, home_settings, mcp_management

FAMILIES = (
    home_settings,
    agent_catalog,
    child_control,
    mcp_management,
    account_list,
)
