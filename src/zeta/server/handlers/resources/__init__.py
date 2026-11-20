"""Serve resource families, each with a local feature flag and method list."""

from . import (
    account_list,
    agent_catalog,
    child_control,
    home_settings,
    library,
    library_writes,
    mcp_management,
    session_read,
)

FAMILIES = (
    home_settings,
    agent_catalog,
    child_control,
    mcp_management,
    account_list,
    library,
    library_writes,
    session_read,
)
