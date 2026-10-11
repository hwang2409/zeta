# MCP server management

Zeta stores MCP definitions in `~/.zeta/mcp.json` (user scope) and
`<repository>/.zeta/mcp.json` (project scope). Project entries override user
entries with the same name. `ZETA_MCP_CONFIG` continues to select the user
configuration file.

## CLI

```console
zeta mcp add filesystem --scope project -- python -m filesystem_server
zeta mcp add linear --scope user --url https://mcp.example --oauth
zeta mcp list --scope effective --json
zeta mcp show filesystem --json
zeta mcp disable filesystem --scope project
zeta mcp enable filesystem --scope project
zeta mcp test filesystem
zeta mcp trust filesystem
zeta mcp untrust filesystem
zeta mcp login linear
zeta mcp login linear --no-browser
zeta mcp logout linear
zeta mcp remove filesystem --scope project
```

`list` and `show` are strictly offline. `test` initializes a temporary
connection and lists tools; it never invokes a tool, trusts a project server,
or changes its enabled state. Login and logout delegate to the existing OAuth
flow and token store. Logout removes credentials, not the server definition. Use
`--no-browser` to print the authorization URL and paste the full callback URL
from another device.

OAuth options are configured under `auth`. RFC discovery is used by default;
`authorization_server_url` and `resource_metadata_url` override its two
well-known lookups. Both must use `https`, except loopback `http`. `scopes` is
an explicit scope list, and `authorization_params` adds provider-neutral query
parameters such as `{"access_type": "offline"}`. It cannot override OAuth flow
parameters such as `state`, `redirect_uri`, `scope`, or PKCE fields:

```json
{
  "transport": "streamable-http",
  "url": "https://mcp.example",
  "auth": {
    "type": "oauth",
    "authorization_server_url": "https://login.example/oauth",
    "resource_metadata_url": "https://mcp.example/.well-known/resource",
    "scopes": ["openid", "offline_access"],
    "authorization_params": {"access_type": "offline"}
  }
}
```

## TUI

In the full-screen TUI, `/mcp` opens the interactive manager. `/mcp status`
keeps textual status output; `/mcp auth`, `/mcp reconnect`, and
`/mcp resources` remain available. Rows show scope, transport, authentication,
tool count, and one of these states:

- `●` connected
- `○` disabled
- `!` login needed or not yet connected
- `◌` pending trust
- `×` degraded/error

Use arrow keys (or `j`/`k`) to select, Enter for redacted details, `a` to add,
`e` to enable/disable, `t` to test, `l`/`o` to login/logout, `d` to remove, `T`
to trust a project command, and Escape to close. Secret values in add flows are
environment references, for example `API_KEY ← $LINEAR_API_KEY`; the manager
does not ask for a raw token.

Persisted TUI mutations reconcile the running MCP mount. Tools and provider
schemas are removed when a server is disabled or removed and republished after
a successful activation. A failed activation leaves its definition saved and
shows a degraded row. CLI changes made during a TUI session are loaded the next
time the manager opens (or when MCP is reconnected).

## Trust and credentials

Project HTTP definitions do not execute local programs. Project `stdio`
definitions are pending until explicitly trusted. Trust is stored in the local
user state and is keyed by both the checkout path and a fingerprint of command,
arguments, environment references, URL, headers, auth, and client settings.
Changing any of those fields invalidates trust. A cloned checkout therefore
cannot inherit permission to execute project commands.

Management output redacts literal environment values, headers, bearer tokens,
client secrets, and URL credentials/query values. Environment references remain
visible. Literal credentials remain readable in existing config files for
compatibility, but are never printed by list/show or the manager. There is no
automatic rewrite or credential migration.

Missing `enabled` retains the historical enabled behavior. Existing `${ENV}`
interpolation, project-over-user precedence, tool/prompt naming, and malformed
or missing-environment reporting remain compatible. Trusted enabled servers
start connecting in the background after the TUI renders; degraded servers keep
their persisted definitions.

## Per-server tool filters

A server definition can restrict the tools that Zeta registers. Patterns match
the server's exact, unqualified tool names and use simple shell-style globs.
For example, this read-only Google connector exposes search and read operations
but never mutation operations:

```json
{
  "servers": {
    "google": {
      "transport": "streamable-http",
      "url": "https://google.example/mcp",
      "allowed_tools": ["search_*", "get_*", "list_*"],
      "disallowed_tools": ["*_create", "*_update", "*_delete"]
    }
  }
}
```

When `allowed_tools` is present, only matching tools are available.
`disallowed_tools` always removes matching tools. The global `--tools` and
`--disallowed-tools` policy also applies, so a tool must pass both allow rules
and neither deny rule. Zeta applies the server filter again after a
`tools/list_changed` notification. Unknown patterns produce a warning. A server
whose filter matches no tools remains mounted with zero tools and reports a
notice. Filtered tools are not registered or discoverable, and direct calls to
them fail closed.
