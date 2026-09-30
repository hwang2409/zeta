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
zeta mcp logout linear
zeta mcp remove filesystem --scope project
```

`list` and `show` are strictly offline. `test` initializes a temporary
connection and lists tools; it never invokes a tool, trusts a project server,
or changes its enabled state. Login and logout delegate to the existing OAuth
flow and token store. Logout removes credentials, not the server definition.

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
