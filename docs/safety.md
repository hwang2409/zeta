# Safety and tool policy

Zeta's file and shell tools run on the host as the current user. Zeta is not a
security sandbox.

## Tool availability

Use `--tools` to allow only matching built-in or MCP tool names. Action tools
also accept `name(action)` selectors; for example, `agent(status)` permits only
that action, while bare `agent` permits every action. Repeat selectors to permit
more than one action. Use `--disallowed-tools` to deny matching capabilities;
the denylist wins. Name patterns use exact names or shell-style globs, and MCP
names use the `server__tool` form:

```sh
zeta --tools 'remote__*' --disallowed-tools 'remote__delete' -p 'inspect the service'
```

A tool that is not allowed is not advertised to the provider and cannot run.
`--require-tools` makes a headless run fail if an exact allowlisted tool is not
available. Tool policy is separate from approval policy: an allowed tool can
still require approval.

Settings can define `tools` and `disallowed_tools` in `~/.zeta/settings.toml`
and in a project `.zeta/settings.toml`. Project allowlists are cumulative with
the global allowlist, and project denylists add to the global denylist. A
project cannot widen the global approval or execution settings. CLI policy
flags are trusted invocation overrides. On resume, the saved and current
policies narrow together; a resume cannot widen a session's policy.

## Approvals

Approval rules are configured in the global settings file under `[approval]`,
with `allow`, `deny`, and `ask` lists. Rules can match a tool name, one action,
or an action-specific subject: `task`, `task(output)`, and
`task(start pytest*)`. For tools without actions, existing subject syntax such
as `bash(git status*)` is unchanged. The default interactive behavior asks before
tools that need approval. `--yolo` sets the default approval decision to allow;
explicit approval rules still apply. `--no-yolo` sets the default to ask; it does
not force every call to prompt. In headless mode, calls that need approval are
denied unless `--yolo` is set.

In the terminal UI, `/permissions` shows the active default, deny/ask/allow
rules, and grants from `/always`. Rules take precedence in that order:
deny > ask > allow > default. The view reads the running policy, including CLI
overrides, without changing permissions. It also shows tool availability
restrictions separately.

Use `/permissions revoke bash` or `/permissions revoke inbox(send)` to remove
the exact session grant shown in the view. `/permissions clear` removes all
session grants. These commands preserve configured rules (including identical
allow rules), deny/ask rules, tool restrictions, and settings files. Future
calls use the remaining policy; revoking a grant does not add a deny rule, so
another allow rule or a default of `allow` can still permit the call. Pending
approvals and already authorized calls are unchanged. Bare `/permissions`
remains read-only.

`/approve [key]` allows one pending request; `/deny [key]` denies it.
`/always [key]` allows the whole tool or the request's resolved action for the
running session, with **any arguments**. For example, a grant for `bash` covers
any command, while `inbox(send)` covers any recipient project but no other inbox
action. Explicit deny/ask rules still win. These grants reset when the session
is reopened and do not edit settings. For persistent, narrower grants, edit
`[approval]` in `~/.zeta/settings.toml` (or `$ZETA_HOME/settings.toml`):

```toml
[approval]
allow = ["bash(git status*)"]
ask = ["bash(git push*)"]
deny = ["bash(rm *)"]
```

Project settings cannot set approval rules. Start a new Zeta process to load
changes to the global settings file.

Availability and approval are independent. A tool must first pass the tool
policy; approval cannot enable a tool that is unavailable. Conversely, an
advertised tool can still stop for approval. Child agents inherit the parent's
policy and can only remove tools.

## Host execution and extensions

Command hooks in `~/.zeta/hooks.toml` execute host shell commands. They are
enabled for unrestricted sessions, but a tool allowlist or denylist disables
them unless the trusted operator passes `--allow-hooks` or sets global
`allow_hooks = true`. Project settings cannot enable hooks.

TUI `!` commands, custom `exec` slash commands, and inline shell spans use the
registered `bash` tool, so tool policy applies to them. Python modules in
`~/.zeta/tools` are not loaded in restricted sessions unless the operator sets
global `allow_external_tools = true`; project modules require explicit `/tools
trust`. Skill and agent files are Markdown, not executable plugins.

Project MCP stdio servers are host processes. They do not start until the user
trusts the exact project definition with `zeta mcp trust`; changing its command,
arguments, or related settings invalidates that trust. MCP HTTP definitions do
not execute local programs. Automation sessions do not load command hooks and
use their approved tool and MCP allowlists.
