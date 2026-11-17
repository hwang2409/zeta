# Native extensions

Native extensions are Python packages that run inside the Zeta process. They have
full access to the process, files, credentials, and session history. Review all
extension code before you install or update it.

Zeta verifies that the installed tree is the tree that you reviewed and trusted.
It does not sandbox extension code. Tampering by another process that runs as the
same Unix user is out of scope because that process can already edit Zeta state
and installed extensions.

## Locations and scopes

User extensions are installed in `$ZETA_HOME/extensions/<name>` (normally
`~/.zeta/extensions/<name>`). Project extensions are installed in
`<repository>/.zeta/extensions/<name>`.

Use `--scope user` or `--scope project`. User scope is the default. Project scope
requires a repository.

## Package and manifest

An extension source is a directory with a `zeta-extension.toml` manifest. A
minimal extension has these files:

```text
hello-extension/
├── hello.py
└── zeta-extension.toml
```

```toml
[extension]
name = "hello"
version = "0.1.0"
api_version = 1
entry = "hello:activate"
description = "A small example extension"
```

The manifest accepts only `name`, `version`, `api_version`, `entry`, and the
optional `description`. Names contain lowercase letters, digits, and hyphens,
and start with a letter or digit. `entry` is `module:function`; the function
defaults to `activate` when it is omitted.

The entry function is synchronous. Zeta calls it once for each session and gives
it the public `zx` interface. Activation records declarations; it must not run
the session itself.

```python
def activate(zx):
    zx.ui.notify("hello extension loaded")
```

Do not import Zeta internals. Use the `zx` object passed to `activate`.

## Install and trust

Install from a local directory:

```console
zeta ext install ./hello-extension
zeta ext install ./hello-extension --scope project
```

A Git source has the form `git+<repository>@<ref>`:

```console
zeta ext install 'git+https://example.com/hello.git@v0.1.0'
```

Zeta stages the source, validates its manifest and files, prints the reviewed
file list, and asks for approval. Install records trust for the exact installed
tree. `--yes` skips the prompt; use it only after another process has reviewed
the same source.

Use these commands to manage an installed extension:

```console
zeta ext update hello
zeta ext trust hello
zeta ext untrust hello
zeta ext remove hello
```

Add `--scope project` for a project extension. `update` works only for an
extension installed from a Git source. Update prints changed files, a diff
summary, and `CHANGELOG.md` when present before it asks for approval. `trust`
reviews the installed tree and trusts its exact digest. A changed or untrusted
tree does not load.

Install, update, trust, untrust, and remove change stored state. Restart all Zeta
processes that can use extensions together before any process uses the changed
store. This includes interactive sessions, background children, and `zeta serve`.
Do not mix processes that use different extension-store versions.

## Session management

Use `/ext` inside a session:

```text
/ext list
/ext info hello
/ext disable hello
/ext enable hello
```

`list` shows the version, scope, and state. `info` shows registered tools and
commands for a loaded extension. Enable and disable changes apply to the next
session. A setting in `settings.toml` can also keep an extension disabled; remove
that setting before `/ext enable` can enable it.

## Registration interface

During activation, `zx` exposes the extension `name`, `scope`, host
`api_version`, Zeta `home`, and current `project_dir` or `None`.

Use `zx.register_command(...)` to declare a slash command. Supply a nonempty
name and exactly one of:

- an async `handler` that receives the command argument string; or
- a string `body` with `kind="prompt"` or `kind="exec"`.

`description` is optional. A command that conflicts with an existing command is
ignored without breaking the session.

Use `zx.register_tool(...)` to declare a tool. Supply a nonempty name and a
callable handler. The public declaration also accepts `description`,
`parameters`, `parallel_safe`, `requires_approval`, `actions`, and `binding`.
`requires_approval` defaults to `True`. Tool declarations use the normal Zeta
tool registry validation and approval rules. Zeta commits declarations only
after `activate` returns without an error.

Use `zx.on(event, handler)` to register an async hook. Supported events are
`session_start`, `user_prompt_submit`, `pre_tool`, `post_tool`, `stop`, and
`tool_result`. The observation hooks do not delay the session. `pre_tool` is the
blocking event: Zeta waits for it before tool execution. The public interface
does not define a hook result that cancels or rewrites the tool call.

Use `zx.ui.notify(text, level="info")` for a transient session notice. `text`
must be nonempty. `level` is `info`, `warning`, or `error`.

## hook execution

extensions run in process with full python access. review an extension before you install it.

async hooks registered with `zx.on` run on one private event loop for each session. they do not run on the session event loop, and each invocation starts with a fresh empty `contextvars.Context`; ambient `ContextVar` values are not inherited. do not await or use `asyncio` objects that belong to the session loop. use the declared `zx.*` interfaces for host interaction.

`zx.ui.notify(...)` is safe to call from a hook. zeta marshals that call to the session loop, and the notice sink runs only on the session loop. `zx.run_blocking(...)` moves synchronous work off the hook's current loop and resumes the returned awaitable on that same hook loop; it does not provide session-loop affinity.

public `zx` and `zx.ui` facades keep only weak, clearable links to the session owner after activation. extension code still has full in-process power and can retain arbitrary objects through its own globals or closures; zeta cannot isolate or revoke those extension-owned references without moving extensions to another process.

## `zeta serve`

Serve clients can negotiate the `extensions` feature. It adds the read-only
`extensions_list` request and transient `extension_notice` events.
`extensions_list` reports each extension's name, scope, version, enabled,
trusted, loaded, and error state for the active session. `extension_notice`
carries the extension name, text, and `info`, `warning`, or `error` level. Clients
that do not negotiate the feature do not receive these events.

## Limitations

Extensions are trusted in-process code, not isolated plugins. Async hooks run on
a private loop, not the session loop. Do not capture, await, or operate on
session-owned `asyncio` objects. Use only the declared `zx` and `zx.ui`
interfaces for host interaction.
