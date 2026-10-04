# zeta

Custom agent harness. Owns the full agent loop — conversation state, context
assembly and compaction, tool dispatch, approval policy, streaming, session
persistence — and talks to Claude and Codex through direct, plan-authenticated
provider APIs (pi-style). No claude/codex CLI or app-server subprocesses.

Experimental. Isolated from the Wiki app; Wiki may consume it later as a
dependency behind a flag.

## composer attachments

Use `@"a b.txt"` for a quoted path, or `@path/to/file` for a path-like
reference. This includes `@./file`, `@../file`, and `@~/file`. Bare words such
as `@user` and `@dataclass` stay as prompt text. Missing path-like files show
a notice and block that message.

Use `/paste` or Ctrl+V to insert an image token such as `[Image #1]` at the
cursor. The token stays in the message text and resolves to the staged image
when you send it. Delete the token to cancel that image. Numbering starts at
`[Image #1]` for each message and does not renumber after edits. Attachment
paths follow the read-tool policy: there is no attachment-specific sandbox;
absolute paths, home paths, and symlink targets are allowed. Repeated
references to one resolved path produce one attachment block.

## Layout

- `src/zeta/` — the package
- `tests/` — pytest suite (`uv run pytest -q`)
- `docs/design.md` — architecture and ticket ladder

## edit tool

`edit` accepts one `old_string`/`new_string` pair or an `edits` array for
several replacements in the same file. Each old string must match once in the
original file. Batch replacements cannot overlap. Zeta checks all matches
before it writes the file.

## cache tracing

Set `ZETA_CACHE_TRACE=1` before starting zeta to write bounded, private
`$ZETA_HOME/logs/cache-trace.jsonl` (default `~/.zeta/logs/cache-trace.jsonl`).
Each agent-loop completion, including child agents, records its model, cache
token counts, tool stability, and how many earlier messages match its last
completed request. Compaction-summary calls are excluded. No prompt text or
tool arguments are recorded; unset the variable to disable it. The file is
mode `0600` and rotates at 1 MiB.

## workflow evals

Run `uv run python evals/run.py` to send five file-edit and coding tasks through
the real Zeta loop with the signed-in Codex provider. Each task gets a fresh
temporary working directory; checks grade the files and commands, not the
agent's claim. Results report artifact success and agent completion separately.
Use `--task ID`, `--repeat N`, or `--instruction TEXT` for focused A/B runs.
`--keep-failures DIR` copies failed workspaces for inspection.
After installing Chromium, run the optional public-page browser suite with
`ZETA_BROWSER=1 uv run --extra browser python evals/run.py --tasks evals/browser_tasks.jsonl`.
It grades the final browser tool result, not the agent's final claim; public
pages can change, so this suite is run manually rather than in CI.

These evals invoke Zeta with `--yolo` and are **not a security sandbox**. Run
them only with a trusted task corpus and credentials you intend to use.
To test a change to the packaged identity, use a fresh `ZETA_HOME` with sign-in
configured: Zeta does not overwrite an existing user-edited `AGENTS.md`.

Custom agents in `~/.zeta/agents/` or `.zeta/agents/` can set
`allow_delegation: false` in YAML frontmatter to hide the `agent` tool from
that child without restricting its other tools. The default is `true`.

## experimental browser tool

Install the optional headless browser with
`uv run --extra browser playwright install --only-shell chromium`, then start
Zeta with `ZETA_BROWSER=1 uv run --extra browser zeta`. This adds one `browser`
tool for opening HTTP(S) pages, reading an accessibility snapshot, and
clicking, filling, pressing, or selecting by role/name (plus zero-based index
when names repeat). A bounded `batch` action groups known sequences into one
call. It uses one isolated, non-persistent tab per agent session
and follows Zeta's normal tool approval policy. It is not a security sandbox; use `--yolo`
only on sites and tasks you trust. Popups and downloads are not supported.

## full-screen transcript keys

- `pageup` and `pagedown` scroll the transcript.
- `ctrl+x ctrl+o` expands or collapses the newest child card.
- `ctrl+o` keeps its native prompt-toolkit behavior in the composer.

## selecting and copying text

The transcript lives on the alternate screen with mouse reporting on, so the
terminal never sees a drag as a selection. Drag over the transcript instead:
the covered text highlights as you go, and releasing the button copies it to
the system clipboard (`pbcopy`, `wl-copy`, `xclip`, or `xsel`) and to the
composer's own clipboard. The status bar reports `copied N lines`. The
selection is pinned to the text it covers, so it stays put while a reply is
still streaming in; you can select and copy mid-reply. Each streamed token
repaints only the message it landed in rather than re-parsing the whole
transcript, so long sessions stay responsive under the pointer. A plain click
clears the highlight and the wheel still scrolls. Provider errors now wrap
their full reason instead of cutting it off at the card edge.

## picking a model

`/model` opens a picker card listing the models the current provider serves:
the built-in table plus the live catalog once it loads, with the current
model marked. `↑`/`↓` move, `enter` selects, and `esc` cancels; the keys only
act while the composer is empty. `/model <text>` switches directly when the
text names a known model or matches exactly one, and otherwise narrows the
picker to the matches, so `/model opus` shows the opus family instead of
sending `opus` to the provider. Text that matches nothing is still sent as
typed, so a brand-new id keeps working. Typing after `/model ` also completes
model names inline.

Typing `/` opens the command menu above the composer in the theme's colors,
with the highlighted row on the accent and up to twelve entries visible.
Arrow keys move, `tab` or `enter` accepts, and a mouse click picks an entry.

## invoking skills

Loaded skills can be invoked at the start of a message as either `/skill-name`
or `$skill-name`. A `$skill-name` mention can also appear later in a normal
message; Zeta loads each distinct mentioned skill in mention order and gives
the skill the full message as its request. Typing `$` at a token boundary
opens a skills-only completion menu, including in the middle of a message.
Dollar expressions in `!` shell mode and inside backticks or fenced code stay
literal, as do names that do not exactly match a loaded skill.

## tool availability

Use `--tools` to give a session an allowlist of exact tool names or shell-style
globs. MCP tools use `server__tool` names. Use `--disallowed-tools` for a
denylist; the denylist wins when both lists match.

```sh
zeta --tools 'computer__*' --disallowed-tools 'computer__shutdown' -p 'inspect the page'
zeta serve --tools 'computer__*' --require-tools
```

A tool that does not pass this policy is not included in provider request
schemas, and a stale or hallucinated call is rejected before execution. This
includes built-in file, shell, agent, background, and task tools. If an MCP
server fails to start, Zeta continues with only matching tools that did start;
it does not restore built-ins. `--require-tools` makes print mode and `serve`
session startup fail when any exact name in `--tools` is unavailable after MCP
startup. Glob patterns do not create a startup requirement because they can
intentionally match zero or many tools.

The same policy can be set globally in `~/.zeta/settings.toml` or per project
in `.zeta/settings.toml`:

```toml
tools = ["computer__*"]
disallowed_tools = ["computer__shutdown"]
```

Project policy is monotonic. A project `tools` list is an additional allowlist:
a tool must match both the global and project lists. A missing list adds no
restriction, while `tools = []` allows no tools. Project denylists are added to
the global denylist; a project cannot remove a global denial. Zeta prints a
startup notice when a project list appears to widen global policy. Invalid
`tools` or `disallowed_tools` values in either settings file stop startup with
an error instead of silently leaving tools unrestricted. TOML syntax and UTF-8
errors in any settings file also stop startup because that file can contain
security policy.

CLI values are trusted invocation policy. `--tools` replaces all configured
allowlist layers, including global restrictions, and `--disallowed-tools`
replaces the complete configured denylist. The effective policy is stored in
session metadata. On resume, the persisted policy and the current invocation
policy both apply: allowlists intersect and
denylists combine. The narrowed result is persisted, so a later resume can
never widen the session. Missing and empty allowlists remain distinct. Child
agents inherit the parent policy and can only remove more tools through their
agent tool list. `/tools` shows cumulative allowlist layers with `AND` when
more than one layer applies. `--require-tools` checks this effective policy.

Tool availability is separate from approval policy. An advertised tool can
still require approval, while an unavailable tool cannot be advertised or
executed regardless of approval settings.

Command hooks from `~/.zeta/hooks.toml` execute host shell commands. They remain
enabled for unrestricted sessions. When any tool allowlist or denylist is
active, Zeta disables command hooks by default. A trusted operator can opt in
with `--allow-hooks`, including `zeta serve --allow-hooks`, or set
`allow_hooks = true` in the global `~/.zeta/settings.toml`. Project settings
cannot enable hooks.

Other host-execution paths follow these rules:

- TUI `!`/`!!`, custom slash `exec` commands, and command inline-shell spans
  execute through the `bash` tool registry path. Tool policy therefore blocks
  them when `bash` is unavailable. Skill and custom-agent definitions are
  Markdown, not imported code; execution they request uses registered tools.
- Python modules in `~/.zeta/tools` are not imported when any allowlist or
  denylist is active. A trusted operator can opt in globally with
  `allow_external_tools = true` in `~/.zeta/settings.toml`; project settings
  cannot enable it. Project `.zeta/tools` modules remain unimported until the
  user explicitly runs `/tools trust`. Zeta has no other plugin loader.
- Automation sessions do not load command hooks. Their approved tool and MCP
  service allowlists continue to govern execution.
- Interactive and headless sessions, including sessions created by
  `zeta serve`, use the same hook gate. `--allow-hooks` is the trusted CLI
  override.
- Project MCP stdio servers are host processes, but they never launch until the
  user has trusted the exact project server definition with `zeta mcp trust`.
  A restricted session can still start trusted configured MCP servers even if
  none of their tools match the session policy; nonmatching tools are filtered
  from registration and cannot execute.

## context compaction

New sessions use deterministic history eviction by default. To use the prior
model-written behavior, pass `--compaction summary` or set it globally or for
a project:

```toml
compaction = "summary"
```

`evict` first replaces old, re-derivable read, shell, and search results with
small deterministic digests. The original structured messages remain in the
append-only session log, and the `recall_history` tool can retrieve their exact
contents from the active branch. Sequence-range recall paginates deterministic
rendered text by character offset, including individual messages larger than a
page. If the deterministic eviction view cannot fit the full token budget,
Zeta uses normal summary compaction. `recall_history` is available whenever the
effective mode is `evict`, including in child agents and unattended sessions.

The selected mode is stored when a session is created. Resume keeps that mode,
even if settings later change. Sessions created before mode persistence have
no stored value and resume in `summary` mode so their behavior does not change
mid-session.

Use the default `evict` mode for long coding or investigation sessions with
large repeated tool outputs. Use `summary` when the conversational narrative
is more important than exact tool-output recall.

## tui themes

Use `/theme list` to see the built-in `dark`, `light`, and `gruvbox-dark`
palettes. `/theme gruvbox-dark` switches the current session. Set
`theme = "gruvbox-dark"` in `~/.zeta/settings.toml` to use it at startup.
Custom palettes live in `~/.zeta/themes/<name>.toml`. They can override a
built-in name and use the keys in [the Gruvbox example](docs/themes/gruvbox-dark.toml).
The `surface` and `tint` keys fill tool and user messages. `read_bg`,
`shell_bg`, `edit_bg`, and `agent_bg` set tool-specific surfaces. Missing keys
use the built-in palette of the same name, or `dark` for a new name.

The `bash` tool accepts timeouts from above zero through 3600 seconds. A
command that creates its own session, such as with `setsid`, can outlive the
timeout because it leaves the process group that zeta kills.

## exec macro example

Create `~/.zeta/commands/rebuild.md` for a local `/rebuild` macro:

```markdown
---
kind: exec
description: rebuild and relaunch the project
timeout: 300
---
git pull --ff-only && make build && ./scripts/relaunch.sh
```

This is an example only. zeta does not install a default macro.

Exec macro arguments use shell positional parameters. `$1` through `$9` and
`"$@"` receive the whitespace-split arguments. `$ARGUMENTS` receives the raw
argument tail. Arguments are passed as argv values, so quotes, shell
metacharacters, dollar signs, and newlines are not evaluated as shell code.
Use `${10}` for the tenth argument; `$10` means `$1` followed by `0` in POSIX
shells. The timeout defaults to 300 seconds and can be set in frontmatter.

## Prior art

- [pi](https://github.com/earendil-works/pi) (MIT) — reference for the
  provider seam, loop shape, session tree, and compaction split. Ideas are
  copied with citations; code is not vendored without a provenance note.
- Wiki's WIKI-361 design (`wiki/vault/wiki-app/wk-owned-agent-loop-design.md`)
  — the earlier in-app plan this project supersedes.
