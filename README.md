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

These evals invoke Zeta with `--yolo` and are **not a security sandbox**. Run
them only with a trusted task corpus and credentials you intend to use.
To test a change to the packaged identity, use a fresh `ZETA_HOME` with sign-in
configured: Zeta does not overwrite an existing user-edited `AGENTS.md`.

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
