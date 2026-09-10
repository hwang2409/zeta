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

## full-screen transcript keys

- `pageup` and `pagedown` scroll the transcript.
- `ctrl+x ctrl+o` expands or collapses the newest child card.
- `ctrl+o` keeps its native prompt-toolkit behavior in the composer.

## selecting and copying text

The transcript lives on the alternate screen with mouse reporting on, so the
terminal never sees a drag as a selection. Drag over the transcript instead:
the covered text highlights as you go, and releasing the button copies it to
the system clipboard (`pbcopy`, `wl-copy`, `xclip`, or `xsel`) and to the
composer's own clipboard. The status bar reports `copied N lines`. A plain
click clears the highlight, the wheel still scrolls, and any transcript
change drops the selection so highlights never drift. Provider errors now
wrap their full reason instead of cutting it off at the card edge.

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
