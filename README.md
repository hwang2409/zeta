# harness

Custom agent harness. Owns the full agent loop — conversation state, context
assembly and compaction, tool dispatch, approval policy, streaming, session
persistence — and talks to Claude and Codex through direct, plan-authenticated
provider APIs (pi-style). No claude/codex CLI or app-server subprocesses.

Experimental. Isolated from the Wiki app; Wiki may consume it later as a
dependency behind a flag.

## Layout

- `src/harness/` — the package
- `tests/` — pytest suite (`uv run pytest -q`)
- `docs/design.md` — architecture and ticket ladder

## Prior art

- [pi](https://github.com/earendil-works/pi) (MIT) — reference for the
  provider seam, loop shape, session tree, and compaction split. Ideas are
  copied with citations; code is not vendored without a provenance note.
- Wiki's WIKI-361 design (`wiki/vault/wiki-app/wk-owned-agent-loop-design.md`)
  — the earlier in-app plan this project supersedes.
