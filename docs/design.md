# zeta design

Re-anchor of the WIKI-361 plan (2026-08-19, Henry decision): the harness is a
standalone project, not a Wiki backend retrofit. It builds its own agent loop
end to end, pi-style, and does NOT spawn claude/codex subprocesses — the
earlier plan's CLI stream-json and app-server seams are dropped along with the
raw-API ban that motivated them. Full context: Wiki vault note
`wk-owned-agent-loop-design` (module map, pi citations, risks); this doc is
the harness-native distillation.

## Provider seam (pi-style, plan-authenticated)

- **Claude**: direct Anthropic Messages API. Auth via Claude Pro/Max
  subscription OAuth (refresh + access tokens), never a raw API key. pi
  reference: `packages/ai/src/providers/anthropic.ts`,
  `packages/ai/src/auth/oauth/anthropic.ts`,
  `packages/ai/src/api/anthropic-messages.ts`.
- **Codex**: direct ChatGPT backend-api (Responses-style, SSE/WebSocket).
  Auth via ChatGPT plan OAuth with account id derived from the access token.
  pi reference: `packages/ai/src/providers/openai-codex.ts`,
  `packages/ai/src/auth/oauth/openai-codex.ts`,
  `packages/ai/src/api/openai-codex-responses.ts`.
- One provider-neutral stream contract: assistant deltas, thinking, tool-call
  deltas, usage. The loop is blind to provider wire formats.
- Keep provider-native prompt-cache identity where the API allows it (we own
  the context, so cache reuse is ours to manage — an advantage the
  subprocess-seam plan had to give up).

## Loop ownership (what the harness owns)

- `AgentLoop`: turns, steering, follow-ups, tool execution, stop conditions.
  One completion per provider call; tool calls stop at the harness boundary.
- `ConversationStore`: append-only JSONL, parent-linked entries, explicit
  compaction markers, torn-tail repair, replay on resume. Derive context from
  the active branch, never from provider history.
- `ContextAssembler` + `CompactionPolicy`: token budget, retained tail,
  no-tools summary completion, block on failed summarization. Never compact
  between tool_call and tool_result.
- `ToolRegistry`: one registry across providers; schema validation, abort
  handling, sequential/parallel execution, pre-execution hook.
- `ApprovalPolicy`: allow / deny / ask with durable pending requests (stronger
  than pi's hook-only model — this we keep from the Wiki design).
- Sessions: `~/.zeta/sessions/` (own home, not ~/.wiki), versioned schema.
- TUI later: inline renderer + sticky composer (prompt_toolkit + rich), reuses
  the wk-tui design decisions; lands after the loop is real.

## Isolation

- No imports from the wiki repo. Wiki integrates later behind a flag by
  consuming this package — never the reverse.
- No shared state with ~/.wiki, ~/.claude, ~/.codex beyond READING the OAuth
  credential stores those apps maintain (with consent, documented per
  provider).

## Ticket ladder (ZETA prefix)

| Ticket | Contract | Depends |
|---|---|---|
| ZETA-1 | Core types + ConversationStore (append-only JSONL, parent links, replay, torn-tail repair) + AgentLoop skeleton driven by a deterministic fake backend + tests | — |
| ZETA-2 | Anthropic backend: subscription OAuth (reuse/refresh existing Claude login), Messages API stream, provider-neutral events; single-completion boundary test | ZETA-1 |
| ZETA-3 | Codex backend: ChatGPT plan OAuth, backend-api Responses stream; same boundary test | ZETA-1 |
| ZETA-4 | ToolRegistry + first tools (read/list/exec with session cwd defaults) + pre-execution hook | ZETA-1 |
| ZETA-5 | ApprovalPolicy with durable pending requests + resume re-emit | ZETA-4 |
| ZETA-6 | ContextAssembler + CompactionPolicy (budget, retained tail, summary completion, digest) | ZETA-2 or ZETA-3 |
| ZETA-7 | Session resume (--continue/--resume), provider-transport recreation | ZETA-6 |
| ZETA-8 | TUI: sticky composer, patch_stdout pump, status bar, commit-on-newline streaming | ZETA-2/3 |
| ZETA-10 | `zeta login` CLI subcommand: wire existing PKCE OAuth machinery (build_authorization_url + exchange_authorization_code, both providers) into a CLI flow with a local redirect server; store tokens via AnthropicCredentialStore / CodexCredentialStore. Motivation: no login surface today; when both stored tokens revoke (session on another machine invalidates them), users have no in-harness recovery path. | ZETA-2/3 |
| ZETA-11 | Refresh-on-401 in the completion path: when a completion raises AuthError from a 401, attempt a token refresh once and retry the completion; fail loudly on the second 401. Motivation: `access_token()` only refreshes on local expiry — server-side revocation (token invalidated by a Claude/ChatGPT re-login elsewhere) leaves the local file "valid" and the completion 401s with no recovery. | ZETA-2/3 |
| ZETA-12 | `list` tool output cap fragility: the 10,000-char cap in ZETA-4's `list` truncates the accumulated pytest tmp_path parent after ~195 tests, silently breaking `tests/test_tools.py::test_paths_outside_session_cwd_are_allowed` when suite size grows. Either raise/remove the cap for the assertion path, or restructure the test to assert the invariant against a structured result rather than the truncated listing. Blocks nothing today (worked around in the test itself); this ticket fixes the underlying fragility so the workaround can be removed. | ZETA-4 |
| ZETA-13 | Structured tool results: adopt MCP's content-array return shape (`{"content": [{"type": "text"|"image"|"resource", ...}], "isError": bool, "structuredContent": <optional>}`). Replaces today's string returns everywhere in the ToolRegistry. Explicit `truncated: bool` + `full_size` metadata on text blocks — no more silent output caps. Same shape MCP servers already emit; foundation for ZETA-18. | ZETA-4 |
| ZETA-14 | `bash` tool: general shell tool folding in `exec` and replacing `list` entirely (list/git/search all become bash invocations). PERSISTENT per-session cwd — `cd foo && ls` state survives across bash calls within one session (serialized into the store's session state). Structured return: stdout, stderr, exit_code, cwd_after. Session-cwd defaults; explicit per-call override still allowed. Drop `list` from the registry. | ZETA-13 |
| ZETA-15 | `write` tool: write file at path with structured result (bytes_written, sha256, was_created vs was_overwritten). No auto-create parent dirs (fail-loud); explicit `create_parents: bool` flag if needed. | ZETA-13 |
| ZETA-16 | `edit` tool: Anthropic-native str_replace shape — `{"path", "old_string", "new_string"}`. `old_string` must be unique in file, else fail with the count. Cache-friendly (matches what claude/codex expect natively), no diff-parsing surface area. | ZETA-13 |
| ZETA-17 | Streaming tool output: long-running `bash` calls emit stream chunks the TUI renders live. Assistant sees ONE final tool_result on completion (or a canceled marker on abort) — no mid-turn partial tool_results in the model context. Preserves ZETA-8/ZETA-9's ordering invariants and keeps prompt cache reuse clean. Live-preview is a rendering concern only. | ZETA-14 + ZETA-13 |
| ZETA-18 | MCP client: one adapter that connects to any MCP server (stdio + streamable-http transports first), translates each server's tools into zeta's structured-tool-result shape. Per-server auth. Server discovery via a config file at `~/.zeta/mcp.json` (or `ZETA_MCP_CONFIG`). Servers appear as first-class tools in the registry; the assistant does not know they are remote. | ZETA-13 |
| ZETA-19 | `/status` slash command + slash-command primitive: minimal slash dispatcher — parse `/word args` at input time, before it reaches the model. First command is `/status`: session id, provider/model, retained tail, tokens used this session, tokens in current context, compaction-marker count, live pending approvals. Primitive stays small — no full DSL, no hooks. Later slash commands are separate tickets. | ZETA-2/3, ZETA-6 |
| ZETA-20 | TUI polish (OpenCode / Codex parity): status bar, sticky composer polish, markdown syntax colors, tool-call rendering (collapsible), streaming indicator, palette tune. Can ship as multiple rounds — first round targets visual parity with OpenCode's chrome. | ZETA-8 |
| ZETA-21 | Codebase restructure: split the flat `src/zeta/*.py` layout into `core/` (loop, store, context, session, approval+gate merged, abort, fake), `providers/` (transport, auth, anthropic, codex), `tools/` (registry + one file per tool: `read`, `list`, `exec` — current tools split out of today's `tools.py`), `skills/` (folder + a tiny `loader.py` stub for future markdown skills — see notes). `tui/` stays as-is. `types.py` and `cli.py` stay flat at top level. `__init__.py` becomes a public barrel only. Skills semantics locked: claude-code-style markdown files (prompt content + trigger keywords), model discovers and loads on demand, NO per-skill tool-mounting (tools stay mounted at session level). Add two governance tests: `tests/test_import_boundaries.py` (`core` imports nothing from above; `providers` can't import `tools`; `skills` can't import `providers`) and `tests/test_module_limits.py` (per-file line cap, per-dir file cap). NO behavior change — every existing test passes unchanged at the same assertion. SEQUENCING: ship BEFORE starting ZETA-13..20 so subsequent tool/tool-adjacent tickets land in the new layout naturally. | ZETA-7 |

## Deferred / open followups (not yet ticketed)

- **Prompt-cache management**: explicit cache breakpoints + hit-rate observability. Anthropic-native (we own the context per design). High value, needs a design discussion first.
- **Sub-agents, hooks, slash-command DSL**: general extensibility framework. Deferred until ZETA-19's primitive is in and one or two real slash commands surface a shape.
- **Rejected (not planned)**: third providers beyond Anthropic + Codex; thinking-mode / reasoning-effort controls in the assistant; sandbox for `exec`/`bash` (LLM is trusted; user retains approval policy for consequential ops).

Gate: `uv run pytest -q`. Review flow: same luna implementer -> sol reviewer
loop as the wiki repo; merges by the orchestrator after a clean pass.
