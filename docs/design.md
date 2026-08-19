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
| ZETA-4 | ToolRegistry + first tools (read/list/exec with cwd jail) + pre-execution hook | ZETA-1 |
| ZETA-5 | ApprovalPolicy with durable pending requests + resume re-emit | ZETA-4 |
| ZETA-6 | ContextAssembler + CompactionPolicy (budget, retained tail, summary completion, digest) | ZETA-2 or ZETA-3 |
| ZETA-7 | Session resume (--continue/--resume), provider-transport recreation | ZETA-6 |
| ZETA-8 | TUI: sticky composer, patch_stdout pump, status bar, commit-on-newline streaming | ZETA-2/3 |

Gate: `uv run pytest -q`. Review flow: same luna implementer -> sol reviewer
loop as the wiki repo; merges by the orchestrator after a clean pass.
