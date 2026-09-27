# Zeta experiment ledger

Each candidate needs a baseline, a changed arm, and a real workflow check before
merge. A green unit suite alone is not evidence of better agent outcomes.

| Idea | Status | Evidence and next gate |
| --- | --- | --- |
| Small, repeatable real-workflow evals for Zeta's normal tools and model team | PR #222 merged; needs broader corpus | `evals/run.py` grades five live Luna workflows in fresh directories and reports artifacts separately from completion. The earlier 5/5 artifact, 2/5 completion run exposed tool/turn overhead that unit tests missed. Expand to real repos and repeat before treating percentages as stable. |
| Skip `todo` for one edit plus verification | PR #223 merged | On the one-edit task, unchanged identity completed 3/8 and the exact appended rule completed 8/8; artifacts were 16/16. Both arms completed 8/8 on four other tasks (two repeats each). The changed packaged identity in a fresh `ZETA_HOME` completed 16/16 across all five tasks; the updated local user identity completed another 4/4 one-edit runs. Existing user-edited identities do not update automatically. |
| Cache reuse on short real workflows | Measured; no client prefix churn found | Opt-in trace across 18 Luna completions in three live workflows: 47,104 cached of 65,060 input tokens (72.4%). All subsequent turns retained the full previous message prefix and the same 16 tools; no compaction. Final inventory turns reached 85.8–87.7%. [OpenAI's caching guide](https://developers.openai.com/api/docs/guides/prompt-caching) explains that newly appended input remains uncached even on a prefix hit. Next: trace longer TUI sessions and test API diagnostics only if the subscription endpoint supports them; do not pad prompts just to inflate hit rate. |
| Reduce turns spent on bookkeeping and child-agent coordination | Investigating | In the 5-task packaged-prompt trial, three correct artifacts still hit their turn caps; traces include repeated `todo`, `agent_status`, and `read` calls. Compare a narrow tool/prompt intervention against the same corpus before merging. |
| One generic interactive `browser` tool, with Jev selecting elements from a bounded page snapshot | Investigating | `jev/harness` reports 36/60 routed vs 26/60 stock completions in an **offline fake-page** matrix. Its seven browser tools conflict with Zeta's minimal-tool preference. Prototype one action-discriminated tool, then compare on live local and public pages before a merge. Source: private `hwang2409/jev/harness/evals/RESULTS.md`. |
| Disposable computer for each agent session | Blocked for cloud validation | [Muse](https://about.fb.com/news/2026/09/introducing-muse-personal-ai-agent/) uses a dedicated browser-equipped VM with separate approval controls. Mock locally first; local Docker is installed but its daemon is unavailable. A real remote VM needs an account/provider choice and spending limit. No cloud deployment yet. |
| Agent-to-agent handoff across users | Researching | The intriguing part is identity, consent, and an auditable handoff, not another chat tool. Find a primary Instinct protocol description; prototype only between two local Zeta sessions before involving another person. |

Evaluation reference: [Harbor](https://github.com/harbor-framework/harbor) runs
agents against containerized, verifiable tasks. Zeta should first use a tiny
local corpus; integrate a full Harbor adapter only if that corpus shows value.

Cache trace reproduced with `ZETA_CACHE_TRACE=1 uv run python evals/run.py
--task csv-parser-repair --task slugify-cli --task inventory-chain`.
