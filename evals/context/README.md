# Context-management benchmark

This benchmark compares Zeta context strategies on six long-horizon repository tasks and one four-turn persisted session. Candidate workspaces contain only fixtures; grader tests and reference overlays stay outside and are introduced only by the trusted grading process.

Run the full matrix against another checkout (for example, the strategies worktree):

```sh
uv run python -m evals.context.bench \
  --zeta-checkout /path/to/zeta \
  --strategies ,recall,budget \
  --reps 3 --concurrency 3 --token-budget 24000
uv run python -m evals.context.report evals/context/results.jsonl
```

Use `--tasks noisy-ledger` for a subset. The special task ID `session` runs four prompts in one persisted session. Results are append-only JSONL and completed `(task, strategy, rep)` keys are skipped on rerun.

The runner always enables `ZETA_CACHE_TRACE=1`, creates a fresh `ZETA_HOME`, and passes `ZETA_CONTEXT_STRATEGY` plus `ZETA_CONTEXT_TELEMETRY` to Zeta. Baseline is represented by the empty strategy string. For Codex authentication, ambient `~/.codex/auth.json` is copied with mode `0600` into that temporary run home; it is never placed in the candidate workspace or retained after the run.

## Pricing

Estimated cost uses this versioned table (USD per million tokens):

| Model | Uncached input | Cache read | Output |
|---|---:|---:|---:|
| `gpt-5.6-luna` | $0.20 | $0.02 | $1.20 |

Cache writes are collected and reported as tokens but currently have no added price for this model.
