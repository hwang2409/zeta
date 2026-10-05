# Context-cap experiments

This directory contains the context benchmark imported from `exp/ctx-final`,
two absolute-length tasks, and a read-only stored-session replay.

## Offline replay

The replay snapshots each `conversation.jsonl` size, reads it directly without a
store/session object, follows the final parent chain, and calls production
`evict_messages`, tool-result fitting, token accounting, target ratio, and
hysteresis ratio. Independent logs can run in bounded worker processes:

```sh
uv run python -m evals.context.replay \
  --limit 15 --concurrency 6 \
  --caps 64000,100000,150000,200000,300000,400000,1000000 \
  --output /tmp/context-cap-phase-a.json
```

No transcript text is printed. Output contains short IDs and aggregate metrics.

## Live benchmark

`long-delayed-clue` and `long-noisy-ledger` each put the useful rule in early
stored evidence, then require two deterministic generated traces totaling about
340k estimated tokens before the fix. Graders and references remain outside the
candidate workspace. The runner copies `~/.codex/auth.json` into a temporary
home with mode `0600` and deletes that home after the run.

```sh
uv run python -m evals.context.bench \
  --zeta-checkout "$PWD" \
  --tasks long-delayed-clue,long-noisy-ledger \
  --token-budgets 100000,200000,400000,1050000 \
  --reps 3 --concurrency 3 \
  --results /tmp/context-cap-phase-b.jsonl
uv run python -m evals.context.report /tmp/context-cap-phase-b.jsonl
```

The 1,050,000 cap is the published `gpt-5.6-luna` window. Before a full run,
estimate the matrix input as task count × reps × the expected cumulative request
contexts. Reduce reps or tasks if that estimate exceeds 60 million input tokens.
The benchmark records grader results, `recall_history` calls, evictions, input,
cache-read, cache-write and output tokens, wall time, and estimated cost.

## Pricing

Estimated cost uses this versioned table (USD per million tokens):

| Model | Uncached input | Cache read | Output |
|---|---:|---:|---:|
| `gpt-5.6-luna` | $0.20 | $0.02 | $1.20 |

Cache writes are collected but have no added price for this model.
