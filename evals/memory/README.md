# Zeta persistent-memory benchmark

This phase-1 benchmark measures whether persistent project memory helps a later agent action across separate sessions. It covers six families with two chains each. Opaque literals make recall tasks non-inferable.

## Isolation

Each cell gets a fresh fixture repository and mode-0700 `ZETA_HOME`. Every phase launches `zeta -p` without `--resume`. Raw session directories are removed between phases, so the chain shares only the Zeta project registry and memory files. Hidden tests remain outside the workspace until trusted grading.

The harness stages `~/.codex/auth.json`, when present, as mode 0600 under the temporary home and points `CODEX_HOME` to it. The temporary directory is deleted after the cell. Failed-workspace retention excludes the provider directory and auth files. No credential content is printed or stored in results.

The agent receives an explicit `read,write` tool allowlist. `--yolo` auto-approves only these advertised tools so a headless final phase can create `answer.json`. Setup phases are graded to ensure they did not write the answer early.

## Strategies

- `S0`: an empty current Zeta project memory.
- `S1`: oracle-quality content written through `ProjectRegistry.update_memory`, equivalent to a prior approved user/agent memory write.
- `oracle-snippet`: the relevant source snippet in the final prompt.
- `oracle-history`: all prior user prompts and assistant acknowledgements in the final prompt.

The strategy seam is limited to project-memory seeding and final-prompt context. S2 and S3 can add adapters there while reusing chain execution, grading, retry, telemetry, and reporting.

## Run

From the repository root:

```sh
# Budget estimate only: 144 cells projects 25.92M input tokens.
uv run python -m evals.memory.bench \
  --zeta-checkout "$PWD" --estimate-only

# Required one-cell smoke.
uv run python -m evals.memory.bench \
  --zeta-checkout "$PWD" \
  --results evals/memory/smoke-results.jsonl \
  --tasks stable-decision-cedar --strategies S1 --reps 1 --concurrency 1

# Baseline matrix: 12 x 4 x 3, bounded concurrency, resumable JSONL.
uv run python -m evals.memory.bench \
  --zeta-checkout "$PWD" \
  --results evals/memory/results.jsonl \
  --reps 3 --concurrency 3

uv run python -m evals.memory.report evals/memory/results.jsonl \
  --output /Users/henry/.zeta/scratch/memory-bench-baselines.md
```

The matrix is append-only and keyed by task, strategy, repetition, model, token budget, and benchmark revision. A network/provider failure is retried once inside its cell and counted. Re-running skips terminal keys.

## Telemetry

Each result records hidden-test pass and partial count; wrong-memory, stale-fact, and abstention outcomes; request-level uncached/cache-read/cache-write/output tokens; requests; tool calls; memory searches; retrieved and stored memory bytes; wall time; actual price-table cost; network drops; and zero-valued proposal/approval fields reserved for S2/S3. Extraction and provenance metrics are `null` for seeded/read-only baselines.
