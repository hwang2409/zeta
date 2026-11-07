# performance probe

`bench.py` measures production session open/replay/messages, context assembly,
`zeta serve` requests, and the real TUI in a real pty. It clones the source
sessions (and their project memory, skills, agents, and `AGENTS.md`) with APFS
copy-on-write into a throwaway home before opening them; the originals are
never opened. The copied home is removed unless `--keep` is supplied.

```sh
lockf -k ~/.zeta/scratch/perf/bench.lock \
  uv run python scripts/bench/bench.py --session phoebe --probe tui
# compare another checkout (for example a worktree of origin/main):
uv run python scripts/bench/bench.py --tree .worktrees/main-ref --session fresh
```

- `--tree` picks the checkout whose `src/` is measured. The script re-executes
  itself with `PYTHONPATH=<tree>/src`, and every TUI process uses the same path.
  The output records `zeta_file` and `child_zeta_file` to prove it.
- `--probe` limits a run to `store`, `scaling`, `context`, `serve`, or `tui`.
- Timed samples run without a profiler. Each profile comes from a separate call
  or process and is written under `~/.zeta/scratch/perf/profiles/`.

The TUI probe starts the console entry point on a pty that is its controlling
tty (50x200), with `--provider ollama` pointed at a local stub server
(`ZETA_OLLAMA_BASE_URL`), so turns use the production HTTP client and no real
provider is called. It records:

- startup to the first prompt, and to the first provider request of a turn typed
  at that prompt (a resumed transcript is rebuilt after the prompt appears);
- one key per sample until that char appears in ANSI-stripped output, idle and
  while a 10 s stub stream runs (30 samples each);
- PageUp to the last rendered byte, Enter to the stub's request, and 60 s idle
  CPU/RSS plus a 10 s profiled window that counts event-loop wakeups.

A failed process reports its exit code and the last 2 KB of output. This is an
exploratory tool and not a CI check or wall-clock gate. Record the load average
next to every number; rank by operation counts and scaling, not wall time.
