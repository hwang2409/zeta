# performance probe

`bench.py` measures production session open/replay/messages and starts the real
`zeta` TUI in a real pty. It clones source sessions with APFS copy-on-write
before opening them. The copied home is removed unless `--keep` is supplied.
No provider request is made: the probe never submits a turn and starts the TUI
with `--no-auto-memory`.

```sh
mkdir -p ~/.zeta/scratch/perf
uv run python scripts/bench/bench.py --session fresh --session phoebe --session orchestrator
```

The JSON output includes operation scale, load average at each measurement
boundary, p50/p95 replay timing, and pty startup/key-to-render timing. This is
an exploratory tool and is not a CI check or wall-clock gate. Run one process
at a time on a quiet machine before comparing wall-clock values.
