# performance probes

these scripts are exploratory performance tools. they are not ci checks and no
wall-clock result is a pass/fail gate. record the load average beside every
number, and rank results by operation counts and scaling before noisy timings.

## shared safety rules

run one probe at a time under the shared `lockf` lock:

```sh
lockf -k ~/.zeta/scratch/perf/bench.lock \
  uv run python scripts/bench/bench.py --session phoebe --probe tui
```

both entry points clone source sessions with `cp -cR` into a throwaway home.
the originals are never opened. project memory and the other required home
files are cloned as needed. the copied home is removed unless a script's
`--keep` option is used. providers are stubbed at the network boundary, so
these probes do not call a real provider or spend model budget.

## `bench.py`

`bench.py` exercises the pty tui, startup, turn start, idle, and `zeta serve`.
its `--probe` values are `store`, `scaling`, `context`, `serve`, and `tui`.
`--tree` selects the checkout whose `src/` is measured; the script re-executes
with that tree on `PYTHONPATH` and reports the imported file paths. profiles are
written under `~/.zeta/scratch/perf/profiles/`.

```sh
uv run python scripts/bench/bench.py --tree .worktrees/main-ref \
  --session fresh --probe tui
```

(the tui run includes startup, turn start, idle, and rendering measurements.)
the tui probe uses the real console entry point on a controlling pty. it uses a
local ollama stub through `ZETA_OLLAMA_BASE_URL` and measures prompt startup,
turn start, key input, idle, rendering, and the stubbed stream. the serve probe
uses the real serve protocol. store and context probes call the production
session store and context assembler; scaling repeats store work at 10%, 50%,
and 100% transcript sizes.

## `store_probes.py` and `store_probes_more.py`

`store_probes.py` is the command-line entry point. it supports these probes:

- `open`: session open and replay work;
- `append`: synchronous and asynchronous append paths;
- `evict`: context eviction;
- `context`: turn-start context assembly;
- `children`: child-agent rpc and receipt work;
- `serve`: `zeta serve` requests;
- `search`: transcript search and memory preparation;
- `clean`: remove the cached copied homes and results.

`store_probes_more.py` supplies the `evict`, `context`, `children`, `serve`, and
`search` implementations loaded by the `store_probes.py` entry point. each
probe runs timed samples, operation counting, and a separate cprofile pass.
results go under `~/.zeta/scratch/perf/store/`; profiles go under
`~/.zeta/scratch/perf/profiles/`.

```sh
lockf -k ~/.zeta/scratch/perf/bench.lock \
  uv run python scripts/bench/store_probes.py open --session phoebe
lockf -k ~/.zeta/scratch/perf/bench.lock \
  uv run python scripts/bench/store_probes.py clean
```
