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

both entry points run inside one per-run sandbox (`sandbox.py`) under
`~/.zeta/scratch/perf/sandbox-*`:

- source sessions and needed project data are cloned with `cp -cR`. after the
  copy, metadata is read only from the clone, and the clone is pinned to the
  local ollama stub (`provider`, `model`, `selected_accounts`), because resume
  uses the provider stored in the session, not the command-line provider.
- credential sources are removed from the environment of the script and every
  child (`*_API_KEY`, `*_TOKEN`, `*_SECRET`, and `ANTHROPIC`/`OPENAI`/`CLAUDE`/
  `CODEX` names). `CODEX_HOME`, `CLAUDE_CONFIG_DIR`, `ZETA_HOME`, and
  `ZETA_ACCOUNT_HOME` point inside the sandbox, so no fallback reaches
  `~/.codex/auth.json`, `~/.claude`, the keychain, or the real account home.
- a network guard (a python audit hook, also loaded in every child through
  `sitecustomize`) refuses any connect or name lookup to a host other than
  `127.0.0.1`/`localhost` and any keychain read. a refusal fails the run.
  before any probe, the run fails if a copied session names another provider.
- on exit, including an exception or `SIGTERM`, owned processes are stopped
  (process group `SIGTERM`, wait, `SIGKILL`, wait) and the sandbox is deleted
  unless `--keep` is given. a `SIGKILL` of the script itself leaves the
  sandbox behind.
- every timed sample stores its own 1-minute load reading:
  `"samples": [[ms, load], ...]`, with p50/p95 from the elapsed values and the
  `load_range` beside them.

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
- `search`: transcript search and memory preparation.

`children` and `search` need a copied session; `--session fresh` is rejected
for them.

`store_probes_more.py` supplies the `evict`, `context`, `children`, `serve`, and
`search` implementations loaded by the `store_probes.py` entry point. each
probe runs timed samples, operation counting, and a separate cprofile pass.
results go under `~/.zeta/scratch/perf/store/`; profiles go under
`~/.zeta/scratch/perf/profiles/`.

```sh
lockf -k ~/.zeta/scratch/perf/bench.lock \
  uv run python scripts/bench/store_probes.py open --session phoebe
```
