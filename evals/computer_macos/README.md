# macos computer-use phase 0 and pilot

set `RUN` to a new directory under `~/.zeta/scratch/computer-use/runs/`. the
commands below never copy account files. they use a disposable `ZETA_HOME` and
`ZETA_ACCOUNT_HOME="$HOME/.zeta"` only for the single allowed status refresh.

```sh
mkdir -p "$RUN"
uv run python -m evals.computer_macos.fixture --port 8765 --state-dir "$RUN/fixture" &
FIXTURE_PID=$!
trap 'kill "$FIXTURE_PID" 2>/dev/null || true' EXIT
uv run python -m evals.computer_macos.run phase0 --run-dir "$RUN"
uv run python -m evals.computer_macos.run pilot --run-dir "$RUN" --trials 3
uv run python -m evals.computer_macos.grade --run-dir "$RUN"
uv run python -m evals.computer_macos.reset --run-dir "$RUN" --verify
```

before any pilot, and only once, run this no-input check. redact output before
putting it in the report:

```sh
env ZETA_HOME="$RUN/zeta-home" ZETA_ACCOUNT_HOME="$HOME/.zeta" \
  uv run zeta accounts status codex --refresh --json
```

phase 0 does not call a provider and does not send input events. it records 20
screenshots and 100 cursor-position calls. stop on any screen-recording,
accessibility, or other tcc prompt. phase 1 is not launched by this harness.
artifacts remain below `RUN`; do not commit them.
