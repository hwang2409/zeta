# macos computer-use phase 0 and pilot

set `RUN` to a new directory under `~/.zeta/scratch/computer-use/runs/`. the pilot owns its fixture, Safari window, disposable home, and artifacts. do not start a second fixture.

```sh
RUN="$HOME/.zeta/scratch/computer-use/runs/$(date +%s)"
mkdir -p "$RUN"
uv run python -m evals.computer_macos.run phase0 --run-dir "$RUN"
uv run python -m evals.computer_macos.run pilot --run-dir "$RUN" --trials 3
uv run python -m evals.computer_macos.grade --run-dir "$RUN"
uv run python -m evals.computer_macos.reset --run-dir "$RUN" --verify
```

before any pilot, and only once, run this no-input check. redact output before putting it in the report:

```sh
env ZETA_HOME="$RUN/zeta-home" ZETA_ACCOUNT_HOME="$HOME/.zeta" \
  uv run zeta accounts status codex --refresh --json
```

phase 0 does not call a provider and does not send input events. it records 20 screenshots and 100 cursor-position calls. stop on any screen-recording, accessibility, or other tcc prompt. all pilot failures still write a truthful atomic `pilot.json` and cleanup artifact. artifacts remain below `RUN`; do not commit them.

`pilot()` makes exactly three sequential `run_trial()` calls. each trial owns its fixture, browser state, backend, and independent zeta session. production uses the Codex `codex`/`gpt-5.6-luna` completion adapter and `MacBackend`; no-desktop tests inject a scripted completion adapter and fake computer backend into the same `AgentLoop.run_turn()` and tool dispatcher. there is no provider subprocess or headless event parser. `--trials 3` is the only accepted pilot value; the scripted test proves 3/3 trials, screenshots, clicks, fixture HTTP, grading, metrics, independent sessions, and observed cleanup without external action.
