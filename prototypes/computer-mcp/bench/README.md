# Computer-use benchmark

This benchmark measures pixel-only work on a disposable 1280x800 Linux desktop.
The model receives only the eight `computer__*` MCP tools. Host tools remain in
the headless schema until the CLI allowlist lands, but temporary settings hard-deny
them and any observed non-computer call automatically fails the run.

Each run starts a fresh networkless, mount-free container. The benchmark streams
fixture archives through `docker exec -i` into guest tmpfs. It never mounts a host
path. Graders read real guest files and application settings through `docker exec`
before the container is destroyed.

The image adds Chromium, PCManFM, and a Tint2 taskbar to the prototype's Mousepad
desktop. Chromium uses `--no-sandbox` because the outer container already drops
all capabilities, uses `no-new-privileges`, and applies Docker's seccomp profile.
It can open local files and a localhost-only form server while the container keeps
`--network none`. Gnumeric is intentionally omitted to keep the arm64 image smaller;
the task set does not need a spreadsheet.

## Tasks

- `edit-save-as`: edit a seeded note and save a copy in a new nested folder.
- `file-organize`: create nested folders and move selected files in PCManFM.
- `web-form`: fill and submit a localhost-only form.
- `web-fact`: navigate linked local pages and record a fact in a file.
- `browser-preference`: enable the home button through Chromium Settings.
- `cross-app`: read an invoice in Chromium and write a summary in Mousepad.
- `multipart`: complete a 15+ action browser, editor, and file-manager workflow.
- `recovery`: finish an edit despite an unsaved-changes interruption.
- `prompt-injection`: extract a fact without following hostile page text.
- `discoverability`: rename through visible UI without keyboard shortcuts.

## Run

Use only the dedicated Lima Docker socket and an empty Docker CLI configuration:

```sh
export ZETA_COMPUTER_DOCKER_HOST="unix://$HOME/.lima/zeta-sandbox/sock/docker.sock"
export DOCKER_HOST="$ZETA_COMPUTER_DOCKER_HOST"
export DOCKER_CONFIG="$(mktemp -d /tmp/zeta-computer-docker.XXXXXX)"
printf '{}\n' > "$DOCKER_CONFIG/config.json"
unset DOCKER_CONTEXT

uv run python prototypes/computer-mcp/bench/validate_graders.py
uv run python prototypes/computer-mcp/bench/run.py \
  --reps 3 --concurrency 3 --model codex:gpt-5.6-luna --suite baseline
```

The runner makes a temporary `ZETA_HOME` for every trial and copies only the Codex
OAuth file into it. Results are written below `/tmp/computer-bench/<suite>/` as
JSON and a Markdown table with 95% Wilson score intervals. Each trial records tool
names, policy violations, steps, screenshots, token usage, wall time, errors, and
cleanup. Failed runs retain their transcript and final screenshot in that tree.

To adopt the pending native allowlist, change `HEADLESS_TOOL_ARGS` in `run.py` from
an empty list to `["--tools", "computer__*"]`; no other runner change is needed.

See [`RESULTS.md`](RESULTS.md) for the 30-run Codex Luna baseline and failure
analysis.
