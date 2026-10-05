# Computer-use MCP prototype

This prototype gives Zeta pixel-based control of a disposable Linux desktop. The
MCP server keeps the model-facing interface independent from the runtime through
`DesktopBackend`. The current backend runs a hardened Docker container inside a
dedicated Lima VM. A later backend can control a Lima guest directly without
changing the MCP tools.

This is a prototype, not a merge-ready security boundary.

## Isolation boundary

Create the dedicated VM once:

```sh
limactl create --name=zeta-sandbox --mount-none --cpus=4 --memory=6 \
  --tty=false template:docker
limactl start zeta-sandbox
```

Before Docker use, verify that the VM has no host mounts:

```sh
limactl shell zeta-sandbox -- mount | grep -E 'virtiofs|9p|sshfs|/Users'
# Expected: no output and exit 1.
limactl shell zeta-sandbox -- ls /Users
# Expected: "No such file or directory" and exit 2.
```

The generated Lima configuration must have no `mounts` and only this socket
forward:

```yaml
portForwards:
  - guestSocket: /run/user/{{.UID}}/docker.sock
    hostSocket: '{{.Dir}}/sock/docker.sock'
```

Every command uses the socket explicitly. It never uses the default Docker
context. The backend also uses an empty, isolated Docker CLI configuration so a
host credential helper or context cannot redirect the command:

```sh
export ZETA_COMPUTER_DOCKER_HOST="unix://$HOME/.lima/zeta-sandbox/sock/docker.sock"
export DOCKER_HOST="$ZETA_COMPUTER_DOCKER_HOST"
export DOCKER_CONFIG="$(mktemp -d /tmp/zeta-computer-docker.XXXXXX)"
printf '{}\n' > "$DOCKER_CONFIG/config.json"
unset DOCKER_CONTEXT
```

The desktop container has no network, host mounts, or published ports. It uses
a read-only root, tmpfs for `/tmp` and `/home/zeta`, UID/GID 65532, all
capabilities dropped, `no-new-privileges`, Docker's default seccomp profile, an
init process, and PID/CPU/memory limits. Runtime inspection rejects a container
if these core controls are absent. No clipboard is shared.

The image contains Xvfb at 1280x800, Openbox, Tint2, Mousepad, PCManFM,
Chromium, xdotool, ImageMagick, and x11vnc. Chromium is for offline local pages
and localhost-only benchmark forms; the container still uses `--network none`.
VNC is off by default. `ZETA_COMPUTER_VNC=1` starts x11vnc on the container
loopback interface only; the backend still publishes no port.

Stop the VM when work is complete:

```sh
limactl stop zeta-sandbox
```

Delete it when it is no longer needed:

```sh
limactl delete zeta-sandbox
```

## Build and test

```sh
DOCKER_HOST="$ZETA_COMPUTER_DOCKER_HOST" \
  docker build --platform linux/arm64 -t zeta-computer-mcp:local \
  prototypes/computer-mcp

env -u ZETA_ANTHROPIC_OAUTH_COMPAT uv run pytest -q
uv run ruff check .

ZETA_COMPUTER_DOCKER=1 \
ZETA_COMPUTER_DOCKER_HOST="$ZETA_COMPUTER_DOCKER_HOST" \
  env -u ZETA_ANTHROPIC_OAUTH_COMPAT \
  uv run pytest -q tests/test_computer_mcp.py
```

## MCP tools and approval

By default, the server exposes `computer_screenshot`, `computer_click`,
`computer_double_click`, `computer_drag`, `computer_type`, `computer_key`,
`computer_scroll`, and `computer_wait`. Each action returns a fresh JPEG.
Screenshots are reduced from the physical 1280x800 display to a fixed 1024x640
model frame. Coordinates are validated in that model frame, scaled to the
physical display, rounded, and clamped to the physical edge. With no feature
environment variable, tool names, schemas, results, and screenshot behavior are
unchanged.

Set `ZETA_COMPUTER_FEATURES` to a comma-separated list to opt in to independent
features:

- `batch` adds `computer_batch`. It validates all coordinates before it starts,
  executes at most 10 actions in order, stops at the first runtime error, and
  returns per-action status with one final screenshot by default. Set its
  `screenshot` argument to false when no final image is needed.
- `zoom` adds `computer_zoom`. Its `x`, `y`, `w`, and `h` describe a crop in the
  global 1024x640 model frame. The physical-screen crop is enlarged to 1024x640.
  The enlarged pixels are not action coordinates: clicks still use global model
  coordinates. Each zoom result includes the exact conversion formula.
- `observe` adds JSON text to every screenshot result. It includes the active
  window, top-level window titles and global model-frame bounds, mouse position,
  current/previous screen hashes, and the focused widget's accessible role, name,
  and text value when AT-SPI provides them. Values are bounded to 50,000
  characters. Clipboard contents are never read.
- `cursor` draws a red and white pointer marker into each returned screenshot.
- `settle` waits for two near-identical sampled frames after each action, capped
  at two seconds, and reports the elapsed settle time.
- `verify` reads accessible focused text after `computer_type` and each batch type
  action. It reports the resulting value and warns when an entry does not exactly
  equal the typed text or an editor buffer does not contain it. Tool descriptions
  tell the model to make precise edits instead of retyping whole documents.
- `plan` adds `computer_plan` and `computer_check`. The server stores one bounded
  checklist without model calls and appends it to every later screenshot result.
  Tool descriptions tell the model to plan multi-part tasks first and verify each
  step on screen before marking it done.

For example:

```sh
ZETA_COMPUTER_FEATURES=batch,observe,settle \
  python prototypes/computer-mcp/server.py
```

The server uses AT-SPI only for focused-text read-back. It does not expose a full
accessibility tree or element-click interface.

Zeta mounts these as names such as `computer__computer_click`. MCP tools use the
normal tool approval path. In the TUI, the user sees an approval card with the
qualified tool name and its arguments. The demo uses `--yolo` because its only
external tools are these networkless sandbox controls. Do not use that flag for
an MCP configuration that includes host tools.

The backend starts on the first computer tool call, enforces a 15-minute default
TTL, and destroys the container on MCP EOF, exit, or termination. The demo
grades the real guest file
and saves a final screenshot before destruction.

## Demo

Run each trial with a temporary `ZETA_HOME`:

```sh
ZETA_COMPUTER_DOCKER_HOST="$ZETA_COMPUTER_DOCKER_HOST" \
  uv run python prototypes/computer-mcp/demo.py --run run-1
```

`demo.py` creates and removes its own temporary `ZETA_HOME`; it never registers
or writes the server in `~/.zeta`. It copies only the existing Codex OAuth file
into the temporary home so any refresh is also temporary. Its temporary global
settings hard-deny every host built-in tool; `--yolo` therefore grants only the
computer MCP controls. The summary also fails if any non-computer tool call is
observed. It invokes Codex `gpt-5.6-luna` with a maximum of 60 turns and asks it
to create `~/notes/demo.txt`. Results, JSONL events, metrics,
the graded file, and `final.jpg` are written below `/tmp/computer-demo/<run>/`.

## Benchmark

`bench/` contains ten seeded desktop tasks, deterministic guest-state graders,
a task/repetition/model matrix runner, and reference-solution validation. See
[`bench/README.md`](bench/README.md) for the task list and commands.

## Next steps

1. Add a direct-Lima backend. Create a per-session guest from an immutable
   template, provision the desktop in the guest, send input and screenshots over
   Lima SSH, and destroy the guest at the backend lifecycle boundary. This
   removes the Docker daemon from the trusted path and makes the VM itself the
   disposable desktop.
2. Add Anthropic's native computer tool for models that support it. Negotiate
   the supported tool version, map native actions to `DesktopBackend`, and keep
   the portable MCP functions for Codex and other providers. Preserve one fixed
   display size per session and the same result-grading boundary.
3. Add a restricted observation/action batching interface if screenshot token
   cost is too high. Do not weaken the rule that grades guest state instead of
   model prose.
