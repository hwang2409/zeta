# Computer use

`zeta --computer` gives the model a disposable Linux desktop and nothing else.
The model sees screenshots and controls the desktop with the mouse and the
keyboard. It cannot read or change files on your computer, run host commands, or
reach the network.

```sh
zeta computer setup                       # once: create the VM, verify it, build the image
zeta --computer                           # TUI session with a sandbox desktop
zeta --computer -p "write a note in Mousepad"   # one headless turn
zeta computer watch --live                # spectator page and live VNC view
zeta computer stop                        # remove desktops and stop the VM
```

In the TUI, `/computer` reopens the current session as a computer session.

## Requirements

- macOS or Linux with [Lima](https://lima-vm.io) 2.x (`limactl`) and the Docker
  CLI (`docker`). Zeta does not use Docker Desktop, Colima, or the default Docker
  context.
- About 1.2 GiB of VM disk for the desktop image (Debian, Xvfb, Openbox, Tint2,
  Mousepad, PCManFM, Chromium, xdotool, ImageMagick, x11vnc).

## Setup and lifecycle

| Command | Effect |
|---|---|
| `zeta computer setup` | Create the `zeta-sandbox` VM with `--mount-none` (if absent), start it, verify isolation, and build the image. |
| `zeta computer status` | Show the VM, the isolation checks, the image, and running desktops. |
| `zeta computer stop` | Remove every desktop and stop the VM. |
| `zeta computer destroy [--yes]` | Delete the VM and its images. |
| `zeta computer watch [SESSION] [--live]` | Serve the spectator page for a session (default: newest); `--live` also opens a VNC view. |

`setup` builds the image from the Dockerfile that ships in the Zeta package
(`zeta/computer/assets/`). The image tag is a digest of the Dockerfile and its
entrypoint, so a Zeta update that changes them requires `zeta computer setup`
again. The build runs inside the VM and needs network access there; desktops do
not.

Size the VM in the global `~/.zeta/settings.toml`. Sizes apply only when `setup`
creates the VM; to resize, run `zeta computer destroy` and `setup` again.

```toml
[computer]
backend = "local"      # desktop backend (only "local" today)
cpus = 4
memory_gib = 6
disk_gib = 50
recording = true       # record frames and actions to the session directory
desktop_minutes = 60   # maximum lifetime of one desktop
```

Project settings cannot configure computer use.

## Security model

The boundary has three layers. Each layer alone keeps host files away from the
model.

**1. Tool policy.** A computer session uses the
[tool allowlist](../README.md#tool-availability) with the exact names of the
computer tools (`computer__screenshot`, `computer__click`, ...) and
`--require-tools`. Every host tool (files, shell, agents, background tasks,
user MCP servers, external tool modules) is absent from provider requests and
rejected if called. Command hooks are off, as in every restricted session. The
session mounts only the computer MCP server: user and project `mcp.json` servers
are not read and never start. The policy is stored in the session; a resumed
computer session stays restricted, and `/computer` can only narrow a session,
never widen it. `--disallowed-tools` can remove more computer tools.

Computer tools run without approval prompts because they act only inside the
sandbox. Approval `deny` and `ask` rules in settings still apply to them.

**2. Dedicated VM.** Desktops run in a Lima VM, `zeta-sandbox`, created with
`--mount-none`. Before it starts a desktop, Zeta verifies the VM and refuses to
continue if one check fails:

- the Lima configuration has no mounts and forwards only the Docker socket,
  with no reverse forwards;
- `/proc/mounts` in the guest has no `virtiofs`, `9p`, or `sshfs` mount and no
  mount at `/Users` or at your home directory;
- `/Users` and your home directory do not exist in the guest;
- the Docker socket reaches the engine named `lima-zeta-sandbox`.

Zeta reaches Docker only through that VM's socket: every command passes
`--host unix://…/zeta-sandbox/sock/docker.sock` and `--config` with a fresh,
empty, private configuration directory, and runs with a minimal environment
(no `DOCKER_CONTEXT`, no credential helpers). It never runs `colima`.

**3. Hardened container.** Each session gets one container started with
`--network none`, `--read-only`, tmpfs `/tmp` and `/home/zeta`, user
65532:65532, `--cap-drop ALL`, `no-new-privileges`, `--init`, 1 CPU, 1 GiB
memory, 256 PIDs, no mounts, and no published ports. Zeta inspects the running
container and destroys it if a control is missing. No clipboard is shared.

### Limits

- Content on the sandbox screen (web pages in the guest, file contents) is
  untrusted input to the model. A prompt injection can make the model misuse the
  desktop, but it cannot reach the host through the tools.
- Guest Chromium has no network. Tasks that need the internet do not work.
- Each desktop lives at most `desktop_minutes`; the next action starts a fresh
  desktop. The container also exits by itself shortly after that limit, so a
  killed Zeta process cannot leave a desktop running for long.
- The outer boundary is the hypervisor. This design does not defend against a
  VM escape.

## Lifecycle of a desktop

The desktop starts on the first computer tool call, not at session start. At
session end (exit, Ctrl-C, or an aborted headless turn) Zeta removes every
container labeled with the session ID, even if the MCP server was killed.
`zeta computer stop` removes all desktops.

## Watching

Recording is on by default. The MCP server writes `metadata.json`,
`events.jsonl`, and numbered JPEG frames to `<session dir>/computer/`. At
session start Zeta prints two lines:

```text
computer · spectator http://127.0.0.1:PORT/?token=…
computer · live view: zeta computer watch --live SESSION_ID
```

The spectator page shows the latest frame with click, drag, and scroll markers,
an action timeline, and replay controls. It is live while the session runs and
a replay afterward; `zeta computer watch [SESSION]` serves it again later.

`zeta computer watch --live` starts x11vnc inside the desktop on the guest
loopback with a random one-time password and bridges it to a random
`127.0.0.1` port through `docker exec` standard I/O. Open the printed
`vnc://` URL (on macOS, `open vnc://127.0.0.1:PORT`) and enter the password. The
view is always read-only. One live view is
allowed per desktop. Ctrl-C closes it and removes the VNC server.

Both listeners bind only to `127.0.0.1`. The spectator page needs its random
token, loads no external assets, and sends a strict Content Security Policy.
Treat the token and the VNC password as local secrets: other processes of the
same user can read them.

## Tools

| Tool | Action |
|---|---|
| `screenshot` | Fresh screenshot |
| `click`, `double_click` | Click at `x`, `y` (optional `button`) |
| `drag` | Drag from `x1`, `y1` to `x2`, `y2` |
| `type` | Type text into the focused control; newlines press Return |
| `key` | xdotool key chord, such as `ctrl+s` |
| `scroll` | Scroll at `x`, `y` by `dx`, `dy` |
| `wait` | Wait up to 5 seconds |
| `batch` | Run 1 to 10 of the actions above, then return one screenshot |

Screenshots are 1024x640 JPEG frames of the 1280x800 display. Coordinates use
the 1024x640 frame; Zeta validates them and scales them to the display. A batch
is validated completely before its first action runs, and stops at the first
runtime error with a status for each action. After every action the server
waits up to 2 seconds for the screen to settle, then returns the screenshot and
an observation: the active window, window titles and bounds, the focused
widget's role and text (through AT-SPI), the pointer, and whether the screen
changed.

## Benchmark results

The tool set is the winner of three rounds of experiments on 20 desktop tasks
(Codex `gpt-5.6-luna`, isolated desktops, deterministic graders that read the
real guest state).

| Configuration | Hard tasks passed (10 tasks x 3) | Tool calls | Screenshots |
|---|---:|---:|---:|
| Baseline (single actions only) | 26/30 | 1,162 | 1,169 |
| **batch + observe + settle (shipped)** | **28/30** | **738** | **727** |
| batch + observe + settle + plan | 19/30 | 947 | 934 |

The shipped set used 37% fewer tool calls and screenshots and 28% fewer
uncached input tokens than the baseline. A checklist tool (`plan`) lowered the
pass rate because the model marked steps done that the graders found incorrect.
Typed-text read-back (`verify`), a zoom tool, and a drawn cursor did not help,
so they are not included.

## Backends

`--computer-backend local` (the default) is the Lima and Docker backend above.
The server talks to a backend only through the `DesktopBackend` interface
(`start`, `destroy`, `close`, `screenshot`, `input`, `observe`, `settle`), and
the X11 command layer is shared, so a hosted backend (for example Modal or E2B)
is one new entry in `zeta.computer.backend.BACKENDS`.

## Development

```sh
env -u ZETA_ANTHROPIC_OAUTH_COMPAT uv run pytest -q tests/test_computer.py tests/test_computer_session.py
# Against the VM (after `zeta computer setup`):
ZETA_COMPUTER_DOCKER=1 env -u ZETA_ANTHROPIC_OAUTH_COMPAT uv run pytest -q tests/test_computer_integration.py
```
