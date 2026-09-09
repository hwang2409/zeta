# zeta gui

The native GUI uses [GPUI Kit](https://github.com/longbridge/gpui-kit) 0.6.
The plain Rust protocol client and connection actor speak the existing
[`zeta serve` protocol](../docs/serve-protocol.md).

```sh
make gui
# Connect to an existing server:
cargo run --manifest-path gui/Cargo.toml -- --socket /path/to/serve.sock
```

The GUI starts a real-provider server by default. Set `ZETA_BIN` if `zeta` is
not on `PATH`. For local tests, set `ZETA_HOME` to a temporary directory.
A fake server can use `zeta serve --provider fake --socket /tmp/demo.sock`.

## Core chat loop

- Create a session or select one from the virtual sidebar. Rows show its name
  or first-message preview and relative age. An empty session shows
  `New conversation`; an older server without previews gets the same fallback.
- Enter sends; Shift-Enter inserts a newline. Rejected sends retain the draft.
  The composer explains why sending is disabled.
- Assistant messages use Kit rich text: headings, lists, emphasis, code fences,
  selection, and syntax colors for Rust, Python, Bash, JSON, and TypeScript.
  Streaming previews retain the latest 8 KiB / 40 lines until the committed
  message restores the complete source.
- Kit's virtual message scroller follows the tail. Scrolling up pauses follow;
  `Jump to latest` restores it. Tool receipts are collapsed status lines.
- Approval dialogs use Enter to approve and Escape to deny. The dialog stays
  until the server confirms the decision. Escape or Ctrl-C aborts an active turn.
- Connection loss shows a reconnect button. Session RPC errors show an error
  without changing the connection state.
- The status bar shows the model, token count, and cache rate at turn boundaries.
  Missing values show an em dash. Tokens include input, cache reads, cache writes,
  and output. Cache rate divides cache reads by all input tokens.

Kit semantic themes follow system appearance. JetBrains Mono 2.304 Regular,
Medium, Bold, and Italic are embedded and registered at startup. Their OFL
license is in `assets/fonts/OFL.txt`.

## Session ergonomics

- The sidebar lists branches under the session picker. Click a branch to switch;
  the current branch is disabled with a `*` marker. A `Settings` control opens a
  Kit overlay with a grouped Claude/Codex model picker (the current model is
  labelled `current` and auto-scrolled into view) and an approval-mode selector.
  Applying a cross-provider swap that fails RPC (missing credentials, model
  denied) keeps the modal open with an inline error.
- User messages carry a `Fork here` action that dispatches `fork_message`.
  Branches and forks are hidden on legacy servers without session extensions.
- The composer accepts PNG, JPEG, GIF, and WebP images through the `Attach
  image` button or Cmd-V paste. Attachments render as chips (name + byte count)
  above the textarea; validation errors show inline. `Send` dispatches
  `SendImages` and the confirmation records the attachment on the transcript
  row so replays retain the file listing.

## Build and targeted checks

GPUI Kit brings matching published `gpui-pre` crates (locked at 0.3.4), replacing
the previous Zed git revision. Kit always enables runtime shaders, so Command
Line Tools hosts need no offline Metal compiler. `runtime-shaders` remains an
empty compatibility feature for existing commands.

```sh
cargo fmt --manifest-path gui/Cargo.toml -- --check
cargo clippy --manifest-path gui/Cargo.toml --all-targets -- -D warnings
ZETA_HOME="$(mktemp -d)" cargo test --manifest-path gui/Cargo.toml
```

## Developer Mac app

```sh
make gui-app
open dist/Zeta.app
```

The unsigned bundle contains the GUI and a server launcher tied to this checkout
and the current `uv` executable. Keep this checkout in place. Packaging changes
and distribution signing are outside this milestone.

## Native smoke capture

The optional `smoke-test` feature adds a driver that clicks New session, enters
text, sends it, approves the scripted tool, and captures native renderer pixels.
It uses the real connection actor and server. It exits after capture and fails
if the turn does not complete within 30 seconds. Normal builds omit this driver.

In one terminal:

```sh
ZETA_HOME="$(mktemp -d)" uv run python gui/tests/smoke_server.py --socket /tmp/zeta-smoke.sock
```

In another:

```sh
ZETA_HOME="$(mktemp -d)" ZETA_GUI_SMOKE_IMAGE=/tmp/zeta-smoke.png \
  cargo run --manifest-path gui/Cargo.toml --features smoke-test -- --socket /tmp/zeta-smoke.sock
```
