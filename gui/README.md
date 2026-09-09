# zeta gui

the native gui is a separate rust crate. it speaks the version 1.x `zeta serve`
protocol documented in [`../docs/serve-protocol.md`](../docs/serve-protocol.md).

run it with the default zeta home:

```sh
cargo run --manifest-path gui/Cargo.toml
```

connect to an existing server:

```sh
cargo run --manifest-path gui/Cargo.toml -- --socket /path/to/serve.sock
```

set `ZETA_BIN` when the `zeta` executable is not on `PATH`. set `ZETA_HOME` to
use an isolated home for local verification.

shift-enter inserts a newline. enter sends the composer. escape aborts the
active turn. when an approval modal is open, enter approves and escape denies.

assistant messages render headings, lists, emphasis, inline code, and syntax-colored
code fences. user messages remain plain text. long code lines scroll inside their
fence; syntax themes never paint a background.

tool receipts start collapsed. click a receipt, or focus it with tab and press
enter, to show its last 20 output lines. tails also have a 16,000-character cap;
cut output has an `output truncated` header. delegated receipts keep their agent
identity and use an indented border. background agents stay marked working until
their terminal receipt arrives.

the bottom bar shows the model, token count, and cache hit rate from `status`.
it refreshes at turn boundaries, after aborts, and when selecting a session.
usage notifications do not change displayed metrics during streaming. missing
values show an em dash. token counts include input, cache reads, cache writes,
and output; cache rate divides reads by all input tokens.

the window follows system light/dark appearance, including the composer and code
colors. body and syntax text meet AA contrast against their surfaces. no motion
is needed to operate disclosures.

local validation on hosts without the offline Metal compiler:

```sh
cargo fmt --manifest-path gui/Cargo.toml --all -- --check
cargo clippy --manifest-path gui/Cargo.toml --features 'gui runtime-shaders' --all-targets -- -D warnings
cargo test --manifest-path gui/Cargo.toml --features runtime-shaders
```

appearance tests force both palettes. GPUI's platform appearance simulator is
private, so those tests do not simulate macOS appearance notifications.


## Session tools (protocol 1.1)

The sidebar lists branch heads, with indentation at each divergence and a blue
rule on the current branch. Labels stay on one line; hover to read the label.
Click a branch to switch. `fork here` on a user message forks at that message.
These actions use the core conversation tree. They do not restore workspace
snapshots. Resume and branch switches load the persisted transcript.

Settings apply only to the active session and persist with it. The model list
comes from the server's built-in provider catalog, including the current model.
Approval mode changes the default decision (`ask`, `allow`, or `deny`); explicit
tool rules still apply. In settings, tab selects the field, arrows select its
value, enter applies, and escape closes. Settings and branch changes wait until
foreground turns, pending approvals, and background agents finish.

Paste an image with cmd-v or drop image files onto the composer. PNG, JPEG, GIF,
and WebP are supported, with at most four images totaling 512 KiB per message.
Click a draft chip to remove it. The server saves each image under the session
directory and sends it through the existing provider image path. The transcript
shows the filename and byte count, without an image preview. Rejected sends
retain the draft and show the error.

The GUI starts with a 1.0 hello and advertises `client_version: "1.1"`. A 1.0
server ignores that field; the GUI hides session tools and image input. A 1.1
server returns the negotiated version. Old clients receive 1.0 and keep working.

## Developer Mac app

```sh
make gui-app
open dist/Zeta.app
```

This builds `dist/Zeta.app` with runtime shaders, even on hosts without Apple's
Metal compiler. The unsigned bundle contains the GUI and a server launcher tied
to this checkout and the current `uv` executable. Keep this checkout in place.
The launcher honors `ZETA_BIN`, `ZETA_HOME`, and `--socket`; it does not package a
Python runtime. It is intended for the local development loop.

For isolated verification against an existing server:

```sh
ZETA_HOME=/path/to/isolated-home dist/Zeta.app/Contents/MacOS/zeta --socket /path/to/isolated.sock
```

The release/default-shader build path requires full Xcode with the Metal
compiler. A distributable build would package the Python server first, then sign
nested executables and the finished app bundle before notarization. This target
does not sign, notarize, or add CI packaging jobs.
