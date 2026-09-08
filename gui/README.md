# zeta gui

the native gui is a separate rust crate. it speaks the version 1.0 `zeta serve`
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
