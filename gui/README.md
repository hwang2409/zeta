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
