# GPUI Kit enables runtime shaders, including on Command Line Tools hosts.
.PHONY: gui
gui:
	cargo run --manifest-path gui/Cargo.toml

.PHONY: gui-app
gui-app:
	RUSTFLAGS="$(RUSTFLAGS) --remap-path-prefix=$(CURDIR)=." cargo build --release --manifest-path gui/Cargo.toml
	python3 gui/package_app.py
