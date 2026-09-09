# Hosts without Apple's offline Metal compiler (Command Line Tools only)
# must use gpui's runtime-shaders feature; full Xcode installs can use the
# default precompiled shader path.
METAL := $(shell xcrun -sdk macosx -f metal 2>/dev/null)
GUI_FEATURES := $(if $(METAL),,--features runtime-shaders)

.PHONY: gui
gui:
	cargo run --manifest-path gui/Cargo.toml $(GUI_FEATURES)

.PHONY: gui-app
gui-app:
	cargo build --manifest-path gui/Cargo.toml --features 'gui runtime-shaders'
	python3 gui/package_app.py
