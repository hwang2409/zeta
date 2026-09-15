# GPUI Kit enables runtime shaders, including on Command Line Tools hosts.
.PHONY: gui
gui:
	cargo run --manifest-path gui/Cargo.toml

.PHONY: gui-app
gui-app:
	RUSTFLAGS="$(RUSTFLAGS) --remap-path-prefix=$(CURDIR)=." cargo build --release --manifest-path gui/Cargo.toml
	python3 gui/package_app.py

.PHONY: gui-native-guards
gui-native-guards:
	@set -eu; \
	tmp_dir=$$(mktemp -d); \
	mkdir "$$tmp_dir/home"; \
	socket="$$tmp_dir/smoke.sock"; \
	server_log="$$tmp_dir/server.log"; \
	ZETA_HOME="$$tmp_dir/home" uv run --frozen python gui/tests/smoke_server.py --socket "$$socket" >"$$server_log" 2>&1 & \
	server_pid=$$!; \
	trap 'kill "$$server_pid" 2>/dev/null || true; wait "$$server_pid" 2>/dev/null || true; rm -rf "$$tmp_dir"' EXIT INT TERM; \
	for attempt in $$(seq 1 100); do \
		if grep -q '^ready$$' "$$server_log"; then break; fi; \
		if ! kill -0 "$$server_pid" 2>/dev/null; then cat "$$server_log"; exit 1; fi; \
		sleep 0.1; \
	done; \
	grep -q '^ready$$' "$$server_log" || { cat "$$server_log"; exit 1; }; \
	ZETA_HOME="$$tmp_dir/home" ZETA_GUI_NATIVE_GUARDS=1 cargo run --manifest-path gui/Cargo.toml --features smoke-test -- --socket "$$socket"
