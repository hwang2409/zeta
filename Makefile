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

# ZETA-129 poison-canary: silences the `InlineFlow::prepaint` MaxContent
# fix from `gui/vendor/gpui-base/` by setting `ZETA_GUI_INLINE_FLOW_DEFINITE=1`,
# reinstating the upstream `Definite(...)` shape that drops the last
# glyph of any 9-char code chip and paints its overflow onto the next
# line at a stale x-position. The A1 phantom lands inside the content
# column (never in the pixel-gutter scan band) and coincides with the
# scrollbar x-band at ladder positions, so the native pixel-gutter guard
# CANNOT resolve this bug class. The headless mutation-killer is
# `zeta129_inline_code_chip_ladder_paints_one_widening_chip_per_length`
# in `gui/src/tests.rs`: it drives a 1..=16 backtick ladder through
# `TextView::markdown` and asserts on painted_quads (chip count, chip
# widths, and below-line phantom paint). Running that test with the env
# set MUST FAIL — a green run here means the fix silently reverted (a
# rebase picked the wrong side, a peer refactor threw the env branch
# away) or the assertion loosened past the mutation.
.PHONY: gui-native-guards-inline-flow-mutation
gui-native-guards-inline-flow-mutation:
	@set -eu; \
	if ZETA_GUI_INLINE_FLOW_DEFINITE=1 \
	  cargo test --manifest-path gui/Cargo.toml \
	    zeta129_inline_flow_never_wraps_a_text_fragment_inside_its_own_fragment \
	    -- --nocapture; then \
		echo "FAIL: recorder guard PASSED with ZETA_GUI_INLINE_FLOW_DEFINITE=1 (fix silenced)"; \
		exit 1; \
	else \
		echo "OK: recorder guard FAILED under the mutation as required"; \
	fi

# Poison-canary: the native pixel-gutter guard MUST still fail when a
# real prose-column overflow is introduced. The `NATIVE_GUARD_FORCE_TEXT_WIDTH`
# knob widens the assistant TextView beyond its prose column so glyphs
# actually escape into the gutter. This target inverts the exit code —
# the guard is expected to panic. A green run here would prove the guard
# has been silenced (by e.g. a masking regression); we exit non-zero.
.PHONY: gui-native-guards-mutation
gui-native-guards-mutation:
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
	if ZETA_HOME="$$tmp_dir/home" ZETA_GUI_NATIVE_GUARDS=1 ZETA_GUI_NATIVE_GUARDS_FORCE_TEXT_WIDTH=1 \
	  cargo run --manifest-path gui/Cargo.toml --features smoke-test -- --socket "$$socket"; then \
		echo "FAIL: guard mutation PASSED (guard has been silenced — masking regressed)"; \
		exit 1; \
	else \
		echo "OK: guard mutation FAILED the guard as required"; \
	fi
