# Vendored gpui-base 0.6.1

This directory holds a temporary in-repo fork of the `gpui-base` crate at
version `0.6.1`, byte-identical to the crates.io release except for the
one change described below. `gui/Cargo.toml` points `[patch.crates-io.gpui-base]`
at `vendor/gpui-base` so both `zeta-gui` and its transitive `gpui-kit`
dependency compile against the patched crate.

## Why the fork exists (ZETA-129)

Two inline-code rendering bugs share one root cause in
`gpui-base/src/text/inline_flow.rs`:

* A1 — a backtick span of exactly 9 characters loses its last glyph
  (`task_kill` painted as `task_kil`, `websearch` as `websearc`,
  `thumbnail` as `thumbnai`). Present on old sessions, not test-induced.
* A2 — a code chip at a wrap boundary paints overflow glyphs onto the
  next line at a stale x-position, over existing text (`gpui::image`
  painted `gpui::im` with a bare `g` on the row below).

The mechanism: `InlineFlow::prepaint` prepaints the inner `StyledText`
that fills a text fragment with an available width of
`Definite(fragment_size.width - INLINE_CODE_PADDING * 2)`, which equals
`shape_line.width()` for a code chip. `StyledText::layout` threads that
into `shape_text`, which forwards it as `wrap_width` to
`line_layout_cache.layout_wrapped_line`. Under real macOS CoreText
metrics (`gpui-pre-macos::layout_line` applies `font_size.next_up()` on
the first font-run to break ligatures, and `typographic_bounds.width`
rounds independently of individual glyph advances), the cached
`unwrapped_layout.width` drifts a fraction of a pixel above the
`wrap_width` passed in. `compute_wrap_boundaries` iterates glyphs and
inserts a boundary the moment `width > wrap_width`; for a bare
identifier with no space, `last_candidate_ix` stays `None`, so the
boundary lands ON the last glyph. That glyph then paints
`line_height` below the fragment origin — invisible inside a chip that
sits on its own row (A1), or over the next prose line when the outer
flow already placed the chip on line 2 of a wrap (A2). Length 9 is
where the drift crosses the sub-pixel edge on the shipped
13px × 0.875 mono metrics; 8 and 10 both sit comfortably clear.

The outer flow (`layout_flow` and `push_text_wrap_fragments`) already
splits over-wide code spans at word / grapheme boundaries and hands the
inner `Inline` a portion that fits inside its fragment. The inner
`StyledText` has no legitimate reason to re-wrap. Removing the inner
`wrap_width` constraint is the minimum-surface fix.

## Edit 1 — functional `MaxContent` swap in `InlineFlow::prepaint`

```diff
--- a/src/text/inline_flow.rs
+++ b/src/text/inline_flow.rs
@@ -348,7 +348,11 @@ impl Element for InlineFlow {
                     element.prepaint_as_root(
                         bounds.origin + origin + point(padding, Pixels::ZERO),
                         size(
-                            AvailableSpace::Definite(fragment_size.width - padding * 2.),
+                            if std::env::var_os("ZETA_GUI_INLINE_FLOW_DEFINITE").is_some() {
+                                AvailableSpace::Definite(fragment_size.width - padding * 2.)
+                            } else {
+                                AvailableSpace::MaxContent
+                            },
                             AvailableSpace::Definite(fragment_size.height),
                         ),
                         window,
```

The height axis stays `Definite`, so line-height and vertical
positioning are unchanged. Every text fragment (code chip and regular
text alike) now returns its intrinsic shaped width to Taffy; the outer
flow already carries the wrap decisions those fragments need.
`ZETA_GUI_INLINE_FLOW_DEFINITE` reinstates the upstream shape so the
paired mutation arm (see below) can prove the CI guard catches the bug
when the fix is silenced.

## Edit 2 — test-only recorder in `InlineFlow::prepaint`

Immediately before `element.prepaint_as_root(...)` in the Text arm,
gated on `#[cfg(any(test, feature = "test-support"))]`:

```diff
+                    let width_available =
+                        if std::env::var_os("ZETA_GUI_INLINE_FLOW_DEFINITE").is_some() {
+                            AvailableSpace::Definite(fragment_size.width - padding * 2.)
+                        } else {
+                            AvailableSpace::MaxContent
+                        };
+                    #[cfg(any(test, feature = "test-support"))]
+                    {
+                        let probe_wrap_width = match width_available {
+                            AvailableSpace::Definite(x) => Some(x),
+                            _ => None,
+                        };
+                        let text_style_local = window.text_style();
+                        let probe_runs =
+                            text_runs(text.len(), &text_style_local, &highlights);
+                        if let Ok(lines) = window.text_system().shape_text(
+                            text.clone(), font_size, &probe_runs, probe_wrap_width, None,
+                        ) {
+                            let wrap_boundaries: usize =
+                                lines.iter().map(|l| l.wrap_boundaries().len()).sum();
+                            crate::zeta129_wrap_recorder::record(
+                                crate::zeta129_wrap_recorder::Sample {
+                                    text: text.clone(),
+                                    wrap_boundaries,
+                                    font_size,
+                                },
+                            );
+                        }
+                    }
                     element.prepaint_as_root(
                         bounds.origin + origin + point(padding, Pixels::ZERO),
-                        size(
-                            ...
-                            AvailableSpace::Definite(fragment_size.height),
-                        ),
+                        size(width_available, AvailableSpace::Definite(fragment_size.height)),
                         window, cx,
                     );
```

The recorder module lives at `src/zeta129_wrap_recorder.rs`, also
`#[cfg]`-gated so a release build compiles it out entirely. It exposes
`clear()`, `samples()`, and a private `record(...)`. Downstream tests
read the recorder via `gpui_kit::base::zeta129_wrap_recorder`.

The recorder is the mutation-killer. The invariant the fix establishes
is "an inline text fragment NEVER wraps inside its own fragment"; the
probe uses the SAME wrap_width the actual `prepaint_as_root` call uses
(via the env-gated `width_available`), so:

* Under the fix (env unset): `probe_wrap_width = None` → `shape_text`
  never inserts a wrap boundary → every recorded sample carries
  `wrap_boundaries == 0`.
* Under mutation (env set): `probe_wrap_width = Some(shape_line.width())`
  → `compute_wrap_boundaries` re-enters the sub-pixel drift path and
  drops a boundary onto the last glyph of the tripping ladder length.
  At least one sample carries `wrap_boundaries >= 1`.

The paired downstream check is `scan_inline_flow_recorder` in
`gui/src/smoke.rs`. It runs INSIDE the native smoke driver (real macOS
window, real CoreText metrics — the only place the sub-pixel drift
shows through headlessly-deterministic text shaping never reproduces
it) and panics if any recorded sample carries a non-zero wrap boundary
count. `gui-native-guards` invokes the smoke driver directly and thus
runs the recorder scan as a durable guard on every CI run.
`gui-native-guards-inline-flow-mutation` invokes the same smoke driver
with `ZETA_GUI_INLINE_FLOW_DEFINITE=1` and inverts the exit code — the
mutation MUST panic the recorder scan on the ladder shape (18px is the
demonstrated CoreText trip on macOS-latest CI). CI's `cargo
(macos-latest)` job runs both targets.

## Poison-canary (`ZETA_GUI_INLINE_FLOW_DEFINITE`)

The env is checked in TWO places:
1. `InlineFlow::prepaint`'s `width_available` binding — chooses between
   `MaxContent` (fix) and the upstream `Definite(...)` (mutation).
2. The test-only recorder probe — mirrors the same choice so the sample
   captures what actually happens in `prepaint_as_root`.

Both checks read the same env; setting `ZETA_GUI_INLINE_FLOW_DEFINITE=1`
silences the fix and reproduces the bug end-to-end — the CI mutation
target inverts the exit code so a green run FAILS CI, catching a
silent revert (a stray `git checkout`, a merge conflict resolved the
wrong way) rather than letting it ship.

## Deviation from byte-identical crates.io content

Two deliberately-scoped deviations from the crates.io 0.6.1 release:
1. The `MaxContent` swap (the functional fix).
2. The `#[cfg]`-gated recorder + probe (test-only, compiles out of
   release builds; needed to make the mutation arm CI-visible per
   ZETA-129 kickoff constraint 5).

Every other file, including `Cargo.toml`, `Cargo.toml.orig`,
`Cargo.lock`, `LICENSE-APACHE`, and the entire `src/` and `tests/`
trees, is byte-identical to the crates.io content.

## Unfork condition

Delete this directory and the `[patch.crates-io.gpui-base]` entry in
`gui/Cargo.toml` once a released `gpui-kit` version depends on a
`gpui-base` release that carries the equivalent fix (in whichever
shape — `MaxContent`, a wrap tolerance in
`compute_wrap_boundaries`, or a different upstream call site).

The follow-up is tracked in `docs/design.md` under deferred / open
follow-ups, keyed to ZETA-129.
