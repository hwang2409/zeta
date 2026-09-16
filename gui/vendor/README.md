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

## The diff

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

## Poison-canary (`ZETA_GUI_INLINE_FLOW_DEFINITE`)

Per ZETA-129 kickoff constraint 5, guard proof must be CI-visible. The
env check above reinstates the upstream `Definite(...)` shape when
`ZETA_GUI_INLINE_FLOW_DEFINITE` is set — the `Makefile` target
`gui-native-guards-inline-flow-mutation` runs the smoke driver with
that env active and expects the native pixel-gutter guard (or the
`code_ladder` shape scan) to panic, mirroring the shape of the
existing `gui-native-guards-mutation` poison-canary. The default
(env unset) path stays clean and CI runs the standard
`gui-native-guards` target.

The env-mutation arm is a second, deliberately-scoped deviation from
byte-identical crates.io content. The functional fix is the
`MaxContent` swap; the env branch exists solely so a peer refactor that
silently reverts the swap (a stray `git checkout`, a merge conflict
resolved the wrong way) is caught by CI rather than shipped.

## Unfork condition

Delete this directory and the `[patch.crates-io.gpui-base]` entry in
`gui/Cargo.toml` once a released `gpui-kit` version depends on a
`gpui-base` release that carries the equivalent fix (in whichever
shape — `MaxContent`, a wrap tolerance in
`compute_wrap_boundaries`, or a different upstream call site).

The follow-up is tracked in `docs/design.md` under deferred / open
follow-ups, keyed to ZETA-129.
