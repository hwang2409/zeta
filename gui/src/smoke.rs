//! Opt-in smoke driver: real input events, real worker/socket, native Metal pixels.
use super::*;
use gpui::{px, Pixels};
use gpui_kit::test::TestWindowExt;

const NATIVE_GUARD_COLOR_THRESHOLD: u8 = 10;
const NATIVE_GUARD_MIN_CONSECUTIVE: usize = 2;
const NATIVE_GUARD_SCROLLBAR_WIDTH: Pixels = px(8.);
/// Composer chrome height reserved BELOW the transcript scan y-range.
/// Reads `theme::composer_chrome_reserve()` — the SAME function the
/// composer render sums from its named children — so a bump to any
/// composer chrome constant propagates without a paired smoke-side
/// edit (ZETA-135 review r1 finding 5).
fn native_guard_composer_height() -> Pixels {
    theme::composer_chrome_reserve()
}
/// Backticked identifier length that exceeds every tested column at every
/// tested base font size. `TRANSCRIPT_MAX_WIDTH` caps the widest column at
/// 1024px; at 11px the mono advance is roughly `11 * MONO_CH_ADVANCE`
/// (~6.82px), so 192 chars renders ~1310px — comfortably wider than the
/// cap. At 18px the same token renders ~2143px, well past even the
/// narrower 0.7-viewport column. The outer flow's
/// `push_text_wrap_fragments` grapheme-splits the token onto multiple
/// lines; the guard's job is to confirm no split fragment paints past
/// `content_right` (`scan_native_gutter`) and no inner `Inline` re-wraps
/// inside its own fragment (`scan_inline_flow_recorder`).
const OVER_WIDE_CODE_TOKEN_LEN: usize = 192;

/// One entry in the native guard matrix. `prose_only` selects between
/// two transcript layouts and two scan gutters:
///
/// * `prose_only = false` — the mixed transcript with tool receipts and
///   a scrollbar-triggering row set (see `native_guard_transcript`).
///   The pixel scan starts at the wider `TRANSCRIPT_MAX_WIDTH` content
///   edge because tool rows legitimately paint out to that cap. Prose
///   overshoots between the prose edge and the tool edge are
///   under-scanned here — the round-3 pixel-gutter finding.
/// * `prose_only = true` — an assistant-only transcript (see
///   `native_guard_prose_only_transcript`). The scan starts at the prose
///   body edge inside the centered unified frame.
struct GuardShape {
    name: &'static str,
    source: String,
    prose_only: bool,
    mutation_probe: bool,
}

fn native_guard_shapes() -> Vec<GuardShape> {
    let over_wide_ident = "a".repeat(OVER_WIDE_CODE_TOKEN_LEN);
    vec![
    GuardShape {
        name: "wedge",
        source: "2. `zeta serve` session hardening — half-written session dirs \
         (`conversation.jsonl` without `meta.json`) wedge status/list. Atomic dir \
         creation via `meta.json` tmp+rename.\n3. Follow-up work with additional \
         wrapping to exercise the hanging indent so the paragraph reliably breaks \
         onto a continuation line even at 2204px.".into(),
        prose_only: false,
        mutation_probe: false,
    },
    GuardShape {
        name: "adjacent",
        source: "1. Outer numbered item with plenty of prose to force wrapping onto multiple \
         continuation lines at every picker step.\n2. Second outer numbered item \
         to prove the second sibling wraps in the same column geometry as the first \
         with more filler prose here now.".into(),
        prose_only: false,
        mutation_probe: false,
    },
    GuardShape {
        name: "nested",
        source: "1. Outer item with room to spare.\n   - Nested bullet A that itself carries \
         enough hanging-indent text to force wrap boundaries near the prose cap at \
         every base picker step.\n   - Nested bullet B with more prose — deeper \
         nesting stays inside the same column even when the marker indent has \
         consumed a few characters.".into(),
        prose_only: false,
        mutation_probe: false,
    },
    GuardShape {
        name: "long_token",
        source: "Prose leading up to a very long unbroken token that the wrap engine cannot \
         break: \
         supercalifragilisticexpialidocious_but_much_longer_than_any_column_should_ever_be_aaaaaaaaaaaaaaaaaaaa \
         and then some trailing prose after it.".into(),
        prose_only: false,
        mutation_probe: false,
    },
    // ZETA-129: exercise inline-code chips of length 1..16 in a bullet list.
    // The upstream `InlineFlow::prepaint` bug drops the last glyph of any
    // 9-char chip (locally) and paints overflow glyphs onto the next line
    // at a stale x-position — the drift threshold is CoreText-metric
    // dependent, so the ladder runs past 12 to guarantee the mutation
    // crosses it on every CI font resolution. A1 / A2 detection is owned
    // by `scan_inline_flow_recorder` below (see the vendored
    // `zeta129_wrap_recorder` in `gui/vendor/gpui-base/`), which runs
    // AFTER this shape's native paint and asserts every inner text
    // fragment recorded zero wrap boundaries. The headless
    // `zeta129_inline_code_chip_ladder_structure` in `tests.rs` is a
    // chip-structure regression cover only — it inspects background
    // quads and cannot see the phantom-glyph paint (which is a text
    // sprite). Prose-only so the pixel scan runs against the prose body
    // edge inside the unified frame.
    GuardShape {
        name: "code_ladder",
        source: "- `a` len=1\n- `ab` len=2\n- `abc` len=3\n- `abcd` len=4\n- `abcde` len=5\n\
         - `abcdef` len=6\n- `abcdefg` len=7\n- `abcdefgh` len=8\n- `abcdefghi` len=9\n\
         - `abcdefghij` len=10\n- `abcdefghijk` len=11\n- `abcdefghijkl` len=12\n\
         - `abcdefghijklm` len=13\n- `abcdefghijklmn` len=14\n\
         - `abcdefghijklmno` len=15\n- `abcdefghijklmnop` len=16".into(),
        prose_only: true,
        mutation_probe: false,
    },
    // ZETA-129 round 2: a backticked identifier wider than
    // `TRANSCRIPT_MAX_WIDTH` at every picker font size (see
    // `OVER_WIDE_CODE_TOKEN_LEN`). The MaxContent fix's main regression
    // condition is a code span that the outer flow must break at
    // grapheme boundaries: the outer `push_text_wrap_fragments` splits
    // the identifier into multiple `Inline` fragments; each fragment's
    // inner `StyledText` must NOT re-wrap. `scan_native_gutter` asserts
    // no split fragment paints past the prose body edge;
    // `scan_inline_flow_recorder` asserts each inner fragment records
    // zero wrap boundaries. Under `ZETA_GUI_INLINE_FLOW_DEFINITE=1` the
    // Definite width axis re-enters shape_text and CoreText drift can
    // trip either scan depending on the fragment landing.
    GuardShape {
        name: "code_wide_token",
        source: format!(
            "Prose leading up to a code span wider than every tested column: \
             `{over_wide_ident}` and then trailing prose after it."
        ),
        prose_only: true,
        mutation_probe: false,
    },
    GuardShape {
        name: "prose_edge_probe",
        source: format!("prose edge probe {}", "edge ".repeat(220)),
        prose_only: true,
        mutation_probe: true,
    },
    ]
}

fn native_guard_enabled() -> bool {
    env::var_os("ZETA_GUI_NATIVE_GUARDS").as_deref() == Some(std::ffi::OsStr::new("1"))
}

fn rgb8(color: gpui::Hsla) -> [u8; 3] {
    let color = color.to_rgb();
    [
        (color.r * 255.).round() as u8,
        (color.g * 255.).round() as u8,
        (color.b * 255.).round() as u8,
    ]
}

/// Pixel rectangle in image coordinates. Used to punch holes in the scan
/// band for painted overlays (currently just the transcript scrollbar
/// thumb) that render inside the column's padding zone. Every hole
/// records the ACTUAL painted rect — no blanket tolerance on
/// `content_right` — so a real glyph escape adjacent to the overlay still
/// trips the guard.
#[derive(Debug, Clone, Copy)]
struct PixelRect {
    x_start: u32,
    x_end: u32,
    y_start: u32,
    y_end: u32,
}

impl PixelRect {
    fn contains(&self, x: u32, y: u32) -> bool {
        x >= self.x_start && x < self.x_end && y >= self.y_start && y < self.y_end
    }
}

/// Return the first x range with at least two adjacent pixels that differ
/// from the active canvas token. One isolated anti-aliased pixel is noise;
/// adjacent pixels are the minimum evidence for an escaped glyph stroke.
/// Pixels inside any `exclude` rect are treated as background — this is
/// how the scan skips the scrollbar-thumb overlay without loosening the
/// content-right coordinate for every other paint.
fn escaped_glyph_range(
    image: &image::RgbaImage,
    x_start: u32,
    x_end: u32,
    y_start: u32,
    y_end: u32,
    background: [u8; 3],
    exclude: &[PixelRect],
) -> Option<(u32, u32)> {
    for y in y_start..y_end {
        let mut run_start = None;
        for x in x_start..x_end {
            let masked = exclude.iter().any(|rect| rect.contains(x, y));
            let over_threshold = if masked {
                false
            } else {
                let pixel = image.get_pixel(x, y).0;
                pixel[..3].iter().zip(background).any(|(actual, expected)| {
                    actual.abs_diff(expected) >= NATIVE_GUARD_COLOR_THRESHOLD
                })
            };
            if over_threshold {
                run_start.get_or_insert(x);
            } else if let Some(start) = run_start.take() {
                if x - start >= NATIVE_GUARD_MIN_CONSECUTIVE as u32 {
                    return Some((start, x - 1));
                }
            }
        }
        if let Some(start) = run_start {
            if x_end - start >= NATIVE_GUARD_MIN_CONSECUTIVE as u32 {
                return Some((start, x_end - 1));
            }
        }
    }
    None
}

/// Identify the transcript scrollbar-thumb rectangles the current frame
/// painted, so the pixel-gutter scan can skip them without loosening its
/// content-right coordinate. The scrollbar THUMB is proportional to the
/// viewport / content ratio — at the taller 922x610 viewport it can be
/// only a few dozen pixels tall — so this filter identifies it by
/// **X-BAND** signature: left edge at (or immediately at) the gutter
/// start AND width matching the `SCROLLBAR_THUMB_WIDTH` token
/// (± 2px slack for subpixel rounding). Height is unbounded so a short
/// proportional thumb is still masked.
///
/// Real glyph escapes are unmaskable by construction: glyphs paint as
/// text sprites (a separate scene primitive), NOT as quads, so this
/// filter cannot accidentally cover a real prose overshoot even if a
/// future refactor were to add a narrow chrome quad in the same x-band.
/// The `gui-native-guards-mutation` poison-canary pins that
/// non-masking property in CI.
fn scrollbar_scan_masks(window: &Window, gutter_x_start: u32, gutter_x_end: u32) -> Vec<PixelRect> {
    let scale = window.scale_factor();
    let scrollbar_width_scaled = f32::from(theme::SCROLLBAR_THUMB_WIDTH) * scale;
    let min_width_scaled = (scrollbar_width_scaled - 1.0).max(0.0);
    let max_width_scaled = scrollbar_width_scaled + 2.0;
    window
        .painted_quads()
        .into_iter()
        .filter_map(|quad| {
            let width_scaled = quad.bounds.size.width.0;
            if width_scaled < min_width_scaled || width_scaled > max_width_scaled {
                return None;
            }
            let x_start = quad.bounds.origin.x.0.floor() as u32;
            let x_end = x_start + width_scaled.ceil() as u32;
            let y_start = quad.bounds.origin.y.0.floor() as u32;
            let y_end = y_start + quad.bounds.size.height.0.ceil() as u32;
            // Left edge must sit at or PAST the gutter start — narrow
            // chrome painted inside the content area (icons, focus
            // rings, chip borders) is never at content_right, so we
            // never mask it. Allow 1px of subpixel slack on the left
            // to accept a thumb that landed just before the ceil-ed
            // gutter start.
            if x_start + 1 < gutter_x_start {
                return None;
            }
            if x_end <= gutter_x_start || x_start >= gutter_x_end {
                return None;
            }
            Some(PixelRect {
                x_start,
                x_end,
                y_start,
                y_end,
            })
        })
        .collect()
}

fn scan_native_gutter(
    image: &image::RgbaImage,
    window: &Window,
    font_size: Pixels,
    shape: &str,
    prose_only: bool,
    achieved_width: u32,
    achieved_height: u32,
) {
    let scale = window.scale_factor();
    let window_width = f32::from(window.bounds().size.width);
    let main_left = f32::from(theme::SIDEBAR_WIDTH);
    let main_width = (window_width - main_left).max(0.);
    // `content_right` is derived from the same unified frame, gutter, and
    // body cap the renderer uses:
    //
    // * MIXED transcript rows (tool receipts + assistant prose) — the
    //   scan uses `TRANSCRIPT_MAX_WIDTH`. Tool receipts and fenced
    //   error blocks legitimately paint out to that wider cap; a
    //   narrower gutter would flag every receipt paint at a wide
    //   centered viewport as a glyph escape.
    // * PROSE-ONLY transcript rows — the scan starts at the prose body edge
    //   inside the centered unified frame. This catches a prose glyph that
    //   escapes the body but stays inside the frame's right padding.
    //
    let frame_width = f32::from(theme::TRANSCRIPT_MAX_WIDTH).min(main_width);
    let frame_left = main_left + (main_width - frame_width) / 2.;
    let body_cap = if prose_only {
        f32::from(theme::prose_body_max_width(font_size))
    } else {
        f32::from(theme::wide_body_max_width())
    };
    let available_body =
        (frame_width - 2. * theme::PROSE_ROW_PADDING_X - f32::from(theme::LEADING_GUTTER_WIDTH))
            .max(0.);
    let body_width = body_cap.min(available_body);
    let content_right = frame_left
        + theme::PROSE_ROW_PADDING_X
        + f32::from(theme::LEADING_GUTTER_WIDTH)
        + body_width;
    let x_start = (content_right * scale).ceil() as u32;
    let x_end = image
        .width()
        .saturating_sub((f32::from(NATIVE_GUARD_SCROLLBAR_WIDTH) * scale).ceil() as u32);
    // Larger picker sizes can make the header's content-driven height exceed
    // its 44px minimum. Leave the scan below that dynamic edge.
    let y_start = ((f32::from(theme::HEADER_BAND1_MIN_HEIGHT) + 8.) * scale).ceil() as u32;
    // The live composer stays visible during the guard. Reserve its fixed
    // chrome height (shared `theme::composer_chrome_reserve()`) below the
    // scan so composer paint (bg fill, ZETA-135 label chip, footer) never
    // looks like a transcript glyph escape.
    let y_end = image
        .height()
        .saturating_sub((f32::from(native_guard_composer_height()) * scale).ceil() as u32 + 1);
    let background = rgb8(theme::palette::canvas());
    let scrollbar_masks = scrollbar_scan_masks(window, x_start, x_end);
    if let Some((escape_start, escape_end)) = escaped_glyph_range(
        image,
        x_start,
        x_end,
        y_start,
        y_end,
        background,
        &scrollbar_masks,
    ) {
        panic!(
            "native pixel gutter guard failed: shape={shape} size={font_size:?} \
             prose_only={prose_only} x_range={escape_start}..={escape_end} \
             gutter={x_start}..{x_end} window_width={window_width} scale={scale} \
             frame_width={frame_width} body_width={body_width} content_right={content_right} \
             y_range={y_start}..{y_end} background={background:?} \
             masked={scrollbar_masks:?}"
        );
    }
    println!(
        "NATIVE-GUARD-PASS: shape={shape} size={font_size:?} prose_only={prose_only} \
         viewport={achieved_width}x{achieved_height} gutter={x_start}..{x_end} \
         scrollbar_masks={} rects={scrollbar_masks:?}",
        scrollbar_masks.len()
    );
}

/// ZETA-129 recorder scan. `InlineFlow::prepaint` in the vendored
/// `gpui-base` pushes a
/// `gpui_kit::base::zeta129_wrap_recorder::Sample` per inner text
/// fragment it renders, using the SAME `wrap_width` its actual
/// `prepaint_as_root` call uses (`MaxContent` under the fix,
/// `Definite(fragment_size.width - padding * 2.)` under
/// `ZETA_GUI_INLINE_FLOW_DEFINITE=1`). The fix's invariant is "an
/// inline text fragment NEVER wraps inside its own fragment"; this
/// scan reads the recorder after each native render and panics if any
/// sample carries a non-zero wrap boundary count.
///
/// Runs on the NATIVE macOS text system so CoreText's real drift shows
/// through. The paired `gui-native-guards-inline-flow-mutation`
/// Makefile target invokes `make gui-native-guards` with
/// `ZETA_GUI_INLINE_FLOW_DEFINITE=1` and inverts the exit code — the
/// mutation MUST trip this scan. Pinned CI trip evidence:
/// shape=`wedge`, size=13px, sample text=`meta.json`, wrap_boundaries=1
/// — the audit's length-9 code chip on the 13px × 0.875 mono metrics
/// crossing the sub-pixel `shape_line.width()` boundary.
fn scan_inline_flow_recorder(shape: &str, font_size: Pixels, achieved: (u32, u32)) {
    let samples = gpui_kit::base::zeta129_wrap_recorder::samples();
    for sample in &samples {
        if sample.wrap_boundaries > 0 {
            panic!(
                "ZETA-129 recorder guard tripped: shape={shape} size={font_size:?} \
                 viewport={}x{} text={:?} font_size={:?} wrap_boundaries={} — the \
                 inner `Inline`'s `shape_text` inserted a wrap boundary at this text \
                 span, violating the ZETA-129 invariant that an inline text fragment \
                 never wraps inside its own fragment. If this fired under \
                 `ZETA_GUI_INLINE_FLOW_DEFINITE=1` (the poison-canary mutation), it \
                 is the expected failure — the paired \
                 `gui-native-guards-inline-flow-mutation` Makefile target inverts \
                 the exit code. If it fired without the env var, the vendored \
                 `MaxContent` fix has silently regressed.",
                achieved.0, achieved.1, sample.text, sample.font_size, sample.wrap_boundaries,
            );
        }
    }
}

fn native_guard_viewports(window: &Window, cx: &App) -> [gpui::Size<Pixels>; 2] {
    let display_size = window
        .display(cx)
        .map(|display| display.visible_bounds().size)
        .unwrap_or(window.bounds().size);
    let maximum = gpui::size(
        display_size.width.min(px(2204.)),
        display_size.height.min(px(1608.)),
    );
    [
        gpui::size(maximum.width * 0.7, maximum.height * 0.7),
        gpui::size(maximum.width * 0.9, maximum.height * 0.9),
    ]
}

async fn run_native_wrap_guards(view: Entity<ZetaView>, cx: &mut gpui::AsyncWindowContext) {
    let viewports = cx
        .update(|window, cx| native_guard_viewports(window, cx))
        .expect("native guard window remains open");
    let mutation = env::var_os(row_text::sel::NATIVE_GUARD_FORCE_TEXT_WIDTH_ENV).is_some();
    let font_sizes = if mutation {
        vec![theme::DEFAULT_FONT_SIZE]
    } else {
        vec![
            px(theme::MIN_FONT_SIZE_PX),
            theme::DEFAULT_FONT_SIZE,
            px(theme::MAX_FONT_SIZE_PX),
        ]
    };
    let shapes: Vec<_> = native_guard_shapes()
        .into_iter()
        .filter(|shape| shape.mutation_probe == mutation)
        .collect();
    let mut achieved_viewports: Vec<(u32, u32)> = Vec::new();
    let mut matrix_entries = 0;
    let mut appearance = theme::Appearance::default();
    cx.update(|_, cx| {
        view.update(cx, |view, cx| {
            view.state.streaming = false;
            view.pending_command = false;
            cx.notify();
        });
    })
    .expect("native guard window remains open");
    for requested in viewports {
        cx.update(|window, _| window.resize(requested))
            .expect("native guard window remains open");
        // macOS delivers setContentSize_ on the foreground executor. Yield so
        // the capture observes the native size instead of the prior frame.
        cx.background_executor()
            .timer(Duration::from_millis(100))
            .await;
        let achieved = cx
            .update(|window, cx| {
                window.bounds_changed(cx);
                window.render_frame(cx);
                let image = window
                    .render_to_image()
                    .expect("native renderer viewport capture");
                (image.width(), image.height())
            })
            .expect("native guard window remains open");
        assert!(
            achieved.0 > 0 && achieved.1 > 0,
            "native guard produced an empty capture for requested viewport {requested:?}"
        );
        if let Some(previous) = achieved_viewports
            .iter()
            .find(|previous| **previous == achieved)
        {
            eprintln!(
                "NATIVE-GUARD-WARN: requested viewport {requested:?} achieved duplicate \
                 capture {}x{}; skipping duplicate matrix entry (first was {}x{})",
                achieved.0, achieved.1, previous.0, previous.1
            );
            continue;
        }
        achieved_viewports.push(achieved);
        println!(
            "NATIVE-GUARD-VIEWPORT: requested={requested:?} achieved={}x{}",
            achieved.0, achieved.1
        );
        for font_size in font_sizes {
            appearance.font_size = font_size;
            cx.update(|_, cx| theme::apply_with(cx, &appearance))
                .expect("native guard window remains open");
            for entry in &shapes {
                let shape = entry.name;
                let source = entry.source.as_str();
                let prose_only = entry.prose_only;
                // ZETA-129: clear the wrap recorder before each shape so
                // the post-paint scan reads samples from THIS render only.
                // The recorder is populated by the vendored
                // `InlineFlow::prepaint` for every inner text fragment;
                // `scan_inline_flow_recorder` fails the guard if any
                // sample carries a non-zero wrap boundary count.
                gpui_kit::base::zeta129_wrap_recorder::clear();
                cx.update(|window, cx| {
                    view.update(cx, |view, cx| {
                        view.state.connection = ConnectionState::Connected;
                        // r3 finding 7: seed grouped + ungrouped tool
                        // receipts alongside the wrap-prose so glyph
                        // escapes in the tool-receipt paint paths ride
                        // the same pixel gutter. The taller-transcript
                        // shape entry keeps the SCROLLBAR-PRESENT path
                        // exercised at 18px — `scrollbar_scan_masks`
                        // punches ONLY the scrollbar rect out of the
                        // scan band, so a real overshoot adjacent to
                        // the scrollbar (or on any other row) still
                        // trips the guard cleanly.
                        //
                        // Prose-only shapes render an assistant-only
                        // transcript so the whole content area is prose;
                        // the pixel scan then uses the prose body edge
                        // inside the unified frame.
                        view.state.transcript = if prose_only {
                            native_guard_prose_only_transcript(source)
                        } else {
                            native_guard_transcript(source)
                        };
                        let count = view.state.transcript.len();
                        view.transcript
                            .update(cx, |scroll, cx| scroll.reset(count, cx));
                        cx.notify();
                    });
                    window.render_frame(cx);
                    let image = window
                        .render_to_image()
                        .expect("native renderer capture for pixel guard");
                    assert_eq!(
                        (image.width(), image.height()),
                        achieved,
                        "native guard viewport changed during matrix for requested \
                             {requested:?}"
                    );
                    if let Some(dir) = env::var_os("ZETA_GUI_NATIVE_GUARDS_CAPTURE_DIR") {
                        let path = PathBuf::from(dir).join(format!(
                            "{shape}-{}-{}x{}.png",
                            f32::from(font_size),
                            achieved.0,
                            achieved.1
                        ));
                        image.save(path).expect("save native guard capture");
                    }
                    scan_native_gutter(
                        &image, window, font_size, shape, prose_only, achieved.0, achieved.1,
                    );
                    scan_inline_flow_recorder(shape, font_size, achieved);
                })
                .expect("native guard window remains open");
                matrix_entries += 1;
            }
        }
    }
    assert_eq!(
        achieved_viewports.len(),
        2,
        "native guard requires two distinct achieved viewports, got {achieved_viewports:?}"
    );
    assert_ne!(
        achieved_viewports[0].0, achieved_viewports[1].0,
        "native guard requires two distinct achieved viewport widths"
    );
    let achieved_list = achieved_viewports
        .iter()
        .map(|(width, height)| format!("{width}x{height}"))
        .collect::<Vec<_>>()
        .join(",");
    println!("NATIVE-GUARD-PASS: matrix={matrix_entries} achieved_viewports={achieved_list}");
}

/// Prose-only transcript for shapes that need to be pixel-scanned
/// against the narrower prose content edge (see the round-3 pixel
/// gutter finding). Contains ONLY the assistant row so no tool receipt
/// (which legitimately paints out to `TRANSCRIPT_MAX_WIDTH`) lands in
/// the y-band the prose gutter scans. The mixed transcript below stays
/// the default for non-prose shapes.
fn native_guard_prose_only_transcript(source: &str) -> Vec<TranscriptEntry> {
    use zeta_gui::state::TranscriptEntry;
    vec![TranscriptEntry::Assistant(source.into())]
}

/// Mixed transcript for shapes whose escapes we want to trip against
/// the wider tool-receipt gutter. The assistant row carries the
/// wrapping prose the guard was originally designed to stress. Alongside
/// it we seed a 3-receipt grouped run AND a solo receipt so glyph
/// escapes in the tool-receipt paint paths ride the same pixel gutter —
/// the r3 review flagged that receipt rows were previously invisible to
/// the guard. The transcript is deliberately tall enough to trigger the
/// scrollbar at 18px on the smaller viewport (717x474), so the
/// scrollbar-mask code path stays exercised — a future change that
/// regresses scrollbar geometry (moves its thumb OUT of the mask zone,
/// or paints extra chrome next to it) still shows up in CI here rather
/// than passing silently.
fn native_guard_transcript(source: &str) -> Vec<TranscriptEntry> {
    use zeta_gui::cards::{Card, OutputTail};
    use zeta_gui::state::{tool_excerpt, ToolReceiptKey, TranscriptEntry};
    let tool = |id: &str, name: &str, key: &str, value: &str, bytes: usize| -> TranscriptEntry {
        let mut arguments = serde_json::Map::new();
        arguments.insert(key.into(), serde_json::Value::String(value.into()));
        TranscriptEntry::Tool {
            key: ToolReceiptKey {
                session_id: None,
                agent_instance_id: None,
                tool_call_id: id.into(),
            },
            name: name.into(),
            excerpt: tool_excerpt(name, &arguments),
            summary: String::new(),
            complete: true,
            error: false,
            canceled: false,
            card: Card {
                tail: OutputTail {
                    text: "x".repeat(bytes),
                    truncated: false,
                    bytes_seen: bytes,
                },
                ..Default::default()
            },
        }
    };
    vec![
        TranscriptEntry::Assistant(source.into()),
        // Grouped run (3 receipts, same-turn adjacent) — collapses to
        // one header row when expansion is unset.
        tool("g1", "bash", "command", "grep -rn TODO src/", 900),
        tool("g2", "read", "path", "src/main.rs", 3_940),
        tool("g3", "read", "path", "src/lib.rs", 1_180),
        // Second assistant row splits the run so the ungrouped tool
        // below paints as its own individual receipt row.
        TranscriptEntry::Assistant("Spot-checking one more.".into()),
        tool("s1", "read", "path", "Cargo.toml", 252),
    ]
}

/// Encode a tiny checkerboard PNG for the ZETA-112 attachment-chrome shot.
/// A one-shot helper — the smoke driver seeds a real attachment so the
/// chip decodes into a Valid variant with a live thumbnail (the only path
/// that paints a preview; a decode failure would surface as an error chip
/// instead).
/// Seed a mixed run of tool receipts for the ZETA-125 shot. Five receipts
/// in a row so they collapse into one summary row on paint, followed by an
/// assistant reply and two more receipts (below the grouping threshold) so
/// the shot demonstrates BOTH the redesigned individual receipt and the
/// collapsed-group summary in one image. Every excerpt runs through
/// `tool_excerpt` so the shot exercises the same derivation path the
/// production render uses.
fn seed_zeta_125_tool_run(state: &mut zeta_gui::state::AppState) {
    use zeta_gui::cards::{Card, OutputTail};
    use zeta_gui::state::{tool_excerpt, ToolReceiptKey, TranscriptEntry};
    let build_tool =
        |id: &str, name: &str, key: &str, value: &str, bytes: usize| -> TranscriptEntry {
            let mut arguments = serde_json::Map::new();
            arguments.insert(
                key.to_string(),
                serde_json::Value::String(value.to_string()),
            );
            let excerpt = tool_excerpt(name, &arguments);
            TranscriptEntry::Tool {
                key: ToolReceiptKey {
                    session_id: None,
                    agent_instance_id: None,
                    tool_call_id: id.to_string(),
                },
                name: name.to_string(),
                excerpt,
                summary: String::new(),
                complete: true,
                error: false,
                canceled: false,
                card: Card {
                    tail: OutputTail {
                        text: "x".repeat(bytes),
                        truncated: false,
                        bytes_seen: bytes,
                    },
                    ..Default::default()
                },
            }
        };
    state
        .transcript
        .push(TranscriptEntry::User("Do a repo sweep.".into()));
    state.transcript.push(TranscriptEntry::Assistant(
        "Checking the source tree.".into(),
    ));
    state.transcript.push(build_tool(
        "t1",
        "bash",
        "command",
        "grep -rn TODO src/",
        900,
    ));
    state
        .transcript
        .push(build_tool("t2", "read", "path", "src/main.rs", 3_940));
    state
        .transcript
        .push(build_tool("t3", "read", "path", "src/lib.rs", 1_180));
    state
        .transcript
        .push(build_tool("t4", "edit", "path", "src/main.rs", 820));
    state.transcript.push(build_tool(
        "t5",
        "bash",
        "command",
        "cargo check --workspace",
        7_800,
    ));
    state.transcript.push(TranscriptEntry::Assistant(
        "Spot-checking a couple of files.".into(),
    ));
    state
        .transcript
        .push(build_tool("t6", "read", "path", "Cargo.toml", 252));
    state.transcript.push(build_tool(
        "t7",
        "bash",
        "command",
        "cargo test --lib -q",
        7_800,
    ));
}

/// Seed two OLDER session rows below the active one so the ZETA-134 A6
/// sidebar shot demonstrates the truthful order the fix promises. Each
/// row carries its own `first_message_preview` so `session_label`
/// renders a distinct label, and each `updated_at` sits far enough in
/// the past that `relative_age` prints a non-`now` marker.
fn seed_zeta_134_sidebar_sessions(state: &mut zeta_gui::state::AppState) {
    use chrono::{Duration, Utc};
    use zeta_gui::client::SessionMetadata;
    if let Some(active_id) = state.active_session.clone() {
        if let Some(row) = state
            .sessions
            .iter_mut()
            .find(|row| row.session_id == active_id)
        {
            row.first_message_preview = "Show the core chat loop and a small Rust example.".into();
        }
    }
    let now = Utc::now();
    let older = |minutes: i64, id: &str, preview: &str| -> SessionMetadata {
        serde_json::from_value(serde_json::json!({
            "session_id": id,
            "updated_at": (now - Duration::minutes(minutes)).to_rfc3339(),
            "first_message_preview": preview,
        }))
        .expect("static SessionMetadata seed decodes")
    };
    state.sessions.push(older(
        5,
        "aa11deadbeef",
        "Rebuild the sidebar sort so selection stops bumping updated_at.",
    ));
    state.sessions.push(older(
        20,
        "bb22deadbeef",
        "Draft release notes for the GUI polish arc.",
    ));
}

/// Seed an EXPANDED bash tool receipt whose reshaped tail carries the
/// `exit: 0` line reshape_bash_content appends. Runs after a user
/// message + assistant preamble so the row reads in context, and the
/// smoke shot proves both invariants: card body paints (expanded) and
/// the tail includes the exit code.
/// ZETA-135 (Trait 2 — diff card). Seed an expanded edit receipt whose
/// `Card::edit_data` carries a small old/new pair so the after-shot proves
/// the side-by-side diff card renders end-to-end (path header + two
/// tinted panes + line-numbered gutters). Also seeds an assistant preamble
/// so the transcript reads like a real turn. Requires the caller to also
/// seed a session with wire timestamps + provider + model if they want
/// the turn footer to paint below the last row.
fn seed_zeta_135_edit_diff(state: &mut zeta_gui::state::AppState) {
    use zeta_gui::cards::{Card, EditData, OutputTail};
    use zeta_gui::state::{extract_edit_data, tool_excerpt, ToolReceiptKey, TranscriptEntry};
    state.transcript.push(TranscriptEntry::User(
        "Bring hot.md up to date with the ZETA arc.".into(),
    ));
    state.transcript.push(TranscriptEntry::Assistant(
        "Updating the zeta line in `hot.md` to reflect the current arc state.".into(),
    ));
    let mut arguments = serde_json::Map::new();
    arguments.insert(
        "path".into(),
        serde_json::Value::String("/Users/henry/me/fun/wiki/vault/hot.md".into()),
    );
    arguments.insert(
        "old_string".into(),
        serde_json::Value::String(
            "- **Zeta UI-POLISH-2 arc RESUMED post-reset (zeta orch, 09-16 ~20:40Z).**\n\
             MERGED: ZETA-129 (#172, codespan glyph fixes) and ZETA-130\n\
             (#171, GUI slash commands) — main at `49ca130`. LIVE:\n\
             ZETA-131-PR2 (cc opus-4.7, worktree `zeta-131-approvals`) fixing REVI…"
                .into(),
        ),
    );
    arguments.insert(
        "new_string".into(),
        serde_json::Value::String(
            "- **Zeta UI-POLISH-2 arc (zeta orch): 4 of 6 lanes MERGED.**\n\
             ZETA-129 (#172), ZETA-130 (#171), ZETA-131 (#173, merged\n\
             22:09Z after 4 review rounds — allow indicator now projects\n\
             the EFFECTIVE policy mode into new_session/resume/status;\n\
             stored-null metadata…"
                .into(),
        ),
    );
    let edit_data = extract_edit_data("edit", &arguments).unwrap_or(EditData {
        old_text: String::new(),
        new_text: String::new(),
    });
    state.transcript.push(TranscriptEntry::Tool {
        key: ToolReceiptKey {
            session_id: None,
            agent_instance_id: None,
            tool_call_id: "z135-edit-hot".into(),
        },
        name: "edit".into(),
        excerpt: tool_excerpt("edit", &arguments),
        summary: String::new(),
        complete: true,
        error: false,
        canceled: false,
        card: Card {
            expanded: true,
            edit_data: Some(edit_data),
            tail: OutputTail {
                text: "hot.md updated (1 hunk)".into(),
                truncated: false,
                bytes_seen: 24,
            },
            ..Default::default()
        },
    });
}

fn seed_zeta_134_expanded_bash_receipt(state: &mut zeta_gui::state::AppState) {
    use zeta_gui::cards::{Card, OutputTail};
    use zeta_gui::state::{tool_excerpt, ToolReceiptKey, TranscriptEntry};
    state
        .transcript
        .push(TranscriptEntry::User("Run the smoke tests.".into()));
    state.transcript.push(TranscriptEntry::Assistant(
        "Running `cargo test --lib -q` and reading its tail.".into(),
    ));
    let mut arguments = serde_json::Map::new();
    arguments.insert(
        "command".into(),
        serde_json::Value::String("cargo test --lib -q".into()),
    );
    let reshaped = zeta_gui::state::reshape_bash_content(
        "",
        Some(&serde_json::json!({
            "stdout": "test result: ok. 320 passed; 0 failed; 0 ignored; 0 measured\n",
            "stderr": "",
            "exit_code": 0,
        })),
    );
    let bytes = reshaped.len();
    state.transcript.push(TranscriptEntry::Tool {
        key: ToolReceiptKey {
            session_id: None,
            agent_instance_id: None,
            tool_call_id: "z134-bash-expanded".into(),
        },
        name: "bash".into(),
        excerpt: tool_excerpt("bash", &arguments),
        summary: String::new(),
        complete: true,
        error: false,
        canceled: false,
        card: Card {
            expanded: true,
            tail: OutputTail {
                text: reshaped,
                truncated: false,
                bytes_seen: bytes,
            },
            ..Default::default()
        },
    });
}

fn png_seed_bytes() -> Vec<u8> {
    let pixels = image::RgbaImage::from_fn(48, 32, |x, y| {
        if ((x / 8) + (y / 8)) % 2 == 0 {
            image::Rgba([210, 180, 90, 255])
        } else {
            image::Rgba([40, 40, 50, 255])
        }
    });
    let mut bytes = std::io::Cursor::new(Vec::new());
    pixels
        .write_to(&mut bytes, image::ImageFormat::Png)
        .expect("encode seed png");
    bytes.into_inner()
}

pub fn start(view: &Entity<ZetaView>, window: &mut Window, cx: &mut App) {
    let path = env::var_os("ZETA_GUI_SMOKE_IMAGE");
    let guard = native_guard_enabled();
    if path.is_none() && !guard {
        return;
    }
    // Optional second capture — the ZETA-112 composer chrome with pending
    // attachment chips visible, before the settings modal covers them. Emits
    // a separate PNG so the primary shot stays comparable with prior tickets.
    let attachment_path = env::var_os("ZETA_GUI_SMOKE_ATTACHMENT_IMAGE");
    // Optional third capture — the run chrome (header + composer + sidebar)
    // WITHOUT the settings modal covering it, for review comparisons that
    // need to read the composer and header cluster directly. Saved after
    // the attachment/settings seeding but before `view.settings_open`
    // fires, so the transcript column stays visible.
    let composer_path = env::var_os("ZETA_GUI_SMOKE_COMPOSER_IMAGE");
    // Optional ZETA-125 capture — a mixed sequence of tool receipts (one
    // grouped run of 5, plus 2 individual receipts after an assistant
    // reply) so the after-shot proves the new receipt layout, the
    // metadata-adjacency rule, AND the collapsed-group summary row all at
    // once. Saved before the settings-modal shot so the transcript column
    // stays visible.
    let tools_path = env::var_os("ZETA_GUI_SMOKE_TOOLS_IMAGE");
    // Optional ZETA-128 capture — the same settings modal after switching
    // to the 18px picker MAX so the after-screenshot pair proves the
    // layout holds at both extremes of the appearance picker. Saved
    // AFTER the primary capture, then the driver quits.
    let modal_18px_path = env::var_os("ZETA_GUI_SMOKE_MODAL_18PX_IMAGE");
    // ZETA-131 captures. Each targets one lane of the approval-UX arc.
    // Rendered via `render_to_image()` so no screen-recording permission
    // is needed. Saved BEFORE the primary settings-modal capture so the
    // driver's later mutations (modal open, connection-lost banner) do
    // not overwrite the state the approval shots need.
    let zeta131_mode_path = env::var_os("ZETA_GUI_SMOKE_ZETA131_MODE_IMAGE");
    let zeta131_indicator_path = env::var_os("ZETA_GUI_SMOKE_ZETA131_INDICATOR_IMAGE");
    let zeta131_approval_path = env::var_os("ZETA_GUI_SMOKE_ZETA131_APPROVAL_IMAGE");
    // ZETA-134 captures. Sidebar shot needs multiple rows so the truthful
    // ordering (recent-first, older rows keep their timestamp position) is
    // observable; the shipped one-row after-shot could not show it.
    // Receipt shot needs an expanded bash card with the reshaped tail
    // (`exit: N` line present, empty `stderr:` label dropped); the shipped
    // collapsed-group shot showed neither.
    let zeta134_sidebar_path = env::var_os("ZETA_GUI_SMOKE_ZETA134_SIDEBAR_IMAGE");
    let zeta134_receipt_path = env::var_os("ZETA_GUI_SMOKE_ZETA134_RECEIPT_IMAGE");
    // ZETA-135 captures. Round 2 finding 2 required real screenshots
    // of the wiki-look states; the round-1 shot was a byte-identical
    // copy of the reference PNG. Each env var here seeds a specific
    // state (default with restyle, expanded receipt with kind glyph +
    // panel header, expanded EDIT receipt with diff card) so the
    // reviewer can diff the three states against the reference.
    let zeta135_diff_path = env::var_os("ZETA_GUI_SMOKE_ZETA135_DIFF_IMAGE");
    // ZETA-133 D1 (`AFTER`) capture. Seeds a mixed transcript (prose +
    // tools + turn footer) so the after-shot shows every row kind
    // sharing ONE body left edge under the wiki-look shell. D3
    // (bottom-anchor for short transcripts) is REPORTED BLOCKED in this
    // PR — the pt-on-list approach the contract's "top-fills-first
    // spacing" hint suggested was found to alter ListState positioning
    // semantics (tripping the ZETA-107 view-sync stability test and
    // pushing single-row content off the visible list viewport), which
    // the hard constraint forbids. See the ladder row and PR body.
    let zeta133_after_path = env::var_os("ZETA_GUI_SMOKE_ZETA133_AFTER_IMAGE");
    view.update(cx, |_, cx| {
        cx.spawn_in(window, async move |view, cx| {
            let mut phase = 0;
            for _ in 0..600 {
                cx.background_executor()
                    .timer(Duration::from_millis(50))
                    .await;
                let (finished, run_guard) = cx
                    .update(|window, cx| {
                        let entity = view.upgrade().expect("smoke view remains alive");
                        let (ready, active, idle, ready_to_capture) = {
                            let view = entity.read(cx);
                            if let Some(error) = &view.command_error {
                                panic!("smoke command failed: {error}");
                            }
                            let has_thinking_marker = view
                                .state
                                .transcript
                                .iter()
                                .any(|entry| matches!(entry, TranscriptEntry::Thinking));
                            let has_assistant = view
                                .state
                                .transcript
                                .iter()
                                .any(|entry| matches!(entry, TranscriptEntry::Assistant(_)));
                            (
                                view.state.connection == ConnectionState::Connected,
                                view.state.active_session.is_some(),
                                !view.state.streaming && !view.pending_command,
                                view.state.streaming && has_thinking_marker && has_assistant,
                            )
                        };
                        window.render_frame(cx);
                        match phase {
                            0 if ready => {
                                window.click("new-session", cx);
                                phase = 1;
                            }
                            1 if active && idle => {
                                // Seed the composer value directly and submit via
                                // `send_composer`, rather than dispatching per-character
                                // keystrokes. The native `.input(...)` helper spends the
                                // whole message duration listening on the real macOS
                                // window; any stray character typed at the terminal
                                // during the capture flight lands on the focused
                                // composer and rides through to the transcript. That
                                // capture-time input leak surfaced as a stray leading
                                // "j" in the shipped ZETA-128 after-screenshots
                                // ("jShow the core chat loop..."). Direct state assignment
                                // keeps the rendered turn deterministic on any dev machine.
                                entity.update(cx, |view, cx| {
                                    view.composer.update(cx, |input, cx| {
                                        input.set_value(
                                            "Show the core chat loop and a small Rust example.",
                                            window,
                                            cx,
                                        );
                                    });
                                    view.send_composer(cx);
                                });
                                phase = 2;
                            }
                            // Once the transcript carries the "+ Thought"
                            // marker AND the assistant preamble, seed the
                            // extra chrome the after-screenshot must show:
                            // a branch row, a connection-lost banner (so
                            // the danger-rail attention lights up), and a
                            // popup modal. Sequence matters — draw once so
                            // the branch tree lands before the settings
                            // overlay occludes the transcript.
                            2 if ready_to_capture => {
                                entity.update(cx, |view, cx| {
                                    view.state.session_view.available = true;
                                    view.state.session_view.branches = vec![
                                        zeta_gui::session::Branch {
                                            id: "main".into(),
                                            label: "main".into(),
                                            current: true,
                                            depth: 0,
                                        },
                                        zeta_gui::session::Branch {
                                            id: "review".into(),
                                            label: "review".into(),
                                            current: false,
                                            depth: 1,
                                        },
                                    ];
                                    view.state.session_view.message_ids.insert(0, "m1".into());
                                    view.state.mark_connection_lost("socket closed");
                                    cx.notify();
                                });
                                window.render_frame(cx);
                                phase = 3;
                            }
                            3 => {
                                if guard {
                                    phase = 4;
                                    return (false, true);
                                }
                                // ZETA-112 attachment capture — seed a mixed
                                // batch (one valid chip with a real decoded
                                // thumbnail, one decode-failure chip whose
                                // header parses but whose body cannot decode,
                                // and one format-reject chip) so the shot
                                // proves the typed pending model: a valid
                                // chip always paints a thumbnail, and every
                                // failure mode surfaces its own error chip.
                                // Clear before the modal shot below so the
                                // primary after-screenshot stays unchanged.
                                if let Some(ref attach_path) = attachment_path {
                                    entity.update(cx, |view, cx| {
                                        // The `add_pending_attachments` guard
                                        // requires a live connection and an
                                        // idle turn — the smoke driver's prior
                                        // phase deliberately synthesises a
                                        // lost-connection banner, so lift the
                                        // guard to attach the seeded chips.
                                        view.state.streaming = false;
                                        view.state.connection = ConnectionState::Connected;
                                        let mut items: Vec<
                                            Result<
                                                zeta_gui::session::ImageAttachment,
                                                (String, String),
                                            >,
                                        > = Vec::new();
                                        if let Ok(image) =
                                            zeta_gui::session::ImageAttachment::from_bytes(
                                                "diagram.png".into(),
                                                &png_seed_bytes(),
                                            )
                                        {
                                            items.push(Ok(image));
                                        }
                                        if let Ok(broken) =
                                            zeta_gui::session::ImageAttachment::from_bytes(
                                                "sketch.png".into(),
                                                b"\x89PNG\r\n\x1a\n",
                                            )
                                        {
                                            items.push(Ok(broken));
                                        }
                                        items.push(Err((
                                            "notes.bmp".into(),
                                            "choose a PNG, JPEG, GIF, or WebP image".into(),
                                        )));
                                        view.add_pending_attachments(items, cx);
                                    });
                                    window.render_frame(cx);
                                    window
                                        .render_to_image()
                                        .expect("native renderer capture")
                                        .save(PathBuf::from(attach_path))
                                        .expect("save attachment screenshot");
                                    entity.update(cx, |view, cx| {
                                        view.clear_composer_images(cx);
                                    });
                                }
                                // Round-2 review: the primary shot below
                                // opens the settings modal, which occludes
                                // the composer half of the chrome. If the
                                // caller wants a clean chrome shot for the
                                // composer/header comparison, save one HERE
                                // — connection restored, no chips, no modal.
                                if let Some(ref composer_path) = composer_path {
                                    entity.update(cx, |view, cx| {
                                        view.state.connection = ConnectionState::Connected;
                                        cx.notify();
                                    });
                                    window.render_frame(cx);
                                    window
                                        .render_to_image()
                                        .expect("native renderer capture")
                                        .save(PathBuf::from(composer_path))
                                        .expect("save composer screenshot");
                                }
                                if let Some(ref tools_path) = tools_path {
                                    entity.update(cx, |view, cx| {
                                        view.state.connection = ConnectionState::Connected;
                                        seed_zeta_125_tool_run(&mut view.state);
                                        let count = view.state.transcript.len();
                                        view.transcript.update(cx, |scroll, cx| {
                                            scroll.reset(count, cx);
                                        });
                                        cx.notify();
                                    });
                                    window.render_frame(cx);
                                    window
                                        .render_to_image()
                                        .expect("native renderer capture")
                                        .save(PathBuf::from(tools_path))
                                        .expect("save tools screenshot");
                                }
                                // ZETA-131 C2: header run-band with the
                                // `auto-approve` chip painted. Server-applied
                                // mode set to `allow` so `render_run_header`
                                // shows the chip; banner cleared so the
                                // header owns the frame. Metrics carry a
                                // model name so the header cluster reads.
                                if let Some(ref indicator_path) = zeta131_indicator_path {
                                    entity.update(cx, |view, cx| {
                                        view.state.connection = ConnectionState::Connected;
                                        view.state.session_view.applied_mode = "allow".into();
                                        view.state.metrics.model = Some("claude-opus-4-7".into());
                                        view.settings_open = false;
                                        cx.notify();
                                    });
                                    window.render_frame(cx);
                                    window
                                        .render_to_image()
                                        .expect("native renderer zeta-131 indicator capture")
                                        .save(PathBuf::from(indicator_path))
                                        .expect("save zeta-131 indicator screenshot");
                                }
                                // ZETA-131 B3: an approval dialog with the
                                // "Always allow <tool>" button visible and
                                // the shortcut hint in the footer. Seed a
                                // pending approval through the same seam
                                // `sync_approval` watches; two frames so
                                // the dialog paints AFTER
                                // `sync_approval` reads state.
                                if let Some(ref approval_path) = zeta131_approval_path {
                                    entity.update(cx, |view, cx| {
                                        use zeta_gui::client::{Approval, ServerEvent, ToolCall};
                                        view.state.connection = ConnectionState::Connected;
                                        view.state.session_view.applied_mode = "allow".into();
                                        view.settings_open = false;
                                        let session_id = view.state.active_session.clone();
                                        view.apply_worker_message(
                                            WorkerMessage::Event(ServerEvent::ApprovalRequest {
                                                session_id,
                                                approval: Approval {
                                                    request_id: "zeta131-demo".into(),
                                                    tool_call: ToolCall {
                                                        id: "zeta131-demo".into(),
                                                        name: "bash".into(),
                                                        arguments: serde_json::from_value(
                                                            serde_json::json!({
                                                                "command":
                                                                    "rg zeta ~/src"
                                                            }),
                                                        )
                                                        .unwrap(),
                                                    },
                                                },
                                            }),
                                            window,
                                            cx,
                                        );
                                        cx.notify();
                                    });
                                    window.render_frame(cx);
                                    window.render_frame(cx);
                                    window
                                        .render_to_image()
                                        .expect("native renderer zeta-131 approval capture")
                                        .save(PathBuf::from(approval_path))
                                        .expect("save zeta-131 approval screenshot");
                                    // Reset: close the dialog before the
                                    // primary settings capture opens the
                                    // modal — the dialog occludes the
                                    // approval-mode section otherwise.
                                    entity.update(cx, |view, cx| {
                                        view.state.approvals.clear();
                                        view.dialog_request = None;
                                        window.close_dialog(cx);
                                    });
                                    window.render_frame(cx);
                                }
                                // ZETA-134 A6: prove the sidebar sort stays
                                // truthful across selection. Seed two older
                                // rows below the active one so the shot
                                // shows three distinct labels with distinct
                                // age markers (now / 5m / 20m). A shipped
                                // one-row shot could not demonstrate order.
                                if let Some(sidebar_path) = &zeta134_sidebar_path {
                                    entity.update(cx, |view, cx| {
                                        view.state.connection = ConnectionState::Connected;
                                        view.state.approvals.clear();
                                        view.dialog_request = None;
                                        view.settings_open = false;
                                        seed_zeta_134_sidebar_sessions(&mut view.state);
                                        cx.notify();
                                    });
                                    window.render_frame(cx);
                                    window.render_frame(cx);
                                    window
                                        .render_to_image()
                                        .expect("native renderer zeta-134 sidebar capture")
                                        .save(PathBuf::from(sidebar_path))
                                        .expect("save zeta-134 sidebar screenshot");
                                    entity.update(cx, |view, cx| {
                                        view.state.sessions.retain(|row| {
                                            Some(&row.session_id)
                                                == view.state.active_session.as_ref()
                                        });
                                        cx.notify();
                                    });
                                    window.render_frame(cx);
                                }
                                // ZETA-135 (Trait 2 — diff card): expanded
                                // edit receipt with typed old/new strings so
                                // the after-shot proves the side-by-side
                                // diff card renders end-to-end. Also seeds
                                // wire session data so the turn footer
                                // paints below the last row.
                                if let Some(diff_path) = &zeta135_diff_path {
                                    entity.update(cx, |view, cx| {
                                        view.state.connection = ConnectionState::Connected;
                                        view.state.transcript.clear();
                                        seed_zeta_135_edit_diff(&mut view.state);
                                        // Session metadata drives the turn
                                        // footer — provider · model ·
                                        // duration below the last row.
                                        view.state.sessions.clear();
                                        view.state.sessions.push(
                                            zeta_gui::client::SessionMetadata {
                                                version: 1,
                                                session_id: "z135".into(),
                                                created_at: "2026-09-17T12:00:00Z".into(),
                                                updated_at: "2026-09-17T12:06:32Z".into(),
                                                provider: "cc".into(),
                                                model: "claude-fable-5".into(),
                                                cwd: String::new(),
                                                retained_tail: 0,
                                                compaction_budget: 0,
                                                override_audit: Vec::new(),
                                                system_prompt: String::new(),
                                                context_files: Vec::new(),
                                                vim_mode: false,
                                                budget_pinned: false,
                                                plan_mode: false,
                                                name: String::new(),
                                                first_message_preview: String::new(),
                                                approval_mode: String::new(),
                                            },
                                        );
                                        view.state.active_session = Some("z135".into());
                                        view.state.metrics.model = Some("claude-fable-5".into());
                                        let count = view.state.transcript.len();
                                        view.transcript.update(cx, |scroll, cx| {
                                            scroll.reset(count, cx);
                                        });
                                        cx.notify();
                                    });
                                    window.render_frame(cx);
                                    window.render_frame(cx);
                                    window
                                        .render_to_image()
                                        .expect("native renderer zeta-135 diff capture")
                                        .save(PathBuf::from(diff_path))
                                        .expect("save zeta-135 diff screenshot");
                                }
                                // ZETA-133 (D1): mixed transcript — prose +
                                // tool receipts (bash + read + edit) +
                                // assistant reply — so the after-shot shows
                                // every row kind sharing ONE body left edge
                                // under the wiki-look shell. Session
                                // metadata seeds the turn footer.
                                if let Some(after_path) = &zeta133_after_path {
                                    entity.update(cx, |view, cx| {
                                        view.state.connection = ConnectionState::Connected;
                                        view.state.transcript.clear();
                                        seed_zeta_125_tool_run(&mut view.state);
                                        view.state.sessions.clear();
                                        view.state.sessions.push(
                                            zeta_gui::client::SessionMetadata {
                                                version: 1,
                                                session_id: "z133".into(),
                                                created_at: "2026-09-17T12:00:00Z".into(),
                                                updated_at: "2026-09-17T12:00:42Z".into(),
                                                provider: "cc".into(),
                                                model: "claude-fable-5".into(),
                                                cwd: String::new(),
                                                retained_tail: 0,
                                                compaction_budget: 0,
                                                override_audit: Vec::new(),
                                                system_prompt: String::new(),
                                                context_files: Vec::new(),
                                                vim_mode: false,
                                                budget_pinned: false,
                                                plan_mode: false,
                                                name: String::new(),
                                                first_message_preview: String::new(),
                                                approval_mode: String::new(),
                                            },
                                        );
                                        view.state.active_session = Some("z133".into());
                                        view.state.metrics.model = Some("claude-fable-5".into());
                                        let count = view.state.transcript.len();
                                        view.transcript.update(cx, |scroll, cx| {
                                            scroll.reset(count, cx);
                                        });
                                        cx.notify();
                                    });
                                    window.render_frame(cx);
                                    window.render_frame(cx);
                                    window
                                        .render_to_image()
                                        .expect("native renderer zeta-133 after capture")
                                        .save(PathBuf::from(after_path))
                                        .expect("save zeta-133 after screenshot");
                                }
                                // ZETA-134 D6: expanded bash receipt with
                                // the reshaped tail. Card.expanded=true so
                                // the body paints, and the tail carries the
                                // `exit: 0` line reshape_bash_content
                                // appends — the shipped shot was collapsed
                                // and demonstrated neither.
                                if let Some(receipt_path) = &zeta134_receipt_path {
                                    entity.update(cx, |view, cx| {
                                        view.state.connection = ConnectionState::Connected;
                                        view.state.transcript.clear();
                                        seed_zeta_134_expanded_bash_receipt(&mut view.state);
                                        let count = view.state.transcript.len();
                                        view.transcript.update(cx, |scroll, cx| {
                                            scroll.reset(count, cx);
                                        });
                                        cx.notify();
                                    });
                                    window.render_frame(cx);
                                    window.render_frame(cx);
                                    window
                                        .render_to_image()
                                        .expect("native renderer zeta-134 receipt capture")
                                        .save(PathBuf::from(receipt_path))
                                        .expect("save zeta-134 receipt screenshot");
                                }
                                entity.update(cx, |view, cx| {
                                    view.state.connection = ConnectionState::Connected;
                                    view.settings_open = true;
                                    view.state.session_view.models =
                                        vec!["claude-opus-4-7".into(), "claude-fable-5".into()];
                                    view.state.session_view.current_model =
                                        "claude-opus-4-7".into();
                                    view.state.session_view.selected_model = 0;
                                    view.state.session_view.selected_mode = 0;
                                    view.state
                                        .session_view
                                        .model_providers
                                        .insert("claude-opus-4-7".into(), "claude".into());
                                    view.state
                                        .session_view
                                        .model_providers
                                        .insert("claude-fable-5".into(), "claude".into());
                                    // Restore the lost-connection banner so
                                    // the shot carries every piece of chrome
                                    // the reviewer named.
                                    view.state.mark_connection_lost("socket closed");
                                    cx.notify();
                                });
                                // Two frames: the first opens the modal and
                                // its `on_children_prepainted` hook measures
                                // rows to compute the row-snapped scroll-cue
                                // height, which is queued for the next
                                // frame via `on_next_frame`. The second draw
                                // paints the mask at its settled height so
                                // the captured pixels show the final
                                // geometry (a single frame would capture
                                // the raw, un-snapped mask).
                                window.render_frame(cx);
                                window.render_frame(cx);
                                if let Some(path) = &path {
                                    window
                                        .render_to_image()
                                        .expect("native renderer capture")
                                        .save(PathBuf::from(path))
                                        .expect("save smoke screenshot");
                                    println!("SMOKE-PASS: {}", PathBuf::from(path).display());
                                }
                                // ZETA-131 C1: same modal, `selected_mode`
                                // moved off the default `ask` (index 0) to
                                // `allow` (index 1) so the after-screenshot
                                // shows the FILLED primary variant on the
                                // selected segment — the shipped
                                // ghost().selected(true) fill was invisible.
                                if let Some(mode_path) = &zeta131_mode_path {
                                    entity.update(cx, |view, cx| {
                                        view.state.session_view.selected_mode = 1;
                                        cx.notify();
                                    });
                                    window.render_frame(cx);
                                    window.render_frame(cx);
                                    window
                                        .render_to_image()
                                        .expect("native renderer zeta-131 mode capture")
                                        .save(PathBuf::from(mode_path))
                                        .expect("save zeta-131 mode screenshot");
                                    entity.update(cx, |view, cx| {
                                        view.state.session_view.selected_mode = 0;
                                        cx.notify();
                                    });
                                    window.render_frame(cx);
                                }
                                // Optional second modal capture at the 18px
                                // picker MAX. Same panel, larger type — a
                                // reviewer can walk the before/after pair at
                                // the picker extreme without a second run.
                                if let Some(modal_18px) = &modal_18px_path {
                                    entity.update(cx, |_, cx| {
                                        let appearance = theme::Appearance {
                                            theme: theme::ThemeId::default(),
                                            font_family: gpui::SharedString::new_static(
                                                theme::DEFAULT_FONT_FAMILY,
                                            ),
                                            font_size: theme::clamp_font_size(
                                                theme::MAX_FONT_SIZE_PX,
                                            ),
                                        };
                                        theme::apply_with(cx, &appearance);
                                    });
                                    // Two frames for the same reason as the
                                    // 13px capture above: the appearance
                                    // change re-measures the rows at the
                                    // new base font, and the snapped
                                    // scroll-cue height only lands after
                                    // the `on_next_frame` follow-up draw.
                                    window.render_frame(cx);
                                    window.render_frame(cx);
                                    window
                                        .render_to_image()
                                        .expect("native renderer 18px capture")
                                        .save(PathBuf::from(modal_18px))
                                        .expect("save 18px modal screenshot");
                                }
                                cx.quit();
                                return (true, false);
                            }
                            _ => {}
                        }
                        (false, false)
                    })
                    .expect("smoke window update");
                if run_guard {
                    let entity = view.upgrade().expect("smoke view remains alive");
                    run_native_wrap_guards(entity, &mut *cx).await;
                    cx.update(|_, cx| cx.quit())
                        .expect("smoke window remains open");
                    return;
                }
                if finished {
                    return;
                }
            }
            panic!("smoke session did not complete within 30 seconds");
        })
        .detach();
    });
}
