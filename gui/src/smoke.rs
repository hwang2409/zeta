//! Opt-in smoke driver: real input events, real worker/socket, native Metal pixels.
use super::*;
use gpui::{px, Pixels};
use gpui_kit::test::TestWindowExt;

const NATIVE_GUARD_COLOR_THRESHOLD: u8 = 10;
const NATIVE_GUARD_MIN_CONSECUTIVE: usize = 2;
const NATIVE_GUARD_SCROLLBAR_WIDTH: Pixels = px(8.);
const NATIVE_GUARD_COMPOSER_HEIGHT: Pixels = px(80.);
const NATIVE_GUARD_SHAPES: &[(&str, &str)] = &[
    (
        "wedge",
        "2. `zeta serve` session hardening — half-written session dirs \
         (`conversation.jsonl` without `meta.json`) wedge status/list. Atomic dir \
         creation via `meta.json` tmp+rename.\n3. Follow-up work with additional \
         wrapping to exercise the hanging indent so the paragraph reliably breaks \
         onto a continuation line even at 2204px.",
    ),
    (
        "adjacent",
        "1. Outer numbered item with plenty of prose to force wrapping onto multiple \
         continuation lines at every picker step.\n2. Second outer numbered item \
         to prove the second sibling wraps in the same column geometry as the first \
         with more filler prose here now.",
    ),
    (
        "nested",
        "1. Outer item with room to spare.\n   - Nested bullet A that itself carries \
         enough hanging-indent text to force wrap boundaries near the prose cap at \
         every base picker step.\n   - Nested bullet B with more prose — deeper \
         nesting stays inside the same column even when the marker indent has \
         consumed a few characters.",
    ),
    (
        "long_token",
        "Prose leading up to a very long unbroken token that the wrap engine cannot \
         break: \
         supercalifragilisticexpialidocious_but_much_longer_than_any_column_should_ever_be_aaaaaaaaaaaaaaaaaaaa \
         and then some trailing prose after it.",
    ),
];

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

/// Return the first x range with at least two adjacent pixels that differ
/// from the active canvas token. One isolated anti-aliased pixel is noise;
/// adjacent pixels are the minimum evidence for an escaped glyph stroke.
fn escaped_glyph_range(
    image: &image::RgbaImage,
    x_start: u32,
    x_end: u32,
    y_start: u32,
    y_end: u32,
    background: [u8; 3],
) -> Option<(u32, u32)> {
    for y in y_start..y_end {
        let mut run_start = None;
        for x in x_start..x_end {
            let pixel = image.get_pixel(x, y).0;
            let over_threshold = pixel[..3].iter().zip(background).any(|(actual, expected)| {
                actual.abs_diff(expected) >= NATIVE_GUARD_COLOR_THRESHOLD
            });
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

fn scan_native_gutter(
    image: &image::RgbaImage,
    window: &Window,
    font_size: Pixels,
    shape: &str,
    achieved_width: u32,
    achieved_height: u32,
) {
    let scale = window.scale_factor();
    let window_width = f32::from(window.bounds().size.width);
    let main_left = f32::from(theme::SIDEBAR_WIDTH);
    let main_width = (window_width - main_left).max(0.);
    let column_width = f32::from(theme::prose_max_width(font_size)).min(main_width);
    let content_right =
        main_left + (main_width - column_width) / 2. + column_width - theme::PROSE_ROW_PADDING_X;
    let x_start = (content_right * scale).ceil() as u32;
    let x_end = image
        .width()
        .saturating_sub((f32::from(NATIVE_GUARD_SCROLLBAR_WIDTH) * scale).ceil() as u32);
    // Larger picker sizes can make the header's content-driven height exceed
    // its 44px minimum. Leave the scan below that dynamic edge.
    let y_start = ((f32::from(theme::HEADER_BAND1_MIN_HEIGHT) + 8.) * scale).ceil() as u32;
    // The live composer stays visible during the guard. Its fixed children
    // occupy 8px top padding + 44px input row + 4px gap + 16px footer + 8px
    // bottom padding, so stop before composer chrome can look like a glyph.
    let y_end = image
        .height()
        .saturating_sub((f32::from(NATIVE_GUARD_COMPOSER_HEIGHT) * scale).ceil() as u32 + 1);
    let background = rgb8(theme::palette::canvas());
    if let Some((escape_start, escape_end)) =
        escaped_glyph_range(image, x_start, x_end, y_start, y_end, background)
    {
        panic!(
            "native pixel gutter guard failed: shape={shape} size={font_size:?} \
             x_range={escape_start}..={escape_end} gutter={x_start}..{x_end} \
             window_width={window_width} scale={scale} column_width={column_width} \
             content_right={content_right} y_range={y_start}..{y_end} background={background:?}"
        );
    }
    println!(
        "NATIVE-GUARD-PASS: shape={shape} size={font_size:?} \
         viewport={achieved_width}x{achieved_height} gutter={x_start}..{x_end}"
    );
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
    let font_sizes = [
        px(theme::MIN_FONT_SIZE_PX),
        theme::DEFAULT_FONT_SIZE,
        px(theme::MAX_FONT_SIZE_PX),
    ];
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
            for &(shape, source) in NATIVE_GUARD_SHAPES {
                cx.update(|window, cx| {
                    view.update(cx, |view, cx| {
                        view.state.connection = ConnectionState::Connected;
                        // r3 finding 7: the guard's matrix used to hold a
                        // single Assistant row per shape, so glyph escapes
                        // in the tool-receipt render paths were never
                        // scanned. Seed BOTH grouped and ungrouped
                        // receipts alongside the wrapping prose so
                        // receipt-row escapes ride the same pixel gutter.
                        view.state.transcript = native_guard_transcript(source);
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
                    scan_native_gutter(&image, window, font_size, shape, achieved.0, achieved.1);
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

/// Build the transcript the native pixel-gutter guard renders for one
/// shape entry. The assistant row carries the wrapping prose the guard
/// was originally designed to stress. Alongside it we seed a 3-receipt
/// grouped run AND a solo receipt so glyph escapes in the tool-receipt
/// paint paths ride the same pixel gutter — the r3 review flagged that
/// receipt rows were previously invisible to the guard.
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
        // one header row when expansion is unset; expands to three
        // interior rows if streaming forces expansion. Both shapes are
        // in play depending on state at scan time.
        tool("g1", "bash", "command", "grep -rn TODO src/", 900),
        tool("g2", "read", "path", "src/main.rs", 3_940),
        tool("g3", "read", "path", "src/lib.rs", 1_180),
        // Second assistant row splits the run — the ungrouped tool
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
                                let focus = entity.read(cx).composer.focus_handle(cx);
                                window.focus(&focus, cx);
                                window
                                    .input("Show the core chat loop and a small Rust example.", cx);
                                window.press("enter", cx);
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
                                window.render_frame(cx);
                                if let Some(path) = &path {
                                    window
                                        .render_to_image()
                                        .expect("native renderer capture")
                                        .save(PathBuf::from(path))
                                        .expect("save smoke screenshot");
                                    println!("SMOKE-PASS: {}", PathBuf::from(path).display());
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
