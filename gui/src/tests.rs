use super::*;
use gpui::{InputEvent as _, TestAppContext, VisualTestContext, WindowHandle};
use gpui_kit::component::Theme;
use serde_json::json;
use std::path::PathBuf;
use std::sync::mpsc::Receiver;
use std::sync::LazyLock;
use zeta_gui::client::{ModelCatalog, ServerEvent, SessionMetadata, StatusResult, ToolCall};
use zeta_gui::session::{self, Branch, ImageAttachment, SessionSettings};

/// Isolated `ZETA_HOME` shared by every test in this file. Set once on first
/// access via `LazyLock` so any test that reads or writes `prefs::prefs_path`
/// (directly, or through `prefs::commit`/`prefs::save`) lands under a temp
/// dir instead of the developer's real `~/.zeta`. The `prefs_path` guard
/// panics if a test forgets to force this before touching the file.
static SCOPED_ZETA_HOME: LazyLock<PathBuf> = LazyLock::new(|| {
    let dir = env::temp_dir().join(format!("zeta-gui-tests-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scoped ZETA_HOME");
    env::set_var("ZETA_HOME", &dir);
    dir
});

/// Wipe any prefs file left by an earlier test. Forces the scoped
/// `ZETA_HOME` first so `prefs::prefs_path()` never falls back to the real
/// user home under `cargo test`.
fn wipe_scoped_prefs() {
    let _ = &*SCOPED_ZETA_HOME;
    let _ = std::fs::remove_file(prefs::prefs_path());
}

fn session() -> SessionMetadata {
    serde_json::from_value(
        json!({"session_id":"ab12deadbeef", "updated_at":"2026-09-09T12:00:00Z"}),
    )
    .unwrap()
}

fn setup(
    cx: &mut TestAppContext,
) -> (
    WindowHandle<Root>,
    Entity<ZetaView>,
    Receiver<CommandMessage>,
) {
    static ENVIRONMENT: std::sync::Once = std::sync::Once::new();
    ENVIRONMENT.call_once(|| {
        env::set_var("TERM", "dumb");
        env::set_var("COLORTERM", "");
    });
    // Force the scoped ZETA_HOME BEFORE anything touches prefs::prefs_path().
    // `theme::apply` below reads no disk, but any test that clicks the
    // appearance picker will call `prefs::commit` and land here.
    let _ = &*SCOPED_ZETA_HOME;
    cx.update(init);
    // Baseline appearance: shipped default (opencode / JetBrains Mono / 13px).
    // Runs before `ZetaView::new` so every test sees a deterministic theme,
    // regardless of what a peer test flipped through the appearance picker.
    cx.update(theme::apply);
    let (commands, receiver) = mpsc::channel();
    let mut view = None;
    let window = cx.open_window(gpui::size(px(1100.), px(760.)), |window, cx| {
        let entity = cx.new(|cx| {
            let mut view = ZetaView::new(window, cx, commands);
            view.state.connection = ConnectionState::Connected;
            view.state.active_session = Some(session().session_id.clone());
            view.state.sessions.push(session());
            view
        });
        view = Some(entity.clone());
        Root::new(entity, window, cx)
    });
    (window, view.unwrap(), receiver)
}

#[test]
fn prose_wrap_budget_floors_fractional_widths_and_fits_the_content_box() {
    // r4 finding 5: the wrap budget MUST be floor()-ed so a fractional
    // `prose_max_width` cannot let the painter's rounding push a glyph
    // one pixel past `content_right`. The r3 pixel-gutter guard flagged
    // that pattern at 11px on the 922×610 viewport — glyphs, not quads,
    // painting one column past the column content edge.
    //
    // This is a headless mutation-sensitive test: it hard-fails if the
    // `.floor()` call in `theme::prose_wrap_budget` is removed or swapped
    // for `.ceil()` / `.round()` / a bare cast. It does NOT rely on the
    // wide-viewport pixel guard, which under-scans the fractional strip
    // by design.
    let base = gpui::px(11.);
    let pre_floor =
        f32::from(theme::prose_max_width(base)) - 2.0 * theme::PROSE_ROW_PADDING_X - 2.0;
    // Premise: 11 * 0.62 * 88 + 32 - 32 - 2 = 597.68 — must be fractional
    // so the floor()/no-floor split is observable.
    assert!(
        (pre_floor - pre_floor.floor()).abs() > f32::EPSILON,
        "test premise: pre-floor budget for {base:?} is {pre_floor} — must \
         be fractional to make the floor mutation observable"
    );
    let budget = theme::prose_wrap_budget(base);
    let budget_f = f32::from(budget);
    assert!(
        (budget_f - budget_f.round()).abs() < f32::EPSILON,
        "wrap budget must be integer-valued (floored) but is {budget_f}"
    );
    assert!(
        (budget_f - pre_floor.floor()).abs() < f32::EPSILON,
        "wrap budget {budget_f} must equal floor(pre_floor) {} — a \
         mutation that removed .floor() or swapped it for .ceil()/.round() \
         would trip here",
        pre_floor.floor(),
    );
    // Synthetic fractional-width layout: the row's inner content box is
    // `prose_max_width - 2 * padding` (fractional at this base). The
    // wrap budget must fit inside that box strictly — a caller that
    // stopped flooring would sit at 597.68 and pass the box check by
    // luck, but the integer-valued assertion above catches it. A caller
    // that ceil()-ed to 598 would push the row's advertised wrap width
    // above the content box and glyphs shape past `content_right`.
    let content_box_right =
        f32::from(theme::prose_max_width(base)) - 2.0 * theme::PROSE_ROW_PADDING_X;
    assert!(
        budget_f <= content_box_right,
        "wrap budget {budget_f} must fit inside the content box {content_box_right}"
    );
}

#[test]
fn sidebar_uses_name_then_preview_and_never_session_id() {
    let mut session = session();
    assert_eq!(sidebar::session_label(&session, None), "New conversation");
    session.first_message_preview = "first\n\tmessage".into();
    assert_eq!(sidebar::session_label(&session, None), "first message");
    session.name = "named\n conversation".into();
    assert_eq!(sidebar::session_label(&session, None), "named conversation");
    session.name.clear();
    let rows = [TranscriptEntry::User("current\nfirst message".into())];
    assert_eq!(
        sidebar::session_label(&session, Some(&rows)),
        "current first message"
    );
    let now = chrono::DateTime::parse_from_rfc3339("2026-09-09T13:05:00Z")
        .unwrap()
        .with_timezone(&chrono::Utc);
    assert_eq!(sidebar::relative_age(&session.updated_at, now), "1h");
    assert_eq!(sidebar::relative_age("bad", now), "");
    assert_eq!(sidebar::relative_age("2026-09-10T00:00:00Z", now), "now");
}

#[gpui::test]
fn kit_composer_sends_enter_and_preserves_newlines_and_rejected_drafts(cx: &mut TestAppContext) {
    let (window, view, receiver) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.simulate_keystrokes("h i shift-enter t h e r e");
    assert!(receiver.try_recv().is_err());
    assert_eq!(
        view.read_with(&visual, |view, cx| view
            .composer
            .read(cx)
            .value()
            .to_string()),
        "hi\nthere"
    );
    visual.simulate_keystrokes("enter enter");
    let sent = receiver.try_recv();
    assert!(
        matches!(&sent, Ok(CommandMessage::Send(text)) if text == "hi\nthere"),
        "got {sent:?}"
    );
    assert!(receiver.try_recv().is_err());
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.apply_worker_message(WorkerMessage::Rejected("try again".into()), window, cx);
            assert_eq!(view.composer.read(cx).value().as_ref(), "hi\nthere");
            assert_eq!(view.state.connection, ConnectionState::Connected);
        })
    });
    visual.simulate_keystrokes("enter");
    assert!(matches!(receiver.try_recv(), Ok(CommandMessage::Send(_))));
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.apply_worker_message(WorkerMessage::Sent("hi\nthere".into()), window, cx);
            assert!(view.composer.read(cx).value().is_empty());
            assert!(view.state.streaming);
        })
    });
    visual.simulate_keystrokes("escape");
    assert!(matches!(receiver.try_recv(), Ok(CommandMessage::Abort)));
}

#[gpui::test]
fn connection_banner_reconnect_and_session_rejection(cx: &mut TestAppContext) {
    let (window, view, receiver) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.apply_worker_message(WorkerMessage::Lost("socket closed".into()), window, cx);
            assert_eq!(view.composer_hint(), "Reconnect to send a message");
            view.send_composer(cx);
        })
    });
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let bounds = visual
        .debug_bounds("reconnect-button")
        .expect("visible reconnect");
    visual.simulate_click(bounds.center(), Default::default());
    assert!(matches!(receiver.try_recv(), Ok(CommandMessage::Reconnect)));
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            assert_eq!(view.state.connection, ConnectionState::Reconnecting);
            view.apply_worker_message(WorkerMessage::Connected, window, cx);
            view.apply_worker_message(
                WorkerMessage::Rejected("session could not load".into()),
                window,
                cx,
            );
            assert_eq!(view.state.connection, ConnectionState::Connected);
            assert!(view.can_change_session());
        })
    });
    visual.update(|window, cx| window.draw(cx).clear(cx));
    assert!(visual.debug_bounds("reconnect-button").is_none());
}

#[gpui::test]
fn approval_dialog_dispatches_approve_and_deny_once(cx: &mut TestAppContext) {
    let (window, view, receiver) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    for (key, approve) in [("enter", true), ("escape", false)] {
        visual.update(|window, cx| {
            view.update(cx, |view, cx| {
                view.apply_worker_message(
                    WorkerMessage::Event(ServerEvent::ApprovalRequest {
                        session_id: view.state.active_session.clone(),
                        approval: Approval {
                            request_id: key.into(),
                            tool_call: ToolCall {
                                id: key.into(),
                                name: "bash".into(),
                                arguments: serde_json::from_value(json!({"command":"pwd"}))
                                    .unwrap(),
                            },
                        },
                    }),
                    window,
                    cx,
                );
            })
        });
        visual.update(|window, cx| window.draw(cx).clear(cx));
        assert!(visual.debug_bounds("dialog-layer").is_some());
        visual.simulate_keystrokes(&format!("{key} {key}"));
        match receiver.try_recv().unwrap() {
            CommandMessage::Approve(id) => {
                assert!(approve);
                assert_eq!(id, key);
            }
            CommandMessage::Deny(id) => {
                assert!(!approve);
                assert_eq!(id, key);
            }
            other => panic!("unexpected {other:?}"),
        }
        assert!(receiver.try_recv().is_err());
        visual.update(|window, cx| {
            view.update(cx, |view, cx| {
                view.apply_worker_message(
                    WorkerMessage::Status(StatusResult {
                        session: Some(session()),
                        state: "idle".into(),
                        pending_approvals: vec![],
                        usage: json!({}),
                        compaction_markers: 0,
                    }),
                    window,
                    cx,
                );
            })
        });
        visual.update(|window, cx| window.draw(cx).clear(cx));
        assert!(visual.debug_bounds("dialog-layer").is_none());
    }
}

#[gpui::test]
fn virtual_transcript_and_session_rows_fit_their_viewports(cx: &mut TestAppContext) {
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|_, cx| {
        view.update(cx, |view, cx| {
            view.state.sessions[0].name = "long name\n".repeat(60);
            view.state.transcript = (0..1000)
                .map(|index| {
                    if index % 2 == 1 {
                        TranscriptEntry::User("a wrapped user message ".repeat(100))
                    } else {
                        TranscriptEntry::Assistant(
                            format!(
                                "# Heading\n\n- item\n\n```rust\nlet value = \"{}\";\n```",
                                "x".repeat(2000)
                            )
                            .into(),
                        )
                    }
                })
                .collect();
            view.transcript
                .update(cx, |scroll, cx| scroll.reset(1000, cx));
            cx.notify();
        })
    });
    for mode in [
        gpui_kit::component::ThemeMode::Light,
        gpui_kit::component::ThemeMode::Dark,
    ] {
        visual.update(|window, cx| {
            Theme::change(mode, Some(window), cx);
            window.draw(cx).clear(cx);
        });
        let transcript = visual.debug_bounds("transcript-viewport").unwrap();
        let composer = visual.debug_bounds("composer").unwrap();
        let row = visual.debug_bounds("transcript-row").unwrap();
        let session = visual.debug_bounds("session-row").unwrap();
        assert!(transcript.bottom() <= composer.top());
        assert!(row.size.width <= transcript.size.width);
        assert!(session.size.height <= px(44.));
        assert!(session.size.width <= px(280.));
        assert!(transcript.size.height > px(300.));
        visual.update(|window, cx| {
            let viewport = transcript.scale(window.scale_factor());
            let composer_bounds = composer.scale(window.scale_factor());
            // Identify transcript quads by their PAINT attributes alone —
            // primary-color left rail — and exclude the composer's rail
            // explicitly by bounds. The old test filtered by content_mask
            // inside the viewport, which silently DROPPED any leaking quad
            // rather than failing on it. Here we identify without that
            // filter, then assert every quad's content mask (the clipping
            // rectangle the virtual list assigns) stays inside the viewport.
            let quads: Vec<_> = window
                .painted_quads()
                .into_iter()
                .filter(|quad| {
                    quad.border_color == cx.theme().primary
                        && quad.border_widths.left > gpui::ScaledPixels::default()
                        && !(quad.bounds.top() >= composer_bounds.top()
                            && quad.bounds.bottom() <= composer_bounds.bottom())
                })
                .collect();
            assert!(!quads.is_empty(), "user message borders were painted");
            // The 760px window minus header, banner, composer, and footer
            // leaves ≲520px of transcript viewport. With the 22px row rhythm
            // that fits ~24 rows; a healthy virtual list over-renders a small
            // buffer above and below. Anything past that means the list is
            // materialising off-screen work.
            assert!(quads.len() < 20, "the virtual list paints only nearby rows");
            for quad in quads {
                assert!(
                    quad.content_mask.bounds.top() >= viewport.top()
                        && quad.content_mask.bounds.bottom() <= viewport.bottom(),
                    "transcript row content-mask leaks vertically outside the viewport"
                );
                assert!(
                    quad.content_mask.bounds.left() >= viewport.left()
                        && quad.content_mask.bounds.right() <= viewport.right(),
                    "transcript row content-mask leaks horizontally outside the viewport"
                );
            }
        });
    }
}

fn png_bytes() -> Vec<u8> {
    b"\x89PNG\r\n\x1a\n".to_vec()
}

/// A tiny but decodable PNG. Used in tests that need `image_source` to
/// succeed — the header-only `png_bytes()` above passes the parser's magic
/// check but fails the decoder, and now demotes to an Invalid chip.
fn valid_png_bytes() -> Vec<u8> {
    let pixels = image::RgbaImage::from_pixel(4, 4, image::Rgba([255, 0, 0, 255]));
    let mut bytes = std::io::Cursor::new(Vec::new());
    pixels
        .write_to(&mut bytes, image::ImageFormat::Png)
        .unwrap();
    bytes.into_inner()
}

#[gpui::test]
fn transcript_prose_column_caps_at_reading_measure_and_centers(cx: &mut TestAppContext) {
    // ZETA-124 narrowed the prose measure: user / assistant / thinking rows
    // now cap at `prose_max_width(base)` — ~88ch of the base font — while
    // tool receipts and errors keep the wider `TRANSCRIPT_MAX_WIDTH`. The
    // default 1100px window minus the 216px sidebar leaves ~884px of
    // transcript viewport — WIDER than the prose measure at the shipped 13px
    // base — but this test resizes anyway so both the cap and the centering
    // branch stay exercised at every viewport size a peer test might reuse.
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.simulate_resize(gpui::size(px(1600.), px(760.)));
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.transcript = vec![TranscriptEntry::User("wide user turn ".repeat(500))];
            view.transcript.update(cx, |scroll, cx| scroll.reset(1, cx));
            cx.notify();
        });
        window.draw(cx).clear(cx);
    });
    let row = visual.debug_bounds("transcript-row").unwrap();
    let transcript = visual.debug_bounds("transcript-viewport").unwrap();
    let base = visual.update(|_, cx| cx.theme().font_size);
    let prose_cap = theme::prose_max_width(base);
    // The transcript viewport must exceed the prose cap so the centering
    // branch actually activates — otherwise the row would just fill the
    // available width and the asymmetry check below would trivially pass.
    assert!(
        transcript.size.width > prose_cap,
        "viewport {:?} must exceed the prose cap {prose_cap:?} for centering \
         to matter",
        transcript.size.width
    );
    assert!(row.size.width <= transcript.size.width);
    visual.update(|window, cx| {
        let scale = window.scale_factor();
        let scaled_viewport = transcript.scale(scale);
        let scaled_row = row.scale(scale);
        let scaled_column_cap = px(f32::from(prose_cap)).scale(scale);
        let user_quads: Vec<_> = window
            .painted_quads()
            .into_iter()
            .filter(|quad| {
                quad.border_color == cx.theme().primary
                    && quad.border_widths.left > gpui::ScaledPixels::default()
                    && quad.content_mask.bounds.top() >= scaled_row.top()
                    && quad.content_mask.bounds.bottom() <= scaled_row.bottom()
            })
            .collect();
        assert!(!user_quads.is_empty(), "user rail was painted");
        // The user rectangle sits inside the bounded inner column: prose
        // cap minus 16px horizontal padding on each side (`.px_4()`). The
        // rectangle's quad bounds are its border-box, so a 3px left-rail
        // adds up to 3px to the observed width — allow that plus a few
        // pixels of rendering pipeline rounding.
        let inner_column = scaled_column_cap - px(32.).scale(scale);
        let tolerance = px(8.).scale(scale);
        for quad in user_quads {
            let width = quad.bounds.size.width;
            let delta = if width > inner_column {
                width - inner_column
            } else {
                inner_column - width
            };
            assert!(
                delta <= tolerance,
                "user rectangle width {:?} must land on the inner prose \
                 column {:?} (prose_cap {prose_cap:?} minus 32px padding)",
                width,
                inner_column,
            );
            let left_gap = quad.bounds.left() - scaled_viewport.left();
            let right_gap = scaled_viewport.right() - quad.bounds.right();
            let asymmetry = if left_gap > right_gap {
                left_gap - right_gap
            } else {
                right_gap - left_gap
            };
            assert!(
                asymmetry <= tolerance,
                "column not centered: left {left_gap:?}, right {right_gap:?}"
            );
        }
    });
}

#[gpui::test]
fn assistant_row_is_naked_and_carries_no_bg_or_rail(cx: &mut TestAppContext) {
    // The wiki assistant turn has NO frame — 2px vertical breath, no bg, no
    // border, no rail. Regressing to a card style would fight the mono prose.
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.transcript = vec![TranscriptEntry::Assistant("plain answer".into())];
            view.transcript.update(cx, |scroll, cx| scroll.reset(1, cx));
            cx.notify();
        });
        window.draw(cx).clear(cx);
    });
    let row = visual.debug_bounds("transcript-row").unwrap();
    visual.update(|window, cx| {
        let theme = cx.theme();
        let scaled_row = row.scale(window.scale_factor());
        let framed: Vec<_> = window
            .painted_quads()
            .into_iter()
            .filter(|quad| {
                let in_row = quad.content_mask.bounds.top() >= scaled_row.top()
                    && quad.content_mask.bounds.bottom() <= scaled_row.bottom();
                let has_left_rail = quad.border_widths.left > gpui::ScaledPixels::default();
                let element_fill = quad.background == theme.muted.into()
                    || quad.background == theme.sidebar.into();
                in_row && (has_left_rail || element_fill)
            })
            .collect();
        assert!(
            framed.is_empty(),
            "naked assistant row painted framing chrome: {} quads",
            framed.len()
        );
    });
}

#[gpui::test]
fn assistant_markdown_style_pins_the_zeta_110_clipping_guards(cx: &mut TestAppContext) {
    // ZETA-110 r1 fixed the assistant-row table clipping by (a) opting the
    // TextView table into gpui-base's SCROLL layout so column widths come
    // from measured glyph runs, and (b) forcing per-cell nowrap so the
    // column floors are raised to the full content width — mentally revert
    // either lever and inline `code` chips in tool tables lose their
    // trailing glyph again. r1 also flattened the cell borders and moved
    // the inline chip off the raw accent onto a subtle text-normal wash
    // sourced from the theme (routed through `cx.theme()` so the gpui-kit
    // rich-text default — solid accent — cannot leak in through any
    // future TextView caller that forgets a local style override).
    //
    // Only paint-probe / color tests guarded these fields before; a
    // scroll-mode or nowrap regression would compile and paint the same
    // colors while quietly reintroducing the clipping. This test pins the
    // style struct's shape directly, so each of the four levers below
    // fails a named assert if reverted.
    cx.update(gpui_kit::init);
    cx.update(theme::apply);
    cx.update(|cx| {
        let style = super::transcript_render::assistant_markdown_style(cx);
        let theme = cx.theme();

        // Scroll mode — without this, the table falls back to the wrap
        // layout that budgets columns by character count and clamps cells
        // to `overflow_hidden`, which is what sliced the chip glyphs in
        // the smoke shot before the fix.
        assert_eq!(
            style.table.overflow.x,
            Some(gpui::Overflow::Scroll),
            "table must opt into gpui-base's SCROLL layout — reverting this \
             brings back the wrap layout's character-count column budget",
        );
        assert!(
            style.table.overflow.y.is_none(),
            "table must leave overflow.y unset so vertical scroll stays \
             with the transcript column, not the individual table",
        );

        // Per-cell nowrap — the load-bearing floor-raise. gpui-base's own
        // docs on `style.table_cell.white_space = Nowrap` say it "keeps
        // the cell text on a single line, and the floors are raised to
        // the full content widths so the single-line columns never
        // shrink." That is what pushes the Tool column's floor out to the
        // widest chip's shaped width so the trailing glyph lands inside
        // the cell rather than outside its `overflow_hidden()`.
        assert_eq!(
            style.table_cell.text.white_space,
            Some(gpui::WhiteSpace::Nowrap),
            "table cells must carry nowrap so column floors are raised \
             to their shaped-glyph widths — reverting this restarts the \
             r1 inline-code-chip clipping",
        );

        // Transparent cell border — kills the per-cell vertical grid so
        // rows read flat (the row-bottom rule rides on the row div, not
        // the cell, and survives this override).
        assert_eq!(
            style.table_cell.border_color,
            Some(gpui::transparent_black()),
            "cell border must be transparent so the table reads as flat \
             rows without a per-cell grid",
        );

        // Inline-code chip routed through the theme, not palette::
        // directly. `secondary_hover` is the ~6% text-normal wash the
        // wiki paints on `.markdown-preview-view code`; `foreground`
        // paints the glyph at normal-tier text — together they keep the
        // chip from competing with real accent chrome.
        assert_eq!(
            style.inline_code.background_color,
            Some(theme.secondary_hover),
            "inline-code chip bg must ride the theme's secondary_hover \
             wash — a fallback to accent paints the solid violet slab \
             the ticket set out to remove",
        );
        assert_eq!(
            style.inline_code.color,
            Some(theme.foreground),
            "inline-code glyph must be normal-tier text so the chip \
             reads as a quiet annotation, not accent chrome",
        );

        // The wash the theme routes into the chip must itself stay
        // subtle and share the text hue — a regression that promoted
        // secondary_hover to a solid fill would silently loud-up every
        // inline chip AND every hover state at once, so a named assert
        // here beats waiting for the visible regression.
        let wash = theme.secondary_hover;
        let text = theme.foreground;
        assert!(
            wash.a > 0.02 && wash.a < 0.10,
            "inline-code wash alpha {:.3} must land in the wiki's \
             ~6% text-normal band",
            wash.a
        );
        assert_eq!(wash.h, text.h);
        assert_eq!(wash.s, text.s);
        assert_eq!(wash.l, text.l);
    });
}

/// A minimal Render that drops one UNSTYLED `TextView::markdown` (inline
/// code, no local `.style(...)`) into a window. It exists only so the
/// r3 default-install guard below can inspect the paint scene without
/// dragging the whole `ZetaView`/`Root` chrome into a probe render.
struct InlineCodeDefaultProbe;

impl gpui::Render for InlineCodeDefaultProbe {
    fn render(
        &mut self,
        _: &mut gpui::Window,
        _: &mut gpui::Context<Self>,
    ) -> impl gpui::IntoElement {
        gpui::div()
            .size_full()
            .p_4()
            .child(gpui_kit::component::text::TextView::markdown(
                "app-default-inline-code-probe",
                "hello `world` there",
            ))
    }
}

#[gpui::test]
fn app_wide_text_view_default_paints_inline_code_on_the_subtle_wash(cx: &mut TestAppContext) {
    // ZETA-110 r3 root cause: gpui-component's `base_text_view_style`
    // hardcodes the app-wide inline-code default to
    // `HighlightStyle { background_color: Some(theme.accent) }`. Round 2
    // only styled the ASSISTANT renderer locally, so every OTHER
    // `TextView::markdown` (input popovers, error surfaces, previews —
    // anywhere a caller does not pass a local `.style(...)`) still
    // painted inline code on the solid violet slab.
    //
    // `theme::apply` overwrites `TextViewDefaults::global(cx)` AFTER
    // `Theme::sync_base(cx)` so the app-wide default rides the same
    // ~6% text-normal wash the assistant renderer already uses. This
    // probe exercises that path: an UNSTYLED `TextView::markdown` with
    // one inline `code` chip goes through the paint pipeline, and the
    // resulting quads must carry the subtle wash — NEVER the accent.
    //
    // If the install line in `theme::apply` is removed, `sync_base`
    // still puts gpui-component's accent-inline default in place and
    // the inline chip paints on solid violet, so the second assertion
    // (no accent-backed quad) fails.
    cx.update(gpui_kit::init);
    cx.update(theme::apply);
    let window = cx.open_window(gpui::size(px(400.), px(160.)), |_, _| {
        InlineCodeDefaultProbe
    });
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| window.draw(cx).clear(cx));
    visual.update(|window, cx| {
        let theme = cx.theme();
        let subtle = theme.secondary_hover;
        let accent = theme.accent;
        assert_ne!(
            subtle, accent,
            "the subtle wash and solid accent must be distinct tokens \
             or this guard degenerates to a tautology",
        );
        let quads = window.painted_quads();
        let hit = |color: gpui::Hsla| quads.iter().any(|quad| quad.background == color.into());
        assert!(
            hit(subtle),
            "unstyled TextView::markdown must paint the inline-code chip \
             on the theme's secondary_hover wash — a removal of the \
             `TextViewDefaults::install` call in theme::apply restores \
             gpui-component's solid-accent default",
        );
        assert!(
            !hit(accent),
            "unstyled TextView::markdown must NOT paint any quad on \
             solid theme.accent — the r3 regression paints the inline \
             chip on the raw violet slab this test guards against",
        );
        // The Base-layer defaults must still carry the code-block
        // syntax highlighter installed by `Theme::sync_base` — the
        // r3 install rebuilds `TextViewDefaults` and any refactor that
        // drops the `TextViewDefaults::global(cx).with_style(...)` clone
        // (and installs a fresh `TextViewDefaults::new()` instead)
        // would silently strip fenced-code coloring app-wide.
        assert!(
            gpui_kit::base::TextViewDefaults::global(cx).has_code_block_highlighter(),
            "the installed defaults must retain the code-block syntax \
             highlighter — dropping it kills fence colors app-wide",
        );
    });
}

#[gpui::test]
fn tool_state_paints_by_color_alone_and_expanded_body_borders_by_error(cx: &mut TestAppContext) {
    // The wiki contract encodes tool state through COLOR ONLY on the actual
    // verb and detail text: running paints at foreground, done fades to
    // muted, failed lands on danger. No textual "[working]/[done]/[failed]"
    // marker may reach the row. This guard drives real tool rows through the
    // state layer and asserts on: (a) the `visible_text` seam (no bracketed
    // marker), (b) the `tool_state_color` helper (contract token per state),
    // and (c) the paint-probe (the ACTUAL color the render layer applied via
    // `.text_color(state_color)`). Since gpui's public test surface exposes
    // painted quads but not the scene's glyph sprites, the probe binds the
    // renderer's applied color to the assertion; a regression that
    // hard-codes a wrong token instead of routing through the helper is
    // caught because it either records the wrong color or records nothing.
    use zeta_gui::state::ToolState;
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    for case in [ToolState::Running, ToolState::Done, ToolState::Failed] {
        visual.update(|window, cx| {
            view.update(cx, |view, cx| {
                view.state.transcript.clear();
                view.transcript.update(cx, |scroll, cx| scroll.reset(0, cx));
                let call = ToolCall {
                    id: "receipt".into(),
                    name: "bash".into(),
                    arguments: Default::default(),
                };
                view.apply_worker_message(
                    WorkerMessage::Event(ServerEvent::ToolStart {
                        session_id: view.state.active_session.clone(),
                        tool_call: call.clone(),
                        data: json!({}),
                    }),
                    window,
                    cx,
                );
                if !matches!(case, ToolState::Running) {
                    let is_error = matches!(case, ToolState::Failed);
                    view.apply_worker_message(
                        WorkerMessage::Event(ServerEvent::ToolEnd {
                            session_id: view.state.active_session.clone(),
                            tool_call: call,
                            tool_result: Some(zeta_gui::client::ToolResult {
                                tool_call_id: "receipt".into(),
                                content: "line".into(),
                                is_error,
                                is_canceled: false,
                                structured_content: None,
                                content_blocks: Vec::new(),
                            }),
                            data: json!({}),
                        }),
                        window,
                        cx,
                    );
                    // Expand collapsed successful rows so the indent-rail body paints.
                    if matches!(case, ToolState::Done) {
                        view.state.toggle_card(0);
                        cx.notify();
                    }
                }
            });
            window.draw(cx).clear(cx);
        });
        // Contract mapping: running=foreground, done=muted_foreground,
        // failed=danger. Same helper the render function reads.
        visual.update(|_, cx| {
            let theme = cx.theme();
            let expected = match case {
                ToolState::Running => theme.foreground,
                ToolState::Done => theme.muted_foreground,
                ToolState::Failed => theme.danger,
            };
            let entry_state = view.read(cx).state.transcript[0].tool_state();
            assert_eq!(entry_state, case, "state classification regressed");
            assert_eq!(
                super::tool_state_color(entry_state, cx),
                expected,
                "tool row state color regressed off the contract token"
            );
            // render_log: every state-colored element in the tool row goes
            // through `state_text` or `record_state`, so the samples for
            // row 0 (verb, detail, chevron) are the actual colors reaching
            // `.text_color(...)`. A mutation that swaps the color argument
            // at any call site either records the wrong color OR drops a
            // required row id from the recorded set — both fail here.
            let samples = super::render_log::samples();
            let recorded: std::collections::HashSet<&str> = samples
                .iter()
                .map(|sample| sample.row_id.as_str())
                .filter(|id| id.starts_with("tool-"))
                .collect();
            // ZETA-125: the tool row's state-colored elements are the
            // excerpt (row's primary text) and the chevron. The tool_label
            // and metadata paint at the muted-foreground tier regardless of
            // state, so they are NOT recorded here.
            let expected_ids: std::collections::HashSet<&str> =
                ["tool-excerpt-0", "tool-chevron-0"].into_iter().collect();
            assert_eq!(
                recorded, expected_ids,
                "render_tool_row must record excerpt and chevron samples for {case:?}"
            );
            for sample in samples
                .iter()
                .filter(|s| expected_ids.contains(s.row_id.as_str()))
            {
                assert_eq!(
                    sample.color, expected,
                    "tool row {case:?} painted {} at the wrong color",
                    sample.row_id
                );
            }
        });
        // The tool_label and excerpt elements paint their bounds — the
        // color check above proves the contract token is on the excerpt;
        // this pins the ZETA-125 debug selectors so a rename regresses.
        assert!(
            visual.debug_bounds("tool-label-0").is_some(),
            "tool_label element must paint for state {case:?}"
        );
        assert!(
            visual.debug_bounds("tool-excerpt-0").is_some(),
            "excerpt element must paint for state {case:?}"
        );
        // Typed row-text model: the tool row's paint set (verb + detail +
        // optional peek/hover/omitted/body) is a `ToolRowText` built by
        // `row_text::build`. No bracketed state marker may reach any field
        // in any state — that would revert contract line 83.
        view.read_with(&visual, |view, _| {
            let entry = &view.state.transcript[0];
            let row = zeta_gui::row_text::build(entry, 0, &view.state.session_view, true);
            for marker in ["[working]", "[done]", "[failed]", "[canceled]"] {
                for text in row.visible_strings() {
                    assert!(
                        !text.contains(marker),
                        "tool row text carried state marker {marker} for {case:?}: {text:?}"
                    );
                }
            }
        });
        // Failed tool bodies auto-expand (see state::ServerEvent::ToolEnd),
        // so the indent-rail assertion still runs for that case. The Done
        // case toggles above; Running has no expanded body.
        if matches!(case, ToolState::Done | ToolState::Failed) {
            let body = visual
                .debug_bounds("tool-output-0")
                .expect("expanded tool body renders");
            visual.update(|window, cx| {
                let theme = cx.theme();
                let scaled_body = body.scale(window.scale_factor());
                let thin_rail = px(f32::from(theme::RAIL_WIDTH_THIN)).scale(window.scale_factor());
                let thick_rail =
                    px(f32::from(theme::RAIL_WIDTH_THICK)).scale(window.scale_factor());
                let rails: Vec<_> = window
                    .painted_quads()
                    .into_iter()
                    .filter(|quad| {
                        let in_body = quad.bounds.top() >= scaled_body.top()
                            && quad.bounds.bottom() <= scaled_body.bottom();
                        let is_thin_rail = quad.border_widths.left >= thin_rail
                            && quad.border_widths.left < thick_rail;
                        in_body && is_thin_rail
                    })
                    .collect();
                let danger_count = rails
                    .iter()
                    .filter(|quad| quad.border_color == theme.danger)
                    .count();
                let neutral_count = rails
                    .iter()
                    .filter(|quad| quad.border_color == theme.border)
                    .count();
                if matches!(case, ToolState::Failed) {
                    assert!(
                        danger_count > 0,
                        "failed tool body must paint its rail in danger"
                    );
                    assert_eq!(
                        neutral_count, 0,
                        "failed tool body must not paint any neutral rails"
                    );
                } else {
                    assert!(
                        neutral_count > 0,
                        "successful tool body must paint its rail in the neutral border color"
                    );
                    assert_eq!(
                        danger_count, 0,
                        "successful tool body must not paint any danger rails"
                    );
                }
            });
        }
    }
}

#[gpui::test]
fn pending_user_rail_paints_a_single_one_pixel_rail(cx: &mut TestAppContext) {
    // Item 8 guard: the queued strip pins to a 1px dashed rail. A revert to
    // the thick 3px rail (contract line 85) or a fill-rail hybrid would
    // read as an active user turn. Assert the exact left-border width via
    // painted_quads — the scene reflects it directly, no probes needed.
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, _| {
            view.pending_user_turn = Some(super::PendingUserTurn {
                text: "queued message".into(),
                failed: false,
            });
        });
        window.draw(cx).clear(cx);
    });
    let pending = visual
        .debug_bounds("composer-pending")
        .expect("queued strip renders");
    visual.update(|window, _| {
        let scaled = pending.scale(window.scale_factor());
        let thin = px(f32::from(theme::RAIL_WIDTH_THIN)).scale(window.scale_factor());
        let thick = px(f32::from(theme::RAIL_WIDTH_THICK)).scale(window.scale_factor());
        let rails: Vec<_> = window
            .painted_quads()
            .into_iter()
            .filter(|quad| {
                quad.bounds.top() >= scaled.top() - gpui::ScaledPixels::from(0.5)
                    && quad.bounds.bottom() <= scaled.bottom() + gpui::ScaledPixels::from(0.5)
                    && quad.border_widths.left > gpui::ScaledPixels::default()
            })
            .collect();
        assert!(!rails.is_empty(), "queued strip must paint a left rail");
        for rail in &rails {
            // 1px rail exactly — a 3px revert lands at or above `thick`.
            assert!(
                rail.border_widths.left <= thin + gpui::ScaledPixels::from(0.5),
                "queued rail width {:?} exceeds RAIL_WIDTH_THIN {:?} — thick revert",
                rail.border_widths.left,
                thin
            );
            assert!(
                rail.border_widths.left < thick,
                "queued rail width must never reach RAIL_WIDTH_THICK"
            );
            // Style must be dashed — a revert to solid drops the "queued"
            // signal and reads as an active-turn rail. `Quad::border_style`
            // defaults to `Solid`, so this catches a `.border_dashed()`
            // removal directly at the paint layer.
            assert_eq!(
                rail.border_style,
                gpui::BorderStyle::Dashed,
                "queued rail must paint dashed, not solid"
            );
        }
    });
}

#[test]
fn scroll_sync_translates_every_edit_in_order() {
    // Round-10 structural pin. `apply` returns an ordered edit list; the
    // view MUST apply every edit as its own scroller op, in order. A
    // mutation that drops all but one edit (the round-9 shape) or reverts
    // any removal to a tail splice fails these assertions.
    use zeta_gui::state::TranscriptEdit;
    assert_eq!(
        scroll_sync(6, &[TranscriptEdit::Remove(4)], false),
        vec![ScrollSync::Splice(4..5, 0)],
    );
    assert_eq!(
        scroll_sync(4, &[TranscriptEdit::Insert(3)], false),
        vec![ScrollSync::Splice(3..3, 1)],
    );
    assert_eq!(
        scroll_sync(5, &[TranscriptEdit::Remeasure(2)], false),
        vec![ScrollSync::Remeasure(2)],
    );
    assert_eq!(scroll_sync(5, &[], false), Vec::<ScrollSync>::new());
    // Replace (history swap / session switch) resets regardless of edits.
    assert_eq!(scroll_sync(8, &[], true), vec![ScrollSync::Reset(8)],);
    // Compound edit: append a Thinking row AND remove a middle assistant
    // row in ONE reconcile. The view must splice BOTH operations — a
    // one-action signal collapses to a single op and desyncs the list.
    assert_eq!(
        scroll_sync(
            5,
            &[TranscriptEdit::Insert(4), TranscriptEdit::Remove(2)],
            false,
        ),
        vec![ScrollSync::Splice(4..4, 1), ScrollSync::Splice(2..3, 0)],
    );
}

#[gpui::test]
fn middle_row_removal_syncs_the_virtual_list_at_the_exact_index(cx: &mut TestAppContext) {
    // End-to-end pin for the state → view sync. Streams a mixed turn that
    // ends with the final message equal to the FIRST streamed fragment, so
    // the reconciler drops the middle assistant row rather than blanking
    // it. The virtual list must land on the reconciled item count and the
    // full flow — apply_worker_message → scroll_sync → MessageScrollerState
    // — must not panic on the middle-slot splice.
    use zeta_gui::client::{ContentBlock, Message};
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    let session_id = view.read_with(&visual, |view, _| view.state.active_session.clone());
    let stream_events = vec![
        ServerEvent::TurnStart {
            session_id: session_id.clone(),
            data: json!({}),
        },
        ServerEvent::AssistantDelta {
            session_id: session_id.clone(),
            delta: "pre".into(),
            kind: "assistant".into(),
        },
        ServerEvent::ToolStart {
            session_id: session_id.clone(),
            tool_call: ToolCall {
                id: "tool-a".into(),
                name: "read".into(),
                arguments: serde_json::from_value(json!({"path": "README.md"})).unwrap(),
            },
            data: json!({}),
        },
        ServerEvent::AssistantDelta {
            session_id: session_id.clone(),
            delta: "post".into(),
            kind: "assistant".into(),
        },
        ServerEvent::ToolStart {
            session_id: session_id.clone(),
            tool_call: ToolCall {
                id: "tool-b".into(),
                name: "list".into(),
                arguments: serde_json::from_value(json!({"path": "gui/src"})).unwrap(),
            },
            data: json!({}),
        },
    ];
    for event in stream_events {
        visual.update(|window, cx| {
            view.update(cx, |view, cx| {
                view.apply_worker_message(WorkerMessage::Event(event), window, cx);
            });
        });
    }
    visual.update(|window, cx| window.draw(cx).clear(cx));
    visual.update(|_, cx| {
        view.update(cx, |view, cx| {
            assert_eq!(view.state.transcript.len(), 4);
            assert_eq!(view.transcript.read(cx).item_count(), 4);
        });
    });
    // Final "pre" trims the trailing Assistant("post") at index 2, leaving
    // a Tool row past the removed slot. The sync must splice the middle
    // slot's cache — a tail splice would corrupt the survivor's metadata.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.apply_worker_message(
                WorkerMessage::Event(ServerEvent::AssistantMessage {
                    session_id,
                    message: Message {
                        role: "assistant".into(),
                        content: vec![ContentBlock::Text { text: "pre".into() }],
                    },
                }),
                window,
                cx,
            );
        });
    });
    visual.update(|window, cx| window.draw(cx).clear(cx));
    visual.update(|_, cx| {
        view.update(cx, |view, cx| {
            assert_eq!(view.state.transcript.len(), 3);
            assert_eq!(
                view.transcript.read(cx).item_count(),
                3,
                "virtual list item count must track the reconciled transcript"
            );
        });
    });
    // Post-reconcile transcript: [Assistant("pre"), Tool, Tool]. Adjacent
    // tool rows carry a zero row-gap per the wiki contract; a splice at
    // the wrong slot would leave the second tool anchored below the
    // removed assistant's cached height.
    let tool_a = visual
        .debug_bounds("tool-receipt-1")
        .expect("first tool row must paint");
    let tool_b = visual
        .debug_bounds("tool-receipt-2")
        .expect("second tool row must paint");
    assert!(
        tool_b.top() >= tool_a.top(),
        "tool rows must remain in transcript order"
    );
    let gap = tool_b.top() - tool_a.bottom();
    assert!(
        gap < px(8.),
        "adjacent tool rows must sit at zero gap (measured gap={gap:?})"
    );
}

#[gpui::test]
fn compound_reconcile_syncs_the_virtual_list_row_for_row(cx: &mut TestAppContext) {
    // Round-10 mutation gate. A single AssistantMessage can append a
    // Thinking marker AND drop a middle assistant row in the same
    // reconcile. `apply` returns BOTH edits in order and the view must
    // splice each — a mutation that keeps only one edit leaves the
    // virtual list off by one item from the transcript. This end-to-end
    // pin fails any such collapse (name: compound-edit desync).
    use zeta_gui::client::{ContentBlock, Message};
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    let session_id = view.read_with(&visual, |view, _| view.state.active_session.clone());
    let stream_events = vec![
        ServerEvent::TurnStart {
            session_id: session_id.clone(),
            data: json!({}),
        },
        ServerEvent::AssistantDelta {
            session_id: session_id.clone(),
            delta: "pre".into(),
            kind: "assistant".into(),
        },
        ServerEvent::ToolStart {
            session_id: session_id.clone(),
            tool_call: ToolCall {
                id: "tool-a".into(),
                name: "read".into(),
                arguments: serde_json::from_value(json!({"path": "README.md"})).unwrap(),
            },
            data: json!({}),
        },
        ServerEvent::AssistantDelta {
            session_id: session_id.clone(),
            delta: "post".into(),
            kind: "assistant".into(),
        },
        ServerEvent::ToolStart {
            session_id: session_id.clone(),
            tool_call: ToolCall {
                id: "tool-b".into(),
                name: "list".into(),
                arguments: serde_json::from_value(json!({"path": "gui/src"})).unwrap(),
            },
            data: json!({}),
        },
    ];
    for event in stream_events {
        visual.update(|window, cx| {
            view.update(cx, |view, cx| {
                view.apply_worker_message(WorkerMessage::Event(event), window, cx);
            });
        });
    }
    visual.update(|window, cx| window.draw(cx).clear(cx));
    visual.update(|_, cx| {
        view.update(cx, |view, cx| {
            assert_eq!(view.state.transcript.len(), 4);
            assert_eq!(view.transcript.read(cx).item_count(), 4);
        });
    });
    // Final "pre" with a Thinking block: append Thinking AND drop the
    // middle Assistant("post"). Transcript ends at four items — the
    // scroller item count MUST equal that; the compound reconcile is
    // exactly the shape that the round-9 single-action signal missed.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.apply_worker_message(
                WorkerMessage::Event(ServerEvent::AssistantMessage {
                    session_id,
                    message: Message {
                        role: "assistant".into(),
                        content: vec![
                            ContentBlock::Thinking {
                                text: "hidden".into(),
                            },
                            ContentBlock::Text { text: "pre".into() },
                        ],
                    },
                }),
                window,
                cx,
            );
        });
    });
    visual.update(|window, cx| window.draw(cx).clear(cx));
    visual.update(|_, cx| {
        view.update(cx, |view, cx| {
            assert_eq!(view.state.transcript.len(), 4);
            assert_eq!(
                view.transcript.read(cx).item_count(),
                4,
                "compound reconcile MUST leave the virtual list aligned to \
                 the transcript — a dropped edit desyncs the counts"
            );
        });
    });
}

#[gpui::test]
fn variable_height_survivor_positions_stay_stable_after_middle_removal(cx: &mut TestAppContext) {
    // Round-10 stability pin. Build several rows of DIFFERENT painted
    // heights after the removed slot, scroll AWAY from the tail so the
    // splice cannot rely on tail follow-mode, then drop a middle row.
    // Every surviving row past the removal must sit at exactly its
    // pre-removal top MINUS the removed row's height — otherwise the
    // splice landed on the wrong slot and the survivors carry stale
    // metadata. A tail-splice mutation shifts these numbers.
    use zeta_gui::client::{ContentBlock, Message};
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    let session_id = view.read_with(&visual, |view, _| view.state.active_session.clone());
    let bulk = "line ".repeat(60);
    let stream_events = vec![
        ServerEvent::TurnStart {
            session_id: session_id.clone(),
            data: json!({}),
        },
        // Row 0: short assistant fragment (streamed).
        ServerEvent::AssistantDelta {
            session_id: session_id.clone(),
            delta: "pre".into(),
            kind: "assistant".into(),
        },
        // Row 1: first tool receipt.
        ServerEvent::ToolStart {
            session_id: session_id.clone(),
            tool_call: ToolCall {
                id: "tool-a".into(),
                name: "read".into(),
                arguments: serde_json::from_value(json!({"path": "README.md"})).unwrap(),
            },
            data: json!({}),
        },
        // Row 2: streamed assistant fragment that will be dropped.
        ServerEvent::AssistantDelta {
            session_id: session_id.clone(),
            delta: "post".into(),
            kind: "assistant".into(),
        },
        // Row 3: second tool receipt with a large output (different height).
        ServerEvent::ToolStart {
            session_id: session_id.clone(),
            tool_call: ToolCall {
                id: "tool-b".into(),
                name: "list".into(),
                arguments: serde_json::from_value(json!({"path": "gui/src"})).unwrap(),
            },
            data: json!({}),
        },
        ServerEvent::ToolOutput {
            session_id: session_id.clone(),
            tool_call: ToolCall {
                id: "tool-b".into(),
                name: "list".into(),
                arguments: serde_json::from_value(json!({"path": "gui/src"})).unwrap(),
            },
            output: bulk.clone(),
            data: json!({}),
        },
        // Row 4: third tool receipt (short again).
        ServerEvent::ToolStart {
            session_id: session_id.clone(),
            tool_call: ToolCall {
                id: "tool-c".into(),
                name: "read".into(),
                arguments: serde_json::from_value(json!({"path": "Cargo.toml"})).unwrap(),
            },
            data: json!({}),
        },
    ];
    for event in stream_events {
        visual.update(|window, cx| {
            view.update(cx, |view, cx| {
                view.apply_worker_message(WorkerMessage::Event(event), window, cx);
            });
        });
    }
    // Expand the tall tool receipt so its painted height differs from
    // the short ones — the point of the test is heights that are NOT
    // uniform.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.toggle_card(3);
            view.transcript.update(cx, |scroll, cx| {
                scroll.remeasure_items(3..4, cx);
            });
            cx.notify();
        });
        window.draw(cx).clear(cx);
    });
    // Scroll away from the tail. `is_scrolled_up()` is truthy only when
    // the user has left tail-follow — assert the split before mutating.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.transcript.update(cx, |scroll, cx| {
                scroll.scroll_to_item(0, cx);
            });
        });
        window.draw(cx).clear(cx);
    });
    visual.update(|_, cx| {
        view.read_with(cx, |view, cx| {
            assert_eq!(view.state.transcript.len(), 5);
            assert_eq!(view.transcript.read(cx).item_count(), 5);
        });
    });
    let tool_b_top_before = visual
        .debug_bounds("tool-receipt-3")
        .expect("tall tool row must paint")
        .top();
    let tool_c_top_before = visual
        .debug_bounds("tool-receipt-4")
        .expect("trailing short tool row must paint")
        .top();
    // Reconcile: final "pre" drops Assistant("post") at slot 2. Every
    // surviving row past the removal must shift up by the SAME amount —
    // the height of the dropped row. A tail-splice mutation would leave
    // the tall Tb anchored to Assistant("post")'s stale cached height,
    // so Tb and Tc would end up at inconsistent shifts.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.apply_worker_message(
                WorkerMessage::Event(ServerEvent::AssistantMessage {
                    session_id,
                    message: Message {
                        role: "assistant".into(),
                        content: vec![ContentBlock::Text { text: "pre".into() }],
                    },
                }),
                window,
                cx,
            );
        });
        window.draw(cx).clear(cx);
    });
    visual.update(|_, cx| {
        view.read_with(cx, |view, cx| {
            assert_eq!(view.state.transcript.len(), 4);
            assert_eq!(
                view.transcript.read(cx).item_count(),
                4,
                "middle removal MUST keep scroller count aligned to \
                 the transcript"
            );
        });
    });
    let tool_b_top_after = visual
        .debug_bounds("tool-receipt-2")
        .expect("tall tool row must still paint after removal")
        .top();
    let tool_c_top_after = visual
        .debug_bounds("tool-receipt-3")
        .expect("trailing tool row must still paint after removal")
        .top();
    let tolerance = px(1.);
    let tall_shift = tool_b_top_before - tool_b_top_after;
    let tail_shift = tool_c_top_before - tool_c_top_after;
    assert!(
        tall_shift > px(0.),
        "tall tool row must shift up after the middle removal \
         (shift={tall_shift:?})"
    );
    assert!(
        tail_shift > px(0.),
        "trailing tool row must shift up after the middle removal \
         (shift={tail_shift:?})"
    );
    assert!(
        (tall_shift - tail_shift).abs() <= tolerance,
        "surviving rows past the removal MUST shift up by an equal \
         amount — differing shifts (tall={tall_shift:?}, \
         tail={tail_shift:?}) mean the splice landed on the wrong slot \
         and the cache under one survivor is stale"
    );
    // Additional invariant: adjacent tool rows sit at zero row-gap per
    // the wiki contract. A wrong-slot splice leaves the second tool
    // anchored below the removed row's cached height, opening a gap.
    let tool_a_after = visual
        .debug_bounds("tool-receipt-1")
        .expect("first tool row must paint")
        .bottom();
    let gap = tool_b_top_after - tool_a_after;
    assert!(
        gap < px(8.),
        "adjacent tool rows must sit at zero gap after middle removal \
         (measured gap={gap:?})"
    );
}

#[test]
fn every_row_text_flows_through_the_typed_row_text_model() {
    // Sentinel sweep across every `TranscriptEntry` variant: the row-text
    // model iterates every field a renderer paints (content + attachments +
    // labels + hints + body), so a marker anywhere on the row fails here.
    // Together with the renderer-literal fence, the sweep guarantees no
    // stray "[working]/[done]/[failed]/[canceled]" text lands on a row.
    use zeta_gui::cards::Card;
    use zeta_gui::row_text::{self, chrome};
    use zeta_gui::state::ToolReceiptKey;
    let markers = ["[working]", "[done]", "[failed]", "[canceled]"];
    let session_view = zeta_gui::session::SessionView::default();
    let entries = [
        TranscriptEntry::User("hi".into()),
        TranscriptEntry::Assistant("hello".into()),
        TranscriptEntry::Thinking,
        TranscriptEntry::Error {
            message: "boom".into(),
            settings_action: true,
            login_provider: None,
        },
        TranscriptEntry::Tool {
            key: ToolReceiptKey {
                session_id: None,
                agent_instance_id: None,
                tool_call_id: "id".into(),
            },
            name: "bash".into(),
            excerpt: "echo hello".into(),
            summary: "echo".into(),
            complete: true,
            error: false,
            canceled: false,
            card: Card {
                expanded: true,
                ..Default::default()
            },
        },
    ];
    for entry in &entries {
        let row = row_text::build(entry, 0, &session_view, true);
        for text in row.visible_strings() {
            for marker in markers {
                assert!(
                    !text.contains(marker),
                    "row text carried {marker} in {text:?}"
                );
            }
        }
    }
    for literal in chrome::ALL {
        for marker in markers {
            assert!(
                !literal.contains(marker),
                "chrome literal {literal:?} carried state marker {marker}"
            );
        }
    }
    // The Thinking row's paint set is exactly the generic header — no
    // reasoning body ever leaks into the row's visible strings.
    let thinking = row_text::build(&TranscriptEntry::Thinking, 0, &session_view, true);
    assert_eq!(
        thinking.visible_strings(),
        vec![zeta_gui::state::THINKING_HEADER_LABEL]
    );
}

#[gpui::test]
fn composer_focus_promotes_the_rail_and_lightens_the_fill(cx: &mut TestAppContext) {
    // Focus is the ONLY chrome cue on the composer: rail promotes to full
    // accent AND fill lightens one step. Any border ring or extra outline
    // would fail the wiki contract — so this test paints both blurred and
    // focused states and asserts exact rail width, the actual painted fill
    // promotion, and that no other border/ring lives in the composer bounds.
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);

    // Blurred: force focus off the composer via a fresh focus handle.
    visual.update(|window, cx| {
        let handle = cx.focus_handle();
        window.focus(&handle, cx);
        window.draw(cx).clear(cx);
    });
    let composer = visual.debug_bounds("composer").unwrap();
    let (blurred_rail_alpha, blurred_fill) = visual.update(|window, cx| {
        let theme = cx.theme();
        let scaled = composer.scale(window.scale_factor());
        let rail_width_scaled = px(f32::from(theme::RAIL_WIDTH_THICK)).scale(window.scale_factor());
        let composer_quads: Vec<_> = window
            .painted_quads()
            .into_iter()
            .filter(|quad| {
                quad.bounds.top() >= scaled.top() && quad.bounds.bottom() <= scaled.bottom()
            })
            .collect();
        let rail = composer_quads
            .iter()
            .find(|quad| {
                quad.border_widths.left >= rail_width_scaled
                    && quad.bounds.left() <= scaled.left() + gpui::ScaledPixels::from(1.0)
            })
            .expect("blurred composer paints a left rail");
        // Rail must land on the DIM accent (rail alpha < full accent).
        assert!(rail.border_color.a < theme.primary.a);
        assert!((rail.border_color.h - theme.primary.h).abs() < 0.01);
        // Rail width is EXACTLY the wiki thick rail — a 2px regression would
        // pass a `>=` check but slip under this equality.
        assert!(rail.border_widths.left <= rail_width_scaled + gpui::ScaledPixels::from(0.5));
        // No ring: no non-left border on any composer quad. The rail is the
        // only chrome; a border ring elsewhere would trip this.
        for quad in &composer_quads {
            assert!(
                quad.border_widths.top == gpui::ScaledPixels::default(),
                "composer must not paint a top border"
            );
            assert!(
                quad.border_widths.right == gpui::ScaledPixels::default(),
                "composer must not paint a right border"
            );
            assert!(
                quad.border_widths.bottom == gpui::ScaledPixels::default(),
                "composer must not paint a bottom border"
            );
        }
        // Blurred fill is the ambient element surface.
        let fill_quad = composer_quads
            .iter()
            .find(|quad| {
                quad.background == theme.muted.into()
                    || quad.background == theme::palette::composer_focus_fill().into()
            })
            .expect("blurred composer paints its fill");
        assert_eq!(
            fill_quad.background,
            theme.muted.into(),
            "blurred composer must sit on the ambient element surface"
        );
        (rail.border_color.a, fill_quad.background)
    });

    // Focused: restore focus to the composer input.
    visual.update(|window, cx| {
        let handle = view.read(cx).composer.read(cx).focus_handle(cx);
        window.focus(&handle, cx);
        window.draw(cx).clear(cx);
    });
    visual.update(|window, cx| {
        let theme = cx.theme();
        let scaled = composer.scale(window.scale_factor());
        let rail_width_scaled = px(f32::from(theme::RAIL_WIDTH_THICK)).scale(window.scale_factor());
        let composer_quads: Vec<_> = window
            .painted_quads()
            .into_iter()
            .filter(|quad| {
                quad.bounds.top() >= scaled.top() && quad.bounds.bottom() <= scaled.bottom()
            })
            .collect();
        let rail = composer_quads
            .iter()
            .find(|quad| {
                quad.border_widths.left >= rail_width_scaled
                    && quad.bounds.left() <= scaled.left() + gpui::ScaledPixels::from(1.0)
                    && quad.border_color == theme.primary
            })
            .expect("focused composer paints a full-accent left rail");
        // Rail width remains at the wiki thick rail — not the ring style.
        assert!(rail.border_widths.left <= rail_width_scaled + gpui::ScaledPixels::from(0.5));
        // No border ring anywhere in the composer.
        for quad in &composer_quads {
            assert!(
                quad.border_widths.top == gpui::ScaledPixels::default(),
                "focused composer must not paint a top border"
            );
            assert!(
                quad.border_widths.right == gpui::ScaledPixels::default(),
                "focused composer must not paint a right border"
            );
            assert!(
                quad.border_widths.bottom == gpui::ScaledPixels::default(),
                "focused composer must not paint a bottom border"
            );
        }
        // Focus must promote the rail alpha above the blurred alpha AND change
        // the composer fill to the lighter step — asserting both prevents a
        // regression that promotes only one.
        assert!(
            rail.border_color.a > blurred_rail_alpha,
            "focus must strengthen the rail alpha"
        );
        let fill_quad = composer_quads
            .iter()
            .find(|quad| {
                quad.background == theme.muted.into()
                    || quad.background == theme::palette::composer_focus_fill().into()
            })
            .expect("focused composer paints its fill");
        assert_eq!(
            fill_quad.background,
            theme::palette::composer_focus_fill().into(),
            "focus must lighten the composer fill one step"
        );
        assert_ne!(
            fill_quad.background, blurred_fill,
            "focused fill must differ from the blurred fill"
        );
    });
}

#[gpui::test]
fn adjacent_tool_rows_have_zero_gap_between_them(cx: &mut TestAppContext) {
    // A run of tool receipts is visually a single column in the wiki. Non-tool
    // neighbours reintroduce the 14px rhythm on both sides so a mixed sequence
    // still breathes.
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    let tool = |id: &str| TranscriptEntry::Tool {
        key: zeta_gui::state::ToolReceiptKey {
            session_id: None,
            agent_instance_id: None,
            tool_call_id: id.into(),
        },
        name: "bash".into(),
        excerpt: id.into(),
        summary: id.into(),
        complete: true,
        error: false,
        canceled: false,
        card: zeta_gui::cards::Card::default(),
    };
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.transcript = vec![
                TranscriptEntry::User("q".into()),
                tool("a"),
                tool("b"),
                tool("c"),
                TranscriptEntry::Assistant("plain answer".into()),
            ];
            // ZETA-125: a run of 3+ tool receipts collapses into a group
            // summary row unless the group is expanded. This test measures
            // the zero-gap contract between INDIVIDUAL receipts, so expand
            // the group so the individual rows paint their bounds.
            view.state.tool_group_expanded.insert("a".into(), true);
            view.transcript.update(cx, |scroll, cx| scroll.reset(5, cx));
            cx.notify();
        });
        window.draw(cx).clear(cx);
    });
    let tool_a = visual.debug_bounds("tool-receipt-1").unwrap();
    let tool_b = visual.debug_bounds("tool-receipt-2").unwrap();
    let tool_c = visual.debug_bounds("tool-receipt-3").unwrap();
    // Zero gap between adjacent tools; the receipts stack immediately.
    let gap_ab = tool_b.top() - tool_a.bottom();
    let gap_bc = tool_c.top() - tool_b.bottom();
    assert!(gap_ab <= px(2.), "adjacent tool rows must sit flush");
    assert!(gap_bc <= px(2.));
}

#[gpui::test]
fn branches_render_and_click_dispatches_switch_branch(cx: &mut TestAppContext) {
    let (window, view, receiver) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.session_view.available = true;
            view.state.session_view.branches = vec![
                Branch {
                    id: "trunk".into(),
                    label: "main".into(),
                    depth: 0,
                    current: true,
                },
                Branch {
                    id: "alt".into(),
                    label: "alternate".into(),
                    depth: 1,
                    current: false,
                },
            ];
            cx.notify();
        });
        window.draw(cx).clear(cx);
    });
    let trunk = visual
        .debug_bounds("branch-row-trunk")
        .expect("current branch renders");
    assert!(trunk.size.width <= px(280.));
    let alt = visual
        .debug_bounds("branch-row-alt")
        .expect("alternate branch renders");
    visual.simulate_click(alt.center(), Default::default());
    let sent = receiver.try_recv().expect("branch switch dispatched");
    assert!(matches!(sent, CommandMessage::SwitchBranch(id) if id == "alt"));
    // An older server without extensions must not surface branches.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.session_view.available = false;
            cx.notify();
        });
        window.draw(cx).clear(cx);
    });
    assert!(visual.debug_bounds("branch-row-trunk").is_none());
    assert!(visual.debug_bounds("branch-row-alt").is_none());
}

#[gpui::test]
fn fork_button_appears_on_user_rows_and_dispatches_fork_message(cx: &mut TestAppContext) {
    let (window, view, receiver) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.session_view.available = true;
            view.state.transcript = vec![TranscriptEntry::User("a wrapped question".into())];
            view.state
                .session_view
                .message_ids
                .insert(0, "msg-1".into());
            view.transcript.update(cx, |scroll, cx| scroll.reset(1, cx));
            cx.notify();
        });
        window.draw(cx).clear(cx);
    });
    let fork = visual
        .debug_bounds("fork-button-0")
        .expect("fork button renders on user row");
    visual.simulate_click(fork.center(), Default::default());
    let sent = receiver.try_recv().expect("fork dispatched");
    assert!(matches!(sent, CommandMessage::ForkMessage(id) if id == "msg-1"));
    // Without an id (server never surfaced this message) the button hides.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.session_view.message_ids.clear();
            cx.notify();
        });
        window.draw(cx).clear(cx);
    });
    assert!(visual.debug_bounds("fork-button-0").is_none());
}

#[gpui::test]
fn refresh_rpc_failure_clears_the_stale_transcript_and_keeps_the_connection(
    cx: &mut TestAppContext,
) {
    // ZETA-96 recovery: when the worker's refresh RPC fails after a branch
    // switch or fork, it emits a Rejected message followed by a Status with
    // session=None. The GUI must drop the previous transcript and keep the
    // connection healthy so the user can create a fresh session.
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.transcript = vec![
                TranscriptEntry::User("stale question".into()),
                TranscriptEntry::User("older question".into()),
            ];
            view.transcript.update(cx, |scroll, cx| scroll.reset(2, cx));
            view.state.session_view.available = true;
            cx.notify();
            view.apply_worker_message(
                WorkerMessage::Rejected("session metadata is missing; refresh failed".into()),
                window,
                cx,
            );
            view.apply_worker_message(
                WorkerMessage::Status(StatusResult {
                    session: None,
                    state: "idle".into(),
                    pending_approvals: vec![],
                    usage: json!({}),
                    compaction_markers: 0,
                }),
                window,
                cx,
            );
        });
        window.draw(cx).clear(cx);
    });
    view.read_with(&visual, |view, _| {
        assert!(
            view.state.transcript.is_empty(),
            "the stale transcript is dropped when the refresh drops the session"
        );
        assert!(view.state.active_session.is_none());
        assert_eq!(view.state.connection, ConnectionState::Connected);
        assert!(view
            .command_error
            .as_ref()
            .is_some_and(|error| error.contains("refresh failed")));
    });
}

#[gpui::test]
fn appearance_controls_reapply_theme_font_and_size_live(cx: &mut TestAppContext) {
    // The prefs file lives under the scoped $ZETA_HOME. Wipe any leftover
    // from a prior failed run so the baseline reads as the shipped default.
    // `wipe_scoped_prefs` forces the scoped home before deriving the path
    // so `cargo test` never touches the developer's real ~/.zeta.
    wipe_scoped_prefs();
    let (window, view, _receiver) = setup(cx);
    // Reset the client's active appearance to defaults so this test does not
    // inherit a theme flipped by a peer test.
    cx.update(theme::apply);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.session_view.available = true;
            view.apply_worker_message(
                WorkerMessage::Settings(
                    SessionSettings {
                        model: "claude-sonnet-4-6".into(),
                        approval_mode: "ask".into(),
                    },
                    ModelCatalog {
                        models: vec!["claude-sonnet-4-6".into()],
                        providers: [("claude-sonnet-4-6".into(), "claude".into())]
                            .into_iter()
                            .collect(),
                    },
                ),
                window,
                cx,
            );
        });
        window.draw(cx).clear(cx);
    });
    // Baseline: opencode / JetBrains Mono / DEFAULT_FONT_SIZE. Assertions
    // read the PER-APP `cx.theme()` so a peer test racing the process-wide
    // ACTIVE cannot flip them out from under us.
    let baseline_bg = visual.update(|_, cx| cx.theme().background);
    let baseline_font_size = visual.update(|_, cx| cx.theme().font_size);
    assert_eq!(baseline_bg, theme::ThemeId::Opencode.palette().canvas);
    assert_eq!(
        visual.update(|_, cx| cx.theme().font_family.as_ref().to_string()),
        theme::DEFAULT_FONT_FAMILY
    );

    // Click a non-default theme; the app-local theme moves off opencode.
    let gruvbox_light = visual
        .debug_bounds("theme-row-gruvbox-light")
        .expect("gruvbox-light theme row renders");
    visual.simulate_click(gruvbox_light.center(), Default::default());
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let after_theme_bg = visual.update(|_, cx| cx.theme().background);
    assert_eq!(
        after_theme_bg,
        theme::ThemeId::GruvboxLight.palette().canvas
    );
    assert_ne!(after_theme_bg, baseline_bg);

    // Click a non-default font family; theme.font_family follows.
    let menlo = visual
        .debug_bounds("font-row-Menlo")
        .expect("Menlo font row renders");
    visual.simulate_click(menlo.center(), Default::default());
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let after_font = visual.update(|_, cx| cx.theme().font_family.as_ref().to_string());
    assert_eq!(after_font, "Menlo");

    // Shrink font size by one step; grow it back. Both should land on the
    // whole-px pick window.
    let shrink = visual
        .debug_bounds("font-size-shrink")
        .expect("shrink control renders");
    visual.simulate_click(shrink.center(), Default::default());
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let smaller = visual.update(|_, cx| cx.theme().font_size);
    assert!(
        f32::from(smaller) < f32::from(baseline_font_size)
            && f32::from(smaller) >= theme::MIN_FONT_SIZE_PX,
        "shrink moved {baseline_font_size:?} -> {smaller:?} within the picker window"
    );
    let grow = visual
        .debug_bounds("font-size-grow")
        .expect("grow control renders");
    visual.simulate_click(grow.center(), Default::default());
    visual.update(|window, cx| window.draw(cx).clear(cx));
    assert_eq!(
        visual.update(|_, cx| cx.theme().font_size),
        baseline_font_size
    );

    // Reset for the next test — the global appearance and the on-disk prefs
    // both persist across the in-process test run.
    cx.update(theme::apply);
    wipe_scoped_prefs();
}

#[gpui::test]
fn settings_modal_swaps_models_and_credential_errors_keep_it_open(cx: &mut TestAppContext) {
    let (window, view, receiver) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.session_view.available = true;
            view.apply_worker_message(
                WorkerMessage::Settings(
                    SessionSettings {
                        model: "claude-sonnet-4-6".into(),
                        approval_mode: "ask".into(),
                    },
                    ModelCatalog {
                        models: vec![
                            "claude-sonnet-4-6".into(),
                            "claude-opus-4-7".into(),
                            "codex-gpt-5".into(),
                        ],
                        providers: [
                            ("claude-sonnet-4-6".into(), "claude".into()),
                            ("claude-opus-4-7".into(), "claude".into()),
                            ("codex-gpt-5".into(), "codex".into()),
                        ]
                        .into_iter()
                        .collect(),
                    },
                ),
                window,
                cx,
            );
        });
        window.draw(cx).clear(cx);
    });
    // The overlay renders; grouped models mark the current one.
    assert!(visual.debug_bounds("settings-overlay").is_some());
    assert!(visual.debug_bounds("model-row-0").is_some());
    let codex_row = visual
        .debug_bounds("model-row-2")
        .expect("codex row renders");
    // Selecting a codex model and clicking Apply dispatches SetSettings across
    // providers with the retained approval mode.
    visual.simulate_click(codex_row.center(), Default::default());
    let apply = visual
        .debug_bounds("settings-apply")
        .expect("apply button renders");
    visual.simulate_click(apply.center(), Default::default());
    let dispatched = receiver.try_recv().expect("settings dispatch");
    assert!(matches!(
        &dispatched,
        CommandMessage::SetSettings(settings)
            if settings.model == "codex-gpt-5" && settings.approval_mode == "ask"
    ));
    // A credential rejection keeps the modal open and surfaces the error inline.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.apply_worker_message(
                WorkerMessage::Rejected("codex credentials missing; run zeta login".into()),
                window,
                cx,
            );
        });
        window.draw(cx).clear(cx);
    });
    assert!(
        visual.debug_bounds("settings-overlay").is_some(),
        "credential errors do not close the settings modal"
    );
    view.read_with(&visual, |view, _| {
        assert!(view
            .settings_error
            .as_ref()
            .is_some_and(|error| error.contains("codex credentials")));
    });
    // Clicking Close dismisses the overlay after an error.
    let close = visual
        .debug_bounds("settings-close")
        .expect("close button renders");
    visual.simulate_click(close.center(), Default::default());
    visual.update(|window, cx| window.draw(cx).clear(cx));
    assert!(visual.debug_bounds("settings-overlay").is_none());
}

#[gpui::test]
fn settings_selection_scrolls_the_current_model_into_view(cx: &mut TestAppContext) {
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.session_view.available = true;
            let mut models: Vec<String> = (0..30).map(|index| format!("claude-{index}")).collect();
            let current = models[24].clone();
            let providers = models
                .iter()
                .map(|model| (model.clone(), "claude".into()))
                .collect();
            models.push(current.clone());
            view.apply_worker_message(
                WorkerMessage::Settings(
                    SessionSettings {
                        model: current,
                        approval_mode: "ask".into(),
                    },
                    ModelCatalog { models, providers },
                ),
                window,
                cx,
            );
        });
        window.draw(cx).clear(cx);
    });
    // A second redraw consumes the scroll target now that child bounds exist.
    visual.update(|window, cx| window.draw(cx).clear(cx));
    // The current model is the 25th; without a scroll it would sit far below
    // the list's 280px viewport (25 × ~32 px rows plus headings).
    let list = visual
        .debug_bounds("model-list")
        .expect("model list bounds");
    let selected = visual
        .debug_bounds("model-row-24")
        .expect("selected model row renders");
    assert!(
        selected.top() >= list.top() && selected.bottom() <= list.bottom(),
        "the current model must be scrolled into the visible list bounds"
    );
    // Keyboard navigation wraps around the list bounds.
    visual.update(|_, cx| {
        view.update(cx, |view, cx| {
            view.move_settings_model(1, cx);
            view.move_settings_model(-3, cx);
        });
    });
    view.read_with(&visual, |view, _| {
        assert_eq!(view.state.session_view.selected_model, 22);
    });
}

#[gpui::test]
fn attachments_paste_shows_chip_and_send_dispatches_send_images(cx: &mut TestAppContext) {
    let (window, view, receiver) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    // Prime the clipboard with a real decodable PNG; the paste hook adds the
    // chip. `png_bytes()` is a header-only stub — decoding fails and the app
    // now (correctly) demotes it to an error chip, but this test wants the
    // happy-path chip to Send.
    visual.update(|_, cx| {
        cx.write_to_clipboard(gpui::ClipboardItem::new_image(&gpui::Image {
            format: gpui::ImageFormat::Png,
            bytes: valid_png_bytes(),
            id: 42,
        }))
    });
    visual.simulate_keystrokes("cmd-v");
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let chip = visual
        .debug_bounds("composer-chip")
        .expect("pasted image becomes a chip");
    assert!(chip.size.width > px(0.));
    view.read_with(&visual, |view, _| {
        assert_eq!(view.valid_attachment_names(), vec!["pasted-image.png"]);
    });
    // Click Send dispatches SendImages with the pending attachments.
    let send = visual
        .debug_bounds("send-button")
        .expect("send button renders");
    visual.simulate_click(send.center(), Default::default());
    let dispatched = receiver.try_recv().expect("SendImages dispatched");
    assert!(matches!(
        &dispatched,
        CommandMessage::SendImages(text, images)
            if text.is_empty() && images.len() == 1 && images[0].name == "pasted-image.png"
    ));
    // Worker ack clears the chip and records the attachment on the transcript row.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            let images = view.valid_attachments();
            view.apply_worker_message(WorkerMessage::ImagesSent(String::new(), images), window, cx);
        });
        window.draw(cx).clear(cx);
    });
    let recorded_size = valid_png_bytes().len();
    view.read_with(&visual, |view, _| {
        assert!(view.composer_attachments.is_empty());
        assert_eq!(
            view.state.session_view.attachments.get(&0),
            Some(&vec![("pasted-image.png".to_string(), recorded_size)])
        );
        // ImagesSent clears the queued-strip state alongside Sent — otherwise
        // an image-only send leaves a phantom dashed strip beside the solid
        // transcript turn.
        assert!(
            view.pending_user_turn.is_none(),
            "ImagesSent must clear the queued strip"
        );
    });
    assert!(visual.debug_bounds("composer-chip").is_none());
    assert!(visual.debug_bounds("composer-pending").is_none());
}

#[gpui::test]
fn per_file_parse_errors_render_their_own_chip_and_never_ship(cx: &mut TestAppContext) {
    // A bad clipboard payload lands as its own inline error chip — never in
    // the batch banner and never sendable — while good siblings and later
    // additions keep their own chips.
    let (window, view, receiver) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.add_pending_attachments(
                vec![
                    ImageAttachment::from_bytes("bad.png".into(), b"not a real png")
                        .map_err(|error| ("bad.png".to_string(), error)),
                ],
                cx,
            );
        });
        window.draw(cx).clear(cx);
    });
    view.read_with(&visual, |view, _| {
        assert!(
            view.composer_image_error.is_none(),
            "per-file errors must NOT set the batch banner: {:?}",
            view.composer_image_error,
        );
        assert_eq!(view.valid_attachment_count(), 0);
        assert_eq!(view.invalid_attachment_count(), 1);
    });
    assert!(
        visual.debug_bounds("composer-chip-error-0").is_some(),
        "bad file still paints an error chip so the user can see and remove it"
    );
    // Adding a valid sibling leaves the invalid chip in place. `png_bytes()`
    // above passes the header parser but fails the thumbnail decoder, so we
    // hand a real decodable PNG here — otherwise the sibling would also
    // demote to Invalid and the Send-filter assertion below has nothing to
    // ship.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.add_pending_attachments(
                vec![
                    ImageAttachment::from_bytes("good.png".into(), &valid_png_bytes())
                        .map_err(|error| ("good.png".to_string(), error)),
                ],
                cx,
            );
        });
        window.draw(cx).clear(cx);
    });
    view.read_with(&visual, |view, _| {
        assert_eq!(view.valid_attachment_count(), 1);
        assert_eq!(view.invalid_attachment_count(), 1);
    });
    // Send fires with the valid image only — the invalid one is filtered out.
    visual.update(|_, cx| {
        view.update(cx, |view, cx| view.send_composer(cx));
    });
    let dispatched = receiver.try_recv().expect("send fires with valid images");
    match dispatched {
        CommandMessage::SendImages(_, images) => {
            assert_eq!(images.len(), 1);
            assert_eq!(images[0].name, "good.png");
        }
        other => panic!("expected SendImages, got {other:?}"),
    }
}

#[gpui::test]
fn header_valid_but_undecodable_bytes_demote_to_an_error_chip(cx: &mut TestAppContext) {
    // `png_bytes()` is the PNG magic prefix with no IHDR — it passes the
    // header parser inside `ImageAttachment::from_bytes` so `attach_from_*`
    // hands the app an Ok(image), but the thumbnail decoder fails. The app
    // must catch that failure at the attach step and demote the item to an
    // Invalid chip — otherwise Send would ship an image the preview never
    // rendered, and the server would receive base64 the model cannot read.
    let (window, view, receiver) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            let broken = ImageAttachment::from_bytes("corrupt.png".into(), &png_bytes())
                .expect("header parses; the decoder should be the one to reject");
            view.add_pending_attachments(vec![Ok(broken)], cx);
        });
        window.draw(cx).clear(cx);
    });
    view.read_with(&visual, |view, _| {
        assert_eq!(
            view.valid_attachment_count(),
            0,
            "corrupt bytes must not sit as a Valid chip — Send would ship them"
        );
        assert_eq!(
            view.invalid_attachment_count(),
            1,
            "the corrupt payload paints its own error chip"
        );
    });
    assert!(
        visual.debug_bounds("composer-chip-error-0").is_some(),
        "error chip renders for the corrupt attachment"
    );
    // Send must NOT dispatch an image for the corrupt entry. With no other
    // text or valid images the composer surfaces its empty-hint instead.
    visual.update(|_, cx| {
        view.update(cx, |view, cx| view.send_composer(cx));
    });
    assert!(
        receiver.try_recv().is_err(),
        "corrupt-only Send must dispatch nothing to the worker channel"
    );
    view.read_with(&visual, |view, _| {
        assert!(
            view.composer_empty_hint,
            "corrupt-only Send should surface the empty-hint"
        );
    });
}

#[gpui::test]
fn text_send_with_lingering_invalid_chips_clears_them_on_worker_ack(cx: &mut TestAppContext) {
    // A user can type text into the composer AND have a leftover invalid chip
    // (a corrupt paste, a wrong-format drop). `send_composer` filters the
    // invalid entry out of the outgoing payload — so the worker path is
    // `Send(text)`, not `SendImages` — but until r4 the acknowledgment left
    // the invalid chip stranded beside the landed turn. `WorkerMessage::Sent`
    // must now clear the pending strip AND the chip row so the composer
    // resets to empty, matching the `ImagesSent` branch.
    let (window, view, receiver) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.add_pending_attachments(
                vec![Err((
                    "corrupt.png".into(),
                    "could not decode this image".into(),
                ))],
                cx,
            );
            view.composer
                .update(cx, |input, cx| input.set_value("hello", window, cx));
            view.send_composer(cx);
        });
        window.draw(cx).clear(cx);
    });
    let dispatched = receiver.try_recv().expect("text send fires");
    assert!(
        matches!(&dispatched, CommandMessage::Send(text) if text == "hello"),
        "invalid chip must be filtered out — expected Send, got {dispatched:?}",
    );
    view.read_with(&visual, |view, _| {
        assert_eq!(view.invalid_attachment_count(), 1);
        assert_eq!(view.valid_attachment_count(), 0);
    });
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.apply_worker_message(WorkerMessage::Sent("hello".into()), window, cx);
        });
        window.draw(cx).clear(cx);
    });
    view.read_with(&visual, |view, cx| {
        assert!(
            view.composer_attachments.is_empty(),
            "Sent must clear leftover invalid chips, not just the text",
        );
        assert!(
            view.composer.read(cx).value().is_empty(),
            "Sent still clears the composer text",
        );
        assert!(
            view.pending_user_turn.is_none(),
            "Sent still clears the queued dashed strip",
        );
    });
    assert!(
        visual.debug_bounds("composer-chip-error-0").is_none(),
        "the leftover error chip must be gone from the paint",
    );
}

#[gpui::test]
fn error_chip_advertises_alert_role_and_full_label(cx: &mut TestAppContext) {
    // Screen readers need the chip to announce as an alert AND carry the
    // filename + full error text, since the visible label truncates at the
    // chip's max width. A regression that dropped either would silently
    // ship an inaccessible chip.
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.add_pending_attachments(
                vec![Err((
                    "notes.bmp".into(),
                    "choose a PNG, JPEG, GIF, or WebP image".into(),
                ))],
                cx,
            );
        });
        window.draw(cx).clear(cx);
    });
    visual.update(|window, _cx| {
        use gpui_kit::test::TestWindowExt as _;
        let snapshot = window.find(("composer-chip-error", 0usize));
        assert_eq!(
            snapshot.role(),
            Some(gpui::Role::Alert),
            "error chip must advertise the Alert role for screen readers",
        );
        assert_eq!(
            snapshot.label(),
            Some("Attachment error: notes.bmp — choose a PNG, JPEG, GIF, or WebP image"),
            "chip label must pair the filename with the full error text",
        );
    });
}

#[gpui::test]
fn attachment_cap_counts_only_valid_chips(cx: &mut TestAppContext) {
    // The 4-image cap gates PAYLOADS THAT WILL SHIP. An invalid chip is
    // visible but never ships, so a mixed batch — three valid + two invalid
    // files — must attach all five entries. A regression that counted the
    // vec length would reject the whole batch and force the user to retry.
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            let items: Vec<Result<ImageAttachment, (String, String)>> = vec![
                Ok(thumbnail_attachment(8, 6, 1)),
                Ok(thumbnail_attachment(8, 6, 2)),
                Ok(thumbnail_attachment(8, 6, 3)),
                Err(("bad-a.png".into(), "junk bytes".into())),
                Err(("bad-b.png".into(), "junk bytes".into())),
            ];
            let appended = view.add_pending_attachments(items, cx);
            assert!(appended, "cap must not fire on 3 valid + 2 invalid");
        });
        window.draw(cx).clear(cx);
    });
    view.read_with(&visual, |view, _| {
        assert_eq!(view.valid_attachment_count(), 3);
        assert_eq!(view.invalid_attachment_count(), 2);
        assert!(
            view.composer_image_error.is_none(),
            "batch banner must not fire when the valid count fits: {:?}",
            view.composer_image_error,
        );
    });
    // A follow-up attach of one more valid image is fine — 3 + 1 = 4 valid.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            let appended =
                view.add_pending_attachments(vec![Ok(thumbnail_attachment(8, 6, 4))], cx);
            assert!(appended, "fourth valid attachment must slot under the cap");
        });
        window.draw(cx).clear(cx);
    });
    view.read_with(&visual, |view, _| {
        assert_eq!(view.valid_attachment_count(), 4);
        assert!(view.composer_image_error.is_none());
    });
    // The fifth valid attachment DOES trip the cap.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            let appended =
                view.add_pending_attachments(vec![Ok(thumbnail_attachment(8, 6, 5))], cx);
            assert!(
                !appended,
                "fifth valid attachment must fail the cap and leave chips intact"
            );
        });
        window.draw(cx).clear(cx);
    });
    view.read_with(&visual, |view, _| {
        assert_eq!(view.valid_attachment_count(), 4);
        assert!(
            view.composer_image_error
                .as_ref()
                .is_some_and(|e| e.contains('4')),
            "batch banner must fire on the fifth valid image, got {:?}",
            view.composer_image_error,
        );
    });
}

#[gpui::test]
fn attachment_parse_errors_cover_all_four_kinds(cx: &mut TestAppContext) {
    // Each of the four parse-error categories the attach path can raise —
    // unreadable path, oversize file, undecodable bytes, unsupported format
    // — lands as its own per-file chip, never rejects a batch, and each
    // survives Send filtering.
    let temp_dir = std::env::temp_dir();
    let unique = std::process::id();
    let oversize_path = temp_dir.join(format!("zeta-parse-oversize-{unique}.png"));
    let unsupported_path = temp_dir.join(format!("zeta-parse-unsupported-{unique}.bmp"));
    let undecodable_path = temp_dir.join(format!("zeta-parse-undecodable-{unique}.png"));
    let missing_path = temp_dir.join(format!("zeta-parse-missing-{unique}.png"));
    // Oversize: a PNG magic-header prefix padded past the 512 KiB cap.
    let mut oversize_bytes = png_bytes();
    oversize_bytes.resize(session::MAX_IMAGE_BYTES + 1, 0u8);
    std::fs::write(&oversize_path, &oversize_bytes).unwrap();
    // Unsupported: a valid BMP header the parser rejects.
    std::fs::write(&unsupported_path, b"BM\x00\x00\x00\x00\x00\x00").unwrap();
    // Undecodable: a `.png`-named file with junk bytes.
    std::fs::write(&undecodable_path, b"junk bytes").unwrap();
    // Missing: never written, so the read fails first.
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.attach_from_paths(
                &[
                    missing_path.clone(),
                    oversize_path.clone(),
                    undecodable_path.clone(),
                    unsupported_path.clone(),
                ],
                cx,
            );
        });
        window.draw(cx).clear(cx);
    });
    view.read_with(&visual, |view, _| {
        assert_eq!(
            view.invalid_attachment_count(),
            4,
            "every parse error becomes its own chip"
        );
        assert_eq!(view.valid_attachment_count(), 0);
        assert!(view.composer_image_error.is_none());
    });
    for selector in [
        "composer-chip-error-0",
        "composer-chip-error-1",
        "composer-chip-error-2",
        "composer-chip-error-3",
    ] {
        assert!(
            visual.debug_bounds(selector).is_some(),
            "error chip {selector} paints"
        );
    }
    for path in [oversize_path, unsupported_path, undecodable_path] {
        let _ = std::fs::remove_file(path);
    }
}

#[gpui::test]
fn attach_button_paints_an_icon_hit_target_at_send_button_height(cx: &mut TestAppContext) {
    // The icon-only attach affordance keeps the Send row compact but must
    // still meet the 40px hit-area floor from make-interfaces-feel-better so
    // pointer users, tab focus, and touch targets all land on the same box.
    let (window, _, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let attach = visual
        .debug_bounds("attach-button")
        .expect("attach button renders");
    assert!(
        attach.size.height >= px(40.) && attach.size.width >= px(40.),
        "attach hit area must be at least 40x40 (got {:?})",
        attach.size,
    );
    let send = visual
        .debug_bounds("send-button")
        .expect("send button renders");
    // Both controls share the composer row's vertical rhythm.
    assert_eq!(
        attach.size.height, send.size.height,
        "attach and send buttons must share the composer action-row height",
    );
}

#[gpui::test]
fn drag_over_composer_paints_drop_target_and_drop_adds_attachments(cx: &mut TestAppContext) {
    // The full drag-and-drop path: entering the composer bounds with an
    // ExternalPaths payload lights up the overlay; submitting the drop
    // dispatches the same batch flow as the file picker, so a real PNG on
    // disk lands as a pending attachment. Exiting without a drop must clear
    // the overlay without touching the chip row.
    let temp = std::env::temp_dir().join(format!("zeta-drop-{}.png", std::process::id()));
    std::fs::write(&temp, valid_png_bytes()).expect("write drop-source png");
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let composer = visual.debug_bounds("composer").expect("composer renders");
    let inside = composer.center();
    // Enter → overlay paints on the next draw; Exit → overlay clears; no
    // chip yet. `on_drag_move` updates the hover flag BEFORE the render
    // that reads it, so one draw per drag event suffices.
    visual.update(|window, cx| {
        window.dispatch_event(
            gpui::FileDropEvent::Entered {
                position: inside,
                paths: gpui::ExternalPaths([temp.clone()].into_iter().collect()),
            }
            .to_platform_input(),
            cx,
        );
        window.draw(cx).clear(cx);
    });
    assert!(
        visual.debug_bounds("composer-drop-target").is_some(),
        "drop-target overlay must paint while an external drag is active",
    );
    view.read_with(&visual, |view, _| {
        assert!(
            view.composer_attachments.is_empty(),
            "hover alone must not attach"
        );
    });
    visual.update(|window, cx| {
        window.dispatch_event(gpui::FileDropEvent::Exited.to_platform_input(), cx);
        window.draw(cx).clear(cx);
    });
    assert!(
        visual.debug_bounds("composer-drop-target").is_none(),
        "drop-target overlay must clear when the drag leaves the window",
    );
    // Re-enter and submit → the drop hits attach_from_paths, which decodes
    // the file into an ImageAttachment and renders the pending chip.
    visual.update(|window, cx| {
        window.dispatch_event(
            gpui::FileDropEvent::Entered {
                position: inside,
                paths: gpui::ExternalPaths([temp.clone()].into_iter().collect()),
            }
            .to_platform_input(),
            cx,
        );
        window.dispatch_event(
            gpui::FileDropEvent::Submit { position: inside }.to_platform_input(),
            cx,
        );
        window.draw(cx).clear(cx);
    });
    view.read_with(&visual, |view, _| {
        assert_eq!(view.valid_attachment_count(), 1);
        assert!(view.valid_attachment_names()[0].ends_with(".png"));
    });
    assert!(visual.debug_bounds("composer-chip").is_some());
    let _ = std::fs::remove_file(&temp);
}

#[gpui::test]
fn drop_overlay_hides_when_the_drag_leaves_the_composer_bounds(cx: &mut TestAppContext) {
    // The overlay must scope to the composer's hitbox — a drag that starts
    // over the composer and moves onto the sidebar clears the overlay even
    // though `has_active_drag()` stays true window-wide. `on_drag_move`
    // updates hover state BEFORE the render that reads it, so ONE draw per
    // drag event paints the correct state — a mutation that reverts the
    // hover path to a paint-time side effect would need two draws to catch
    // up and fail this test.
    let temp = std::env::temp_dir().join(format!("zeta-drop-scope-{}.png", std::process::id()));
    std::fs::write(&temp, valid_png_bytes()).expect("write drop-source png");
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let composer = visual.debug_bounds("composer").expect("composer renders");
    let sidebar = visual
        .debug_bounds("sidebar-header")
        .expect("sidebar renders");
    // Enter the composer → overlay paints after a SINGLE draw.
    visual.update(|window, cx| {
        window.dispatch_event(
            gpui::FileDropEvent::Entered {
                position: composer.center(),
                paths: gpui::ExternalPaths([temp.clone()].into_iter().collect()),
            }
            .to_platform_input(),
            cx,
        );
        window.draw(cx).clear(cx);
    });
    assert!(
        visual.debug_bounds("composer-drop-target").is_some(),
        "overlay must paint after ONE draw once the drag enters the composer",
    );
    view.read_with(&visual, |view, _| {
        assert!(view.drag_over_composer.get());
    });
    // Move the drag pointer onto the sidebar → overlay clears after ONE
    // draw, though the window-wide drag is still active.
    visual.update(|window, cx| {
        window.dispatch_event(
            gpui::FileDropEvent::Pending {
                position: sidebar.center(),
            }
            .to_platform_input(),
            cx,
        );
        window.draw(cx).clear(cx);
    });
    view.read_with(&visual, |view, _| {
        assert!(
            !view.drag_over_composer.get(),
            "drag pointer left the composer bounds — overlay must clear",
        );
    });
    assert!(
        visual.debug_bounds("composer-drop-target").is_none(),
        "overlay must not paint while the drag hovers a peer element",
    );
    // Re-enter the composer → overlay lights back up on the very next draw.
    // Guards against a regression where the flag stays stuck false after a
    // sidebar excursion.
    visual.update(|window, cx| {
        window.dispatch_event(
            gpui::FileDropEvent::Pending {
                position: composer.center(),
            }
            .to_platform_input(),
            cx,
        );
        window.draw(cx).clear(cx);
    });
    assert!(
        visual.debug_bounds("composer-drop-target").is_some(),
        "re-entering the composer bounds must relight the overlay after ONE draw",
    );
    let _ = std::fs::remove_file(&temp);
}

#[gpui::test]
fn attach_button_click_invokes_the_file_picker(cx: &mut TestAppContext) {
    // Clicking the attach button routes through `attach_from_files`, which
    // opens the platform path prompt. The test harness records the prompt
    // so we can observe the click actually reached the picker.
    let (window, _view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| window.draw(cx).clear(cx));
    assert!(!visual.did_prompt_for_paths());
    let attach = visual
        .debug_bounds("attach-button")
        .expect("attach button renders");
    visual.simulate_click(attach.center(), Default::default());
    visual.run_until_parked();
    assert!(
        visual.did_prompt_for_paths(),
        "the attach button must open the platform path prompt"
    );
    // Cancel the prompt so it does not linger for later tests.
    visual.simulate_path_prompt_response(|_options| None);
}

#[gpui::test]
fn attachment_chip_renders_remove_button_and_multi_attachments(cx: &mut TestAppContext) {
    // Multiple pending attachments each render their own chip with a
    // remove-button hit target; clicking a chip's remove drops just that
    // attachment and keeps the others intact.
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            let items = vec![
                Ok(thumbnail_attachment(8, 6, 1)),
                Ok(thumbnail_attachment(8, 6, 2)),
                Ok(thumbnail_attachment(8, 6, 3)),
            ];
            view.add_pending_attachments(items, cx);
        });
        window.draw(cx).clear(cx);
    });
    view.read_with(&visual, |view, _| {
        assert_eq!(view.valid_attachment_count(), 3);
    });
    let remove_middle = visual
        .debug_bounds("chip-remove-1")
        .expect("middle chip remove button renders");
    assert!(
        remove_middle.size.height >= px(20.) && remove_middle.size.width >= px(20.),
        "remove target must be at least visible-sized (got {:?})",
        remove_middle.size,
    );
    visual.simulate_click(remove_middle.center(), Default::default());
    visual.update(|window, cx| window.draw(cx).clear(cx));
    view.read_with(&visual, |view, _| {
        assert_eq!(view.valid_attachment_count(), 2);
    });
    // The chip row still paints for the two survivors.
    assert!(visual.debug_bounds("composer-chip").is_some());
    assert!(visual.debug_bounds("chip-remove-0").is_some());
    assert!(visual.debug_bounds("chip-remove-1").is_some());
    assert!(visual.debug_bounds("chip-remove-2").is_none());
}

#[gpui::test]
fn removing_an_invalid_attachment_chip_drops_only_that_entry(cx: &mut TestAppContext) {
    // Invalid chips are removable via their own remove button; the survivor
    // list keeps its valid siblings intact.
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            let items = vec![
                Ok(thumbnail_attachment(8, 6, 1)),
                Err(("junk.bin".to_string(), "unsupported format".to_string())),
                Ok(thumbnail_attachment(8, 6, 2)),
            ];
            view.add_pending_attachments(items, cx);
        });
        window.draw(cx).clear(cx);
    });
    view.read_with(&visual, |view, _| {
        assert_eq!(view.valid_attachment_count(), 2);
        assert_eq!(view.invalid_attachment_count(), 1);
    });
    assert!(visual.debug_bounds("composer-chip-error-1").is_some());
    let remove = visual
        .debug_bounds("chip-remove-1")
        .expect("error-chip remove button");
    visual.simulate_click(remove.center(), Default::default());
    visual.update(|window, cx| window.draw(cx).clear(cx));
    view.read_with(&visual, |view, _| {
        assert_eq!(view.valid_attachment_count(), 2);
        assert_eq!(view.invalid_attachment_count(), 0);
    });
    assert!(visual.debug_bounds("composer-chip-error-1").is_none());
}

#[gpui::test]
fn drop_reuses_the_batch_limit_error_and_leaves_chips_intact(cx: &mut TestAppContext) {
    // Drop routes through `add_pending_attachments` so the shared 4-image
    // cap fires the same batch-limit banner the paste and file-picker paths
    // surface. A dropped fifth file must land in the error surface, not
    // silently, and the four existing chips survive.
    let temp = std::env::temp_dir().join(format!("zeta-drop-limit-{}.png", std::process::id()));
    std::fs::write(&temp, valid_png_bytes()).expect("write drop-source png");
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            for _ in 0..4 {
                view.add_pending_attachments(vec![Ok(thumbnail_attachment(8, 6, 1))], cx);
            }
            assert_eq!(view.valid_attachment_count(), 4);
        });
        window.draw(cx).clear(cx);
    });
    let composer = visual.debug_bounds("composer").unwrap();
    visual.update(|window, cx| {
        window.dispatch_event(
            gpui::FileDropEvent::Entered {
                position: composer.center(),
                paths: gpui::ExternalPaths([temp.clone()].into_iter().collect()),
            }
            .to_platform_input(),
            cx,
        );
        window.dispatch_event(
            gpui::FileDropEvent::Submit {
                position: composer.center(),
            }
            .to_platform_input(),
            cx,
        );
        window.draw(cx).clear(cx);
    });
    view.read_with(&visual, |view, _| {
        assert_eq!(view.valid_attachment_count(), 4, "cap must not be exceeded");
        assert!(
            view.composer_image_error
                .as_ref()
                .is_some_and(|e| e.contains("512")),
            "the batch limit banner must fire on the offending drop, got {:?}",
            view.composer_image_error,
        );
    });
    let _ = std::fs::remove_file(&temp);
}

#[gpui::test]
fn removing_and_clearing_pending_chips_evict_thumbnail_assets(cx: &mut TestAppContext) {
    // Repeated attach → remove / attach → clear cycles must free the GPU
    // asset cache slot, or long compose sessions leak textures. `remove` and
    // `clear` both route through `PendingAttachment::Valid`'s thumbnail
    // Arc so `remove_asset` fires per handle.
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    // First pass: add + render (populates the asset cache) → remove →
    // asset must be evicted.
    let cached_after_remove = visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.add_pending_attachments(vec![Ok(thumbnail_attachment(16, 12, 1))], cx);
        });
        window.draw(cx).clear(cx);
        view.update(cx, |view, cx| {
            let handle = match &view.composer_attachments[0] {
                PendingAttachment::Valid { thumbnail, .. } => thumbnail.clone(),
                PendingAttachment::Invalid { .. } => panic!("expected valid attachment"),
            };
            handle.clone().get_render_image(window, cx);
            assert!(handle.is_asset_cached(cx), "asset must cache after render");
            view.remove_attached_image(0, cx);
            handle.is_asset_cached(cx)
        })
    });
    assert!(
        !cached_after_remove,
        "the removed thumbnail must be evicted from the asset cache"
    );
    // Second pass: add + render → clear → asset must be evicted.
    let cached_after_clear = visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.add_pending_attachments(vec![Ok(thumbnail_attachment(16, 12, 2))], cx);
        });
        window.draw(cx).clear(cx);
        view.update(cx, |view, cx| {
            let handle = match &view.composer_attachments[0] {
                PendingAttachment::Valid { thumbnail, .. } => thumbnail.clone(),
                PendingAttachment::Invalid { .. } => panic!("expected valid attachment"),
            };
            handle.clone().get_render_image(window, cx);
            assert!(handle.is_asset_cached(cx), "asset must cache after render");
            view.clear_composer_images(cx);
            handle.is_asset_cached(cx)
        })
    });
    assert!(
        !cached_after_clear,
        "clear must evict every held thumbnail from the asset cache"
    );
}

#[gpui::test]
fn attachment_chip_scales_with_the_appearance_font_size(cx: &mut TestAppContext) {
    // Chip dimensions route through theme tokens keyed on font size, so a
    // chip painted at the picker's 11px floor is visibly smaller than the
    // same chip at the 18px ceiling. Measures RENDERED sub-parts — the
    // whole chip's outer bounds could stay the same if only the thumbnail
    // shrank while padding grew, so we probe the thumbnail img and the
    // remove-button bounds directly. A regression that hardcoded either
    // inner size to a pixel literal is caught here.
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.add_pending_attachments(vec![Ok(thumbnail_attachment(16, 12, 1))], cx);
        });
        window.draw(cx).clear(cx);
    });
    struct ChipParts {
        chip: gpui::Bounds<gpui::Pixels>,
        thumbnail: gpui::Bounds<gpui::Pixels>,
        remove: gpui::Bounds<gpui::Pixels>,
    }
    let mut appearance = theme::Appearance::default();
    let mut chip_at = |base: f32, visual: &mut VisualTestContext| {
        appearance.font_size = theme::clamp_font_size(base);
        visual.update(|window, cx| {
            theme::apply_with(cx, &appearance);
            view.update(cx, |_, cx| cx.notify());
            window.draw(cx).clear(cx);
        });
        ChipParts {
            chip: visual.debug_bounds("composer-chip").expect("chip renders"),
            thumbnail: visual
                .debug_bounds("composer-chip-thumbnail-0")
                .expect("chip thumbnail renders"),
            remove: visual
                .debug_bounds("chip-remove-0")
                .expect("chip remove button renders"),
        }
    };
    let small = chip_at(theme::MIN_FONT_SIZE_PX, &mut visual);
    let large = chip_at(theme::MAX_FONT_SIZE_PX, &mut visual);
    assert!(
        large.chip.size.height > small.chip.size.height
            && large.chip.size.width > small.chip.size.width,
        "outer chip bounds must scale (11px→{:?}, 18px→{:?})",
        small.chip.size,
        large.chip.size,
    );
    assert!(
        large.thumbnail.size.height > small.thumbnail.size.height
            && large.thumbnail.size.width > small.thumbnail.size.width,
        "chip thumbnail must scale — a mutation that hardcoded the inner \
         image dimensions would leave these equal (11px→{:?}, 18px→{:?})",
        small.thumbnail.size,
        large.thumbnail.size,
    );
    assert!(
        large.remove.size.height > small.remove.size.height
            && large.remove.size.width > small.remove.size.width,
        "chip remove button must scale — a mutation that hardcoded the hit \
         target would leave these equal (11px→{:?}, 18px→{:?})",
        small.remove.size,
        large.remove.size,
    );
    // Reset back to the default so downstream tests see the baseline theme.
    visual.update(|_, cx| theme::apply(cx));
}

#[gpui::test]
fn attachment_chips_render_legibly_across_every_shipped_theme(cx: &mut TestAppContext) {
    // Every appearance-picker theme must keep the chip surface + chip text
    // visually distinct. Asserts the RENDERED colors — the chip's fill quad
    // comes from `painted_quads` and the chip name's text color from the
    // `record_state` sample logged at draw time — so a mutation that painted
    // the label with the fill color, or the fill quad with the text color,
    // would slip past a palette-field check and fail here.
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.add_pending_attachments(vec![Ok(thumbnail_attachment(16, 12, 1))], cx);
        });
        window.draw(cx).clear(cx);
    });
    for id in theme::ThemeId::ALL {
        let appearance = theme::Appearance {
            theme: *id,
            ..theme::Appearance::default()
        };
        super::render_log::clear();
        let (expected_fill, expected_text) = visual.update(|window, cx| {
            theme::apply_with(cx, &appearance);
            view.update(cx, |_, cx| cx.notify());
            window.draw(cx).clear(cx);
            let theme = cx.theme();
            (theme.muted, theme.foreground)
        });
        let chip = visual
            .debug_bounds("composer-chip")
            .unwrap_or_else(|| panic!("chip must paint under {:?}", id.slug()));
        assert!(
            chip.size.width > px(0.) && chip.size.height > px(0.),
            "chip must have positive bounds under {:?}",
            id.slug(),
        );
        // Rendered fill: a painted quad whose background matches the
        // theme.muted token AND whose bounds sit inside the chip.
        visual.update(|window, _| {
            let scale = window.scale_factor();
            let scaled_chip = chip.scale(scale);
            let fill = window.painted_quads().into_iter().find(|quad| {
                quad.background == expected_fill.into()
                    && quad.bounds.top() >= scaled_chip.top() - px(1.).scale(scale)
                    && quad.bounds.bottom() <= scaled_chip.bottom() + px(1.).scale(scale)
                    && quad.bounds.left() >= scaled_chip.left() - px(1.).scale(scale)
                    && quad.bounds.right() <= scaled_chip.right() + px(1.).scale(scale)
            });
            assert!(
                fill.is_some(),
                "chip fill must paint theme.muted ({:?}) under {:?}",
                expected_fill,
                id.slug(),
            );
        });
        // Rendered text color: the render_log sample for chip-name-0 is the
        // color that reached the `.text_color(...)` call at draw time.
        let sample = super::render_log::samples()
            .into_iter()
            .find(|s| s.row_id == "chip-name-0")
            .unwrap_or_else(|| panic!("chip name text must record under {:?}", id.slug()));
        assert_eq!(
            sample.color,
            expected_text,
            "chip name text must paint theme.foreground under {:?}",
            id.slug(),
        );
        assert_ne!(
            sample.color,
            expected_fill,
            "chip name text must not paint the fill color under {:?} — \
             filename would render invisible against the chip",
            id.slug(),
        );
    }
    // Reset back to the default so downstream tests see the baseline theme.
    visual.update(|_, cx| theme::apply(cx));
}

#[gpui::test]
fn tool_receipts_expand_collapse_and_show_failures(cx: &mut TestAppContext) {
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    for failed in [false, true] {
        visual.update(|window, cx| {
            view.update(cx, |view, cx| {
                view.state.transcript.clear();
                view.transcript.update(cx, |scroll, cx| scroll.reset(0, cx));
                view.apply_worker_message(
                    WorkerMessage::Event(ServerEvent::ToolEnd {
                        session_id: view.state.active_session.clone(),
                        tool_call: ToolCall {
                            id: "receipt".into(),
                            name: "bash".into(),
                            arguments: Default::default(),
                        },
                        tool_result: Some(zeta_gui::client::ToolResult {
                            tool_call_id: "receipt".into(),
                            content: "first line\nsecond line\nlast line".into(),
                            is_error: failed,
                            is_canceled: false,
                            structured_content: None,
                            content_blocks: Vec::new(),
                        }),
                        data: json!({}),
                    }),
                    window,
                    cx,
                );
            });
        });
        visual.update(|window, cx| window.draw(cx).clear(cx));
        assert_eq!(visual.debug_bounds("tool-output-0").is_some(), failed);
        for expanded in [!failed, failed] {
            let bounds = visual.debug_bounds("tool-receipt-0").unwrap();
            // Click the header, which remains at the top when output expands.
            visual.simulate_click(
                bounds.origin + gpui::point(px(50.), px(20.)),
                Default::default(),
            );
            visual.update(|window, cx| window.draw(cx).clear(cx));
            assert_eq!(visual.debug_bounds("tool-output-0").is_some(), expanded);
            if expanded {
                assert!(visual.debug_bounds("tool-output-0").unwrap().size.height > px(40.));
            }
            view.read_with(&visual, |view, _| {
                assert!(
                    matches!(&view.state.transcript[0], TranscriptEntry::Tool {card, ..}
                    if card.expanded == expanded && card.tail.text.ends_with("last line"))
                );
            });
        }
    }
}

#[gpui::test]
fn error_block_keeps_full_text_wraps_and_opens_settings(cx: &mut TestAppContext) {
    for detail in ["provider detail ".repeat(80), "x".repeat(1200)] {
        let (window, view, receiver) = setup(cx);
        let mut visual = VisualTestContext::from_window(window.into(), cx);
        let message = format!("Model not supported: {}\nlast error line", detail);
        visual.update(|window, cx| {
            view.update(cx, |view, cx| {
                view.state.session_view.available = true;
                view.apply_worker_message(
                    WorkerMessage::Event(ServerEvent::Error {
                        session_id: view.state.active_session.clone(),
                        error: zeta_gui::client::EventError {
                            code: "model_access_error".into(),
                            message: message.clone(),
                        },
                        data: json!({}),
                    }),
                    window,
                    cx,
                );
            });
        });
        visual.update(|window, cx| window.draw(cx).clear(cx));
        view.read_with(&visual, |view, _| {
        assert!(matches!(&view.state.transcript[0], TranscriptEntry::Error { message: text, settings_action: true, .. } if *text == message));
    });
        let text_bounds = visual
            .debug_bounds("error-message-0")
            .expect("full error text block");
        assert!(
            text_bounds.size.height > px(60.),
            "long error must wrap across several lines"
        );
        let button = visual.debug_bounds("error-settings-0").unwrap();
        visual.simulate_click(button.center(), Default::default());
        assert!(matches!(
            receiver.try_recv(),
            Ok(CommandMessage::LoadSettings)
        ));
        visual.update(|window, cx| {
            view.update(cx, |view, cx| {
                view.apply_worker_message(
                    WorkerMessage::Settings(
                        SessionSettings {
                            model: "working".into(),
                            approval_mode: "ask".into(),
                        },
                        ModelCatalog {
                            models: vec!["working".into()],
                            providers: Default::default(),
                        },
                    ),
                    window,
                    cx,
                )
            });
            window.draw(cx).clear(cx);
        });
        assert!(visual.debug_bounds("settings-overlay").is_some());
    }
}

#[gpui::test]
fn errors_without_settings_recovery_have_no_action(cx: &mut TestAppContext) {
    for (available, code, message) in [
        (false, "model_access_error", "Login required"),
        (true, "auth_error", "MCP OAuth credentials expired"),
    ] {
        let (window, view, _) = setup(cx);
        let mut visual = VisualTestContext::from_window(window.into(), cx);
        visual.update(|window, cx| {
            view.update(cx, |view, cx| {
                view.state.session_view.available = available;
                view.apply_worker_message(
                    WorkerMessage::Event(ServerEvent::Error {
                        session_id: view.state.active_session.clone(),
                        error: zeta_gui::client::EventError {
                            code: code.into(),
                            message: message.into(),
                        },
                        data: json!({}),
                    }),
                    window,
                    cx,
                );
            });
            window.draw(cx).clear(cx);
        });
        assert!(visual.debug_bounds("error-block-0").is_some());
        assert!(visual.debug_bounds("error-settings-0").is_none());
    }
}

#[gpui::test]
fn login_controls_share_progress_cancel_and_retry_across_surfaces(cx: &mut TestAppContext) {
    for surface in ["first", "settings", "error"] {
        let (window, view, receiver) = setup(cx);
        let mut visual = VisualTestContext::from_window(window.into(), cx);
        visual.update(|window, cx| {
            view.update(cx, |view, cx| {
                view.apply_worker_message(
                    WorkerMessage::LoginProviders(vec![LoginProvider {
                        provider: "claude".into(),
                        credentials_present: false,
                        progress: LoginProgress::Idle,
                    }]),
                    window,
                    cx,
                );
                match surface {
                    "settings" => view.settings_open = true,
                    "error" => {
                        view.apply_worker_message(
                            WorkerMessage::Event(ServerEvent::Error {
                                session_id: view.state.active_session.clone(),
                                error: zeta_gui::client::EventError {
                                    code: "model_access_error".into(),
                                    message: "Sign in required".into(),
                                },
                                data: json!({"login_provider":"claude"}),
                            }),
                            window,
                            cx,
                        );
                    }
                    _ => view.state.active_session = None,
                }
                cx.notify();
            });
            window.draw(cx).clear(cx);
        });
        let selector = match surface {
            "first" => "first-login-claude-start",
            "settings" => "settings-login-claude-start",
            _ => "error-login-0-claude-start",
        };
        let button = visual.debug_bounds(selector).unwrap();
        assert!(button.size.height >= px(40.));
        visual.simulate_click(button.center(), Default::default());
        assert!(
            matches!(receiver.try_recv(), Ok(CommandMessage::LoginStart(provider)) if provider == "claude")
        );
        visual.update(|window, cx| {
            view.update(cx, |view, cx| {
                assert_eq!(view.login_providers[0].progress, LoginProgress::Starting);
                view.apply_worker_message(
                    WorkerMessage::Login(
                        "claude".into(),
                        LoginProgress::Pending {
                            authorization_url: None,
                        },
                    ),
                    window,
                    cx,
                );
                cx.notify();
            });
            window.draw(cx).clear(cx);
        });
        // First-run progress remains visible after its initial button disappears.
        let selector = match surface {
            "first" => "login-progress-claude-cancel",
            "settings" => "settings-login-claude-cancel",
            _ => "error-login-0-claude-cancel",
        };
        let cancel = visual.debug_bounds(selector).unwrap();
        visual.simulate_click(cancel.center(), Default::default());
        assert!(
            matches!(receiver.try_recv(), Ok(CommandMessage::LoginCancel(provider)) if provider == "claude")
        );
        visual.update(|window, cx| {
            view.update(cx, |view, cx| {
                view.apply_worker_message(
                    WorkerMessage::Login("claude".into(), LoginProgress::Cancelled),
                    window,
                    cx,
                );
                view.start_login("claude", cx);
                view.apply_worker_message(
                    WorkerMessage::Login(
                        "claude".into(),
                        LoginProgress::failed("Browser sign-in timed out".into()),
                    ),
                    window,
                    cx,
                );
                assert!(!view.login_providers[0].progress.busy());
                view.start_login("claude", cx);
                view.apply_worker_message(
                    WorkerMessage::Login("claude".into(), LoginProgress::Succeeded),
                    window,
                    cx,
                );
                assert!(view.login_providers[0].credentials_present);
                assert!(view.can_change_session());
            });
        });
    }
}

#[gpui::test]
fn legacy_login_controls_stay_hidden_and_disconnect_clears_pending(cx: &mut TestAppContext) {
    let (window, view, receiver) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.start_login("claude", cx);
            assert!(receiver.try_recv().is_err());
            view.login_providers.push(LoginProvider {
                provider: "claude".into(),
                credentials_present: false,
                progress: LoginProgress::Starting,
            });
            view.apply_worker_message(WorkerMessage::Lost("offline".into()), window, cx);
            assert!(!view.login_providers[0].progress.busy());
            view.apply_worker_message(WorkerMessage::LoginProviders(Vec::new()), window, cx);
            cx.notify();
        });
        window.draw(cx).clear(cx);
    });
    assert!(visual.debug_bounds("first-login-claude-start").is_none());
    assert!(visual.debug_bounds("login-progress-claude-start").is_none());
}

#[gpui::test]
fn settings_trap_typing_editing_paste_and_shortcuts(cx: &mut TestAppContext) {
    let (window, view, receiver) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.simulate_keystrokes("d r a f t");
    visual.update(|window, cx| {
        cx.write_to_clipboard(gpui::ClipboardItem::new_string("pasted".into()));
        view.update(cx, |view, cx| {
            view.apply_worker_message(
                WorkerMessage::Settings(
                    SessionSettings {
                        model: "one".into(),
                        approval_mode: "ask".into(),
                    },
                    ModelCatalog {
                        models: vec!["one".into(), "two".into()],
                        providers: Default::default(),
                    },
                ),
                window,
                cx,
            );
        });
        window.draw(cx).clear(cx);
    });
    for selector in ["mode-row-ask", "mode-row-allow", "mode-row-deny"] {
        assert!(visual.debug_bounds(selector).is_some());
    }
    visual.simulate_keystrokes("x backspace cmd-a cmd-v shift-enter tab cmd-n down");
    view.read_with(&visual, |view, cx| {
        assert_eq!(view.composer.read(cx).value().as_ref(), "draft");
        assert!(view.composer_attachments.is_empty());
        assert_eq!(view.state.session_view.selected_model, 1);
    });
    assert!(receiver.try_recv().is_err());
    visual.simulate_keystrokes("escape x");
    view.read_with(&visual, |view, cx| {
        assert!(!view.settings_open);
        assert_eq!(view.composer.read(cx).value().as_ref(), "draftx");
    });
}

#[gpui::test]
fn session_changes_clear_ui_state_but_same_session_status_preserves_it(cx: &mut TestAppContext) {
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    for via_status in [false, true] {
        visual.update(|window, cx| {
            view.update(cx, |view, cx| {
                let current = view.state.active_session.clone().unwrap();
                let metadata = SessionMetadata {
                    session_id: current,
                    ..session()
                };
                view.add_pending_attachments(
                    vec![Ok(ImageAttachment::from_bytes(
                        "test.png".into(),
                        &valid_png_bytes(),
                    )
                    .unwrap())],
                    cx,
                );
                view.composer_image_error = Some("old image error".into());
                view.settings_open = true;
                view.settings_error = Some("old settings error".into());
                let status = StatusResult {
                    session: Some(metadata.clone()),
                    state: "idle".into(),
                    pending_approvals: vec![],
                    usage: json!({}),
                    compaction_markers: 0,
                };
                view.apply_worker_message(WorkerMessage::Status(status.clone()), window, cx);
                assert_eq!(view.valid_attachment_count(), 1);
                assert!(view.settings_open);
                let next = SessionMetadata {
                    session_id: format!("next-{via_status}"),
                    ..metadata
                };
                let change = if via_status {
                    WorkerMessage::Status(StatusResult {
                        session: Some(next),
                        ..status
                    })
                } else {
                    WorkerMessage::Session(next)
                };
                view.apply_worker_message(change, window, cx);
                assert!(view.composer_attachments.is_empty());
                assert!(view.composer_image_error.is_none());
                assert!(!view.settings_open);
                assert!(view.settings_error.is_none());
            });
        });
    }
}

#[gpui::test]
fn image_paste_is_claimed_only_when_an_image_is_accepted(cx: &mut TestAppContext) {
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|_, cx| {
        cx.write_to_clipboard(gpui::ClipboardItem::new_image(&gpui::Image::from_bytes(
            gpui::ImageFormat::Png,
            valid_png_bytes(),
        )));
        view.update(cx, |view, cx| {
            view.state.active_session = None;
            assert!(!view.attach_from_clipboard(cx));
            view.state.active_session = Some(session().session_id);
            view.state.streaming = true;
            assert!(!view.attach_from_clipboard(cx));
            view.state.streaming = false;
            view.pending_command = true;
            assert!(!view.attach_from_clipboard(cx));
            view.pending_command = false;
            for _ in 0..4 {
                assert!(view.attach_from_clipboard(cx));
            }
            assert!(!view.attach_from_clipboard(cx));
            assert_eq!(view.valid_attachment_count(), 4);
            assert!(view.composer_image_error.is_some());
            view.clear_composer_images(cx);
        });
        cx.write_to_clipboard(gpui::ClipboardItem::new_image(&gpui::Image::from_bytes(
            gpui::ImageFormat::Png,
            b"invalid".to_vec(),
        )));
        view.update(cx, |view, cx| assert!(!view.attach_from_clipboard(cx)));
        cx.write_to_clipboard(gpui::ClipboardItem::new_string("normal paste".into()));
        view.update(cx, |view, cx| assert!(!view.attach_from_clipboard(cx)));
    });
    visual.simulate_keystrokes("cmd-v");
    view.read_with(&visual, |view, cx| {
        assert_eq!(view.composer.read(cx).value().as_ref(), "normal paste")
    });
}

#[gpui::test]
fn thinking_feedback_stops_on_text_and_turn_boundaries(cx: &mut TestAppContext) {
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            let start = ServerEvent::TurnStart {
                session_id: None,
                data: json!({}),
            };
            let thinking = ServerEvent::AssistantDelta {
                session_id: None,
                delta: "private reasoning".into(),
                kind: "thinking".into(),
            };
            view.apply_worker_message(WorkerMessage::Event(start.clone()), window, cx);
            view.apply_worker_message(WorkerMessage::Event(thinking.clone()), window, cx);
            assert_eq!(
                view.composer_hint(),
                "zeta is thinking… · Esc stops the turn"
            );
            // Thinking now paints a header-only marker row — never the
            // private reasoning text, which the final assertion of this test
            // still guards against below.
            assert!(matches!(
                view.state.transcript.as_slice(),
                [TranscriptEntry::Thinking]
            ));
            view.apply_worker_message(
                WorkerMessage::Event(ServerEvent::AssistantDelta {
                    session_id: None,
                    delta: String::new(),
                    kind: "assistant".into(),
                }),
                window,
                cx,
            );
            assert!(view.state.thinking);
            view.apply_worker_message(
                WorkerMessage::Event(ServerEvent::AssistantDelta {
                    session_id: None,
                    delta: "hello".into(),
                    kind: "assistant".into(),
                }),
                window,
                cx,
            );
            assert!(!view.state.thinking);
            view.apply_worker_message(WorkerMessage::Event(thinking.clone()), window, cx);
            assert!(!view.state.thinking);
            for end in [
                ServerEvent::TurnEnd {
                    session_id: None,
                    data: json!({}),
                },
                ServerEvent::AgentEnd {
                    session_id: None,
                    data: json!({}),
                },
                ServerEvent::TurnAborted {
                    session_id: None,
                    data: json!({}),
                },
            ] {
                view.apply_worker_message(WorkerMessage::Event(start.clone()), window, cx);
                view.apply_worker_message(WorkerMessage::Event(thinking.clone()), window, cx);
                assert!(view.state.thinking);
                view.apply_worker_message(WorkerMessage::Event(end), window, cx);
                assert!(!view.state.thinking);
            }
            view.apply_worker_message(WorkerMessage::Event(start), window, cx);
            view.apply_worker_message(WorkerMessage::Event(thinking), window, cx);
            view.apply_worker_message(WorkerMessage::Lost("disconnected".into()), window, cx);
            assert!(!view.state.thinking);
            assert!(!format!("{:?}", view.state.transcript).contains("private reasoning"));
        });
    });
}

#[gpui::test]
fn thinking_row_paints_a_generic_header_and_never_leaks_private_reasoning(cx: &mut TestAppContext) {
    // Privacy guard for the thinking chrome. Zeta's provider protocol has no
    // display-safe summary channel — `ContentBlock::Thinking` carries raw
    // reasoning (codex.py Thinking assembly, Anthropic raw thinking) — so
    // the GUI must never render its text. A streamed thinking delta AND a
    // finalized Thinking block both drop their payloads on the way in; the
    // transcript keeps only a header-only marker. This test feeds a sentinel
    // through both paths and asserts the sentinel never surfaces in state or
    // in painted text, and that the generic "+ Thought" header renders.
    use zeta_gui::client::{ContentBlock, Message};
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    let sentinel = "SECRET-PRIVATE-REASONING-NEVER-DISPLAY";
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.apply_worker_message(
                WorkerMessage::Event(ServerEvent::TurnStart {
                    session_id: None,
                    data: json!({}),
                }),
                window,
                cx,
            );
            view.apply_worker_message(
                WorkerMessage::Event(ServerEvent::AssistantDelta {
                    session_id: None,
                    delta: sentinel.into(),
                    kind: "thinking".into(),
                }),
                window,
                cx,
            );
            view.apply_worker_message(
                WorkerMessage::Event(ServerEvent::AssistantMessage {
                    session_id: None,
                    message: Message {
                        role: "assistant".into(),
                        content: vec![ContentBlock::Thinking {
                            text: sentinel.into(),
                        }],
                    },
                }),
                window,
                cx,
            );
        });
        window.draw(cx).clear(cx);
    });
    // State: the transcript holds only a header-only marker; the sentinel
    // reached no field on the way in.
    view.read_with(&visual, |view, _| {
        assert!(
            matches!(
                view.state.transcript.as_slice(),
                [TranscriptEntry::Thinking]
            ),
            "transcript must hold a single header-only Thinking marker, got {:?}",
            view.state.transcript
        );
        // Row-text model: every visible string the render layer paints for
        // any row is built by `row_text::build`. The Thinking row's only
        // contribution is the generic header — no row's model may carry
        // the sentinel on any field.
        for (index, entry) in view.state.transcript.iter().enumerate() {
            let row = zeta_gui::row_text::build(entry, index, &view.state.session_view, true);
            for text in row.visible_strings() {
                assert!(
                    !text.contains(sentinel),
                    "sentinel reached a row's visible text: {text:?}"
                );
            }
        }
        let thinking =
            zeta_gui::row_text::build(&view.state.transcript[0], 0, &view.state.session_view, true);
        assert_eq!(
            thinking.visible_strings(),
            vec![zeta_gui::state::THINKING_HEADER_LABEL],
            "Thinking row model text must be exactly the generic header"
        );
    });
    // Paint: the generic header renders.
    let header_bounds = visual
        .debug_bounds("thinking-header-0")
        .expect("generic thinking header renders");
    assert!(
        header_bounds.size.width > px(0.),
        "header bounds must have non-zero width so glyphs paint"
    );
    // render_log: `render_thinking_row` records the exact color it applied
    // through `state_text(...)`. The thinking header sits at
    // `muted_foreground`; a regression that repaints it at accent or danger
    // fails here.
    visual.update(|_, cx| {
        let expected = cx.theme().muted_foreground;
        let samples = super::render_log::samples();
        let thinking_samples: Vec<_> = samples
            .iter()
            .filter(|sample| sample.row_id.starts_with("thinking-header-"))
            .collect();
        assert!(
            !thinking_samples.is_empty(),
            "render_thinking_row must record a render_log sample"
        );
        for sample in &thinking_samples {
            assert_eq!(
                sample.color, expected,
                "thinking header painted off the muted-foreground token"
            );
        }
    });
    // Row-level fields the previous chrome would have exposed (title,
    // duration, body) must not paint under any selector.
    for stale in ["thinking-title-0", "thinking-duration-0", "thinking-body-0"] {
        assert!(
            visual.debug_bounds(stale).is_none(),
            "removed thinking chrome resurfaced under selector {stale}"
        );
    }
}

#[gpui::test]
fn disabled_send_button_paints_transparent_fill_and_semantic_outline(cx: &mut TestAppContext) {
    // Guard for contract line 85. Kit's Custom variant derives its border
    // color FROM the fill color, so a naive `.color(transparent)` on the
    // variant kills the outline too. The disabled state paints its outline
    // through a plain div instead: transparent fill AND an explicit semantic
    // border color, with the whole presentation dimmed to 0.55 opacity.
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    // Force disable by clearing the active session so `can_send` flips false
    // — the composer keeps rendering, the send button falls into the
    // disabled branch.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.active_session = None;
            cx.notify();
        });
        window.draw(cx).clear(cx);
    });
    let send = visual
        .debug_bounds("send-button")
        .expect("send button paints when disabled");
    let outline_token = theme::palette::border_active();
    visual.update(|window, _| {
        let scaled = send.scale(window.scale_factor());
        let button_quads: Vec<_> = window
            .painted_quads()
            .into_iter()
            .filter(|quad| {
                quad.bounds.top() >= scaled.top() - gpui::ScaledPixels::from(0.5)
                    && quad.bounds.bottom() <= scaled.bottom() + gpui::ScaledPixels::from(0.5)
                    && quad.bounds.left() >= scaled.left() - gpui::ScaledPixels::from(0.5)
                    && quad.bounds.right() <= scaled.right() + gpui::ScaledPixels::from(0.5)
            })
            .collect();
        // Locate the outline quad by border color — asserting on the ACTUAL
        // painted border color, not on a theme constant we chose ourselves.
        let outline = button_quads
            .iter()
            .find(|quad| {
                let color = quad.border_color;
                color.h == outline_token.h
                    && color.s == outline_token.s
                    && color.l == outline_token.l
                    && quad.border_widths.top > gpui::ScaledPixels::default()
            })
            .expect("disabled send button paints a semantic outline");
        // Complete presentation at 0.55 opacity: element opacity multiplies
        // into every painted color's alpha, so the outline alpha lands near
        // outline_token.a * 0.55.
        let expected_alpha = outline_token.a * 0.55;
        assert!(
            (outline.border_color.a - expected_alpha).abs() < 0.02,
            "outline alpha {} must land near 0.55 * token ({expected_alpha})",
            outline.border_color.a
        );
        // No fill quad — background is transparent. The outline quad itself
        // may carry `background = transparent`; a REGRESSION would paint a
        // separate quad with a non-transparent background inside the button
        // bounds. Assert no such quad has visible alpha.
        for quad in &button_quads {
            let bg_alpha = quad.background.as_solid().map_or(0.0, |color| color.a);
            assert!(
                bg_alpha < 0.02,
                "disabled send button must not paint a fill (got alpha {bg_alpha})"
            );
        }
    });
}

#[gpui::test]
fn whitespace_send_shows_hint_until_the_draft_changes(cx: &mut TestAppContext) {
    let (window, view, receiver) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.composer
                .update(cx, |input, cx| input.set_value(" \n\t", window, cx))
        });
    });
    visual.simulate_keystrokes("enter");
    assert!(receiver.try_recv().is_err());
    view.read_with(&visual, |view, _| {
        assert_eq!(
            view.composer_hint(),
            "Type a message or attach an image to send"
        )
    });
    visual.simulate_keystrokes("h");
    view.read_with(&visual, |view, _| {
        assert_eq!(
            view.composer_hint(),
            "Enter sends · Shift-Enter adds a line"
        )
    });
    visual.simulate_keystrokes("enter");
    assert!(matches!(receiver.try_recv(), Ok(CommandMessage::Send(_))));
}

#[gpui::test]
fn new_session_action_uses_the_same_busy_gate_as_the_button(cx: &mut TestAppContext) {
    let (window, view, receiver) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.simulate_keystrokes("cmd-n cmd-n");
    assert!(matches!(
        receiver.try_recv(),
        Ok(CommandMessage::NewSession)
    ));
    assert!(receiver.try_recv().is_err());
    view.read_with(&visual, |view, _| assert!(view.pending_command));
}

#[test]
fn status_and_approval_summaries_are_readable_without_raw_placeholders() {
    assert_eq!(
        polish::status_label(&Default::default()),
        "Usage appears after the first turn"
    );
    let metrics = zeta_gui::state::StatusMetrics {
        model: Some("model".into()),
        tokens: Some(12),
        cache_hit_rate: Some(50.),
    };
    assert_eq!(polish::status_label(&metrics), "12 tokens · 50.0% cache");
    let mut state = AppState::default();
    let mut status = StatusResult {
        session: Some(SessionMetadata {
            model: "model".into(),
            ..session()
        }),
        state: "idle".into(),
        pending_approvals: vec![],
        usage: json!({}),
        compaction_markers: 0,
    };
    state.apply_status(status.clone());
    assert_eq!(
        polish::status_label(&state.metrics),
        "Usage appears after the first turn"
    );
    status.usage = json!({
        "input_tokens": 4, "output_tokens": 4, "cache_read_input_tokens": 4
    });
    state.apply_status(status);
    assert_eq!(
        polish::status_label(&state.metrics),
        "12 tokens · 50.0% cache"
    );
    for (name, args, expected) in [
        (
            "bash",
            json!({"command":"echo one\necho two"}),
            Some("echo one echo two"),
        ),
        (
            "bash",
            json!({"cmd":"echo legacy\npwd"}),
            Some("echo legacy pwd"),
        ),
        (
            "bash",
            json!({"command":"echo current", "cmd":"echo legacy"}),
            Some("echo current"),
        ),
        ("read", json!({"path":"/tmp/test"}), Some("/tmp/test")),
        ("custom", json!({"path":"/tmp/test"}), None),
    ] {
        let call = ToolCall {
            id: "tool".into(),
            name: name.into(),
            arguments: args.as_object().unwrap().clone(),
        };
        assert_eq!(polish::approval_summary(&call).as_deref(), expected);
    }
}

fn thumbnail_attachment(width: u32, height: u32, color: u8) -> ImageAttachment {
    let pixels = image::RgbaImage::from_pixel(width, height, image::Rgba([color, 0, 0, 255]));
    let mut bytes = std::io::Cursor::new(Vec::new());
    pixels
        .write_to(&mut bytes, image::ImageFormat::Png)
        .unwrap();
    ImageAttachment::from_bytes("pixel.png".into(), bytes.get_ref()).unwrap()
}

#[test]
fn thumbnails_downscale_large_images_and_reject_decode_failures() {
    for (width, height, expected) in [(1024, 768, (128, 96)), (768, 1024, (72, 96))] {
        let attachment = thumbnail_attachment(width, height, 1);
        let thumbnail = polish::image_source(&attachment).unwrap();
        let decoded = image::load_from_memory(thumbnail.bytes()).unwrap();
        assert_eq!((decoded.width(), decoded.height()), expected);
        assert!(thumbnail.bytes().len() < attachment.size);
    }
    let invalid = ImageAttachment::from_bytes("broken.png".into(), &png_bytes()).unwrap();
    assert!(polish::image_source(&invalid).is_none());
}

#[gpui::test]
fn sent_image_thumbnail_survives_history_refresh_and_clears_on_switch(cx: &mut TestAppContext) {
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    let image = thumbnail_attachment(16, 12, 1);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.apply_worker_message(
                WorkerMessage::ImagesSent("image".into(), vec![image.clone()]),
                window,
                cx,
            );
            let history = serde_json::from_value(json!([{
                "id":"message", "role":"user", "content":[
                    {"type":"text", "text":"image"},
                    {"type":"attachment", "name":"pixel.png", "size":image.size}
                ]
            }]))
            .unwrap();
            view.apply_worker_message(WorkerMessage::History(history, false), window, cx);
            assert!(view.sent_images.contains_key(&(0, 0)));
        });
        window.draw(cx).clear(cx);
    });
    assert!(visual.debug_bounds("attachment-thumbnail").is_some());
    assert!(visual.debug_bounds("attachment-chip").is_some());
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            let thumbnail = view.sent_images[&(0, 0)].clone();
            assert!(thumbnail.is_asset_cached(cx));
            view.apply_worker_message(
                WorkerMessage::Session(SessionMetadata {
                    session_id: "other".into(),
                    ..session()
                }),
                window,
                cx,
            );
            assert!(view.sent_images.is_empty());
            assert!(!thumbnail.is_asset_cached(cx));
        });
        window.draw(cx).clear(cx);
    });
    assert!(visual.debug_bounds("attachment-thumbnail").is_none());
}

#[gpui::test]
fn sent_thumbnail_cache_evicts_old_assets_and_reset_releases_remaining_assets(
    cx: &mut TestAppContext,
) {
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            let mut sent = Vec::new();
            for index in 0..polish::SENT_IMAGE_LIMIT * 3 {
                view.apply_worker_message(
                    WorkerMessage::ImagesSent(
                        "image".into(),
                        vec![thumbnail_attachment(16, 12, index as u8)],
                    ),
                    window,
                    cx,
                );
                let thumbnail = view.sent_images[&(index, 0)].clone();
                thumbnail.clone().get_render_image(window, cx);
                sent.push(thumbnail);
                assert_eq!(
                    view.sent_images.len(),
                    (index + 1).min(polish::SENT_IMAGE_LIMIT)
                );
                let first_retained = (index + 1).saturating_sub(polish::SENT_IMAGE_LIMIT);
                for (index, thumbnail) in sent.iter().enumerate() {
                    assert_eq!(thumbnail.is_asset_cached(cx), index >= first_retained);
                }
            }
            view.apply_worker_message(WorkerMessage::History(vec![], true), window, cx);
            assert!(view.sent_images.is_empty());
            assert!(sent.iter().all(|image| !image.is_asset_cached(cx)));
        });
    });
}

#[gpui::test]
fn sent_image_decode_failure_keeps_attachment_text(cx: &mut TestAppContext) {
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.apply_worker_message(
                WorkerMessage::ImagesSent(
                    "image".into(),
                    vec![ImageAttachment::from_bytes("broken.png".into(), &png_bytes()).unwrap()],
                ),
                window,
                cx,
            );
            assert!(view.sent_images.is_empty());
            assert_eq!(
                view.state.session_view.attachments[&0],
                vec![("broken.png".into(), 8)]
            );
        });
        window.draw(cx).clear(cx);
    });
    assert!(visual.debug_bounds("attachment-chip").is_some());
    assert!(visual.debug_bounds("attachment-thumbnail").is_none());
}

#[gpui::test]
fn text_only_history_user_row_paints_the_empty_attachment_gap(cx: &mut TestAppContext) {
    // r3 rendered-bounds guard for the attachment tri-state.
    //
    // Text-only history rows carry `Some(vec![])` in
    // `session_view.attachments` (attachment key present but empty) while
    // pre-history rows carry `None`. The typed-seam renderer paints the
    // `.mt_1()` container whenever the value is `Some` — even when the
    // list is empty — so the row height matches the pre-seam behaviour
    // for text-only history rows. A revert that collapses the tri-state
    // to "check for empty list" would skip the container in the
    // `Some(vec![])` case, shrinking the row height by the `.mt_1()`
    // gap. This test measures the actual rendered row bounds so the
    // regression fails here even if the unit test in `row_text.rs`
    // remains green.
    let render_single_user_row =
        |cx: &mut TestAppContext, present_empty: bool| -> gpui::Bounds<gpui::Pixels> {
            let (window, view, _) = setup(cx);
            let mut visual = VisualTestContext::from_window(window.into(), cx);
            visual.update(|window, cx| {
                view.update(cx, |view, cx| {
                    view.state.transcript = vec![TranscriptEntry::User("hi".into())];
                    if present_empty {
                        view.state.session_view.attachments.insert(0, Vec::new());
                    }
                    view.transcript.update(cx, |scroll, cx| scroll.reset(1, cx));
                    cx.notify();
                });
                window.draw(cx).clear(cx);
            });
            visual
                .debug_bounds("transcript-row")
                .expect("transcript row renders")
        };
    let absent = render_single_user_row(cx, false);
    let present_empty = render_single_user_row(cx, true);
    // The attachment container is `.mt_1()` on top of an empty flex row,
    // so the present-empty case must be at least ~4px taller. Allow a
    // sub-logical-pixel slack for scaling arithmetic.
    let delta = present_empty.size.height - absent.size.height;
    assert!(
        delta >= gpui::px(3.),
        "text-only history user row must paint the `.mt_1()` attachment \
         container (Some(vec![]) tri-state); present_empty={:?}, \
         absent={:?}, delta={:?}",
        present_empty.size.height,
        absent.size.height,
        delta,
    );
}

#[gpui::test]
fn refused_image_paste_falls_through_to_clipboard_text(cx: &mut TestAppContext) {
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|_, cx| {
        cx.write_to_clipboard(gpui::ClipboardItem {
            entries: vec![
                gpui::ClipboardEntry::Image(gpui::Image::from_bytes(
                    gpui::ImageFormat::Png,
                    valid_png_bytes(),
                )),
                gpui::ClipboardEntry::String(gpui::ClipboardString::new("fallback text".into())),
            ],
        });
    });
    visual.simulate_keystrokes("cmd-v");
    view.read_with(&visual, |view, cx| {
        assert_eq!(view.valid_attachment_count(), 1);
        assert!(view.composer.read(cx).value().is_empty());
    });
    visual.update(|_, cx| {
        view.update(cx, |view, cx| {
            let base = view.valid_attachments()[0].clone();
            view.clear_composer_images(cx);
            view.add_pending_attachments(vec![Ok(base.clone()); 4], cx);
        });
    });
    visual.simulate_keystrokes("cmd-v");
    view.read_with(&visual, |view, cx| {
        assert_eq!(view.valid_attachment_count(), 4);
        assert_eq!(view.composer.read(cx).value().as_ref(), "fallback text");
    });
}

#[gpui::test]
fn session_menu_requires_management_capabilities(cx: &mut TestAppContext) {
    let (window, view, receiver) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.open_session_edit(session().session_id, true, window, cx);
            assert!(view.session_edit.is_none());
        });
        window.draw(cx).clear(cx);
    });
    assert!(visual.debug_bounds("session-menu").is_none());
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.apply_worker_message(WorkerMessage::SessionManagement(true), window, cx);
        });
        window.draw(cx).clear(cx);
    });
    let row = visual.debug_bounds("session-menu").unwrap();
    visual.simulate_click(row.center(), Default::default());
    visual.run_until_parked();
    visual.update(|window, cx| {
        window.draw(cx).clear(cx);
    });
    visual.simulate_keystrokes("down enter");
    visual.run_until_parked();
    assert!(view.read_with(&visual, |view, _| matches!(
        view.session_edit,
        Some(session_management::SessionEdit::Rename { .. })
    )));
    visual.simulate_keystrokes("n enter");
    assert!(
        matches!(receiver.try_recv(), Ok(CommandMessage::RenameSession(_, name)) if name == "n")
    );
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.apply_worker_message(WorkerMessage::Renamed(session()), window, cx);
        });
        window.draw(cx).clear(cx);
    });
    visual.run_until_parked();
}

#[gpui::test]
fn session_rename_keyboard_cancel_commit_clear_and_error(cx: &mut TestAppContext) {
    let (window, view, receiver) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.session_management = true;
            view.open_session_edit(session().session_id, true, window, cx);
        });
        window.draw(cx).clear(cx);
    });
    visual.simulate_keystrokes("n a m e escape");
    assert!(receiver.try_recv().is_err());
    assert!(view.read_with(&visual, |view, _| view.session_edit.is_none()));
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.open_session_edit(session().session_id, true, window, cx)
        });
    });
    visual.simulate_keystrokes("n a m e enter enter");
    let command = receiver.try_recv().unwrap();
    assert!(
        matches!(command, CommandMessage::RenameSession(id, name) if id == session().session_id && name == "name")
    );
    assert!(receiver.try_recv().is_err());
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.apply_worker_message(
                WorkerMessage::Rejected("name could not be saved".into()),
                window,
                cx,
            );
            assert!(!view.pending_command);
            let Some(session_management::SessionEdit::Rename { input, .. }) = &view.session_edit
            else {
                panic!("rename stays open");
            };
            assert_eq!(input.read(cx).value().as_ref(), "name");
        });
        window.draw(cx).clear(cx);
    });
    assert!(visual.debug_bounds("session-edit-error").is_some());
    visual.simulate_keystrokes("enter");
    assert!(matches!(
        receiver.try_recv(),
        Ok(CommandMessage::RenameSession(..))
    ));
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            let mut renamed = session();
            renamed.name = "name".into();
            view.apply_worker_message(WorkerMessage::Renamed(renamed), window, cx);
            assert!(view.session_edit.is_none());
            assert!(!view.pending_command);
            assert_eq!(
                sidebar::session_label(&view.state.sessions[0], None),
                "name"
            );
            view.open_session_edit(session().session_id, true, window, cx);
            let Some(session_management::SessionEdit::Rename { input, .. }) = &view.session_edit
            else {
                panic!();
            };
            input.update(cx, |input, cx| input.set_value("", window, cx));
        });
    });
    visual.simulate_keystrokes("enter");
    assert!(
        matches!(receiver.try_recv(), Ok(CommandMessage::RenameSession(_, name)) if name.is_empty())
    );
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.apply_worker_message(WorkerMessage::Renamed(session()), window, cx);
            assert_eq!(
                sidebar::session_label(&view.state.sessions[0], None),
                "New conversation"
            );
        });
    });
}

#[gpui::test]
fn session_delete_confirmation_error_and_success_preserve_active_transcript(
    cx: &mut TestAppContext,
) {
    let (window, view, receiver) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.session_management = true;
            view.state
                .transcript
                .push(TranscriptEntry::User("keep this".into()));
            view.open_session_edit(session().session_id, false, window, cx);
        });
        window.draw(cx).clear(cx);
    });
    assert!(receiver.try_recv().is_err());
    visual.simulate_keystrokes("escape");
    assert!(receiver.try_recv().is_err());
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.open_session_edit(session().session_id, false, window, cx)
        });
        window.draw(cx).clear(cx);
    });
    let confirm = visual.debug_bounds("session-edit-confirm").unwrap();
    visual.simulate_click(confirm.center(), Default::default());
    assert!(
        matches!(receiver.try_recv(), Ok(CommandMessage::DeleteSession(id)) if id == session().session_id)
    );
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.apply_worker_message(
                WorkerMessage::Rejected(
                    "select another session before deleting this session".into(),
                ),
                window,
                cx,
            );
            assert!(!view.pending_command);
            assert_eq!(view.state.active_session, Some(session().session_id));
            assert_eq!(
                view.state.transcript,
                [TranscriptEntry::User("keep this".into())]
            );
        });
        window.draw(cx).clear(cx);
    });
    assert!(visual.debug_bounds("session-edit-error").is_some());
    visual.simulate_keystrokes("escape");
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            let mut other = session();
            other.session_id = "other".into();
            view.state.sessions.push(other);
            view.state.saved_transcripts.insert(
                "other".into(),
                vec![TranscriptEntry::User("remove this".into())],
            );
            view.open_session_edit("other".into(), false, window, cx);
        });
    });
    visual.simulate_keystrokes("enter enter");
    assert!(matches!(receiver.try_recv(), Ok(CommandMessage::DeleteSession(id)) if id == "other"));
    assert!(receiver.try_recv().is_err());
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.apply_worker_message(WorkerMessage::Deleted("other".into()), window, cx);
            assert!(!view.pending_command);
            assert!(view.session_edit.is_none());
            assert_eq!(view.state.sessions.len(), 1);
            assert!(!view.state.saved_transcripts.contains_key("other"));
            assert_eq!(view.state.active_session, Some(session().session_id));
            assert_eq!(
                view.state.transcript,
                [TranscriptEntry::User("keep this".into())]
            );
        });
    });
}

#[gpui::test]
fn session_rename_during_slow_stream_keeps_delete_disabled(cx: &mut TestAppContext) {
    let (window, view, receiver) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.session_management = true;
            view.apply_worker_message(
                WorkerMessage::Event(ServerEvent::TurnStart {
                    session_id: None,
                    data: json!({}),
                }),
                window,
                cx,
            );
            view.apply_worker_message(
                WorkerMessage::Event(ServerEvent::AssistantDelta {
                    session_id: None,
                    delta: "first chunk".into(),
                    kind: "assistant".into(),
                }),
                window,
                cx,
            );
            assert!(view.state.streaming);
            view.open_session_edit(session().session_id, false, window, cx);
            assert!(view.session_edit.is_none());
        });
        window.draw(cx).clear(cx);
    });
    // Pause the event stream between chunks while the user renames through the menu.
    let menu = visual.debug_bounds("session-menu").unwrap();
    visual.simulate_click(menu.center(), Default::default());
    visual.run_until_parked();
    visual.update(|window, cx| {
        window.draw(cx).clear(cx);
    });
    visual.simulate_keystrokes("down enter");
    visual.run_until_parked();
    assert!(view.read_with(&visual, |view, _| matches!(
        view.session_edit,
        Some(session_management::SessionEdit::Rename { .. })
    )));
    visual.simulate_keystrokes("n e w enter");
    assert!(
        matches!(receiver.try_recv(), Ok(CommandMessage::RenameSession(_, name)) if name == "new")
    );
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            let mut renamed = session();
            renamed.name = "new".into();
            view.apply_worker_message(WorkerMessage::Renamed(renamed), window, cx);
            assert!(view.state.streaming);
            view.open_session_edit(session().session_id, false, window, cx);
            assert!(view.session_edit.is_none());
            // A delete overlay opened before the turn must also refuse commit.
            view.session_edit = Some(session_management::SessionEdit::Delete {
                id: session().session_id,
                label: "new".into(),
            });
            view.commit_session_edit(cx);
            assert!(!view.pending_command);
            view.session_edit = None;
            view.apply_worker_message(
                WorkerMessage::Event(ServerEvent::AssistantDelta {
                    session_id: None,
                    delta: " next chunk".into(),
                    kind: "assistant".into(),
                }),
                window,
                cx,
            );
            assert_eq!(view.state.sessions[0].name, "new");
        });
        window.draw(cx).clear(cx);
    });
    assert!(receiver.try_recv().is_err());
    visual.run_until_parked();
}

#[gpui::test]
fn sidebar_pins_to_the_wiki_column_width_and_row_height(cx: &mut TestAppContext) {
    // Contract line 81: 216px sidebar with 40px rows. A regression that
    // widens the column back to 280px would leak into every screenshot and
    // shrink the transcript column across the app.
    let (window, _view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let header = visual
        .debug_bounds("sidebar-header")
        .expect("header renders");
    let row = visual
        .debug_bounds("session-row")
        .expect("session row renders");
    // Allow one sub-logical-pixel drift on either side; gpui rounds layout
    // to physical pixels, so the row width may land at 215 or 216 depending
    // on scale factor without violating the contract.
    let width_delta = if header.size.width >= theme::SIDEBAR_WIDTH {
        header.size.width - theme::SIDEBAR_WIDTH
    } else {
        theme::SIDEBAR_WIDTH - header.size.width
    };
    assert!(
        width_delta <= px(1.),
        "sidebar header width {:?} must land near the 216px pin",
        header.size.width
    );
    let row_delta = if row.size.width >= theme::SIDEBAR_WIDTH {
        row.size.width - theme::SIDEBAR_WIDTH
    } else {
        theme::SIDEBAR_WIDTH - row.size.width
    };
    assert!(
        row_delta <= px(1.),
        "session row width {:?} must fit the 216px sidebar",
        row.size.width
    );
    // Row min-height clamps at 40 (contract). Slight sub-logical-pixel
    // rounding is fine as long as the height stays within a tolerance.
    let delta = if row.size.height >= theme::SIDEBAR_ROW_HEIGHT {
        row.size.height - theme::SIDEBAR_ROW_HEIGHT
    } else {
        theme::SIDEBAR_ROW_HEIGHT - row.size.height
    };
    assert!(
        delta <= px(1.),
        "session row height {:?} must land on the 40px floor",
        row.size.height
    );
}

#[gpui::test]
fn current_session_row_paints_no_fill_and_gets_an_accent_dot(cx: &mut TestAppContext) {
    // Contract line 81: the current session paints NO fill (transparent bg,
    // accent text 600) and a small accent dot in the left gutter. A ghost
    // Button's selected-state fill would violate this — the guard checks
    // both the missing fill and the presence of a solid-accent dot.
    let (window, _view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let row = visual
        .debug_bounds("session-row")
        .expect("session row renders");
    let dot = visual
        .debug_bounds("session-current-dot")
        .expect("current session paints its accent dot");
    assert!(
        row.contains(&dot.origin),
        "the accent dot sits inside the row's gutter"
    );
    visual.update(|window, cx| {
        let theme = cx.theme();
        let scaled_row = row.scale(window.scale_factor());
        let scaled_dot = dot.scale(window.scale_factor());
        // Contract line 81: the CURRENT row paints NO row-sized fill —
        // transparent bg, accent text, dot in the gutter. The guard walks
        // every painted quad that spans a full row's width AND lives
        // entirely inside the row's bounds, then rejects anything with a
        // non-transparent background. Mentally swap in a panel fill: the
        // test must fail; swap in a hover tint: it must fail. Only the
        // small dot quad passes because its bounds sit well below the
        // row-width threshold.
        let scale = window.scale_factor();
        let row_width_threshold = scaled_row.size.width - px(4.).scale(scale);
        let dot_bounds_padded = gpui::Bounds {
            origin: gpui::Point {
                x: scaled_dot.origin.x - px(2.).scale(scale),
                y: scaled_dot.origin.y - px(2.).scale(scale),
            },
            size: gpui::Size {
                width: scaled_dot.size.width + px(4.).scale(scale),
                height: scaled_dot.size.height + px(4.).scale(scale),
            },
        };
        let dot_contains = |quad_bounds: gpui::Bounds<gpui::ScaledPixels>| {
            quad_bounds.top() >= dot_bounds_padded.top()
                && quad_bounds.bottom() <= dot_bounds_padded.bottom()
                && quad_bounds.left() >= dot_bounds_padded.left()
                && quad_bounds.right() <= dot_bounds_padded.right()
        };
        let filled: Vec<_> = window
            .painted_quads()
            .into_iter()
            .filter(|quad| {
                let in_row = quad.bounds.top() >= scaled_row.top()
                    && quad.bounds.bottom() <= scaled_row.bottom()
                    && quad.bounds.left() >= scaled_row.left()
                    && quad.bounds.right() <= scaled_row.right();
                // Row-sized fill = wider than most of the row. That excludes
                // the small dot quad (~9px) but catches any hover / active /
                // panel / arbitrary tint that a regression could paint under
                // the label.
                let row_sized = quad.bounds.size.width >= row_width_threshold;
                let has_fill: gpui::Background = quad.background;
                let transparent = has_fill == gpui::transparent_black().into();
                in_row && row_sized && !transparent && !dot_contains(quad.bounds)
            })
            .collect();
        assert!(
            filled.is_empty(),
            "the current session row painted a row-sized fill: {} quads, first={:?}",
            filled.len(),
            filled.first().map(|q| q.background)
        );
        // Legacy pin: even if a future fill were narrower than a full row,
        // the sidebar_accent / list_active tints are the two Kit tokens that
        // a `.ghost().selected()` regression would paint here. Keep both
        // named assertions so the guard reads as intentional.
        let selected_tint = window.painted_quads().into_iter().any(|quad| {
            let in_row = quad.bounds.top() >= scaled_row.top()
                && quad.bounds.bottom() <= scaled_row.bottom()
                && quad.bounds.left() >= scaled_row.left()
                && quad.bounds.right() <= scaled_row.right();
            in_row
                && (quad.background == theme.sidebar_accent.into()
                    || quad.background == theme.list_active.into())
        });
        assert!(
            !selected_tint,
            "the current session row painted a selected-tint fill"
        );
        // A solid accent quad exists somewhere on the row — the dot.
        let scaled_dot = dot.scale(window.scale_factor());
        let dot_quads: Vec<_> = window
            .painted_quads()
            .into_iter()
            .filter(|quad| {
                quad.background == theme.primary.into()
                    && quad.bounds.top() >= scaled_dot.top() - px(1.).scale(window.scale_factor())
                    && quad.bounds.bottom()
                        <= scaled_dot.bottom() + px(1.).scale(window.scale_factor())
            })
            .collect();
        assert!(!dot_quads.is_empty(), "the accent dot painted its fill");
    });
}

#[gpui::test]
fn connection_lost_paints_a_blocker_row_with_a_danger_rail(cx: &mut TestAppContext) {
    // Contract line 83: blocker row = border-left 2px danger + danger 10%
    // tint bg. The banner replaces the old Alert::error card — this pin
    // catches a regression that would restore the framed alert.
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.apply_worker_message(WorkerMessage::Lost("socket closed".into()), window, cx);
        });
        window.draw(cx).clear(cx);
    });
    let banner = visual
        .debug_bounds("connection-lost")
        .expect("banner renders");
    visual.update(|window, cx| {
        let theme = cx.theme();
        let scaled_banner = banner.scale(window.scale_factor());
        let rail = window
            .painted_quads()
            .into_iter()
            .find(|quad| {
                quad.border_color == theme.danger
                    && quad.border_widths.left > gpui::ScaledPixels::default()
                    && quad.bounds.top()
                        >= scaled_banner.top() - px(1.).scale(window.scale_factor())
                    && quad.bounds.bottom()
                        <= scaled_banner.bottom() + px(1.).scale(window.scale_factor())
            })
            .expect("blocker rail paints on the connection-lost banner");
        let expected = px(f32::from(theme::ATTENTION_RAIL_WIDTH)).scale(window.scale_factor());
        let delta = if rail.border_widths.left > expected {
            rail.border_widths.left - expected
        } else {
            expected - rail.border_widths.left
        };
        assert!(
            delta <= px(0.5).scale(window.scale_factor()),
            "blocker rail width {:?} must land on the 2px contract",
            rail.border_widths.left
        );
        let tint = window.painted_quads().into_iter().find(|quad| {
            quad.background == theme::palette::danger_tint().into()
                && quad.bounds.top() >= scaled_banner.top() - px(1.).scale(window.scale_factor())
                && quad.bounds.bottom()
                    <= scaled_banner.bottom() + px(1.).scale(window.scale_factor())
        });
        assert!(tint.is_some(), "banner painted its 10% danger tint fill");
    });
}

#[gpui::test]
fn status_dot_paints_accent_at_rest_and_danger_when_offline(cx: &mut TestAppContext) {
    // ZETA-123: the state indicator is a small dot next to a mode word,
    // not a filled pill — the header reads as a quiet status band, not a
    // call-to-action. The dot still carries the single load-bearing
    // color on the strip: neutral = accent, offline = danger. A
    // regression that dropped the color (or restored a full-width pill
    // fill) would show up here.
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let mode = visual
        .debug_bounds("footer-mode")
        .expect("footer mode indicator renders");
    let dot = visual
        .debug_bounds("run-header-status-dot")
        .expect("state dot renders");
    // The dot lives inside the footer-mode cluster.
    assert!(
        mode.contains(&dot.center()),
        "state dot must sit inside the footer-mode cluster"
    );
    // Dot is a small square (round via border-radius), not the wide
    // filled pill it replaced.
    assert!(
        dot.size.width <= px(12.),
        "state dot must stay a small glyph, saw width {:?}",
        dot.size.width
    );
    visual.update(|window, cx| {
        let theme = cx.theme();
        let scaled = dot.scale(window.scale_factor());
        let neutral = window.painted_quads().into_iter().find(|quad| {
            quad.background == theme.primary.into()
                && quad.bounds.top() >= scaled.top() - px(1.).scale(window.scale_factor())
                && quad.bounds.bottom() <= scaled.bottom() + px(1.).scale(window.scale_factor())
        });
        assert!(neutral.is_some(), "neutral state paints the dot in accent");
    });
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.apply_worker_message(WorkerMessage::Lost("network gone".into()), window, cx);
        });
        window.draw(cx).clear(cx);
    });
    let dot = visual
        .debug_bounds("run-header-status-dot")
        .expect("state dot renders while offline");
    visual.update(|window, cx| {
        let theme = cx.theme();
        let scaled = dot.scale(window.scale_factor());
        let danger = window.painted_quads().into_iter().find(|quad| {
            quad.background == theme.danger.into()
                && quad.bounds.top() >= scaled.top() - px(1.).scale(window.scale_factor())
                && quad.bounds.bottom() <= scaled.bottom() + px(1.).scale(window.scale_factor())
        });
        assert!(danger.is_some(), "offline state paints the dot in danger");
    });
}

#[gpui::test]
fn modals_paint_a_flat_panel_on_the_scrim_at_the_wiki_top_offset(cx: &mut TestAppContext) {
    // Contract line 91: flat panel — bg panel, no shadow, no border, width
    // 480, seated below a scrim at 25% of the viewport height. The guard
    // pins both settings and session-edit; regressing either to a bordered
    // card would show up here first.
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.session_view.available = true;
            view.apply_worker_message(
                WorkerMessage::Settings(
                    SessionSettings {
                        model: "claude-opus-4-7".into(),
                        approval_mode: "ask".into(),
                    },
                    ModelCatalog {
                        models: vec!["claude-opus-4-7".into()],
                        providers: [("claude-opus-4-7".into(), "claude".into())]
                            .into_iter()
                            .collect(),
                    },
                ),
                window,
                cx,
            );
        });
        window.draw(cx).clear(cx);
    });
    let panel = visual
        .debug_bounds("settings-panel")
        .expect("settings panel renders");
    assert_eq!(panel.size.width, theme::MODAL_WIDTH);
    visual.update(|window, cx| {
        let theme = cx.theme();
        let scaled_panel = panel.scale(window.scale_factor());
        // Panel bg paints as the panel/sidebar token, not the app canvas.
        let filled = window.painted_quads().into_iter().find(|quad| {
            quad.background == theme.sidebar.into()
                && quad.bounds.top() >= scaled_panel.top() - px(1.).scale(window.scale_factor())
                && quad.bounds.bottom()
                    <= scaled_panel.bottom() + px(1.).scale(window.scale_factor())
                && quad.bounds.left() >= scaled_panel.left() - px(1.).scale(window.scale_factor())
                && quad.bounds.right() <= scaled_panel.right() + px(1.).scale(window.scale_factor())
        });
        assert!(filled.is_some(), "settings panel paints on the panel token");
        // No border rail on the panel itself — a regression that restored
        // .border_1() would paint a bordered quad on the panel bounds.
        let bordered: Vec<_> = window
            .painted_quads()
            .into_iter()
            .filter(|quad| {
                let border = quad.border_widths.left
                    + quad.border_widths.right
                    + quad.border_widths.top
                    + quad.border_widths.bottom;
                border > gpui::ScaledPixels::default() && quad.bounds == scaled_panel
            })
            .collect();
        assert!(
            bordered.is_empty(),
            "settings panel painted a border rail: {} quads",
            bordered.len()
        );
    });
    // Sits at 25% of the viewport height — the exact wiki `top` offset.
    // A 25% mark lands cleanly on a pixel grid, so drift beyond layout
    // rounding (~1 logical px) means someone shifted the offset itself,
    // not a fractional-pixel rounding wobble.
    let overlay = visual
        .debug_bounds("settings-overlay")
        .expect("settings overlay renders");
    let target = overlay.top() + overlay.size.height * theme::MODAL_TOP_FRACTION;
    let drift = if panel.top() > target {
        panel.top() - target
    } else {
        target - panel.top()
    };
    assert!(
        drift <= px(1.),
        "settings panel top {:?} must land within 1px (layout rounding) \
         of the 25% mark ({:?})",
        panel.top(),
        target,
    );
    // Zero radius on the panel — contract line 91 pins a flat rectangle.
    visual.update(|window, cx| {
        let scaled_panel = panel.scale(window.scale_factor());
        let rounded: Vec<_> = window
            .painted_quads()
            .into_iter()
            .filter(|quad| {
                quad.bounds == scaled_panel
                    && (quad.corner_radii.top_left > gpui::ScaledPixels::default()
                        || quad.corner_radii.top_right > gpui::ScaledPixels::default()
                        || quad.corner_radii.bottom_left > gpui::ScaledPixels::default()
                        || quad.corner_radii.bottom_right > gpui::ScaledPixels::default())
            })
            .collect();
        assert!(
            rounded.is_empty(),
            "settings panel painted a rounded corner ({} quads)",
            rounded.len()
        );
        // No shadow behind the panel. gpui lowers `box-shadow` to a
        // Shadow primitive (dedicated GPU pass) rather than a Quad, so
        // `painted_quads()` cannot see shadow primitives directly. What
        // it CAN see is the theme flags every shadow pass reads at paint
        // time — Kit skips shadow emission entirely when both are false,
        // and this assertion runs inside the same paint frame as the
        // panel above, so it captures the paint-time state (not a
        // constant). A mutation that flips either flag would trip here.
        let theme = cx.theme();
        assert!(
            !theme.shadow,
            "flat modal must paint with theme.shadow = false; got {}",
            theme.shadow
        );
        assert!(
            !theme.tile_shadow,
            "flat modal must paint with theme.tile_shadow = false; got {}",
            theme.tile_shadow
        );
    });
}

#[gpui::test]
fn session_edit_modal_matches_the_wiki_flat_panel_shape(cx: &mut TestAppContext) {
    // Twin of the settings modal: same flat-panel shape must land on the
    // rename/delete overlay too. Contract line 91.
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.session_management = true;
            view.open_session_edit(session().session_id, true, window, cx);
        });
        window.draw(cx).clear(cx);
    });
    let panel = visual
        .debug_bounds("session-edit-panel")
        .expect("session-edit panel renders");
    assert_eq!(panel.size.width, theme::MODAL_WIDTH);
    visual.update(|window, cx| {
        let theme = cx.theme();
        let scaled_panel = panel.scale(window.scale_factor());
        let filled = window.painted_quads().into_iter().find(|quad| {
            quad.background == theme.sidebar.into()
                && quad.bounds.top() >= scaled_panel.top() - px(1.).scale(window.scale_factor())
                && quad.bounds.bottom()
                    <= scaled_panel.bottom() + px(1.).scale(window.scale_factor())
                && quad.bounds.left() >= scaled_panel.left() - px(1.).scale(window.scale_factor())
                && quad.bounds.right() <= scaled_panel.right() + px(1.).scale(window.scale_factor())
        });
        assert!(
            filled.is_some(),
            "session-edit panel paints on the panel token"
        );
        // Zero radius — contract line 91 pins a flat rectangle. A future
        // rounded card here would fail this even if the panel color and
        // scrim offset stay put.
        let rounded: Vec<_> = window
            .painted_quads()
            .into_iter()
            .filter(|quad| {
                quad.bounds == scaled_panel
                    && (quad.corner_radii.top_left > gpui::ScaledPixels::default()
                        || quad.corner_radii.top_right > gpui::ScaledPixels::default()
                        || quad.corner_radii.bottom_left > gpui::ScaledPixels::default()
                        || quad.corner_radii.bottom_right > gpui::ScaledPixels::default())
            })
            .collect();
        assert!(
            rounded.is_empty(),
            "session-edit panel painted a rounded corner ({} quads)",
            rounded.len()
        );
        // Twin shadow guard — see the settings-modal test for why this
        // reads the theme flags at paint time (gpui shadows are Shadow
        // primitives, not Quads, so `painted_quads()` cannot observe
        // them directly). A mutation that flips either flag to true
        // trips this assertion.
        let theme = cx.theme();
        assert!(
            !theme.shadow,
            "flat modal must paint with theme.shadow = false; got {}",
            theme.shadow
        );
        assert!(
            !theme.tile_shadow,
            "flat modal must paint with theme.tile_shadow = false; got {}",
            theme.tile_shadow
        );
    });
    // 25% top offset — same shelf as the settings modal, contract line 91.
    let overlay = visual
        .debug_bounds("session-edit")
        .expect("session-edit overlay renders");
    let target = overlay.top() + overlay.size.height * theme::MODAL_TOP_FRACTION;
    let drift = if panel.top() > target {
        panel.top() - target
    } else {
        target - panel.top()
    };
    assert!(
        drift <= px(1.),
        "session-edit panel top {:?} must land within 1px (layout rounding) \
         of the 25% mark ({:?})",
        panel.top(),
        target,
    );
    // The input frame paints a bottom-only underline — contract line 91:
    // "Inputs: no box, border-bottom 1px only, focus promotes underline".
    // The rename modal grabs focus on open, so the underline paints in the
    // ring color; a blurred re-render drops it back to the plain border.
    let frame = visual
        .debug_bounds("session-edit-input-frame")
        .expect("session-edit input frame renders");
    visual.update(|window, cx| {
        let theme = cx.theme();
        let scaled_frame = frame.scale(window.scale_factor());
        let underline = window.painted_quads().into_iter().find(|quad| {
            (quad.border_color == theme.ring || quad.border_color == theme.border)
                && quad.border_widths.bottom > gpui::ScaledPixels::default()
                && quad.border_widths.top == gpui::ScaledPixels::default()
                && quad.border_widths.left == gpui::ScaledPixels::default()
                && quad.border_widths.right == gpui::ScaledPixels::default()
                && quad.bounds.top() >= scaled_frame.top() - px(1.).scale(window.scale_factor())
                && quad.bounds.bottom()
                    <= scaled_frame.bottom() + px(1.).scale(window.scale_factor())
        });
        assert!(
            underline.is_some(),
            "session-edit input paints its bottom-only underline (border or ring)"
        );
        // Focused input promotes the underline to the ring color. Rip the
        // focus off the textarea and assert it falls back to `theme.border`.
        assert_eq!(
            underline.unwrap().border_color,
            theme.ring,
            "the freshly opened rename modal grabs input focus, so the underline must paint on the ring"
        );
    });
    visual.update(|window, cx| {
        // Blur the input by focusing a fresh unattached handle; the
        // underline should fall back to `theme.border`.
        let elsewhere = cx.focus_handle();
        window.focus(&elsewhere, cx);
        window.draw(cx).clear(cx);
    });
    visual.update(|window, cx| {
        let theme = cx.theme();
        let scaled_frame = frame.scale(window.scale_factor());
        let underline_border = window.painted_quads().into_iter().find(|quad| {
            quad.border_color == theme.border
                && quad.border_widths.bottom > gpui::ScaledPixels::default()
                && quad.bounds.top() >= scaled_frame.top() - px(1.).scale(window.scale_factor())
                && quad.bounds.bottom()
                    <= scaled_frame.bottom() + px(1.).scale(window.scale_factor())
        });
        assert!(
            underline_border.is_some(),
            "blurred session-edit input drops the underline back onto `theme.border`"
        );
    });
}

#[gpui::test]
fn run_header_rules_metadata_cluster_with_two_vertical_separators(cx: &mut TestAppContext) {
    // The single-row header (ZETA-123) carries three right-aligned
    // metadata pieces — tokens/cache, dot + state word, and the model
    // chip — separated by two 1x14 vertical rules at the border tier.
    // A regression that dropped a rule would fuse the metadata slots
    // into one uniform run; one that added extra rules would paint
    // over the strip. The old two-band strip is gone, so `run-header`
    // now hosts these rules directly.
    let (window, _view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let header = visual
        .debug_bounds("run-header")
        .expect("run header renders");
    visual.update(|window, cx| {
        let theme = cx.theme();
        let scaled = header.scale(window.scale_factor());
        let rules: Vec<_> = window
            .painted_quads()
            .into_iter()
            .filter(|quad| {
                quad.background == theme.border.into()
                    && quad.bounds.top() >= scaled.top()
                    && quad.bounds.bottom() <= scaled.bottom()
                    && quad.bounds.size.width <= px(2.).scale(window.scale_factor())
            })
            .collect();
        assert_eq!(
            rules.len(),
            2,
            "expected two vertical rules between metadata slots, saw {}",
            rules.len()
        );
    });
}

#[gpui::test]
fn sidebar_rows_are_tab_focusable_paint_a_focus_cursor_and_activate_on_enter_and_space(
    cx: &mut TestAppContext,
) {
    // A11y regression guard (finding #2 — expanded round 3). Sidebar rows
    // must be:
    //   1. reachable by Tab (`.tab_index(0)` populates the window's tab
    //      stops that `window.focus_next` walks),
    //   2. paint the solid-accent keyboard cursor on the FOCUSED row so a
    //      keyboard-only user sees which row Enter/Space would activate —
    //      the current row must show the cursor too; the "no fill"
    //      contract only applies to the UNFOCUSED current row,
    //   3. activate on Enter AND Space (button-role keyboard contract).
    // Both session and branch rows share this contract. Mutations that
    // must fail: dropping the focus-cursor branch, dropping a key
    // handler, or dropping `.tab_index(0)` from either row type.
    let (window, view, receiver) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    let session_target = "cd34beef1234";
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            let mut other = session();
            other.session_id = session_target.into();
            other.name = "other".into();
            view.state.sessions.push(other);
            view.state.session_view.available = true;
            view.state.session_view.branches = vec![
                Branch {
                    id: "trunk".into(),
                    label: "main".into(),
                    depth: 0,
                    current: true,
                },
                Branch {
                    id: "alt".into(),
                    label: "alt".into(),
                    depth: 1,
                    current: false,
                },
            ];
            view.apply_worker_message(WorkerMessage::Connected, window, cx);
        });
        window.draw(cx).clear(cx);
    });
    while receiver.try_recv().is_ok() {}

    let session_handle = visual
        .update(|_, cx| {
            view.read(cx)
                .sidebar_row_focus
                .borrow()
                .get(session_target)
                .cloned()
        })
        .expect("sidebar row focus handle exists after render");
    let current_session_handle = visual
        .update(|_, cx| {
            view.read(cx)
                .sidebar_row_focus
                .borrow()
                .get(&session().session_id)
                .cloned()
        })
        .expect("current session row focus handle exists after render");
    let branch_handle = visual
        .update(|_, cx| {
            view.read(cx)
                .sidebar_row_focus
                .borrow()
                .get("branch:alt")
                .cloned()
        })
        .expect("branch row focus handle exists after render");
    let current_branch_handle = visual
        .update(|_, cx| {
            view.read(cx)
                .sidebar_row_focus
                .borrow()
                .get("branch:trunk")
                .cloned()
        })
        .expect("current branch row focus handle exists after render");

    // Row-sized paint check. `debug_bounds("session-row")` is ambiguous
    // when multiple rows share the selector, so we assert on the SIZE of
    // the accent quad — the focus cursor fills the whole row rectangle
    // (~SIDEBAR_WIDTH × SIDEBAR_ROW_HEIGHT). The current-item dot is
    // painted in the same accent color, but at 9×9px, so a size floor
    // near the row's own footprint rejects it.
    let row_sized_accent = |visual: &mut VisualTestContext, height: gpui::Pixels| -> bool {
        visual.update(|window, cx| {
            let theme = cx.theme();
            let scale = window.scale_factor();
            let width_floor = (theme::SIDEBAR_WIDTH * 0.9).scale(scale);
            let height_floor = (height * 0.9).scale(scale);
            window.painted_quads().into_iter().any(|q| {
                q.background == theme.primary.into()
                    && q.bounds.size.width >= width_floor
                    && q.bounds.size.height >= height_floor
            })
        })
    };

    // --- The CURRENT session row (active) MUST paint the accent cursor
    // when focused. Contract: the current-item "no fill" rule applies to
    // the UNFOCUSED state only; a focused row overrides it. A mutation
    // that reintroduces the old `focused && !active` guard would leave
    // the current row with no visible focus cursor, and this check
    // fails. ---
    visual.update(|window, cx| {
        window.focus(&current_session_handle, cx);
        window.draw(cx).clear(cx);
    });
    assert!(
        row_sized_accent(&mut visual, theme::SIDEBAR_ROW_HEIGHT),
        "the CURRENT session row must still paint the accent cursor \
         when focused (no-fill rule applies only to the unfocused state)"
    );

    // --- Non-active session row also paints the cursor on focus. ---
    visual.update(|window, cx| {
        window.focus(&session_handle, cx);
        window.draw(cx).clear(cx);
    });
    assert!(
        row_sized_accent(&mut visual, theme::SIDEBAR_ROW_HEIGHT),
        "focused session row must paint the accent focus cursor"
    );

    // --- Enter activates the focused session row. ---
    visual.simulate_keystrokes("enter");
    let mut resumed = None;
    while let Ok(msg) = receiver.try_recv() {
        if let CommandMessage::Resume(id) = msg {
            resumed = Some(id);
            break;
        }
    }
    assert_eq!(
        resumed.as_deref(),
        Some(session_target),
        "Enter on a focused session row must dispatch Resume for that id"
    );

    // --- Space activates the focused session row. ---
    visual.update(|window, cx| {
        view.update(cx, |view, _| view.pending_command = false);
        window.focus(&session_handle, cx);
        window.draw(cx).clear(cx);
    });
    visual.simulate_keystrokes("space");
    let mut resumed_space = None;
    while let Ok(msg) = receiver.try_recv() {
        if let CommandMessage::Resume(id) = msg {
            resumed_space = Some(id);
            break;
        }
    }
    assert_eq!(
        resumed_space.as_deref(),
        Some(session_target),
        "Space on a focused session row must dispatch Resume for that id"
    );

    // --- The CURRENT branch row (trunk) must also paint the accent
    // cursor when focused. Same contract as sessions — focused wins
    // over the unfocused "no fill" rule. Reset `pending_command` first
    // (session activation set it, and the branch row captures
    // `can_activate` at render time). ---
    visual.update(|window, cx| {
        view.update(cx, |view, _| view.pending_command = false);
        window.focus(&current_branch_handle, cx);
        window.draw(cx).clear(cx);
    });
    assert!(
        row_sized_accent(&mut visual, theme::SIDEBAR_NESTED_ROW_HEIGHT),
        "the CURRENT branch row must still paint the accent cursor \
         when focused (no-fill rule applies only to the unfocused state)"
    );

    // --- Non-current branch row (alt) also paints on focus. ---
    visual.update(|window, cx| {
        window.focus(&branch_handle, cx);
        window.draw(cx).clear(cx);
    });
    assert!(
        row_sized_accent(&mut visual, theme::SIDEBAR_NESTED_ROW_HEIGHT),
        "focused branch row must paint the accent focus cursor"
    );

    // --- Enter on a focused branch row dispatches SwitchBranch. ---
    visual.simulate_keystrokes("enter");
    let mut switched = None;
    while let Ok(msg) = receiver.try_recv() {
        if let CommandMessage::SwitchBranch(id) = msg {
            switched = Some(id);
            break;
        }
    }
    assert_eq!(
        switched.as_deref(),
        Some("alt"),
        "Enter on a focused branch row must dispatch SwitchBranch for that id"
    );

    // --- Space on a focused branch row dispatches SwitchBranch. ---
    visual.update(|window, cx| {
        view.update(cx, |view, _| view.pending_command = false);
        window.focus(&branch_handle, cx);
        window.draw(cx).clear(cx);
    });
    visual.simulate_keystrokes("space");
    let mut switched_space = None;
    while let Ok(msg) = receiver.try_recv() {
        if let CommandMessage::SwitchBranch(id) = msg {
            switched_space = Some(id);
            break;
        }
    }
    assert_eq!(
        switched_space.as_deref(),
        Some("alt"),
        "Space on a focused branch row must dispatch SwitchBranch for that id"
    );
}

#[gpui::test]
fn current_session_dot_lands_on_the_9px_left_4px_contract(cx: &mut TestAppContext) {
    // Contract line 81: current-item dot ~0.58em (9px at the 15px base)
    // pinned to `left 4px` inside the row's gutter. Finding #11 flagged
    // the previous 5px-centred sizing; this guard pins the new layout so
    // a regression to the old constants shows up here.
    let (window, _view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let row = visual
        .debug_bounds("session-row")
        .expect("session row renders");
    let dot = visual
        .debug_bounds("session-current-dot")
        .expect("current session paints its accent dot");
    // Size lands on the SIDEBAR_CURRENT_DOT_SIZE token.
    let size_delta = if dot.size.width > theme::SIDEBAR_CURRENT_DOT_SIZE {
        dot.size.width - theme::SIDEBAR_CURRENT_DOT_SIZE
    } else {
        theme::SIDEBAR_CURRENT_DOT_SIZE - dot.size.width
    };
    assert!(
        size_delta <= px(1.),
        "dot width {:?} must land on the 9px contract",
        dot.size.width
    );
    // Left inset — dot.left - row.left ~= 4px.
    let inset = dot.left() - row.left();
    let inset_delta = if inset > theme::SIDEBAR_CURRENT_DOT_INSET {
        inset - theme::SIDEBAR_CURRENT_DOT_INSET
    } else {
        theme::SIDEBAR_CURRENT_DOT_INSET - inset
    };
    assert!(
        inset_delta <= px(1.),
        "dot left inset {:?} must land on the 4px contract",
        inset
    );
}

#[gpui::test]
fn sidebar_right_rule_paints_the_subtle_border(cx: &mut TestAppContext) {
    // Contract line 3: sidebar right rule uses the SUBTLE border tier so
    // the column reads as a seam, not a hard rule. A regression back to
    // `palette::border()` shifts the seam one tint step darker; the paint
    // probe below catches it by scanning for a full-height border quad on
    // the sidebar's right edge.
    let (window, _view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| window.draw(cx).clear(cx));
    visual.update(|window, _cx| {
        let subtle = theme::palette::border_subtle();
        let bright = theme::palette::border();
        let scale = window.scale_factor();
        // Any painted quad whose right border color matches the subtle
        // token counts as a passing match. We also assert the bright
        // border token is NOT the one used, so a future regression that
        // routed sidebar_border back to palette::border() is caught.
        let sidebar_right = theme::SIDEBAR_WIDTH.scale(scale);
        let matches: Vec<_> = window
            .painted_quads()
            .into_iter()
            .filter(|quad| {
                quad.border_color == subtle
                    && quad.border_widths.right > gpui::ScaledPixels::default()
                    && quad.bounds.left() <= sidebar_right
            })
            .collect();
        assert!(
            !matches.is_empty(),
            "expected the sidebar edge to paint on the subtle border tier"
        );
        let tolerance = px(2.).scale(scale);
        let bright_matches = window.painted_quads().into_iter().any(|quad| {
            let right_delta = if quad.bounds.right() > sidebar_right {
                quad.bounds.right() - sidebar_right
            } else {
                sidebar_right - quad.bounds.right()
            };
            quad.border_color == bright
                && quad.border_widths.right > gpui::ScaledPixels::default()
                && quad.bounds.top() < px(200.).scale(scale)
                && quad.bounds.left() <= sidebar_right
                && right_delta <= tolerance
        });
        assert!(
            !bright_matches,
            "sidebar edge painted on the bright border tier"
        );
    });
}

#[gpui::test]
fn run_header_paints_a_single_row_without_the_keyboard_hint(cx: &mut TestAppContext) {
    // ZETA-123: the run header collapses to ONE 44px row. The session
    // title anchors the left; a right-aligned metadata cluster carries
    // quiet tokens/cache, a dot + state word (not a filled pill), and
    // the model name. The keyboard shortcut hint that used to sit here
    // now lives in the composer footer — metadata sits next to what it
    // describes (laws-of-ux: Proximity). A regression that reintroduced
    // the second band, brought the "Enter sends" fallback back to the
    // header, or dropped the model chip out of it would show up here.
    let (window, _view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let header = visual
        .debug_bounds("run-header")
        .expect("run header renders");
    // The header is one row on the 44px floor — a second band would
    // push its height past ~52px.
    let delta = if header.size.height > theme::HEADER_BAND1_MIN_HEIGHT {
        header.size.height - theme::HEADER_BAND1_MIN_HEIGHT
    } else {
        theme::HEADER_BAND1_MIN_HEIGHT - header.size.height
    };
    assert!(
        delta <= px(4.),
        "run header height {:?} must land on the 44px floor",
        header.size.height
    );
    // Old two-band selectors must be gone.
    assert!(
        visual.debug_bounds("run-header-band1").is_none(),
        "the two-band selector `run-header-band1` must be gone",
    );
    assert!(
        visual.debug_bounds("status-bar").is_none(),
        "the two-band selector `status-bar` must be gone",
    );
    assert!(
        visual.debug_bounds("run-header-step").is_none(),
        "step text must not sit in the header — the composer footer owns the hint",
    );
    // The single-row header carries the title, dot+word, and model chip.
    assert!(
        visual.debug_bounds("run-header-title").is_some(),
        "header must render the session title"
    );
    assert!(
        visual.debug_bounds("footer-mode").is_some(),
        "header must render the state indicator"
    );
    assert!(
        visual.debug_bounds("run-header-status-dot").is_some(),
        "state indicator must paint as a dot glyph, not a filled pill"
    );
    assert!(
        visual.debug_bounds("run-header-model").is_some(),
        "header must render the model chip"
    );
    assert!(
        visual.debug_bounds("status-metrics").is_some(),
        "header must render the quiet metrics slot"
    );
    // The keyboard hint moved to the composer footer.
    let footer = visual
        .debug_bounds("composer-footer")
        .expect("composer footer renders");
    let hint = visual
        .debug_bounds("composer-hint")
        .expect("keyboard hint renders in the composer footer");
    assert!(
        footer.contains(&hint.center()),
        "the keyboard hint must sit inside the composer footer, not the header"
    );
    // The model chip has moved out of the input row and now sits in
    // the composer footer next to the hint.
    let target = visual
        .debug_bounds("composer-target")
        .expect("composer target renders in the footer");
    assert!(
        footer.contains(&target.center()),
        "the composer model target must sit in the footer, not above the input row"
    );
}

#[gpui::test]
fn run_header_title_survives_narrow_widths_and_metadata_never_overflows(cx: &mut TestAppContext) {
    // ZETA-123 round 2, finding 2: at 760px the title measured 0px and
    // the model text painted past the right edge — quiet metadata was
    // pinned as `flex_shrink_0` and starved the title. The fix keeps
    // the title as a flex_1 spacer with `min_w_0` (no max_w cap) and
    // lets tokens/model shrink and truncate first. Guard against a
    // regression at three widths: narrow-ish (600), the reproduced
    // failure (760), and wide (1200).
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    // Populate a realistic metrics load — long model id + tokens/cache
    // string — so the shrink path exercises real content.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.metrics.model = Some("claude-opus-4-7-super-long-model-identifier".into());
            view.state.metrics.tokens = Some(123_456);
            view.state.metrics.cache_hit_rate = Some(0.42);
            cx.notify();
            let _ = window;
        });
    });
    for probe_width in [px(600.), px(760.), px(1200.)] {
        visual.simulate_resize(gpui::size(probe_width, px(760.)));
        visual.update(|window, cx| window.draw(cx).clear(cx));
        let header = visual
            .debug_bounds("run-header")
            .expect("run header renders at every probe width");
        let title = visual
            .debug_bounds("run-header-title")
            .expect("title renders");
        let model = visual
            .debug_bounds("run-header-model")
            .expect("model chip renders");
        let metrics = visual
            .debug_bounds("status-metrics")
            .expect("status-metrics slot renders");
        // Title must survive with a scannable measure — at least ~48px
        // (a few characters). Zero-width title reads as "the header
        // has no identity" and is the exact bug we're guarding.
        assert!(
            title.size.width >= px(48.),
            "title width {:?} collapsed at width {:?} — metadata cluster ate the row",
            title.size.width,
            probe_width,
        );
        // No metadata slot paints past the header's right edge.
        let right_edge = header.right();
        for (name, bounds) in [("model", model), ("status-metrics", metrics)] {
            assert!(
                bounds.right() <= right_edge + px(1.),
                "{name} bounds {:?} paint past the header right edge {:?} at width {:?}",
                bounds,
                right_edge,
                probe_width,
            );
        }
    }
}

#[gpui::test]
fn run_header_status_dot_is_the_only_dot_and_pulses_when_busy(cx: &mut TestAppContext) {
    // ZETA-123 round 2, finding 4: streaming/thinking used to paint a
    // SECOND dot next to the status dot — the reader saw two pulses
    // and wondered which one was authoritative. One dot per state
    // (Selective Attention). The same run-header-status-dot pulses
    // when busy; the separate `streaming-dot` selector is gone.
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| window.draw(cx).clear(cx));
    // Rest: one status dot renders, no streaming-dot selector exists.
    assert!(
        visual.debug_bounds("run-header-status-dot").is_some(),
        "status dot renders at rest",
    );
    assert!(
        visual.debug_bounds("streaming-dot").is_none(),
        "the second streaming-dot selector must be gone — one dot per state",
    );
    // Enter streaming state and confirm we STILL have exactly one dot.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.apply_worker_message(
                WorkerMessage::Event(ServerEvent::TurnStart {
                    session_id: None,
                    data: json!({}),
                }),
                window,
                cx,
            );
            view.apply_worker_message(
                WorkerMessage::Event(ServerEvent::AssistantDelta {
                    session_id: None,
                    delta: "chunk".into(),
                    kind: "assistant".into(),
                }),
                window,
                cx,
            );
            assert!(view.state.streaming);
        });
        window.draw(cx).clear(cx);
    });
    assert!(
        visual.debug_bounds("run-header-status-dot").is_some(),
        "status dot still renders while streaming",
    );
    assert!(
        visual.debug_bounds("streaming-dot").is_none(),
        "streaming state must NOT paint a second dot — one dot pulses in place",
    );
}

#[gpui::test]
fn sidebar_new_session_content_hugs_the_left_edge(cx: &mut TestAppContext) {
    // ZETA-123 round 2, finding 3: the full-width Kit Button was
    // centering its `+` glyph and label in the middle of the sidebar
    // slot. The fix drops `.w_full()` and left-anchors the button in
    // its slot. Guard: the button's LEFT edge lands inside the outer
    // slot padding (SIDEBAR_ROW_PADDING_X), not in the slot's center.
    let (window, _view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let slot = visual
        .debug_bounds("sidebar-new-session")
        .expect("new-session slot renders");
    let button = visual
        .debug_bounds("new-session-button")
        .expect("new-session action button renders");
    // The button's left edge sits within a couple of pixels of the
    // slot's inner-left (slot.left + SIDEBAR_ROW_PADDING_X). A
    // regression that re-adds `.w_full()` or `.justify_center()` on
    // the slot puts the button center at slot.center().x — the button
    // left edge would be roughly slot.left + (slot.width - button.width)/2,
    // far to the right of the padding line.
    let inner_left = slot.left() + theme::SIDEBAR_ROW_PADDING_X;
    let left_gap = if button.left() >= inner_left {
        button.left() - inner_left
    } else {
        inner_left - button.left()
    };
    assert!(
        left_gap <= px(4.),
        "new-session button left edge {:?} must sit near the slot's left padding {:?} (slot {:?})",
        button.left(),
        inner_left,
        slot,
    );
    // Sanity: the button width must NOT span the whole slot minus
    // padding — that would mean w_full is back and the button still
    // centers its content internally.
    let full_width_span = slot.size.width - theme::SIDEBAR_ROW_PADDING_X * 2.0;
    assert!(
        button.size.width < full_width_span - px(4.),
        "new-session button width {:?} spans the full slot — content still centered",
        button.size.width,
    );
}

#[gpui::test]
fn sidebar_new_session_reads_as_an_action_button(cx: &mut TestAppContext) {
    // ZETA-123: the top of the sidebar exposes a "New session" ACTION
    // — a ghost button with a `+` glyph, not a large centered heading.
    // Guards the button-affordance shape so a regression that drops the
    // icon or reverts it to a plain label surfaces here.
    let (window, _view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let button = visual
        .debug_bounds("new-session-button")
        .expect("new-session action button renders");
    let slot = visual
        .debug_bounds("sidebar-new-session")
        .expect("new-session slot renders");
    assert!(
        slot.contains(&button.center()),
        "the new-session button must sit inside its sidebar slot"
    );
    // Compact ghost action — height stays within one row of the
    // sidebar rhythm, well under a modal CTA.
    assert!(
        button.size.height <= theme::SIDEBAR_ROW_HEIGHT + px(2.),
        "new-session button height {:?} must not exceed one sidebar row",
        button.size.height
    );
}

#[gpui::test]
fn sidebar_row_menu_stays_visible_when_tab_moves_focus_from_the_row_to_the_menu_button(
    cx: &mut TestAppContext,
) {
    // ZETA-123 round 3, finding 1 (second round; extended round 4).
    // The wrapper reveal originally keyed on the row's OWN focus handle.
    // Tab from the row lands on the menu button — its own tab stop —
    // and row focus goes false. Under a row-only predicate the wrapper
    // opacity returned to 0 and the focused menu button paints its
    // focus ring at alpha 0. Enter still activates a control the user
    // cannot see; WCAG 2.4.7 focus-visible.
    //
    // The fix moves the wrapper's opacity to a container-level
    // `contains_focused` check spanning both the row and the menu
    // button. This test walks the full keyboard sequence via REAL Tab
    // key events dispatched through the keymap (Root binds `tab` →
    // `focus_next`), never `window.focus(handle)` which bypasses the
    // tab-stops registry:
    //   1. Tab → the row (bounded walk; asserts a real Tab keystroke
    //      reaches the row's tracked focus handle, not just that
    //      `window.focus()` can jam focus onto it).
    //   2. Tab → the menu button (its own tab stop). Assert the
    //      button's focus ring paints as a VISIBLE quad. Under the
    //      pre-fix predicate the count would be 0 because the
    //      wrapper's `.opacity(0.)` multiplies every descendant color
    //      alpha (including the ring border) to 0.
    //   3. Enter → the dropdown popup paints (background quad at
    //      `theme.popover`) — an invisible focus ring the user can
    //      still Enter through is the exact WCAG failure this whole
    //      arc set out to fix.
    //   4. Escape → the popup dismisses (popover quads drop out) and
    //      focus restores to the trigger button.
    //   5. Shift-Tab back to the row (still inside the container —
    //      menu stays revealed via `contains_focused`).
    //   6. Shift-Tab OUT of the row+menu container → the ellipsis
    //      button paints INVISIBLY at rest, same contract as the
    //      hover-only reveal test.
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.apply_worker_message(WorkerMessage::SessionManagement(true), window, cx);
        });
        window.draw(cx).clear(cx);
    });

    let target = session().session_id;
    let row_handle = visual
        .update(|_, cx| {
            view.read(cx)
                .sidebar_row_focus
                .borrow()
                .get(&target)
                .cloned()
        })
        .expect("session row focus handle registered after render");

    // --- Step 1: Tab reaches the row. ---
    //
    // Blur first so the walk starts from the beginning of the tab
    // order — the sequence stays deterministic no matter what the
    // composer or any kit control grabbed at construction time. Then
    // walk the window's tab-stops registry via `focus_next` (what the
    // Root `tab` keybinding invokes under the hood — see gpui-component
    // `root::init` → `Tab` → `window.focus_next`). This is the same
    // machinery a real Tab keystroke drives; the point is to route
    // through the tab-stops table and NOT jam focus onto the row
    // handle directly with `window.focus(&row_handle)`, which would
    // succeed even if the row were not registered as a tab stop at
    // all — the exact hole the reviewer flagged.
    visual.update(|window, cx| {
        window.blur(cx);
        window.draw(cx).clear(cx);
    });
    let max_tab_steps = 64;
    let mut steps_to_row = None;
    for step in 0..max_tab_steps {
        visual.update(|window, cx| window.focus_next(cx));
        if visual.update(|window, _| row_handle.is_focused(window)) {
            steps_to_row = Some(step + 1);
            break;
        }
    }
    let steps_to_row = steps_to_row.expect(
        "a Tab walk must land on the sidebar row within a bounded loop \
         — proves the row focus handle is reachable from the keyboard \
         tab-stops registry, not just via `window.focus(handle)` which \
         would succeed even for handles that are not tab stops at all",
    );
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let menu = visual
        .debug_bounds("session-menu")
        .expect("session menu renders when session-management is enabled");
    let (ring, row_focus_ring_hits) = visual.update(|window, cx| {
        let ring = cx.theme().ring;
        (ring, count_visible_ring_quads(window, menu, ring))
    });
    assert_eq!(
        row_focus_ring_hits, 0,
        "with the row focused the menu button is NOT focused — no ring \
         should be painted (theme ring {ring:?}, reached row in {steps_to_row} tabs)",
    );

    // --- Step 2: Tab moves focus to the menu button. ---
    //
    // Another `focus_next` step advances focus to the ellipsis button
    // (its own tab stop right after the row inside the wrapper). The
    // button's focus ring must paint as a VISIBLE border quad on the
    // menu bounds — the mutation-sensitive assertion that failed
    // under the row-only predicate (opacity 0 → alpha-0 ring).
    visual.update(|window, cx| {
        window.focus_next(cx);
        window.draw(cx).clear(cx);
    });
    // Prove focus DID leave the row — that is the exact case the
    // row-only predicate could not see, and the case the reviewer's
    // probe (menu-button-focus visible paints = 0) caught in the bug.
    let row_still_focused = visual.update(|window, _cx| row_handle.is_focused(window));
    assert!(
        !row_still_focused,
        "advancing the tab-stops registry from the row must move focus \
         off the row — if it stays on the row the tab-order regressed \
         and the menu-button-focus case would never be exercised",
    );
    let button_focus = visual
        .update(|window, cx| window.focused(cx))
        .expect("Tab from the row must land on the menu button focus handle");
    assert_ne!(
        button_focus, row_handle,
        "focus must have advanced past the row onto the menu button",
    );
    // The failing case: menu button focused, wrapper reveal must
    // cover its focus. Look for the button's focus-ring quad landing
    // as a VISIBLE border (alpha > 0) near the menu bounds. Under the
    // old row-only predicate the ring paints with alpha 0 → zero hits.
    let menu = visual
        .debug_bounds("session-menu")
        .expect("session menu still renders after Tab");
    let menu_focus_ring_hits =
        visual.update(|window, cx| count_visible_ring_quads(window, menu, cx.theme().ring));
    assert!(
        menu_focus_ring_hits > 0,
        "menu button focus ring must paint visibly when Tab lands on it \
         — the wrapper reveal must cover focus WITHIN the row+menu \
         container, not just the row's own focus (theme ring {ring:?})",
    );

    // --- Step 3: Enter opens the popup menu. ---
    //
    // The dropdown Popover binds `enter` in its "Popover" key context
    // → Confirm → toggle_open. The popup itself renders inside a
    // deferred layer with `popover_style(cx)` — a rounded panel with
    // `background = theme.popover`. Snapshot the baseline
    // `theme.popover` quad count BEFORE opening so a persistent
    // popover-styled surface elsewhere in the chrome (tooltip layer,
    // etc.) doesn't skew the check, then assert the count strictly
    // INCREASES on open and drops back to the baseline on dismiss.
    let popover_baseline = visual.update(|window, cx| {
        let popover_bg: gpui::Background = cx.theme().popover.into();
        window
            .painted_quads()
            .into_iter()
            .filter(|quad| quad.background == popover_bg)
            .count()
    });
    visual.simulate_keystrokes("enter");
    visual.run_until_parked();
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let popover_quads_open = visual.update(|window, cx| {
        let popover_bg: gpui::Background = cx.theme().popover.into();
        window
            .painted_quads()
            .into_iter()
            .filter(|quad| quad.background == popover_bg)
            .count()
    });
    assert!(
        popover_quads_open > popover_baseline,
        "Enter on the focused menu button must open the dropdown popup \
         — the `theme.popover` quad count did not increase over the \
         baseline ({popover_baseline}). An invisible focus ring the \
         user can still Enter through is the exact WCAG 2.4.7 failure \
         this arc set out to fix.",
    );
    let popup_focus = visual
        .update(|window, cx| window.focused(cx))
        .expect("the opened popup menu must own focus");
    assert_ne!(
        popup_focus, button_focus,
        "opening the popup must transfer focus off the trigger button \
         onto the popup menu itself",
    );

    // --- Step 4: Escape dismisses the popup. ---
    //
    // Escape in "PopupMenu" context → Cancel → emit DismissEvent →
    // Popover subscribes and closes → previous focus (the button) is
    // restored. Assert both the paint AND the focus restore — a
    // dismiss that leaks either half is a regression.
    visual.simulate_keystrokes("escape");
    visual.run_until_parked();
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let popover_quads_after_dismiss = visual.update(|window, cx| {
        let popover_bg: gpui::Background = cx.theme().popover.into();
        window
            .painted_quads()
            .into_iter()
            .filter(|quad| quad.background == popover_bg)
            .count()
    });
    assert_eq!(
        popover_quads_after_dismiss, popover_baseline,
        "Escape must dismiss the popup — `theme.popover` quad count \
         must return to the pre-open baseline ({popover_baseline}, \
         got {popover_quads_after_dismiss})",
    );
    let focus_after_dismiss = visual
        .update(|window, cx| window.focused(cx))
        .expect("focus must return somewhere after Escape dismisses the popup");
    assert_eq!(
        focus_after_dismiss, button_focus,
        "dismiss must restore focus to the trigger button so the user \
         does not lose their place in the tab order",
    );

    // --- Step 5: Shift-Tab back to the row (still inside container). ---
    //
    // `focus_prev` walks the tab-stops registry backwards — Root's
    // `shift-tab` keybinding calls this same method. With focus on
    // the row (its own tab stop inside the container),
    // `contains_focused` stays true and the wrapper stays visible.
    visual.update(|window, cx| {
        window.focus_prev(cx);
        window.draw(cx).clear(cx);
    });
    assert!(
        visual.update(|window, _| row_handle.is_focused(window)),
        "Shift-Tab from the menu button must land back on the row \
         (its immediate previous tab stop inside the same container)",
    );

    // --- Step 6: Shift-Tab OUT of the row+menu container. ---
    //
    // Once nothing in the container is focused, the container-level
    // `contains_focused` predicate goes false, the wrapper opacity
    // drops to 0, and the ellipsis button must paint INVISIBLY — same
    // contract as `sidebar_row_menu_stays_hidden_until_the_row_is_hovered`.
    // A regression that swapped `contains_focused` back to the row's
    // own `is_focused` would already have failed step 2, but a
    // regression that dropped the opacity gate entirely (or leaked
    // reveal past focus) surfaces here.
    visual.update(|window, cx| {
        window.focus_prev(cx);
        window.draw(cx).clear(cx);
    });
    assert!(
        !visual.update(|window, _| row_handle.is_focused(window)),
        "second Shift-Tab must move focus off the row and out of the \
         row+menu container",
    );
    let menu = visual
        .debug_bounds("session-menu")
        .expect("session menu still exists after focus leaves the container");
    visual.update(|window, _cx| {
        let scaled = menu.scale(window.scale_factor());
        let opaque = window.painted_quads().into_iter().any(|quad| {
            let inside = quad.bounds.top() >= scaled.top() - px(1.).scale(window.scale_factor())
                && quad.bounds.bottom() <= scaled.bottom() + px(1.).scale(window.scale_factor())
                && quad.bounds.left() >= scaled.left() - px(1.).scale(window.scale_factor())
                && quad.bounds.right() <= scaled.right() + px(1.).scale(window.scale_factor());
            inside && quad.background != gpui::transparent_black().into()
        });
        assert!(
            !opaque,
            "with focus outside the row+menu container the ellipsis \
             button must paint invisibly at rest — wrapper reveal leaked past focus",
        );
    });
}

/// Count painted quads whose border reads as the theme's focus ring on
/// the menu bounds — the ring paints outside the button's own border
/// (see gpui-component `focus_ring_style`) so widen the probe rectangle
/// by a few device pixels. A quad only counts when its border alpha is
/// above zero; under `.opacity(0.)` the wrapper multiplies every
/// descendant color's alpha by 0, and the ring drops out of visible
/// paint even though the primitive is still in the scene.
fn count_visible_ring_quads(
    window: &gpui::Window,
    menu_bounds: gpui::Bounds<gpui::Pixels>,
    ring: gpui::Hsla,
) -> usize {
    let scaled = menu_bounds.scale(window.scale_factor());
    let slack = px(8.).scale(window.scale_factor());
    window
        .painted_quads()
        .into_iter()
        .filter(|quad| {
            let overlaps = quad.bounds.right() >= scaled.left() - slack
                && quad.bounds.left() <= scaled.right() + slack
                && quad.bounds.bottom() >= scaled.top() - slack
                && quad.bounds.top() <= scaled.bottom() + slack;
            let border = quad.border_color;
            overlaps
                && border.h == ring.h
                && border.s == ring.s
                && border.l == ring.l
                && border.a > 0.0
        })
        .count()
}

#[gpui::test]
fn sidebar_row_menu_stays_hidden_until_the_row_is_hovered(cx: &mut TestAppContext) {
    // ZETA-123: the per-row `...` menu clutters the sidebar when it's
    // permanently visible. It now sits at opacity 0 at rest, revealed
    // by the row's own `.group()` hover — one pointer position only
    // lights up ONE row's menu, never every row at once. A regression
    // that dropped the `opacity(0)` gate (or the group scoping) would
    // show every menu again.
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.apply_worker_message(WorkerMessage::SessionManagement(true), window, cx);
        });
        window.draw(cx).clear(cx);
    });
    let menu = visual
        .debug_bounds("session-menu")
        .expect("session menu renders when session-management is enabled");
    // The menu paints inside the sidebar column so its hit target
    // stays reachable, but must land under an opacity-0 wrapper at
    // rest — the rendered ellipsis icon must NOT paint any visible
    // foreground quad on the header/menu bounds before hover.
    visual.update(|window, _cx| {
        let scaled = menu.scale(window.scale_factor());
        let opaque_paint = window.painted_quads().into_iter().any(|quad| {
            let inside = quad.bounds.top() >= scaled.top() - px(1.).scale(window.scale_factor())
                && quad.bounds.bottom() <= scaled.bottom() + px(1.).scale(window.scale_factor())
                && quad.bounds.left() >= scaled.left() - px(1.).scale(window.scale_factor())
                && quad.bounds.right() <= scaled.right() + px(1.).scale(window.scale_factor());
            // Any non-transparent background fill drawn tightly on the
            // menu bounds fails this contract — the reveal-on-hover
            // treatment must keep the menu invisible at rest.
            inside && quad.background != gpui::transparent_black().into()
        });
        assert!(
            !opaque_paint,
            "session menu paints an opaque quad at rest — hover-reveal broke",
        );
    });
}

/// Probe view for the scrollbar guard. Renders a Kit `Scrollbar` in
/// `Always` mode inside a fixed-size viewport with an oversized scroll
/// area, so the thumb paints in the very first frame.
struct ScrollbarProbe {
    handle: gpui::ScrollHandle,
    scroll_size: gpui::Size<gpui::Pixels>,
    viewport: gpui::Size<gpui::Pixels>,
}

impl gpui::Render for ScrollbarProbe {
    fn render(
        &mut self,
        _: &mut gpui::Window,
        _: &mut gpui::Context<Self>,
    ) -> impl gpui::IntoElement {
        gpui::div()
            .relative()
            .w(self.viewport.width)
            .h(self.viewport.height)
            .child(
                gpui_kit::base::Scrollbar::vertical(&self.handle)
                    .mode(gpui_kit::base::ScrollbarMode::Always)
                    .scroll_size(self.scroll_size)
                    .viewport_from_layout(),
            )
    }
}

#[gpui::test]
fn scrollbar_thumb_lands_on_the_wiki_8px_and_text_alpha_mix(cx: &mut TestAppContext) {
    // Contract line 93: transcript scrollbar rides at 8px wide, painted
    // in text-normal color at 20% alpha (rest) / 40% alpha (hover). The
    // `base.scrollbar.with_styles(...)` block in `theme::apply` pushes
    // those tokens into `gpui-base::ScrollbarTheme` so any Scrollbar
    // consumer picks them up.
    //
    // Mutation to catch: deleting the `base.scrollbar.with_styles(...)`
    // block in `theme::apply`. Kit's Scrollbar falls back to the 6px
    // default and a foreground-derived 35% mix — both differences are
    // observable directly in `painted_quads()`.
    cx.update(init);
    let handle = gpui::ScrollHandle::new();
    let viewport = gpui::size(px(160.), px(80.));
    let scroll_size = gpui::size(px(160.), px(320.));
    let handle_for_probe = handle.clone();
    let (_view, cx) = cx.add_window_view(move |_, cx| {
        theme::apply(cx);
        ScrollbarProbe {
            handle: handle_for_probe,
            scroll_size,
            viewport,
        }
    });
    cx.update(|window, cx| window.draw(cx).clear(cx));

    // --- Rest: thumb paints at 8px in the text-normal 20% mix. ---
    let (thumb_bounds, rest_scale) = cx.update(|window, _| {
        let scale = window.scale_factor();
        let rest_bg: gpui::Background = theme::palette::scrollbar_thumb().into();
        let thumb = window
            .painted_quads()
            .into_iter()
            .find(|q| q.background == rest_bg)
            .expect("scrollbar paints its resting thumb at the 20% text mix");
        (thumb.bounds, scale)
    });
    let thumb_target = theme::SCROLLBAR_THUMB_WIDTH.scale(rest_scale);
    let width_delta = if thumb_bounds.size.width > thumb_target {
        thumb_bounds.size.width - thumb_target
    } else {
        thumb_target - thumb_bounds.size.width
    };
    assert!(
        width_delta <= px(1.).scale(rest_scale),
        "thumb width {:?} must land on the 8px contract",
        thumb_bounds.size.width
    );

    // --- Hover: same 8px width, hover mix (40% alpha). ---
    // Hover the mouse over the middle of the resting thumb bounds; Kit
    // flips `hovered_on_thumb` on the very next MouseMoveEvent and the
    // next paint reads `style_for_hovered_thumb`. Kit's mouse events
    // arrive in logical (unscaled) pixels, so undo the device scale.
    let center = thumb_bounds.center();
    let hover_pos = gpui::point(
        px(center.x.as_f32() / rest_scale),
        px(center.y.as_f32() / rest_scale),
    );
    cx.simulate_mouse_move(hover_pos, None, gpui::Modifiers::default());
    cx.update(|window, cx| window.draw(cx).clear(cx));
    cx.update(|window, _| {
        let hover_bg: gpui::Background = theme::palette::scrollbar_thumb_hover().into();
        let hover_thumb = window
            .painted_quads()
            .into_iter()
            .find(|q| q.background == hover_bg)
            .expect("scrollbar paints its hovered thumb at the 40% text mix");
        let scale = window.scale_factor();
        let hover_delta = if hover_thumb.bounds.size.width > thumb_target {
            hover_thumb.bounds.size.width - thumb_target
        } else {
            thumb_target - hover_thumb.bounds.size.width
        };
        assert!(
            hover_delta <= px(1.).scale(scale),
            "hovered thumb width {:?} must also land on the 8px contract",
            hover_thumb.bounds.size.width
        );
    });
}

#[gpui::test]
fn sidebar_row_focus_handles_are_real_tab_stops_reached_via_focus_next(cx: &mut TestAppContext) {
    // Round-4 a11y guard. `cx.focus_handle()` defaults to `tab_stop=false`,
    // and the div's `.tab_index(0)` does NOT propagate to a tracked focus
    // handle. Without setting `tab_stop(true)` on the handle itself,
    // `window.focus_next` walks past every sidebar row. The prior row-3
    // guard only exercised `window.focus(&handle)` directly, so a missing
    // tab-stop flag never showed. This test uses `focus_next` and MUST
    // fail if the handle is not registered as a tab stop.
    let (window, view, _receiver) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    let session_target = "cd34beef1234";
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            let mut other = session();
            other.session_id = session_target.into();
            other.name = "other".into();
            view.state.sessions.push(other);
            view.state.session_view.available = true;
            view.state.session_view.branches = vec![
                Branch {
                    id: "trunk".into(),
                    label: "main".into(),
                    depth: 0,
                    current: true,
                },
                Branch {
                    id: "alt".into(),
                    label: "alt".into(),
                    depth: 1,
                    current: false,
                },
            ];
            view.apply_worker_message(WorkerMessage::Connected, window, cx);
        });
        window.draw(cx).clear(cx);
    });

    let session_handle = visual
        .update(|_, cx| {
            view.read(cx)
                .sidebar_row_focus
                .borrow()
                .get(session_target)
                .cloned()
        })
        .expect("sidebar row focus handle exists after render");
    let branch_handle = visual
        .update(|_, cx| {
            view.read(cx)
                .sidebar_row_focus
                .borrow()
                .get("branch:alt")
                .cloned()
        })
        .expect("branch row focus handle exists after render");
    // Blur so `focus_next` starts from the beginning of the tab order —
    // sequence becomes deterministic regardless of what the composer or
    // any kit control grabbed at construction time.
    visual.update(|window, cx| window.blur(cx));

    // Bound the walk. `focus_next` wraps around, so we cap at a very
    // generous ceiling to defend against a runaway loop while still
    // proving reachability.
    let max_steps = 512;
    let mut saw_session_at = None;
    let mut saw_branch_at = None;
    for step in 0..max_steps {
        visual.update(|window, cx| window.focus_next(cx));
        let focused = visual.update(|window, cx| window.focused(cx));
        if focused.as_ref() == Some(&session_handle) && saw_session_at.is_none() {
            saw_session_at = Some(step);
        }
        if focused.as_ref() == Some(&branch_handle) && saw_branch_at.is_none() {
            saw_branch_at = Some(step);
        }
        if saw_session_at.is_some() && saw_branch_at.is_some() {
            break;
        }
    }

    let session_step = saw_session_at.expect(
        "focus_next must land on a session row — proves the row focus handle \
         is registered as a real tab stop, not just a tracked handle with \
         tab_stop=false",
    );
    let branch_step = saw_branch_at.expect(
        "focus_next must land on a branch row — proves the branch row focus \
         handle is registered as a real tab stop",
    );
    assert!(
        session_step < branch_step,
        "session rows paint before branch rows and must be reached first via \
         focus_next (session at step {session_step}, branch at step {branch_step})"
    );
}

#[gpui::test]
fn sidebar_row_focus_map_prunes_removed_rows_and_keeps_survivors(cx: &mut TestAppContext) {
    // Round-4 leak guard. Every new branch head mints a fresh UUID, so an
    // insert-only handle map grows for the app's lifetime. `render_sidebar`
    // prunes to the live key set on every paint. This test renders once
    // with a set of rows, mutates the state (drop one session, replace
    // one branch head, keep one of each), renders again, and asserts:
    //   1. stale keys are gone from `sidebar_row_focus`,
    //   2. surviving keys retain the SAME handle instance so tab focus
    //      does not drift across redraws.
    let (window, view, _receiver) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    let keep_session = "cd34beef1234";
    let drop_session = "aa11aa11aa11";
    let keep_branch = "trunk";
    let old_branch = "old-branch-head";
    let new_branch = "new-branch-head";
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            let mut a = session();
            a.session_id = keep_session.into();
            let mut b = session();
            b.session_id = drop_session.into();
            view.state.sessions.push(a);
            view.state.sessions.push(b);
            view.state.session_view.available = true;
            view.state.session_view.branches = vec![
                Branch {
                    id: keep_branch.into(),
                    label: "main".into(),
                    depth: 0,
                    current: true,
                },
                Branch {
                    id: old_branch.into(),
                    label: "old".into(),
                    depth: 1,
                    current: false,
                },
            ];
            view.apply_worker_message(WorkerMessage::Connected, window, cx);
        });
        window.draw(cx).clear(cx);
    });

    let (keep_session_before, keep_branch_before, drop_session_before, old_branch_before) = visual
        .update(|_, cx| {
            let map = view.read(cx).sidebar_row_focus.borrow();
            (
                map.get(keep_session).cloned(),
                map.get(&format!("branch:{keep_branch}")).cloned(),
                map.get(drop_session).cloned(),
                map.get(&format!("branch:{old_branch}")).cloned(),
            )
        });
    assert!(
        keep_session_before.is_some() && drop_session_before.is_some(),
        "both session rows must register a focus handle on first render"
    );
    assert!(
        keep_branch_before.is_some() && old_branch_before.is_some(),
        "both branch rows must register a focus handle on first render"
    );

    // Mutate: drop the second session, replace the second branch head
    // with a fresh UUID (the real bug — branch heads churn UUIDs on
    // every turn). Keep the first of each.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state
                .sessions
                .retain(|s| s.session_id.as_str() == keep_session);
            view.state.session_view.branches = vec![
                Branch {
                    id: keep_branch.into(),
                    label: "main".into(),
                    depth: 0,
                    current: true,
                },
                Branch {
                    id: new_branch.into(),
                    label: "new".into(),
                    depth: 1,
                    current: false,
                },
            ];
            view.apply_worker_message(WorkerMessage::Connected, window, cx);
        });
        window.draw(cx).clear(cx);
    });

    visual.update(|_, cx| {
        let map = view.read(cx).sidebar_row_focus.borrow();
        assert!(
            !map.contains_key(drop_session),
            "dropped session must be pruned from the focus map"
        );
        assert!(
            !map.contains_key(&format!("branch:{old_branch}")),
            "replaced branch head must be pruned from the focus map"
        );
        assert!(
            map.contains_key(keep_session),
            "surviving session must remain in the focus map"
        );
        assert!(
            map.contains_key(&format!("branch:{keep_branch}")),
            "surviving branch must remain in the focus map"
        );
        assert!(
            map.contains_key(&format!("branch:{new_branch}")),
            "new branch head must be registered on the second render"
        );
        assert_eq!(
            map.get(keep_session),
            keep_session_before.as_ref(),
            "surviving session must keep the same focus handle across redraws"
        );
        assert_eq!(
            map.get(&format!("branch:{keep_branch}")),
            keep_branch_before.as_ref(),
            "surviving branch must keep the same focus handle across redraws"
        );
    });
}

/// Renderer-literal fence — the ZETA-109 AST-based static guard.
///
/// The typed `row_text::RowText` / `LoginRowText` model is the sole source
/// of every user-visible string a transcript row paints. This fence proves
/// it stays that way by parsing the WHOLE `transcript_render.rs` module
/// with `syn` on every run and rejecting every string, byte-string, or
/// C-string literal in expression position — no ambient allowance from any
/// method-call subtree. The only literals that pass are the ones carried
/// by an allowlisted macro payload.
///
/// AST context, not string shape, is what distinguishes an ID from
/// visible text — that closes the r1 review's bypasses:
///
///   * `.child("[done]")` — literal outside allowed subtree
///   * `.child("done")`  — lowercase-safe shape does NOT save it
///   * `Alert::error(..., "hardcoded")` — plain string second arg
///   * `format!("hello {n}")` — bare format outside allowed subtree
///   * `format!("tool-verb-{i}")` — ID-shaped format still flagged
///   * `String::from_utf8_lossy(b"…")` — byte strings are flagged too
///   * `c"leaked"` — C-string literals are flagged too
///   * `stringify!(leaked)` / `concat!("a", "b")` — forbidden macros
///     (r3 finding — accidental non-`format!` string builders)
///   * a new `render_login_row` helper hiding text — the WHOLE module is
///     scanned, not a fixed six-fn allowlist, so renaming or splitting
///     renderers cannot smuggle a literal past the fence.
///
/// Allowed macro payloads (every other literal is rejected):
///
/// * Diagnostic macros — `panic!`, `unreachable!`, `todo!`,
///   `unimplemented!`, `assert{,_eq,_ne}!`, `debug_assert{,_eq,_ne}!` —
///   whose payloads never reach the user.
/// * `matches!` — the ONE pattern-only macro the render module uses;
///   its payload is a pattern, never visible text.
/// * `format!` — literal fragments in the payload are still checked
///   against the (always-zero) ambient depth, so any literal there
///   still trips. `format!` calls in the render module are rejected
///   because there is no legitimate use — widget-ID composition lives
///   in `row_text::sel::*` and returns a `String` back to the module.
///
/// Every OTHER macro (`stringify!`, `concat!`, `write!`, `println!`,
/// arbitrary imported macros) is rejected outright — the module has no
/// legitimate use for them.
///
/// The `chrome`-coverage arm parses `row_text.rs` for the `pub mod chrome`
/// submodule and proves (a) every `pub const NAME: &str = "…"` sits at
/// exactly `pub` visibility (no `pub(super)`/`pub(crate)`/private bypass)
/// and (b) is a member of `chrome::ALL`. Adding a new chrome constant
/// without listing it in `ALL` silently escapes the seam sweep — this
/// test flags that regression.
#[test]
fn renderer_literal_fence_rejects_literals_outside_allowed_contexts() {
    // r4 finding 6: the tool-receipt and group renderers moved to
    // `tool_receipts.rs`. The fence's scanned set MUST include the
    // extracted module or a stray literal there slips past the ZETA-109
    // guard silently. Both files ride the same fence rules.
    const SOURCES: &[(&str, &str)] = &[
        ("transcript_render.rs", include_str!("transcript_render.rs")),
        ("tool_receipts.rs", include_str!("tool_receipts.rs")),
    ];
    let mut failures: Vec<String> = Vec::new();
    for (name, source) in SOURCES {
        for failure in fence::run(source) {
            failures.push(format!("{name}: {failure}"));
        }
    }
    assert!(
        failures.is_empty(),
        "renderer_literal_fence tripped on the render source set:\n  - {}",
        failures.join("\n  - "),
    );
}

#[test]
fn renderer_literal_fence_ast_visitor_flags_the_probe_bypasses() {
    // Mutation battery — the six probes the r1 review named MUST all
    // trip. Runs the same AST visitor over synthetic module snippets so
    // the guard's guard fires on every push. If any probe stops
    // failing, the fence has weakened and the review's finding is
    // silently back.
    let probes: &[(&str, &str)] = &[
        (
            "bracketed state marker",
            r#"impl X { fn f(&self) -> D { div().child("[done]") } }"#,
        ),
        (
            "lowercase prose reaches child",
            r#"impl X { fn f(&self) -> D { div().child("done") } }"#,
        ),
        (
            "prose format! fragment outside allowed context",
            r#"impl X { fn f(&self, n: usize) -> D { div().child(format!("hello {n}")) } }"#,
        ),
        (
            "ID-shaped format! fragment outside allowed context",
            r#"impl X { fn f(&self, i: usize) -> D { div().child(format!("tool-verb-{i}")) } }"#,
        ),
        (
            "byte-string literal bypasses via from_utf8_lossy",
            r#"impl X { fn f(&self) -> D { div().child(String::from_utf8_lossy(b"leaked").to_string()) } }"#,
        ),
        (
            "renamed/new helper fn in the module still scanned",
            r#"impl X { fn f(&self) -> D { self.helper() } fn helper(&self) -> D { div().child("leaked-via-helper") } }"#,
        ),
        (
            "C-string literal reaches child",
            r#"impl X { fn f(&self) -> D { div().child(c"leaked".to_str().unwrap()) } }"#,
        ),
        (
            "stringify! macro assembles a leaked string",
            r#"impl X { fn f(&self) -> D { div().child(stringify!(LEAKED_IDENT)) } }"#,
        ),
        (
            "concat! macro joins literal fragments",
            r#"impl X { fn f(&self) -> D { div().child(concat!("a", "-", "b")) } }"#,
        ),
        (
            "write! macro (imports unlisted machinery)",
            r#"impl X { fn f(&self, out: &mut String) { let _ = write!(out, "hi"); } }"#,
        ),
        (
            "debug_selector method allowance is gone — bare literal still trips",
            r#"impl X { fn f(&self) -> D { div().debug_selector(|| "transcript-row".into()) } }"#,
        ),
        (
            "id method allowance is gone — bare literal still trips",
            r#"impl X { fn f(&self, i: usize) -> D { div().id(("tool-receipt", i)) } }"#,
        ),
        (
            "aria_label method allowance is gone — bare literal still trips",
            r#"impl X { fn f(&self) -> D { div().aria_label("dialog") } }"#,
        ),
    ];
    for (label, probe) in probes {
        let failures = fence::run(probe);
        assert!(
            !failures.is_empty(),
            "mutation probe MUST trip the fence — {label}\nsnippet: {probe}\ngot failures: {failures:?}"
        );
    }
}

#[test]
fn renderer_literal_fence_accepts_the_legitimate_shapes() {
    // Positive fixtures — every allowed usage the render module actually
    // emits. If ANY of these starts failing, the fence has become too
    // strict and legitimate render code cannot compile.
    //
    // r3 fence: the method-name allowance for `debug_selector`/`id`/
    // `aria_label`/`role` is gone. Every selector call in the module
    // takes a const path or a `sel::*` helper's `String`, so no bare
    // literal ever sits inside those args. Every positive fixture here
    // is literal-free outside diagnostic-macro payloads.
    let positives: &[&str] = &[
        // Selector const routed through .debug_selector — no literal.
        r#"impl X { fn f(&self) -> D { div().debug_selector(|| sel::TRANSCRIPT_ROW.into()) } }"#,
        // Selector helper returning a String routed through .debug_selector.
        r#"impl X { fn f(&self, i: usize) -> D { div().debug_selector(move || sel::tool_verb(i)) } }"#,
        // Selector const paired with an index tuple — no literal.
        r#"impl X { fn f(&self, i: usize) -> D { div().id((sel::TOOL_RECEIPT_TAG, i)) } }"#,
        // Owned selector ID routed through .id.
        r#"impl X { fn f(&self, id: String) -> D { div().id(id) } }"#,
        // Diagnostic macro escape — `unreachable!` is the only diagnostic
        // the round-4-tightened allowlist keeps.
        r#"impl X { fn f(&self) { unreachable!("row-inner-entry-mismatch") } }"#,
        // matches! is a pattern-only macro the module legitimately uses.
        r#"impl X { fn f(&self, e: &E) -> bool { matches!(e, E::Tool { .. }) } }"#,
        // Passing a chrome-const path through .child — no literal.
        r#"impl X { fn f(&self) -> D { div().child(row_text::chrome::FORK_HERE) } }"#,
    ];
    for probe in positives {
        let failures = fence::run(probe);
        assert!(
            failures.is_empty(),
            "positive fixture MUST pass the fence:\nsnippet: {probe}\ngot failures: {failures:?}",
        );
    }
}

#[test]
fn renderer_literal_fence_mutation_battery_against_the_real_module() {
    // Round-2 mutation battery — each entry names a review-attested
    // bypass and mutates the ACTUAL `transcript_render.rs` source. The
    // fence MUST trip on every one. This is the same set the orchestrator
    // re-runs on the final head; if any stops failing, the guard has
    // weakened and the review's finding is silently back.
    const SOURCE: &str = include_str!("transcript_render.rs");
    let mutations: &[(&str, &str, &str)] = &[
        // (label, injection point — matched verbatim, mutated snippet)
        (
            "child bracketed state marker",
            ".child(header)\n            .into_any_element()\n    }",
            ".child(\"[done]\")\n            .child(header)\n            .into_any_element()\n    }",
        ),
        (
            "child lowercase prose",
            ".child(header)\n            .into_any_element()\n    }",
            ".child(\"done\")\n            .child(header)\n            .into_any_element()\n    }",
        ),
        (
            "rename render_thinking_row",
            "fn render_thinking_row(&self",
            "fn render_thinking_pane(&self",
        ),
        (
            "text moved into a fresh helper fn in the module",
            "impl ZetaView {\n    pub(crate) fn render_row(",
            "impl ZetaView {\n    fn newly_added_helper(&self) -> &str { \"leaked-via-helper\" }\n    pub(crate) fn render_row(",
        ),
        (
            "prose format! fragment outside allowed context",
            ".child(header)",
            ".child(format!(\"You have {} messages\", 3)).child(header)",
        ),
        (
            "ID-shaped format! fragment outside allowed context",
            ".child(header)",
            ".child(format!(\"tool-verb-{}\", 3)).child(header)",
        ),
        (
            "c-string literal leaks into a child slot",
            ".child(header)",
            ".child(c\"leaked\".to_str().unwrap()).child(header)",
        ),
        (
            "stringify! macro leaks into a child slot",
            ".child(header)",
            ".child(stringify!(LEAKED_IDENT)).child(header)",
        ),
        (
            "concat! macro leaks into a child slot",
            ".child(header)",
            ".child(concat!(\"a\", \"-\", \"b\")).child(header)",
        ),
    ];
    let mut ran = 0usize;
    for (label, needle, replacement) in mutations {
        assert!(
            SOURCE.contains(needle),
            "mutation battery: injection anchor {needle:?} not found for probe {label:?}"
        );
        let mutated = SOURCE.replacen(needle, replacement, 1);
        let failures = fence::run(&mutated);
        // The `rename` mutation does NOT introduce a new literal; the fence
        // MUST NOT trip for it. Every other mutation SHOULD trip.
        let expect_trip = *label != "rename render_thinking_row";
        if expect_trip {
            assert!(
                !failures.is_empty(),
                "mutation MUST trip the fence — {label}\ngot failures: {failures:?}"
            );
        } else {
            assert!(
                failures.is_empty(),
                "rename mutation MUST NOT trip the fence (fence scans WHOLE module, \
                 not a fn-name list) — got failures: {failures:?}"
            );
        }
        ran += 1;
    }
    assert_eq!(ran, 9, "battery must exercise every review-named probe");
}

#[test]
fn renderer_literal_fence_scans_the_extracted_tool_receipts_module() {
    // r4 finding 6: the tool-receipt and group renderers moved to
    // `tool_receipts.rs`. If the fence's scanned set does not include
    // the new module, a stray literal there silently regresses the
    // ZETA-109 guard. This mutation proves the extracted module is
    // scanned: injecting a `.child("x")` literal into tool_receipts.rs
    // MUST trip the fence — the same rule the parent module enforces.
    const SOURCE: &str = include_str!("tool_receipts.rs");
    // Anchor on a stable render-time call the module actually emits so
    // this test does not go stale on unrelated refactors of the tool
    // renderers.
    let needle = ".into_any_element()\n    }";
    assert!(
        SOURCE.contains(needle),
        "mutation anchor {needle:?} not found in tool_receipts.rs — the \
         mutation battery has drifted from the module's actual shape",
    );
    let mutated = SOURCE.replacen(needle, ".child(\"x\").into_any_element()\n    }", 1);
    let failures = fence::run(&mutated);
    assert!(
        !failures.is_empty(),
        "injecting `.child(\"x\")` into tool_receipts.rs MUST trip the fence — \
         got failures: {failures:?}. The fence's scanned set is not covering \
         the extracted module.",
    );
    // Also prove the unmodified tool_receipts.rs is CLEAN so a real fence
    // trip would not blend into background failures.
    let baseline = fence::run(SOURCE);
    assert!(
        baseline.is_empty(),
        "tool_receipts.rs must pass the fence unmodified — baseline failures: {baseline:?}",
    );
}

#[test]
fn renderer_literal_fence_chrome_consts_are_public_and_listed_in_all() {
    // Parse row_text.rs's `pub mod chrome` and check:
    //   (a) every `pub const NAME: &str = "…"` sits at exactly `pub`
    //       visibility — `pub(super)` / `pub(crate)` / private are the
    //       exact bypass this test names,
    //   (b) every declared constant appears in `chrome::ALL`, so the
    //       seam-sweep tests that iterate `ALL` pick it up.
    const SOURCE: &str = include_str!("row_text.rs");
    let (declared, all) = fence::chrome_summary(SOURCE);
    assert!(
        !declared.is_empty(),
        "fence: chrome module scan returned zero constants — parser regressed?"
    );
    for name in &declared {
        assert!(
            all.contains(name),
            "chrome::{name} is declared but missing from chrome::ALL — \
             add it so the seam sweep and the fence pick it up"
        );
    }
}

mod fence {
    //! AST fence internals — a `syn`-based visitor over a Rust module's
    //! source. Kept in a submodule so the tests read cleanly and the
    //! visitor stays testable on its own.

    use syn::visit::{self, Visit};

    /// Diagnostic macros whose payloads never reach the user. Their
    /// token stream is scanned inside an allowed subtree so any literal
    /// payload passes. Round-4 tightening: reduced to exactly what the
    /// render module uses (`unreachable!`). Every other diagnostic macro
    /// (`panic!`, `todo!`, `unimplemented!`, `assert*!`, `debug_assert*!`)
    /// is default-deny — adding one to the module trips the fence.
    const DIAGNOSTIC_MACROS: &[&str] = &["unreachable"];

    /// Pattern-only macros the render module legitimately uses. Their
    /// payload carries no visible text.
    const PATTERN_MACROS: &[&str] = &["matches"];

    /// `format!` is scanned WITHOUT bumping the allowed subtree — any
    /// literal fragment in its payload is checked against the ambient
    /// depth (which is always zero now that the method-name allowance
    /// is gone) and trips the fence.
    const FORMAT_MACRO: &str = "format";

    /// Parse `source` and return every literal the fence flags. Empty
    /// vector means the source is clean.
    pub(super) fn run(source: &str) -> Vec<String> {
        let file = match syn::parse_file(source) {
            Ok(file) => file,
            Err(err) => return vec![format!("fence: parse error: {err}")],
        };
        let mut visitor = Visitor::default();
        visit::visit_file(&mut visitor, &file);
        visitor.failures
    }

    /// Extract `(declared_names, all_names)` from `row_text.rs`'s
    /// `pub mod chrome { ... }` submodule. Panics on unexpected shape so
    /// the fence stays authoritative on chrome layout.
    pub(super) fn chrome_summary(source: &str) -> (Vec<String>, Vec<String>) {
        let file = syn::parse_file(source).expect("parse row_text.rs");
        let chrome = file
            .items
            .iter()
            .find_map(|item| match item {
                syn::Item::Mod(m) if m.ident == "chrome" => Some(m),
                _ => None,
            })
            .expect("row_text.rs must declare `pub mod chrome`");
        assert!(
            matches!(chrome.vis, syn::Visibility::Public(_)),
            "chrome module must be `pub`"
        );
        let content = &chrome
            .content
            .as_ref()
            .expect("chrome module must be inline")
            .1;
        let mut declared = Vec::new();
        let mut all = Vec::new();
        for item in content {
            let syn::Item::Const(c) = item else { continue };
            let name = c.ident.to_string();
            if name == "ALL" {
                if let syn::Expr::Reference(refexpr) = c.expr.as_ref() {
                    if let syn::Expr::Array(arr) = refexpr.expr.as_ref() {
                        for elem in &arr.elems {
                            if let syn::Expr::Path(p) = elem {
                                let last = p
                                    .path
                                    .segments
                                    .last()
                                    .expect("chrome::ALL entry has at least one segment");
                                all.push(last.ident.to_string());
                            }
                        }
                    }
                }
                continue;
            }
            // Only inspect &str consts (the visible-string set). Other
            // shapes (e.g. `pub const ALL: &[&str]`) are handled above.
            if !is_str_ref_type(&c.ty) {
                continue;
            }
            assert!(
                matches!(c.vis, syn::Visibility::Public(_)),
                "chrome::{name} must be `pub` (no `pub(super)` / `pub(crate)` / private) — \
                 the visibility bypass fails the fence."
            );
            declared.push(name);
        }
        (declared, all)
    }

    fn is_str_ref_type(ty: &syn::Type) -> bool {
        let syn::Type::Reference(r) = ty else {
            return false;
        };
        let syn::Type::Path(p) = r.elem.as_ref() else {
            return false;
        };
        p.path.is_ident("str")
    }

    #[derive(Default)]
    struct Visitor {
        failures: Vec<String>,
        // Non-zero while the visitor is inside an allowed method-call
        // argument subtree OR inside a diagnostic macro's token stream.
        allowed_depth: usize,
    }

    impl<'ast> Visit<'ast> for Visitor {
        fn visit_expr_lit(&mut self, node: &'ast syn::ExprLit) {
            self.check_lit(&node.lit);
            visit::visit_expr_lit(self, node);
        }

        fn visit_lit(&mut self, lit: &'ast syn::Lit) {
            // Also called for patterns and other non-Expr contexts. The
            // depth counter still governs — anything outside allowed
            // scope is flagged.
            self.check_lit(lit);
        }

        fn visit_macro(&mut self, node: &'ast syn::Macro) {
            let name = node
                .path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default();
            if DIAGNOSTIC_MACROS.iter().any(|d| *d == name)
                || PATTERN_MACROS.iter().any(|p| *p == name)
            {
                // Diagnostic payloads never surface. Pattern macros
                // (`matches!`) carry patterns, not visible text. Both
                // ride an allowed subtree so any literal token passes.
                self.allowed_depth += 1;
                self.scan_tokens(node.tokens.clone());
                self.allowed_depth -= 1;
            } else if name == FORMAT_MACRO {
                // `format!` is scanned at ambient depth (always 0 now).
                // Any literal fragment in its payload trips the fence.
                self.scan_tokens(node.tokens.clone());
            } else {
                self.failures.push(format!(
                    "forbidden macro `{name}!` in the render module — the fence \
                     allowlists only `unreachable!`, the pattern-only `matches!`, \
                     and `format!` (whose literal fragments are still checked). \
                     Every other macro (`panic!`, `todo!`, `unimplemented!`, \
                     `assert*!`, `debug_assert*!`, `stringify!`, `concat!`, \
                     `write!`, unknown/imported macros) is rejected — route \
                     every visible string through the RowText / LoginRowText \
                     model or `row_text::chrome`."
                ));
                self.scan_tokens(node.tokens.clone());
            }
        }

        fn visit_attribute(&mut self, _node: &'ast syn::Attribute) {
            // Skip attribute contents entirely — `#[doc = "..."]`,
            // `#[cfg_attr(..., allow_unused_variables)]`, `#[cfg(...)]`
            // literals never reach the render surface.
        }

        fn visit_item_const(&mut self, node: &'ast syn::ItemConst) {
            // Const items in the render module MUST NOT declare visible
            // strings — chrome consts live in row_text.rs. A `const FOO:
            // &str = "..."` in transcript_render.rs is a bypass.
            let is_str_const = is_str_ref_type(&node.ty);
            if is_str_const {
                self.failures.push(format!(
                    "forbidden `const {}: &str = ...` in the render module — \
                     move it to `row_text::chrome` so the seam-sweep tests \
                     iterate it too.",
                    node.ident
                ));
            }
            visit::visit_item_const(self, node);
        }
    }

    impl Visitor {
        fn check_lit(&mut self, lit: &syn::Lit) {
            let content = match lit {
                syn::Lit::Str(s) => Some(("str", s.value())),
                syn::Lit::ByteStr(b) => {
                    Some(("byte-str", String::from_utf8_lossy(&b.value()).into_owned()))
                }
                syn::Lit::CStr(c) => Some(("c-str", c.value().to_string_lossy().into_owned())),
                _ => None,
            };
            if let Some((kind, text)) = content {
                if self.allowed_depth == 0 {
                    self.failures.push(format!(
                        "forbidden {kind} literal {text:?} — route this string through \
                         the RowText / LoginRowText model, the row_text::chrome constants, \
                         or a selector helper (row_text::sel::*). The only allowed literal \
                         context is a diagnostic / matches! macro payload."
                    ));
                }
            }
        }

        fn scan_tokens(&mut self, ts: proc_macro2::TokenStream) {
            for tt in ts {
                match tt {
                    proc_macro2::TokenTree::Group(g) => self.scan_tokens(g.stream()),
                    proc_macro2::TokenTree::Literal(l) => {
                        let text = l.to_string();
                        let bytes = text.as_bytes();
                        let is_str = matches!(bytes.first(), Some(b'"'))
                            || bytes.starts_with(b"r\"")
                            || bytes.starts_with(b"r#");
                        let is_byte = bytes.starts_with(b"b\"")
                            || bytes.starts_with(b"br\"")
                            || bytes.starts_with(b"br#");
                        let is_cstr = bytes.starts_with(b"c\"")
                            || bytes.starts_with(b"cr\"")
                            || bytes.starts_with(b"cr#");
                        if (is_str || is_byte || is_cstr) && self.allowed_depth == 0 {
                            self.failures.push(format!(
                                "forbidden macro-token literal {text} — route this string \
                                 through the RowText / LoginRowText model. The only allowed \
                                 literal context is a diagnostic / matches! macro payload."
                            ));
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ZETA-124: Type-scale role guard, prose-measure cap, and the wedge-list
// wrap regression. Every text site outside `theme.rs` and the smoke shot
// driver must route through a NAMED role — a raw `.text_size(px(...))`
// fails the guard. The measure cap keeps assistant prose readable at every
// picker size, and the wedge regression pins the exact list content whose
// wrap layout dropped an orphan character in the ZETA-124 critique shot.
// ---------------------------------------------------------------------------

/// Role sizes ride an ordered ladder (title > body > label > label_small >=
/// label_micro) at every base the appearance picker exposes. At the picker's
/// MIN base the two smallest roles both land on `MIN_LABEL_PX` (they clip
/// against the legibility floor), so the ladder relaxes to non-strict below
/// `label`; at the picker's DEFAULT and MAX the ladder is strict all the
/// way down — a refactor that flattens `label` onto `body`, or shifts
/// `label_small` above `label`, fails here.
#[test]
fn role_scale_lands_on_an_ordered_ladder() {
    for base_px in [
        theme::MIN_FONT_SIZE_PX as i32,
        f32::from(theme::DEFAULT_FONT_SIZE) as i32,
        theme::MAX_FONT_SIZE_PX as i32,
    ] {
        let base = px(base_px as f32);
        let title = theme::title(base);
        let body = theme::body(base);
        let label = theme::label(base);
        let small = theme::label_small(base);
        let micro = theme::label_micro(base);
        assert!(
            f32::from(title) > f32::from(body),
            "title {title:?} must sit above body {body:?} at base {base:?}"
        );
        assert_eq!(body, base, "body role must equal base at every picker step");
        assert!(
            f32::from(label) < f32::from(body),
            "label {label:?} must sit below body {body:?}"
        );
        assert!(
            f32::from(small) <= f32::from(label),
            "label_small {small:?} must sit at or below label {label:?}"
        );
        assert!(
            f32::from(micro) <= f32::from(small),
            "label_micro {micro:?} must sit at or below label_small {small:?}"
        );
        // Floors: even at the picker's MIN, the smallest role stays >= the
        // legibility floor so a shrink to 11px does not vanish micro chips.
        assert!(f32::from(micro) >= theme::MIN_LABEL_PX);
    }
    // At the shipped default the ladder is strictly ordered — a refactor
    // that lost the +2 title step or the -1 label step fails here even
    // when the floor hides the collapse at the MIN base.
    let base = theme::DEFAULT_FONT_SIZE;
    assert!(f32::from(theme::title(base)) > f32::from(theme::body(base)));
    assert!(f32::from(theme::body(base)) > f32::from(theme::label(base)));
    assert!(f32::from(theme::label(base)) > f32::from(theme::label_small(base)));
    assert!(f32::from(theme::label_small(base)) > f32::from(theme::label_micro(base)));
}

/// Prose measure caps assistant reading rows at ~88ch of the base font,
/// scaling with the picker: an 18px reader keeps a wider column than an
/// 11px reader, but both stay narrower than `TRANSCRIPT_MAX_WIDTH`.
#[test]
fn prose_max_width_scales_with_the_appearance_picker() {
    let low = theme::prose_max_width(px(theme::MIN_FONT_SIZE_PX));
    let high = theme::prose_max_width(px(theme::MAX_FONT_SIZE_PX));
    assert!(f32::from(low) < f32::from(high));
    assert!(
        f32::from(high) < f32::from(theme::TRANSCRIPT_MAX_WIDTH),
        "prose cap must sit BELOW the wide TRANSCRIPT_MAX_WIDTH even at MAX \
         picker size — otherwise the reading measure is a no-op",
    );
    // Approx guard: at the shipped default (13px) the measure lands in a
    // 500-800px window — a scannable ~88ch column. A regression that
    // dropped the multiplier past 0.5 or above 0.75 fails here.
    let default = f32::from(theme::prose_max_width(theme::DEFAULT_FONT_SIZE));
    assert!(
        (500.0..=800.0).contains(&default),
        "prose max width {default} at default base drifted outside the \
         scannable 90ch band"
    );
}

/// Every `.text_size(...)` call in the run-UI source (main.rs, sidebar.rs,
/// transcript_render.rs, session_management.rs, polish.rs) must feed one
/// of the FIVE verified type roles — `theme::title(`, `theme::body(`,
/// `theme::label(`, `theme::label_small(`, `theme::label_micro(`. Anything
/// else — a raw `px(N.)`, an ambient alias like `label_size` / `meta_size`
/// / `small` / `fallback_size` / `base_size`, a leaked `current_font_size(`,
/// or a `prose_max_width(` fed as a font size — fails the sweep.
///
/// The allowlist deliberately does NOT include `theme::current_font_size(`
/// (r2 finding 2: the picker's raw pixel value is not a role — sites that
/// need the "normal reading text" tier must route through `theme::body`)
/// or `theme::prose_max_width(` (a WIDTH cap in pixels; feeding a width
/// into `.text_size(...)` was the r2 finding 2 loophole where an unrelated
/// pixel value looked like a valid role call).
#[test]
fn every_text_size_call_routes_through_a_named_theme_role() {
    let sources: &[(&str, &str)] = &[
        ("main.rs", include_str!("main.rs")),
        ("sidebar.rs", include_str!("sidebar.rs")),
        ("transcript_render.rs", include_str!("transcript_render.rs")),
        (
            "session_management.rs",
            include_str!("session_management.rs"),
        ),
        ("polish.rs", include_str!("polish.rs")),
    ];
    // ONLY the five type roles are accepted. `theme::body(theme::current_font_size())`
    // still starts with `theme::body(` so ambient-context sites (no `cx` in
    // scope) can route through `body` explicitly — that is the r2 fix for
    // the last raw `current_font_size()` call at the root text_size site.
    let allowed = [
        "theme::title(",
        "theme::body(",
        "theme::label(",
        "theme::label_small(",
        "theme::label_micro(",
    ];
    let mut offenders = Vec::new();
    for (name, src) in sources {
        let needle = ".text_size(";
        let mut cursor = 0usize;
        while let Some(pos) = src[cursor..].find(needle) {
            let start = cursor + pos + needle.len();
            let tail = &src[start..];
            let matched = allowed.iter().any(|prefix| tail.starts_with(prefix));
            if !matched {
                // Report line + 40-char preview so a reviewer can locate the
                // offending call without opening the file.
                let line = src[..start].matches('\n').count() + 1;
                let preview: String = tail.chars().take(40).collect();
                offenders.push(format!("{name}:{line}: .text_size({preview}"));
            }
            cursor = start;
        }
    }
    assert!(
        offenders.is_empty(),
        "text_size sites must feed one of theme::{{title,body,label,\
         label_small,label_micro}}(...) — no raw px literal, no ambient \
         alias (label_size / meta_size / small / fallback_size / base_size), \
         no current_font_size(), no prose_max_width():\n  - {}",
        offenders.join("\n  - ")
    );
}

/// Foreground / canvas contrast must stay above WCAG AA at EVERY role size
/// on every shipped theme. The check is size-agnostic (contrast is a color
/// ratio, not a pixel ratio) but the assertion iterates every role so a
/// future palette that fails at the promoted title tier — where the wider
/// column exposes more glyph mass — fails here first.
#[test]
fn every_role_clears_wcag_aa_against_canvas_on_every_theme() {
    fn relative_luminance(color: gpui::Hsla) -> f32 {
        let rgba = color.to_rgb();
        let channel = |c: f32| {
            if c <= 0.03928 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        };
        0.2126 * channel(rgba.r) + 0.7152 * channel(rgba.g) + 0.0722 * channel(rgba.b)
    }
    fn contrast_ratio(a: gpui::Hsla, b: gpui::Hsla) -> f32 {
        let la = relative_luminance(a);
        let lb = relative_luminance(b);
        let (lmax, lmin) = if la >= lb { (la, lb) } else { (lb, la) };
        (lmax + 0.05) / (lmin + 0.05)
    }
    for id in theme::ThemeId::ALL {
        let p = id.palette();
        // Roles below title are all "normal" text under WCAG, so the AA bar
        // is 4.5:1 across the board. Title sits at ~15-20px, well below the
        // 18pt / 24px large-text threshold, so it takes the same bar.
        for role_name in ["title", "body", "label", "label_small", "label_micro"] {
            let ratio = contrast_ratio(p.text, p.canvas);
            assert!(
                ratio >= 4.5,
                "{}: role {role_name} text/canvas contrast {ratio:.2}:1 fails \
                 WCAG AA (need >=4.5:1)",
                id.label()
            );
        }
        // Muted foreground on canvas — used for hints/metadata at label_small.
        // The bar drops to 4.5:1 at small sizes; make sure every theme clears
        // it so hint text does not disappear at the picker's MIN base.
        let muted_ratio = contrast_ratio(p.text_muted, p.canvas);
        assert!(
            muted_ratio >= 4.5,
            "{}: muted foreground/canvas contrast {muted_ratio:.2}:1 fails \
             WCAG AA at label_small hint sites",
            id.label()
        );
    }
}

/// Tool rows keep the wider `TRANSCRIPT_MAX_WIDTH` column so a long
/// command line or code block does not re-wrap at the prose measure. The
/// dual of `transcript_prose_column_caps_at_reading_measure_and_centers`
/// (which pins the narrower prose cap for user / assistant / thinking).
#[gpui::test]
fn tool_rows_keep_the_wide_transcript_column(cx: &mut TestAppContext) {
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.simulate_resize(gpui::size(px(1600.), px(760.)));
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.transcript = vec![TranscriptEntry::Tool {
                key: zeta_gui::state::ToolReceiptKey {
                    session_id: None,
                    agent_instance_id: None,
                    tool_call_id: "wide-tool".into(),
                },
                name: "bash".into(),
                excerpt: "run a very long command line ".repeat(30),
                summary: "run a very long command line ".repeat(30),
                complete: true,
                error: false,
                canceled: false,
                card: zeta_gui::cards::Card::default(),
            }];
            view.transcript.update(cx, |scroll, cx| scroll.reset(1, cx));
            cx.notify();
        });
        window.draw(cx).clear(cx);
    });
    let base = visual.update(|_, cx| cx.theme().font_size);
    let prose_cap = f32::from(theme::prose_max_width(base));
    let wide_cap = f32::from(theme::TRANSCRIPT_MAX_WIDTH);
    // The `transcript-column` debug selector sits on the inner max_w'd div
    // — the actual visible cap. `transcript-row` is the outer full-width
    // wrapper that centers the column.
    let column = visual
        .debug_bounds("transcript-column")
        .expect("tool column draws");
    assert!(
        f32::from(column.size.width) <= wide_cap + 4.0,
        "tool column width {:?} exceeded wide cap {wide_cap}",
        column.size.width,
    );
    assert!(
        f32::from(column.size.width) > prose_cap + 50.0,
        "tool column width {:?} must clearly exceed the prose cap \
         {prose_cap} — otherwise the split gate did not activate",
        column.size.width,
    );
}

/// Fenced code blocks INSIDE an assistant markdown row ride the same prose
/// cap as the surrounding prose. The r2 finding 4 review flagged that the
/// PR body implied code fences kept the wide `TRANSCRIPT_MAX_WIDTH` column;
/// they do not, and the accepted r2 shape is the narrower cap so the
/// reading rhythm around the fence stays intact. A refactor that hoists a
/// per-block splitter (fence → wide, prose → narrow) fails this test and
/// forces the change to update the review note in
/// `transcript_render.rs` at the same time.
#[gpui::test]
fn assistant_code_fence_rides_the_prose_cap(cx: &mut TestAppContext) {
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.simulate_resize(gpui::size(px(1600.), px(760.)));
    // Assistant turn with a paragraph plus a fenced code block. At a
    // 1600px viewport the row would otherwise fit the whole content on a
    // wide column; the prose cap must clamp it back to the reading
    // measure regardless.
    let source = "\
Here is a code block:\n\
\n\
```rust\n\
fn main() {\n\
    println!(\"a long line to prove the code fence never widens the row\");\n\
}\n\
```\n\
\n\
And a paragraph after the fence.";
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.transcript = vec![TranscriptEntry::Assistant(source.into())];
            view.transcript.update(cx, |scroll, cx| scroll.reset(1, cx));
            cx.notify();
        });
        window.draw(cx).clear(cx);
    });
    let column = visual
        .debug_bounds("transcript-column")
        .expect("assistant column draws");
    let base = visual.update(|_, cx| cx.theme().font_size);
    let prose_cap = f32::from(theme::prose_max_width(base));
    assert!(
        f32::from(column.size.width) <= prose_cap + 4.0,
        "assistant column width {:?} exceeded prose cap {prose_cap} — a \
         per-block splitter that gave code fences the wide cap regressed \
         the r2 accepted shape",
        column.size.width,
    );
    // Tool rows keep the wider cap — reasserted here to guard the flip
    // side of the choice: if a future refactor merged prose and tool
    // rows onto the same cap, both would end up at whichever was wider.
    assert!(
        f32::from(column.size.width) < f32::from(theme::TRANSCRIPT_MAX_WIDTH),
        "prose cap {prose_cap} must sit strictly below TRANSCRIPT_MAX_WIDTH \
         {:?} — otherwise the split gate for tool receipts is a no-op",
        theme::TRANSCRIPT_MAX_WIDTH,
    );
}

/// Full-layout bounds at the picker's MIN 11px and MAX 18px extremes.
/// The five ZETA-123 chrome regions — run header, sidebar, modal panel,
/// composer, transcript viewport — must all sit inside the viewport
/// and not overlap each other at either extreme. This is the r2
/// finding 3 gate: the existing chip-scale test proved that individual
/// chip sub-parts scale with the picker, but it never asked whether the
/// WHOLE chrome still fits at 11px (tighter mono → composer footer risks
/// stealing header space) or at 18px (wider glyphs → sidebar rows risk
/// pushing transcript-column into the sidebar). Runs the layout at both
/// extremes and reads every debug bound; a regression that clips one of
/// the regions inside another fails here.
#[gpui::test]
fn full_layout_regions_fit_at_11px_and_18px(cx: &mut TestAppContext) {
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    // Wide + tall viewport so the 18px case has room; the extremes case
    // proves each region computes its own size correctly, not that the
    // window is huge. 1600x900 is a common laptop shape.
    visual.simulate_resize(gpui::size(px(1600.), px(900.)));
    // Seed a transcript row so `transcript-viewport` paints and the
    // composer sits above pending work rather than clinging to the top.
    // Also open Settings later per extreme to reach `settings-panel`.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.transcript = vec![TranscriptEntry::User("hello".into())];
            view.transcript.update(cx, |scroll, cx| scroll.reset(1, cx));
            cx.notify();
        });
        window.draw(cx).clear(cx);
    });
    let mut appearance = theme::Appearance::default();
    let mut assert_regions_at = |base_px: f32, visual: &mut VisualTestContext| {
        appearance.font_size = theme::clamp_font_size(base_px);
        let viewport_size = visual.update(|window, cx| {
            theme::apply_with(cx, &appearance);
            view.update(cx, |_, cx| cx.notify());
            window.draw(cx).clear(cx);
            window.viewport_size()
        });
        // Viewport bounds in gpui-content coordinates (origin at 0,0).
        let viewport = gpui::Bounds {
            origin: gpui::point(px(0.), px(0.)),
            size: viewport_size,
        };
        let header = visual
            .debug_bounds("run-header")
            .unwrap_or_else(|| panic!("run-header at {base_px}px"));
        let sidebar = visual
            .debug_bounds("sidebar-header")
            .unwrap_or_else(|| panic!("sidebar-header at {base_px}px"));
        let transcript = visual
            .debug_bounds("transcript-viewport")
            .unwrap_or_else(|| panic!("transcript-viewport at {base_px}px"));
        let composer = visual
            .debug_bounds("composer")
            .unwrap_or_else(|| panic!("composer at {base_px}px"));
        // Every region sits inside the window viewport at both extremes.
        for (name, region) in [
            ("run-header", &header),
            ("sidebar-header", &sidebar),
            ("transcript-viewport", &transcript),
            ("composer", &composer),
        ] {
            assert!(
                region.left() >= viewport.left() - px(1.)
                    && region.right() <= viewport.right() + px(1.)
                    && region.top() >= viewport.top() - px(1.)
                    && region.bottom() <= viewport.bottom() + px(1.),
                "{name} at {base_px}px leaks outside viewport {viewport:?}: {region:?}"
            );
            assert!(
                region.size.width > px(0.) && region.size.height > px(0.),
                "{name} at {base_px}px has zero size {:?}",
                region.size,
            );
        }
        // Run-header-title content-visibility floor (r3 finding 2).
        // Container-bounds checks above pass even when a caller clips
        // the title to 1px, so a paint-time regression that hides the
        // session identity slips through. Assert the title's rendered
        // width lands AT LEAST at `HEADER_TITLE_MIN_WIDTH` — the floor
        // the layout hands the title before flex 1 grows it. A caller
        // that sets `.max_w(px(1.))` over the title, or drops the
        // `min_w(HEADER_TITLE_MIN_WIDTH)` guard, collapses the title to
        // a hairline and fails here. (See r3 attestation: the same
        // mutation reproduced in-code drops this assertion to 1.0 vs
        // the current floor of 80px.)
        let title = visual
            .debug_bounds("run-header-title")
            .unwrap_or_else(|| panic!("run-header-title at {base_px}px"));
        assert!(
            title.size.width >= theme::HEADER_TITLE_MIN_WIDTH - px(1.),
            "run-header-title collapsed to {:?} at {base_px}px — the title \
             content is clipped below the {:?} floor (r3 finding 2: bounds \
             checks alone are blind to a 1px clip mutation)",
            title.size.width,
            theme::HEADER_TITLE_MIN_WIDTH,
        );
        // Header sits above the transcript; transcript sits above composer;
        // composer sits inside the main column (right of sidebar). At the
        // picker extremes a broken layout typically manifests as the
        // header stealing composer space or the sidebar bleeding into the
        // transcript viewport, both of which fail here.
        assert!(
            header.bottom() <= transcript.top() + px(1.),
            "run-header overlaps transcript at {base_px}px (header {header:?}, transcript {transcript:?})",
        );
        assert!(
            transcript.bottom() <= composer.top() + px(1.),
            "transcript overlaps composer at {base_px}px (transcript {transcript:?}, composer {composer:?})",
        );
        assert!(
            sidebar.right() <= transcript.left() + px(1.),
            "sidebar bleeds into transcript at {base_px}px (sidebar {sidebar:?}, transcript {transcript:?})",
        );
        assert!(
            sidebar.right() <= composer.left() + px(1.),
            "sidebar bleeds into composer at {base_px}px (sidebar {sidebar:?}, composer {composer:?})",
        );
        // Modal panel: open Settings by flipping the state flag directly
        // (peer tests do the same — the panel gate depends on session
        // state we do not model here) and re-read. The modal must sit
        // inside the viewport at both extremes; at 18px a modal that
        // hardcodes its inner text size (rather than routing through the
        // roles) could push the panel past the viewport bottom.
        visual.update(|window, cx| {
            view.update(cx, |view, cx| {
                view.settings_open = true;
                cx.notify();
            });
            window.draw(cx).clear(cx);
        });
        let modal = visual
            .debug_bounds("settings-panel")
            .unwrap_or_else(|| panic!("settings-panel at {base_px}px"));
        assert!(
            modal.left() >= viewport.left() - px(1.)
                && modal.right() <= viewport.right() + px(1.)
                && modal.top() >= viewport.top() - px(1.)
                && modal.bottom() <= viewport.bottom() + px(1.),
            "settings-panel at {base_px}px leaks outside viewport {viewport:?}: {modal:?}",
        );
        // Close the modal so the next extreme starts from the same state.
        visual.update(|window, cx| {
            view.update(cx, |view, cx| {
                view.settings_open = false;
                cx.notify();
            });
            window.draw(cx).clear(cx);
        });
    };
    assert_regions_at(theme::MIN_FONT_SIZE_PX, &mut visual);
    assert_regions_at(theme::MAX_FONT_SIZE_PX, &mut visual);
}

/// Regression harness for the ZETA-124 wedge list content: an assistant
/// markdown row carrying a numbered list where one item's paragraph
/// combines inline code chips (`meta.json` / `conversation.jsonl`) with a
/// long hanging-indent continuation. The r1 critique screenshot at
/// 2204x1608 showed a stray `n` shaping past the row column edge.
///
/// The r2 fix here is at the RENDERER's wrap-boundary math: `prose_max_width`
/// now includes the row's `.px_4()` horizontal padding on each side, so
/// the effective TEXT area lands at exactly `PROSE_MEASURE_CH` glyph
/// advances rather than that minus ~4 chars the padding used to steal.
/// The regression harness runs at the exact 2204x1608 shape the critique
/// captured and asserts (a) glyph containment — every painted quad
/// inside the row sits inside the row's inner text column, so a
/// subpixel-rounded wrap that pokes a glyph rectangle past the edge
/// fails here — and (b) hanging-indent continuation — the row wraps
/// onto multiple lines, so a regression that widens the column past the
/// prose cap collapses it back to a single line and fails.
#[gpui::test]
fn wedge_list_item_wraps_inside_the_prose_column(cx: &mut TestAppContext) {
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    // The critique shot: 2204x1608, the shape that put a stray `n` past
    // the row edge under the r1 prose cap. Any regression to a wider column
    // (or a padding-not-in-cap variant that shortens the effective measure)
    // shifts the wrap point relative to this shape, so glyphs land in
    // different places — the containment check below catches that.
    visual.simulate_resize(gpui::size(px(2204.), px(1608.)));
    let wedge = "\
2. `zeta serve` session hardening — half-written session dirs \
(`conversation.jsonl` without `meta.json`) wedge status/list. Atomic dir \
creation via `meta.json` tmp+rename.\n\
3. Follow-up work with additional wrapping to exercise the hanging indent \
so the paragraph reliably breaks onto a continuation line even at 2204px.";
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.transcript = vec![TranscriptEntry::Assistant(wedge.into())];
            view.transcript.update(cx, |scroll, cx| scroll.reset(1, cx));
            cx.notify();
        });
        window.draw(cx).clear(cx);
    });
    let row = visual
        .debug_bounds("transcript-row")
        .expect("wedge row draws");
    let column = visual
        .debug_bounds("transcript-column")
        .expect("wedge column draws");
    let transcript = visual
        .debug_bounds("transcript-viewport")
        .expect("transcript viewport draws");
    let base = visual.update(|_, cx| cx.theme().font_size);
    let prose_cap = f32::from(theme::prose_max_width(base));
    let text_measure = f32::from(theme::prose_text_measure(base));
    // The wedge assistant is a prose row — its inner column must sit at
    // the narrower measure so the hanging-indent list content wraps at a
    // scannable width. `+4` guards against the pipeline's subpixel rounding.
    assert!(
        f32::from(column.size.width) <= prose_cap + 4.0,
        "wedge assistant column width {:?} exceeded prose cap {prose_cap} \
         — the ZETA-124 measure gate is off",
        column.size.width,
    );
    // The column sits centered inside the transcript viewport: left and
    // right gaps balance within a few pixels. A regression that shifts the
    // wrap point past the visible column (the r0 orphan-glyph shape)
    // drifts the centering here first.
    let viewport_center = transcript.left() + transcript.size.width / 2.0;
    let column_center = column.left() + column.size.width / 2.0;
    let drift = if viewport_center > column_center {
        viewport_center - column_center
    } else {
        column_center - viewport_center
    };
    assert!(
        f32::from(drift) < 8.0,
        "wedge column not centered inside transcript viewport: drift {drift:?}",
    );
    // Glyph containment: every quad painted inside the row bounds sits
    // inside the inner text column (column bounds minus the row's
    // horizontal padding). GPUI does not expose glyph-level bounds, but
    // rich text paints backgrounds/rails/underlines as quads inside the
    // same wrap path, so a glyph that shapes past the edge takes its
    // painted rectangle with it. A tolerance of 4 scaled pixels covers
    // the pipeline's subpixel rounding without letting a whole glyph
    // slip past.
    visual.update(|window, _cx| {
        let scale = window.scale_factor();
        let scaled_row = row.scale(scale);
        let scaled_column = column.scale(scale);
        let pad_scaled = px(theme::PROSE_ROW_PADDING_X).scale(scale);
        let inner_left = scaled_column.left() + pad_scaled;
        let inner_right = scaled_column.right() - pad_scaled;
        let tolerance = px(4.).scale(scale);
        // Quads inside the row's vertical AND the transcript column's
        // horizontal band: this excludes the sidebar / composer strips
        // that paint at the same y-range but on the other side of the
        // window. A quad that starts left of the column left edge is
        // outside the transcript column entirely and cannot regress the
        // wrap boundary this test guards.
        let row_quads: Vec<_> = window
            .painted_quads()
            .into_iter()
            .filter(|quad| {
                quad.bounds.top() >= scaled_row.top()
                    && quad.bounds.bottom() <= scaled_row.bottom() + tolerance
                    && quad.bounds.left() >= scaled_column.left() - tolerance
                    && quad.bounds.size.width > gpui::ScaledPixels::default()
            })
            .collect();
        // At least one quad should have been painted inside the row — a
        // hard zero would mean the assertion body below is vacuous. The
        // wedge content paints the assistant row's markdown, which shapes
        // at least one primitive per line.
        assert!(
            !row_quads.is_empty(),
            "no quads observed inside wedge row bounds {scaled_row:?} — the \
             draw pass never landed a primitive inside the row",
        );
        for quad in row_quads {
            assert!(
                quad.bounds.left() >= inner_left - tolerance,
                "quad {:?} started left of the inner text column {inner_left:?}",
                quad.bounds,
            );
            assert!(
                quad.bounds.right() <= inner_right + tolerance,
                "quad {:?} shaped past the inner text column {inner_right:?} \
                 — the wrap boundary regressed",
                quad.bounds,
            );
        }
        // Hanging-indent continuation: the wedge content is long enough
        // that at the 90ch prose measure it wraps onto multiple visible
        // lines. Assert the row is taller than a single line at the
        // current base — this catches a regression that reverts to
        // `TRANSCRIPT_MAX_WIDTH` (which would let the whole paragraph fit
        // on one line at 2204px) and it catches a padding-included cap
        // that quietly grew the measure back past 90ch on this shape.
        let single_line = f32::from(base) * 1.65;
        let row_height = f32::from(row.size.height);
        assert!(
            row_height > single_line * 2.5,
            "wedge row height {row_height} did not exceed 2.5 line-heights \
             ({}) — the hanging-indent continuation did not paint on \
             separate lines (prose_cap {prose_cap}, text_measure \
             {text_measure})",
            single_line * 2.5,
        );
    });
}

/// Text-run recorder acceptance: closes the r3 review's blind-fix gap
/// on the wedge defect. `painted_quads()` reports rectangles only, so
/// laid-out glyphs paint as sprite primitives no test accessor exposes.
/// The recorder (`super::record_text_geometry`) closes that gap by
/// shaping the assistant row's source through `shape_line` +
/// `LineWrapper::wrap_line` at the SAME wrap width the row hands to the
/// text system, and reporting each wrap segment's shaped extent —
/// exactly the measurement `painted_quads()` cannot see.
///
/// The acceptance bar the r3 review pinned: "no laid-out text extends
/// past the column's content box." The test reads the row's ACTUAL
/// inner text column (column bounds minus the row's horizontal padding)
/// and asserts every recorder sample's `max_wrap_segment_width` fits
/// inside it — so a formula regression that shrinks the row past the
/// promised prose measure (r0/r1 shape: `prose_max_width` did NOT
/// include the row's `.px_4()` padding, leaving the effective text
/// column ~4ch short of the promised 88ch) shows up here as an overflow
/// even though `painted_quads` sees nothing wrong.
///
/// Runs across the r3 SHAPE MATRIX: the r1 critique wedge, the
/// reviewer's adjacent numbered-list shape, a deeper-nesting list, and
/// long-token content that stresses the wrap engine's unbreakable-token
/// path. Each shape is drawn at BOTH the r1 shot (2204x1608) and the
/// setup shape (1600x760) AND at 11px / 13px / 18px so a width tweak
/// that passes ONE cell but fails the matrix (the exact way r0 and r1
/// slipped through) fails here.
#[gpui::test]
fn zeta124_wrap_segments_fit_inside_the_prose_column_across_the_matrix(cx: &mut TestAppContext) {
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);

    // Shape matrix — every entry the r3 review named. Together they
    // cover the wedge critique, the reviewer's adjacent list, deeper
    // nesting, and the wrap engine's unbreakable-token path.
    let wedge = "\
2. `zeta serve` session hardening — half-written session dirs \
(`conversation.jsonl` without `meta.json`) wedge status/list. Atomic dir \
creation via `meta.json` tmp+rename.\n\
3. Follow-up work with additional wrapping to exercise the hanging indent \
so the paragraph reliably breaks onto a continuation line even at 2204px.";
    let adjacent = "\
1. Outer numbered item with plenty of prose to force wrapping onto \
multiple continuation lines at every picker step.\n\
2. Second outer numbered item to prove the second sibling wraps in the \
same column geometry as the first with more filler prose here now.";
    let nested = "\
1. Outer item with room to spare.\n\
   - Nested bullet A that itself carries enough hanging-indent text to \
force wrap boundaries near the prose cap at every base picker step.\n\
   - Nested bullet B with more prose — deeper nesting stays inside the \
same column even when the marker indent has consumed a few characters.";
    let long_token = "\
Prose leading up to a very long unbroken token that the wrap engine \
cannot break: \
supercalifragilisticexpialidocious_but_much_longer_than_any_column_should_ever_be_aaaaaaaaaaaaaaaaaaaa \
and then some trailing prose after it.";
    let shapes: &[(&str, &str)] = &[
        ("wedge", wedge),
        ("adjacent", adjacent),
        ("nested", nested),
        ("long_token", long_token),
    ];
    let viewports = &[(px(1600.), px(760.)), (px(2204.), px(1608.))];
    let bases = &[
        theme::MIN_FONT_SIZE_PX,
        f32::from(theme::DEFAULT_FONT_SIZE),
        theme::MAX_FONT_SIZE_PX,
    ];
    let mut appearance = theme::Appearance::default();
    for &(vw, vh) in viewports {
        visual.simulate_resize(gpui::size(vw, vh));
        for &base_px in bases {
            appearance.font_size = theme::clamp_font_size(base_px);
            for &(label, source) in shapes {
                visual.update(|window, cx| {
                    theme::apply_with(cx, &appearance);
                    view.update(cx, |view, cx| {
                        view.state.transcript = vec![TranscriptEntry::Assistant(source.into())];
                        view.transcript.update(cx, |scroll, cx| scroll.reset(1, cx));
                        cx.notify();
                    });
                    window.draw(cx).clear(cx);
                });
                let column = visual.debug_bounds("transcript-column").unwrap_or_else(|| {
                    panic!("assistant column at {label} {vw:?}x{vh:?} {base_px}")
                });
                // Inner text column = column bounds minus row .px_4() on
                // each side. This is the ACTUAL width the TextView had
                // to wrap into, regardless of what `prose_max_width`
                // promised at this base. A formula that shrinks the
                // column past the promised measure exposes the gap here
                // because recorder samples are shaped at the PROMISED
                // wrap width (`prose_text_measure(base)`), not at the
                // row's actual inner width — so a shape-vs-column drift
                // lands as an overflow the assertion catches.
                let inner_width = f32::from(column.size.width) - 2.0 * theme::PROSE_ROW_PADDING_X;
                let samples = super::text_run_log::samples();
                assert!(
                    !samples.is_empty(),
                    "text_run_log must record an assistant sample at {label} \
                     {vw:?}x{vh:?} {base_px}px",
                );
                for sample in &samples {
                    let wrap = f32::from(sample.wrap_width);
                    let seg = f32::from(sample.max_wrap_segment_width);
                    let unwrapped = f32::from(sample.max_unwrapped_line_width);
                    // Assertion 1: wrap width promised at THIS base must
                    // equal the row's actual inner text width. A caller
                    // that hands `prose_text_measure(base)` to the text
                    // system while giving the row a narrower inner
                    // column produces glyph overflow no matter how the
                    // wrap engine breaks the source.
                    assert!(
                        (wrap - inner_width).abs() < 2.0,
                        "prose row's inner width {inner_width} does not match \
                         wrap width {wrap} at {label} {vw:?}x{vh:?} {base_px}px \
                         — the r2 formula gap is back (row {:?})",
                        sample.row_id,
                    );
                    // Assertion 2: every wrap segment's shaped extent
                    // fits inside the wrap width. An unbreakable token
                    // wider than the column is the only shape that can
                    // trip this; when it happens, the recorder catches
                    // exactly the class of defect painted_quads misses.
                    assert!(
                        seg <= wrap + 1.0,
                        "wrap segment shaped past the wrap width at {label} \
                         {vw:?}x{vh:?} {base_px}px on row {:?}: \
                         max_wrap_segment_width={seg} > wrap_width={wrap} \
                         (max unwrapped line width {unwrapped}, wrap segments \
                         {segs}, source_len {source_len})",
                        sample.row_id,
                        source_len = sample.source_len,
                        segs = sample.wrap_segment_count,
                    );
                    // Assertion 3: every wrap segment fits inside the
                    // row's ACTUAL inner text column. This is the r3
                    // acceptance bar — "no laid-out text extends past
                    // the column's content box" — restated as a
                    // recorder-based check. Any drift between wrap
                    // width and column geometry surfaces here.
                    assert!(
                        seg <= inner_width + 1.0,
                        "wrap segment shaped past the row's inner column at \
                         {label} {vw:?}x{vh:?} {base_px}px on row {:?}: \
                         max_wrap_segment_width={seg} > inner_width \
                         {inner_width}",
                        sample.row_id,
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ZETA-125: Tool receipt redesign.
//
// This region is deliberately separated from earlier tests so a sibling lane
// (ZETA-127) that rebuilds test infra elsewhere in this file can rebase
// cleanly. Every test below covers ONE part of the ZETA-125 contract:
// excerpt extraction and truncation, metadata adjacency, grouping at 3+,
// mouse + keyboard expansion, streaming-forces-expanded, and correctness
// across the 11px and 18px scale.
// ---------------------------------------------------------------------------

/// Excerpt extraction routes bash/read/write/edit/fetch through the
/// argument key the tool actually reads, and falls back to the tool name
/// for tools whose arguments carry nothing useful. A pathological command
/// longer than the character cap truncates with a single-character
/// ellipsis so a wide argument still fits on one row.
#[test]
fn zeta125_excerpts_route_by_kind_and_truncate() {
    use serde_json::json;
    let excerpt = |name: &str, args: serde_json::Value| {
        let map = args.as_object().cloned().unwrap_or_default();
        zeta_gui::state::tool_excerpt(name, &map)
    };
    // Bash-family tools read the "command" argument's first line.
    assert_eq!(
        excerpt("bash", json!({"command": "grep -rn TODO src/"})),
        "grep -rn TODO src/"
    );
    assert_eq!(
        excerpt("exec", json!({"command": "ls -la\nsecond line"})),
        "ls -la"
    );
    // Read/write/edit route through "path".
    assert_eq!(
        excerpt("read", json!({"path": "src/main.rs"})),
        "src/main.rs"
    );
    assert_eq!(excerpt("write", json!({"path": "notes.txt"})), "notes.txt");
    assert_eq!(
        excerpt("edit", json!({"path": "docs/design.md"})),
        "docs/design.md"
    );
    // Fetch reads "url" and keeps the whole URL under the cap.
    assert_eq!(
        excerpt("fetch", json!({"url": "https://example.com/api/v1/data"})),
        "https://example.com/api/v1/data"
    );
    // Unknown tools fall back to the first primitive argument.
    assert_eq!(excerpt("weather", json!({"city": "Paris"})), "Paris");
    // With no primitive argument, fall back to the tool name.
    assert_eq!(excerpt("noop", json!({})), "noop");
    // Truncation trims to `EXCERPT_CHARS` and appends a single-character
    // ellipsis marker. The output length is at most cap + 1 char.
    let long = "a".repeat(zeta_gui::state::EXCERPT_CHARS * 2);
    let truncated = excerpt("bash", json!({"command": long}));
    assert!(truncated.ends_with('…'));
    assert_eq!(
        truncated.chars().count(),
        zeta_gui::state::EXCERPT_CHARS + 1,
        "truncated excerpt must be cap + ellipsis",
    );
}

/// Metadata adjacency: with a collapsed receipt that has output bytes on
/// the tail, the metadata paints DIRECTLY next to the excerpt end. The
/// gap between the excerpt's right edge and the metadata's left edge
/// stays under a small proximity bound — laws-of-ux proximity, the fix
/// for the pre-ZETA-125 "huge right-aligned gap".
#[gpui::test]
fn zeta125_metadata_paints_directly_after_the_excerpt(cx: &mut TestAppContext) {
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.transcript = vec![TranscriptEntry::Tool {
                key: zeta_gui::state::ToolReceiptKey {
                    session_id: None,
                    agent_instance_id: None,
                    tool_call_id: "solo".into(),
                },
                name: "bash".into(),
                excerpt: "cargo check".into(),
                summary: String::new(),
                complete: true,
                error: false,
                canceled: false,
                card: zeta_gui::cards::Card {
                    tail: zeta_gui::cards::OutputTail {
                        text: "x".repeat(4096),
                        truncated: false,
                        bytes_seen: 4096,
                    },
                    expanded: false,
                    ..Default::default()
                },
            }];
            view.transcript.update(cx, |scroll, cx| scroll.reset(1, cx));
            cx.notify();
        });
        window.draw(cx).clear(cx);
    });
    let excerpt = visual
        .debug_bounds("tool-excerpt-0")
        .expect("excerpt paints");
    let metadata = visual
        .debug_bounds("tool-metadata-0")
        .expect("metadata paints for a collapsed receipt with output");
    // Metadata sits on the same visual baseline as the excerpt.
    assert!(
        metadata.top() < excerpt.bottom() && metadata.bottom() > excerpt.top(),
        "metadata must sit on the same row as the excerpt (excerpt {excerpt:?}, metadata {metadata:?})"
    );
    // The excerpt gets the flex_1 slot so it occupies the middle; a raw
    // gap under 64px keeps the two elements visually adjacent even after
    // the excerpt truncates.
    let gap = metadata.left() - excerpt.right();
    assert!(
        gap < px(64.),
        "metadata must sit adjacent to the excerpt end (gap={gap:?}); the \
         pre-ZETA-125 layout right-aligned this element across the whole \
         column and the fix was 'kill that gap'"
    );
}

/// Grouping: a run of 3+ consecutive tool receipts renders as ONE group
/// summary row when collapsed. The summary carries the correct count and
/// the total output bytes. Below the threshold each receipt renders on
/// its own row.
#[gpui::test]
fn zeta125_grouping_at_three_or_more_receipts(cx: &mut TestAppContext) {
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    let tool = |id: &str, bytes: usize| TranscriptEntry::Tool {
        key: zeta_gui::state::ToolReceiptKey {
            session_id: None,
            agent_instance_id: None,
            tool_call_id: id.into(),
        },
        name: "bash".into(),
        excerpt: format!("cargo test {id}"),
        summary: String::new(),
        complete: true,
        error: false,
        canceled: false,
        card: zeta_gui::cards::Card {
            tail: zeta_gui::cards::OutputTail {
                text: "x".repeat(bytes),
                truncated: false,
                bytes_seen: bytes,
            },
            ..Default::default()
        },
    };
    // Below threshold: 2 consecutive receipts remain individual.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.transcript = vec![tool("a", 100), tool("b", 200)];
            view.transcript.update(cx, |scroll, cx| scroll.reset(2, cx));
            cx.notify();
        });
        window.draw(cx).clear(cx);
    });
    assert!(
        visual.debug_bounds("tool-receipt-0").is_some(),
        "below the threshold both receipts render individually"
    );
    assert!(
        visual.debug_bounds("tool-receipt-1").is_some(),
        "below the threshold both receipts render individually"
    );
    assert!(
        visual.debug_bounds("tool-group-0").is_none(),
        "below the threshold NO group summary row paints"
    );
    // At threshold: 3 consecutive receipts collapse into ONE summary row
    // when the group is not expanded.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.transcript = vec![tool("a", 100), tool("b", 200), tool("c", 300)];
            view.transcript.update(cx, |scroll, cx| scroll.reset(3, cx));
            cx.notify();
        });
        window.draw(cx).clear(cx);
    });
    assert!(
        visual.debug_bounds("tool-group-0").is_some(),
        "at threshold the group summary row paints on the start index"
    );
    assert!(
        visual.debug_bounds("tool-receipt-1").is_none(),
        "interior receipt rows paint nothing when the group is collapsed"
    );
    assert!(
        visual.debug_bounds("tool-receipt-2").is_none(),
        "interior receipt rows paint nothing when the group is collapsed"
    );
    // The row-text model composes count + total from the run.
    view.read_with(&visual, |view, _| {
        let group = view
            .state
            .tool_group_position(0)
            .expect("group position at index 0");
        assert_eq!(group.count(), 3);
        let bytes = view.state.tool_group_output_bytes(&group);
        assert_eq!(bytes, 600);
        let excerpts: Vec<&str> = (group.first_index..=group.last_index)
            .filter_map(|i| match &view.state.transcript[i] {
                TranscriptEntry::Tool { excerpt, .. } => Some(excerpt.as_str()),
                _ => None,
            })
            .collect();
        let text = zeta_gui::row_text::build_tool_group(
            &excerpts,
            bytes,
            zeta_gui::state::TOOL_GROUP_PREVIEW_MAX,
            false,
        );
        assert_eq!(text.count_label, "3 tool calls");
        assert_eq!(text.total_label.as_deref(), Some("600B"));
        assert_eq!(text.preview_excerpts.len(), 2, "preview capped at 2");
        assert!(
            text.aria_label.contains(", collapsed"),
            "accessible label announces the collapsed state: {:?}",
            text.aria_label
        );
    });
}

/// r2 review finding 4: the group header/button must remain painted
/// while the group is EXPANDED so keyboard-only users can still collapse
/// it. Before the r2 fix the header disappeared on expansion and only
/// individual receipts remained — a keyboard user had no target to focus.
/// This test drives real Enter keystrokes against the tab-stop header,
/// verifies the header keeps painting after expansion, and drives a
/// second Enter to collapse the group again. Enter/Space are the two
/// activation keys the on_key_down handler accepts.
#[gpui::test]
fn zeta125_group_header_persists_when_expanded_and_toggles_via_real_keystrokes(
    cx: &mut TestAppContext,
) {
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    let tool = |id: &str, bytes: usize| TranscriptEntry::Tool {
        key: zeta_gui::state::ToolReceiptKey {
            session_id: None,
            agent_instance_id: None,
            tool_call_id: id.into(),
        },
        name: "read".into(),
        excerpt: format!("src/{id}.rs"),
        summary: String::new(),
        complete: true,
        error: false,
        canceled: false,
        card: zeta_gui::cards::Card {
            tail: zeta_gui::cards::OutputTail {
                text: "x".repeat(bytes),
                truncated: false,
                bytes_seen: bytes,
            },
            ..Default::default()
        },
    };
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.transcript = vec![tool("a", 100), tool("b", 100), tool("c", 100)];
            view.transcript.update(cx, |scroll, cx| scroll.reset(3, cx));
            cx.notify();
        });
        window.draw(cx).clear(cx);
    });
    assert!(
        visual.debug_bounds("tool-group-0").is_some(),
        "collapsed group paints its header row",
    );
    // r4 finding 3: reach the group header through a REAL tab-key
    // walk — `simulate_keystrokes("tab")` drives the Root keymap's
    // Tab -> `focus_next` binding end-to-end (gpui-component
    // `root::init`), not just the `focus_next` method the pre-r4
    // test called directly. Jamming focus onto the handle via
    // `window.focus(&handle)` succeeds even for handles that are not
    // real tab stops; the key-walk proves the header is reachable
    // from the keyboard tab-stops registry via the same key path a
    // real user drives.
    let focus_key = zeta_gui::row_text::sel::tool_group_focus_key("a");
    let group_handle = visual
        .update(|_, cx| {
            view.read(cx)
                .tool_group_focus
                .borrow()
                .get(&focus_key)
                .cloned()
        })
        .expect("group focus handle registered on first paint");
    visual.update(|window, cx| {
        window.blur(cx);
        window.draw(cx).clear(cx);
    });
    let max_tab_steps = 64;
    let mut steps_to_group = None;
    for step in 0..max_tab_steps {
        visual.simulate_keystrokes("tab");
        if visual.update(|window, _| group_handle.is_focused(window)) {
            steps_to_group = Some(step + 1);
            break;
        }
    }
    let _ = steps_to_group.expect(
        "a Tab walk must land on the tool-group header within a bounded loop \
         — proves the header is a real tab stop reachable from the keyboard \
         registry, not just via `window.focus(handle)`",
    );
    visual.update(|window, cx| {
        view.update(cx, |_, cx| cx.notify());
        window.draw(cx).clear(cx);
    });
    visual.simulate_keystrokes("enter");
    visual.update(|window, cx| {
        view.update(cx, |_, cx| cx.notify());
        window.draw(cx).clear(cx);
    });
    // Header must PERSIST when expanded so the tab stop and toggle
    // remain reachable. The chevron flips to Down and the aria label
    // announces the expanded state; both are covered by the aria label
    // assertions further down.
    assert!(
        visual.debug_bounds("tool-group-0").is_some(),
        "expanded group KEEPS the header row visible — a keyboard user \
         needs a target to collapse back",
    );
    // The individual receipts also paint under the expanded header.
    assert!(visual.debug_bounds("tool-receipt-0").is_some());
    assert!(visual.debug_bounds("tool-receipt-2").is_some());
    // Space collapses the group again through the same key path.
    visual.simulate_keystrokes("space");
    visual.update(|window, cx| {
        view.update(cx, |_, cx| cx.notify());
        window.draw(cx).clear(cx);
    });
    assert!(
        visual.debug_bounds("tool-group-0").is_some(),
        "collapsed group still paints its header row",
    );
    assert!(
        visual.debug_bounds("tool-receipt-0").is_none(),
        "space toggles the group back to collapsed",
    );
}

/// A collapsed tool-group summary row is a real tab stop (matches
/// ZETA-108 a11y precedent) and toggles the group open on both mouse
/// click and Enter. After expansion, the individual tool-receipt rows
/// paint again and the aria label swaps to the "expanded" wording.
#[gpui::test]
fn zeta125_group_toggles_on_mouse_and_keyboard(cx: &mut TestAppContext) {
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    let tool = |id: &str, bytes: usize| TranscriptEntry::Tool {
        key: zeta_gui::state::ToolReceiptKey {
            session_id: None,
            agent_instance_id: None,
            tool_call_id: id.into(),
        },
        name: "read".into(),
        excerpt: format!("src/{id}.rs"),
        summary: String::new(),
        complete: true,
        error: false,
        canceled: false,
        card: zeta_gui::cards::Card {
            tail: zeta_gui::cards::OutputTail {
                text: "x".repeat(bytes),
                truncated: false,
                bytes_seen: bytes,
            },
            ..Default::default()
        },
    };
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.transcript = vec![tool("a", 100), tool("b", 100), tool("c", 100)];
            view.transcript.update(cx, |scroll, cx| scroll.reset(3, cx));
            cx.notify();
        });
        window.draw(cx).clear(cx);
    });
    // Initially collapsed — one group row, no receipt rows.
    assert!(visual.debug_bounds("tool-group-0").is_some());
    assert!(visual.debug_bounds("tool-receipt-0").is_none());
    // Toggle via the state seam (the same seam the mouse click handler
    // dispatches through). Expansion paints the individual receipts.
    visual.update(|window, cx| {
        view.update(cx, |view, _| view.state.toggle_tool_group("a"));
        view.update(cx, |_, cx| cx.notify());
        window.draw(cx).clear(cx);
    });
    assert!(
        visual.debug_bounds("tool-receipt-0").is_some(),
        "after expansion the first receipt row paints",
    );
    assert!(
        visual.debug_bounds("tool-receipt-2").is_some(),
        "after expansion the tail receipt row paints",
    );
    // Toggling again collapses the group — the same code path a
    // second Enter/Space keystroke drives.
    visual.update(|window, cx| {
        view.update(cx, |view, _| view.state.toggle_tool_group("a"));
        view.update(cx, |_, cx| cx.notify());
        window.draw(cx).clear(cx);
    });
    assert!(
        visual.debug_bounds("tool-receipt-0").is_none(),
        "collapsing hides the interior receipt rows again",
    );
    // The accessible label carries the expansion state so keyboard-only
    // users hear what changes on Enter/Space.
    view.read_with(&visual, |view, _| {
        let group = view.state.tool_group_position(0).expect("group position");
        let excerpts: Vec<&str> = (group.first_index..=group.last_index)
            .filter_map(|i| match &view.state.transcript[i] {
                TranscriptEntry::Tool { excerpt, .. } => Some(excerpt.as_str()),
                _ => None,
            })
            .collect();
        let collapsed = zeta_gui::row_text::build_tool_group(
            &excerpts,
            0,
            zeta_gui::state::TOOL_GROUP_PREVIEW_MAX,
            false,
        );
        let expanded = zeta_gui::row_text::build_tool_group(
            &excerpts,
            0,
            zeta_gui::state::TOOL_GROUP_PREVIEW_MAX,
            true,
        );
        assert!(collapsed.aria_label.contains(", collapsed"));
        assert!(expanded.aria_label.contains(", expanded"));
    });
}

/// Streaming turns force EVERY tool group in the current turn expanded
/// regardless of the user's explicit toggle map, so live tool activity
/// stays visible on screen without a click. Completing the turn
/// (streaming=false) restores the default collapsed state.
#[gpui::test]
fn zeta125_streaming_forces_current_turn_groups_expanded(cx: &mut TestAppContext) {
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    let tool = |id: &str| TranscriptEntry::Tool {
        key: zeta_gui::state::ToolReceiptKey {
            session_id: None,
            agent_instance_id: None,
            tool_call_id: id.into(),
        },
        name: "bash".into(),
        excerpt: format!("cmd {id}"),
        summary: String::new(),
        complete: true,
        error: false,
        canceled: false,
        card: zeta_gui::cards::Card::default(),
    };
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.transcript = vec![tool("a"), tool("b"), tool("c")];
            view.state.streaming = true;
            view.transcript.update(cx, |scroll, cx| scroll.reset(3, cx));
            cx.notify();
        });
        window.draw(cx).clear(cx);
    });
    // Streaming forces every current-turn group expanded — the interior
    // receipts paint.
    assert!(
        visual.debug_bounds("tool-receipt-0").is_some(),
        "streaming must expand every current-turn group without a user click",
    );
    // Ending the stream collapses the group back to the summary row.
    visual.update(|window, cx| {
        view.update(cx, |view, _| {
            view.state.streaming = false;
        });
        view.update(cx, |_, cx| cx.notify());
        window.draw(cx).clear(cx);
    });
    assert!(
        visual.debug_bounds("tool-group-0").is_some(),
        "after streaming ends the group returns to its collapsed default",
    );
    assert!(
        visual.debug_bounds("tool-receipt-0").is_none(),
        "after streaming ends the interior receipts hide again",
    );
}

/// Receipt layout stays correct at the picker's 11px floor AND 18px
/// ceiling: at both ends the label + excerpt + metadata all paint and the
/// metadata still sits adjacent to the excerpt (not right-aligned across
/// the column). A regression that hardcoded a font size to 13px would
/// leave the elements at the wrong size on either end.
#[gpui::test]
fn zeta125_receipt_layout_holds_at_11px_and_18px(cx: &mut TestAppContext) {
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.transcript = vec![TranscriptEntry::Tool {
                key: zeta_gui::state::ToolReceiptKey {
                    session_id: None,
                    agent_instance_id: None,
                    tool_call_id: "solo".into(),
                },
                name: "bash".into(),
                excerpt: "cargo test --lib".into(),
                summary: String::new(),
                complete: true,
                error: false,
                canceled: false,
                card: zeta_gui::cards::Card {
                    tail: zeta_gui::cards::OutputTail {
                        text: "x".repeat(2048),
                        truncated: false,
                        bytes_seen: 2048,
                    },
                    ..Default::default()
                },
            }];
            view.transcript.update(cx, |scroll, cx| scroll.reset(1, cx));
            cx.notify();
        });
        window.draw(cx).clear(cx);
    });
    let mut appearance = theme::Appearance::default();
    let mut bounds_at = |base: f32, visual: &mut VisualTestContext| {
        appearance.font_size = theme::clamp_font_size(base);
        visual.update(|window, cx| {
            theme::apply_with(cx, &appearance);
            view.update(cx, |_, cx| cx.notify());
            window.draw(cx).clear(cx);
        });
        (
            visual.debug_bounds("tool-label-0").expect("label paints"),
            visual
                .debug_bounds("tool-excerpt-0")
                .expect("excerpt paints"),
            visual
                .debug_bounds("tool-metadata-0")
                .expect("metadata paints"),
        )
    };
    let (small_label, small_excerpt, small_meta) = bounds_at(theme::MIN_FONT_SIZE_PX, &mut visual);
    let small_gap = small_meta.left() - small_excerpt.right();
    assert!(
        small_gap < px(64.),
        "11px metadata must sit adjacent to the excerpt end (gap={small_gap:?})"
    );
    assert!(small_label.size.width > px(0.));
    assert!(small_excerpt.size.width > px(0.));
    assert!(small_meta.size.width > px(0.));
    let (large_label, large_excerpt, large_meta) = bounds_at(theme::MAX_FONT_SIZE_PX, &mut visual);
    let large_gap = large_meta.left() - large_excerpt.right();
    assert!(
        large_gap < px(96.),
        "18px metadata must sit adjacent to the excerpt end (gap={large_gap:?})"
    );
    assert!(
        large_label.size.height > small_label.size.height,
        "tool_label height must scale (11px→{:?}, 18px→{:?})",
        small_label.size,
        large_label.size,
    );
    assert!(
        large_excerpt.size.height > small_excerpt.size.height,
        "excerpt height must scale (11px→{:?}, 18px→{:?})",
        small_excerpt.size,
        large_excerpt.size,
    );
    // Reset back to the default so downstream tests see the baseline theme.
    visual.update(|_, cx| theme::apply(cx));
}
