use super::*;
use gpui::{TestAppContext, VisualTestContext, WindowHandle};
use gpui_kit::component::Theme;
use serde_json::json;
use std::sync::mpsc::Receiver;
use zeta_gui::client::{ModelCatalog, ServerEvent, SessionMetadata, StatusResult, ToolCall};
use zeta_gui::session::{Branch, ImageAttachment, SessionSettings};

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
    cx.update(init);
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

#[gpui::test]
fn transcript_column_caps_at_wiki_readable_measure_and_centers(cx: &mut TestAppContext) {
    // Wiki agent-run column pins at 1024px centered. The default 1100px window
    // minus the 216px sidebar leaves ~884px of transcript viewport — narrower
    // than the column cap, which means the centering branch never runs. Resize
    // to a wide window here so the cap and centering are both exercised.
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
    assert!(row.size.width <= transcript.size.width);
    // The transcript viewport must clear the column cap, otherwise this test
    // regresses to the old "cap never activates" hole.
    assert!(
        transcript.size.width > theme::TRANSCRIPT_MAX_WIDTH,
        "viewport {:?} must exceed the 1024px cap for centering to matter",
        transcript.size.width
    );
    visual.update(|window, cx| {
        let scale = window.scale_factor();
        let scaled_viewport = transcript.scale(scale);
        let scaled_row = row.scale(scale);
        let scaled_column_cap = px(f32::from(theme::TRANSCRIPT_MAX_WIDTH)).scale(scale);
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
        // The user rectangle sits inside a bounded inner column: 1024px minus
        // 16px horizontal padding on each side (`.px_4()`). The rectangle's
        // quad bounds are its border-box, so a 3px left-rail adds up to 3px
        // to the observed width — allow that plus a sub-logical-pixel wiggle.
        let inner_column = scaled_column_cap - px(32.).scale(scale);
        let tolerance = px(4.).scale(scale);
        for quad in user_quads {
            let width = quad.bounds.size.width;
            let delta = if width > inner_column {
                width - inner_column
            } else {
                inner_column - width
            };
            assert!(
                delta <= tolerance,
                "user rectangle width {:?} must land on the inner 1024-32px column {:?}",
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
            let expected_ids: std::collections::HashSet<&str> =
                ["tool-verb-0", "tool-detail-0", "tool-chevron-0"]
                    .into_iter()
                    .collect();
            assert_eq!(
                recorded, expected_ids,
                "render_tool_row must record verb, detail, and chevron samples for {case:?}"
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
        // The verb and detail elements paint their bounds — the color check
        // above proves the contract token is on the entry; this pins the
        // debug selectors so a rename regresses.
        assert!(
            visual.debug_bounds("tool-verb-0").is_some(),
            "verb element must paint for state {case:?}"
        );
        assert!(
            visual.debug_bounds("tool-detail-0").is_some(),
            "detail element must paint for state {case:?}"
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
    // Prime the clipboard with a valid PNG stub; the paste hook adds the chip.
    visual.update(|_, cx| {
        cx.write_to_clipboard(gpui::ClipboardItem::new_image(&gpui::Image {
            format: gpui::ImageFormat::Png,
            bytes: png_bytes(),
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
        assert_eq!(view.composer_images.len(), 1);
        assert_eq!(view.composer_images[0].name, "pasted-image.png");
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
            let images = view.composer_images.clone();
            view.apply_worker_message(WorkerMessage::ImagesSent(String::new(), images), window, cx);
        });
        window.draw(cx).clear(cx);
    });
    view.read_with(&visual, |view, _| {
        assert!(view.composer_images.is_empty());
        assert_eq!(
            view.state.session_view.attachments.get(&0),
            Some(&vec![("pasted-image.png".to_string(), 8)])
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
fn attachment_validation_error_renders_and_clears_on_a_good_image(cx: &mut TestAppContext) {
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.add_attached_images(
                ImageAttachment::from_bytes("bad.png".into(), b"not a real png")
                    .map(|image| vec![image]),
                cx,
            );
        });
        window.draw(cx).clear(cx);
    });
    view.read_with(&visual, |view, _| {
        assert!(view.composer_image_error.is_some());
        assert!(view.composer_images.is_empty());
    });
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.add_attached_images(
                ImageAttachment::from_bytes("good.png".into(), &png_bytes())
                    .map(|image| vec![image]),
                cx,
            );
        });
        window.draw(cx).clear(cx);
    });
    view.read_with(&visual, |view, _| {
        assert!(view.composer_image_error.is_none());
        assert_eq!(view.composer_images.len(), 1);
    });
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
        assert!(view.composer_images.is_empty());
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
                view.composer_images =
                    vec![ImageAttachment::from_bytes("test.png".into(), &png_bytes()).unwrap()];
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
                assert_eq!(view.composer_images.len(), 1);
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
                assert!(view.composer_images.is_empty());
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
            png_bytes(),
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
            assert_eq!(view.composer_images.len(), 4);
            assert!(view.composer_image_error.is_some());
            view.composer_images.clear();
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
    assert_eq!(
        polish::status_label(&metrics),
        "model · 12 tokens · 50.0% cache"
    );
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
        "model · Usage appears after the first turn"
    );
    status.usage = json!({
        "input_tokens": 4, "output_tokens": 4, "cache_read_input_tokens": 4
    });
    state.apply_status(status);
    assert_eq!(
        polish::status_label(&state.metrics),
        "model · 12 tokens · 50.0% cache"
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
fn refused_image_paste_falls_through_to_clipboard_text(cx: &mut TestAppContext) {
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|_, cx| {
        cx.write_to_clipboard(gpui::ClipboardItem {
            entries: vec![
                gpui::ClipboardEntry::Image(gpui::Image::from_bytes(
                    gpui::ImageFormat::Png,
                    png_bytes(),
                )),
                gpui::ClipboardEntry::String(gpui::ClipboardString::new("fallback text".into())),
            ],
        });
    });
    visual.simulate_keystrokes("cmd-v");
    view.read_with(&visual, |view, cx| {
        assert_eq!(view.composer_images.len(), 1);
        assert!(view.composer.read(cx).value().is_empty());
    });
    visual.update(|_, cx| {
        view.update(cx, |view, _| {
            view.composer_images = vec![view.composer_images[0].clone(); 4]
        });
    });
    visual.simulate_keystrokes("cmd-v");
    view.read_with(&visual, |view, cx| {
        assert_eq!(view.composer_images.len(), 4);
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
fn status_pill_paints_a_solid_fill_and_flips_to_danger_when_offline(cx: &mut TestAppContext) {
    // Contract line 83: neutral state = solid accent, negative = solid
    // danger. The pill carries the single load-bearing color on the strip;
    // a regression that dropped the fill back to text-tone would erase the
    // wiki-run recognisability.
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let pill = visual
        .debug_bounds("footer-mode")
        .expect("footer mode pill renders");
    visual.update(|window, cx| {
        let theme = cx.theme();
        let scaled_pill = pill.scale(window.scale_factor());
        let neutral = window.painted_quads().into_iter().find(|quad| {
            quad.background == theme.primary.into()
                && quad.bounds.top() >= scaled_pill.top() - px(1.).scale(window.scale_factor())
                && quad.bounds.bottom()
                    <= scaled_pill.bottom() + px(1.).scale(window.scale_factor())
        });
        assert!(neutral.is_some(), "neutral pill paints a solid accent fill");
    });
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.apply_worker_message(WorkerMessage::Lost("network gone".into()), window, cx);
        });
        window.draw(cx).clear(cx);
    });
    let pill = visual
        .debug_bounds("footer-mode")
        .expect("footer mode pill renders offline");
    visual.update(|window, cx| {
        let theme = cx.theme();
        let scaled_pill = pill.scale(window.scale_factor());
        let danger = window.painted_quads().into_iter().find(|quad| {
            quad.background == theme.danger.into()
                && quad.bounds.top() >= scaled_pill.top() - px(1.).scale(window.scale_factor())
                && quad.bounds.bottom()
                    <= scaled_pill.bottom() + px(1.).scale(window.scale_factor())
        });
        assert!(
            danger.is_some(),
            "offline mode paints the negative pill in danger"
        );
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
fn footer_status_strip_paints_vertical_rules_between_metadata(cx: &mut TestAppContext) {
    // The wiki header pattern rules adjacent metadata with 1x14 vertical
    // separators. Two rules ride between the pill, the metrics label, and
    // the hint — a regression that dropped them would fuse the strip into
    // one uniform run.
    let (window, _view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let bar = visual
        .debug_bounds("status-bar")
        .expect("status bar renders");
    visual.update(|window, cx| {
        let theme = cx.theme();
        let scaled_bar = bar.scale(window.scale_factor());
        let rules: Vec<_> = window
            .painted_quads()
            .into_iter()
            .filter(|quad| {
                quad.background == theme.border.into()
                    && quad.bounds.top() >= scaled_bar.top()
                    && quad.bounds.bottom() <= scaled_bar.bottom()
                    && quad.bounds.size.width <= px(2.).scale(window.scale_factor())
            })
            .collect();
        assert!(
            rules.len() >= 2,
            "expected two vertical rules on the status strip, saw {}",
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
fn run_header_paints_two_stacked_bands(cx: &mut TestAppContext) {
    // Contract line 83 pins the run header at two stacked bands — band 1
    // at 44px with the session label + state pill + step, band 2 at 40px
    // with the metrics + rules. A regression that dropped either band
    // (or merged them into one) would drift the header height and hide
    // one signal.
    let (window, _view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let band1 = visual
        .debug_bounds("run-header-band1")
        .expect("run header band 1 renders");
    // Band 2 keeps the "status-bar" selector for backwards compatibility
    // with the existing status-strip guard so both share one paint probe.
    let band2 = visual
        .debug_bounds("status-bar")
        .expect("run header band 2 (status-bar) renders");
    // Bands stack — band 2 sits directly below band 1.
    assert!(
        band2.top() >= band1.bottom() - px(2.),
        "band 2 must sit below band 1 (band1.bottom={:?}, band2.top={:?})",
        band1.bottom(),
        band2.top()
    );
    // Heights land on the contract floors within one logical pixel.
    let h1_delta = if band1.size.height > theme::HEADER_BAND1_MIN_HEIGHT {
        band1.size.height - theme::HEADER_BAND1_MIN_HEIGHT
    } else {
        theme::HEADER_BAND1_MIN_HEIGHT - band1.size.height
    };
    assert!(
        h1_delta <= px(2.),
        "band 1 height {:?} must land on the 44px floor",
        band1.size.height
    );
    let h2_delta = if band2.size.height > theme::HEADER_BAND2_MIN_HEIGHT {
        band2.size.height - theme::HEADER_BAND2_MIN_HEIGHT
    } else {
        theme::HEADER_BAND2_MIN_HEIGHT - band2.size.height
    };
    assert!(
        h2_delta <= px(2.),
        "band 2 height {:?} must land on the 40px floor",
        band2.size.height
    );
    // Band 1 carries the state pill and the step — both must render.
    assert!(
        visual.debug_bounds("footer-mode").is_some(),
        "band 1 must render the state pill"
    );
    assert!(
        visual.debug_bounds("run-header-step").is_some(),
        "band 1 must render the step text"
    );
    // Band 2 carries the metrics rules.
    assert!(
        visual.debug_bounds("run-header-model").is_some(),
        "band 2 must render the model chip"
    );
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

/// Renderer-literal fence — the ZETA-109 static guard.
///
/// The typed row-text model (`row_text::RowText`) is the sole source of
/// every user-visible string a transcript row paints. The fence proves it
/// stays that way by parsing `main.rs` on every run, extracting each
/// targeted `fn` body via a hand-rolled Rust tokenizer, and rejecting
/// every inline string literal that does not match the tight widget-ID
/// allowlist (or sit inside a diagnostic macro).
///
/// Negative fixtures the fence rejects — DO NOT paste any of these into
/// a renderer body; they exist in this comment as documentation:
///
///     .child("[done]")                    // bracketed state marker
///     .child("Send message")              // uppercase prose
///     .child("show details")              // has a space
///     .child("Try again")                 // uppercase + space
///     .label("Cancel")                    // capitalized action label
///     .child(format!("You have {n}"))     // format string is prose
///
/// Positive fixtures the fence accepts:
///
///     .debug_selector(|| "transcript-row".into())
///     .debug_selector(move || format!("tool-verb-{index}"))
///     Button::new(("fork", index))
///     format!("error-login-{index}")
///     unreachable!("row-inner-entry-mismatch")   // diagnostic macro escape
///
/// Rules:
///
/// * The fence walks the SIX transcript render functions by exact
///   signature: `render_row_inner`, `render_thinking_row`,
///   `render_user_row`, `render_assistant_row`, `render_tool_row`,
///   `render_error_row`. The set is spelled out below so a rename or
///   split fails loudly instead of silently dropping a fn from the scan.
///
/// * Bodies are extracted by hand-rolled Rust tokenizer that skips line
///   and (nested) block comments, char literals, byte strings, and both
///   normal and raw string literals — a comment-marker parser was
///   rejected by the ZETA-107 r7 reviewer for being bypassable by moving
///   a fn or renaming markers. The tokenizer keys off Rust's actual
///   syntax, not documentation cues.
///
/// * String literals inside diagnostic macros
///   (`unreachable!/panic!/todo!/unimplemented!/assert{,_eq,_ne}!/
///   debug_assert{,_eq,_ne}!`) are allowed — those payloads never reach
///   the user. Every OTHER literal must match the widget-ID allowlist:
///   empty, OR first char ASCII lowercase, remaining chars in
///   `[a-z0-9_-]` with at most one trailing `-{ident}` format placeholder
///   where `ident` is `[a-z_][a-z0-9_]*`. Everything else — spaces,
///   uppercase, punctuation, prose — fails.
///
/// * The chrome-coverage arm proves `row_text::chrome::ALL` mentions
///   every constant declared in the `chrome` module. Adding a new
///   chrome constant without adding it to `ALL` regresses the seam
///   tests that iterate `ALL`.
#[test]
fn renderer_literal_fence_rejects_user_visible_strings_in_transcript_renderers() {
    const SOURCE: &str = include_str!("main.rs");
    const TARGET_FN_NAMES: &[&str] = &[
        "render_row_inner",
        "render_thinking_row",
        "render_user_row",
        "render_assistant_row",
        "render_tool_row",
        "render_error_row",
    ];

    let mut failures: Vec<String> = Vec::new();
    for name in TARGET_FN_NAMES {
        let (body_start, body_end) = fence::locate_fn_body(SOURCE, name).unwrap_or_else(|err| {
            panic!("fence: locating `{name}` failed: {err}. If the fn was renamed, update TARGET_FN_NAMES.")
        });
        let body = &SOURCE[body_start..body_end];
        let literals = fence::collect_string_literals(body)
            .unwrap_or_else(|err| panic!("fence: tokenizing `{name}` body failed: {err}"));
        for literal in &literals {
            if literal.in_diagnostic_macro {
                continue;
            }
            if fence::is_widget_id_shape(&literal.content) {
                continue;
            }
            failures.push(format!(
                "fn {name}: forbidden inline literal {:?} at byte offset {} \
                 inside the fn body — route this string through the RowText \
                 model or the row_text::chrome constants module.",
                literal.content, literal.offset,
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "renderer_literal_fence tripped:\n  - {}",
        failures.join("\n  - "),
    );
}

#[test]
fn renderer_literal_fence_widget_id_shape_accepts_ids_and_rejects_prose() {
    for id in [
        "",
        "fork",
        "attachment-chip",
        "transcript-row",
        "user-row-{index}",
        "tool-receipt-{index}",
        "error-login-{index}",
        "tool-verb-{index}",
    ] {
        assert!(
            fence::is_widget_id_shape(id),
            "widget-ID shape must accept {id:?}"
        );
    }
    for prose in [
        "Fork here",
        "Open Settings",
        "Cancel",
        "Send",
        "show output",
        "Earlier output omitted",
        "[done]",
        "[working]",
        "{index}",
        "-leading-hyphen",
        "You have {n} messages",
        "tool row",
        "Tool-Row",
    ] {
        assert!(
            !fence::is_widget_id_shape(prose),
            "widget-ID shape must reject {prose:?}"
        );
    }
}

#[test]
fn renderer_literal_fence_tokenizer_ignores_comments_and_diagnostic_macros() {
    // Mutation guard: prove the tokenizer skips comments, char literals,
    // and raw strings, and that literals inside diagnostic macros are
    // marked as such. A regression in the tokenizer that fails to skip
    // a `//` line comment would surface here.
    let sample = r##"
        // "line-comment-literal"
        /* "block-comment-literal" /* nested */ */
        let a = '\n';
        let b = r#""raw-inside""#;
        panic!("panic message body");
        div().child("Prose sneaks in");
        format!("tool-verb-{index}");
    "##;
    let literals = fence::collect_string_literals(sample).expect("tokenize");
    let seen: Vec<(&str, bool)> = literals
        .iter()
        .map(|lit| (lit.content.as_str(), lit.in_diagnostic_macro))
        .collect();
    assert_eq!(
        seen,
        vec![
            ("\"raw-inside\"", false),
            ("panic message body", true),
            ("Prose sneaks in", false),
            ("tool-verb-{index}", false),
        ],
        "tokenizer must skip comments and char literals, keep raw strings, \
         and mark panic!-argument as diagnostic. Got: {seen:?}"
    );
}

#[test]
fn renderer_literal_fence_chrome_all_lists_every_chrome_constant() {
    // Chrome-coverage arm: parse the row_text.rs chrome module and prove
    // every `pub const NAME: &str = "..."` appears in `chrome::ALL`. A new
    // chrome constant without an ALL entry silently escapes the row-text
    // seam sweep — this test flags that regression.
    const CHROME_SOURCE: &str = include_str!("row_text.rs");
    let declared = fence::chrome_constants(CHROME_SOURCE);
    let all_names = fence::chrome_all_entries(CHROME_SOURCE);
    assert!(
        !declared.is_empty(),
        "fence: chrome module scan returned zero constants — parser regressed?"
    );
    for name in &declared {
        assert!(
            all_names.contains(name),
            "chrome constant {name} is declared but missing from chrome::ALL — \
             add it so the seam sweep and the fence pick it up"
        );
    }
}

mod fence {
    //! Fence internals: hand-rolled Rust tokenizer used only by the
    //! renderer-literal fence tests. Kept in a submodule so the tests
    //! remain readable and the tokenizer stays testable in isolation.

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(super) struct Literal {
        pub content: String,
        pub offset: usize,
        pub in_diagnostic_macro: bool,
    }

    const DIAGNOSTIC_MACROS: &[&str] = &[
        "unreachable",
        "panic",
        "todo",
        "unimplemented",
        "assert",
        "assert_eq",
        "assert_ne",
        "debug_assert",
        "debug_assert_eq",
        "debug_assert_ne",
    ];

    /// Widget-ID allowlist. Empty string is allowed; otherwise the string
    /// must be `[a-z][a-z0-9_-]*(-\{[a-z_][a-z0-9_]*\})?`.
    pub(super) fn is_widget_id_shape(s: &str) -> bool {
        if s.is_empty() {
            return true;
        }
        let bytes = s.as_bytes();
        if !bytes[0].is_ascii_lowercase() {
            return false;
        }
        let mut i = 0;
        // Kebab-lowercase prefix.
        while i < bytes.len() {
            match bytes[i] {
                b'a'..=b'z' | b'0'..=b'9' | b'_' | b'-' => i += 1,
                b'{' => break,
                _ => return false,
            }
        }
        if i == bytes.len() {
            // Must not end on `-` or `_` — but that's cosmetic; allow.
            return bytes[i - 1] != b'-';
        }
        // Must be `-{ident}` at end.
        if bytes[i] != b'{' || i == 0 || bytes[i - 1] != b'-' {
            return false;
        }
        i += 1;
        // First char of ident: [a-z_]
        let ident_start = i;
        if i >= bytes.len() || !matches!(bytes[i], b'a'..=b'z' | b'_') {
            return false;
        }
        i += 1;
        while i < bytes.len() && matches!(bytes[i], b'a'..=b'z' | b'0'..=b'9' | b'_') {
            i += 1;
        }
        if i == ident_start {
            return false;
        }
        if i >= bytes.len() || bytes[i] != b'}' {
            return false;
        }
        i += 1;
        i == bytes.len()
    }

    /// Locate the byte range of a fn's body — the inclusive-start /
    /// exclusive-end offsets of everything between the opening `{` (after
    /// the signature) and the matching closing `}`. Errors if the fn
    /// isn't found, if it appears more than once (rename ambiguity), or
    /// if the body is unterminated.
    pub(super) fn locate_fn_body(source: &str, fn_name: &str) -> Result<(usize, usize), String> {
        let needle = format!("fn {fn_name}(");
        let mut occurrences: Vec<usize> = Vec::new();
        let bytes = source.as_bytes();
        let needle_bytes = needle.as_bytes();
        let mut i = 0;
        while i + needle_bytes.len() <= bytes.len() {
            if &bytes[i..i + needle_bytes.len()] == needle_bytes {
                // Guard: the byte before must be non-alphanumeric so we
                // don't match `fn render_thinking_row_v2(`.
                let starts_word = i == 0
                    || !matches!(bytes[i - 1], b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_');
                if starts_word {
                    occurrences.push(i);
                }
                i += needle_bytes.len();
            } else {
                i += 1;
            }
        }
        match occurrences.len() {
            0 => return Err(format!("`{fn_name}` not found")),
            1 => {}
            n => return Err(format!("`{fn_name}` matched {n} times (rename ambiguity)")),
        }
        let sig_start = occurrences[0];
        // From sig_start, find the first `{` at paren-depth zero AFTER the
        // matching `)` of the signature. Rust tokenizer-lite.
        let mut cursor = sig_start;
        let mut paren_depth = 0i32;
        let mut in_signature = true;
        while cursor < bytes.len() {
            let (new_cursor, event) = advance_token(bytes, cursor)?;
            cursor = new_cursor;
            match event {
                TokenEvent::None => {}
                TokenEvent::Open(b'(') => paren_depth += 1,
                TokenEvent::Close(b')') => paren_depth -= 1,
                TokenEvent::Open(b'{') if in_signature && paren_depth == 0 => {
                    // Body starts.
                    let body_start = cursor; // cursor is now just past the `{`
                    let mut brace_depth = 1i32;
                    let mut inner = cursor;
                    while inner < bytes.len() {
                        let (next, event) = advance_token(bytes, inner)?;
                        inner = next;
                        match event {
                            TokenEvent::Open(b'{') => brace_depth += 1,
                            TokenEvent::Close(b'}') => {
                                brace_depth -= 1;
                                if brace_depth == 0 {
                                    let body_end = inner - 1; // exclude the `}`
                                    return Ok((body_start, body_end));
                                }
                            }
                            _ => {}
                        }
                    }
                    return Err(format!("`{fn_name}` body unterminated"));
                }
                _ => {
                    in_signature = paren_depth != 0 || in_signature;
                }
            }
        }
        Err(format!("`{fn_name}` body not found after signature"))
    }

    /// Collect every string literal in `body`, recording its content,
    /// byte offset (within `body`), and whether it sits inside a
    /// diagnostic macro's argument list.
    pub(super) fn collect_string_literals(body: &str) -> Result<Vec<Literal>, String> {
        let bytes = body.as_bytes();
        let mut i = 0;
        let mut literals: Vec<Literal> = Vec::new();
        let mut macro_stack: Vec<bool> = Vec::new(); // one entry per open bracket
        while i < bytes.len() {
            // Comment and literal-body skip is handled by advance_token; we
            // need a variant that ALSO surfaces string literals to us.
            if starts_with(bytes, i, b"//") {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
                continue;
            }
            if starts_with(bytes, i, b"/*") {
                let mut depth = 1;
                i += 2;
                while i < bytes.len() && depth > 0 {
                    if starts_with(bytes, i, b"/*") {
                        depth += 1;
                        i += 2;
                    } else if starts_with(bytes, i, b"*/") {
                        depth -= 1;
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                continue;
            }
            // Char literal — heuristic: `'` followed by (\?any)`'`.
            if bytes[i] == b'\'' && likely_char_literal(bytes, i) {
                i += 1;
                if i < bytes.len() && bytes[i] == b'\\' {
                    i += 1;
                    if i < bytes.len() {
                        i += 1;
                    }
                } else if i < bytes.len() {
                    i += 1;
                }
                while i < bytes.len() && bytes[i] != b'\'' {
                    i += 1;
                }
                if i < bytes.len() {
                    i += 1;
                }
                continue;
            }
            // Byte string / raw / normal string detection.
            let (prefix_len, is_raw, is_byte) = scan_string_prefix(bytes, i);
            let (hash_len, expects_string) = if is_raw {
                let mut h = 0;
                let mut cur = i + prefix_len;
                while cur < bytes.len() && bytes[cur] == b'#' {
                    h += 1;
                    cur += 1;
                }
                (h, cur < bytes.len() && bytes[cur] == b'"')
            } else if prefix_len > 0 {
                (
                    0,
                    i + prefix_len < bytes.len() && bytes[i + prefix_len] == b'"',
                )
            } else if bytes[i] == b'"' {
                (0, true)
            } else {
                (0, false)
            };
            if expects_string {
                let content_start = i + prefix_len + hash_len + 1;
                let mut j = content_start;
                if is_raw {
                    // find closing `"` followed by hash_len `#`.
                    loop {
                        if j >= bytes.len() {
                            return Err(format!("unterminated raw string at offset {i}"));
                        }
                        if bytes[j] == b'"' {
                            let mut k = j + 1;
                            let mut matched = 0;
                            while matched < hash_len && k < bytes.len() && bytes[k] == b'#' {
                                matched += 1;
                                k += 1;
                            }
                            if matched == hash_len {
                                let content =
                                    String::from_utf8_lossy(&bytes[content_start..j]).into_owned();
                                let in_diagnostic_macro =
                                    macro_stack.last().copied().unwrap_or(false);
                                if !is_byte {
                                    literals.push(Literal {
                                        content,
                                        offset: i,
                                        in_diagnostic_macro,
                                    });
                                }
                                i = k;
                                break;
                            }
                        }
                        j += 1;
                    }
                    continue;
                } else {
                    while j < bytes.len() {
                        match bytes[j] {
                            b'\\' => {
                                j += 2;
                            }
                            b'"' => break,
                            _ => j += 1,
                        }
                    }
                    if j >= bytes.len() {
                        return Err(format!("unterminated string at offset {i}"));
                    }
                    let content = decode_escapes(&bytes[content_start..j])
                        .map_err(|e| format!("escape decode failed at {i}: {e}"))?;
                    let in_diagnostic_macro = macro_stack.last().copied().unwrap_or(false);
                    if !is_byte {
                        literals.push(Literal {
                            content,
                            offset: i,
                            in_diagnostic_macro,
                        });
                    }
                    i = j + 1;
                    continue;
                }
            }
            // Bracket tracking. When we see `(`, `[`, or `{`, look back for
            // an identifier ending with `!` to mark diagnostic-macro frames.
            match bytes[i] {
                b'(' | b'[' | b'{' => {
                    let is_diag = preceding_identifier(bytes, i)
                        .filter(|(_, bang)| *bang)
                        .map(|(ident, _)| DIAGNOSTIC_MACROS.iter().any(|name| *name == ident))
                        .unwrap_or(false);
                    macro_stack.push(is_diag);
                    i += 1;
                }
                b')' | b']' | b'}' => {
                    macro_stack.pop();
                    i += 1;
                }
                _ => i += 1,
            }
        }
        Ok(literals)
    }

    /// Return (identifier, ended_with_bang) for the identifier immediately
    /// before `pos` in `bytes`, skipping whitespace. If no identifier is
    /// found, returns None.
    fn preceding_identifier(bytes: &[u8], pos: usize) -> Option<(String, bool)> {
        if pos == 0 {
            return None;
        }
        let mut i = pos;
        // Skip whitespace.
        while i > 0 && matches!(bytes[i - 1], b' ' | b'\t' | b'\n' | b'\r') {
            i -= 1;
        }
        // Optional trailing `!`.
        let mut bang = false;
        if i > 0 && bytes[i - 1] == b'!' {
            bang = true;
            i -= 1;
        }
        let ident_end = i;
        while i > 0 && matches!(bytes[i - 1], b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_') {
            i -= 1;
        }
        if i == ident_end {
            return None;
        }
        Some((
            String::from_utf8_lossy(&bytes[i..ident_end]).into_owned(),
            bang,
        ))
    }

    fn starts_with(bytes: &[u8], pos: usize, needle: &[u8]) -> bool {
        bytes.get(pos..pos + needle.len()) == Some(needle)
    }

    fn likely_char_literal(bytes: &[u8], pos: usize) -> bool {
        // Rust lifetime tokens use `'`, e.g. `'a`, `'static`. Distinguish
        // by looking for a closing `'` within a plausible range.
        let mut j = pos + 1;
        if j < bytes.len() && bytes[j] == b'\\' {
            j += 2;
        } else {
            j += 1;
        }
        j < bytes.len() && bytes[j] == b'\''
    }

    fn scan_string_prefix(bytes: &[u8], pos: usize) -> (usize, bool, bool) {
        // Returns (prefix_len, is_raw, is_byte). Handles `r`, `b`, `br`,
        // `rb` prefixes.
        let mut is_raw = false;
        let mut is_byte = false;
        let mut i = pos;
        loop {
            match bytes.get(i) {
                Some(b'r') if !is_raw => {
                    is_raw = true;
                    i += 1;
                }
                Some(b'b') if !is_byte => {
                    is_byte = true;
                    i += 1;
                }
                _ => break,
            }
        }
        (i - pos, is_raw, is_byte)
    }

    fn decode_escapes(bytes: &[u8]) -> Result<String, String> {
        // Best-effort: decode common `\n`, `\t`, `\"`, `\\`, `\'` and Unicode
        // escapes. Fence assertions compare against literal content, so a
        // partial escape decode is fine — as long as we don't lose track of
        // string boundaries, which is handled by the scanner above.
        let mut out = String::new();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'\\' && i + 1 < bytes.len() {
                match bytes[i + 1] {
                    b'n' => out.push('\n'),
                    b't' => out.push('\t'),
                    b'r' => out.push('\r'),
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'\'' => out.push('\''),
                    b'0' => out.push('\0'),
                    b'u' => {
                        // \u{XXXX} — skip through the closing `}`.
                        let mut k = i + 2;
                        while k < bytes.len() && bytes[k] != b'}' {
                            k += 1;
                        }
                        if k < bytes.len() {
                            out.push('?');
                            i = k + 1;
                            continue;
                        }
                    }
                    other => out.push(other as char),
                }
                i += 2;
            } else {
                out.push(bytes[i] as char);
                i += 1;
            }
        }
        Ok(out)
    }

    #[derive(Debug)]
    pub(super) enum TokenEvent {
        None,
        Open(u8),
        Close(u8),
    }

    /// Advance one token from `pos`, returning the new position and any
    /// bracket/paren event of interest. Skips comments, char literals,
    /// and string literals in bulk.
    pub(super) fn advance_token(bytes: &[u8], pos: usize) -> Result<(usize, TokenEvent), String> {
        let mut i = pos;
        if starts_with(bytes, i, b"//") {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            return Ok((i, TokenEvent::None));
        }
        if starts_with(bytes, i, b"/*") {
            let mut depth = 1;
            i += 2;
            while i < bytes.len() && depth > 0 {
                if starts_with(bytes, i, b"/*") {
                    depth += 1;
                    i += 2;
                } else if starts_with(bytes, i, b"*/") {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            return Ok((i, TokenEvent::None));
        }
        if i < bytes.len() && bytes[i] == b'\'' && likely_char_literal(bytes, i) {
            i += 1;
            if i < bytes.len() && bytes[i] == b'\\' {
                i += 2;
            } else if i < bytes.len() {
                i += 1;
            }
            while i < bytes.len() && bytes[i] != b'\'' {
                i += 1;
            }
            if i < bytes.len() {
                i += 1;
            }
            return Ok((i, TokenEvent::None));
        }
        let (prefix_len, is_raw, _is_byte) = scan_string_prefix(bytes, i);
        let starts_string = if is_raw {
            let mut cur = i + prefix_len;
            while cur < bytes.len() && bytes[cur] == b'#' {
                cur += 1;
            }
            cur < bytes.len() && bytes[cur] == b'"'
        } else if prefix_len > 0 {
            i + prefix_len < bytes.len() && bytes[i + prefix_len] == b'"'
        } else if i < bytes.len() {
            bytes[i] == b'"'
        } else {
            false
        };
        if starts_string {
            let mut j = i + prefix_len;
            let mut hashes = 0;
            while j < bytes.len() && bytes[j] == b'#' {
                hashes += 1;
                j += 1;
            }
            j += 1; // past the opening `"`
            if is_raw {
                loop {
                    if j >= bytes.len() {
                        return Err(format!("unterminated raw string at offset {i}"));
                    }
                    if bytes[j] == b'"' {
                        let mut k = j + 1;
                        let mut matched = 0;
                        while matched < hashes && k < bytes.len() && bytes[k] == b'#' {
                            matched += 1;
                            k += 1;
                        }
                        if matched == hashes {
                            return Ok((k, TokenEvent::None));
                        }
                    }
                    j += 1;
                }
            } else {
                while j < bytes.len() {
                    match bytes[j] {
                        b'\\' => {
                            j += 2;
                        }
                        b'"' => break,
                        _ => j += 1,
                    }
                }
                if j >= bytes.len() {
                    return Err(format!("unterminated string at offset {i}"));
                }
                return Ok((j + 1, TokenEvent::None));
            }
        }
        if i < bytes.len() {
            let byte = bytes[i];
            match byte {
                b'(' | b'[' | b'{' => Ok((i + 1, TokenEvent::Open(byte))),
                b')' | b']' | b'}' => Ok((i + 1, TokenEvent::Close(byte))),
                _ => Ok((i + 1, TokenEvent::None)),
            }
        } else {
            Ok((i, TokenEvent::None))
        }
    }

    /// Extract every `pub const NAME: &str = "..."` from the chrome module
    /// in `row_text.rs`. Uses simple line-based parsing; the chrome module
    /// is deliberately kept in that shape so this parser is trivial.
    pub(super) fn chrome_constants(source: &str) -> Vec<String> {
        let mut names = Vec::new();
        let mut in_chrome = false;
        let mut depth = 0i32;
        for line in source.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("pub mod chrome {") {
                in_chrome = true;
                depth = 1;
                continue;
            }
            if !in_chrome {
                continue;
            }
            for ch in trimmed.chars() {
                if ch == '{' {
                    depth += 1;
                } else if ch == '}' {
                    depth -= 1;
                    if depth == 0 {
                        return names;
                    }
                }
            }
            if let Some(rest) = trimmed.strip_prefix("pub const ") {
                if let Some(end) = rest.find(':') {
                    let name = rest[..end].trim().to_string();
                    if name != "ALL" {
                        names.push(name);
                    }
                }
            }
        }
        names
    }

    /// Extract identifiers listed inside `pub const ALL: &[&str] = &[ ... ]`.
    pub(super) fn chrome_all_entries(source: &str) -> Vec<String> {
        let mut collecting = false;
        let mut items = Vec::new();
        for line in source.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("pub const ALL:") {
                collecting = true;
                continue;
            }
            if !collecting {
                continue;
            }
            if trimmed.starts_with("];") {
                break;
            }
            let ident: String = trimmed
                .trim_end_matches(',')
                .chars()
                .take_while(|ch| ch.is_ascii_alphanumeric() || *ch == '_')
                .collect();
            if !ident.is_empty() {
                items.push(ident);
            }
        }
        items
    }
}
