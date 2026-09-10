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
    // marker may reach the row, and no synthetic canary quad substitutes for
    // the real text. This guard drives real tool rows through the state
    // layer, then verifies the semantic mapping tuple that main.rs consumes.
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
        // State classification lands on the wiki contract's palette map.
        // main.rs::tool_state_color reads the SAME helper, so a regression on
        // either side fails this assertion.
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
        });
        // The verb and detail carry the state color as their own text_color
        // refinement. Their presence proves the row was painted (a shape
        // regression that dropped the verb would leave no bounds to find).
        assert!(
            visual.debug_bounds("tool-verb-0").is_some(),
            "verb element must paint for state {case:?}"
        );
        assert!(
            visual.debug_bounds("tool-detail-0").is_some(),
            "detail element must paint for state {case:?}"
        );
        // Canary is gone: a hidden 1x1 quad would trivialise the color test.
        // Any regression that reintroduced one — under this or the earlier
        // marker names — would trip here.
        for stale in [
            "tool-state-canary-0",
            "tool-state-label-0",
            "tool-state-marker-0",
            "tool-marker-0",
        ] {
            assert!(
                visual.debug_bounds(stale).is_none(),
                "removed state chrome resurfaced under selector {stale}"
            );
        }
        // No textual state marker in any visible tool row content — the
        // transcript entry's rendered fields (name + summary) come from the
        // server, and neither must contain "[working]/[done]/[failed]/
        // [canceled]" because the display layer no longer synthesises them.
        view.read_with(&visual, |view, _| {
            if let TranscriptEntry::Tool { name, summary, .. } = &view.state.transcript[0] {
                for marker in ["[working]", "[done]", "[failed]", "[canceled]"] {
                    assert!(
                        !name.contains(marker),
                        "tool name must not carry state marker {marker}"
                    );
                    assert!(
                        !summary.contains(marker),
                        "tool summary must not carry state marker {marker}"
                    );
                }
            } else {
                panic!("expected a Tool entry for state {case:?}");
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
            // Thinking now paints a display-safe placeholder row — never the
            // private reasoning text, which the final assertion of this test
            // still guards against below.
            assert!(matches!(
                view.state.transcript.as_slice(),
                [TranscriptEntry::Thinking { title, body, .. }] if title.is_empty() && body.is_empty()
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
fn thinking_row_renders_display_safe_summary_and_hides_private_deltas(cx: &mut TestAppContext) {
    // Guard for contract line 83's thinking chrome. Private streamed deltas
    // must never surface in the row's title/body; only the finalized
    // Thinking block from the server's AssistantMessage supplies displayable
    // content. Once populated, the header shows `+ Thought: <title>` plus a
    // `· <duration>` segment when known, and the expanded body indents 2ch
    // with no bg/border chrome around it.
    use zeta_gui::client::{ContentBlock, Message};
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    let secret = "SECRET-PRIVATE-DELTA-NEVER-DISPLAY";
    let displayable = "Chose read over grep\nBecause the file is small";
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
                    delta: secret.into(),
                    kind: "thinking".into(),
                }),
                window,
                cx,
            );
            // Deltas seed only the placeholder — title/body remain empty so
            // the private text never reaches the display state.
            assert!(matches!(
                &view.state.transcript[0],
                TranscriptEntry::Thinking { title, body, .. }
                    if title.is_empty() && body.is_empty()
            ));
            view.apply_worker_message(
                WorkerMessage::Event(ServerEvent::AssistantMessage {
                    session_id: None,
                    message: Message {
                        role: "assistant".into(),
                        content: vec![ContentBlock::Thinking {
                            text: displayable.into(),
                        }],
                    },
                }),
                window,
                cx,
            );
        });
        window.draw(cx).clear(cx);
    });
    // Finalized message populated title/body from the displayable source.
    view.read_with(&visual, |view, _| {
        let TranscriptEntry::Thinking {
            title,
            body,
            expanded,
            ..
        } = &view.state.transcript[0]
        else {
            panic!("thinking entry missing after AssistantMessage");
        };
        assert_eq!(title, "Chose read over grep");
        assert_eq!(body, displayable);
        assert!(!*expanded, "thinking rows start collapsed");
        // The private delta must never appear in visible fields.
        assert!(!title.contains(secret));
        assert!(!body.contains(secret));
        assert!(!format!("{:?}", view.state.transcript).contains(secret));
    });
    // Collapsed: header paints, body does not. Duration segment is absent
    // when duration_ms is None.
    assert!(visual.debug_bounds("thinking-header-0").is_some());
    assert!(visual.debug_bounds("thinking-body-0").is_none());
    assert!(visual.debug_bounds("thinking-duration-0").is_none());
    // Expand + set a duration; header now includes the duration span and the
    // body renders indented at 2ch (≈ THINKING_BODY_INDENT) with no chrome.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            if let Some(TranscriptEntry::Thinking {
                expanded,
                duration_ms,
                ..
            }) = view.state.transcript.get_mut(0)
            {
                *expanded = true;
                *duration_ms = Some(463);
            }
            view.transcript.update(cx, |scroll, cx| scroll.reset(1, cx));
            cx.notify();
        });
        window.draw(cx).clear(cx);
    });
    assert!(visual.debug_bounds("thinking-duration-0").is_some());
    let header = visual.debug_bounds("thinking-header-0").unwrap();
    let body = visual
        .debug_bounds("thinking-body-0")
        .expect("expanded thinking body renders");
    // Body sits BELOW the header (stacked) and is INDENTED — its left edge
    // starts to the right of the header's left edge by the 2ch indent
    // (allow a small paint tolerance).
    assert!(
        body.top() >= header.bottom() - px(0.5),
        "body must sit below the header"
    );
    let indent = body.left() - header.left();
    assert!(
        indent >= theme::THINKING_BODY_INDENT - px(1.5)
            && indent <= theme::THINKING_BODY_INDENT + px(1.5),
        "expanded body indent {indent:?} must land on 2ch ({:?})",
        theme::THINKING_BODY_INDENT
    );
    // The expanded body must carry no bg/border chrome — the header sits
    // muted, the body is the only content — so any framing quad inside the
    // body's bounds is a regression.
    visual.update(|window, cx| {
        let theme = cx.theme();
        let scaled_body = body.scale(window.scale_factor());
        let framed: Vec<_> = window
            .painted_quads()
            .into_iter()
            .filter(|quad| {
                let in_body = quad.bounds.top() >= scaled_body.top()
                    && quad.bounds.bottom() <= scaled_body.bottom()
                    && quad.bounds.left() >= scaled_body.left()
                    && quad.bounds.right() <= scaled_body.right();
                let has_border = quad.border_widths.left > gpui::ScaledPixels::default()
                    || quad.border_widths.top > gpui::ScaledPixels::default()
                    || quad.border_widths.right > gpui::ScaledPixels::default()
                    || quad.border_widths.bottom > gpui::ScaledPixels::default();
                let has_fill = quad.background == theme.muted.into()
                    || quad.background == theme.sidebar.into();
                in_body && (has_border || has_fill)
            })
            .collect();
        assert!(
            framed.is_empty(),
            "expanded thinking body must have no chrome (got {} framing quads)",
            framed.len()
        );
    });
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
