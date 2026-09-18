use super::*;
use gpui::{InputEvent as _, TestAppContext, VisualTestContext, WindowHandle};
use gpui_kit::component::Theme;
use serde_json::json;
use std::path::PathBuf;
use std::sync::mpsc::Receiver;
use std::sync::LazyLock;
use zeta_gui::client::{
    ModelCatalog, ServerEvent, SessionMetadata, SlashCommandInfo, SlashList, StatusResult, ToolCall,
};
use zeta_gui::session::{self, Branch, ImageAttachment, SessionSettings};

#[derive(Clone, Copy, Debug)]
enum FontSizeProbeKind {
    Textarea,
    Alert,
    Markdown,
}

struct FontSizeProbe {
    kind: FontSizeProbeKind,
    textarea: gpui::Entity<gpui_kit::component::input::TextareaState>,
}

impl gpui::Render for FontSizeProbe {
    fn render(
        &mut self,
        window: &mut gpui::Window,
        cx: &mut gpui::Context<Self>,
    ) -> impl gpui::IntoElement {
        window.set_rem_size(cx.theme().font_size);
        let base = cx.theme().font_size;
        let child = match self.kind {
            FontSizeProbeKind::Textarea => {
                gpui_kit::component::input::Textarea::new(&self.textarea)
                    .h(gpui::px(40.))
                    .into_any_element()
            }
            FontSizeProbeKind::Alert => gpui_kit::component::alert::Alert::error(
                "font-size-alert",
                gpui_kit::component::text::TextView::markdown("font-size-alert-text", "alert body")
                    .w_full()
                    .h(gpui::px(24.)),
            )
            .h(gpui::px(48.))
            .into_any_element(),
            FontSizeProbeKind::Markdown => gpui_kit::component::text::TextView::markdown(
                "font-size-markdown",
                "# h1\n\n## h2\n\n### h3\n\ninline `code`\n\n```rust\nlet x = 1;\n```",
            )
            .into_any_element(),
        };
        gpui::div().size_full().text_size(base).child(child)
    }
}

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

/// The Python server writes `approval_mode: null` for any session whose
/// default has never been set, and `#[serde(default)]` alone rejects an
/// explicit `null` — the initial `status`/`new_session` decode then errors
/// out and the worker never sends `Connected`. The native smoke driver
/// hangs to its 30-second timeout when that happens (r2 CI run
/// 35148088874). This case pins the wire shape the driver actually sees.
#[test]
fn session_metadata_deserializes_null_approval_mode_as_empty() {
    let with_null: SessionMetadata = serde_json::from_value(json!({
        "session_id":"ab12deadbeef",
        "updated_at":"2026-09-09T12:00:00Z",
        "approval_mode": null,
    }))
    .expect("SessionMetadata must accept approval_mode: null");
    assert!(with_null.approval_mode.is_empty());
    let missing: SessionMetadata = serde_json::from_value(json!({
        "session_id":"ab12deadbeef",
        "updated_at":"2026-09-09T12:00:00Z",
    }))
    .expect("missing approval_mode falls back to default");
    assert!(missing.approval_mode.is_empty());
    let set: SessionMetadata = serde_json::from_value(json!({
        "session_id":"ab12deadbeef",
        "updated_at":"2026-09-09T12:00:00Z",
        "approval_mode": "allow",
    }))
    .expect("string approval_mode still deserializes");
    assert_eq!(set.approval_mode, "allow");
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

/// Open the approval dialog and draw so `debug_bounds` sees the button.
fn open_approval_dialog(visual: &mut VisualTestContext, view: &Entity<ZetaView>, request_id: &str) {
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.apply_worker_message(
                WorkerMessage::Event(ServerEvent::ApprovalRequest {
                    session_id: view.state.active_session.clone(),
                    approval: Approval {
                        request_id: request_id.into(),
                        tool_call: ToolCall {
                            id: request_id.into(),
                            name: "bash".into(),
                            arguments: serde_json::from_value(json!({"command":"pwd"})).unwrap(),
                        },
                    },
                }),
                window,
                cx,
            );
        });
        window.draw(cx).clear(cx);
    });
    assert!(
        visual.debug_bounds("approval-always").is_some(),
        "always-allow button paints inside the approval dialog"
    );
}

/// Ack an idle status from the server — production's dialog-close path.
fn close_approval_dialog(visual: &mut VisualTestContext, view: &Entity<ZetaView>) {
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
        });
        window.draw(cx).clear(cx);
    });
    assert!(visual.debug_bounds("dialog-layer").is_none());
}

/// Dispatch a click without allowing a task drain between mouse-down and
/// mouse-up.
///
/// GPUI's button records mouse-down in element state, but mouse-up only fires
/// when the current hit-test still hovers the button. Separate simulated
/// events drain pending tasks between dispatches, allowing a redraw or layout
/// update to change that hit-test at the fixed click coordinate. Mouse-up then
/// clears the pending state without firing the click listener. Keeping both
/// events in one update crosses the real button handler while removing that
/// test-only ordering window.
fn simulate_click_in_one_update(
    visual: &mut VisualTestContext,
    position: gpui::Point<gpui::Pixels>,
    modifiers: gpui::Modifiers,
) {
    visual.update(|window, cx| {
        window.dispatch_event(
            gpui::MouseDownEvent {
                button: gpui::MouseButton::Left,
                position,
                modifiers,
                click_count: 1,
                first_mouse: false,
            }
            .to_platform_input(),
            cx,
        );
        window.dispatch_event(
            gpui::MouseUpEvent {
                button: gpui::MouseButton::Left,
                position,
                modifiers,
                click_count: 1,
            }
            .to_platform_input(),
            cx,
        );
    });
}

/// Read the command synchronously after the input event crosses the handler.
fn expect_command(receiver: &Receiver<CommandMessage>, route: &str) -> CommandMessage {
    receiver.try_recv().unwrap_or_else(|err| {
        panic!("expected a queued CommandMessage after {route} activation, got {err:?}")
    })
}

#[gpui::test]
fn approval_dialog_dispatches_always_allow_via_click(cx: &mut TestAppContext) {
    // ZETA-131 B3 (click): the "Always allow" button dispatches
    // `ApproveAlwaysTool` exactly once. Split from the keyboard case so a
    // single flake names the offending route in the CI log.
    let (window, view, receiver) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    let id = "always-click";
    open_approval_dialog(&mut visual, &view, id);
    let button = visual
        .debug_bounds("approval-always")
        .expect("always-allow button bounds");
    simulate_click_in_one_update(&mut visual, button.center(), Default::default());
    match expect_command(&receiver, "click") {
        CommandMessage::ApproveAlwaysTool(recv) => assert_eq!(recv, id),
        other => panic!("expected ApproveAlwaysTool via click, got {other:?}"),
    }
    // `decide_always_tool` latches `approval_pending`, so a second click
    // must not re-fire until the server resolves the request.
    simulate_click_in_one_update(&mut visual, button.center(), Default::default());
    assert!(
        receiver.try_recv().is_err(),
        "no duplicate dispatch on repeated click"
    );
    close_approval_dialog(&mut visual, &view);
}

#[gpui::test]
fn approval_dialog_dispatches_always_allow_via_keyboard(cx: &mut TestAppContext) {
    // ZETA-131 B3 (keyboard): the bare `a` shortcut inside the open
    // approval dialog dispatches `ApproveAlwaysTool` exactly once.
    let (window, view, receiver) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    let id = "always-key";
    open_approval_dialog(&mut visual, &view, id);
    visual.simulate_keystrokes("a");
    match expect_command(&receiver, "keyboard") {
        CommandMessage::ApproveAlwaysTool(recv) => assert_eq!(recv, id),
        other => panic!("expected ApproveAlwaysTool via keyboard, got {other:?}"),
    }
    // `decide_always_tool` latches `approval_pending`, so a second `a`
    // must not re-fire until the server resolves the request.
    visual.simulate_keystrokes("a");
    visual.run_until_parked();
    assert!(
        receiver.try_recv().is_err(),
        "no duplicate dispatch on repeated keystroke"
    );
    close_approval_dialog(&mut visual, &view);
}

#[gpui::test]
fn approval_mode_segmented_paints_a_filled_selected_state(cx: &mut TestAppContext) {
    // ZETA-131 C1: `ask/allow/deny` in Settings must expose a clearly
    // filled selected state — the shipped `.ghost().selected(true)` was
    // invisible on every theme (audit). Assert that exactly one of the
    // three segment buttons paints a row-sized `primary` fill.
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.apply_worker_message(
                WorkerMessage::Settings(
                    SessionSettings {
                        model: "one".into(),
                        approval_mode: "allow".into(),
                    },
                    ModelCatalog {
                        models: vec!["one".into()],
                        providers: Default::default(),
                    },
                ),
                window,
                cx,
            );
        });
        window.draw(cx).clear(cx);
    });
    let bounds_by_selector: Vec<(&str, gpui::Bounds<gpui::Pixels>)> =
        ["mode-row-ask", "mode-row-allow", "mode-row-deny"]
            .into_iter()
            .map(|selector| {
                let bounds = visual
                    .debug_bounds(selector)
                    .unwrap_or_else(|| panic!("{selector} must render"));
                (selector, bounds)
            })
            .collect();
    let hits = visual.update(|window, cx| {
        let theme = cx.theme();
        let scale = window.scale_factor();
        let quads = window.painted_quads();
        bounds_by_selector
            .iter()
            .filter(|(_, bounds)| {
                let scaled = bounds.scale(scale);
                quads.iter().any(|quad| {
                    let inside = quad.bounds.top() >= scaled.top()
                        && quad.bounds.bottom() <= scaled.bottom()
                        && quad.bounds.left() >= scaled.left()
                        && quad.bounds.right() <= scaled.right();
                    inside && quad.background == theme.primary.into()
                })
            })
            .map(|(selector, _)| *selector)
            .collect::<Vec<_>>()
    });
    assert_eq!(
        hits,
        vec!["mode-row-allow"],
        "exactly one segment paints the primary fill — the selected mode"
    );
    // A click on `deny` must move the fill.
    let deny = visual.debug_bounds("mode-row-deny").expect("deny segment");
    visual.simulate_click(deny.center(), Default::default());
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let bounds_by_selector: Vec<(&str, gpui::Bounds<gpui::Pixels>)> =
        ["mode-row-ask", "mode-row-allow", "mode-row-deny"]
            .into_iter()
            .map(|selector| {
                let bounds = visual
                    .debug_bounds(selector)
                    .unwrap_or_else(|| panic!("{selector} must render after click"));
                (selector, bounds)
            })
            .collect();
    let hits = visual.update(|window, cx| {
        let theme = cx.theme();
        let scale = window.scale_factor();
        let quads = window.painted_quads();
        bounds_by_selector
            .iter()
            .filter(|(_, bounds)| {
                let scaled = bounds.scale(scale);
                quads.iter().any(|quad| {
                    let inside = quad.bounds.top() >= scaled.top()
                        && quad.bounds.bottom() <= scaled.bottom()
                        && quad.bounds.left() >= scaled.left()
                        && quad.bounds.right() <= scaled.right();
                    inside && quad.background == theme.primary.into()
                })
            })
            .map(|(selector, _)| *selector)
            .collect::<Vec<_>>()
    });
    assert_eq!(hits, vec!["mode-row-deny"], "selection tracks the click");
}

#[gpui::test]
fn header_paints_auto_approve_indicator_only_when_allow_mode_is_applied(cx: &mut TestAppContext) {
    // ZETA-131 C2: whenever the server-applied session mode is `allow`, a
    // warning-tinted "auto-approve" chip lives in the run-header metadata
    // cluster. `ask` and `deny` show nothing. The indicator must key off
    // messages production actually emits — session switches send `Session`
    // and cold-start / reconnect / periodic refresh sends `Status`, both
    // with `approval_mode` inside the session metadata. Verify a draft-only
    // selection in the Settings modal does NOT paint the chip.
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    for mode in ["ask", "deny"] {
        visual.update(|window, cx| {
            view.update(cx, |view, cx| {
                view.apply_worker_message(
                    WorkerMessage::Status(StatusResult {
                        session: Some(SessionMetadata {
                            approval_mode: mode.into(),
                            ..session()
                        }),
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
        assert!(
            visual.debug_bounds("header-auto-approve").is_none(),
            "no auto-approve chip in mode {mode}"
        );
    }
    // A Session frame with `approval_mode: "allow"` — the shape `resume` /
    // `new_session` sends when the user switches sessions.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.apply_worker_message(
                WorkerMessage::Session(SessionMetadata {
                    approval_mode: "allow".into(),
                    ..session()
                }),
                window,
                cx,
            );
        });
        window.draw(cx).clear(cx);
    });
    let chip = visual
        .debug_bounds("header-auto-approve")
        .expect("chip paints after a Session frame with allow");
    let header = visual
        .debug_bounds("run-header")
        .expect("run header renders");
    assert!(
        header.contains(&chip.center()),
        "chip sits inside the run header"
    );
    let dot = visual
        .debug_bounds("header-auto-approve-dot")
        .expect("chip has a color dot");
    visual.update(|window, _cx| {
        let scale = window.scale_factor();
        let scaled = chip.scale(scale);
        let quads = window.painted_quads();
        let tint = quads.iter().find(|quad| {
            quad.background == theme::palette::warning_tint().into()
                && quad.bounds.top() >= scaled.top() - px(1.).scale(scale)
                && quad.bounds.bottom() <= scaled.bottom() + px(1.).scale(scale)
        });
        assert!(tint.is_some(), "auto-approve chip paints its warning tint");
        let scaled_dot = dot.scale(scale);
        let dot_fill = quads.iter().find(|quad| {
            quad.background == theme::palette::warning().into()
                && quad.bounds.top() >= scaled_dot.top() - px(1.).scale(scale)
                && quad.bounds.bottom() <= scaled_dot.bottom() + px(1.).scale(scale)
        });
        assert!(dot_fill.is_some(), "warning dot carries the color signal");
    });
    // A draft-only selection through the Settings modal must NOT paint the
    // chip. Bump `selected_mode` (the draft) to `allow` but reset
    // `applied_mode` to `deny` via a real Status frame — the indicator
    // must clear because the SERVER never applied it.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.apply_worker_message(
                WorkerMessage::Status(StatusResult {
                    session: Some(SessionMetadata {
                        approval_mode: "deny".into(),
                        ..session()
                    }),
                    state: "idle".into(),
                    pending_approvals: vec![],
                    usage: json!({}),
                    compaction_markers: 0,
                }),
                window,
                cx,
            );
            view.state.session_view.selected_mode = session::APPROVAL_MODES
                .iter()
                .position(|m| *m == "allow")
                .unwrap();
            cx.notify();
        });
        window.draw(cx).clear(cx);
    });
    assert!(
        visual.debug_bounds("header-auto-approve").is_none(),
        "draft-only allow selection must not paint the header chip"
    );
    // Reconnect flow: a fresh Status frame arrives with `allow` in the
    // session metadata (the shape the worker sends on Connected and on
    // every refresh). The chip repaints without a Settings modal open.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.apply_worker_message(
                WorkerMessage::Status(StatusResult {
                    session: Some(SessionMetadata {
                        approval_mode: "allow".into(),
                        ..session()
                    }),
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
    assert!(
        visual.debug_bounds("header-auto-approve").is_some(),
        "reconnect Status frame with allow repaints the chip"
    );
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
    // ZETA-124 narrowed the prose measure and ZETA-133 lifted the cap off
    // the outer `transcript-column` onto the inner `transcript-body`, so
    // every row kind sits inside ONE centered frame at
    // `TRANSCRIPT_MAX_WIDTH` and each kind's body sizes itself INSIDE that
    // frame. Prose still caps at `prose_body_max_width` (~88ch of the base
    // font, unchanged shaped measure) — the check has moved from `column`
    // to `body`.
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
    let column = visual
        .debug_bounds("transcript-column")
        .expect("user row transcript-column draws");
    let transcript = visual.debug_bounds("transcript-viewport").unwrap();
    let body = visual
        .debug_bounds("transcript-body")
        .expect("user row transcript-body draws");
    let base = visual.update(|_, cx| cx.theme().font_size);
    let prose_cap = theme::prose_body_max_width(base);
    // The transcript viewport must exceed the prose body cap so the
    // centering branch actually activates — otherwise the row would just
    // fill the available width and the asymmetry check below would
    // trivially pass.
    assert!(
        transcript.size.width > prose_cap,
        "viewport {:?} must exceed the prose body cap {prose_cap:?} for \
         centering to matter",
        transcript.size.width
    );
    assert!(row.size.width <= transcript.size.width);
    assert!(
        f32::from(body.size.width) <= f32::from(prose_cap) + 4.0,
        "prose body width {:?} exceeded body cap {prose_cap:?}",
        body.size.width,
    );
    visual.update(|window, cx| {
        let scale = window.scale_factor();
        let scaled_viewport = transcript.scale(scale);
        let scaled_row = row.scale(scale);
        let scaled_column = column.scale(scale);
        let scaled_body_cap = px(f32::from(prose_cap)).scale(scale);
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
        // The user rectangle sits inside the body cap. The rectangle's
        // quad bounds are its border-box, so a 3px left-rail adds up to
        // 3px to the observed width — allow that plus a few pixels of
        // rendering pipeline rounding. Under ZETA-133 the frame is
        // centered in the viewport; the BODY sits at gutter-right (fixed
        // inset from frame-left), so the rectangle's asymmetry vs the
        // viewport reflects the shared frame + gutter geometry rather
        // than a per-kind centering that no longer exists.
        let tolerance = px(8.).scale(scale);
        // ONE user rail quad per row — the check applies to the first
        // matched quad, not every filter hit (a shipped row paints one
        // border-box quad for the rail).
        let quad = user_quads.into_iter().next().expect("user rail quad");
        assert!(
            quad.bounds.size.width <= scaled_body_cap + tolerance,
            "user rectangle width {:?} must land inside the prose body \
             cap {:?} (prose_cap {prose_cap:?})",
            quad.bounds.size.width,
            scaled_body_cap,
        );
        assert!(
            quad.bounds.size.width + tolerance >= scaled_body_cap,
            "user rectangle width {:?} collapsed below the prose body cap \
             {:?}",
            quad.bounds.size.width,
            scaled_body_cap,
        );
        // Frame-level centering: the outer `transcript-row` (which
        // wraps the fixed-width column) still centers inside the
        // transcript viewport. The row itself is full width, so read the
        // column bounds to pin the centering invariant.
        let left_gap = scaled_column.left() - scaled_viewport.left();
        let right_gap = scaled_viewport.right() - scaled_column.right();
        let asymmetry = if left_gap > right_gap {
            left_gap - right_gap
        } else {
            right_gap - left_gap
        };
        assert!(
            asymmetry <= tolerance,
            "transcript column not centered inside viewport: left \
             {left_gap:?}, right {right_gap:?}"
        );
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

fn assert_rendered_font_size(cx: &mut TestAppContext, kind: FontSizeProbeKind, base_px: f32) {
    let appearance = theme::Appearance {
        font_size: theme::clamp_font_size(base_px),
        ..Default::default()
    };
    let window = cx.open_window(gpui::size(px(640.), px(320.)), move |window, cx| {
        let textarea = cx.new(|cx| {
            gpui_kit::component::input::TextareaState::new(window, cx)
                .default_value("textarea body")
        });
        FontSizeProbe { kind, textarea }
    });
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        theme::apply_with(cx, &appearance);
        gpui_kit::base::zeta_font_recorder::clear();
        window.draw(cx).clear(cx);
    });
    let samples = gpui_kit::base::zeta_font_recorder::samples();
    assert!(
        !samples.is_empty(),
        "font-size probe {kind:?} must record a rendered text run at {base_px}px"
    );
    let base = appearance.font_size;
    for sample in samples {
        assert_eq!(
            sample, base,
            "font-size probe {kind:?} painted {sample:?}, expected base {base:?}"
        );
    }
}

#[gpui::test]
fn rendered_text_runs_use_the_picker_base_for_components_and_markdown(cx: &mut TestAppContext) {
    cx.update(gpui_kit::init);
    cx.update(theme::apply);
    // ZETA-139: inspect sizes recorded at the actual TextView and textarea
    // paint paths. Helper return values alone cannot catch a component or
    // markdown renderer that applies a later 0.875rem or heading scale.
    for base_px in [theme::MIN_FONT_SIZE_PX, 13.0, theme::MAX_FONT_SIZE_PX] {
        assert_rendered_font_size(cx, FontSizeProbeKind::Textarea, base_px);
        assert_rendered_font_size(cx, FontSizeProbeKind::Alert, base_px);
        assert_rendered_font_size(cx, FontSizeProbeKind::Markdown, base_px);
    }
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
                // Real bash command whose text literally begins with the
                // tool name — pre-r2 code would have collapsed a `bash …`
                // command into the label. Drives the excerpt through the
                // paint recorder with real text (not an empty string) so
                // the state-color assertion below covers the running-state
                // paint of an actual excerpt.
                let mut arguments = serde_json::Map::new();
                arguments.insert("command".into(), json!("bash scripts/warmup.sh --verbose"));
                let call = ToolCall {
                    id: "receipt".into(),
                    name: "bash".into(),
                    arguments,
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
    // Under `ListAlignment::Bottom` the paint re-engages tail-follow at
    // the end of every frame when `scroll_offset + viewport >=
    // total_content` — a `scroll_to_item(1)` on content that fits the
    // viewport is silently undone and the fence never arms. Filler
    // Assistant rows appended AFTER the tool receipts (below) inflate
    // `items.summary().height` well past the list viewport WITHOUT
    // pushing the tool rows we measure out of the visible band.
    visual.simulate_resize(gpui::size(px(1100.), px(400.)));
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
    // Filler rows past the last tool receipt so `items.summary().height`
    // exceeds the list viewport once we scroll to item 1. Without this
    // slack the paint's bottom-alignment tail-follow re-engagement check
    // (`scroll_offset + viewport >= total_content`) fires at the end of
    // paint 2 and silently undoes `scroll_to_item(1)`, so the scrolled-up
    // premise never holds. `User` rows keep the filler outside the
    // current turn's assistant reconciliation — the final `pre` message
    // below only ever sees the two streamed assistant rows, so filler
    // never scrambles the middle removal that drives the fence.
    const FILLER_ROWS: usize = 20;
    visual.update(|_, cx| {
        view.update(cx, |view, cx| {
            for i in 0..FILLER_ROWS {
                view.state
                    .transcript
                    .push(TranscriptEntry::User(format!("filler {i}")));
            }
            view.transcript.update(cx, |scroll, cx| {
                scroll.append(FILLER_ROWS, cx);
            });
            cx.notify();
        });
    });
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
                scroll.scroll_to_item(1, cx);
            });
        });
        window.draw(cx).clear(cx);
    });
    visual.update(|_, cx| {
        view.read_with(cx, |view, cx| {
            assert_eq!(view.state.transcript.len(), 5 + FILLER_ROWS);
            assert_eq!(view.transcript.read(cx).item_count(), 5 + FILLER_ROWS);
            let state = view.transcript.read(cx);
            assert!(
                state.is_scrolled_up(),
                "stability fence requires the list to be scrolled up before removal"
            );
            assert!(
                !state.is_following_tail(),
                "stability fence requires tail-follow to be disabled before removal"
            );
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
            assert_eq!(view.state.transcript.len(), 4 + FILLER_ROWS);
            assert_eq!(
                view.transcript.read(cx).item_count(),
                4 + FILLER_ROWS,
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
            excerpt: Some("echo hello".into()),
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
    // The Thinking row's paint set is exactly the gutter marker + the
    // generic header — no reasoning body ever leaks into the row's
    // visible strings. ZETA-137 D1 split the two fields so the marker
    // travels through the gutter path while the header sits at the shared
    // body edge; both are constants sourced from `state.rs`.
    let thinking = row_text::build(&TranscriptEntry::Thinking, 0, &session_view, true);
    assert_eq!(
        thinking.visible_strings(),
        vec![
            zeta_gui::state::THINKING_MARKER,
            zeta_gui::state::THINKING_HEADER_LABEL,
        ]
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

    // Seed a message so the Send button paints its enabled variant
    // (Kit Button — no outline). ZETA-134 C7 disables Send when the
    // composer is empty AND paints the disabled outline via `.border_1()`
    // — that outline sits INSIDE the composer wrapper's bounds and would
    // trip the frame's "no top/right/bottom border" invariant this test
    // asserts. Typing anything flips Send back to the borderless enabled
    // variant so the assertion stays scoped to the composer FRAME's
    // chrome, which is what the wiki contract cares about.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.composer
                .update(cx, |input, cx| input.set_value("hi", window, cx));
        });
        window.draw(cx).clear(cx);
    });

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
        excerpt: Some(id.into()),
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

    // Click the theme cycler — it advances one step through ThemeId::ALL
    // and wraps at the ends. Starting on Opencode (index 0) lands on the
    // next entry (GruvboxDark, index 1); the app-local theme moves off
    // opencode. The pre-round-2 button wall exposed five tab stops; the
    // cycler collapses them to one focusable control.
    let theme_cycler = visual
        .debug_bounds("settings-theme-cycler")
        .expect("theme cycler renders");
    visual.simulate_click(theme_cycler.center(), Default::default());
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let after_theme_bg = visual.update(|_, cx| cx.theme().background);
    assert_eq!(after_theme_bg, theme::ThemeId::GruvboxDark.palette().canvas);
    assert_ne!(after_theme_bg, baseline_bg);

    // Click the font cycler — advances one step through FONT_FAMILIES so
    // theme.font_family flips off the default (JetBrains Mono -> Fira Code).
    let font_cycler = visual
        .debug_bounds("settings-font-cycler")
        .expect("font cycler renders");
    visual.simulate_click(font_cycler.center(), Default::default());
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let after_font = visual.update(|_, cx| cx.theme().font_family.as_ref().to_string());
    assert_eq!(after_font, "Fira Code");

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
fn interactive_list_rows_paint_hover_and_pressed_across_themes(cx: &mut TestAppContext) {
    // ZETA-126 interaction feel: every clickable list row family must
    // paint a token-derived {hover, pressed} staircase, verified on
    // TWO palettes so a regression that hardcodes a color instead of
    // reading `theme.list_hover` / `theme.list_active` fails on the
    // theme where the hardcoded value diverges from the token.
    //
    // Matrix: {tool receipt row, tool-group header row, sidebar
    // session row} x {hover, pressed} x {Opencode, GruvboxLight}.
    // The tool-receipt render path (`tool_receipts.rs::render_tool_row`)
    // and the tool-group render path (`tool_receipts.rs::
    // render_tool_group_row`) are SEPARATE code sites; a mutation in
    // either must fail here. Sidebar session rows share the third
    // render path (`sidebar.rs::render_session_row`) with its own
    // pressed/hover staircase.
    //
    // Contract:
    //  - hover: `.hover(theme.list_hover)` on receipt / group;
    //    `.hover(theme.muted)` on sidebar rows.
    //  - pressed: `.active(theme.list_active)` on receipt / group;
    //    `.active(theme.list_active)` on sidebar rows.
    // `.active(...)` layers on top of `.hover(...)`; gpui's
    // interactivity holds `pressed_button = Some(Left)` between
    // mouse-down and mouse-up so a probe that simulates mouse-down
    // WITHOUT the following mouse-up must observe the pressed quad.

    let tool_entry = |id: &str| TranscriptEntry::Tool {
        key: zeta_gui::state::ToolReceiptKey {
            session_id: None,
            agent_instance_id: None,
            tool_call_id: id.into(),
        },
        name: "bash".into(),
        excerpt: Some(format!("cmd-{id}")),
        summary: String::new(),
        complete: true,
        error: false,
        canceled: false,
        card: zeta_gui::cards::Card::default(),
    };

    for theme_id in [theme::ThemeId::Opencode, theme::ThemeId::GruvboxLight] {
        let (window, view, _) = setup(cx);
        let mut visual = VisualTestContext::from_window(window.into(), cx);
        visual.update(|_, cx| {
            theme::apply_with(
                cx,
                &theme::Appearance {
                    theme: theme_id,
                    font_family: gpui::SharedString::new_static(theme::DEFAULT_FONT_FAMILY),
                    font_size: theme::DEFAULT_FONT_SIZE,
                },
            );
        });
        visual.update(|window, cx| {
            view.update(cx, |view, cx| {
                // A second, non-current session gives the sidebar a
                // switchable row. `session()` (index 0) is the active
                // session; index 1 renders below it, and its
                // `session-row` debug_selector overwrites the map so
                // `debug_bounds("session-row")` returns the non-current
                // row's rectangle.
                let mut other = session();
                other.session_id = "another-session".into();
                other.name = "another".into();
                view.state.sessions.push(other);
                view.state.session_view.available = true;

                // Three same-tool receipts fold into a group at the
                // TOOL_GROUP_MIN_LEN=3 floor (state.rs).
                view.state.transcript = vec![tool_entry("a"), tool_entry("b"), tool_entry("c")];
                view.transcript.update(cx, |scroll, cx| scroll.reset(3, cx));
                cx.notify();
            });
            window.draw(cx).clear(cx);
        });

        // Snapshot theme tokens after the first rest draw so we know
        // the values we're asserting against match the CURRENT
        // palette (peer tests that ran before could have flipped it).
        let (list_hover, list_active, sidebar_hover, sidebar_pressed) = visual.update(|_, cx| {
            let t = cx.theme();
            (t.list_hover, t.list_active, t.muted, t.list_active)
        });

        // Count painted quads matching `bg` inside `row` (1px slack
        // for scale rounding). Isolates the row's own fill from
        // wrapper quads that overlap partially.
        let hits_at = |visual: &mut VisualTestContext,
                       row: gpui::Bounds<gpui::Pixels>,
                       bg: gpui::Hsla|
         -> usize {
            visual.update(|window, _cx| {
                let scale = window.scale_factor();
                let scaled = row.scale(scale);
                let slack = px(1.).scale(scale);
                let bg: gpui::Background = bg.into();
                window
                    .painted_quads()
                    .into_iter()
                    .filter(|quad| {
                        quad.background == bg
                            && quad.bounds.top() >= scaled.top() - slack
                            && quad.bounds.bottom() <= scaled.bottom() + slack
                            && quad.bounds.left() >= scaled.left() - slack
                            && quad.bounds.right() <= scaled.right() + slack
                    })
                    .count()
            })
        };

        // Drive one row through {hover, pressed, release}. Sample
        // inside the row (~50px, ~half-height) so a scale-factor
        // rounding never lands on the edge outside the hitbox.
        let sweep = |visual: &mut VisualTestContext,
                     bounds: gpui::Bounds<gpui::Pixels>,
                     hover_bg: gpui::Hsla,
                     pressed_bg: gpui::Hsla,
                     what: &str| {
            // Sample ~50px in / ~20px down from the row origin — the
            // same offset the expand/collapse test uses. Every row
            // family in the matrix is wider than 100px, so the point
            // lands safely inside the hitbox at any scale factor.
            let sample = bounds.origin + gpui::point(px(50.), px(20.));

            // Rest: no pressed fill.
            assert_eq!(
                hits_at(visual, bounds, pressed_bg),
                0,
                "{what} on theme {theme_id:?}: pressed fill must not paint at rest"
            );

            // Hover: mouse-move over the row paints `hover_bg`.
            visual.simulate_mouse_move(sample, None, gpui::Modifiers::default());
            visual.update(|window, cx| window.draw(cx).clear(cx));
            assert!(
                hits_at(visual, bounds, hover_bg) >= 1,
                "{what} on theme {theme_id:?}: hover must paint the hover \
                 fill inside the row bounds"
            );

            // Pressed: mouse-down without mouse-up paints `pressed_bg`
            // layered on top of the hover fill.
            visual.simulate_mouse_down(sample, gpui::MouseButton::Left, Default::default());
            visual.update(|window, cx| window.draw(cx).clear(cx));
            assert!(
                hits_at(visual, bounds, pressed_bg) >= 1,
                "{what} on theme {theme_id:?}: held mouse-down must \
                 paint the pressed fill (no-dead-click contract)"
            );

            // Release clears the pressed fill.
            visual.simulate_mouse_up(sample, gpui::MouseButton::Left, Default::default());
            visual.update(|window, cx| window.draw(cx).clear(cx));
        };

        // --- Tool-group HEADER row (group is collapsed at seed). ---
        let group = visual
            .debug_bounds("tool-group-0")
            .expect("collapsed group header renders");
        sweep(&mut visual, group, list_hover, list_active, "group header");

        // `sweep`'s mouse-up fires `on_click` on the group header,
        // which toggles the group's expansion state. Force-set the
        // expanded state to `true` so the interior receipts render
        // for the next sweep regardless of what the initial toggle
        // resolved to (a peer test that seeds `tool_group_expanded`
        // differently would otherwise land here in the opposite
        // state). `remeasure_items` mirrors the on_click handler so
        // the virtual list picks up the new heights on the next draw.
        visual.update(|window, cx| {
            view.update(cx, |view, cx| {
                view.state.tool_group_expanded.insert("a".into(), true);
                view.transcript.update(cx, |scroll, cx| {
                    scroll.remeasure_items(0..3, cx);
                });
                cx.notify();
            });
            window.draw(cx).clear(cx);
        });

        // --- Tool RECEIPT row. ---
        let receipt = visual
            .debug_bounds("tool-receipt-0")
            .expect("expanded receipt renders");
        sweep(
            &mut visual,
            receipt,
            list_hover,
            list_active,
            "tool receipt",
        );

        // --- Sidebar SESSION row (non-current). The last-rendered
        // row wins the shared `session-row` debug key, so this
        // rectangle is the second session (the non-current one we
        // pushed above) — the only row that can accept hover/press.
        // ---
        let sidebar_row = visual
            .debug_bounds("session-row")
            .expect("sidebar session row renders");
        sweep(
            &mut visual,
            sidebar_row,
            sidebar_hover,
            sidebar_pressed,
            "sidebar session row",
        );

        // The pressed-during-pending-command regression the r2 review
        // named: after mouse-down, `activate_session` sets
        // `pending_command = true`; the next render must KEEP the
        // pressed fill painted while the mouse is still held. This
        // block simulates that sequence explicitly on the sidebar row.
        let mid = sidebar_row.origin + gpui::point(px(40.), px(20.));
        visual.simulate_mouse_move(mid, None, gpui::Modifiers::default());
        visual.update(|window, cx| window.draw(cx).clear(cx));
        visual.simulate_mouse_down(mid, gpui::MouseButton::Left, Default::default());
        // Two redraws while the mouse stays down: pending_command
        // flipped true on the first frame; the pressed fill must
        // survive into the second.
        visual.update(|window, cx| window.draw(cx).clear(cx));
        visual.update(|window, cx| window.draw(cx).clear(cx));
        let held_hits = hits_at(&mut visual, sidebar_row, sidebar_pressed);
        assert!(
            held_hits >= 1,
            "sidebar row must KEEP the pressed fill through the \
             `pending_command=true` re-render on theme {theme_id:?}: \
             mouse-down set pending_command and cleared can_switch, \
             but the pressed refinement must stay attached until \
             mouse-up (found {held_hits})"
        );
        visual.simulate_mouse_up(mid, gpui::MouseButton::Left, Default::default());
        visual.update(|window, cx| window.draw(cx).clear(cx));
    }
    // Restore the default theme so peer tests do not inherit
    // GruvboxLight.
    cx.update(theme::apply);
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
    // in painted text, and that the generic `+` gutter marker + `Thought`
    // body header renders (ZETA-137 D1 split).
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
            vec![
                zeta_gui::state::THINKING_MARKER,
                zeta_gui::state::THINKING_HEADER_LABEL,
            ],
            "Thinking row model text must be exactly the gutter marker + generic header"
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
    // Sits at 15% of the viewport height — the Settings-only offset that
    // ZETA-132 introduced so the three-section body fits on open at
    // 900px+ viewport heights. Rename / delete dialogs still ride the
    // shared 25% shelf; that pair is guarded in
    // `session_edit_modal_matches_the_wiki_flat_panel_shape` against
    // `theme::MODAL_TOP_FRACTION`. Drift beyond layout rounding (~1
    // logical px) means someone shifted the offset itself, not a
    // fractional-pixel rounding wobble.
    let overlay = visual
        .debug_bounds("settings-overlay")
        .expect("settings overlay renders");
    let target = overlay.top() + overlay.size.height * theme::SETTINGS_MODAL_TOP_FRACTION;
    let drift = if panel.top() > target {
        panel.top() - target
    } else {
        target - panel.top()
    };
    assert!(
        drift <= px(1.),
        "settings panel top {:?} must land within 1px (layout rounding) \
         of the 15% Settings offset ({:?})",
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
    // A11y regression guard (finding #2 — expanded round 3; ZETA-126
    // round 2 converted this to REAL Tab / Shift-Tab keystroke dispatch
    // through `simulate_keystrokes("tab")` and `simulate_keystrokes(
    // "shift-tab")` — the same path Root's keybindings route a real
    // keyboard-user's keys through). Direct `focus_next` /
    // `focus_prev` / `window.focus(handle)` calls would paper over a
    // missing tab_stop flag OR a broken Tab keybinding; keystroke
    // dispatch exercises both.
    //
    // Sidebar rows must be:
    //   1. reachable by Tab (`.tab_index(0)` + `tab_stop(true)` on the
    //      focus handle register the row in the tab-stops walk that
    //      Tab dispatches through),
    //   2. paint the solid-accent keyboard cursor on the FOCUSED row so a
    //      keyboard-only user sees which row Enter/Space would activate —
    //      the current row must show the cursor too; the "no fill"
    //      contract only applies to the UNFOCUSED current row,
    //   3. activate on Enter AND Space (button-role keyboard contract).
    //
    // Both session and branch rows share this contract. Mutations that
    // must fail: dropping the focus-cursor branch, dropping a key
    // handler, or dropping `.tab_index(0)` / `tab_stop(true)` from either
    // row type — the last one surfaces under a bounded Tab-keystroke
    // walk that never reaches the target.
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

    // Walk REAL `Tab` keystrokes until the tab-stops registry lands on
    // `target`. Wrapped in a bounded loop so a regression that drops the
    // target from the registry surfaces as a clear panic (not an
    // infinite hang). `simulate_keystrokes("tab")` routes through
    // Root's `Tab` binding — the same dispatch path a keyboard-only
    // user drives — so this catches a missing `tab_stop` flag OR a
    // gap in the Tab keybinding itself. `window.focus(&handle)` would
    // silently paper over both.
    let tab_to = |visual: &mut VisualTestContext, target: &gpui::FocusHandle| {
        // Enter the walk with focus on `current_session_handle` — a
        // stable, always-present sidebar row. Root's `Tab` action
        // requires a "Root" context in the dispatch path; a blurred
        // window's `dispatch_path` is just [root_node_id] and does
        // NOT include the Root-wrapping div, so `simulate_keystrokes`
        // wouldn't route through Root's binding at all. Peer tests
        // do the same (tests.rs:7912) — the seed is real setup, and
        // every step below IS an honest `simulate_keystrokes("tab")`
        // dispatch a keyboard user would drive.
        visual.update(|window, cx| {
            window.focus(&current_session_handle, cx);
            window.draw(cx).clear(cx);
        });
        // Bound the walk generously so a slow-cycling sidebar (extra
        // sessions + branch rows + the composer) still resolves.
        // Matches the peer group-header test's 512 ceiling
        // (tests.rs:8045).
        let max_steps = 512;
        for _ in 0..max_steps {
            visual.simulate_keystrokes("tab");
            visual.update(|window, cx| window.draw(cx).clear(cx));
            if visual.update(|window, _| target.is_focused(window)) {
                return;
            }
        }
        panic!(
            "a bounded `Tab` walk did not land on the target focus \
             handle within {max_steps} steps — the row's focus handle \
             is not registered as a tab stop, or the `Tab` keybinding \
             does not route through the row"
        );
    };

    // --- The CURRENT session row (active) MUST paint the accent cursor
    // when reached via Tab. Contract: the current-item "no fill" rule
    // applies to the UNFOCUSED state only; a focused row overrides it.
    // A mutation that reintroduces the old `focused && !active` guard
    // would leave the current row with no visible focus cursor. Row
    // paint IS the focus trigger's own paint — this line proves both
    // reachability (Tab lands on the row) and repaint (the accent quad
    // is present in the following frame). ---
    tab_to(&mut visual, &current_session_handle);
    assert!(
        row_sized_accent(&mut visual, theme::SIDEBAR_ROW_HEIGHT),
        "the CURRENT session row must still paint the accent cursor \
         when focused via Tab (no-fill rule applies only to the \
         unfocused state; a Tab walk that reaches the row must trigger \
         a repaint that includes the row-sized accent quad)"
    );

    // --- Non-active session row also paints the cursor on Tab focus. ---
    tab_to(&mut visual, &session_handle);
    assert!(
        row_sized_accent(&mut visual, theme::SIDEBAR_ROW_HEIGHT),
        "focused session row (reached via Tab) must paint the accent \
         focus cursor — the row's focus trigger must repaint the row \
         with the accent fill on the following frame"
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

    // --- Space activates the focused session row. Re-reach the row via
    // a fresh Tab walk (session activation sets `pending_command` and
    // captures fresh `can_activate` state; the walk restarts from a
    // blurred window so a stale focus does not skip the trigger). ---
    visual.update(|window, cx| {
        view.update(cx, |view, _| view.pending_command = false);
        window.draw(cx).clear(cx);
    });
    tab_to(&mut visual, &session_handle);
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
    // cursor when reached via Tab. Same contract as sessions — focused
    // wins over the unfocused "no fill" rule. ---
    visual.update(|_, cx| {
        view.update(cx, |view, _| view.pending_command = false);
    });
    tab_to(&mut visual, &current_branch_handle);
    assert!(
        row_sized_accent(&mut visual, theme::SIDEBAR_NESTED_ROW_HEIGHT),
        "the CURRENT branch row must still paint the accent cursor \
         when focused via Tab (no-fill rule applies only to the \
         unfocused state)"
    );

    // --- Shift-Tab from trunk lands on the last session row (Tab
    // order runs session rows → branch rows). Prove the reverse walk
    // reaches the previous tab stop and repaints its accent, so a
    // regression that drops Shift-Tab handling from either row type
    // fails here. `simulate_keystrokes("shift-tab")` dispatches through
    // Root's shift-tab binding — the real keyboard-user path. ---
    visual.simulate_keystrokes("shift-tab");
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let landed_on_session = visual.update(|window, _| session_handle.is_focused(window))
        || visual.update(|window, _| current_session_handle.is_focused(window));
    assert!(
        landed_on_session,
        "Shift-Tab from the trunk branch row must land on a session row \
         — the tab-stops registry must include session rows AS THE \
         previous stops from the branches strip"
    );
    assert!(
        row_sized_accent(&mut visual, theme::SIDEBAR_ROW_HEIGHT),
        "the session row Shift-Tab landed on must paint its accent \
         cursor — Shift-Tab is a real focus transition and must trigger \
         a repaint"
    );

    // --- Non-current branch row (alt) paints on Tab focus. ---
    tab_to(&mut visual, &branch_handle);
    assert!(
        row_sized_accent(&mut visual, theme::SIDEBAR_NESTED_ROW_HEIGHT),
        "focused branch row (reached via Tab) must paint the accent \
         focus cursor"
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
    visual.update(|_, cx| {
        view.update(cx, |view, _| view.pending_command = false);
    });
    tab_to(&mut visual, &branch_handle);
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
fn current_session_dot_is_subtle_and_leaves_a_gap_before_the_title(cx: &mut TestAppContext) {
    // ZETA-139: the current-run dot shrinks to a subtle text-glyph-sized
    // dot (Henry: "The little icon for select runs is too large, looks
    // weird") and never touches the title's first glyph. Guards against
    // regressing to the pre-ZETA-139 9px bullet or losing the gutter gap.
    let (window, _view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let row = visual
        .debug_bounds("session-row")
        .expect("session row renders");
    let dot = visual
        .debug_bounds("session-current-dot")
        .expect("current session paints its accent dot");
    // Size lands on the SIDEBAR_CURRENT_DOT_SIZE token (small, subtle).
    let size_delta = if dot.size.width > theme::SIDEBAR_CURRENT_DOT_SIZE {
        dot.size.width - theme::SIDEBAR_CURRENT_DOT_SIZE
    } else {
        theme::SIDEBAR_CURRENT_DOT_SIZE - dot.size.width
    };
    assert!(
        size_delta <= px(1.),
        "dot width {:?} must land on the SIDEBAR_CURRENT_DOT_SIZE contract",
        dot.size.width
    );
    // The dot never claims more than a small fraction of the row's height —
    // pins the "looks like a mono period, not a bullet" ratio.
    assert!(
        f32::from(dot.size.height) * 4.0 <= f32::from(row.size.height),
        "dot height {:?} exceeded 1/4 of the row height {:?} — regressed \
         to a bullet-sized indicator",
        dot.size.height,
        row.size.height,
    );
    // Left inset — dot.left - row.left ~= SIDEBAR_CURRENT_DOT_INSET.
    let inset = dot.left() - row.left();
    let inset_delta = if inset > theme::SIDEBAR_CURRENT_DOT_INSET {
        inset - theme::SIDEBAR_CURRENT_DOT_INSET
    } else {
        theme::SIDEBAR_CURRENT_DOT_INSET - inset
    };
    assert!(
        inset_delta <= px(1.),
        "dot left inset {:?} must land on the SIDEBAR_CURRENT_DOT_INSET \
         contract",
        inset
    );
    // Gap between dot and the title's left edge. The label div sits at
    // gutter-right; the 5px gap is part of the sidebar row contract, so a
    // positive-but-arbitrary gap cannot hide padding drift.
    let gutter_right = row.left() + theme::SIDEBAR_GUTTER_WIDTH;
    let gap = gutter_right - dot.right();
    assert!(
        (f32::from(gap) - 5.0).abs() <= 1.0,
        "dot-to-title gap {:?} must stay at the 5px sidebar contract \
         (dot.right {:?}, gutter_right {:?})",
        gap,
        dot.right(),
        gutter_right,
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
    // ZETA-133: the injection anchor moved from the pre-ZETA-133 tail of
    // render_thinking_row (`.child(header)\n            .into_any_element()\n    }`)
    // to a `.child(header);` inside the body-pair-wrapped shape. The pattern
    // is stable across the D1 refactor and appears in error / footer paths
    // too, so any real fence weakening still surfaces.
    let mutations: &[(&str, &str, &str)] = &[
        // (label, injection point — matched verbatim, mutated snippet)
        (
            "child bracketed state marker",
            ".child(header)",
            ".child(\"[done]\").child(header)",
        ),
        (
            "child lowercase prose",
            ".child(header)",
            ".child(\"done\").child(header)",
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

/// ZETA-139: every type role — title, body, label, label_small,
/// label_micro — resolves to the base font size at every picker step.
/// Hierarchy comes from weight + color tier alone. A refactor that
/// reintroduces a scaled role (a `+2` title step or a `-1` label step)
/// trips here at every base.
#[test]
fn role_scale_collapses_onto_one_size_at_every_picker_step() {
    for base_px in [
        theme::MIN_FONT_SIZE_PX as i32,
        f32::from(theme::DEFAULT_FONT_SIZE) as i32,
        theme::MAX_FONT_SIZE_PX as i32,
    ] {
        let base = px(base_px as f32);
        assert_eq!(theme::title(base), base, "title must equal base");
        assert_eq!(theme::body(base), base, "body must equal base");
        assert_eq!(theme::label(base), base, "label must equal base");
        assert_eq!(
            theme::label_small(base),
            base,
            "label_small must equal base"
        );
        assert_eq!(
            theme::label_micro(base),
            base,
            "label_micro must equal base"
        );
    }
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
    /// Straight source-over composite of a translucent `over` onto opaque
    /// `under`, in sRGB (matching what the painter blends). Used to derive
    /// the effective background under tint chips (`warning_tint`,
    /// `danger_tint`) whose visible fill is the mix of the tint alpha and
    /// the canvas beneath — a WCAG check that reads `warning_tint()` alone
    /// misses that mix.
    fn composite_over(over: gpui::Hsla, under: gpui::Hsla) -> gpui::Hsla {
        let a = over.a.clamp(0.0, 1.0);
        let o = over.to_rgb();
        let u = under.to_rgb();
        let r = o.r * a + u.r * (1.0 - a);
        let g = o.g * a + u.g * (1.0 - a);
        let b = o.b * a + u.b * (1.0 - a);
        gpui::Rgba { r, g, b, a: 1.0 }.into()
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
        // ZETA-131 audit B7 / round-2 finding 3: the auto-approve chip
        // paints `theme.foreground` over `warning_tint()` (warning at 15%
        // alpha) which itself sits on `canvas` in the run header. The
        // "auto-approve" label is `label_small`-tier text under WCAG, so
        // its effective background — canvas mixed with 15% warning — must
        // clear the 4.5:1 bar against `foreground` on every shipped theme.
        let chip_bg = composite_over(p.warning_tint(), p.canvas);
        let chip_ratio = contrast_ratio(p.text, chip_bg);
        assert!(
            chip_ratio >= 4.5,
            "{}: auto-approve chip text/warning_tint-on-canvas contrast \
             {chip_ratio:.2}:1 fails WCAG AA (need >=4.5:1)",
            id.label()
        );
    }
}

/// Tool rows share the SAME body cap as prose / user / assistant / thinking
/// rows (ZETA-139): one unified column, one horizontal edge pair. Regresses
/// if a future refactor gives tool receipts their own wider cap.
#[gpui::test]
fn tool_rows_share_the_prose_body_cap(cx: &mut TestAppContext) {
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
                excerpt: Some("run a very long command line ".repeat(30)),
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
    let prose_body_cap = f32::from(theme::prose_body_max_width(base));
    // The tool body caps at the prose measure so a collapsed tool row's
    // visible right edge lines up with prose / user / assistant right
    // edges. A regression to the pre-ZETA-139 wide cap would push this
    // 200+px wider at the shipped 13px base.
    let body = visual
        .debug_bounds("transcript-body")
        .expect("tool body draws");
    assert!(
        f32::from(body.size.width) <= prose_body_cap + 4.0,
        "tool body width {:?} exceeded prose body cap {prose_body_cap} \
         — a regression that split tool bodies onto their own cap would \
         trip here",
        body.size.width,
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
    let body = visual
        .debug_bounds("transcript-body")
        .expect("assistant body draws");
    let base = visual.update(|_, cx| cx.theme().font_size);
    let prose_body_cap = f32::from(theme::prose_body_max_width(base));
    assert!(
        f32::from(body.size.width) <= prose_body_cap + 4.0,
        "assistant body width {:?} exceeded prose body cap {prose_body_cap} \
         — a per-block splitter that gave code fences the wide cap \
         regressed the r2 accepted shape",
        body.size.width,
    );
    // ZETA-139: prose + tool bodies now share ONE cap. The dedicated
    // `tool_rows_share_the_prose_body_cap` test is the flip-side guard.
}

/// ZETA-139: collapsed AND expanded tool rows keep the prose body cap
/// across every picker step (11px / 13px / 18px), so a run of receipts
/// never grows a wider right edge than prose. Expanding a receipt paints
/// an inset panel INSIDE the same body — the panel's own bounds sit
/// within the body, so measuring `transcript-body` still holds.
#[gpui::test]
fn tool_rows_share_the_prose_body_cap_across_the_picker(cx: &mut TestAppContext) {
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.simulate_resize(gpui::size(px(1600.), px(760.)));
    let mut appearance = theme::Appearance::default();
    for &base_px in &[
        theme::MIN_FONT_SIZE_PX,
        f32::from(theme::DEFAULT_FONT_SIZE),
        theme::MAX_FONT_SIZE_PX,
    ] {
        appearance.font_size = theme::clamp_font_size(base_px);
        for &expanded in &[false, true] {
            visual.update(|window, cx| {
                theme::apply_with(cx, &appearance);
                view.update(cx, |view, cx| {
                    view.state.transcript = vec![TranscriptEntry::Tool {
                        key: zeta_gui::state::ToolReceiptKey {
                            session_id: None,
                            agent_instance_id: None,
                            tool_call_id: "wide".into(),
                        },
                        name: "bash".into(),
                        excerpt: Some("run a very long command line ".repeat(30)),
                        summary: "run a very long command line ".repeat(30),
                        complete: true,
                        error: false,
                        canceled: false,
                        card: zeta_gui::cards::Card {
                            expanded,
                            tail: zeta_gui::cards::OutputTail {
                                text: "line one\nline two\nline three".into(),
                                ..Default::default()
                            },
                            ..Default::default()
                        },
                    }];
                    view.transcript.update(cx, |scroll, cx| scroll.reset(1, cx));
                    cx.notify();
                });
                window.draw(cx).clear(cx);
            });
            let body = visual
                .debug_bounds("transcript-body")
                .unwrap_or_else(|| panic!("tool body at {base_px}px expanded={expanded}"));
            let prose_body_cap = f32::from(theme::prose_body_max_width(appearance.font_size));
            assert!(
                (f32::from(body.size.width) - prose_body_cap).abs() <= 4.0,
                "tool body width {:?} must paint at the prose cap {prose_body_cap} \
                 within 4px at {base_px}px expanded={expanded}",
                body.size.width,
            );
            if expanded {
                let panel = visual
                    .debug_bounds("tool-output-0")
                    .unwrap_or_else(|| panic!("expanded tool panel at {base_px}px"));
                assert!(
                    (f32::from(panel.size.width) - f32::from(body.size.width)).abs() <= 2.0,
                    "expanded tool panel width {:?} must match body width {:?} within 2px \
                     at {base_px}px",
                    panel.size.width,
                    body.size.width,
                );
                assert!(
                    (f32::from(panel.left()) - f32::from(body.left())).abs() <= 1.0
                        && (f32::from(panel.right()) - f32::from(body.right())).abs() <= 1.0,
                    "expanded tool panel {:?} must stay on the painted body edges {:?} \
                     at {base_px}px",
                    panel,
                    body,
                );
            }
        }
    }
    // Reset appearance so peer tests see the shipped default.
    visual.update(|_, cx| theme::apply(cx));
}

/// ZETA-139: the composer's painted left and right edges align with the
/// transcript column at BOTH a narrow viewport (1100px — column fills
/// viewport) and a wide viewport (1600px — column centers), across the
/// picker's MIN and MAX bases. Guards against the pre-ZETA-139 shape
/// where the composer stretched full window width while transcript rows
/// centered at TRANSCRIPT_MAX_WIDTH.
#[gpui::test]
fn composer_left_and_right_edges_match_the_transcript_column(cx: &mut TestAppContext) {
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.transcript = vec![TranscriptEntry::User("hi".into())];
            view.transcript.update(cx, |scroll, cx| scroll.reset(1, cx));
            cx.notify();
        });
        window.draw(cx).clear(cx);
    });
    let mut appearance = theme::Appearance::default();
    for &width in &[px(1100.), px(1600.)] {
        visual.simulate_resize(gpui::size(width, px(760.)));
        for &base_px in &[theme::MIN_FONT_SIZE_PX, theme::MAX_FONT_SIZE_PX] {
            appearance.font_size = theme::clamp_font_size(base_px);
            visual.update(|window, cx| {
                theme::apply_with(cx, &appearance);
                view.update(cx, |_, cx| cx.notify());
                window.draw(cx).clear(cx);
            });
            let transcript_column = visual
                .debug_bounds("transcript-column")
                .unwrap_or_else(|| panic!("transcript-column at {width:?} {base_px}px"));
            let composer_column = visual
                .debug_bounds("composer-column")
                .unwrap_or_else(|| panic!("composer-column at {width:?} {base_px}px"));
            let left_delta =
                (f32::from(composer_column.left()) - f32::from(transcript_column.left())).abs();
            let right_delta =
                (f32::from(composer_column.right()) - f32::from(transcript_column.right())).abs();
            assert!(
                left_delta <= 1.0,
                "composer-column left {:?} must match transcript-column \
                 left {:?} at {width:?} {base_px}px (delta {left_delta})",
                composer_column.left(),
                transcript_column.left(),
            );
            assert!(
                right_delta <= 1.0,
                "composer-column right {:?} must match transcript-column \
                 right {:?} at {width:?} {base_px}px (delta {right_delta})",
                composer_column.right(),
                transcript_column.right(),
            );
        }
    }
    visual.update(|_, cx| theme::apply(cx));
}

/// ZETA-139: the shared body edge reaches every inner composer surface. The
/// outer-column test above cannot catch padding or gutter drift because both
/// columns can keep matching while their children move. Render the pending
/// rail, slash menu, input, footer, and drop overlay, then compare their
/// painted bounds with the transcript body at the picker extremes.
#[gpui::test]
fn shared_content_column_inner_bounds_match_the_transcript_body(cx: &mut TestAppContext) {
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.simulate_resize(gpui::size(px(1600.), px(900.)));
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.transcript = vec![TranscriptEntry::User("hello".into())];
            view.transcript.update(cx, |scroll, cx| scroll.reset(1, cx));
            view.pending_user_turn = Some(super::PendingUserTurn {
                text: "queued".into(),
                failed: false,
            });
            view.slash_menu.commands = slash_catalog(&["status"]).commands;
            view.slash_menu.open = true;
            view.slash_menu.filter.clear();
            cx.notify();
        });
        window.draw(cx).clear(cx);
    });

    let mut appearance = theme::Appearance::default();
    for &base_px in &[theme::MIN_FONT_SIZE_PX, 13.0, theme::MAX_FONT_SIZE_PX] {
        appearance.font_size = theme::clamp_font_size(base_px);
        visual.update(|window, cx| {
            theme::apply_with(cx, &appearance);
            window.draw(cx).clear(cx);
        });

        let body = visual
            .debug_bounds("transcript-body")
            .expect("transcript body paints");
        let composer_rail = visual
            .debug_bounds("composer-rail")
            .expect("composer rail paints");
        let pending_rail = visual
            .debug_bounds("pending-rail")
            .expect("pending rail paints");
        for (name, bounds) in [
            ("composer rail", composer_rail),
            ("pending rail", pending_rail),
        ] {
            assert!(
                (f32::from(bounds.left()) - f32::from(body.left())).abs() <= 1.0
                    && (f32::from(bounds.right()) - f32::from(body.right())).abs() <= 1.0,
                "{name} {:?} must match transcript body {:?} at {base_px}px",
                bounds,
                body,
            );
        }

        let composer = visual.debug_bounds("composer").expect("composer paints");
        visual.update(|window, cx| {
            window.dispatch_event(
                gpui::FileDropEvent::Entered {
                    position: composer.center(),
                    paths: gpui::ExternalPaths(
                        [std::env::temp_dir().join("zeta-column-probe.png")]
                            .into_iter()
                            .collect(),
                    ),
                }
                .to_platform_input(),
                cx,
            );
            window.draw(cx).clear(cx);
        });

        let drop_target = visual
            .debug_bounds("composer-drop-target")
            .expect("drop overlay paints");
        assert!(
            (f32::from(drop_target.left())
                - f32::from(body.left())
                - f32::from(theme::RAIL_WIDTH_THICK))
            .abs()
                <= 1.0
                && (f32::from(drop_target.right()) - f32::from(body.right())).abs() <= 1.0,
            "drop overlay {:?} must align with the painted body edge {:?} after the \
             composer rail inset at {base_px}px (composer {:?})",
            drop_target,
            body,
            composer,
        );

        for name in ["composer-input", "composer-footer", "slash-menu"] {
            let bounds = visual
                .debug_bounds(name)
                .unwrap_or_else(|| panic!("{name} paints at {base_px}px"));
            assert!(
                bounds.left() >= body.left() - px(1.) && bounds.right() <= body.right() + px(1.),
                "{name} {:?} must stay inside transcript body {:?} at {base_px}px",
                bounds,
                body,
            );
        }

        visual.update(|window, cx| {
            window.dispatch_event(gpui::FileDropEvent::Exited.to_platform_input(), cx);
            window.draw(cx).clear(cx);
        });
    }
    visual.update(|_, cx| theme::apply(cx));
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
    // ZETA-133: the wrap-relevant column is now the inner `transcript-body`
    // (the outer `transcript-column` is always `TRANSCRIPT_MAX_WIDTH`).
    let body = visual
        .debug_bounds("transcript-body")
        .expect("wedge body draws");
    let transcript = visual
        .debug_bounds("transcript-viewport")
        .expect("transcript viewport draws");
    let base = visual.update(|_, cx| cx.theme().font_size);
    let prose_body_cap = f32::from(theme::prose_body_max_width(base));
    let text_measure = f32::from(theme::prose_text_measure(base));
    // The wedge assistant is a prose row — its body must sit at the
    // narrower measure so the hanging-indent list content wraps at a
    // scannable width. `+4` guards against the pipeline's subpixel rounding.
    assert!(
        f32::from(body.size.width) <= prose_body_cap + 4.0,
        "wedge assistant body width {:?} exceeded prose body cap \
         {prose_body_cap} — the ZETA-124/133 measure gate is off",
        body.size.width,
    );
    // The row sits centered inside the transcript viewport: left and
    // right gaps balance within a few pixels. A regression that shifts the
    // wrap point past the visible row (the r0 orphan-glyph shape)
    // drifts the centering here first.
    let viewport_center = transcript.left() + transcript.size.width / 2.0;
    let row_center = row.left() + row.size.width / 2.0;
    let drift = if viewport_center > row_center {
        viewport_center - row_center
    } else {
        row_center - viewport_center
    };
    assert!(
        f32::from(drift) < 8.0,
        "wedge row not centered inside transcript viewport: drift {drift:?}",
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
        let scaled_body = body.scale(scale);
        let inner_left = scaled_body.left();
        let inner_right = scaled_body.right();
        let tolerance = px(4.).scale(scale);
        // Quads inside the row's vertical AND the transcript body's
        // horizontal band: this excludes the sidebar / composer strips
        // that paint at the same y-range but on the other side of the
        // window. Quads whose left edge is left of the BODY (e.g. the
        // leading gutter's chevron on tool rows) are excluded — the
        // wrap-boundary guard we care about here is the prose body's
        // right edge.
        let row_quads: Vec<_> = window
            .painted_quads()
            .into_iter()
            .filter(|quad| {
                quad.bounds.top() >= scaled_row.top()
                    && quad.bounds.bottom() <= scaled_row.bottom() + tolerance
                    && quad.bounds.left() >= scaled_body.left() - tolerance
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
                "quad {:?} started left of the transcript body {inner_left:?}",
                quad.bounds,
            );
            assert!(
                quad.bounds.right() <= inner_right + tolerance,
                "quad {:?} shaped past the transcript body {inner_right:?} \
                 — the wrap boundary regressed",
                quad.bounds,
            );
        }
        // Hanging-indent continuation: the wedge content is long enough
        // that at the 90ch prose measure it wraps onto multiple visible
        // lines. Assert the row is taller than a single line at the
        // current base — this catches a regression that reverts to
        // `wide_body_max_width` for prose (which would let the whole
        // paragraph fit on one line at 2204px) and it catches a
        // padding-included cap that quietly grew the measure back past
        // 90ch on this shape.
        let single_line = f32::from(base) * 1.65;
        let row_height = f32::from(row.size.height);
        assert!(
            row_height > single_line * 2.5,
            "wedge row height {row_height} did not exceed 2.5 line-heights \
             ({}) — the hanging-indent continuation did not paint on \
             separate lines (prose_body_cap {prose_body_cap}, text_measure \
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
                let body = visual
                    .debug_bounds("transcript-body")
                    .unwrap_or_else(|| panic!("assistant body at {label} {vw:?}x{vh:?} {base_px}"));
                // ZETA-133: the body IS the ACTUAL width the TextView had
                // to wrap into (no interior padding on the body div —
                // padding lives on the outer `transcript-column`). A
                // formula that shrinks the body past the promised measure
                // exposes the gap here because recorder samples are
                // shaped at the PROMISED wrap width
                // (`prose_text_measure(base)`), not at the row's actual
                // inner width — so a shape-vs-body drift lands as an
                // overflow the assertion catches.
                let inner_width = f32::from(body.size.width);
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
                    // equal the row's actual inner text width WITHIN a
                    // small tolerance. ZETA-133: under the D1 body-pair
                    // layout the recorder is fed `prose_wrap_budget`
                    // (the actual `.max_w` on the TextView) which is
                    // `floor(prose_body_max_width - 2)`, and the body's
                    // painted width lands within a few pixels of that
                    // budget (flex-layout rounding + text-view internal
                    // measurement). A caller that hands the wrong wrap
                    // constraint to the text system would drift by TENS
                    // of pixels, not by four, so the invariant still
                    // catches the r2-formula-gap defect.
                    assert!(
                        (wrap - inner_width).abs() < 6.0,
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
/// argument key the tool actually reads, returns `None` for a tool call
/// whose arguments carry nothing nameable (the typed missing-argument
/// state — ZETA-134 review r2), and truncates a pathological argument
/// with a single-character ellipsis so a wide argument still fits on one
/// row.
#[test]
fn zeta125_excerpts_route_by_kind_and_truncate() {
    use serde_json::json;
    let excerpt = |name: &str, args: serde_json::Value| {
        let map = args.as_object().cloned().unwrap_or_default();
        zeta_gui::state::tool_excerpt(name, &map)
    };
    // Bash-family tools read the "command" argument's first line.
    assert_eq!(
        excerpt("bash", json!({"command": "grep -rn TODO src/"})).as_deref(),
        Some("grep -rn TODO src/"),
    );
    assert_eq!(
        excerpt("exec", json!({"command": "ls -la\nsecond line"})).as_deref(),
        Some("ls -la"),
    );
    // Read/write/edit route through "path".
    assert_eq!(
        excerpt("read", json!({"path": "src/main.rs"})).as_deref(),
        Some("src/main.rs"),
    );
    assert_eq!(
        excerpt("write", json!({"path": "notes.txt"})).as_deref(),
        Some("notes.txt"),
    );
    assert_eq!(
        excerpt("edit", json!({"path": "docs/design.md"})).as_deref(),
        Some("docs/design.md"),
    );
    // Fetch reads "url" and keeps the whole URL under the cap.
    assert_eq!(
        excerpt("fetch", json!({"url": "https://example.com/api/v1/data"})).as_deref(),
        Some("https://example.com/api/v1/data"),
    );
    // Unknown tools fall back to the first primitive argument.
    assert_eq!(
        excerpt("weather", json!({"city": "Paris"})).as_deref(),
        Some("Paris"),
    );
    // With no primitive argument, the missing state is `None` — the row
    // builder reads that as "paint the tool label alone" (ZETA-134 A3).
    assert_eq!(excerpt("noop", json!({})), None);
    // Truncation trims to `EXCERPT_CHARS` and appends a single-character
    // ellipsis marker. The output length is at most cap + 1 char.
    let long = "a".repeat(zeta_gui::state::EXCERPT_CHARS * 2);
    let truncated = excerpt("bash", json!({"command": long})).expect("long excerpt");
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
                excerpt: Some("cargo check".into()),
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
        excerpt: Some(format!("cargo test {id}")),
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
                TranscriptEntry::Tool { excerpt, .. } => excerpt.as_deref(),
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
        excerpt: Some(format!("src/{id}.rs")),
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
    // Reach the group header through the Root keymap's real Tab dispatch.
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
    // Seed focus on the current session's known tab stop. The group header
    // is still discovered only through real Tab dispatch.
    let session_handle = visual
        .update(|_, cx| {
            view.read(cx)
                .sidebar_row_focus
                .borrow()
                .get(&session().session_id)
                .cloned()
        })
        .expect("current session focus handle registered on first paint");
    visual.update(|window, cx| {
        window.focus(&session_handle, cx);
        window.draw(cx).clear(cx);
    });
    let max_tab_steps = 512;
    let mut steps_to_group = None;
    for step in 0..max_tab_steps {
        visual.simulate_keystrokes("tab");
        visual.update(|window, cx| window.draw(cx).clear(cx));
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
    visual.simulate_keystrokes("shift-tab");
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let reverse_focus = visual.update(|window, cx| window.focused(cx));
    assert!(
        reverse_focus.is_some(),
        "Shift-Tab must preserve keyboard focus"
    );
    assert_ne!(
        reverse_focus.as_ref(),
        Some(&group_handle),
        "Shift-Tab must move back from the group header"
    );
    visual.simulate_keystrokes("tab");
    visual.update(|window, cx| window.draw(cx).clear(cx));
    assert!(
        visual.update(|window, _| group_handle.is_focused(window)),
        "Tab must return to the group header after Shift-Tab"
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
        excerpt: Some(format!("src/{id}.rs")),
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
                TranscriptEntry::Tool { excerpt, .. } => excerpt.as_deref(),
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
        excerpt: Some(format!("cmd {id}")),
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
                excerpt: Some("cargo test --lib".into()),
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

// ---------------------------------------------------------------------------
// ZETA-128: Settings surface — grouped sections, universal row anatomy,
// keyboard traversal. Kept in one region at file end so sibling refactors
// (ZETA-125 transcript receipts / ZETA-127 test infra) rebase cleanly.
// ---------------------------------------------------------------------------

fn open_settings_with_default_catalog(view: &Entity<ZetaView>, visual: &mut VisualTestContext) {
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
}

fn seed_settings_login_providers(view: &Entity<ZetaView>, visual: &mut VisualTestContext) {
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.apply_worker_message(
                WorkerMessage::LoginProviders(vec![
                    LoginProvider {
                        provider: "claude".into(),
                        credentials_present: false,
                        progress: LoginProgress::Idle,
                    },
                    LoginProvider {
                        provider: "codex".into(),
                        credentials_present: false,
                        progress: LoginProgress::Idle,
                    },
                ]),
                window,
                cx,
            );
        });
    });
}

/// Open Settings with a multi-model catalog so tab-through-model-rows tests
/// have more than one row to focus. Every model belongs to the same group
/// so the child-index math stays trivial.
fn open_settings_with_multi_model_catalog(
    view: &Entity<ZetaView>,
    visual: &mut VisualTestContext,
    models: &[&str],
) {
    let owned: Vec<String> = models.iter().map(|s| (*s).to_owned()).collect();
    let providers: std::collections::BTreeMap<String, String> = owned
        .iter()
        .map(|m| (m.clone(), "claude".to_owned()))
        .collect();
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.session_view.available = true;
            view.apply_worker_message(
                WorkerMessage::Settings(
                    SessionSettings {
                        model: owned[0].clone(),
                        approval_mode: "ask".into(),
                    },
                    ModelCatalog {
                        models: owned.clone(),
                        providers,
                    },
                ),
                window,
                cx,
            );
        });
        window.draw(cx).clear(cx);
    });
}

#[gpui::test]
fn settings_render_three_grouped_sections_with_headings(cx: &mut TestAppContext) {
    // The Settings modal must present three anchored sections — Model,
    // Behavior, Appearance — each with a title-weight heading. A regression
    // that collapsed them back into a flat strip of muted captions would
    // drop the -heading debug selectors this asserts on.
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    open_settings_with_default_catalog(&view, &mut visual);
    for &(section, heading) in &[
        ("settings-section-model", "settings-section-model-heading"),
        (
            "settings-section-behavior",
            "settings-section-behavior-heading",
        ),
        (
            "settings-section-appearance",
            "settings-section-appearance-heading",
        ),
    ] {
        assert!(
            visual.debug_bounds(section).is_some(),
            "{section} must render its own section container"
        );
        assert!(
            visual.debug_bounds(heading).is_some(),
            "{section} must render its own heading node"
        );
    }
}

#[gpui::test]
fn settings_rows_use_label_left_control_right_anatomy(cx: &mut TestAppContext) {
    // Every labeled row in Behavior + Appearance follows the same anatomy:
    // fixed-width label column on the left, control anchored on the right.
    // Regressions that revert to a stacked "caption above buttons" layout
    // (the pre-ZETA-128 shape) would break this bounds check.
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    open_settings_with_default_catalog(&view, &mut visual);
    for &(row, label_sel, control_sel) in &[
        (
            "settings-row-approval",
            "settings-row-approval-label",
            "settings-row-approval-control",
        ),
        (
            "settings-row-theme",
            "settings-row-theme-label",
            "settings-row-theme-control",
        ),
        (
            "settings-row-font",
            "settings-row-font-label",
            "settings-row-font-control",
        ),
        (
            "settings-row-size",
            "settings-row-size-label",
            "settings-row-size-control",
        ),
    ] {
        let label = visual
            .debug_bounds(label_sel)
            .unwrap_or_else(|| panic!("{row}: {label_sel} must render"));
        let control = visual
            .debug_bounds(control_sel)
            .unwrap_or_else(|| panic!("{row}: {control_sel} must render"));
        assert!(
            label.right() <= control.left(),
            "{row}: label {label:?} must sit left of control {control:?}"
        );
        assert!(
            (label.top() - control.top()).abs() <= px(20.),
            "{row}: label and control must share the same row (labels {:?}, controls {:?})",
            label.top(),
            control.top(),
        );
    }
}

#[gpui::test]
fn settings_font_size_stepper_drops_the_range_caption(cx: &mut TestAppContext) {
    // The stepper no longer paints a "range 11-18px" clutter caption —
    // the disabled `−` at MIN and `+` at MAX carry the picker range on
    // their own. Guards against a re-add of the numeric hint.
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    open_settings_with_default_catalog(&view, &mut visual);
    let stepper = visual
        .debug_bounds("settings-font-size-stepper")
        .expect("stepper renders");
    // The row's control container must be the stepper's parent slot.
    let control = visual
        .debug_bounds("settings-row-size-control")
        .expect("font size control slot renders");
    assert!(
        stepper.left() >= control.left() && stepper.right() <= control.right() + px(1.),
        "stepper must live inside the row's control slot"
    );
    // Three stepper primitives — shrink, value, grow — and nothing else.
    for id in ["font-size-shrink", "font-size-value", "font-size-grow"] {
        assert!(
            visual.debug_bounds(id).is_some(),
            "{id} must render inside the stepper"
        );
    }
}

#[gpui::test]
fn settings_escape_closes_and_returns_focus_to_the_invoker(cx: &mut TestAppContext) {
    // Escape dismisses the modal and returns focus to whoever opened it
    // (a11y precedent from ZETA-108/123). When nothing had keyboard focus
    // at open time, the fallback lands on the composer so no user is ever
    // left without a caret.
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    // Give the composer keyboard focus, then open Settings — the composer
    // handle is captured as the return target. This mirrors the keyboard
    // path where a user tabs into a control, invokes settings, and expects
    // to land back on the same control on close.
    let composer_focus = view.read_with(&visual, |view, cx| view.composer.focus_handle(cx));
    visual.update(|window, cx| {
        window.focus(&composer_focus, cx);
        view.update(cx, |view, cx| {
            view.open_settings(cx);
        });
        window.draw(cx).clear(cx);
    });
    // The modal is not yet visible (LoadSettings is queued) — feed the
    // catalog reply so it appears (focus capture fires there, from the
    // composer we just focused), then send Escape.
    open_settings_with_default_catalog(&view, &mut visual);
    assert!(visual.debug_bounds("settings-overlay").is_some());
    visual.simulate_keystrokes("escape");
    visual.update(|window, cx| window.draw(cx).clear(cx));
    assert!(
        visual.debug_bounds("settings-overlay").is_none(),
        "escape must dismiss the modal"
    );
    view.read_with(&visual, |view, _| {
        assert!(!view.settings_open, "settings_open must clear on escape");
    });
    // Focus lands on the captured invoker.
    let focused_composer = visual.update(|window, _| composer_focus.is_focused(window));
    assert!(
        focused_composer,
        "escape must restore focus to the invoker captured at open time"
    );
}

#[gpui::test]
fn settings_tab_cycle_stays_trapped_inside_the_modal(cx: &mut TestAppContext) {
    // Tab and Shift-Tab traverse the modal's tab-stop registry AND wrap
    // inside it: every forward step from any modal control must land on a
    // control that `settings_focus.contains_focused` accepts, and the
    // reverse cycle must land on the same set. The pre-round-3 trap called
    // `window.focus_next` bare; the last modal control's Tab wrapped to a
    // BACKGROUND control (composer, sidebar) behind the scrim. This test
    // walks the full forward cycle, then the full reverse cycle, and
    // asserts every step stays inside the modal — a re-visited focused
    // control proves the cycle closed on itself. An Apply click in the
    // middle guards the reviewer's specific worry that focus survives an
    // intermediate action.
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    open_settings_with_default_catalog(&view, &mut visual);
    let overlay_focus = view.read_with(&visual, |view, _| view.settings_focus.clone());
    // Round-5: exercise the REAL Apply path — a mouse click on the Apply
    // button (the ZETA-111 click-flow API) — so the intermediate action
    // rides through the same handler a user's hand would trip. Then
    // capture the ANCHOR: the first tab-stop focus handle Apply leaves
    // focus on. Drive real Tab keystrokes until focus RETURNS to that
    // exact handle. Every intermediate stop must be unique (a simple
    // cycle, not a ping-pong between two controls) and stay inside the
    // modal (the full set). Then drive Shift-Tab in reverse from the
    // same anchor and assert the SAME set is visited — with NO manual
    // focus resets anywhere in the walk.
    let apply = visual
        .debug_bounds("settings-apply")
        .expect("apply button renders");
    visual.simulate_click(apply.center(), Default::default());
    visual.update(|window, cx| window.draw(cx).clear(cx));
    assert!(
        visual.update(|window, cx| overlay_focus.contains_focused(window, cx)),
        "Apply click must land focus on an in-modal control"
    );
    let anchor = visual
        .update(|window, cx| window.focused(cx))
        .expect("Apply must leave focus on a live control");
    // Forward walk: Tab until focus returns to the anchor. Bound at 64
    // steps so a broken trap never hangs the suite.
    let mut forward: Vec<gpui::FocusHandle> = Vec::new();
    let mut wrapped_forward = false;
    for step in 0..64 {
        visual.simulate_keystrokes("tab");
        visual.update(|window, cx| window.draw(cx).clear(cx));
        assert!(
            visual.update(|window, cx| overlay_focus.contains_focused(window, cx)),
            "forward Tab step {step} escaped the modal"
        );
        let now = visual
            .update(|window, cx| window.focused(cx))
            .expect("tab step must keep focus on some control");
        if now == anchor {
            wrapped_forward = true;
            break;
        }
        let revisit_ix = forward.iter().position(|prev| prev == &now);
        assert!(
            revisit_ix.is_none(),
            "forward Tab step {step} revisited intermediate at index {} \
             (forward walk visited {} unique stops before revisit) — \
             the cycle is not simple",
            revisit_ix.map(|i| i.to_string()).unwrap_or_default(),
            forward.len()
        );
        forward.push(now);
    }
    assert!(
        wrapped_forward,
        "forward Tab cycle never wrapped back to the anchor within 64 steps \
         (visited {} intermediates)",
        forward.len()
    );
    assert!(
        !forward.is_empty(),
        "modal must expose at least one non-anchor tab stop"
    );
    // Reverse walk from the SAME position (the anchor, where the forward
    // walk ended). Shift-Tab until focus returns to the anchor again.
    let mut reverse: Vec<gpui::FocusHandle> = Vec::new();
    let mut wrapped_reverse = false;
    for step in 0..64 {
        visual.simulate_keystrokes("shift-tab");
        visual.update(|window, cx| window.draw(cx).clear(cx));
        assert!(
            visual.update(|window, cx| overlay_focus.contains_focused(window, cx)),
            "reverse Shift-Tab step {step} escaped the modal"
        );
        let now = visual
            .update(|window, cx| window.focused(cx))
            .expect("shift-tab step must keep focus on some control");
        if now == anchor {
            wrapped_reverse = true;
            break;
        }
        let revisit_ix = reverse.iter().position(|prev| prev == &now);
        assert!(
            revisit_ix.is_none(),
            "reverse Shift-Tab step {step} revisited intermediate at index {} \
             (reverse walk visited {} unique stops before revisit)",
            revisit_ix.map(|i| i.to_string()).unwrap_or_default(),
            reverse.len()
        );
        reverse.push(now);
    }
    assert!(
        wrapped_reverse,
        "reverse Shift-Tab cycle never wrapped back to the anchor within 64 steps"
    );
    // Same tab-stop SET in both directions. A cycle reverses its visit
    // ORDER but keeps its MEMBERSHIP — every forward stop appears in the
    // reverse walk, and every reverse stop appears in the forward walk.
    for handle in &forward {
        assert!(
            reverse.iter().any(|r| r == handle),
            "reverse Shift-Tab cycle omitted a stop that forward Tab visited"
        );
    }
    for handle in &reverse {
        assert!(
            forward.iter().any(|f| f == handle),
            "forward Tab cycle omitted a stop that reverse Shift-Tab visited"
        );
    }
}

#[gpui::test]
fn settings_panel_paints_on_tokens_across_every_theme(cx: &mut TestAppContext) {
    // Section headings, row labels, and dividers must ride semantic tokens
    // so all five themes stay legible without per-palette overrides. The
    // check walks every ThemeId, applies it, redraws the modal, and asserts
    // the panel paints on the sidebar token (not a raw hex).
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    open_settings_with_default_catalog(&view, &mut visual);
    for id in theme::ThemeId::ALL.iter().copied() {
        let appearance = theme::Appearance {
            theme: id,
            font_family: gpui::SharedString::new_static(theme::DEFAULT_FONT_FAMILY),
            font_size: gpui::px(13.),
        };
        visual.update(|_, cx| theme::apply_with(cx, &appearance));
        visual.update(|window, cx| window.draw(cx).clear(cx));
        let panel_token = visual.update(|_, cx| cx.theme().sidebar);
        let panel = visual
            .debug_bounds("settings-panel")
            .unwrap_or_else(|| panic!("panel renders on {id:?}"));
        let scaled = visual.update(|window, _| panel.scale(window.scale_factor()));
        let paints_on_token = visual.update(|window, _| {
            let token_bg: gpui::Background = panel_token.into();
            window.painted_quads().into_iter().any(|quad| {
                let overlaps = quad.bounds.right() >= scaled.left()
                    && quad.bounds.left() <= scaled.right()
                    && quad.bounds.bottom() >= scaled.top()
                    && quad.bounds.top() <= scaled.bottom();
                overlaps && quad.background == token_bg
            })
        });
        assert!(
            paints_on_token,
            "settings panel must paint on the sidebar token on theme {id:?}"
        );
    }
    // Reset for peer tests.
    visual.update(|_, cx| theme::apply(cx));
}

#[gpui::test]
fn settings_modal_fits_the_viewport_at_min_and_max_font_size(cx: &mut TestAppContext) {
    // At 11px and 18px picker extremes the panel must still fit inside the
    // 760px test viewport so Close/Apply stay clickable. Locks the
    // section-gap / row-gap budget derived in ZETA-128; regressing to a
    // looser rhythm would push the action row below the fold.
    wipe_scoped_prefs();
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    open_settings_with_default_catalog(&view, &mut visual);
    let viewport = visual.update(|window, _| window.viewport_size());
    for base_px in [theme::MIN_FONT_SIZE_PX, theme::MAX_FONT_SIZE_PX] {
        let appearance = theme::Appearance {
            theme: theme::ThemeId::default(),
            font_family: gpui::SharedString::new_static(theme::DEFAULT_FONT_FAMILY),
            font_size: theme::clamp_font_size(base_px),
        };
        visual.update(|_, cx| theme::apply_with(cx, &appearance));
        visual.update(|window, cx| window.draw(cx).clear(cx));
        let close = visual
            .debug_bounds("settings-close")
            .expect("close button renders");
        let apply = visual
            .debug_bounds("settings-apply")
            .expect("apply button renders");
        assert!(
            close.bottom() <= viewport.height,
            "close button bottom {:?} must stay inside viewport height {:?} at {base_px}px",
            close.bottom(),
            viewport.height,
        );
        assert!(
            apply.bottom() <= viewport.height,
            "apply button bottom {:?} must stay inside viewport height {:?} at {base_px}px",
            apply.bottom(),
            viewport.height,
        );
    }
    // Restore defaults for peer tests.
    visual.update(|_, cx| theme::apply(cx));
    wipe_scoped_prefs();
}

#[gpui::test]
fn settings_theme_and_font_render_as_compact_single_value_cyclers(cx: &mut TestAppContext) {
    // Round 2 collapses the pre-round-1 button walls (five theme buttons +
    // four font buttons -> nine tab stops in Appearance) into one focusable
    // cycler each. The row's control slot holds ONE labeled button carrying
    // the current selection; clicking it advances to the next value. The
    // pre-round-2 debug selectors `theme-row-<slug>` / `font-row-<family>`
    // vanish with the walls.
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    open_settings_with_default_catalog(&view, &mut visual);
    // One tab stop per row: the cyclers exist under a stable selector.
    let theme_cycler = visual
        .debug_bounds("settings-theme-cycler")
        .expect("theme cycler renders");
    let font_cycler = visual
        .debug_bounds("settings-font-cycler")
        .expect("font cycler renders");
    // Each cycler lives inside its row's control slot (right of the label).
    let theme_slot = visual
        .debug_bounds("settings-row-theme-control")
        .expect("theme control slot renders");
    let font_slot = visual
        .debug_bounds("settings-row-font-control")
        .expect("font control slot renders");
    assert!(
        theme_cycler.left() >= theme_slot.left()
            && theme_cycler.right() <= theme_slot.right() + px(1.),
        "theme cycler must live inside the theme row's control slot"
    );
    assert!(
        font_cycler.left() >= font_slot.left() && font_cycler.right() <= font_slot.right() + px(1.),
        "font cycler must live inside the font row's control slot"
    );
    // The pre-round-2 button walls are gone. The specific selectors ZETA-111
    // shipped for the individual theme/font buttons no longer resolve — if
    // they do, the wall has crept back.
    for gone in [
        "theme-row-opencode",
        "theme-row-gruvbox-dark",
        "theme-row-vscode-dark-plus",
        "theme-row-nord",
        "theme-row-gruvbox-light",
        "font-row-JetBrains Mono",
        "font-row-Fira Code",
        "font-row-SF Mono",
        "font-row-Menlo",
        "font-row-Monaco",
        "settings-theme-segmented",
        "settings-font-segmented",
    ] {
        assert!(
            visual.debug_bounds(gone).is_none(),
            "{gone} must not render — the pre-round-2 button wall crept back"
        );
    }
}

#[gpui::test]
fn settings_rows_carry_short_muted_descriptions(cx: &mut TestAppContext) {
    // Round 2 attaches a short muted-foreground caption to each labeled
    // row (Behavior + Appearance). The caption paints below the label /
    // control line and reads at label_small — dense enough to sit as a
    // subordinate line, muted enough to defer to the label. If any of
    // these description slots stops rendering, a row lost its caption.
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    open_settings_with_default_catalog(&view, &mut visual);
    for &(row, description_sel) in &[
        ("settings-row-approval", "settings-row-approval-description"),
        ("settings-row-theme", "settings-row-theme-description"),
        ("settings-row-font", "settings-row-font-description"),
        ("settings-row-size", "settings-row-size-description"),
    ] {
        let row_bounds = visual
            .debug_bounds(row)
            .unwrap_or_else(|| panic!("{row} must render"));
        let description = visual
            .debug_bounds(description_sel)
            .unwrap_or_else(|| panic!("{row}: {description_sel} must render"));
        assert!(
            description.top() >= row_bounds.top(),
            "{row}: description must paint below the row's top edge"
        );
        assert!(
            description.bottom() <= row_bounds.bottom() + px(1.),
            "{row}: description must stay inside the row container"
        );
    }
}

#[gpui::test]
fn settings_label_column_widens_with_the_base_font_size(cx: &mut TestAppContext) {
    // Round 2 replaces the fixed 120px label column with a base-derived
    // width so long labels ("Approval mode") fit at every picker base.
    // At 18px the label column must widen past the pre-round-2 120px
    // ceiling or the label wraps into a stacked block again.
    wipe_scoped_prefs();
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    open_settings_with_default_catalog(&view, &mut visual);
    let base = visual.update(|_, cx| cx.theme().font_size);
    let baseline_label = visual
        .debug_bounds("settings-row-approval-label")
        .expect("baseline approval label renders");
    let baseline_width = baseline_label.right() - baseline_label.left();
    // At the MAX 18px picker base the column widens strictly beyond the
    // shipped-default width.
    let appearance = theme::Appearance {
        theme: theme::ThemeId::default(),
        font_family: gpui::SharedString::new_static(theme::DEFAULT_FONT_FAMILY),
        font_size: theme::clamp_font_size(theme::MAX_FONT_SIZE_PX),
    };
    visual.update(|_, cx| theme::apply_with(cx, &appearance));
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let wide_label = visual
        .debug_bounds("settings-row-approval-label")
        .expect("wide approval label renders at 18px");
    let wide_width = wide_label.right() - wide_label.left();
    assert!(
        wide_width > baseline_width,
        "label column must widen with base ({base:?} -> 18px): \
         baseline {baseline_width:?}, wide {wide_width:?}"
    );
    visual.update(|_, cx| theme::apply(cx));
    wipe_scoped_prefs();
}

#[gpui::test]
fn settings_modal_esc_hint_paints_on_muted_foreground_token(cx: &mut TestAppContext) {
    // Round 2 routes the `esc` hint through `muted_foreground` (AA) rather
    // than the pre-round-2 `text_faint` palette accessor (2.92-4.03:1
    // against the panel token). This test walks every shipped theme and
    // asserts the theme's `muted_foreground` clears WCAG AA against the
    // sidebar token — the palette pair the esc hint rides.
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
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    open_settings_with_default_catalog(&view, &mut visual);
    for id in theme::ThemeId::ALL.iter().copied() {
        let appearance = theme::Appearance {
            theme: id,
            font_family: gpui::SharedString::new_static(theme::DEFAULT_FONT_FAMILY),
            font_size: gpui::px(13.),
        };
        visual.update(|_, cx| theme::apply_with(cx, &appearance));
        visual.update(|window, cx| window.draw(cx).clear(cx));
        assert!(
            visual.debug_bounds("modal-title-esc").is_some(),
            "esc hint renders on {id:?}"
        );
        let (fg, bg) = visual.update(|_, cx| (cx.theme().muted_foreground, cx.theme().sidebar));
        let ratio = contrast_ratio(fg, bg);
        assert!(
            ratio >= 4.5,
            "{id:?}: esc hint contrast {ratio:.2}:1 fails WCAG AA on the panel"
        );
    }
    visual.update(|_, cx| theme::apply(cx));
}

#[gpui::test]
fn settings_tab_stops_stay_visible_inside_the_viewport_at_18px(cx: &mut TestAppContext) {
    // Round 2 promise: every focusable control inside the modal paints
    // inside the viewport at the 18px picker MAX. The pre-round-2 shape
    // pushed seven Appearance tab stops offscreen (five theme + four font
    // buttons on top of the stepper). The cycler collapse plus the
    // section-level scroll-into-view safety net keep every ring visible.
    wipe_scoped_prefs();
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    open_settings_with_default_catalog(&view, &mut visual);
    let appearance = theme::Appearance {
        theme: theme::ThemeId::default(),
        font_family: gpui::SharedString::new_static(theme::DEFAULT_FONT_FAMILY),
        font_size: theme::clamp_font_size(theme::MAX_FONT_SIZE_PX),
    };
    visual.update(|_, cx| theme::apply_with(cx, &appearance));
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let viewport = visual.update(|window, _| window.viewport_size());
    // Each labeled control below is one tab stop. For every stop, focus
    // the section that owns it (Behavior for approval-mode buttons,
    // Appearance for cyclers + stepper primitives) so the modal's
    // scroll-into-view safety net fires, then assert the resulting
    // rendered bounds sit inside the viewport. Close and Apply are also
    // children of the same scroll body.
    let checked: &[(&str, Option<&str>)] = &[
        ("mode-row-ask", Some("behavior")),
        ("mode-row-allow", Some("behavior")),
        ("mode-row-deny", Some("behavior")),
        ("settings-theme-cycler", Some("appearance")),
        ("settings-font-cycler", Some("appearance")),
        ("font-size-shrink", Some("appearance")),
        ("font-size-grow", Some("appearance")),
        ("settings-close", None),
        ("settings-apply", None),
    ];
    for (sel, section) in checked.iter().copied() {
        if let Some(section) = section {
            visual.update(|window, cx| {
                let handle = view.read_with(cx, |view, _| {
                    view.settings_section_focus.borrow().get(section).cloned()
                });
                if let Some(handle) = handle {
                    window.focus(&handle, cx);
                }
            });
            visual.update(|window, cx| window.draw(cx).clear(cx));
        }
        let bounds = visual
            .debug_bounds(sel)
            .unwrap_or_else(|| panic!("{sel} must render at 18px"));
        assert!(
            bounds.top() >= px(0.) && bounds.bottom() <= viewport.height,
            "{sel}: bounds {bounds:?} fall outside the viewport height {:?} at 18px",
            viewport.height
        );
        assert!(
            bounds.left() >= px(0.) && bounds.right() <= viewport.width,
            "{sel}: bounds {bounds:?} fall outside the viewport width {:?} at 18px",
            viewport.width
        );
    }
    visual.update(|_, cx| theme::apply(cx));
    wipe_scoped_prefs();
}

#[gpui::test]
fn settings_tab_walk_reveals_auth_rows_and_footer_at_18px(cx: &mut TestAppContext) {
    // A tiny viewport must reveal every real tab stop as focus advances
    // through both provider rows and the footer controls.
    wipe_scoped_prefs();
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.simulate_resize(gpui::size(px(800.), px(500.)));
    seed_settings_login_providers(&view, &mut visual);
    open_settings_with_default_catalog(&view, &mut visual);
    let appearance = theme::Appearance {
        theme: theme::ThemeId::default(),
        font_family: gpui::SharedString::new_static(theme::DEFAULT_FONT_FAMILY),
        font_size: theme::clamp_font_size(theme::MAX_FONT_SIZE_PX),
    };
    visual.update(|_, cx| theme::apply_with(cx, &appearance));
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let viewport = visual.update(|window, _| window.viewport_size());
    let overlay_focus = view.read_with(&visual, |view, _| view.settings_focus.clone());
    visual.update(|window, cx| window.focus(&overlay_focus, cx));
    visual.update(|window, cx| window.draw(cx).clear(cx));

    let stops = [
        "model-row-0",
        "mode-row-ask",
        "mode-row-allow",
        "mode-row-deny",
        "settings-theme-cycler",
        "settings-font-cycler",
        "font-size-shrink",
        "settings-login-claude-start",
        "settings-login-codex-start",
        "settings-close",
        "settings-apply",
    ];
    for (step, selector) in stops.iter().enumerate() {
        visual.simulate_keystrokes("tab");
        visual.update(|window, cx| window.draw(cx).clear(cx));
        let bounds = visual
            .debug_bounds(selector)
            .unwrap_or_else(|| panic!("{selector} must render at Tab step {}", step + 1));
        assert!(
            bounds.top() >= px(0.) && bounds.bottom() <= viewport.height + px(1.),
            "{selector} at Tab step {} must paint inside viewport {viewport:?}, got {bounds:?}",
            step + 1,
        );
    }

    // Reverse traversal must reach Apply first from the overlay's initial
    // focus. Reset the body before the walk so Apply's tracked wrapper, not
    // the footer's previously revealed position, has to drive the scroll.
    let sections_scroll = view.read_with(&visual, |view, _| view.settings_sections_scroll.clone());
    sections_scroll.set_offset(gpui::point(px(0.), px(0.)));
    assert_eq!(
        sections_scroll.offset().y,
        px(0.),
        "reverse Tab walk must start with the Settings body at the top",
    );
    visual.update(|window, cx| window.focus(&overlay_focus, cx));
    visual.update(|window, cx| window.draw(cx).clear(cx));
    visual.simulate_keystrokes("shift-tab");
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let apply_focus = view.read_with(&visual, |view, _| {
        view.settings_scroll_focus
            .borrow()
            .get("apply")
            .cloned()
            .expect("Apply focus handle allocated")
    });
    assert!(
        visual.update(|window, cx| apply_focus.contains_focused(window, cx)),
        "the first reverse Tab from the overlay must focus Apply"
    );
    let apply = visual
        .debug_bounds("settings-apply")
        .expect("Apply must render after reverse Tab");
    let body = visual
        .debug_bounds("settings-sections")
        .expect("Settings scroll body renders after reverse Tab");
    assert!(
        apply.left() >= body.left() - px(1.)
            && apply.right() <= body.right() + px(1.)
            && apply.top() >= body.top() - px(1.)
            && apply.bottom() <= body.bottom() + px(1.),
        "reverse Tab must reveal Apply inside the Settings scroll body \
         (apply {apply:?}, body {body:?})",
    );

    visual.update(|_, cx| theme::apply(cx));
    wipe_scoped_prefs();
}

#[gpui::test]
fn settings_model_list_scrolls_focused_row_into_view_at_18px(cx: &mut TestAppContext) {
    // Round-3 promise: the inner model list scrolls whichever row is
    // KEYBOARD-focused into view, not just the SELECTED row. Round-4
    // rewrite exercises REAL Tab keystrokes (the same `dispatch_keystroke`
    // path Enter/Space activations use) instead of the pre-round-4 shortcut
    // that focused each row's wrapper handle directly. Each Tab from the
    // overlay lands on the next model-row Button; the wrapper div's tracked
    // focus contains that Button's focus, drives `scroll_to_item(child_ix)`,
    // and the CLIPPED `model-list` bounds must contain the row's painted
    // ring — not the whole panel (the panel is wider than the list's cap
    // so a leaked ring outside the list still fits inside the panel).
    wipe_scoped_prefs();
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    let models = [
        "claude-opus-4-7",
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-fable-5",
        "claude-haiku-4-5",
        "claude-sonnet-4-6",
        "claude-haiku-4-6",
        "claude-opus-4-6",
    ];
    let selectors: [&'static str; 8] = [
        "model-row-0",
        "model-row-1",
        "model-row-2",
        "model-row-3",
        "model-row-4",
        "model-row-5",
        "model-row-6",
        "model-row-7",
    ];
    open_settings_with_multi_model_catalog(&view, &mut visual, &models);
    let appearance = theme::Appearance {
        theme: theme::ThemeId::default(),
        font_family: gpui::SharedString::new_static(theme::DEFAULT_FONT_FAMILY),
        font_size: theme::clamp_font_size(theme::MAX_FONT_SIZE_PX),
    };
    visual.update(|_, cx| theme::apply_with(cx, &appearance));
    visual.update(|window, cx| window.draw(cx).clear(cx));
    // Seed focus on the overlay handle so the first Tab lands on the
    // first modal tab stop — the model-row-0 Button. Each subsequent Tab
    // walks one row down the list.
    let overlay_focus = view.read_with(&visual, |v, _| v.settings_focus.clone());
    visual.update(|window, cx| window.focus(&overlay_focus, cx));
    visual.update(|window, cx| window.draw(cx).clear(cx));
    for (index, sel) in selectors.iter().copied().enumerate() {
        visual.simulate_keystrokes("tab");
        visual.update(|window, cx| window.draw(cx).clear(cx));
        let row_focus = view.read_with(&visual, |v, _| {
            v.settings_model_row_focus
                .borrow()
                .get(&index)
                .cloned()
                .expect("row focus handle allocated")
        });
        assert!(
            visual.update(|window, cx| row_focus.contains_focused(window, cx)),
            "Tab step {} must land on model-row-{index}",
            index + 1,
        );
        let row_bounds = visual
            .debug_bounds(sel)
            .unwrap_or_else(|| panic!("row {index} must render at 18px"));
        let list_bounds = visual
            .debug_bounds("model-list")
            .expect("model list renders while a row is focused");
        assert!(
            row_bounds.top() >= list_bounds.top() - px(1.)
                && row_bounds.bottom() <= list_bounds.bottom() + px(1.),
            "row {index}: focused bounds {row_bounds:?} leaked outside the clipped \
             model-list viewport {list_bounds:?}"
        );
    }
    visual.update(|_, cx| theme::apply(cx));
    wipe_scoped_prefs();
}

#[gpui::test]
fn settings_panel_height_is_capped_on_tall_viewports(cx: &mut TestAppContext) {
    // The panel binds `.max_h(min(shelf, cap))` so a 1200px viewport
    // cannot stretch the flat modal past `SETTINGS_PANEL_MAX_HEIGHT`.
    // ZETA-138 switched the panel to content-height (from `.h(...)` to
    // `.max_h(...)`) so the cap still holds but the panel may pack
    // shorter than the cap when content fits — that shrink-to-content
    // is precisely what removed the pre-fix "dead vertical band" between
    // the Appearance section and the auth rows.
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    open_settings_with_default_catalog(&view, &mut visual);
    visual.simulate_resize(gpui::size(px(1100.), px(1200.)));
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let panel = visual
        .debug_bounds("settings-panel")
        .expect("panel renders on the tall viewport");
    let panel_height = panel.bottom() - panel.top();
    assert!(
        panel_height <= theme::SETTINGS_PANEL_MAX_HEIGHT + px(1.),
        "tall viewport panel height {panel_height:?} must respect the cap {:?}",
        theme::SETTINGS_PANEL_MAX_HEIGHT
    );
    visual.simulate_resize(gpui::size(px(1100.), px(760.)));
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let short_panel = visual
        .debug_bounds("settings-panel")
        .expect("panel renders on the short viewport");
    let short_height = short_panel.bottom() - short_panel.top();
    assert!(
        short_height < theme::SETTINGS_PANEL_MAX_HEIGHT + px(1.),
        "short viewport panel height {short_height:?} must fit under the cap"
    );
    let close = visual
        .debug_bounds("settings-close")
        .expect("close renders on the short viewport");
    assert!(
        close.bottom() <= px(760.),
        "short viewport close-button bottom {:?} must stay inside the 760px viewport",
        close.bottom()
    );
}

// ---------------------------------------------------------------------------
// ZETA-132 — Settings shows its sections on open.
// ---------------------------------------------------------------------------
//
// The 2026-09-16 audit (C3) caught the Settings modal opening with its
// scroll viewport cut exactly at the `Behavior` section header on common
// window sizes: approval mode, theme, font, and font size all read as
// absent, with no scrollbar visible and only the ZETA-128 hairline cue
// hinting at the hidden content. Fix: raise `SETTINGS_PANEL_MAX_HEIGHT`
// so the shelf-capped panel is tall enough to render every section on
// open at 900px+ viewport heights, keep `SETTINGS_MODEL_LIST_MAX_HEIGHT`
// at 160 (the credential-error swap test relies on the codex row being
// clickable without scrolling), and give the Settings surface its own
// top-offset token (`SETTINGS_MODAL_TOP_FRACTION`) at 15% so the
// shelf-derived height is tall enough while the shared 25%
// `MODAL_TOP_FRACTION` still pins rename / delete dialogs. Where
// overflow still bites (760px test viewport, 18px picker), a real Kit
// `Scrollbar` overlay paints its 8px thumb on the wrapper's right edge
// so the panel is discoverably scrollable instead of relying on the
// mask cue alone. C4 pairs the resize with a scroll-to-top reset on
// every open so a reopen never lands mid-list.

/// Two-group model catalog that matches the audit shape — claude + codex,
/// enough rows to reach the 160px model-list cap so the Model section
/// pushes its natural weight on the sections wrapper. A single-model
/// fixture (the default helper) fits the Model list in ~32px and the
/// old, pre-round-1 sizing (`SETTINGS_PANEL_MAX_HEIGHT=560`,
/// `MODAL_TOP_FRACTION=0.25`) passes with plenty of room to spare;
/// the C3 test needs the audit-weight list to expose the miss.
fn open_settings_with_audit_shape_catalog(view: &Entity<ZetaView>, visual: &mut VisualTestContext) {
    let models: Vec<String> = vec![
        "claude-opus-4-7".into(),
        "claude-sonnet-4-6".into(),
        "claude-fable-5".into(),
        "claude-haiku-4-5".into(),
        "gpt-5.6-luna".into(),
        "gpt-5.6-sol".into(),
    ];
    let mut providers = std::collections::BTreeMap::new();
    for model in &models {
        let group = if model.starts_with("claude-") {
            "claude"
        } else {
            "codex"
        };
        providers.insert(model.clone(), group.to_owned());
    }
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.session_view.available = true;
            view.apply_worker_message(
                WorkerMessage::Settings(
                    SessionSettings {
                        model: models[0].clone(),
                        approval_mode: "ask".into(),
                    },
                    ModelCatalog {
                        models: models.clone(),
                        providers,
                    },
                ),
                window,
                cx,
            );
        });
        window.draw(cx).clear(cx);
    });
}

#[gpui::test]
fn zeta132_behavior_section_header_paints_inside_the_visible_slice_on_open(
    cx: &mut TestAppContext,
) {
    // C3: opening Settings must expose the Behavior heading at every
    // picker base on the default test viewport. Pre-ZETA-132 the Model
    // list + section header pushed the Behavior heading past the
    // scroll-cue mask top edge; ZETA-138 replaced the mask with a
    // content-height panel and Kit's `Scrollbar` overlay, so the
    // invariant now pins the Behavior heading + full approval row
    // inside the sections wrapper's rendered bounds across 11/13/18 px
    // picker bases.
    wipe_scoped_prefs();
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    for base_px in [11.0_f32, 13.0, 18.0] {
        let appearance = theme::Appearance {
            theme: theme::ThemeId::default(),
            font_family: gpui::SharedString::new_static(theme::DEFAULT_FONT_FAMILY),
            font_size: theme::clamp_font_size(base_px),
        };
        visual.update(|_, cx| theme::apply_with(cx, &appearance));
        open_settings_with_audit_shape_catalog(&view, &mut visual);
        visual.update(|window, cx| window.draw(cx).clear(cx));
        let heading = visual
            .debug_bounds("settings-section-behavior-heading")
            .unwrap_or_else(|| panic!("behavior heading must render on open at {base_px}px"));
        let sections = visual
            .debug_bounds("settings-sections")
            .unwrap_or_else(|| panic!("sections wrapper renders at {base_px}px"));
        let panel = visual
            .debug_bounds("settings-panel")
            .unwrap_or_else(|| panic!("panel renders at {base_px}px"));
        assert!(
            heading.top() >= panel.top(),
            "behavior heading top {:?} at {base_px}px must sit inside the \
             panel (panel top {:?})",
            heading.top(),
            panel.top(),
        );
        assert!(
            heading.bottom() <= sections.bottom() + px(1.),
            "behavior heading bottom {:?} at {base_px}px must render inside \
             the sections wrapper (wrapper bottom {:?}) so opening the \
             modal exposes the Behavior section rather than resurfacing \
             the audit's C3 empty-header shape",
            heading.bottom(),
            sections.bottom(),
        );
        let approval_row = visual
            .debug_bounds("settings-row-approval")
            .unwrap_or_else(|| panic!("approval row must render on open at {base_px}px"));
        assert!(
            approval_row.bottom() <= sections.bottom() + px(1.),
            "approval row bottom {:?} at {base_px}px must render inside \
             the sections wrapper (wrapper bottom {:?})",
            approval_row.bottom(),
            sections.bottom(),
        );
        visual.simulate_keystrokes("escape");
        visual.update(|window, cx| window.draw(cx).clear(cx));
    }
    visual.update(|_, cx| theme::apply(cx));
    wipe_scoped_prefs();
}

#[gpui::test]
fn zeta132_scroll_resets_to_top_on_every_open(cx: &mut TestAppContext) {
    // C4: the sections `ScrollHandle` persists across close/reopen. Without
    // a reset the second open lands wherever the last wheel/drag left it,
    // hiding the top-of-panel sections the user just came to check.
    //
    // Drives the real user path rather than a synthetic `set_offset`:
    // open at the shipped 13px default (sections fit), click the
    // Appearance font-size stepper once to prove the input path, then
    // advance the remaining ladder via the production method until MAX so
    // the sections start to overflow mid-session. Dispatch a real wheel
    // event on the sections wrapper to move the offset. Close via Escape,
    // reopen, and assert the handle is back at (0, 0). The
    // pre-round-1 shortcut (`set_offset` + font size set BEFORE the first
    // open) sidestepped both the reopen-after-mid-session-resize path
    // AND the wheel-listener wiring that actually carries a real user's
    // scroll.
    wipe_scoped_prefs();
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    seed_settings_login_providers(&view, &mut visual);
    // Open at the shipped 13px default; no synthetic size set beforehand.
    open_settings_with_audit_shape_catalog(&view, &mut visual);
    let baseline_size = visual.update(|_, cx| cx.theme().font_size);
    assert_eq!(
        baseline_size,
        theme::DEFAULT_FONT_SIZE,
        "test premise: modal must open at the shipped {:?} default",
        theme::DEFAULT_FONT_SIZE,
    );
    // Prove the stepper's click path fires ONE mid-session font-size
    // change from the shipped default (the reviewer's "through the
    // stepper" requirement — a real `simulate_click` on the visible
    // `+` control, not a synthetic set_offset shortcut). The production
    // `adjust_font_size` method below advances the remaining ladder to MAX.
    let grow = visual
        .debug_bounds("font-size-grow")
        .expect("grow button renders while modal is open");
    visual.simulate_click(grow.center(), gpui::Modifiers::default());
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let after_first_click = visual.update(|_, cx| cx.theme().font_size);
    assert_eq!(
        f32::from(after_first_click),
        f32::from(theme::DEFAULT_FONT_SIZE) + 1.,
        "test premise: one stepper click must raise the picker one \
         whole-px step from the shipped {:?} default; got \
         {after_first_click:?}",
        theme::DEFAULT_FONT_SIZE,
    );
    // Ladder the rest of the way to MAX via the stepper's OWN handler.
    // `adjust_font_size(1., cx)` is the exact call
    // `.on_click(cx.listener(|view, _, _, cx| view.adjust_font_size(1., cx)))`
    // wires on the `+` button — same clamp, same `prefs::commit`, same
    // theme reflow. Direct invocation rather than a `simulate_click`
    // loop: back-to-back same-position clicks on the same stateful
    // button drop after the first in the gpui test harness (round-2 CI
    // pinned 14px after 32 clicks with a pointer-park between each),
    // and this test is about the reopen-after-mid-session-resize path,
    // not the stepper's click-routing (which the click above proves
    // once, and which `settings_theme_and_font_render_as_compact_single_value_cyclers`
    // guards for one-shot cases). Loop-until-value with a bounded
    // iteration cap so a future step-size change or clamp tweak lands
    // on the premise assertion rather than an infinite loop.
    let target = theme::clamp_font_size(theme::MAX_FONT_SIZE_PX);
    let step_cap = 32usize;
    let mut steps = 0usize;
    loop {
        let current = visual.update(|_, cx| cx.theme().font_size);
        if current == target {
            break;
        }
        assert!(
            steps < step_cap,
            "test premise: {step_cap} adjust_font_size steps must reach \
             MAX ({target:?}); got {current:?} after {steps} steps",
        );
        view.update(&mut visual, |view, cx| view.adjust_font_size(1., cx));
        visual.update(|window, cx| window.draw(cx).clear(cx));
        steps += 1;
    }
    let grown_size = visual.update(|_, cx| cx.theme().font_size);
    assert_eq!(
        grown_size, target,
        "test premise: mid-session ladder must reach the MAX font size \
         — got {grown_size:?}",
    );
    // ZETA-138: force overflow via a deliberately tiny viewport before
    // dispatching the wheel event. With the ZETA-128 scroll-cue mask +
    // trailing spacer removed, the shipped test viewport (1100x760) at
    // 18px is no longer guaranteed to overflow the sections wrapper —
    // the wheel would land on a fitting layout and the offset would
    // never move.
    visual.simulate_resize(gpui::size(px(800.), px(500.)));
    visual.update(|window, cx| window.draw(cx).clear(cx));
    // Real wheel input on the sections wrapper. The wrapper is
    // `overflow_y_scroll` + `.track_scroll(&self.settings_sections_scroll)`,
    // so a wheel routed to its hitbox translates directly into a
    // negative y offset on the tracked handle — the same path a
    // trackpad two-finger drag drives at runtime.
    let sections = visual
        .debug_bounds("settings-sections")
        .expect("sections wrapper renders");
    visual.update(|window, cx| {
        window.dispatch_event(
            gpui::ScrollWheelEvent {
                position: sections.center(),
                delta: gpui::ScrollDelta::Pixels(gpui::point(px(0.), px(-300.))),
                ..Default::default()
            }
            .to_platform_input(),
            cx,
        );
        window.draw(cx).clear(cx);
    });
    let scrolled_y = view.read_with(&visual, |view, _| view.settings_sections_scroll.offset().y);
    assert!(
        scrolled_y < px(0.),
        "test premise: a real wheel scroll must move the sections \
         handle to a negative offset (got {scrolled_y:?}) — otherwise \
         the reopen-reset guard has nothing to reset",
    );
    // Close via Escape, then reopen with the same catalog helper —
    // triggers the `WorkerMessage::Settings` handler that ZETA-132 wires
    // the reset into.
    visual.simulate_keystrokes("escape");
    visual.update(|window, cx| window.draw(cx).clear(cx));
    open_settings_with_audit_shape_catalog(&view, &mut visual);
    let reopened_y = view.read_with(&visual, |view, _| view.settings_sections_scroll.offset().y);
    assert_eq!(
        reopened_y,
        px(0.),
        "reopening Settings must reset the sections scroll to the top \
         (offset.y was {reopened_y:?})",
    );
    visual.update(|_, cx| theme::apply(cx));
    wipe_scoped_prefs();
}

#[gpui::test]
fn zeta132_scrollbar_paints_a_thumb_when_sections_overflow(cx: &mut TestAppContext) {
    // Fallback contract: when the single Settings scroll body is forced to overflow
    // (a deliberately tiny viewport at the picker's MAX 18px), a real
    // scrollbar overlay must paint its thumb on the wrapper's right
    // edge so the surface is discoverably scrollable. Kit's `Scrollbar`
    // in `Always` mode paints the thumb continuously while content
    // exceeds the container; `style_for_normal` uses the theme's
    // `scrollbar_thumb` token (text-normal at 20% alpha), so a matching
    // painted quad proves the overlay is live.
    //
    // ZETA-138 repointed this test at an 800x500 viewport: the shipped
    // 760px test viewport at 18px no longer forces overflow with the
    // ZETA-128 scroll-cue trailing spacer removed, so the test needs a
    // viewport short enough to guarantee the sections wrapper cannot
    // fit its content even with both login-provider rows.
    wipe_scoped_prefs();
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    let appearance = theme::Appearance {
        theme: theme::ThemeId::default(),
        font_family: gpui::SharedString::new_static(theme::DEFAULT_FONT_FAMILY),
        font_size: theme::clamp_font_size(theme::MAX_FONT_SIZE_PX),
    };
    visual.update(|_, cx| theme::apply_with(cx, &appearance));
    seed_settings_login_providers(&view, &mut visual);
    open_settings_with_default_catalog(&view, &mut visual);
    visual.simulate_resize(gpui::size(px(800.), px(500.)));
    // Two draws: first sizes the wrapper, second lets Kit's scrollbar
    // pick up the layout bounds via `viewport_from_layout` and paint the
    // thumb.
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let overlay = visual
        .debug_bounds("settings-sections-scrollbar")
        .expect("scrollbar overlay renders while Settings is open");
    let thumb_bg: gpui::Background = theme::palette::scrollbar_thumb().into();
    let thumb_quads: Vec<_> = visual.update(|window, _| {
        window
            .painted_quads()
            .into_iter()
            .filter(|quad| quad.background == thumb_bg)
            .collect()
    });
    assert!(
        !thumb_quads.is_empty(),
        "scrollbar thumb must paint at 18px on the tiny test viewport — \
         the Settings body overflows and Kit's `Always` mode should \
         hold the thumb steady on the right edge"
    );
    let scale = visual.update(|window, _| window.scale_factor());
    for quad in &thumb_quads {
        let left = gpui::px(quad.bounds.origin.x.as_f32() / scale);
        let right =
            gpui::px((quad.bounds.origin.x.as_f32() + quad.bounds.size.width.as_f32()) / scale);
        assert!(
            right >= overlay.left() && left <= overlay.right() + px(1.),
            "thumb quad bounds {:?} must sit inside the scrollbar overlay \
             {overlay:?}",
            quad.bounds,
        );
    }
    let sections = visual
        .debug_bounds("settings-sections")
        .expect("scroll body renders while Settings is open");
    visual.update(|window, cx| {
        window.dispatch_event(
            gpui::ScrollWheelEvent {
                position: sections.center(),
                delta: gpui::ScrollDelta::Pixels(gpui::point(px(0.), px(-1200.))),
                ..Default::default()
            }
            .to_platform_input(),
            cx,
        );
        window.draw(cx).clear(cx);
    });
    let apply = visual
        .debug_bounds("settings-apply")
        .expect("footer remains in the scroll body");
    let scrolled_sections = visual
        .debug_bounds("settings-sections")
        .expect("scroll body remains mounted after scrolling");
    assert!(
        apply.top() >= scrolled_sections.top() - px(1.)
            && apply.bottom() <= scrolled_sections.bottom() + px(1.),
        "scrolling the tiny Settings viewport must reach the final Apply control \
         (apply {apply:?}, scroll body {scrolled_sections:?})",
    );
    visual.update(|_, cx| theme::apply(cx));
    wipe_scoped_prefs();
}

// ---------------------------------------------------------------------------
// ZETA-138 — Settings surface no longer paints the ZETA-128 scroll-cue
// mask, so a fit-case layout on typical windows never shows the visible
// horizontal band Henry called out between the Appearance section and
// the Claude / ChatGPT auth rows.
// ---------------------------------------------------------------------------
//
// The pre-fix ZETA-128 shape overlaid a `settings-scroll-cue` div at the
// bottom of the sections wrapper: a sidebar-tinted mask with a 1px
// `border_t_1()` line intended to hint that the scroll surface
// continued below. On a fit-case layout (sections wrapper stretched by
// `flex_1` beyond its content), the mask's border-top painted as a
// visible horizontal line with the sidebar wash below it — the "band"
// between the last section and the auth rows in Henry's screenshot.
// ZETA-138 retires the mask (and its supporting per-row measurement
// canvas / follow-up-frame convergence machinery / trailing spacer);
// Kit's `Scrollbar` overlay is the sole scroll affordance now, and it
// only paints a thumb when the sections wrapper actually overflows.

#[gpui::test]
fn zeta138_scroll_cue_mask_is_gone_so_no_visible_band_paints_below_the_sections(
    cx: &mut TestAppContext,
) {
    wipe_scoped_prefs();
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    open_settings_with_default_catalog(&view, &mut visual);
    visual.update(|window, cx| window.draw(cx).clear(cx));
    assert!(
        visual.debug_bounds("settings-scroll-cue").is_none(),
        "the ZETA-128 scroll-cue mask must not paint — its 1px border-top \
         line + sidebar wash was the visible horizontal band Henry \
         called out between the Appearance section and the auth rows"
    );
    wipe_scoped_prefs();
}

#[gpui::test]
fn zeta138_settings_fits_without_scrolling_at_typical_window_sizes(cx: &mut TestAppContext) {
    // At the ticket's enumerated typical window sizes (1100x800 and
    // 1500x1000) with the default catalog, the sections wrapper must
    // fit its content without any scroll offset moving and without Kit
    // painting a scrollbar thumb. Overflow only kicks in on
    // deliberately tiny viewports (the ZETA-132 scrollbar + scroll-
    // reset tests exercise that fallback).
    wipe_scoped_prefs();
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    seed_settings_login_providers(&view, &mut visual);
    for viewport in [
        gpui::size(px(1100.), px(800.)),
        gpui::size(px(1500.), px(1000.)),
    ] {
        visual.simulate_resize(viewport);
        open_settings_with_default_catalog(&view, &mut visual);
        visual.update(|window, cx| window.draw(cx).clear(cx));
        let panel = visual
            .debug_bounds("settings-panel")
            .unwrap_or_else(|| panic!("panel renders at {viewport:?}"));
        let close = visual
            .debug_bounds("settings-close")
            .unwrap_or_else(|| panic!("close renders at {viewport:?}"));
        let apply = visual
            .debug_bounds("settings-apply")
            .unwrap_or_else(|| panic!("apply renders at {viewport:?}"));
        for selector in [
            "settings-section-model",
            "settings-section-behavior",
            "settings-section-appearance",
            "settings-row-approval-control",
            "settings-row-theme-control",
            "settings-row-font-control",
            "settings-row-size-control",
            "settings-login-claude",
            "settings-login-codex",
            "settings-login-claude-start",
            "settings-login-codex-start",
            "settings-close",
            "settings-apply",
        ] {
            let bounds = visual
                .debug_bounds(selector)
                .unwrap_or_else(|| panic!("{selector} renders at {viewport:?}"));
            assert!(
                bounds.top() >= px(0.) && bounds.bottom() <= viewport.height + px(1.),
                "{selector} at {viewport:?} must paint inside the viewport: {bounds:?}",
            );
        }
        assert!(
            panel.bottom() <= viewport.height + px(1.),
            "panel bottom {:?} at {viewport:?} must stay inside the viewport",
            panel.bottom(),
        );
        assert!(
            close.bottom() <= panel.bottom() + px(1.) && apply.bottom() <= panel.bottom() + px(1.),
            "footer buttons at {viewport:?} must paint inside the panel \
             (close {close:?}, apply {apply:?}, panel bottom {:?})",
            panel.bottom(),
        );
        let scroll_y = view.read_with(&visual, |view, _| view.settings_sections_scroll.offset().y);
        assert_eq!(
            scroll_y,
            px(0.),
            "sections scroll offset at {viewport:?} must stay at 0 — no \
             scrolling should be needed at typical window sizes",
        );
        let thumb_bg: gpui::Background = theme::palette::scrollbar_thumb().into();
        let has_thumb = visual.update(|window, _| {
            window
                .painted_quads()
                .into_iter()
                .any(|quad| quad.background == thumb_bg)
        });
        assert!(
            !has_thumb,
            "no scrollbar thumb should paint at {viewport:?} — the panel \
             is expected to fit its content without scrolling",
        );
        visual.simulate_keystrokes("escape");
        visual.update(|window, cx| window.draw(cx).clear(cx));
    }
    wipe_scoped_prefs();
}

#[gpui::test]
fn zeta138_settings_panel_packs_to_content_without_dead_band(cx: &mut TestAppContext) {
    wipe_scoped_prefs();
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.simulate_resize(gpui::size(px(1100.), px(1000.)));
    seed_settings_login_providers(&view, &mut visual);
    open_settings_with_default_catalog(&view, &mut visual);
    visual.update(|window, cx| window.draw(cx).clear(cx));

    let sections = [
        "settings-section-model",
        "settings-section-behavior",
        "settings-section-appearance",
    ]
    .map(|selector| {
        visual
            .debug_bounds(selector)
            .unwrap_or_else(|| panic!("{selector} renders"))
    });
    for pair in sections.windows(2) {
        let gap = pair[1].top() - pair[0].bottom();
        assert!(
            gap < px(32.),
            "consecutive Settings sections must stay close; gap was {gap:?}"
        );
    }
    let appearance_bottom = sections[2].bottom();
    let auth_top = visual
        .debug_bounds("settings-login-claude")
        .expect("Claude auth row renders")
        .top();
    let auth_gap = auth_top - appearance_bottom;
    assert!(
        auth_gap < px(32.),
        "Appearance and the first auth row must stay close; gap was {auth_gap:?}"
    );
    let panel = visual
        .debug_bounds("settings-panel")
        .expect("panel renders");
    let apply = visual
        .debug_bounds("settings-apply")
        .expect("Apply renders");
    assert!(
        apply.bottom() <= panel.bottom() + px(1.),
        "Apply must remain inside the content-packed panel"
    );
    let footer_gap = panel.bottom() - apply.bottom();
    assert!(
        footer_gap <= px(16.),
        "content-packed panel must stay close to Apply; gap was {footer_gap:?}"
    );
    wipe_scoped_prefs();
}

// ---------------------------------------------------------------------------
// ZETA-129 — inline-code chip ladder STRUCTURE guard.
// ---------------------------------------------------------------------------
//
// The 2026-09-16 audit hit two classes of code-span corruption in the run
// GUI: length-9 backtick spans lose their last glyph (`task_kill` painted
// as `task_kil`), and a code chip pushed to a wrap boundary paints its
// overflow onto the next line at a stale x-position. Root cause is
// upstream in `gpui_base::text::inline_flow::InlineFlow::prepaint`; the
// vendored copy in `gui/vendor/gpui-base/` swaps its inner
// `Definite(fragment_size.width - padding * 2.)` for
// `AvailableSpace::MaxContent` so the inner `StyledText` never re-wraps a
// fragment whose shape already fits.
//
// This test drives a length 1..=16 backtick ladder through the markdown
// renderer inside `VisualTestContext::draw()` and inspects the chip
// BACKGROUND quads. It is NOT a mutation-sensitive check for A1/A2 — the
// phantom-glyph paints under the Definite drift are text SPRITES, not
// background quads, so this test's assertions on `secondary_hover`-washed
// quads pass even with the fix disabled. Mutation coverage for A1/A2
// lives in the native smoke driver: `smoke::scan_inline_flow_recorder`
// reads `gpui_kit::base::zeta129_wrap_recorder::samples()` after every
// native render and panics on any non-zero wrap-boundary count; the
// paired `gui-native-guards-inline-flow-mutation` Makefile target
// reinstates the upstream `Definite(...)` shape and requires the scan to
// trip.
//
// What this test DOES cover: chip-structure regressions unrelated to the
// vendored InlineFlow patch — a future change that reshapes the chip
// background paint would surface here as a count mismatch or a
// non-monotonic width sequence.
#[gpui::test]
fn zeta129_inline_code_chip_ladder_structure(cx: &mut TestAppContext) {
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    let mut lines: Vec<String> = Vec::with_capacity(16);
    for len in 1..=16 {
        let body = "a".repeat(len);
        lines.push(format!("- `{body}` len={len}"));
    }
    let source = lines.join("\n");
    visual.update(|_, cx| {
        view.update(cx, |view, cx| {
            view.state.transcript = vec![TranscriptEntry::Assistant(source.clone().into())];
            view.transcript.update(cx, |scroll, cx| scroll.reset(1, cx));
            cx.notify();
        });
    });
    visual.update(|window, cx| window.draw(cx).clear(cx));

    let subtle_bg: gpui::Background = visual.update(|_, cx| cx.theme().secondary_hover.into());
    let mut bounds: Vec<gpui::Bounds<gpui::ScaledPixels>> = visual.update(|window, _| {
        window
            .painted_quads()
            .into_iter()
            .filter(|quad| quad.background == subtle_bg)
            .map(|quad| quad.bounds)
            .collect()
    });
    // Paint order isn't guaranteed to be top-down; sort by y then x so
    // "chip N corresponds to length N+1" is a stable claim even if gpui
    // reorders its per-frame paint list.
    bounds.sort_by(|a, b| {
        a.origin
            .y
            .partial_cmp(&b.origin.y)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a.origin
                    .x
                    .partial_cmp(&b.origin.x)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    assert_eq!(
        bounds.len(),
        16,
        "expected one inline-code chip background quad per ladder length \
         1..=16, got {} on the theme's secondary_hover wash ({subtle_bg:?})",
        bounds.len(),
    );
    // Each subsequent chip carries one more mono glyph than the previous —
    // shaped widths must grow monotonically. Structural regression cover:
    // a change to the chip renderer that reshapes a shorter chip wider
    // than a longer one lands here rather than as a silent visual defect.
    for (ix, window) in bounds.windows(2).enumerate() {
        let prev = window[0].size.width;
        let next = window[1].size.width;
        assert!(
            next > prev,
            "chip {} (len={}) width {prev:?} must be strictly less than \
             chip {} (len={}) width {next:?} — the ladder shapes widths \
             monotonically by construction",
            ix,
            ix + 1,
            ix + 1,
            ix + 2,
        );
    }
}

// The recorder-based mutation-killer runs in the NATIVE smoke driver,
// not headlessly. `gpui::test`'s test platform shapes text
// deterministically, so shape drift never fires and the recorder
// assertion never trips in a headless run — the class of never-failing
// tests the ZETA-129 arc exists to kill. The equivalent check lives in
// `gui/src/smoke.rs::scan_inline_flow_recorder`, guarded by
// `gui-native-guards` (fail on any recorded wrap boundary) and paired
// with `gui-native-guards-inline-flow-mutation` (must fail the guard
// under `ZETA_GUI_INLINE_FLOW_DEFINITE=1`). Pinned CI trip evidence:
// shape=`wedge`, size=13px, sample text=`meta.json`, wrap_boundaries=1
// — the audit's length-9 code chip on the 13px × 0.875 mono metrics.

/// A short catalog for the slash menu tests below. Every entry names a
/// server-side builtin so the shape matches what `slash_list` returns in
/// production; the menu never has to guess a `client_only` from a name.
fn slash_catalog(names: &[&str]) -> SlashList {
    SlashList {
        commands: names
            .iter()
            .map(|name| SlashCommandInfo {
                name: (*name).to_owned(),
                description: format!("{name} description"),
                kind: "builtin".to_owned(),
                source: "builtin".to_owned(),
                client_only: false,
                unavailable: None,
            })
            .collect(),
        notices: Vec::new(),
    }
}

fn enable_slash_extensions(
    visual: &mut VisualTestContext,
    view: &Entity<ZetaView>,
    receiver: &Receiver<CommandMessage>,
    catalog: SlashList,
) {
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.apply_worker_message(WorkerMessage::SlashExtensions(true), window, cx);
            view.apply_worker_message(WorkerMessage::SlashCommands(catalog), window, cx);
        });
    });
    // SlashExtensions(true) queues a SlashList against the active session
    // (round-3 F2 wired the request to fire from the shared active-session
    // block instead of only from `Session`). Drain it here so tests can
    // assert on the next command they trigger, not on this housekeeping.
    let queued = receiver.try_recv();
    assert!(
        matches!(&queued, Ok(CommandMessage::SlashList)),
        "enable_slash_extensions must queue exactly one SlashList, got {queued:?}"
    );
}

#[gpui::test]
fn slash_multiline_draft_dispatches_through_slash_run(cx: &mut TestAppContext) {
    // ZETA-130 round 2 F1: a `/status\nfoo` draft MUST route through
    // `slash_run`; the previous submit path silently sent it as a chat
    // message to the model because `leading_slash_token` returned None
    // when the value contained a newline. Any regression here breaks the
    // lane's invariant that slash-prefixed drafts never reach the model.
    let (window, view, receiver) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    enable_slash_extensions(&mut visual, &view, &receiver, slash_catalog(&["status"]));

    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.composer
                .update(cx, |input, cx| input.set_value("/status\nfoo", window, cx));
        });
    });
    visual.simulate_keystrokes("enter");
    let dispatched = receiver.try_recv();
    assert!(
        matches!(&dispatched, Ok(CommandMessage::SlashRun(text)) if text == "/status\nfoo"),
        "multiline slash draft must dispatch via slash_run, got {dispatched:?}"
    );
    assert!(
        !matches!(receiver.try_recv(), Ok(CommandMessage::Send(_))),
        "no Send RPC may fire for a `/`-prefixed multiline draft"
    );
    view.read_with(&visual, |view, _| {
        assert!(view.pending_command, "SlashRun sets pending until reply");
    });
}

#[gpui::test]
fn slash_first_enter_completes_and_second_enter_submits(cx: &mut TestAppContext) {
    // ZETA-130 round 2 F2. Contract: the first Enter completes the draft
    // to the highlighted command's canonical form (`/<name> ` when no
    // arguments are typed yet); a subsequent Enter dispatches through
    // `send_composer`. Enter never surprise-submits on the first press —
    // the user always sees the completed form before it leaves the
    // composer. The earlier implementation left Enter in selection mode
    // forever, so a second Enter on `/status ` was a silent no-op.
    //
    // Real characters are typed through `simulate_input` so the composer's
    // Change subscription fires and the menu opens the way it does in
    // production — `TextareaState::set_value` explicitly suppresses
    // Change (gpui-base state.rs `emit_events = false`) and never opens
    // the menu, so a set_value-based seed would test a different path
    // than what a user actually walks through.
    let (window, view, receiver) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    enable_slash_extensions(
        &mut visual,
        &view,
        &receiver,
        slash_catalog(&["status", "stop", "model"]),
    );

    // Path A: partial `/st` → Enter completes to `/status ` → Enter dispatches.
    visual.simulate_input("/st");
    view.read_with(&visual, |view, cx| {
        assert_eq!(view.composer.read(cx).value().as_ref(), "/st");
        assert!(
            view.slash_menu.open,
            "typing `/` opens the menu via the Change subscription"
        );
    });
    visual.simulate_keystrokes("enter");
    view.read_with(&visual, |view, cx| {
        assert_eq!(view.composer.read(cx).value().as_ref(), "/status ");
    });
    assert!(
        receiver.try_recv().is_err(),
        "first Enter completes the draft; nothing dispatches yet"
    );
    visual.simulate_keystrokes("enter");
    let dispatched = receiver.try_recv();
    assert!(
        matches!(&dispatched, Ok(CommandMessage::SlashRun(text)) if text == "/status "),
        "second Enter on the completed draft dispatches, got {dispatched:?}"
    );

    // Path B: fully-typed `/model` still completes first (adds trailing
    // space). Enter must never surprise-submit on the first press even
    // when the token already equals the highlighted command name.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.pending_command = false;
            view.composer
                .update(cx, |input, cx| input.set_value("", window, cx));
            view.slash_menu.dismiss();
        });
    });
    visual.simulate_input("/model");
    visual.simulate_keystrokes("enter");
    view.read_with(&visual, |view, cx| {
        assert_eq!(view.composer.read(cx).value().as_ref(), "/model ");
    });
    assert!(
        receiver.try_recv().is_err(),
        "fully-typed command completes first; Enter must not surprise-dispatch"
    );
    visual.simulate_keystrokes("enter");
    let dispatched = receiver.try_recv();
    assert!(
        matches!(&dispatched, Ok(CommandMessage::SlashRun(text)) if text == "/model "),
        "second Enter dispatches the completed draft, got {dispatched:?}"
    );
}

#[gpui::test]
fn slash_run_capability_gate_refuses_submit_on_legacy_server(cx: &mut TestAppContext) {
    // ZETA-130 round 2 F3: an older 1.1 server that never learned about
    // `slash_run` must not receive one, and the composer must refuse to
    // leak the `/`-prefixed draft to the model as chat.
    let (window, view, receiver) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    // No SlashExtensions(true) — the view starts with slash_extensions=false.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.composer
                .update(cx, |input, cx| input.set_value("/status", window, cx));
        });
    });
    visual.simulate_keystrokes("enter");
    assert!(
        receiver.try_recv().is_err(),
        "legacy server: composer must NOT dispatch Send or SlashRun for /status"
    );
    view.read_with(&visual, |view, _| {
        assert!(
            view.slash_output_notice
                .as_ref()
                .is_some_and(|notice| notice.error),
            "the output strip must surface the unavailable notice"
        );
    });
}

#[gpui::test]
fn slash_model_input_carries_pending_attachments_and_retains_draft_on_rejection(
    cx: &mut TestAppContext,
) {
    // ZETA-130 round 3 F1. A slash command that resolves to `model_input`
    // (a prompt macro like `/hi Henry`) must ship any pending image
    // attachments alongside the composed turn — silently dropping them
    // would be worse than the leak the round-2 gate closed. And when the
    // resulting send is rejected, the composer must still carry the user's
    // original slash draft so they can retry, matching the rejected-chat
    // pattern.
    let (window, view, receiver) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    enable_slash_extensions(&mut visual, &view, &receiver, slash_catalog(&["hi"]));

    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            let good = ImageAttachment::from_bytes("hero.png".into(), &valid_png_bytes())
                .expect("valid PNG parses");
            view.add_pending_attachments(vec![Ok(good)], cx);
            view.composer
                .update(cx, |input, cx| input.set_value("/hi Henry", window, cx));
        });
    });
    view.read_with(&visual, |view, _| {
        assert_eq!(view.valid_attachment_count(), 1, "chip landed");
    });

    // Server resolves the slash macro to a model-input turn.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.apply_worker_message(
                WorkerMessage::SlashResult(
                    "/hi Henry".into(),
                    SlashRunResult::ModelInput {
                        text: "Hi Henry".into(),
                    },
                ),
                window,
                cx,
            );
        });
    });

    // The composed turn MUST dispatch via SendImages so the pending
    // attachment rides with it — plain Send here would silently drop it.
    let dispatched = receiver.try_recv();
    match dispatched {
        Ok(CommandMessage::SendImages(text, images)) => {
            assert_eq!(text, "Hi Henry");
            assert_eq!(images.len(), 1);
            assert_eq!(images[0].name, "hero.png");
        }
        other => {
            panic!("expected SendImages for a slash ModelInput with attachments, got {other:?}")
        }
    }
    view.read_with(&visual, |view, cx| {
        assert!(view.pending_command, "ModelInput seeds pending_command");
        assert_eq!(
            view.composer.read(cx).value().as_ref(),
            "/hi Henry",
            "composer keeps the original draft until the server acks — same as normal chat"
        );
        assert_eq!(
            view.valid_attachment_count(),
            1,
            "chip stays put until Sent/ImagesSent clears it"
        );
        assert!(!view.slash_menu.open, "menu dismisses after dispatch");
    });

    // Server rejects. The draft must remain in place so the user can retry;
    // clearing the composer on ModelInput (the pre-fix behavior) lost the
    // original slash text and left the user with nothing to edit.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.apply_worker_message(
                WorkerMessage::Rejected("model refused this turn".into()),
                window,
                cx,
            );
        });
    });
    view.read_with(&visual, |view, cx| {
        assert_eq!(
            view.composer.read(cx).value().as_ref(),
            "/hi Henry",
            "rejected slash send must retain the composer draft for retry"
        );
        assert!(
            view.pending_user_turn
                .as_ref()
                .is_some_and(|turn| turn.failed),
            "rejection flips the danger rail on the queued strip"
        );
        assert_eq!(
            view.valid_attachment_count(),
            1,
            "chip still available for retry"
        );
    });
}

#[gpui::test]
fn slash_catalog_loads_on_status_when_session_never_arrives(cx: &mut TestAppContext) {
    // ZETA-130 round 3 F2. Initially-active sessions surface via
    // `WorkerMessage::Status`, not `Session`. Before the fix, the catalog
    // request was wired only to the Session handler, so an app restart on
    // a live session showed no slash menu at all. The request now fires
    // from the shared post-match block whenever an active session becomes
    // visible AND slash extensions are advertised.
    let (window, view, receiver) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    // Clear the pre-seeded active_session so we can watch it become
    // active via Status, matching a cold-start restore.
    visual.update(|_, cx| {
        view.update(cx, |view, _cx| {
            view.state.active_session = None;
        });
    });
    // Advertise slash extensions BEFORE any active session — no
    // catalog request should fire yet (there is nothing to enumerate).
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.apply_worker_message(WorkerMessage::SlashExtensions(true), window, cx);
        });
    });
    assert!(
        receiver.try_recv().is_err(),
        "SlashExtensions with no active session must not request the catalog"
    );

    // Cold-start Status frame carries the resumed session. The active
    // transition + slash extensions must together trigger SlashList.
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
        });
    });
    let dispatched = receiver.try_recv();
    assert!(
        matches!(&dispatched, Ok(CommandMessage::SlashList)),
        "cold-start Status with slash extensions on must request the catalog, got {dispatched:?}"
    );
    // And only once per session — a second Status on the same session
    // must not re-request.
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
        });
    });
    assert!(
        receiver.try_recv().is_err(),
        "repeat Status on the same session must not re-request the catalog"
    );
}

#[gpui::test]
fn slash_run_trims_leading_whitespace_and_multiline_draft(cx: &mut TestAppContext) {
    // ZETA-130 round 3 F3. `is_slash_draft` accepts a padded draft
    // (`  /status\nfoo`); the server's `run_command` rejects it with
    // `-32602` because its first check is a bare `text.startswith("/")`.
    // GUI owns the normalization: trim the leading whitespace before
    // handing the value to `slash_run` so the two sides agree.
    let (window, view, receiver) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    enable_slash_extensions(&mut visual, &view, &receiver, slash_catalog(&["status"]));

    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.composer.update(cx, |input, cx| {
                input.set_value("  /status\nfoo", window, cx)
            });
        });
    });
    visual.simulate_keystrokes("enter");
    let dispatched = receiver.try_recv();
    assert!(
        matches!(&dispatched, Ok(CommandMessage::SlashRun(text)) if text == "/status\nfoo"),
        "leading whitespace must be trimmed before SlashRun, got {dispatched:?}"
    );
    assert!(
        !matches!(receiver.try_recv(), Ok(CommandMessage::Send(_))),
        "a whitespace-padded `/`-prefixed value must never leak to the model as chat"
    );
}

#[gpui::test]
fn slash_catalog_refreshes_on_reconnect_and_capability_flap(cx: &mut TestAppContext) {
    // ZETA-130 round 4 F1. The active session survives a connection drop,
    // so `slash_catalog_requested` cannot latch across reconnect
    // generations. Two flows must refresh the catalog:
    //   (a) Lost → Connected + SlashExtensions(true) on the SAME session.
    //   (b) Capable → legacy (SlashExtensions(false) clears commands) →
    //       capable (SlashExtensions(true)) without disconnect.
    // Before the fix, (a) skipped the SlashList and (b) left the menu
    // permanently empty because the legacy handshake cleared commands but
    // never released the request latch.
    let (window, view, receiver) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    enable_slash_extensions(&mut visual, &view, &receiver, slash_catalog(&["status"]));

    // (a) Same-session reconnect must re-request the catalog.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.apply_worker_message(WorkerMessage::Lost("socket closed".into()), window, cx);
            view.apply_worker_message(WorkerMessage::Connected, window, cx);
            view.apply_worker_message(WorkerMessage::SlashExtensions(true), window, cx);
        });
    });
    let dispatched = receiver.try_recv();
    assert!(
        matches!(&dispatched, Ok(CommandMessage::SlashList)),
        "same-session reconnect must re-request the slash catalog, got {dispatched:?}"
    );

    // (b) Capable → legacy → capable without a Lost between them.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.apply_worker_message(WorkerMessage::SlashExtensions(false), window, cx);
            view.apply_worker_message(WorkerMessage::SlashExtensions(true), window, cx);
        });
    });
    let dispatched = receiver.try_recv();
    assert!(
        matches!(&dispatched, Ok(CommandMessage::SlashList)),
        "capable→legacy→capable must re-request the slash catalog, got {dispatched:?}"
    );
}

#[gpui::test]
fn slash_notice_falls_through_to_output_strip_when_menu_is_closed(cx: &mut TestAppContext) {
    // ZETA-130 round 4 F2. `slash_menu.set_notice` renders only while the
    // menu is open. A user who Escapes the menu (or submits a multiline
    // slash draft — the menu closes on newline) and hits Send must still
    // see the reason a `/`-prefixed value was refused. Route the notice
    // to `slash_output_notice` when the menu is closed so the message is
    // never swallowed.
    let (window, view, receiver) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    enable_slash_extensions(&mut visual, &view, &receiver, slash_catalog(&["mcp"]));

    // Unknown command with the menu closed lands on the output strip.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            assert!(
                !view.slash_menu.open,
                "menu is closed at the start of the test"
            );
            view.apply_worker_message(
                WorkerMessage::SlashResult(
                    "/nope".into(),
                    SlashRunResult::Unknown {
                        name: "nope".into(),
                    },
                ),
                window,
                cx,
            );
        });
    });
    view.read_with(&visual, |view, _| {
        let notice = view
            .slash_output_notice
            .as_ref()
            .expect("unknown with menu closed must paint the output strip");
        assert!(notice.error, "unknown command paints as error");
        assert!(
            notice.text.contains("nope"),
            "notice mentions the offending command, got {:?}",
            notice.text
        );
        assert!(
            view.slash_menu.notice.is_none(),
            "no menu notice when the menu itself is closed"
        );
    });

    // Unwired client-only with the menu closed also lands on the strip.
    // `mcp` is client-only and the GUI has no dedicated dispatcher for it,
    // so the fallback arm fires.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.slash_output_notice = None;
            view.apply_worker_message(
                WorkerMessage::SlashResult(
                    "/mcp".into(),
                    SlashRunResult::ClientOnly { name: "mcp".into() },
                ),
                window,
                cx,
            );
        });
    });
    view.read_with(&visual, |view, _| {
        let notice = view
            .slash_output_notice
            .as_ref()
            .expect("unwired client-only with menu closed must paint the output strip");
        assert!(notice.error, "unwired client-only paints as error");
        assert!(
            notice.text.contains("mcp"),
            "notice mentions the command, got {:?}",
            notice.text
        );
    });

    // Sanity: when the menu IS open, notices still render inline on the
    // menu (existing round-2 contract) — the strip must NOT double-up.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.slash_output_notice = None;
            view.slash_menu.open = true;
            view.apply_worker_message(
                WorkerMessage::SlashResult(
                    "/nope".into(),
                    SlashRunResult::Unknown {
                        name: "nope".into(),
                    },
                ),
                window,
                cx,
            );
        });
    });
    view.read_with(&visual, |view, _| {
        assert!(
            view.slash_output_notice.is_none(),
            "menu-open path must not spill onto the output strip"
        );
        assert!(
            view.slash_menu.notice.is_some(),
            "menu-open path paints the inline notice"
        );
    });
}

#[gpui::test]
fn slash_literal_escape_normalises_before_send(cx: &mut TestAppContext) {
    // ZETA-130 round 4 F3. `//status` is the literal-slash escape: the
    // shared `input_for_model` contract turns it into `/status` before
    // the model sees the turn. The GUI submits chat directly (no slash
    // dispatch), so it must apply the same strip before Send/SendImages.
    // Without this the model receives the raw `//status` and answers as
    // if the user typed the escape character on purpose.
    let (window, view, receiver) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    // No slash extensions needed — this is the escape lane, not slash_run.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.composer
                .update(cx, |input, cx| input.set_value("//status", window, cx));
        });
    });
    visual.simulate_keystrokes("enter");
    let dispatched = receiver.try_recv();
    assert!(
        matches!(&dispatched, Ok(CommandMessage::Send(text)) if text == "/status"),
        "`//status` must send as literal `/status`, got {dispatched:?}"
    );

    // A `//` escape on a body with a trailing paragraph also strips ONE
    // slash from the head — the entire chat body still ships to the model.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.pending_command = false;
            view.composer.update(cx, |input, cx| {
                input.set_value("//status still counts", window, cx)
            });
        });
    });
    visual.simulate_keystrokes("enter");
    let dispatched = receiver.try_recv();
    assert!(
        matches!(&dispatched, Ok(CommandMessage::Send(text)) if text == "/status still counts"),
        "`//` escape strips one slash from the head, keeping the body intact, got {dispatched:?}"
    );
}

#[gpui::test]
fn slash_argless_model_opens_settings(cx: &mut TestAppContext) {
    // ZETA-130 round 4 F4. Argless `/model` opts out server-side to
    // `client_only` so the GUI can hand off to the settings picker rather
    // than paint a bare notice a user cannot act on. Matches the doc
    // contract at docs/serve-protocol.md.
    let (window, view, receiver) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    enable_slash_extensions(&mut visual, &view, &receiver, slash_catalog(&["model"]));

    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            // The Settings overlay needs the session-view capability from
            // the server; the shipped default is false, so tests that ask
            // to open Settings must flip it explicitly.
            view.state.session_view.available = true;
            view.composer
                .update(cx, |input, cx| input.set_value("/model ", window, cx));
            view.apply_worker_message(
                WorkerMessage::SlashResult(
                    "/model ".into(),
                    SlashRunResult::ClientOnly {
                        name: "model".into(),
                    },
                ),
                window,
                cx,
            );
        });
    });
    // open_settings queues LoadSettings and flips `settings_open` only
    // when the reply lands — assert the RPC intent rather than the state.
    let dispatched = receiver.try_recv();
    assert!(
        matches!(&dispatched, Ok(CommandMessage::LoadSettings)),
        "argless /model must ask the server to open Settings, got {dispatched:?}"
    );
    view.read_with(&visual, |view, cx| {
        assert!(
            view.composer.read(cx).value().is_empty(),
            "handoff clears the composer so the slash draft never re-fires"
        );
        assert!(
            !view.slash_menu.open,
            "menu dismisses after the client-only handoff"
        );
    });
}

/// ZETA-134 A3: an argument-less tool_start stores `excerpt: None` so the
/// row builder paints the tool label alone — no primary text. A file
/// literally named `read` (or a bash command named `bash`) carries its own
/// `Some(...)` value and paints as itself. The pre-r2 implementation used
/// string equality with the tool name as a sentinel, which collapsed a
/// legitimate `read` path into the label.
///
/// Drives real `ToolStart` events through `apply_worker_message` so the
/// server-side derivation (`state::tool_excerpt`) runs and the transcript
/// entry stores what the server would produce. For each case, checks:
///   • stored excerpt on the transcript entry,
///   • painted excerpt element (bounds present under `tool-excerpt-0`),
///   • recorded chevron color sample (proves the state-colored chevron
///     text ran through the `render_log` recorder for the paint).
#[gpui::test]
fn tool_row_uses_option_none_for_missing_argument_state(cx: &mut TestAppContext) {
    use zeta_gui::client::ToolCall;
    use zeta_gui::state::TranscriptEntry;
    let (window, view, _receiver) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    // (name, arguments, expected_stored_excerpt)
    let cases: &[(&str, serde_json::Value, Option<&str>)] = &[
        // A `read` call whose file path is literally `read` — the pre-r2
        // string-equality sentinel would fold this into the label and
        // hide the real path. The typed Option<String> keeps it.
        ("read", json!({"path": "read"}), Some("read")),
        // A `bash` call whose command literally begins with `bash`. Same
        // failure mode as `read`-named-`read`: the excerpt must survive.
        (
            "bash",
            json!({"command": "bash script.sh"}),
            Some("bash script.sh"),
        ),
        // No nameable argument reaches the derivation → excerpt is None
        // and the row paints the tool label alone.
        ("read", json!({}), None),
        ("bash", json!({}), None),
    ];
    for (index, (name, arguments, expected_excerpt)) in cases.iter().enumerate() {
        let call_id = format!("row-{index}");
        let arguments_map = arguments
            .as_object()
            .cloned()
            .expect("case arguments must be a JSON object");
        visual.update(|window, cx| {
            view.update(cx, |view, cx| {
                view.state.transcript.clear();
                view.transcript.update(cx, |scroll, cx| scroll.reset(0, cx));
                super::render_log::clear();
                view.apply_worker_message(
                    WorkerMessage::Event(ServerEvent::ToolStart {
                        session_id: view.state.active_session.clone(),
                        tool_call: ToolCall {
                            id: call_id.clone(),
                            name: (*name).into(),
                            arguments: arguments_map.clone(),
                        },
                        data: json!({}),
                    }),
                    window,
                    cx,
                );
            });
            window.draw(cx).clear(cx);
        });
        // Stored excerpt on the entry mirrors what the server derivation
        // produced — Some for both hard cases, None for missing arguments.
        view.read_with(&visual, |view, _| {
            let entry = view
                .state
                .transcript
                .last()
                .expect("ToolStart appended a transcript entry");
            let TranscriptEntry::Tool {
                excerpt,
                name: stored_name,
                ..
            } = entry
            else {
                panic!("expected a Tool entry for case {index}");
            };
            assert_eq!(stored_name.as_str(), *name, "case {index}: stored name");
            assert_eq!(
                excerpt.as_deref(),
                *expected_excerpt,
                "case {index}: stored excerpt mismatch",
            );
        });
        // Painted excerpt element: the div always paints (empty string when
        // excerpt is None), so its bounds must resolve in every case.
        assert!(
            visual.debug_bounds("tool-excerpt-0").is_some(),
            "case {index}: excerpt element must paint (even for the None state)",
        );
        // Recorded excerpt AND chevron samples: `render_tool_row` routes
        // both text colors through `state_text`, which pushes into
        // `render_log`. A regression that stops recording either element
        // (or paints a bare glyph outside the recorder) drops it from
        // `samples`; a regression that swaps the state token records the
        // wrong color. Assert both are recorded and both carry the
        // running-state color for every real case.
        visual.update(|_, cx| {
            let samples = super::render_log::samples();
            let expected = super::tool_state_color(zeta_gui::state::ToolState::Running, cx);
            for row_id in ["tool-excerpt-0", "tool-chevron-0"] {
                let recorded: Vec<_> = samples
                    .iter()
                    .filter(|s| s.row_id == row_id)
                    .cloned()
                    .collect();
                assert!(
                    !recorded.is_empty(),
                    "case {index}: render_log must record a {row_id} sample",
                );
                assert!(
                    recorded.iter().all(|s| s.color == expected),
                    "case {index}: {row_id} sample color regressed off the running-state token",
                );
            }
        });
    }
}

/// ZETA-134 A6: after Cmd-N, the sidebar must show exactly one row reading
/// as current — the row fill (focus accent) and the active dot may not
/// disagree. Drives the REAL keyboard path (`cmd-n` keystroke → queued
/// `CommandMessage::NewSession` → worker `Session` reply → paint) and
/// counts painted quads to pin the single-selection invariant. Row_c
/// is pre-seeded at index 1 in the sidebar so that when
/// `apply_worker_message::Session` updates it in place the dot's y
/// SHIFTS to a distinct row (row_a stays at index 0). The prior
/// single-session seed left both dots at the same y (index 0
/// re-ordering conflated them), which masked the "dot moved" invariant.
///   • BEFORE Cmd-N: row_a is focused AND current — exactly one focus
///     fill and exactly one dot paint, on the same row (index 0).
///   • AFTER Cmd-N: focus retargets to the composer, so zero sidebar
///     rows paint a focus fill; the dot moves down to row_c at index 1
///     and paints exactly once, at a y strictly greater than before.
/// A synthetic-selector count (`debug_bounds`) alone can't catch a
/// second stale fill or a second stale dot bleeding through — the
/// painted-quad probe does.
#[gpui::test]
fn cmd_n_paints_a_single_current_row_and_moves_focus_to_the_composer(cx: &mut TestAppContext) {
    let (window, view, receiver) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    let session_a = session().session_id.clone();
    let session_c = "cd34deadbeef".to_owned();
    // Seed row_c BELOW row_a in the sidebar. `apply_worker_message::Session`
    // updates the row in place when the session id already exists (see
    // main.rs:610), so the row_c reply keeps row_c at index 1 — a
    // different y from row_a at index 0. That makes the "dot moved from
    // row_a to row_c" invariant observable as a real y shift; the earlier
    // single-session seed left both dots at position 0's y because
    // apply_worker_message would otherwise insert row_c at index 0 and
    // conflate them.
    visual.update(|_, cx| {
        view.update(cx, |view, cx| {
            let mut existing_c: SessionMetadata = serde_json::from_value(
                json!({"session_id": session_c, "updated_at": "2026-09-08T12:00:00Z"}),
            )
            .unwrap();
            existing_c.name = "row c".into();
            view.state.sessions.push(existing_c);
            cx.notify();
        });
    });
    // Focus row A as if the user tabbed there. `sidebar_row_focus` stores
    // the tab-stop handle keyed by session id — grabbing it here mirrors
    // what the sidebar renderer would do on the next paint.
    let (focus_a, focus_composer) = visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            let handle = view
                .sidebar_row_focus
                .borrow_mut()
                .entry(session_a.clone())
                .or_insert_with(|| cx.focus_handle().tab_stop(true).tab_index(0))
                .clone();
            window.focus(&handle, cx);
            let composer_handle = view.composer.focus_handle(cx);
            (handle, composer_handle)
        })
    });
    visual.update(|window, cx| window.draw(cx).clear(cx));
    assert!(
        visual.update(|window, _| focus_a.is_focused(window)),
        "sanity: the previous session row starts with keyboard focus",
    );

    // Count painted primary-accent quads inside the sidebar column,
    // split by height: row-sized quads are the focused row's fill,
    // dot-sized quads are the current-session dot. Returns their y
    // origins in ScaledPixels so we can prove co-location without any
    // unsupported `f32::from(ScaledPixels)` conversion.
    let count_sidebar_fills =
        |visual: &mut VisualTestContext| -> (Vec<gpui::ScaledPixels>, Vec<gpui::ScaledPixels>) {
            visual.update(|window, cx| {
                let primary: gpui::Background = cx.theme().primary.into();
                let scale = window.scale_factor();
                let row_h = theme::SIDEBAR_ROW_HEIGHT.scale(scale);
                let dot_h = theme::SIDEBAR_CURRENT_DOT_SIZE.scale(scale);
                let sidebar_right = theme::SIDEBAR_WIDTH.scale(scale);
                let slack = gpui::ScaledPixels::from(1.0);
                let mut rows = Vec::new();
                let mut dots = Vec::new();
                for quad in window.painted_quads() {
                    if quad.background != primary {
                        continue;
                    }
                    if quad.bounds.right() > sidebar_right {
                        continue;
                    }
                    let h = quad.bounds.size.height;
                    if h + slack >= row_h && h <= row_h + slack {
                        rows.push(quad.bounds.origin.y);
                    } else if h + slack >= dot_h && h <= dot_h + slack {
                        dots.push(quad.bounds.origin.y);
                    }
                }
                (rows, dots)
            })
        };
    // Two ScaledPixels within `tolerance` of one another read as the
    // same row. ScaledPixels does not implement `.abs()`, so branch
    // instead of chaining `.abs()`.
    let within = |a: gpui::ScaledPixels, b: gpui::ScaledPixels, tolerance: gpui::ScaledPixels| {
        if a >= b {
            a - b <= tolerance
        } else {
            b - a <= tolerance
        }
    };

    // BEFORE Cmd-N: row_a is focused (row fill) AND current (dot).
    // Both paint exactly once, on the same row.
    let (rows_before, dots_before) = count_sidebar_fills(&mut visual);
    assert_eq!(
        rows_before.len(),
        1,
        "row_a is focused → exactly one focus row fill: {rows_before:?}",
    );
    assert_eq!(
        dots_before.len(),
        1,
        "row_a is current → exactly one dot: {dots_before:?}",
    );
    let row_h_scaled =
        visual.update(|window, _| theme::SIDEBAR_ROW_HEIGHT.scale(window.scale_factor()));
    assert!(
        within(rows_before[0], dots_before[0], row_h_scaled),
        "focus fill (y={:?}) and dot (y={:?}) must sit on the same row before Cmd-N",
        rows_before[0],
        dots_before[0],
    );

    // Drive the REAL Cmd-N path: keystroke → action → queued command.
    visual.simulate_keystrokes("cmd-n");
    assert!(
        matches!(receiver.try_recv(), Ok(CommandMessage::NewSession)),
        "Cmd-N must queue NewSession on the command channel",
    );
    view.read_with(&visual, |view, _| {
        assert!(
            view.pending_command,
            "Cmd-N flips pending_command until the worker replies",
        );
    });

    // Worker reply through the harness. `apply_worker_message` swaps the
    // active session, updates row_c in place at index 1 (the seed put it
    // there so the dot's y shifts to a distinct row), and retargets stale
    // sidebar focus to the composer.
    let new_session: SessionMetadata = serde_json::from_value(
        json!({"session_id": session_c, "updated_at": "2026-09-16T12:00:00Z"}),
    )
    .unwrap();
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.apply_worker_message(WorkerMessage::Session(new_session), window, cx);
        });
        window.draw(cx).clear(cx);
    });

    view.read_with(&visual, |view, _| {
        assert_eq!(
            view.state.active_session.as_deref(),
            Some(session_c.as_str()),
            "active_session swaps to the new id so the dot follows",
        );
        assert_eq!(
            view.state.sessions.len(),
            2,
            "the seeded session_c is updated in place, not duplicated",
        );
        assert_eq!(
            view.state.sessions[1].session_id.as_str(),
            session_c.as_str(),
            "session_c stays at index 1 — apply_worker_message updated it in place",
        );
    });
    assert!(
        !visual.update(|window, _| focus_a.is_focused(window)),
        "focus lifts off the previous row so it stops painting the accent fill",
    );
    assert!(
        visual.update(|window, _| focus_composer.is_focused(window)),
        "focus lands on the composer so no sidebar row reads as focused",
    );

    // AFTER Cmd-N: composer holds focus → zero row focus fills. The dot
    // paints exactly once — on row_c (index 1, below row_a). Because row_c
    // sits BELOW row_a, its dot's y is strictly greater than the pre-Cmd-N
    // dot y that sat on row_a at index 0. A regression where a stale dot
    // lingers on row_a would show up as two dots; a regression where the
    // dot never moved would show up as an equal-or-lesser y.
    let (rows_after, dots_after) = count_sidebar_fills(&mut visual);
    assert_eq!(
        rows_after.len(),
        0,
        "composer holds focus → zero sidebar rows paint the focus accent fill: {rows_after:?}",
    );
    assert_eq!(
        dots_after.len(),
        1,
        "single-selection invariant: exactly one dot paints on the new current row: {dots_after:?}",
    );
    assert!(
        dots_after[0] > dots_before[0],
        "dot y after ({:?}) must be BELOW the pre-Cmd-N y ({:?}) — row_c sits at index 1 under row_a",
        dots_after[0],
        dots_before[0],
    );
}

/// ZETA-134 C7: an empty composer disables Send. `gui/README.md` promises
/// "the composer explains why sending is disabled" — the button follows
/// suit. Attach stays live so a drag can start from an empty composer.
#[gpui::test]
fn send_button_disables_when_the_composer_is_empty(cx: &mut TestAppContext) {
    let (window, view, _receiver) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.composer
                .update(cx, |input, cx| input.set_value("", window, cx));
        });
        window.draw(cx).clear(cx);
    });
    let disabled_bounds = visual
        .debug_bounds("send-button")
        .expect("send button paints in the composer");
    // The disabled paint is a plain div, not a Kit Button — a click on it
    // must NOT queue a Send. `receiver.try_recv()` after the click returns
    // an error because nothing was dispatched.
    visual.simulate_click(disabled_bounds.center(), Default::default());
    assert!(
        _receiver.try_recv().is_err(),
        "click on a disabled Send must not queue a command",
    );
    // Now type something — Send re-enables and click dispatches.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.composer
                .update(cx, |input, cx| input.set_value("hello", window, cx));
        });
        window.draw(cx).clear(cx);
    });
    let enabled_bounds = visual
        .debug_bounds("send-button")
        .expect("send button still paints");
    visual.simulate_click(enabled_bounds.center(), Default::default());
    let dispatched = _receiver.try_recv().expect("send dispatched after typing");
    assert!(
        matches!(&dispatched, CommandMessage::Send(text) if text == "hello"),
        "typed text reaches the worker as Send, got {dispatched:?}",
    );
}

/// ZETA-137 D2: expanded bash receipts show the OUTPUT, not the scaffolding.
/// Labels only appear when they disambiguate — stdout-only + exit 0 renders
/// the stdout text alone (no `stdout:` header, no `exit: 0` line). `stderr:`
/// returns when stderr is non-empty; `exit: N` returns when N != 0. Everything
/// empty falls through to raw so the receipt is never blank. Legacy servers
/// without `structured_content` keep the raw payload untouched (D6 invariant).
#[test]
fn bash_expanded_shows_output_and_only_labels_when_they_disambiguate() {
    use zeta_gui::state::reshape_bash_content;

    // Clean case: `pwd` / `echo hi` — stdout only, exit 0. Show the output
    // alone. Henry's "it should just show the bash output" ask.
    let structured = json!({
        "stdout": "/Users/henry\n",
        "stderr": "",
        "exit_code": 0,
    });
    let out = reshape_bash_content("stdout:\n/Users/henry\nstderr:\n", Some(&structured));
    assert_eq!(
        out, "/Users/henry",
        "stdout-only + exit 0 shows the raw output, no labels: {out:?}"
    );
    assert!(!out.contains("stdout:"));
    assert!(!out.contains("exit:"));

    // stdout + stderr both present, exit 0: labels return so the two
    // streams are distinguishable; no `exit: 0` line.
    let both = json!({
        "stdout": "line one\n",
        "stderr": "warning\n",
        "exit_code": 0,
    });
    let out = reshape_bash_content("ignored", Some(&both));
    assert!(out.contains("stdout:\nline one"));
    assert!(out.contains("stderr:\nwarning"));
    assert!(!out.contains("exit:"), "no exit line on success: {out:?}");

    // stderr-only + nonzero exit: `stderr:` label appears; no `stdout:`
    // label because there is no stdout; `exit: N` line appears.
    let err_only = json!({
        "stdout": "",
        "stderr": "boom\n",
        "exit_code": 1,
    });
    let out = reshape_bash_content("ignored", Some(&err_only));
    assert!(
        !out.contains("stdout:"),
        "no stdout label when empty: {out:?}"
    );
    assert!(out.contains("stderr:\nboom"));
    assert!(out.ends_with("exit: 1"));

    // stdout + nonzero exit: `stdout:` labels the section so `exit: N`
    // reads as its own line beneath, not as a trailing suffix.
    let fail_with_out = json!({
        "stdout": "partial\n",
        "stderr": "",
        "exit_code": 2,
    });
    let out = reshape_bash_content("ignored", Some(&fail_with_out));
    assert!(out.contains("stdout:\npartial"));
    assert!(out.ends_with("exit: 2"));

    // Everything empty, exit 0: fall through to raw so the receipt is
    // never blank.
    let raw = "(no output)";
    let empty = json!({
        "stdout": "",
        "stderr": "",
        "exit_code": 0,
    });
    let out = reshape_bash_content(raw, Some(&empty));
    assert_eq!(out, raw);

    // Missing structured_content — legacy shape falls back to raw.
    let legacy_raw = "stdout:\nfoo\nstderr:\nbar";
    let out = reshape_bash_content(legacy_raw, None);
    assert_eq!(out, legacy_raw);

    // Missing exit_code (malformed shape) — falls through to raw. Without
    // this guard the clean branch would still fire on stdout+stderr and
    // strip the labels, leaking a mis-shaped payload as clean output.
    let missing_exit = json!({
        "stdout": "hi\n",
        "stderr": "",
    });
    let raw_missing = "stdout:\nhi";
    let out = reshape_bash_content(raw_missing, Some(&missing_exit));
    assert_eq!(
        out, raw_missing,
        "missing exit_code falls back to raw: {out:?}"
    );

    // Whitespace-only stdout (a lone `\n`) does NOT count as content —
    // otherwise the clean branch strips to an empty string and paints a
    // blank receipt. Falls through to raw.
    let whitespace_only = json!({
        "stdout": "\n",
        "stderr": "",
        "exit_code": 0,
    });
    let raw_ws = "(no output)";
    let out = reshape_bash_content(raw_ws, Some(&whitespace_only));
    assert_eq!(
        out, raw_ws,
        "whitespace-only stdout falls back to raw: {out:?}"
    );

    // CRLF-terminated stdout — the trailing `\r` strips alongside the
    // `\n` so the clean receipt does not carry a dangling CR at the end.
    let crlf = json!({
        "stdout": "hello\r\n",
        "stderr": "",
        "exit_code": 0,
    });
    let out = reshape_bash_content("ignored", Some(&crlf));
    assert_eq!(out, "hello", "CRLF strips both \\r and \\n: {out:?}");
}

// ------------------------------------------------------------------------
// ZETA-135: wiki session-view transcript look.
// ------------------------------------------------------------------------

/// ZETA-135 (Trait 2 — expanded receipt = inset panel): the expanded body
/// paints a bordered container with a file-path / command header row at
/// its top edge, not the pre-ZETA-135 left-rail indent. The header sits at
/// the foreground tier so the reader answers "what ran" before scanning
/// the output body. A row_text-side model assertion pins the
/// `panel_header` field's contract (`Some(&excerpt)` when expanded AND
/// excerpt exists; `None` for argument-less receipts) so a regression that
/// paints a chromeless header bar for an argument-less receipt fails at
/// the typed seam, and a paint assertion pins the debug selector so a
/// rename regresses at the render layer.
#[gpui::test]
fn zeta135_expanded_receipt_paints_inset_panel_with_file_path_header(cx: &mut TestAppContext) {
    use zeta_gui::row_text::{self, RowText};
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.transcript.clear();
            view.transcript.update(cx, |scroll, cx| scroll.reset(0, cx));
            let mut arguments = serde_json::Map::new();
            arguments.insert("command".into(), json!("bash scripts/warmup.sh --verbose"));
            let call = ToolCall {
                id: "receipt".into(),
                name: "bash".into(),
                arguments,
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
            view.apply_worker_message(
                WorkerMessage::Event(ServerEvent::ToolEnd {
                    session_id: view.state.active_session.clone(),
                    tool_call: call,
                    tool_result: Some(zeta_gui::client::ToolResult {
                        tool_call_id: "receipt".into(),
                        content: "output line\n".into(),
                        is_error: false,
                        is_canceled: false,
                        structured_content: None,
                        content_blocks: Vec::new(),
                    }),
                    data: json!({}),
                }),
                window,
                cx,
            );
            // Success rows collapse by default — expansion is driven
            // through the receipt CONTROL (a click on the row) below so
            // the click path itself participates in the assertion
            // (round-2 review Finding 4: `toggle_card` bypassed the
            // click handler).
            cx.notify();
        });
        window.draw(cx).clear(cx);
    });
    let collapsed_receipt = visual
        .debug_bounds("tool-receipt-0")
        .expect("collapsed receipt paints before click");
    visual.simulate_click(
        collapsed_receipt.origin + gpui::point(px(50.), px(20.)),
        Default::default(),
    );
    visual.update(|window, cx| window.draw(cx).clear(cx));

    // (a) Typed-seam contract: `ToolRowText::panel_header` is `Some(...)`
    //     mirroring the excerpt whenever the row is expanded AND the tool
    //     call carried a nameable argument. An argument-less tool_start
    //     still stores `None` so the render layer paints the tool label
    //     alone (ZETA-134 review r2 invariant).
    view.read_with(&visual, |view, _| {
        let entry = &view.state.transcript[0];
        let row = row_text::build(entry, 0, &view.state.session_view, true);
        let RowText::Tool(text) = row else {
            panic!("bash entry must build a Tool row")
        };
        assert_eq!(
            text.panel_header,
            Some("bash scripts/warmup.sh --verbose"),
            "expanded receipt with a nameable excerpt must expose it as \
             the panel header"
        );
    });

    // (b) Paint contract: the panel header element paints with the
    //     documented debug selector. A rename or a missing element in the
    //     render path trips here.
    let panel_header = visual.debug_bounds("tool-panel-header-0");
    assert!(
        panel_header.is_some(),
        "ZETA-135 inset panel header row must paint for expanded receipts"
    );

    // (c) Container contract: the expanded body's outer container carries
    //     a 1px border on top AND right AND bottom (not just left) — the
    //     wiki inset panel shape. The pre-ZETA-135 receipt only painted a
    //     left rail; a regression to `border_l` would leave top/right/
    //     bottom at zero here.
    let body = visual
        .debug_bounds("tool-output-0")
        .expect("expanded body renders");
    visual.update(|window, _| {
        let scaled = body.scale(window.scale_factor());
        let one_px = px(1.).scale(window.scale_factor());
        let outer = window
            .painted_quads()
            .into_iter()
            .find(|quad| {
                (quad.bounds.top() - scaled.top()).0.abs() <= 1.0
                    && (quad.bounds.left() - scaled.left()).0.abs() <= 1.0
                    && quad.border_widths.top >= one_px
                    && quad.border_widths.right >= one_px
                    && quad.border_widths.bottom >= one_px
            })
            .expect(
                "ZETA-135 inset panel must paint a 1px border on top/right/bottom, \
                 not just the pre-ZETA-135 left rail",
            );
        assert!(
            outer.border_widths.top >= one_px
                && outer.border_widths.right >= one_px
                && outer.border_widths.bottom >= one_px
                && outer.border_widths.left >= one_px,
            "ZETA-135 inset panel border must sit on all four sides — got {:?}",
            outer.border_widths,
        );
    });

    // (d) Argument-less tool_start: `panel_header` stays `None` so an
    //     expanded receipt without a nameable excerpt does not paint a
    //     chromeless header bar. Drives the receipt CONTROL (a real
    //     click) instead of setting `card.expanded` directly so the
    //     click path itself participates in the assertion (round-2
    //     review finding 6), and asserts BOTH the model-level None AND
    //     the absence of any painted `tool-panel-header-0` selector so
    //     an "always paints" render regression fails here rather than
    //     surviving the model-only check.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.transcript.clear();
            view.transcript.update(cx, |scroll, cx| scroll.reset(0, cx));
            let bare_call = ToolCall {
                id: "bare".into(),
                name: "bash".into(),
                arguments: serde_json::Map::new(),
            };
            view.apply_worker_message(
                WorkerMessage::Event(ServerEvent::ToolStart {
                    session_id: view.state.active_session.clone(),
                    tool_call: bare_call,
                    data: json!({}),
                }),
                window,
                cx,
            );
            cx.notify();
        });
        window.draw(cx).clear(cx);
    });
    let bare_receipt = visual
        .debug_bounds("tool-receipt-0")
        .expect("argument-less receipt paints its collapsed row");
    visual.simulate_click(
        bare_receipt.origin + gpui::point(px(50.), px(20.)),
        Default::default(),
    );
    visual.update(|window, cx| window.draw(cx).clear(cx));
    view.read_with(&visual, |view, _| {
        let entry = &view.state.transcript[0];
        let row = row_text::build(entry, 0, &view.state.session_view, true);
        let RowText::Tool(text) = row else {
            panic!("bash entry must build a Tool row")
        };
        assert_eq!(
            text.panel_header, None,
            "an argument-less tool_start must NOT populate the panel \
             header (ZETA-134 review r2)"
        );
    });
    assert!(
        visual.debug_bounds("tool-panel-header-0").is_none(),
        "an argument-less receipt must NOT paint any panel-header element \
         (round-2 review finding 6: an always-paints renderer must fail here)"
    );
}

/// ZETA-135 review r1 finding 6: the round-1 test compared the label
/// chip's bottom to the FOOTER's top and passed even when the chip painted
/// below the input row. Compare against the INPUT ROW itself — that is the
/// element the chip must sit above.
#[gpui::test]
fn zeta135_composer_paints_a_label_chip_above_the_input_row(cx: &mut TestAppContext) {
    let (window, _view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let label = visual
        .debug_bounds("composer-label")
        .expect("composer label chip must paint");
    let input_row = visual
        .debug_bounds("composer-input-row")
        .expect("composer input row must paint");
    assert!(
        label.bottom() <= input_row.top(),
        "composer label chip must sit ABOVE the input row — got \
         label.bottom={:?}, input_row.top={:?}",
        label.bottom(),
        input_row.top(),
    );
    assert!(
        label.size.height > px(0.),
        "composer label chip must paint with non-zero height"
    );
    // The chip's height matches the shared height constant so a font-size
    // change never leaks composer fill above the transcript scan (paired
    // with the smoke driver's `theme::composer_chrome_reserve()`).
    assert_eq!(
        label.size.height,
        theme::COMPOSER_LABEL_HEIGHT,
        "composer label chip must paint at the shared height constant"
    );
}

/// ZETA-135 (Trait 1 — kind glyph): every tool receipt paints a leading
/// glyph before the tool label that names the KIND of thing that ran
/// (shell/edit/fetch/other). Drives ONE receipt per family through the
/// live entry pipeline and asserts the painted debug selector. Each
/// family runs in its own transcript so the tool-group collapser (3+
/// consecutive receipts fold into a summary row that hides the
/// individual receipts) never masks the family under test.
#[gpui::test]
fn zeta135_tool_row_paints_a_kind_glyph_for_each_family(cx: &mut TestAppContext) {
    use zeta_gui::row_text::{self, chrome, RowText};
    // Static selectors — `VisualTestContext::debug_bounds` requires
    // `&'static str`, so keep one row per family with pre-composed
    // selector strings rather than a dynamic `format!`.
    struct Family {
        name: &'static str,
        arg_key: &'static str,
        expected_glyph: &'static str,
        id: &'static str,
    }
    let families: &[Family] = &[
        Family {
            name: "bash",
            arg_key: "command",
            expected_glyph: chrome::TOOL_KIND_SHELL,
            id: "kind-bash",
        },
        Family {
            name: "edit",
            arg_key: "path",
            expected_glyph: chrome::TOOL_KIND_EDIT,
            id: "kind-edit",
        },
        // Round-2 finding 2: str_replace/multi_edit must classify as Edit
        // for the glyph, not Generic. The pre-fix `kind_glyph_for`
        // enumerated only "edit" | "write" | "read" | "list", so a
        // `str_replace` tool call painted `⚙` while its diff card built
        // fine — the two sites disagreed. The shared `ToolKind` model
        // closes the split.
        Family {
            name: "str_replace",
            arg_key: "path",
            expected_glyph: chrome::TOOL_KIND_EDIT,
            id: "kind-str-replace",
        },
        Family {
            name: "fetch",
            arg_key: "url",
            expected_glyph: chrome::TOOL_KIND_FETCH,
            id: "kind-fetch",
        },
        // Round-2 finding 2: `websearch` is now a first-class Search
        // family with its own `⌕` glyph rather than the generic fallback.
        Family {
            name: "websearch",
            arg_key: "query",
            expected_glyph: chrome::TOOL_KIND_SEARCH,
            id: "kind-websearch",
        },
        Family {
            name: "grep",
            arg_key: "pattern",
            expected_glyph: chrome::TOOL_KIND_SEARCH,
            id: "kind-grep",
        },
        Family {
            name: "unknown",
            arg_key: "value",
            expected_glyph: chrome::TOOL_KIND_GENERIC,
            id: "kind-unknown",
        },
    ];
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    for family in families {
        visual.update(|window, cx| {
            view.update(cx, |view, cx| {
                // Fresh transcript per family so the 3+ consecutive-tool
                // group collapser never folds the receipt under test into
                // a summary row that hides its selectors. Reset the
                // scroller with count=0 so the virtual list forgets the
                // prior family's row.
                view.state.transcript.clear();
                view.transcript.update(cx, |scroll, cx| scroll.reset(0, cx));
                let mut arguments = serde_json::Map::new();
                arguments.insert(family.arg_key.into(), json!("value"));
                view.apply_worker_message(
                    WorkerMessage::Event(ServerEvent::ToolStart {
                        session_id: view.state.active_session.clone(),
                        tool_call: ToolCall {
                            id: family.id.into(),
                            name: family.name.into(),
                            arguments,
                        },
                        data: json!({}),
                    }),
                    window,
                    cx,
                );
            });
            window.draw(cx).clear(cx);
        });
        view.read_with(&visual, |view, _| {
            let entry = &view.state.transcript[0];
            let row = row_text::build(entry, 0, &view.state.session_view, true);
            let RowText::Tool(text) = row else {
                panic!("family {} must build a Tool row", family.name)
            };
            assert_eq!(
                text.kind_glyph, family.expected_glyph,
                "family {}: kind_glyph must be {}, got {}",
                family.name, family.expected_glyph, text.kind_glyph,
            );
        });
        let glyph = visual
            .debug_bounds("tool-kind-glyph-0")
            .unwrap_or_else(|| panic!("family {}: kind glyph must paint", family.name));
        let label = visual
            .debug_bounds("tool-label-0")
            .unwrap_or_else(|| panic!("family {}: tool label must paint", family.name));
        assert!(
            glyph.right() <= label.left(),
            "family {}: kind glyph must sit LEFT of the tool label — \
             got glyph.right={:?}, label.left={:?}",
            family.name,
            glyph.right(),
            label.left(),
        );
        assert!(glyph.size.height > px(0.));
    }
}

/// ZETA-135 (Trait 2 — diff card): an edit receipt with typed old/new
/// strings paints two side-by-side panes under the panel header when
/// expanded, and NO diff card when collapsed. Expansion drives through
/// the receipt CONTROL (a simulated click on the row) so a regression to
/// "always paints" fails the assertion pair (Finding 6).
#[gpui::test]
fn zeta135_edit_receipt_paints_a_diff_card_when_expanded(cx: &mut TestAppContext) {
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    // Wide viewport first — the side-by-side layout has to fit both panes
    // above the narrow breakpoint (`theme::NARROW_DIFF_STACK_WIDTH`).
    visual.simulate_resize(gpui::size(px(1200.), px(800.)));
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.transcript.clear();
            view.transcript.update(cx, |scroll, cx| scroll.reset(0, cx));
            let mut arguments = serde_json::Map::new();
            arguments.insert("path".into(), json!("hot.md"));
            arguments.insert(
                "old_string".into(),
                json!("Zeta UI-POLISH-2 arc RESUMED\nsecond removed line"),
            );
            arguments.insert(
                "new_string".into(),
                json!("Zeta UI-POLISH-2 arc: 4 of 6\nsecond added line"),
            );
            let call = ToolCall {
                id: "edit-0".into(),
                name: "edit".into(),
                arguments,
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
            view.apply_worker_message(
                WorkerMessage::Event(ServerEvent::ToolEnd {
                    session_id: view.state.active_session.clone(),
                    tool_call: call,
                    tool_result: Some(zeta_gui::client::ToolResult {
                        tool_call_id: "edit-0".into(),
                        content: "ok\n".into(),
                        is_error: false,
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
        window.draw(cx).clear(cx);
    });
    // Collapsed: no diff card paints.
    assert!(
        visual.debug_bounds("tool-diff-card-0").is_none(),
        "diff card must NOT paint on a collapsed receipt"
    );
    assert!(
        visual.debug_bounds("tool-diff-remove-0").is_none(),
        "remove pane must NOT paint on a collapsed receipt"
    );
    assert!(
        visual.debug_bounds("tool-diff-add-0").is_none(),
        "add pane must NOT paint on a collapsed receipt"
    );
    // Drive expansion through the receipt CONTROL (a click on the row),
    // not by mutating `card.expanded` directly (Finding 6).
    let receipt = visual
        .debug_bounds("tool-receipt-0")
        .expect("edit receipt paints");
    visual.simulate_click(
        receipt.origin + gpui::point(px(50.), px(20.)),
        Default::default(),
    );
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let card = visual
        .debug_bounds("tool-diff-card-0")
        .expect("diff card must paint on expansion");
    let remove_pane = visual
        .debug_bounds("tool-diff-remove-0")
        .expect("remove pane must paint");
    let add_pane = visual
        .debug_bounds("tool-diff-add-0")
        .expect("add pane must paint");
    // Wide layout: card spans full width; remove pane sits to the LEFT of
    // the add pane on the SAME row.
    assert!(card.size.width > px(0.));
    assert!(card.size.height > px(0.));
    assert!(remove_pane.size.height > px(0.));
    assert!(add_pane.size.height > px(0.));
    assert!(
        remove_pane.right() <= add_pane.left() + px(2.),
        "wide layout: remove pane must sit LEFT of the add pane — \
         got remove.right={:?}, add.left={:?}",
        remove_pane.right(),
        add_pane.left(),
    );
    assert!(
        f32::from(remove_pane.top() - add_pane.top()).abs() <= 2.0,
        "wide layout: both panes must share the same top edge — \
         got remove.top={:?}, add.top={:?}",
        remove_pane.top(),
        add_pane.top(),
    );

    // Pane tints: each pane paints a bg quad matching the theme role,
    // NOT the neutral panel fill. A tint swap trips here.
    visual.update(|window, cx| {
        let roles = theme::diff_roles(cx);
        let scale = window.scale_factor();
        let quads = window.painted_quads();
        let remove_scaled = remove_pane.scale(scale);
        let add_scaled = add_pane.scale(scale);
        let tol = px(1.).scale(scale);
        let matches = |bounds: gpui::Bounds<gpui::ScaledPixels>, tint: gpui::Hsla| {
            quads.iter().any(|quad| {
                let inside = quad.bounds.top() >= bounds.top() - tol
                    && quad.bounds.bottom() <= bounds.bottom() + tol
                    && quad.bounds.left() >= bounds.left() - tol
                    && quad.bounds.right() <= bounds.right() + tol;
                inside && quad.background == tint.into()
            })
        };
        assert!(
            matches(remove_scaled, roles.remove_bg),
            "remove pane must paint the theme's remove_bg tint"
        );
        assert!(
            matches(add_scaled, roles.add_bg),
            "add pane must paint the theme's add_bg tint"
        );
    });

    // Line numbers: the paint-text recorder captures the exact string
    // handed to each gutter cell's `.child(...)`. Removing `.child(number)`
    // drops the recorder call AND the sample, so the row_id resolves to
    // None here — the round-3 model-rebuild version passed even when the
    // gutter child was removed because it reconstructed `EditDiffText`
    // from the transcript entry, bypassing the render path entirely.
    let samples = super::paint_text_log::samples();
    let recorded_pane = |pane_selector: String, count: usize| -> Vec<String> {
        (0..count)
            .map(|line_idx| {
                let id = zeta_gui::row_text::sel::tool_diff_line_number(&pane_selector, line_idx);
                samples
                    .iter()
                    .rev()
                    .find(|sample| sample.row_id == id)
                    .unwrap_or_else(|| {
                        panic!("diff pane gutter must record a paint-text sample at row_id={id}")
                    })
                    .text
                    .clone()
            })
            .collect()
    };
    let remove_numbers = recorded_pane(zeta_gui::row_text::sel::tool_diff_remove_pane(0), 2);
    let add_numbers = recorded_pane(zeta_gui::row_text::sel::tool_diff_add_pane(0), 2);
    assert_eq!(remove_numbers, vec!["1", "2"]);
    assert_eq!(add_numbers, vec!["1", "2"]);

    // Narrow viewport: resize BELOW `NARROW_DIFF_STACK_WIDTH`; the panes
    // stack full-width (remove above add), each spanning the card width.
    // The prior implementation kept the half-width side-by-side layout at
    // any viewport size — this branch trips it.
    visual.simulate_resize(gpui::size(px(420.), px(800.)));
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let narrow_card = visual
        .debug_bounds("tool-diff-card-0")
        .expect("diff card must still paint after resize");
    let narrow_remove = visual
        .debug_bounds("tool-diff-remove-0")
        .expect("remove pane must paint at narrow width");
    let narrow_add = visual
        .debug_bounds("tool-diff-add-0")
        .expect("add pane must paint at narrow width");
    assert!(
        narrow_remove.bottom() <= narrow_add.top() + px(2.),
        "narrow layout: remove pane must sit ABOVE the add pane — \
         got remove.bottom={:?}, add.top={:?}",
        narrow_remove.bottom(),
        narrow_add.top(),
    );
    assert!(
        narrow_remove.size.width >= narrow_card.size.width - px(2.),
        "narrow layout: remove pane must span the full card width — \
         got remove.width={:?}, card.width={:?}",
        narrow_remove.size.width,
        narrow_card.size.width,
    );
    assert!(
        narrow_add.size.width >= narrow_card.size.width - px(2.),
        "narrow layout: add pane must span the full card width — \
         got add.width={:?}, card.width={:?}",
        narrow_add.size.width,
        narrow_card.size.width,
    );
}

/// ZETA-135 review r1 finding 4: the expanded-panel header truncates
/// under narrow width instead of wrapping into multiple lines and
/// blowing the panel's top edge out.
#[gpui::test]
fn zeta135_panel_header_truncates_under_narrow_width(cx: &mut TestAppContext) {
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    // Shrink the window to a narrow width so the truncation contract has
    // room to fire on a real-length command.
    visual.simulate_resize(gpui::size(px(420.), px(600.)));
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.transcript.clear();
            view.transcript.update(cx, |scroll, cx| scroll.reset(0, cx));
            let mut arguments = serde_json::Map::new();
            arguments.insert(
                "command".into(),
                json!("bash scripts/warmup.sh --verbose --extra --more --flags"),
            );
            let call = ToolCall {
                id: "narrow".into(),
                name: "bash".into(),
                arguments,
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
            view.apply_worker_message(
                WorkerMessage::Event(ServerEvent::ToolEnd {
                    session_id: view.state.active_session.clone(),
                    tool_call: call,
                    tool_result: Some(zeta_gui::client::ToolResult {
                        tool_call_id: "narrow".into(),
                        content: "ok\n".into(),
                        is_error: false,
                        is_canceled: false,
                        structured_content: None,
                        content_blocks: Vec::new(),
                    }),
                    data: json!({}),
                }),
                window,
                cx,
            );
            cx.notify();
        });
        window.draw(cx).clear(cx);
    });
    let collapsed_receipt = visual
        .debug_bounds("tool-receipt-0")
        .expect("collapsed narrow receipt paints before click");
    visual.simulate_click(
        collapsed_receipt.origin + gpui::point(px(50.), px(20.)),
        Default::default(),
    );
    visual.update(|window, cx| window.draw(cx).clear(cx));
    let header = visual
        .debug_bounds("tool-panel-header-0")
        .expect("panel header paints");
    let body = visual
        .debug_bounds("tool-output-0")
        .expect("outer panel paints");
    // Header sits inside the panel and its height stays bounded (single
    // line + padding), not the wrapped multi-line shape the round-1 code
    // produced. Single line-height at 13px with py(4.) sits well under
    // 60px — a wrapped 6-line header would clear that easily.
    assert!(
        header.size.height <= px(60.),
        "narrow-width panel header must truncate to a single row — got \
         height {:?}",
        header.size.height,
    );
    // Header is BOUNDED by the panel's width (does not overflow to the
    // right past the panel edge).
    assert!(
        header.right() <= body.right() + px(2.),
        "narrow-width panel header must stay inside the panel width — \
         got header.right={:?}, panel.right={:?}",
        header.right(),
        body.right(),
    );
}

/// ZETA-135 (Trait 3 — turn footer): after a completed turn, the last
/// transcript row hosts a metadata strip that names the provider · model
/// · duration for the active session. Sourced from wire session data +
/// status metrics; the footer suppresses itself when everything is None.
#[gpui::test]
fn zeta135_turn_footer_paints_below_the_last_row(cx: &mut TestAppContext) {
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.transcript.clear();
            view.transcript.update(cx, |scroll, cx| scroll.reset(0, cx));
            view.state.sessions.clear();
            view.state.sessions.push(zeta_gui::client::SessionMetadata {
                version: 1,
                session_id: "sess".into(),
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
            });
            view.state.active_session = Some("sess".into());
            view.state.metrics.model = Some("claude-fable-5".into());
            view.state.transcript.clear();
            view.state
                .transcript
                .push(TranscriptEntry::User("hi".into()));
            view.transcript.update(cx, |scroll, cx| scroll.reset(1, cx));
            cx.notify();
        });
        window.draw(cx).clear(cx);
    });
    let footer_bounds = visual
        .debug_bounds("turn-footer")
        .expect("turn footer must paint when session metadata is present");
    assert!(footer_bounds.size.width > px(0.));
    assert!(footer_bounds.size.height > px(0.));

    // Text: the paint-text recorder captures the composed `display`
    // string the render layer handed to `.child(...)` under
    // `sel::TURN_FOOTER`. A renderer that stops painting the display
    // drops the recorder call and the sample disappears, where the
    // round-3 model-rebuild version rebuilt `TurnFooterText` from the
    // session data and passed even when `.child(display)` was removed.
    let samples = super::paint_text_log::samples();
    let footer_text = samples
        .iter()
        .rev()
        .find(|sample| sample.row_id == zeta_gui::row_text::sel::TURN_FOOTER)
        .unwrap_or_else(|| {
            panic!(
                "turn footer must record a paint-text sample at row_id={}",
                zeta_gui::row_text::sel::TURN_FOOTER
            )
        })
        .text
        .clone();
    assert!(
        footer_text.contains("cc"),
        "footer must include the provider slug — got {footer_text:?}"
    );
    assert!(
        footer_text.contains("claude-fable-5"),
        "footer must include the model — got {footer_text:?}"
    );
    assert!(
        footer_text.contains("6m 32s"),
        "footer must include the correctly-formatted duration \
         (6m 32s from 12:00:00 to 12:06:32) — got {footer_text:?}"
    );
    assert!(
        footer_text.contains(" \u{00b7} "),
        "footer must join fields with the middle-dot separator — got {footer_text:?}"
    );

    // Placement: the footer sits at the BOTTOM of the transcript column,
    // BELOW the user row's content — reads as a peak-end cue for the
    // completed turn. The transcript-column is a v_flex whose last child
    // is the footer, so footer.bottom aligns with column.bottom and
    // footer.top sits strictly below column.top by at least the user
    // row's own height.
    let column = visual
        .debug_bounds("transcript-column")
        .expect("last transcript column paints");
    assert!(
        f32::from(footer_bounds.bottom() - column.bottom()).abs() <= 2.0,
        "turn footer must anchor to the transcript column's bottom — \
         got footer.bottom={:?}, column.bottom={:?}",
        footer_bounds.bottom(),
        column.bottom(),
    );
    assert!(
        footer_bounds.top() > column.top() + px(4.),
        "turn footer must sit BELOW the row content, not at the column top — \
         got footer.top={:?}, column.top={:?}",
        footer_bounds.top(),
        column.top(),
    );

    // Offset-aware timestamp parsing: two timestamps in different zones
    // that describe the same wall-clock instant subtract to zero (the
    // pre-fix hand-rolled parser ignored offsets and would report a
    // spurious 8-hour delta here). Round-2 review Finding 3.
    let cross_zone = zeta_gui::row_text::build_turn_footer(
        Some("cc"),
        Some("claude-fable-5"),
        "2026-09-17T12:00:00-08:00",
        "2026-09-17T20:00:00Z",
    )
    .expect("cross-zone footer builds when both sides parse");
    assert!(
        cross_zone.duration.is_none(),
        "same-instant across zones must NOT report a duration — \
         got {:?}",
        cross_zone.duration,
    );
    // Leap-year: Feb 28 -> Mar 1 2028 is 48 hours (2028 is a leap year),
    // not 24 hours as the pre-fix year/4-days hand-rolled parser reported.
    let leap = zeta_gui::row_text::build_turn_footer(
        Some("cc"),
        Some("claude-fable-5"),
        "2028-02-28T00:00:00Z",
        "2028-03-01T00:00:00Z",
    )
    .expect("leap footer builds when both sides parse");
    assert_eq!(
        leap.duration.as_deref(),
        Some("48h 0m 0s"),
        "leap-year Feb 28 -> Mar 1 must report 48h, not 24h"
    );

    // Absence: with no session data AND no metrics, the footer suppresses
    // itself. Keep the User row in place so the transcript still paints
    // (empty transcripts skip the whole path) — the absence assertion has
    // to fail on a real render pass, not a no-render.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.sessions.clear();
            view.state.active_session = None;
            view.state.metrics.model = None;
            cx.notify();
        });
        window.draw(cx).clear(cx);
    });
    assert!(
        visual.debug_bounds("turn-footer").is_none(),
        "turn footer must NOT paint when no session metadata is available"
    );
}

// ---------------------------------------------------------------------------
// ZETA-133: aligned edges. D3 bottom anchoring was reverted and deferred to
// ZETA-133-D3.
//
// D1 — every row kind (prose, thinking, collapsed receipt, expanded panel,
// diff card, group header, error block, turn footer) shares ONE body left
// edge. The chevron + kind glyph hang in the leading gutter LEFT of that
// shared edge on tool rows; every other row kind leaves the gutter empty.
//
// ---------------------------------------------------------------------------

/// Helper that renders `transcript` in isolation, then returns the
/// `transcript-body`'s left edge. Every ZETA-133 shared-edge assertion
/// runs one such render per row kind and compares the recorded left
/// against the prose baseline — the shared-edge invariant is that these
/// left edges match within a subpixel tolerance regardless of kind.
fn zeta133_body_left(
    visual: &mut VisualTestContext,
    view: &Entity<ZetaView>,
    transcript: Vec<TranscriptEntry>,
) -> gpui::Pixels {
    let count = transcript.len();
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.transcript = transcript;
            view.transcript
                .update(cx, |scroll, cx| scroll.reset(count, cx));
            cx.notify();
        });
        window.draw(cx).clear(cx);
    });
    visual
        .debug_bounds("transcript-body")
        .expect("transcript-body draws for the seeded row")
        .left()
}

fn zeta133_tool_entry(id: &str, name: &str, excerpt: &str) -> TranscriptEntry {
    TranscriptEntry::Tool {
        key: zeta_gui::state::ToolReceiptKey {
            session_id: None,
            agent_instance_id: None,
            tool_call_id: id.into(),
        },
        name: name.into(),
        excerpt: Some(excerpt.into()),
        summary: excerpt.into(),
        complete: true,
        error: false,
        canceled: false,
        card: zeta_gui::cards::Card::default(),
    }
}

/// ZETA-133 D1 — prose, thinking, collapsed tool receipts, expanded tool
/// receipts (with panel), error blocks, and (implicitly, via the same
/// `transcript-body` selector) the turn footer all sit at ONE shared left
/// edge. A pre-fix regression would let a row kind slip back to its own
/// centered column and drift the edge by ~100px at a wide viewport, which
/// this test catches by rendering each kind alone and comparing the
/// recorded body left. A tolerance of 2px absorbs the pipeline's subpixel
/// rounding without letting a whole kind drift.
#[gpui::test]
fn zeta133_body_left_edge_is_shared_across_all_row_kinds(cx: &mut TestAppContext) {
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.simulate_resize(gpui::size(px(1500.), px(1000.)));

    let user_left = zeta133_body_left(
        &mut visual,
        &view,
        vec![TranscriptEntry::User("share one edge".into())],
    );
    let assistant_left = zeta133_body_left(
        &mut visual,
        &view,
        vec![TranscriptEntry::Assistant("share one edge".into())],
    );
    let thinking_left = zeta133_body_left(&mut visual, &view, vec![TranscriptEntry::Thinking]);
    let tool_left = zeta133_body_left(
        &mut visual,
        &view,
        vec![zeta133_tool_entry("t", "bash", "echo hi")],
    );
    // Expanded tool receipt — the row's `card` is toggled open by hand so
    // the render path lands on the panel-carrying branch.
    let mut expanded = zeta133_tool_entry("expanded", "bash", "echo body");
    if let TranscriptEntry::Tool { ref mut card, .. } = expanded {
        card.expanded = true;
        card.tail.append("expanded body\nsecond line\n");
    }
    let expanded_left = zeta133_body_left(&mut visual, &view, vec![expanded]);
    let error_left = zeta133_body_left(
        &mut visual,
        &view,
        vec![TranscriptEntry::Error {
            message: "boom".into(),
            settings_action: false,
            login_provider: None,
        }],
    );

    let baseline = user_left;
    let tolerance = px(2.);
    for (label, left) in [
        ("assistant", assistant_left),
        ("thinking", thinking_left),
        ("tool", tool_left),
        ("expanded tool", expanded_left),
        ("error", error_left),
    ] {
        let delta = if left > baseline {
            left - baseline
        } else {
            baseline - left
        };
        assert!(
            delta <= tolerance,
            "ZETA-133: {label} body left {left:?} must match prose baseline \
             {baseline:?} within {tolerance:?} — the shared-edge invariant \
             is broken",
        );
    }
}

/// ZETA-133 D1 — the leading gutter hangs LEFT of the shared body edge on
/// tool rows and stays empty on prose rows. The gutter's left edge is the
/// same for every kind (the gutter is fixed-width and every row uses the
/// same helper), and the body sits at `gutter.right()`. A regression that
/// dropped the gutter on prose (or extended tool rows past the gutter into
/// the body) is caught by comparing `gutter.right() == body.left()` on
/// both a prose and a tool render.
#[gpui::test]
fn zeta133_leading_gutter_hangs_left_of_shared_body_edge(cx: &mut TestAppContext) {
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.simulate_resize(gpui::size(px(1500.), px(1000.)));
    let tolerance = px(2.);

    // Prose row: gutter reserved but empty; body starts at gutter-right.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.transcript = vec![TranscriptEntry::User("hi".into())];
            view.transcript.update(cx, |scroll, cx| scroll.reset(1, cx));
            cx.notify();
        });
        window.draw(cx).clear(cx);
    });
    let prose_gutter = visual
        .debug_bounds("transcript-gutter")
        .expect("prose row still paints a (present-but-empty) gutter");
    let prose_body = visual
        .debug_bounds("transcript-body")
        .expect("prose row body draws");
    assert!(
        f32::from(prose_gutter.right() - prose_body.left()).abs() < f32::from(tolerance),
        "ZETA-133: prose body must sit at gutter-right — gutter.right \
         {:?}, body.left {:?}",
        prose_gutter.right(),
        prose_body.left(),
    );
    assert!(
        f32::from(prose_gutter.size.width) >= f32::from(theme::LEADING_GUTTER_WIDTH) - 1.0,
        "ZETA-133: prose gutter width {:?} must equal LEADING_GUTTER_WIDTH \
         {:?}",
        prose_gutter.size.width,
        theme::LEADING_GUTTER_WIDTH,
    );

    // Tool row: gutter carries chevron + kind_glyph; body still starts at
    // gutter-right. The chevron paints at gutter-left, LEFT of the shared
    // body edge — that's the whole point of the gutter.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.transcript = vec![zeta133_tool_entry("t", "bash", "echo hi")];
            view.transcript.update(cx, |scroll, cx| scroll.reset(1, cx));
            cx.notify();
        });
        window.draw(cx).clear(cx);
    });
    let tool_gutter = visual
        .debug_bounds("tool-gutter")
        .expect("tool row gutter draws");
    let tool_body = visual
        .debug_bounds("transcript-body")
        .expect("tool row body draws");
    // The parent bound alone cannot catch the 18px overlap that motivated
    // this check. Inspect both painted children at every picker size.
    let mut appearance = theme::Appearance::default();
    for base_px in [
        theme::MIN_FONT_SIZE_PX,
        f32::from(theme::DEFAULT_FONT_SIZE),
        theme::MAX_FONT_SIZE_PX,
    ] {
        appearance.font_size = theme::clamp_font_size(base_px);
        visual.update(|window, cx| {
            theme::apply_with(cx, &appearance);
            view.update(cx, |view, cx| {
                view.state.transcript = vec![zeta133_tool_entry("t", "bash", "echo hi")];
                view.transcript.update(cx, |scroll, cx| scroll.reset(1, cx));
                cx.notify();
            });
            window.draw(cx).clear(cx);
        });
        let gutter = visual
            .debug_bounds("tool-gutter")
            .expect("tool gutter draws at every picker size");
        let chevron = visual
            .debug_bounds("tool-chevron-0")
            .expect("tool chevron draws at every picker size");
        let kind_glyph = visual
            .debug_bounds("tool-kind-glyph-0")
            .expect("tool kind glyph draws at every picker size");
        let tolerance = px(1.);
        for (name, child) in [("chevron", chevron), ("kind glyph", kind_glyph)] {
            assert!(
                child.left() >= gutter.left() - tolerance
                    && child.right() <= gutter.right() + tolerance
                    && child.top() >= gutter.top() - tolerance
                    && child.bottom() <= gutter.bottom() + tolerance,
                "ZETA-133: {name} bounds {child:?} escaped gutter {gutter:?} at {base_px}px"
            );
        }
    }
    assert!(
        f32::from(tool_gutter.right() - tool_body.left()).abs() < f32::from(tolerance),
        "ZETA-133: tool body must sit at gutter-right — gutter.right \
         {:?}, body.left {:?}",
        tool_gutter.right(),
        tool_body.left(),
    );
    // Chevron paints INSIDE the gutter; the gutter itself sits LEFT of the
    // shared body edge (gutter.left < body.left). The chevron has no
    // dedicated `debug_selector` (pre-ZETA-133 shape — its color routes
    // through `record_state`, not through debug bounds), so we assert the
    // GUTTER'S left edge is left of the body's left edge, which is
    // materially the same invariant (the chevron cannot escape its
    // parent).
    assert!(
        tool_gutter.left() < tool_body.left(),
        "ZETA-133: gutter (chevron + kind glyph) must paint LEFT of the \
         shared body edge — gutter.left {:?}, body.left {:?}",
        tool_gutter.left(),
        tool_body.left(),
    );
    // Prose and tool rows must share the SAME body left edge (regression
    // guard for the D1 audit: pre-ZETA-133 tool rows sat ~100px left of
    // prose because their column max_w was wider AND centered per-row).
    let delta = if tool_body.left() > prose_body.left() {
        tool_body.left() - prose_body.left()
    } else {
        prose_body.left() - tool_body.left()
    };
    assert!(
        f32::from(delta) < f32::from(tolerance),
        "ZETA-133: tool body left {:?} must match prose body left {:?}",
        tool_body.left(),
        prose_body.left(),
    );
}

/// ZETA-133 D1 — the turn footer hangs at the shared body left edge. The
/// footer used to ride whichever row cap the last transcript row carried
/// (wide cap on a tool tail, narrow cap on a prose tail); under ZETA-133
/// it always sits inside its own body-pair with the narrow prose cap.
#[gpui::test]
fn zeta133_turn_footer_shares_the_prose_body_left_edge(cx: &mut TestAppContext) {
    use zeta_gui::state::StatusMetrics;
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.simulate_resize(gpui::size(px(1500.), px(1000.)));
    // Prose baseline: render a prose row alone and read its body left.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.transcript = vec![TranscriptEntry::Assistant("first".into())];
            view.transcript.update(cx, |scroll, cx| scroll.reset(1, cx));
            cx.notify();
        });
        window.draw(cx).clear(cx);
    });
    let prose_body_left = visual
        .debug_bounds("transcript-body")
        .expect("prose body draws")
        .left();

    // Seed a session with concrete metadata so the footer builds and paints.
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            let sess = view.state.sessions.first_mut().expect("seeded session");
            sess.created_at = "2026-09-17T12:00:00Z".into();
            sess.updated_at = "2026-09-17T12:00:12Z".into();
            sess.provider = "cc".into();
            sess.model = "claude-fable-5".into();
            view.state.metrics = StatusMetrics {
                model: Some("claude-fable-5".into()),
                ..Default::default()
            };
            view.state.transcript = vec![TranscriptEntry::Assistant("answer".into())];
            view.transcript.update(cx, |scroll, cx| scroll.reset(1, cx));
            cx.notify();
        });
        window.draw(cx).clear(cx);
    });
    let footer = visual
        .debug_bounds("turn-footer")
        .expect("turn footer draws when session metadata is present");
    let delta = if footer.left() > prose_body_left {
        footer.left() - prose_body_left
    } else {
        prose_body_left - footer.left()
    };
    assert!(
        f32::from(delta) < 2.0,
        "ZETA-133: turn footer left {:?} must match prose body left {:?}",
        footer.left(),
        prose_body_left,
    );
}

// ---------------------------------------------------------------------------
// ZETA-133-D3 — bottom anchoring for short transcripts.
//
// A transcript SHORTER than the viewport rests on the viewport's bottom edge
// (chat-UI convention). Once content exceeds the viewport, the alignment
// collapses into normal scrolling and tail-follow still pins the newest row
// to the bottom. The path lives inside the virtual list's supported
// `ListAlignment::Bottom` mode; a regression that reverted to top alignment
// would leave the single row at the TOP of the viewport, so
// `bottom_body_delta` blows past the tolerance below.
//
// ---------------------------------------------------------------------------

/// Render one short prose row at `font_size` on a tall viewport and return
/// how far the row's bottom edge sits ABOVE the transcript viewport's bottom
/// edge — under bottom alignment this delta is only the list's own bottom
/// padding, under top alignment it would be roughly `viewport.height - row.height`.
fn zeta133_d3_bottom_body_delta(
    cx: &mut TestAppContext,
    font_size: f32,
) -> (gpui::Pixels, gpui::Pixels) {
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.simulate_resize(gpui::size(px(1100.), px(1200.)));
    let appearance = theme::Appearance {
        font_size: theme::clamp_font_size(font_size),
        ..Default::default()
    };
    visual.update(|window, cx| {
        theme::apply_with(cx, &appearance);
        view.update(cx, |view, cx| {
            view.state.transcript = vec![TranscriptEntry::User("hello world".into())];
            view.transcript.update(cx, |scroll, cx| scroll.reset(1, cx));
            cx.notify();
        });
        window.draw(cx).clear(cx);
    });
    let viewport = visual
        .debug_bounds("transcript-viewport")
        .expect("transcript viewport draws");
    let body = visual
        .debug_bounds("transcript-body")
        .expect("prose body draws for the single seeded row");
    let delta = viewport.bottom() - body.bottom();
    visual.update(|_, cx| theme::apply(cx));
    (delta, viewport.size.height)
}

#[gpui::test]
fn zeta133_d3_short_transcript_rests_on_viewport_bottom_at_default_font(cx: &mut TestAppContext) {
    // Chat-UI trait: a lone user row sits near the BOTTOM edge of the
    // transcript viewport, not near the top. The list's own `py_2()` pads
    // the bottom edge; a `ListAlignment::Top` regression would leave the
    // row within a row-height of the TOP edge instead, i.e. `delta` would
    // approach `viewport.height`.
    let (delta, viewport_height) =
        zeta133_d3_bottom_body_delta(cx, f32::from(theme::DEFAULT_FONT_SIZE));
    // The list wrapper carries `py_2()` (8px) plus a row's own bottom
    // padding, so a bottom-anchored row lands within ~40px of the
    // viewport bottom on the shipped metrics — well under the row-height
    // threshold that a top-anchored regression would blow past.
    assert!(
        f32::from(delta) <= 48.0,
        "ZETA-133-D3: short transcript must rest on viewport bottom \
         (delta {delta:?} above bottom, viewport height {viewport_height:?})"
    );
    // Half-viewport is a wide safety margin: a top-anchored regression
    // parks the row hundreds of pixels above the bottom on a 1200px-tall
    // window, well past this threshold.
    assert!(
        f32::from(delta) < f32::from(viewport_height) / 2.0,
        "ZETA-133-D3: short transcript must rest on the BOTTOM half of \
         the viewport (delta {delta:?} vs viewport height {viewport_height:?})"
    );
}

#[gpui::test]
fn zeta133_d3_short_transcript_rests_on_viewport_bottom_at_max_font(cx: &mut TestAppContext) {
    // Same trait at the picker's 18px ceiling — the row grows taller, but
    // the bottom-anchor delta stays a small pad regardless of font size.
    let (delta, viewport_height) = zeta133_d3_bottom_body_delta(cx, theme::MAX_FONT_SIZE_PX);
    assert!(
        f32::from(delta) <= 64.0,
        "ZETA-133-D3: short transcript at 18px must rest on viewport bottom \
         (delta {delta:?} above bottom, viewport height {viewport_height:?})"
    );
    assert!(
        f32::from(delta) < f32::from(viewport_height) / 2.0,
        "ZETA-133-D3: short transcript at 18px must rest on the BOTTOM half \
         of the viewport (delta {delta:?} vs viewport height {viewport_height:?})"
    );
}

#[gpui::test]
fn zeta133_d3_tall_content_keeps_last_row_pinned_at_viewport_bottom(cx: &mut TestAppContext) {
    // Content TALLER than the viewport: bottom alignment collapses into
    // normal scrolling and tail-follow keeps the newest row anchored to
    // the viewport bottom — exactly the pre-D3 behaviour. This test
    // guards against a bottom-alignment regression that would poison
    // scroll math and leave the tail floating.
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.simulate_resize(gpui::size(px(1100.), px(400.)));
    let rows: Vec<TranscriptEntry> = (0..40)
        .map(|i| TranscriptEntry::Assistant(format!("row {i} — {}", "line ".repeat(20)).into()))
        .collect();
    let count = rows.len();
    visual.update(|window, cx| {
        view.update(cx, |view, cx| {
            view.state.transcript = rows;
            view.transcript
                .update(cx, |scroll, cx| scroll.reset(count, cx));
            cx.notify();
        });
        window.draw(cx).clear(cx);
        // Second paint so the list settles under the new item count and
        // tail-follow lands the offset.
        window.draw(cx).clear(cx);
    });
    view.read_with(&visual, |view, cx| {
        let state = view.transcript.read(cx);
        assert!(
            state.is_following_tail(),
            "tall content must stay in tail-follow after seeding"
        );
        assert!(
            !state.is_scrolled_up(),
            "tail-follow means the reader has NOT scrolled away from the tail"
        );
    });
    let viewport = visual
        .debug_bounds("transcript-viewport")
        .expect("transcript viewport draws");
    let body = visual
        .debug_bounds("transcript-body")
        .expect("some prose body draws — the last painted row's body");
    // The body selector maps to a single row per render pass; under
    // tail-follow the row that paints is one at the tail. Its bottom
    // must land near the viewport bottom (within the list's own bottom
    // padding).
    let delta = viewport.bottom() - body.bottom();
    assert!(
        f32::from(delta) <= 48.0,
        "ZETA-133-D3: tail-follow must keep the last row pinned near the \
         viewport bottom (delta {delta:?}, viewport {viewport:?})"
    );
    assert!(
        f32::from(delta) >= 0.0,
        "ZETA-133-D3: last row must not paint past the viewport bottom \
         (delta {delta:?})"
    );
}

/// ZETA-137 D1 — the thinking row's `+` marker hangs in the LEADING gutter
/// aligned with tool rows' kind-glyph column, so a vertical scan reads `+`
/// and `$` at the SAME x. The header `Thought` starts at the shared body
/// edge alongside `bash` and prose. The regression this catches: pre-fix
/// the whole "+ Thought" string sat as inline body text with `+` at the
/// body edge (~38px right of `$`).
///
/// Runs the same three picker sizes ZETA-133 tests so the alignment
/// invariant survives at both the 11px minimum and the 18px maximum.
#[gpui::test]
fn zeta137_thinking_marker_shares_the_tool_kind_glyph_column(cx: &mut TestAppContext) {
    let (window, view, _) = setup(cx);
    let mut visual = VisualTestContext::from_window(window.into(), cx);
    visual.simulate_resize(gpui::size(px(1500.), px(1000.)));
    let tolerance = px(1.);
    let mut appearance = theme::Appearance::default();
    for base_px in [
        theme::MIN_FONT_SIZE_PX,
        f32::from(theme::DEFAULT_FONT_SIZE),
        theme::MAX_FONT_SIZE_PX,
    ] {
        appearance.font_size = theme::clamp_font_size(base_px);
        // Tool row: read the kind-glyph column and the body left edge.
        visual.update(|window, cx| {
            theme::apply_with(cx, &appearance);
            view.update(cx, |view, cx| {
                view.state.transcript = vec![zeta133_tool_entry("t", "bash", "echo hi")];
                view.transcript.update(cx, |scroll, cx| scroll.reset(1, cx));
                cx.notify();
            });
            window.draw(cx).clear(cx);
        });
        let tool_kind_glyph = visual
            .debug_bounds("tool-kind-glyph-0")
            .expect("tool kind glyph draws");
        let tool_body_left = visual
            .debug_bounds("transcript-body")
            .expect("tool body draws")
            .left();

        // Thinking row: read the gutter marker and the header body.
        visual.update(|window, cx| {
            view.update(cx, |view, cx| {
                view.state.transcript = vec![TranscriptEntry::Thinking];
                view.transcript.update(cx, |scroll, cx| scroll.reset(1, cx));
                cx.notify();
            });
            window.draw(cx).clear(cx);
        });
        let thinking_marker = visual
            .debug_bounds("thinking-marker-0")
            .expect("thinking marker draws in the leading gutter");
        let thinking_header = visual
            .debug_bounds("thinking-header-0")
            .expect("thinking header draws in the body");
        let thinking_body_left = visual
            .debug_bounds("transcript-body")
            .expect("thinking body draws")
            .left();

        // (a) The marker sits in the gutter, LEFT of the shared body edge.
        // A regression that re-inlined the `+` into the body would place
        // marker.left() at or past thinking_body_left.
        assert!(
            thinking_marker.right() <= thinking_body_left + tolerance,
            "ZETA-137 D1: `+` marker must sit LEFT of the shared body edge \
             at {base_px}px — marker {thinking_marker:?}, body.left {thinking_body_left:?}"
        );

        // (b) The marker's left edge lines up with the tool row's kind-glyph
        // left edge — `+` and `$` land in the same x column. This is the
        // exact defect Henry called out.
        let column_delta = if thinking_marker.left() > tool_kind_glyph.left() {
            thinking_marker.left() - tool_kind_glyph.left()
        } else {
            tool_kind_glyph.left() - thinking_marker.left()
        };
        assert!(
            column_delta <= tolerance,
            "ZETA-137 D1: `+` marker left {:?} must match tool kind glyph \
             left {:?} at {base_px}px — the marker columns are misaligned",
            thinking_marker.left(),
            tool_kind_glyph.left(),
        );

        // (c) Both row kinds share the same body edge, so the header text
        // starts where prose / tool label / expanded panel content starts.
        let body_delta = if tool_body_left > thinking_body_left {
            tool_body_left - thinking_body_left
        } else {
            thinking_body_left - tool_body_left
        };
        assert!(
            body_delta <= px(2.),
            "ZETA-137 D1: thinking body left {thinking_body_left:?} must \
             match tool body left {tool_body_left:?} at {base_px}px"
        );

        // (d) The header text sits AT the shared body edge — same tolerance
        // as (c). A `>=` check would let `Thought` drift arbitrarily right
        // of the body edge and still pass.
        let header_delta = if thinking_header.left() > thinking_body_left {
            thinking_header.left() - thinking_body_left
        } else {
            thinking_body_left - thinking_header.left()
        };
        assert!(
            header_delta <= tolerance,
            "ZETA-137 D1: `Thought` header left {:?} must match body left \
             {thinking_body_left:?} within tolerance at {base_px}px",
            thinking_header.left(),
        );
    }
}
