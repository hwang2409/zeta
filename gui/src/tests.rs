use super::*;
use gpui::{TestAppContext, VisualTestContext, WindowHandle};
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
            let quads: Vec<_> = window
                .painted_quads()
                .into_iter()
                .filter(|quad| {
                    quad.border_color == cx.theme().primary
                        && quad.border_widths.left > gpui::ScaledPixels::default()
                })
                .collect();
            assert!(!quads.is_empty(), "user message borders were painted");
            assert!(quads.len() < 20, "the virtual list paints only nearby rows");
            let viewport = transcript.scale(window.scale_factor());
            for quad in quads {
                assert!(quad.content_mask.bounds.top() >= viewport.top());
                assert!(quad.content_mask.bounds.bottom() <= viewport.bottom());
                assert!(quad.content_mask.bounds.left() >= viewport.left());
                assert!(quad.content_mask.bounds.right() <= viewport.right());
            }
        });
    }
}

fn png_bytes() -> Vec<u8> {
    b"\x89PNG\r\n\x1a\n".to_vec()
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
    });
    assert!(visual.debug_bounds("composer-chip").is_none());
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
            assert!(view.state.transcript.is_empty());
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
