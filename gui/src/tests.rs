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
    visual.update(|_, cx| {
        view.update(cx, |view, cx| {
            assert!(view.attach_from_clipboard(cx));
        });
    });
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
