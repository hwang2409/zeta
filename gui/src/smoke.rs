//! Opt-in smoke driver: real input events, real worker/socket, native Metal pixels.
use super::*;
use gpui_kit::test::TestWindowExt;

/// Encode a tiny checkerboard PNG for the ZETA-112 attachment-chrome shot.
/// A one-shot helper — the smoke driver seeds a real attachment so the
/// thumbnail slot decodes rather than falling back to the file glyph.
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
    let Some(path) = env::var_os("ZETA_GUI_SMOKE_IMAGE") else {
        return;
    };
    // Optional second capture — the ZETA-112 composer chrome with pending
    // attachment chips visible, before the settings modal covers them. Emits
    // a separate PNG so the primary shot stays comparable with prior tickets.
    let attachment_path = env::var_os("ZETA_GUI_SMOKE_ATTACHMENT_IMAGE");
    view.update(cx, |_, cx| {
        cx.spawn_in(window, async move |view, cx| {
            let mut phase = 0;
            for _ in 0..600 {
                cx.background_executor()
                    .timer(Duration::from_millis(50))
                    .await;
                let finished = cx
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
                                // ZETA-112 attachment capture — seed a mixed
                                // batch (valid decoded thumbnail, valid fallback
                                // glyph, and one error chip surfaced by a per
                                // file parse failure) so the shot proves the
                                // typed pending model. Clear before the modal
                                // shot below so the primary after-screenshot
                                // stays unchanged.
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
                                window
                                    .render_to_image()
                                    .expect("native renderer capture")
                                    .save(PathBuf::from(&path))
                                    .expect("save smoke screenshot");
                                println!("SMOKE-PASS: {}", PathBuf::from(&path).display());
                                cx.quit();
                                return true;
                            }
                            _ => {}
                        }
                        false
                    })
                    .expect("smoke window update");
                if finished {
                    return;
                }
            }
            panic!("smoke session did not complete within 30 seconds");
        })
        .detach();
    });
}
