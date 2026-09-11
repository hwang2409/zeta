//! Opt-in smoke driver: real input events, real worker/socket, native Metal pixels.
use super::*;
use gpui_kit::test::TestWindowExt;

pub fn start(view: &Entity<ZetaView>, window: &mut Window, cx: &mut App) {
    let Some(path) = env::var_os("ZETA_GUI_SMOKE_IMAGE") else {
        return;
    };
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
                            // Capture mid-stream: the transcript carries the
                            // generic "+ Thought" marker AND the assistant
                            // preamble, and `state.streaming` flips can_send
                            // to false so the Send button paints its disabled
                            // outline. No approval modal is used here, so the
                            // transcript is fully visible.
                            2 if ready_to_capture => {
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
