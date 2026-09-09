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
                        let (ready, active, idle, answered, approval) = {
                            let view = entity.read(cx);
                            if let Some(error) = &view.command_error {
                                panic!("smoke command failed: {error}");
                            }
                            (
                                view.state.connection == ConnectionState::Connected,
                                view.state.active_session.is_some(),
                                !view.state.streaming && !view.pending_command,
                                view.state
                                    .transcript
                                    .iter()
                                    .any(|row| matches!(row, TranscriptEntry::Assistant(_))),
                                !view.state.approvals.is_empty(),
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
                            2 if approval => window.press("enter", cx),
                            2 if answered && idle => {
                                phase = 3;
                            }
                            3 => {
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
