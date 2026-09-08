mod composer;
use composer::Composer;
use gpui::{
    div, prelude::*, px, App, Bounds, Context, FocusHandle, Focusable, KeyDownEvent, Render,
    ScrollHandle, Task, Window, WindowBounds, WindowOptions,
};
use gpui_platform::application;
use std::env;
use std::path::PathBuf;
use std::sync::mpsc::{self, Sender};
use std::thread;
use std::time::Duration;
use zeta_gui::client::Approval;
use zeta_gui::state::{AppState, ConnectionState, TranscriptEntry};
use zeta_gui::worker::{CommandMessage, ConnectionWorker, WorkerMessage};

const BG: u32 = 0x111210;
const PANEL: u32 = 0x181a17;
const LINE: u32 = 0x343832;
const TEXT: u32 = 0xd8ddd5;
const MUTED: u32 = 0x8a9287;
const SIGNAL: u32 = 0xe1a84b;

struct ZetaView {
    state: AppState,
    transcript_scroll: ScrollHandle,
    composer: gpui::Entity<Composer>,
    pending_command: bool,
    command_error: Option<String>,
    commands: Sender<CommandMessage>,
    focus_handle: FocusHandle,
    _poll_task: Task<()>,
}

impl Focusable for ZetaView {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl ZetaView {
    fn new(window: &mut Window, cx: &mut Context<Self>, socket: Option<PathBuf>) -> Self {
        let (command_tx, command_rx) = mpsc::channel();
        let (message_tx, message_rx) = mpsc::channel();
        thread::spawn(move || {
            ConnectionWorker {
                commands: command_rx,
                messages: message_tx,
                socket,
            }
            .run()
        });
        let poll_task = cx.spawn_in(window, async move |this, cx| loop {
            cx.background_executor()
                .timer(Duration::from_millis(50))
                .await;
            let updates = message_rx.try_iter().collect::<Vec<_>>();
            if updates.is_empty() {
                continue;
            }
            if this
                .update_in(cx, |view, _window, cx| {
                    for update in updates {
                        view.apply_worker_message(update, cx);
                    }
                    cx.notify();
                })
                .is_err()
            {
                return;
            }
        });
        let composer = cx.new(Composer::new);
        cx.subscribe(&composer, |view, _, _: &composer::Submit, cx| {
            view.send_composer(cx);
        })
        .detach();
        window.focus(&composer.focus_handle(cx), cx);
        Self {
            state: AppState::default(),
            transcript_scroll: ScrollHandle::new(),
            composer,
            pending_command: false,
            command_error: None,
            commands: command_tx,
            focus_handle: cx.focus_handle(),
            _poll_task: poll_task,
        }
    }

    fn apply_worker_message(&mut self, message: WorkerMessage, cx: &mut Context<Self>) {
        match message {
            WorkerMessage::Sessions(sessions) => self.state.sessions = sessions,
            WorkerMessage::Session(session) => {
                self.pending_command = false;
                self.state.active_session = Some(session.session_id.clone());
                if !self
                    .state
                    .sessions
                    .iter()
                    .any(|item| item.session_id == session.session_id)
                {
                    self.state.sessions.push(session);
                }
                self.state.transcript.clear();
                self.transcript_scroll = ScrollHandle::new();
            }
            WorkerMessage::Status(status) => self.state.apply_status(status),
            WorkerMessage::Sent(text) => {
                self.pending_command = false;
                self.state.streaming = true;
                self.state.transcript.push(TranscriptEntry::User(text));
                self.composer.update(cx, |composer, cx| {
                    composer.reset();
                    cx.notify();
                });
            }
            WorkerMessage::Rejected(error) => {
                self.pending_command = false;
                self.command_error = Some(error);
            }
            WorkerMessage::Connected => self.state.connection = ConnectionState::Connected,
            WorkerMessage::Event(event) => self.state.apply(event),
            WorkerMessage::Lost(error) => {
                self.pending_command = false;
                self.state.mark_connection_lost(error);
            }
        }
    }

    fn can_change_session(&self) -> bool {
        matches!(self.state.connection, ConnectionState::Connected)
            && !self.state.streaming
            && self.state.approvals.is_empty()
            && !self.pending_command
    }

    fn queue(&mut self, command: CommandMessage) {
        self.command_error = None;
        if self.commands.send(command).is_err() {
            self.pending_command = false;
            self.state.mark_connection_lost("connection worker stopped");
        }
    }

    fn new_session(&mut self, _: &gpui::ClickEvent, _: &mut Window, cx: &mut Context<Self>) {
        if self.can_change_session() {
            self.pending_command = true;
            self.queue(CommandMessage::NewSession);
            cx.notify();
        }
    }

    fn resume(&mut self, id: String, _: &gpui::ClickEvent, _: &mut Window, cx: &mut Context<Self>) {
        if self.can_change_session() {
            self.pending_command = true;
            self.queue(CommandMessage::Resume(id));
            cx.notify();
        }
    }

    fn control_key(&mut self, event: &KeyDownEvent, _: &mut Window, cx: &mut Context<Self>) {
        if !self.state.approvals.is_empty() {
            let command = match event.keystroke.key.as_str() {
                "enter" => Some(CommandMessage::Approve(
                    self.state.approvals[0].request_id.clone(),
                )),
                "escape" => Some(CommandMessage::Deny(
                    self.state.approvals[0].request_id.clone(),
                )),
                _ => None,
            };
            if let Some(command) = command {
                self.queue(command);
                cx.stop_propagation();
            }
        } else if event.keystroke.key == "escape"
            || (event.keystroke.key == "c" && event.keystroke.modifiers.control)
        {
            self.queue(CommandMessage::Abort);
            cx.stop_propagation();
        }
        cx.notify();
    }

    fn send_composer(&mut self, cx: &mut Context<Self>) {
        if !self.can_change_session() || self.state.active_session.is_none() {
            return;
        }
        let text = self.composer.read(cx).content.to_string();
        if text.trim().is_empty() {
            return;
        }
        self.pending_command = true;
        self.queue(CommandMessage::Send(text));
        cx.notify();
    }

    fn approval_action(
        &mut self,
        approval: Approval,
        approve: bool,
        _: &gpui::ClickEvent,
        _: &mut Window,
        _: &mut Context<Self>,
    ) {
        let command = if approve {
            CommandMessage::Approve(approval.request_id)
        } else {
            CommandMessage::Deny(approval.request_id)
        };
        self.queue(command);
    }

    fn reconnect(&mut self, _: &gpui::ClickEvent, _: &mut Window, _: &mut Context<Self>) {
        self.state.begin_reconnect();
        self.queue(CommandMessage::Reconnect);
    }

    fn render_sidebar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let mut sidebar = div()
            .w(px(260.))
            .h_full()
            .flex()
            .flex_col()
            .p_4()
            .gap_3()
            .bg(gpui::rgb(PANEL))
            .border_r_1()
            .border_color(gpui::rgb(LINE));
        sidebar = sidebar.child(
            div()
                .text_size(px(18.))
                .font_weight(gpui::FontWeight::BOLD)
                .text_color(gpui::rgb(TEXT))
                .child("zeta"),
        );
        sidebar = sidebar.child(div().h(px(1.)).w_full().bg(gpui::rgb(LINE)));
        let new_button = div()
            .id("new-session")
            .w_full()
            .min_h(px(40.))
            .px_3()
            .py_2()
            .border_1()
            .border_color(gpui::rgb(LINE))
            .text_color(gpui::rgb(TEXT))
            .opacity(if self.can_change_session() { 1. } else { 0.4 })
            .child("new session")
            .hover(|this| this.bg(gpui::rgb(LINE)))
            .on_click(cx.listener(Self::new_session));
        sidebar = sidebar.child(new_button);
        for session in &self.state.sessions {
            let id = session.session_id.clone();
            let label = if session.name.is_empty() {
                session.session_id.clone()
            } else {
                session.name.clone()
            };
            sidebar = sidebar.child(
                div()
                    .id(id.clone())
                    .w_full()
                    .min_h(px(40.))
                    .px_3()
                    .py_2()
                    .text_color(gpui::rgb(MUTED))
                    .opacity(if self.can_change_session() { 1. } else { 0.4 })
                    .child(label)
                    .hover(|this| this.text_color(gpui::rgb(TEXT)))
                    .on_click(cx.listener(move |view, event, window, cx| {
                        view.resume(id.clone(), event, window, cx)
                    })),
            );
        }
        sidebar
    }

    fn render_transcript(&self) -> impl IntoElement {
        // Read the previous layout before new output changes the content height.
        // Scrolling away leaves the offset in place until the user returns to the tail.
        if self.transcript_scroll.offset().y + self.transcript_scroll.max_offset().y <= px(1.) {
            self.transcript_scroll.scroll_to_bottom();
        }
        let mut transcript = div()
            .id("transcript")
            .flex_1()
            .min_h_0()
            .overflow_y_scroll()
            .track_scroll(&self.transcript_scroll)
            .flex()
            .flex_col()
            .w_full()
            .max_w(px(760.))
            .self_center()
            .p_6()
            .gap_4();
        for entry in &self.state.transcript {
            let row = match entry {
                TranscriptEntry::User(text) => div()
                    .text_color(gpui::rgb(TEXT))
                    .font_weight(gpui::FontWeight::BOLD)
                    .child(text.clone()),
                TranscriptEntry::Assistant(text) => {
                    div().text_color(gpui::rgb(TEXT)).pl_4().child(text.clone())
                }
                TranscriptEntry::Tool {
                    name,
                    summary,
                    complete,
                    error,
                    ..
                } => {
                    let marker = if *error {
                        "error"
                    } else if *complete {
                        "done"
                    } else {
                        "running"
                    };
                    div()
                        .font_family("monospace")
                        .text_size(px(12.))
                        .text_color(gpui::rgb(if *error { 0xd97979 } else { MUTED }))
                        .child(format!("{marker}  {name}  {summary}"))
                }
            };
            transcript = transcript.child(row.flex_shrink_0());
        }
        transcript
    }

    fn render_composer(&self) -> impl IntoElement {
        let hint = if self.state.streaming {
            "waiting for response; escape aborts"
        } else if self.state.active_session.is_none() {
            "select or create a session to send"
        } else {
            "enter sends; shift-enter adds a line"
        };
        div()
            .flex_shrink_0()
            .w_full()
            .border_t_1()
            .border_color(gpui::rgb(LINE))
            .p_4()
            .child(self.composer.clone())
            .child(
                div()
                    .text_size(px(12.))
                    .text_color(gpui::rgb(MUTED))
                    .child(hint),
            )
            .when_some(self.command_error.clone(), |view, error| {
                view.child(div().text_color(gpui::rgb(0xd97979)).child(error))
            })
    }

    fn render_approval(&self, approval: &Approval, cx: &mut Context<Self>) -> impl IntoElement {
        let approval_for_yes = approval.clone();
        let approval_for_no = approval.clone();
        let approve = div()
            .id("approve")
            .min_w(px(100.))
            .min_h(px(40.))
            .p_3()
            .bg(gpui::rgb(SIGNAL))
            .text_color(gpui::black())
            .child("approve")
            .on_click(cx.listener(move |view, event, window, cx| {
                view.approval_action(approval_for_yes.clone(), true, event, window, cx)
            }));
        let deny = div()
            .id("deny")
            .min_w(px(100.))
            .min_h(px(40.))
            .p_3()
            .border_1()
            .border_color(gpui::rgb(LINE))
            .text_color(gpui::rgb(TEXT))
            .child("deny")
            .on_click(cx.listener(move |view, event, window, cx| {
                view.approval_action(approval_for_no.clone(), false, event, window, cx)
            }));
        div()
            .absolute()
            .inset_0()
            .flex()
            .items_center()
            .justify_center()
            .bg(gpui::black().opacity(0.7))
            .child(
                div()
                    .w(px(430.))
                    .p_6()
                    .bg(gpui::rgb(PANEL))
                    .border_1()
                    .border_color(gpui::rgb(SIGNAL))
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap_4()
                            .child(
                                div()
                                    .text_size(px(16.))
                                    .font_weight(gpui::FontWeight::BOLD)
                                    .text_color(gpui::rgb(TEXT))
                                    .child("approval required"),
                            )
                            .child(
                                div()
                                    .text_color(gpui::rgb(TEXT))
                                    .child(approval.tool_call.name.clone()),
                            )
                            .child(
                                div()
                                    .font_family("monospace")
                                    .text_size(px(12.))
                                    .text_color(gpui::rgb(MUTED))
                                    .child(
                                        serde_json::to_string(&approval.tool_call.arguments)
                                            .unwrap_or_else(|_| "arguments unavailable".to_owned()),
                                    ),
                            )
                            .child(div().flex().gap_3().child(approve).child(deny)),
                    ),
            )
    }
}

impl Render for ZetaView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let enabled = !self.pending_command && self.state.approvals.is_empty();
        self.composer
            .update(cx, |composer, _| composer.enabled = enabled);
        let mut root = div()
            .size_full()
            .flex()
            .bg(gpui::rgb(BG))
            .text_size(px(14.))
            .text_color(gpui::rgb(TEXT))
            .track_focus(&self.focus_handle)
            .capture_action(cx.listener(|view, _: &composer::Submit, _, cx| {
                if let Some(approval) = view.state.approvals.first() {
                    view.queue(CommandMessage::Approve(approval.request_id.clone()));
                    cx.stop_propagation();
                    cx.notify();
                }
            }))
            .on_key_down(cx.listener(Self::control_key))
            .child(self.render_sidebar(cx))
            .child(
                div()
                    .flex_1()
                    .h_full()
                    .flex()
                    .flex_col()
                    .child(self.render_transcript())
                    .child(self.render_composer()),
            );
        root = match &self.state.connection {
            ConnectionState::Connected => root,
            ConnectionState::Reconnecting => root.child(
                div()
                    .absolute()
                    .top_0()
                    .right_0()
                    .m_4()
                    .p_2()
                    .bg(gpui::rgb(LINE))
                    .text_color(gpui::rgb(TEXT))
                    .child("connecting..."),
            ),
            ConnectionState::Lost(error) => root.child(
                div()
                    .absolute()
                    .top_0()
                    .right_0()
                    .m_4()
                    .p_3()
                    .bg(gpui::rgb(PANEL))
                    .border_1()
                    .border_color(gpui::rgb(0xd97979))
                    .child(
                        div()
                            .flex()
                            .gap_3()
                            .items_center()
                            .child(format!("connection lost: {error}"))
                            .child(
                                div()
                                    .id("reconnect")
                                    .min_h(px(40.))
                                    .p_2()
                                    .text_color(gpui::rgb(SIGNAL))
                                    .child("reconnect")
                                    .on_click(cx.listener(Self::reconnect)),
                            ),
                    ),
            ),
        };
        if let Some(approval) = self.state.approvals.first().cloned() {
            root = root.child(self.render_approval(&approval, cx));
        }
        root
    }
}

fn main() {
    let socket = env::args()
        .skip(1)
        .collect::<Vec<_>>()
        .windows(2)
        .find(|pair| pair[0] == "--socket")
        .map(|pair| PathBuf::from(&pair[1]));
    application().run(move |cx: &mut App| {
        composer::bind_keys(cx);
        let bounds = Bounds::centered(None, gpui::size(px(1100.), px(760.)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            move |window, cx| cx.new(|cx| ZetaView::new(window, cx, socket)),
        )
        .expect("open zeta window");
        cx.activate(true);
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use zeta_gui::client::ToolCall;

    #[gpui::test]
    fn transcript_scrolls_and_follows_output_only_at_the_tail(cx: &mut gpui::TestAppContext) {
        use gpui::{point, size, ScrollDelta, ScrollWheelEvent};
        let scroll = ScrollHandle::new();
        let handle = scroll.clone();
        let (commands, _receiver) = mpsc::channel();
        let window = cx.open_window(size(px(1100.), px(760.)), move |_, cx| ZetaView {
            state: AppState {
                connection: ConnectionState::Connected,
                transcript: vec![TranscriptEntry::Assistant("line\n".repeat(100))],
                ..Default::default()
            },
            transcript_scroll: handle,
            composer: cx.new(Composer::new),
            pending_command: false,
            command_error: None,
            commands,
            focus_handle: cx.focus_handle(),
            _poll_task: Task::ready(()),
        });
        let draw = |cx: &mut gpui::TestAppContext| {
            cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx))
                .unwrap();
        };
        let append = |cx: &mut gpui::TestAppContext| {
            window
                .update(cx, |view, _, cx| {
                    if let Some(TranscriptEntry::Assistant(text)) = view.state.transcript.last_mut()
                    {
                        text.push_str(&"new line\n".repeat(20));
                    }
                    cx.notify();
                })
                .unwrap();
        };
        let wheel = |cx: &mut gpui::TestAppContext, delta| {
            gpui::VisualTestContext::from_window(window.into(), cx).simulate_event(
                ScrollWheelEvent {
                    position: scroll.bounds().center(),
                    delta: ScrollDelta::Pixels(point(px(0.), px(delta))),
                    ..Default::default()
                },
            );
            cx.executor().run_until_parked();
        };
        draw(cx);
        assert!(scroll.max_offset().y > px(0.), "long output must overflow");
        assert_eq!(scroll.offset().y, -scroll.max_offset().y);
        assert!(
            scroll.bounds().bottom() < px(760.),
            "the composer stays in the viewport"
        );
        let previous_max = scroll.max_offset().y;
        append(cx);
        draw(cx);
        assert!(scroll.max_offset().y > previous_max);
        assert_eq!(scroll.offset().y, -scroll.max_offset().y);

        wheel(cx, 200.);
        draw(cx);
        let away = scroll.offset().y;
        assert!(away > -scroll.max_offset().y);
        append(cx);
        draw(cx);
        assert_eq!(
            scroll.offset().y,
            away,
            "new output must preserve the reading position"
        );

        wheel(cx, -10000.);
        draw(cx);
        assert_eq!(scroll.offset().y, -scroll.max_offset().y);
        append(cx);
        draw(cx);
        assert_eq!(
            scroll.offset().y,
            -scroll.max_offset().y,
            "returning to the tail resumes following"
        );
    }

    #[gpui::test]
    fn approval_and_abort_keys_reach_worker_from_composer(cx: &mut gpui::TestAppContext) {
        cx.update(composer::bind_keys);
        let (commands, receiver) = mpsc::channel();
        let window = cx.add_window(move |window, cx| {
            let composer = cx.new(Composer::new);
            window.focus(&composer.focus_handle(cx), cx);
            ZetaView {
                state: AppState {
                    connection: ConnectionState::Connected,
                    approvals: vec![Approval {
                        request_id: "one".into(),
                        tool_call: ToolCall {
                            id: "one".into(),
                            name: "read".into(),
                            arguments: Default::default(),
                        },
                    }],
                    ..Default::default()
                },
                composer,
                transcript_scroll: ScrollHandle::new(),
                pending_command: false,
                command_error: None,
                commands,
                focus_handle: cx.focus_handle(),
                _poll_task: Task::ready(()),
            }
        });
        cx.simulate_keystrokes(window.into(), "enter");
        assert!(matches!(receiver.try_recv(), Ok(CommandMessage::Approve(id)) if id == "one"));
        cx.simulate_keystrokes(window.into(), "escape");
        assert!(matches!(receiver.try_recv(), Ok(CommandMessage::Deny(id)) if id == "one"));
        window
            .update(cx, |view, _, cx| {
                view.state.approvals.clear();
                view.state.streaming = true;
                cx.notify();
            })
            .unwrap();
        cx.simulate_keystrokes(window.into(), "escape");
        assert!(matches!(receiver.try_recv(), Ok(CommandMessage::Abort)));
    }
}
