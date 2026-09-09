mod composer;
mod transcript;
use composer::Composer;
use gpui::{
    div, list, prelude::*, px, App, Bounds, Context, FocusHandle, Focusable, FollowMode,
    KeyDownEvent, ListAlignment, ListState, Render, Task, Window, WindowBounds, WindowOptions,
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

use zeta_gui::appearance::{Appearance, Palette};

fn appearance(window: &Window) -> Appearance {
    match window.appearance() {
        gpui::WindowAppearance::Light | gpui::WindowAppearance::VibrantLight => Appearance::Light,
        gpui::WindowAppearance::Dark | gpui::WindowAppearance::VibrantDark => Appearance::Dark,
    }
}

fn transcript_list() -> ListState {
    let state = ListState::new(0, ListAlignment::Top, px(200.));
    state.set_follow_mode(FollowMode::Tail);
    state
}

struct ZetaView {
    state: AppState,
    appearance: Appearance,
    transcript_scroll: ListState,
    composer: gpui::Entity<Composer>,
    pending_command: bool,
    command_error: Option<String>,
    commands: Sender<CommandMessage>,
    focus_handle: FocusHandle,
    _poll_task: Task<()>,
    #[cfg(test)]
    rendered_rows: usize,
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
        cx.observe_window_appearance(window, |view, window, cx| {
            view.appearance = appearance(window);
            cx.notify();
        })
        .detach();
        let composer = cx.new(Composer::new);
        cx.subscribe(&composer, |view, _, _: &composer::Submit, cx| {
            view.send_composer(cx);
        })
        .detach();
        window.focus(&composer.focus_handle(cx), cx);
        Self {
            state: AppState::default(),
            appearance: appearance(window),
            transcript_scroll: transcript_list(),
            composer,
            pending_command: false,
            command_error: None,
            commands: command_tx,
            focus_handle: cx.focus_handle(),
            _poll_task: poll_task,
            #[cfg(test)]
            rendered_rows: 0,
        }
    }

    fn apply_worker_message(&mut self, message: WorkerMessage, cx: &mut Context<Self>) {
        let previous_session = self.state.active_session.clone();
        match message {
            WorkerMessage::Sessions(list) => {
                self.state.sessions = list.sessions;
                self.state.sessions_truncated = list.truncated;
            }
            WorkerMessage::Session(session) => {
                self.pending_command = false;
                self.state.select_session(Some(session.session_id.clone()));
                if !self
                    .state
                    .sessions
                    .iter()
                    .any(|item| item.session_id == session.session_id)
                {
                    self.state.sessions.push(session);
                }
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
            WorkerMessage::Event(event) => {
                if let Some(index) = self.state.apply(event) {
                    self.remeasure_row(index);
                }
            }
            WorkerMessage::Lost(error) => {
                self.pending_command = false;
                self.state.mark_connection_lost(error);
            }
        }
        if self.state.active_session != previous_session {
            self.transcript_scroll = transcript_list();
        }
    }

    fn remeasure_row(&self, index: usize) {
        if index < self.transcript_scroll.item_count() {
            self.transcript_scroll.remeasure_items(index..index + 1);
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
        if self.can_change_session() && self.state.active_session.as_deref() != Some(&id) {
            self.pending_command = true;
            self.queue(CommandMessage::Resume(id));
            cx.notify();
        }
    }

    fn control_key(&mut self, event: &KeyDownEvent, window: &mut Window, cx: &mut Context<Self>) {
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
        } else if event.keystroke.key == "tab" {
            if event.keystroke.modifiers.shift {
                window.focus_prev(cx);
            } else {
                window.focus_next(cx);
            }
            cx.stop_propagation();
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
        let p = self.appearance.palette();
        let mut sidebar = div()
            .w(px(260.))
            .flex_shrink_0()
            .h_full()
            .flex()
            .flex_col()
            .p_4()
            .gap_3()
            .bg(gpui::rgb(p.panel))
            .border_r_1()
            .border_color(gpui::rgb(p.border));
        sidebar = sidebar.child(
            div()
                .text_size(px(18.))
                .font_weight(gpui::FontWeight::BOLD)
                .text_color(gpui::rgb(p.text))
                .child("zeta"),
        );
        sidebar = sidebar.child(div().h(px(1.)).w_full().bg(gpui::rgb(p.border)));
        let new_button = div()
            .id("new-session")
            .w_full()
            .min_h(px(40.))
            .px_3()
            .py_2()
            .border_1()
            .border_color(gpui::rgb(p.border))
            .text_color(gpui::rgb(p.text))
            .opacity(if self.can_change_session() { 1. } else { 0.4 })
            .child("new session")
            .hover(|this| this.bg(gpui::rgb(p.border)))
            .on_click(cx.listener(Self::new_session));
        sidebar = sidebar.child(new_button);
        if self.state.sessions_truncated {
            sidebar = sidebar.child(
                div()
                    .text_color(gpui::rgb(p.muted))
                    .child("showing a partial session list"),
            );
        }
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
                    .text_color(gpui::rgb(p.muted))
                    .opacity(if self.can_change_session() { 1. } else { 0.4 })
                    .child(label)
                    .hover(|this| this.text_color(gpui::rgb(p.text)))
                    .on_click(cx.listener(move |view, event, window, cx| {
                        view.resume(id.clone(), event, window, cx)
                    })),
            );
        }
        sidebar
    }

    fn render_transcript(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let count = self.state.transcript.len();
        let previous_count = self.transcript_scroll.item_count();
        if count != previous_count {
            self.transcript_scroll.splice(
                previous_count.min(count)..previous_count,
                count.saturating_sub(previous_count),
            );
        }
        list(
            self.transcript_scroll.clone(),
            cx.processor(|view, index, _, cx| {
                view.render_transcript_row(index, cx).into_any_element()
            }),
        )
        .flex_1()
        .min_h_0()
        .w_full()
        .max_w(px(760.))
        .self_center()
        .p_6()
    }

    fn render_transcript_row(&mut self, index: usize, cx: &mut Context<Self>) -> gpui::Div {
        #[cfg(test)]
        {
            self.rendered_rows += 1;
        }
        let p = self.appearance.palette();
        let entry = &self.state.transcript[index];
        let row = match entry {
            TranscriptEntry::User(text) => div()
                .text_color(gpui::rgb(p.text))
                .font_weight(gpui::FontWeight::BOLD)
                .child(text.clone()),
            TranscriptEntry::Assistant(text) => match &text.root {
                Some(root) => div().child(transcript::render_block(
                    root,
                    self.appearance,
                    format!("markdown-{index}"),
                )),
                None => div()
                    .when(text.preview_truncated, |view| {
                        view.child(
                            div()
                                .text_color(gpui::rgb(p.muted))
                                .child("showing latest response text"),
                        )
                    })
                    .child(gpui::SharedString::from(text.source.clone())),
            },
            TranscriptEntry::Tool {
                name,
                summary,
                card,
                ..
            } => {
                let marker = entry.tool_marker();
                let label = card
                    .agent_label
                    .as_ref()
                    .map_or_else(|| name.clone(), |label| format!("{label} / {name}"));
                let disclosure = if card.expanded { "collapse" } else { "expand" };
                let border = if card.agent_label.is_some() {
                    p.nested_border
                } else {
                    p.border
                };
                let header = div()
                    .id(format!("card-{index}"))
                    .debug_selector(|| format!("card-{index}"))
                    .tab_index(0)
                    .min_h(px(40.))
                    .px_3()
                    .py_2()
                    .flex()
                    .items_center()
                    .gap_2()
                    .cursor_pointer()
                    .text_size(px(12.))
                    .text_color(gpui::rgb(if entry.unsuccessful() {
                        p.error
                    } else {
                        p.muted
                    }))
                    .hover(|view| view.bg(gpui::rgb(p.panel)))
                    .focus(|view| view.border_color(gpui::rgb(p.accent)).border_1())
                    .child(
                        div()
                            .debug_selector(move || format!("tool-marker-{marker}"))
                            .flex_shrink_0()
                            .font_family("monospace")
                            .child(marker),
                    )
                    .child(
                        div()
                            .min_w_0()
                            .flex_1()
                            .truncate()
                            .child(format!("{label}  {summary}")),
                    )
                    .child(div().flex_shrink_0().child(disclosure))
                    .on_click(cx.listener(move |view, _, _, cx| {
                        view.state.toggle_card(index);
                        view.remeasure_row(index);
                        cx.notify();
                    }))
                    .on_key_down(cx.listener(move |view, event: &KeyDownEvent, _, cx| {
                        if event.keystroke.key == "enter" && view.state.approvals.is_empty() {
                            view.state.toggle_card(index);
                            view.remeasure_row(index);
                            cx.stop_propagation();
                            cx.notify();
                        }
                    }));
                div()
                    .min_w_0()
                    .border_1()
                    .border_color(gpui::rgb(border))
                    .rounded(px(4.))
                    .when(card.agent_label.is_some(), |view| view.ml_4())
                    .child(header)
                    .when(card.expanded, |view| {
                        view.child(
                            div()
                                .id(format!("tail-{index}"))
                                .max_h(px(320.))
                                .overflow_y_scroll()
                                .border_t_1()
                                .border_color(gpui::rgb(border))
                                .px_3()
                                .py_2()
                                .font_family("monospace")
                                .text_size(px(12.))
                                .line_height(px(20.))
                                .when(card.tail.truncated, |view| {
                                    view.child(
                                        div()
                                            .text_color(gpui::rgb(p.muted))
                                            .child("output truncated"),
                                    )
                                })
                                .child(if card.tail.text.is_empty() {
                                    "waiting for output".into()
                                } else {
                                    card.tail.text.clone()
                                }),
                        )
                    })
            }
        };
        div().w_full().min_h(px(24.)).pb_4().child(row)
    }

    fn render_composer(&self) -> impl IntoElement {
        let p = self.appearance.palette();
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
            .border_color(gpui::rgb(p.border))
            .p_4()
            .child(self.composer.clone())
            .child(
                div()
                    .text_size(px(12.))
                    .text_color(gpui::rgb(p.muted))
                    .child(hint),
            )
            .when_some(self.command_error.clone(), |view, error| {
                view.child(div().text_color(gpui::rgb(p.error)).child(error))
            })
    }

    fn render_approval(&self, approval: &Approval, cx: &mut Context<Self>) -> impl IntoElement {
        let p = self.appearance.palette();
        let approval_for_yes = approval.clone();
        let approval_for_no = approval.clone();
        let approve = div()
            .id("approve")
            .min_w(px(100.))
            .min_h(px(40.))
            .p_3()
            .bg(gpui::rgb(p.accent))
            .text_color(gpui::rgb(p.background))
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
            .border_color(gpui::rgb(p.border))
            .text_color(gpui::rgb(p.text))
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
                    .bg(gpui::rgb(p.panel))
                    .border_1()
                    .border_color(gpui::rgb(p.accent))
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap_4()
                            .child(
                                div()
                                    .text_size(px(16.))
                                    .font_weight(gpui::FontWeight::BOLD)
                                    .text_color(gpui::rgb(p.text))
                                    .child("approval required"),
                            )
                            .child(
                                div()
                                    .text_color(gpui::rgb(p.text))
                                    .child(approval.tool_call.name.clone()),
                            )
                            .child(
                                div()
                                    .font_family("monospace")
                                    .text_size(px(12.))
                                    .text_color(gpui::rgb(p.muted))
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

impl ZetaView {
    fn render_status(&self, p: Palette) -> impl IntoElement {
        div()
            .flex()
            .justify_between()
            .items_center()
            .flex_shrink_0()
            .min_h(px(32.))
            .border_t_1()
            .border_color(gpui::rgb(p.border))
            .px_4()
            .gap_4()
            .text_size(px(12.))
            .text_color(gpui::rgb(p.muted))
            .child(
                div()
                    .min_w_0()
                    .truncate()
                    .child(self.state.metrics.model_label().to_owned()),
            )
            .child(
                div()
                    .flex_shrink_0()
                    .font_family("monospace")
                    .child(format!(
                        "tokens {}    cache {}",
                        self.state.metrics.tokens_label(),
                        self.state.metrics.cache_label()
                    )),
            )
    }
}

impl Render for ZetaView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let p = self.appearance.palette();
        let enabled = !self.pending_command && self.state.approvals.is_empty();
        self.composer
            .update(cx, |composer, _| composer.enabled = enabled);
        let mut root = div()
            .size_full()
            .flex()
            .font_family("Helvetica Neue")
            .bg(gpui::rgb(p.background))
            .text_size(px(14.))
            .text_color(gpui::rgb(p.text))
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
                    .min_w_0()
                    .h_full()
                    .flex()
                    .flex_col()
                    .child(self.render_transcript(cx))
                    .child(self.render_composer())
                    .child(self.render_status(p)),
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
                    .bg(gpui::rgb(p.border))
                    .text_color(gpui::rgb(p.text))
                    .child("connecting..."),
            ),
            ConnectionState::Lost(error) => root.child(
                div()
                    .absolute()
                    .top_0()
                    .right_0()
                    .m_4()
                    .p_3()
                    .bg(gpui::rgb(p.panel))
                    .border_1()
                    .border_color(gpui::rgb(p.error))
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
                                    .text_color(gpui::rgb(p.accent))
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
    fn cards_support_click_enter_and_both_appearances(cx: &mut gpui::TestAppContext) {
        cx.update(composer::bind_keys);
        let window = cx.open_window(gpui::size(px(1100.), px(760.)), |window, cx| {
            ZetaView::new(
                window,
                cx,
                Some(PathBuf::from("/tmp/zeta-94-test-no-server.sock")),
            )
        });
        window
            .update(cx, |view, _, cx| {
                view.state.apply(zeta_gui::client::ServerEvent::ToolStart {
                    session_id: None,
                    tool_call: ToolCall {
                        id: "tool".into(),
                        name: "read".into(),
                        arguments: Default::default(),
                    },
                    data: serde_json::json!({}),
                });
                view.state.transcript.push(TranscriptEntry::Assistant(
                    "```rust\nfn main() {}\n```".into(),
                ));
                cx.notify();
            })
            .unwrap();
        window
            .update(cx, |view, _, cx| {
                view.apply_worker_message(
                    WorkerMessage::Event(zeta_gui::client::ServerEvent::ToolEnd {
                        session_id: None,
                        tool_call: ToolCall {
                            id: "tool".into(),
                            name: "read".into(),
                            arguments: Default::default(),
                        },
                        tool_result: Some(
                            serde_json::from_value(serde_json::json!({
                                "tool_call_id": "tool", "content": "tool execution canceled",
                                "is_error": false, "is_canceled": true
                            }))
                            .unwrap(),
                        ),
                        data: serde_json::json!({}),
                    }),
                    cx,
                );
                cx.notify();
            })
            .unwrap();
        // GPUI's platform appearance simulator is private. Force our palette
        // explicitly; production uses the window appearance observer above.
        for mode in [Appearance::Dark, Appearance::Light] {
            window
                .update(cx, |view, _, cx| {
                    view.appearance = mode;
                    cx.notify();
                })
                .unwrap();
            cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx))
                .unwrap();
        }
        let mut visual = gpui::VisualTestContext::from_window(window.into(), cx);
        visual.update(|window, cx| window.draw(cx).clear(cx));
        assert!(visual.debug_bounds("tool-marker-[canceled]").is_some());
        assert!(visual.debug_bounds("tool-marker-[done]").is_none());
        let header = visual.debug_bounds("card-0").expect("receipt header");
        assert!(header.size.height >= px(40.));
        visual.simulate_click(header.center(), Default::default());
        window.update(cx, |view, _, _| assert!(matches!(&view.state.transcript[0], TranscriptEntry::Tool { card, .. } if card.expanded))).unwrap();
        cx.simulate_keystrokes(window.into(), "enter");
        window.update(cx, |view, _, _| assert!(matches!(&view.state.transcript[0], TranscriptEntry::Tool { card, .. } if !card.expanded))).unwrap();
        // Keyboard navigation reaches the receipt from the composer as well.
        window
            .update(cx, |view, window, cx| {
                window.focus(&view.composer.focus_handle(cx), cx)
            })
            .unwrap();
        cx.simulate_keystrokes(window.into(), "tab enter");
        window.update(cx, |view, _, _| assert!(matches!(&view.state.transcript[0], TranscriptEntry::Tool { card, .. } if card.expanded))).unwrap();
    }

    #[gpui::test]
    fn code_scrolls_inside_its_block_in_both_palettes(cx: &mut gpui::TestAppContext) {
        let window = cx.open_window(gpui::size(px(700.), px(500.)), |window, cx| {
            ZetaView::new(
                window,
                cx,
                Some(PathBuf::from("/tmp/zeta-94-test-no-server.sock")),
            )
        });
        window
            .update(cx, |view, _, cx| {
                view.state.transcript = vec![TranscriptEntry::Assistant(
                    format!("```rust\nlet text = \"{}\";\n```", "long line ".repeat(80)).into(),
                )];
                cx.notify();
            })
            .unwrap();
        for mode in [Appearance::Light, Appearance::Dark] {
            window
                .update(cx, |view, _, cx| {
                    view.appearance = mode;
                    cx.notify();
                })
                .unwrap();
            let mut visual = gpui::VisualTestContext::from_window(window.into(), cx);
            visual.update(|window, cx| window.draw(cx).clear(cx));
            let viewport = visual.debug_bounds("code-scroll").unwrap();
            let before = visual.debug_bounds("code-content").unwrap();
            assert!(viewport.right() <= px(700.));
            assert!(
                before.size.width > viewport.size.width,
                "long code must not wrap"
            );
            visual.simulate_event(gpui::ScrollWheelEvent {
                position: viewport.center(),
                delta: gpui::ScrollDelta::Pixels(gpui::point(px(-80.), px(0.))),
                ..Default::default()
            });
            visual.update(|window, cx| window.draw(cx).clear(cx));
            let after = visual.debug_bounds("code-content").unwrap();
            assert!(
                after.left() < before.left(),
                "horizontal scrolling stays inside the code block"
            );
        }
    }

    #[gpui::test]
    fn active_session_click_is_a_no_op_and_switching_restores_transcripts(
        cx: &mut gpui::TestAppContext,
    ) {
        let (commands, receiver) = mpsc::channel();
        let session = |id| serde_json::from_value(serde_json::json!({"session_id":id})).unwrap();
        let window = cx.add_window(move |_, cx| ZetaView {
            state: AppState {
                active_session: Some("one".into()),
                sessions: vec![session("one"), session("two")],
                connection: ConnectionState::Connected,
                transcript: vec![TranscriptEntry::Assistant("first transcript".into())],
                ..Default::default()
            },
            appearance: Appearance::Light,
            transcript_scroll: transcript_list(),
            composer: cx.new(Composer::new),
            pending_command: false,
            command_error: None,
            commands,
            focus_handle: cx.focus_handle(),
            _poll_task: Task::ready(()),
            rendered_rows: 0,
        });
        window
            .update(cx, |view, window, cx| {
                let original = view.state.transcript.clone();
                view.resume("one".into(), &Default::default(), window, cx);
                assert!(
                    receiver.try_recv().is_err(),
                    "active row must not queue a resume"
                );
                assert!(!view.pending_command);
                assert_eq!(view.state.transcript, original);
                // Also tolerate an idempotent session response without erasing output.
                view.apply_worker_message(WorkerMessage::Session(session("one")), cx);
                assert_eq!(view.state.transcript, original);
                view.resume("two".into(), &Default::default(), window, cx);
                assert!(
                    matches!(receiver.try_recv(), Ok(CommandMessage::Resume(id)) if id == "two")
                );
                view.apply_worker_message(WorkerMessage::Session(session("two")), cx);
                assert!(view.state.transcript.is_empty());
                view.state
                    .transcript
                    .push(TranscriptEntry::User("second transcript".into()));
                view.apply_worker_message(WorkerMessage::Session(session("one")), cx);
                assert_eq!(view.state.transcript, original);
                view.apply_worker_message(WorkerMessage::Session(session("two")), cx);
                assert_eq!(
                    view.state.transcript,
                    vec![TranscriptEntry::User("second transcript".into())]
                );
            })
            .unwrap();
    }

    #[gpui::test]
    fn transcript_scrolls_and_follows_output_only_at_the_tail(cx: &mut gpui::TestAppContext) {
        use gpui::{point, size, ScrollDelta, ScrollWheelEvent};
        let scroll = transcript_list();
        let handle = scroll.clone();
        let (commands, _receiver) = mpsc::channel();
        let window = cx.open_window(size(px(1100.), px(760.)), move |_, cx| ZetaView {
            state: AppState {
                connection: ConnectionState::Connected,
                transcript: (0..50)
                    .map(|_| TranscriptEntry::User("line".into()))
                    .chain([TranscriptEntry::Assistant(
                        zeta_gui::markdown::Markdown::streaming("line\n".repeat(10)),
                    )])
                    .collect(),
                ..Default::default()
            },
            appearance: Appearance::Light,
            transcript_scroll: handle,
            composer: cx.new(Composer::new),
            pending_command: false,
            command_error: None,
            commands,
            focus_handle: cx.focus_handle(),
            _poll_task: Task::ready(()),
            rendered_rows: 0,
        });
        let draw = |cx: &mut gpui::TestAppContext| {
            cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx))
                .unwrap();
        };
        let append = |cx: &mut gpui::TestAppContext| {
            window
                .update(cx, |view, _, cx| {
                    view.apply_worker_message(
                        WorkerMessage::Event(zeta_gui::client::ServerEvent::AssistantDelta {
                            session_id: None,
                            kind: "assistant".into(),
                            delta: "new line\n".repeat(5),
                        }),
                        cx,
                    );
                    cx.notify();
                })
                .unwrap();
        };
        let wheel = |cx: &mut gpui::TestAppContext, delta| {
            gpui::VisualTestContext::from_window(window.into(), cx).simulate_event(
                ScrollWheelEvent {
                    position: point(px(700.), px(300.)),
                    delta: ScrollDelta::Pixels(point(px(0.), px(delta))),
                    ..Default::default()
                },
            );
            cx.executor().run_until_parked();
        };
        draw(cx);
        assert!(
            scroll.logical_scroll_top().item_ix > 0,
            "long transcript must overflow"
        );
        assert!(scroll.is_following_tail());
        let bottom = scroll.bounds_for_item(50).unwrap().bottom();
        assert!(bottom < px(760.), "the composer stays in the viewport");
        let before = scroll.logical_scroll_top();
        append(cx);
        draw(cx);
        assert!(
            scroll.logical_scroll_top().item_ix != before.item_ix
                || scroll.logical_scroll_top().offset_in_item != before.offset_in_item
        );
        assert_eq!(scroll.bounds_for_item(50).unwrap().bottom(), bottom);
        assert!(scroll.is_following_tail());

        wheel(cx, 200.);
        draw(cx);
        let away = scroll.logical_scroll_top();
        assert!(!scroll.is_following_tail());
        append(cx);
        draw(cx);
        assert_eq!(
            (
                scroll.logical_scroll_top().item_ix,
                scroll.logical_scroll_top().offset_in_item
            ),
            (away.item_ix, away.offset_in_item),
            "new output must preserve the reading position"
        );

        wheel(cx, -10000.);
        draw(cx);
        assert!(scroll.is_following_tail());
        append(cx);
        draw(cx);
        assert_eq!(scroll.bounds_for_item(50).unwrap().bottom(), bottom);
        assert!(
            scroll.is_following_tail(),
            "returning to the tail resumes following"
        );
    }

    #[gpui::test]
    fn streaming_draw_work_is_independent_of_transcript_length(cx: &mut gpui::TestAppContext) {
        let (commands, _receiver) = mpsc::channel();
        let window = cx.open_window(gpui::size(px(1100.), px(760.)), move |_, cx| ZetaView {
            state: AppState {
                connection: ConnectionState::Connected,
                ..Default::default()
            },
            appearance: Appearance::Light,
            transcript_scroll: transcript_list(),
            composer: cx.new(Composer::new),
            pending_command: false,
            command_error: None,
            commands,
            focus_handle: cx.focus_handle(),
            _poll_task: Task::ready(()),
            rendered_rows: 0,
        });
        let mut work = Vec::new();
        for count in [100, 10_000] {
            window
                .update(cx, |view, _, cx| {
                    view.state.transcript = (0..count)
                        .map(|_| TranscriptEntry::User("history".into()))
                        .collect();
                    view.state.transcript.push(TranscriptEntry::Assistant(
                        zeta_gui::markdown::Markdown::streaming("x".repeat(5 * 1024 * 1024)),
                    ));
                    view.transcript_scroll = transcript_list();
                    cx.notify();
                })
                .unwrap();
            for _ in 0..3 {
                window
                    .update(cx, |view, _, cx| {
                        view.rendered_rows = 0;
                        view.apply_worker_message(
                            WorkerMessage::Event(zeta_gui::client::ServerEvent::AssistantDelta {
                                session_id: None,
                                kind: "assistant".into(),
                                delta: "latest\n".into(),
                            }),
                            cx,
                        );
                        cx.notify();
                    })
                    .unwrap();
                cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx))
                    .unwrap();
                window
                    .update(cx, |view, _, _| {
                        assert!(
                            view.rendered_rows > 0 && view.rendered_rows < 60,
                            "rendered {} rows of {count}",
                            view.rendered_rows
                        );
                        work.push(view.rendered_rows);
                    })
                    .unwrap();
            }
            // Scrolling into history must also render only a viewport, and the
            // offscreen streaming row must not pull the viewport back to the end.
            window
                .update(cx, |view, _, cx| {
                    view.transcript_scroll.scroll_to(gpui::ListOffset {
                        item_ix: 20,
                        offset_in_item: px(0.),
                    });
                    view.rendered_rows = 0;
                    cx.notify();
                })
                .unwrap();
            cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx))
                .unwrap();
            window
                .update(cx, |view, _, _| {
                    assert!(view.rendered_rows > 0 && view.rendered_rows < 60);
                    assert_eq!(
                        view.transcript_scroll.item_is_below_viewport(count),
                        Some(true)
                    );
                })
                .unwrap();
            window
                .update(cx, |view, _, cx| {
                    view.rendered_rows = 0;
                    view.apply_worker_message(
                        WorkerMessage::Event(zeta_gui::client::ServerEvent::AssistantDelta {
                            session_id: None,
                            kind: "assistant".into(),
                            delta: "offscreen update\n".into(),
                        }),
                        cx,
                    );
                    cx.notify();
                })
                .unwrap();
            cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx))
                .unwrap();
            window
                .update(cx, |view, _, _| {
                    assert!(view.rendered_rows > 0 && view.rendered_rows < 60);
                    assert_eq!(view.transcript_scroll.logical_scroll_top().item_ix, 20);
                    assert!(!view.transcript_scroll.is_following_tail());
                })
                .unwrap();
        }
        assert_eq!(
            &work[..3],
            &work[3..],
            "10,000 rows must cost the same as 100 rows"
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
                appearance: Appearance::Light,
                composer,
                transcript_scroll: transcript_list(),
                pending_command: false,
                command_error: None,
                commands,
                focus_handle: cx.focus_handle(),
                _poll_task: Task::ready(()),
                rendered_rows: 0,
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
