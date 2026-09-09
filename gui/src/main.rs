extern crate gpui_kit as gpui;

mod sidebar;
#[cfg(feature = "smoke-test")]
mod smoke;

use gpui::{
    div, prelude::*, px, App, Bounds, Context, Entity, Focusable, KeyDownEvent, Render, Task,
    Window, WindowBounds, WindowOptions,
};
use gpui_kit::component::{
    alert::Alert,
    button::{Button, ButtonVariants},
    dialog::DialogButtonProps,
    input::{InputEvent, Textarea, TextareaState},
    message_scroller::{MessageScroller, MessageScrollerState},
    text::TextView,
    ActiveTheme, Disableable, Root, StyledExt, Theme, WindowExt,
};
use std::{
    borrow::Cow,
    env,
    path::PathBuf,
    sync::mpsc::{self, Sender},
    thread,
    time::Duration,
};
use zeta_gui::{
    client::Approval,
    state::{AppState, ConnectionState, TranscriptEntry},
    worker::{CommandMessage, ConnectionWorker, WorkerMessage},
};

struct DialogLayer;

impl Render for DialogLayer {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div().children(Root::render_dialog_layer(window, cx))
    }
}

struct ZetaView {
    state: AppState,
    dialogs: Entity<DialogLayer>,
    composer: Entity<TextareaState>,
    transcript: Entity<MessageScrollerState>,
    sidebar_scroll: gpui_kit::component::VirtualListScrollHandle,
    pending_command: bool,
    command_error: Option<String>,
    dialog_request: Option<String>,
    approval_pending: bool,
    commands: Sender<CommandMessage>,
    _poll_task: Option<Task<()>>,
}

fn sync_theme(window: &mut Window, cx: &mut App) {
    Theme::sync_system_appearance(Some(window), cx);
    let theme = Theme::global_mut(cx);
    theme.font_family = "JetBrains Mono".into();
    theme.mono_font_family = "JetBrains Mono".into();
    theme.font_size = px(14.);
    theme.mono_font_size = px(13.);
    Theme::sync_base(cx);
}

impl ZetaView {
    fn new(window: &mut Window, cx: &mut Context<Self>, commands: Sender<CommandMessage>) -> Self {
        sync_theme(window, cx);
        cx.observe_window_appearance(window, |_, window, cx| sync_theme(window, cx))
            .detach();
        let composer = cx.new(|cx| {
            TextareaState::new(window, cx)
                .placeholder("Message zeta")
                .submit_on_enter(true)
        });
        cx.subscribe_in(&composer, window, |view, _, event: &InputEvent, _, cx| {
            if matches!(
                event,
                InputEvent::PressEnter {
                    secondary: false,
                    shift: false
                }
            ) {
                view.send_composer(cx);
            }
        })
        .detach();
        window.focus(&composer.focus_handle(cx), cx);
        Self {
            state: AppState::default(),
            dialogs: cx.new(|_| DialogLayer),
            composer,
            transcript: cx.new(|cx| MessageScrollerState::new(0, cx)),
            sidebar_scroll: gpui_kit::component::VirtualListScrollHandle::new(),
            pending_command: false,
            command_error: None,
            dialog_request: None,
            approval_pending: false,
            commands,
            _poll_task: None,
        }
    }

    fn connect(window: &mut Window, cx: &mut Context<Self>, socket: Option<PathBuf>) -> Self {
        let (commands, command_rx) = mpsc::channel();
        let (messages, message_rx) = mpsc::channel();
        thread::spawn(move || {
            ConnectionWorker {
                commands: command_rx,
                messages,
                socket,
            }
            .run()
        });
        let mut view = Self::new(window, cx, commands);
        view._poll_task = Some(cx.spawn_in(window, async move |this, cx| loop {
            cx.background_executor()
                .timer(Duration::from_millis(50))
                .await;
            let updates: Vec<_> = message_rx.try_iter().collect();
            if !updates.is_empty()
                && this
                    .update_in(cx, |view, window, cx| {
                        for update in updates {
                            view.apply_worker_message(update, window, cx);
                        }
                    })
                    .is_err()
            {
                return;
            }
        }));
        view
    }

    fn apply_worker_message(
        &mut self,
        message: WorkerMessage,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let previous_session = self.state.active_session.clone();
        let previous_count = self.state.transcript.len();
        let mut changed_row = None;
        let mut replace = false;
        match message {
            WorkerMessage::Extensions(available) => self.state.session_view.available = available,
            WorkerMessage::Tree(tree) => self.state.session_view.branches = tree.branches,
            WorkerMessage::History(history, reset) => {
                self.state.apply_history(history, reset);
                replace = reset;
                self.pending_command = false;
            }
            WorkerMessage::Settings(settings, _) | WorkerMessage::SettingsApplied(settings) => {
                self.state.session_view.current_model = settings.model;
                self.pending_command = false;
            }
            WorkerMessage::ImagesSent(text, _) | WorkerMessage::Sent(text) => {
                self.pending_command = false;
                self.state.streaming = true;
                self.state.transcript.push(TranscriptEntry::User(text));
                self.composer
                    .update(cx, |input, cx| input.set_value("", window, cx));
            }
            WorkerMessage::Sessions(list) => {
                self.state.sessions = list.sessions;
                self.state.sessions_truncated = list.truncated;
            }
            WorkerMessage::Session(session) => {
                self.pending_command = false;
                self.state.select_session(Some(session.session_id.clone()));
                if let Some(item) = self
                    .state
                    .sessions
                    .iter_mut()
                    .find(|item| item.session_id == session.session_id)
                {
                    *item = session;
                } else {
                    self.state.sessions.insert(0, session);
                }
            }
            WorkerMessage::Status(status) => {
                if let Some(session) = &status.session {
                    if let Some(row) = self
                        .state
                        .sessions
                        .iter_mut()
                        .find(|row| row.session_id == session.session_id)
                    {
                        row.updated_at.clone_from(&session.updated_at);
                        row.name.clone_from(&session.name);
                        row.model.clone_from(&session.model);
                    }
                }
                self.state.apply_status(status);
            }
            WorkerMessage::Rejected(error) => {
                self.pending_command = false;
                self.approval_pending = false;
                self.command_error = Some(error);
            }
            WorkerMessage::Connected => self.state.connection = ConnectionState::Connected,
            WorkerMessage::Event(event) => changed_row = self.state.apply(event),
            WorkerMessage::Lost(error) => {
                self.pending_command = false;
                self.approval_pending = false;
                self.state.mark_connection_lost(error);
            }
        }
        replace |= self.state.active_session != previous_session;
        let count = self.state.transcript.len();
        self.transcript.update(cx, |scroll, cx| {
            if replace {
                scroll.reset(count, cx);
            } else {
                if count != previous_count {
                    scroll.splice(
                        previous_count.min(count)..previous_count,
                        count.saturating_sub(previous_count),
                        cx,
                    );
                }
                if let Some(index) = changed_row {
                    scroll.remeasure_items(index..index + 1, cx);
                }
            }
        });
        self.sync_approval(window, cx);
        cx.notify();
    }

    fn can_change_session(&self) -> bool {
        self.state.connection == ConnectionState::Connected
            && !self.state.streaming
            && self.state.approvals.is_empty()
            && !self.pending_command
    }

    fn composer_hint(&self) -> &'static str {
        match &self.state.connection {
            ConnectionState::Lost(_) => "Reconnect to send a message",
            ConnectionState::Reconnecting => "Connecting to zeta…",
            _ if !self.state.approvals.is_empty() => "Approve or deny the tool request to continue",
            _ if self.pending_command => "Waiting for the server…",
            _ if self.state.streaming => "Responding… Esc stops the turn",
            _ if self.state.active_session.is_none() => "Create or select a session to begin",
            _ => "Enter sends · Shift-Enter adds a line",
        }
    }

    fn queue(&mut self, command: CommandMessage) {
        self.command_error = None;
        if self.commands.send(command).is_err() {
            self.pending_command = false;
            self.state.mark_connection_lost("connection worker stopped");
        }
    }

    fn send_composer(&mut self, cx: &mut Context<Self>) {
        if !self.can_change_session() || self.state.active_session.is_none() {
            return;
        }
        let text = self.composer.read(cx).value().to_string();
        if text.trim().is_empty() {
            return;
        }
        self.pending_command = true;
        self.queue(CommandMessage::Send(text));
        cx.notify();
    }

    fn reconnect(&mut self, cx: &mut Context<Self>) {
        self.state.begin_reconnect();
        self.queue(CommandMessage::Reconnect);
        cx.notify();
    }

    fn decide(&mut self, request: String, approve: bool, cx: &mut Context<Self>) {
        if self.approval_pending
            || !self
                .state
                .approvals
                .iter()
                .any(|item| item.request_id == request)
        {
            return;
        }
        self.approval_pending = true;
        self.queue(if approve {
            CommandMessage::Approve(request)
        } else {
            CommandMessage::Deny(request)
        });
        cx.notify();
    }

    fn sync_approval(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let approval = self.state.approvals.first().cloned();
        let request = approval.as_ref().map(|item| item.request_id.clone());
        if self.dialog_request == request {
            return;
        }
        if self.dialog_request.take().is_some() {
            window.close_dialog(cx);
        }
        self.approval_pending = false;
        self.dialog_request = request;
        if let Some(Approval {
            request_id,
            tool_call,
        }) = approval
        {
            let view = cx.entity().downgrade();
            window.open_dialog(cx, move |dialog, _, cx| {
                let (pending, error) = view
                    .upgrade()
                    .map(|view| {
                        let view = view.read(cx);
                        (view.approval_pending, view.command_error.clone())
                    })
                    .unwrap_or_default();
                let approve_view = view.clone();
                let deny_view = view.clone();
                let approve_id = request_id.clone();
                let deny_id = request_id.clone();
                dialog
                    .title(format!("Allow {}?", tool_call.name))
                    .child(
                        div()
                            .id("approval-arguments")
                            .max_h(px(220.))
                            .overflow_y_scroll()
                            .child(
                                serde_json::to_string_pretty(&tool_call.arguments)
                                    .unwrap_or_default(),
                            ),
                    )
                    .child(if pending {
                        "Waiting for the server…"
                    } else {
                        "Enter approves · Esc denies"
                    })
                    .when_some(error, |dialog, error| {
                        dialog.child(Alert::error("approval-error", error))
                    })
                    .close_button(false)
                    .overlay_closable(false)
                    .button_props(
                        DialogButtonProps::default()
                            .ok_text("Approve")
                            .cancel_text("Deny"),
                    )
                    .on_ok(move |_, _, cx| {
                        let _ = approve_view
                            .update(cx, |view, cx| view.decide(approve_id.clone(), true, cx));
                        false // Authoritative status closes the modal after the server accepts it.
                    })
                    .on_cancel(move |_, _, cx| {
                        let _ = deny_view
                            .update(cx, |view, cx| view.decide(deny_id.clone(), false, cx));
                        false
                    })
            });
        }
    }

    fn control_key(&mut self, event: &KeyDownEvent, _: &mut Window, cx: &mut Context<Self>) {
        if self.state.approvals.is_empty()
            && self.state.streaming
            && (event.keystroke.key == "escape"
                || (event.keystroke.key == "c" && event.keystroke.modifiers.control))
        {
            self.queue(CommandMessage::Abort);
            cx.stop_propagation();
            cx.notify();
        }
    }

    fn render_row(&self, index: usize, cx: &App) -> gpui::AnyElement {
        let row = div()
            .debug_selector(|| "transcript-row".into())
            .w_full()
            .min_w_0()
            .px_6()
            .py_3();
        match &self.state.transcript[index] {
            TranscriptEntry::User(text) => row
                .child(
                    div()
                        .mb_2()
                        .text_size(px(12.))
                        .text_color(cx.theme().muted_foreground)
                        .child("you"),
                )
                .child(
                    div()
                        .pl_3()
                        .border_l_2()
                        .border_color(cx.theme().primary)
                        .child(text.clone()),
                )
                .into_any_element(),
            TranscriptEntry::Assistant(doc) => row
                .child(
                    div()
                        .mb_2()
                        .text_size(px(12.))
                        .text_color(cx.theme().muted_foreground)
                        .child("zeta"),
                )
                .when(doc.preview_truncated, |row| {
                    row.child("Showing the latest streamed text…")
                })
                .child(
                    TextView::markdown(format!("message-{index}"), doc.source.to_string())
                        .selectable(true),
                )
                .into_any_element(),
            entry @ TranscriptEntry::Tool { name, summary, .. } => row
                .py_1()
                .child(
                    div()
                        .px_3()
                        .py_2()
                        .bg(cx.theme().muted)
                        .text_size(px(12.))
                        .text_color(if entry.unsuccessful() {
                            cx.theme().danger
                        } else {
                            cx.theme().muted_foreground
                        })
                        .truncate()
                        .child(format!("{} {name}  {summary}", entry.tool_marker())),
                )
                .into_any_element(),
        }
    }
}

impl Render for ZetaView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let can_send = self.can_change_session() && self.state.active_session.is_some();
        let view = cx.entity();
        let transcript = MessageScroller::new(
            "transcript",
            self.transcript.clone(),
            move |index, _, cx| view.read(cx).render_row(index, cx),
        )
        .with_row_style(gpui::StyleRefinement::default().pb_0())
        .flex_1()
        .min_h_0()
        .min_w_0();
        let banner = match &self.state.connection {
            ConnectionState::Lost(error) => Some(
                div()
                    .v_flex()
                    .gap_2()
                    .p_3()
                    .child(
                        Alert::error("connection-lost", format!("Connection lost: {error}"))
                            .banner(),
                    )
                    .child(
                        Button::new("reconnect")
                            .debug_selector(|| "reconnect-button".into())
                            .label("Reconnect")
                            .on_click(cx.listener(|view, _, _, cx| view.reconnect(cx))),
                    ),
            ),
            ConnectionState::Reconnecting => Some(
                div()
                    .p_3()
                    .child(Alert::info("connecting", "Connecting to zeta…").banner()),
            ),
            ConnectionState::Connected => None,
        };
        let main = div()
            .v_flex()
            .flex_1()
            .min_w_0()
            .h_full()
            .children(banner)
            .when_some(self.command_error.clone(), |main, error| {
                main.child(Alert::error("command-error", error).banner())
            })
            .child(
                div()
                    .id("transcript-viewport")
                    .debug_selector(|| "transcript-viewport".into())
                    .v_flex()
                    .flex_1()
                    .min_h_0()
                    .min_w_0()
                    .overflow_hidden()
                    .when(self.state.transcript.is_empty(), |view| {
                        view.child(
                            div()
                                .p_6()
                                .text_color(cx.theme().muted_foreground)
                                .child("Start a conversation"),
                        )
                    })
                    .child(transcript),
            )
            .child(
                div()
                    .id("composer")
                    .debug_selector(|| "composer".into())
                    .v_flex()
                    .flex_shrink_0()
                    .gap_2()
                    .p_4()
                    .border_t_1()
                    .border_color(cx.theme().border)
                    // The kit emits PressEnter and propagates its action. Consume it
                    // here so native text input cannot insert a newline after submit.
                    .on_action(|_: &gpui_kit::component::input::Enter, _, _| {})
                    .child(
                        Textarea::new(&self.composer)
                            .h(px(96.))
                            .disabled(!can_send)
                            .aria_label("Message zeta"),
                    )
                    .child(
                        div()
                            .h_flex()
                            .justify_between()
                            .gap_3()
                            .child(
                                div()
                                    .text_size(px(12.))
                                    .text_color(cx.theme().muted_foreground)
                                    .child(self.composer_hint()),
                            )
                            .child(
                                Button::new("send")
                                    .primary()
                                    .label("Send")
                                    .disabled(!can_send)
                                    .h(px(40.))
                                    .on_click(cx.listener(|view, _, _, cx| view.send_composer(cx))),
                            ),
                    ),
            )
            .child(
                div()
                    .id("status-bar")
                    .h_flex()
                    .flex_shrink_0()
                    .gap_4()
                    .px_4()
                    .py_2()
                    .text_size(px(12.))
                    .text_color(cx.theme().muted_foreground)
                    .child(self.state.metrics.model_label().to_owned())
                    .child(format!("{} tokens", self.state.metrics.tokens_label()))
                    .child(format!("{} cache", self.state.metrics.cache_label())),
            );
        div()
            .size_full()
            .relative()
            .bg(cx.theme().background)
            .text_color(cx.theme().foreground)
            .font_family("JetBrains Mono")
            .text_size(px(14.))
            .on_key_down(cx.listener(Self::control_key))
            .child(
                div()
                    .h_flex()
                    .size_full()
                    .items_stretch()
                    .child(self.render_sidebar(cx))
                    .child(main),
            )
            .child(self.dialogs.clone())
    }
}

fn init(cx: &mut App) {
    cx.text_system()
        .add_fonts(
            [
                include_bytes!("../assets/fonts/JetBrainsMono-Regular.ttf").as_slice(),
                include_bytes!("../assets/fonts/JetBrainsMono-Medium.ttf").as_slice(),
                include_bytes!("../assets/fonts/JetBrainsMono-Bold.ttf").as_slice(),
                include_bytes!("../assets/fonts/JetBrainsMono-Italic.ttf").as_slice(),
            ]
            .into_iter()
            .map(Cow::Borrowed)
            .collect(),
        )
        .expect("register JetBrains Mono");
    gpui_kit::init(cx);
}

fn main() {
    let socket = env::args()
        .skip(1)
        .collect::<Vec<_>>()
        .windows(2)
        .find(|pair| pair[0] == "--socket")
        .map(|pair| PathBuf::from(&pair[1]));
    gpui_kit::application()
        .with_assets(gpui_kit::assets::Assets)
        .run(move |cx| {
            init(cx);
            let bounds = Bounds::centered(None, gpui::size(px(1100.), px(760.)), cx);
            cx.open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    ..Default::default()
                },
                move |window, cx| {
                    let view = cx.new(|cx| ZetaView::connect(window, cx, socket));
                    #[cfg(feature = "smoke-test")]
                    smoke::start(&view, window, cx);
                    cx.new(|cx| Root::new(view, window, cx))
                },
            )
            .expect("open zeta window");
            cx.activate(true);
        });
}

#[cfg(test)]
mod tests;
