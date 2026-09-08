use gpui::{
    div, prelude::*, px, App, Bounds, Context, FocusHandle, Focusable, KeyDownEvent, Render, Task,
    Window, WindowBounds, WindowOptions,
};
use gpui_platform::application;
use std::env;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use zeta_gui::client::{Approval, ClientError, ProtocolClient, ServerEvent, SessionMetadata};
use zeta_gui::state::{AppState, ConnectionState, TranscriptEntry};

const BG: u32 = 0x111210;
const PANEL: u32 = 0x181a17;
const LINE: u32 = 0x343832;
const TEXT: u32 = 0xd8ddd5;
const MUTED: u32 = 0x8a9287;
const SIGNAL: u32 = 0xe1a84b;

#[derive(Clone, Copy)]
struct Palette {
    bg: u32,
    text: u32,
}

impl Palette {
    fn for_window(window: &Window) -> Self {
        match window.appearance() {
            gpui::WindowAppearance::Dark | gpui::WindowAppearance::VibrantDark => {
                Self { bg: BG, text: TEXT }
            }
            gpui::WindowAppearance::Light | gpui::WindowAppearance::VibrantLight => Self {
                bg: 0xf7f8f5,
                text: 0x1b1e1a,
            },
        }
    }
}

enum CommandMessage {
    NewSession,
    Resume(String),
    Send(String),
    Approve(String),
    Deny(String),
    Abort,
    Reconnect,
}

enum WorkerMessage {
    Sessions(Vec<SessionMetadata>),
    Session(SessionMetadata),
    Connected,
    Event(ServerEvent),
    Lost(String),
}

struct ConnectionWorker {
    commands: Receiver<CommandMessage>,
    messages: Sender<WorkerMessage>,
    socket: Option<PathBuf>,
}

impl ConnectionWorker {
    fn run(self) {
        let mut server_process: Option<Child> = None;
        let mut client = None;
        loop {
            if client.is_none() {
                match connect(&self.socket, &mut server_process) {
                    Ok(mut connected) => match connected
                        .handshake()
                        .and_then(|_| connected.list_sessions())
                    {
                        Ok(sessions) => {
                            let _ = self.messages.send(WorkerMessage::Sessions(sessions));
                            let _ = self.messages.send(WorkerMessage::Connected);
                            client = Some(connected);
                        }
                        Err(error) => {
                            let _ = self.messages.send(WorkerMessage::Lost(error.to_string()));
                            wait_for_reconnect(&self.commands);
                        }
                    },
                    Err(error) => {
                        let _ = self.messages.send(WorkerMessage::Lost(error.to_string()));
                        wait_for_reconnect(&self.commands);
                    }
                }
                continue;
            }

            let mut disconnected = false;
            match self.commands.recv_timeout(Duration::from_millis(40)) {
                Ok(command) => {
                    if command_is_reconnect(&command) {
                        client = None;
                        continue;
                    }
                    let result = client.as_mut().map_or_else(
                        || Err(ClientError::UnexpectedResponse),
                        |active| handle_command(active, command, &self.messages, &self.commands),
                    );
                    if let Err(error) = result {
                        let _ = self.messages.send(WorkerMessage::Lost(error.to_string()));
                        disconnected = true;
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => return,
            }
            if disconnected {
                client = None;
            } else if let Some(event) = client.as_mut().and_then(ProtocolClient::try_event) {
                let _ = self.messages.send(WorkerMessage::Event(event));
            }
        }
    }
}

fn command_is_reconnect(command: &CommandMessage) -> bool {
    matches!(command, CommandMessage::Reconnect)
}

fn wait_for_reconnect(commands: &Receiver<CommandMessage>) {
    while let Ok(command) = commands.recv_timeout(Duration::from_millis(250)) {
        if command_is_reconnect(&command) {
            return;
        }
    }
}

fn handle_command(
    client: &mut ProtocolClient,
    command: CommandMessage,
    messages: &Sender<WorkerMessage>,
    commands: &Receiver<CommandMessage>,
) -> Result<(), ClientError> {
    match command {
        CommandMessage::NewSession => {
            let session = client.new_session(None, None)?;
            let _ = messages.send(WorkerMessage::Session(session));
        }
        CommandMessage::Resume(id) => {
            let session = client.resume(&id)?;
            let _ = messages.send(WorkerMessage::Session(session));
        }
        CommandMessage::Send(text) => {
            client.send(&text)?;
            client.set_read_timeout(Some(Duration::from_millis(100)))?;
            drain_turn(client, messages, commands)?;
            client.set_read_timeout(Some(Duration::from_secs(5)))?;
        }
        CommandMessage::Approve(id) => {
            client.approve(&id)?;
        }
        CommandMessage::Deny(id) => {
            client.deny(&id)?;
        }
        CommandMessage::Abort => {
            client.abort()?;
        }
        CommandMessage::Reconnect => {}
    }
    Ok(())
}

fn drain_turn(
    client: &mut ProtocolClient,
    messages: &Sender<WorkerMessage>,
    commands: &Receiver<CommandMessage>,
) -> Result<(), ClientError> {
    loop {
        match client.next_event() {
            Ok(event) => {
                let done = matches!(
                    event,
                    ServerEvent::TurnEnd { .. } | ServerEvent::TurnAborted { .. }
                );
                let approval_wait = matches!(event, ServerEvent::ApprovalRequest { .. });
                let _ = messages.send(WorkerMessage::Event(event));
                if done {
                    return Ok(());
                }
                if approval_wait {
                    process_turn_commands(client, commands)?;
                }
            }
            Err(ClientError::Io(error)) => {
                if error.kind() == std::io::ErrorKind::TimedOut {
                    process_available_turn_command(client, commands)?;
                } else {
                    return Err(ClientError::Io(error));
                }
            }
            Err(error) => return Err(error),
        }
    }
}

fn process_available_turn_command(
    client: &mut ProtocolClient,
    commands: &Receiver<CommandMessage>,
) -> Result<(), ClientError> {
    if let Ok(command) = commands.try_recv() {
        process_command_during_turn(client, command)?;
    }
    Ok(())
}

fn process_turn_commands(
    client: &mut ProtocolClient,
    commands: &Receiver<CommandMessage>,
) -> Result<(), ClientError> {
    loop {
        match commands.recv() {
            Ok(command) => {
                if matches!(
                    command,
                    CommandMessage::Approve(_) | CommandMessage::Deny(_) | CommandMessage::Abort
                ) {
                    return process_command_during_turn(client, command);
                }
            }
            Err(_) => return Err(ClientError::UnexpectedResponse),
        }
    }
}

fn process_command_during_turn(
    client: &mut ProtocolClient,
    command: CommandMessage,
) -> Result<(), ClientError> {
    match command {
        CommandMessage::Approve(id) => {
            client.approve(&id)?;
        }
        CommandMessage::Deny(id) => {
            client.deny(&id)?;
        }
        CommandMessage::Abort => {
            client.abort()?;
        }
        CommandMessage::NewSession
        | CommandMessage::Resume(_)
        | CommandMessage::Send(_)
        | CommandMessage::Reconnect => {}
    }
    Ok(())
}

fn connect(
    socket: &Option<PathBuf>,
    process: &mut Option<Child>,
) -> Result<ProtocolClient, ClientError> {
    let path = socket.clone().unwrap_or_else(default_socket);
    if !path.exists() {
        let mut command = Command::new(zeta_binary());
        command.arg("serve").arg("--socket").arg(&path);
        *process = Some(command.spawn().map_err(ClientError::Io)?);
        for _ in 0..50 {
            if path.exists() {
                break;
            }
            thread::sleep(Duration::from_millis(100));
        }
    }
    ProtocolClient::connect_socket(path)
}

fn zeta_binary() -> String {
    env::var("ZETA_BIN").unwrap_or_else(|_| "zeta".to_owned())
}

fn default_socket() -> PathBuf {
    if let Some(home) = env::var_os("ZETA_HOME") {
        return PathBuf::from(home).join("run/serve.sock");
    }
    PathBuf::from(env::var_os("HOME").unwrap_or_default()).join(".zeta/run/serve.sock")
}

struct ZetaView {
    state: AppState,
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
        let messages = Arc::new(Mutex::new(message_rx));
        let worker_messages = message_tx;
        thread::spawn(move || {
            ConnectionWorker {
                commands: command_rx,
                messages: worker_messages,
                socket,
            }
            .run()
        });
        let messages_for_poll = Arc::clone(&messages);
        let poll_task = cx.spawn_in(window, async move |this, cx| loop {
            cx.background_executor()
                .timer(Duration::from_millis(50))
                .await;
            let Ok(receiver) = messages_for_poll.lock() else {
                return;
            };
            let updates = receiver.try_iter().collect::<Vec<_>>();
            drop(receiver);
            if updates.is_empty() {
                continue;
            }
            if this
                .update_in(cx, |view, _window, cx| {
                    for update in updates {
                        view.apply_worker_message(update);
                    }
                    cx.notify();
                })
                .is_err()
            {
                return;
            }
        });
        Self {
            state: AppState::default(),
            commands: command_tx,
            focus_handle: cx.focus_handle(),
            _poll_task: poll_task,
        }
    }

    fn apply_worker_message(&mut self, message: WorkerMessage) {
        match message {
            WorkerMessage::Sessions(sessions) => self.state.sessions = sessions,
            WorkerMessage::Session(session) => {
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
            }
            WorkerMessage::Connected => self.state.connection = ConnectionState::Connected,
            WorkerMessage::Event(event) => self.state.apply(event),
            WorkerMessage::Lost(error) => self.state.mark_connection_lost(error),
        }
    }

    fn new_session(&mut self, _: &gpui::ClickEvent, _: &mut Window, _: &mut Context<Self>) {
        let _ = self.commands.send(CommandMessage::NewSession);
        self.state.transcript.clear();
        self.state.active_session = None;
    }

    fn resume(&mut self, id: String, _: &gpui::ClickEvent, _: &mut Window, _: &mut Context<Self>) {
        let _ = self.commands.send(CommandMessage::Resume(id.clone()));
        self.state.active_session = Some(id);
        self.state.transcript.clear();
    }

    fn composer_key(&mut self, event: &KeyDownEvent, window: &mut Window, cx: &mut Context<Self>) {
        if self.state.approval.is_some() {
            return;
        }
        let key = event.keystroke.key.as_str();
        if key == "escape" || (key == "c" && event.keystroke.modifiers.control) {
            let _ = self.commands.send(CommandMessage::Abort);
            self.state.streaming = false;
        } else if key == "enter" {
            if event.keystroke.modifiers.shift {
                self.state.composer.push('\n');
            } else {
                self.send_composer();
            }
        } else if key == "backspace" {
            self.state.composer.pop();
        } else if !event.keystroke.modifiers.modified() {
            if let Some(character) = event.keystroke.key_char.as_deref().or(Some(key)) {
                self.state.composer.push_str(character);
            }
        }
        window.focus(&self.focus_handle, cx);
        cx.notify();
    }

    fn approval_key(&mut self, event: &KeyDownEvent, _: &mut Window, cx: &mut Context<Self>) {
        let Some(approval) = self.state.approval.take() else {
            return;
        };
        match event.keystroke.key.as_str() {
            "enter" => {
                let _ = self
                    .commands
                    .send(CommandMessage::Approve(approval.request_id));
            }
            "escape" => {
                let _ = self
                    .commands
                    .send(CommandMessage::Deny(approval.request_id));
            }
            _ => self.state.approval = Some(approval),
        }
        cx.notify();
    }

    fn send_composer(&mut self) {
        let text = std::mem::take(&mut self.state.composer);
        if text.trim().is_empty() {
            return;
        }
        self.state
            .transcript
            .push(TranscriptEntry::User(text.clone()));
        let _ = self.commands.send(CommandMessage::Send(text));
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
        let _ = self.commands.send(command);
    }

    fn reconnect(&mut self, _: &gpui::ClickEvent, _: &mut Window, _: &mut Context<Self>) {
        self.state.begin_reconnect();
        let _ = self.commands.send(CommandMessage::Reconnect);
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
        let mut transcript = div()
            .flex_1()
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
            transcript = transcript.child(row);
        }
        transcript
    }

    fn render_composer(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let placeholder = if self.state.composer.is_empty() {
            "write a message...".to_owned()
        } else {
            self.state.composer.clone()
        };
        div()
            .w_full()
            .border_t_1()
            .border_color(gpui::rgb(LINE))
            .p_4()
            .child(
                div()
                    .id("composer")
                    .min_h(px(72.))
                    .w_full()
                    .p_3()
                    .border_1()
                    .border_color(gpui::rgb(LINE))
                    .text_color(gpui::rgb(if self.state.composer.is_empty() {
                        MUTED
                    } else {
                        TEXT
                    }))
                    .focusable()
                    .track_focus(&self.focus_handle)
                    .on_key_down(cx.listener(Self::composer_key))
                    .child(placeholder),
            )
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
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let palette = Palette::for_window(window);
        let mut root = div()
            .size_full()
            .flex()
            .bg(gpui::rgb(palette.bg))
            .text_size(px(14.))
            .text_color(gpui::rgb(palette.text))
            .on_key_down(cx.listener(Self::approval_key))
            .child(self.render_sidebar(cx))
            .child(
                div()
                    .flex_1()
                    .h_full()
                    .flex()
                    .flex_col()
                    .child(self.render_transcript())
                    .child(self.render_composer(cx)),
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
        if let Some(approval) = self.state.approval.clone() {
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
