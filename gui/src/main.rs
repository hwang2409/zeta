extern crate gpui_kit as gpui;

mod polish;
mod session_management;
mod sidebar;
#[cfg(feature = "smoke-test")]
mod smoke;
mod theme;

use gpui::{
    div, ease_in_out, prelude::*, px, Animation, AnimationExt, App, Bounds, Context, Entity,
    Focusable, KeyDownEvent, Render, Task, Window, WindowBounds, WindowOptions,
};
use gpui_kit::component::{
    alert::Alert,
    button::{Button, ButtonCustomVariant, ButtonVariants},
    dialog::DialogButtonProps,
    input::{InputEvent, Textarea, TextareaState},
    message_scroller::{MessageScroller, MessageScrollerState},
    text::TextView,
    ActiveTheme, Disableable, Icon, IconName, Root, Selectable, StyledExt, WindowExt,
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
    login::{LoginProgress, LoginProvider},
    row_text::{
        self, AssistantRowText, ErrorRowText, RowText, ThinkingRowText, ToolRowText, UserRowText,
    },
    session::{ImageAttachment, SessionSettings, APPROVAL_MODES},
    state::{AppState, ConnectionState, TranscriptEdit, TranscriptEntry},
    worker::{CommandMessage, ConnectionWorker, WorkerMessage},
};

struct DialogLayer;

impl Render for DialogLayer {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div().children(Root::render_dialog_layer(window, cx))
    }
}

/// Local queued state for a user turn that has been submitted but not yet
/// echoed by the server. Rendered as a dashed strip between the transcript
/// and the composer so users see the message is on its way, and the strip
/// promotes to a solid user turn on `Sent` or flips its rail to danger on
/// `Rejected`.
#[derive(Debug, Clone, PartialEq)]
struct PendingUserTurn {
    text: String,
    failed: bool,
}

/// Ordered virtual-list operations that mirror the ordered edit list from
/// `AppState::apply`. Extracted as a pure function so a mutation that
/// collapses the compound-edit case to a tail splice (or drops all but one
/// edit) is caught by a unit test — the visual sync path only observes the
/// bug when off-viewport rows drift or the item count desyncs, which is
/// impractical to reproduce here.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ScrollSync {
    Reset(usize),
    Splice(std::ops::Range<usize>, usize),
    Remeasure(usize),
}

fn scroll_sync(new_count: usize, edits: &[TranscriptEdit], replace: bool) -> Vec<ScrollSync> {
    // A session switch or history swap replaces the transcript wholesale.
    // The edit list from any concurrent apply cannot describe the new state,
    // so reset is authoritative — the view drops its cache and rebuilds from
    // the new item count.
    if replace {
        return vec![ScrollSync::Reset(new_count)];
    }
    // One-to-one translation of the ordered edits — each edit describes an
    // operation against the transcript state the view currently holds, so
    // applying them in the same order keeps the virtual-list metadata cache
    // aligned to the transcript row-for-row, whether the reconcile is a
    // single append, a middle removal, or a compound append + remove +
    // remeasure.
    edits
        .iter()
        .map(|edit| match *edit {
            TranscriptEdit::Insert(index) => ScrollSync::Splice(index..index, 1),
            TranscriptEdit::Remove(index) => ScrollSync::Splice(index..index + 1, 0),
            TranscriptEdit::Remeasure(index) => ScrollSync::Remeasure(index),
        })
        .collect()
}

struct ZetaView {
    state: AppState,
    dialogs: Entity<DialogLayer>,
    composer: Entity<TextareaState>,
    transcript: Entity<MessageScrollerState>,
    sidebar_scroll: gpui_kit::component::VirtualListScrollHandle,
    model_scroll: gpui::ScrollHandle,
    pending_command: bool,
    pending_user_turn: Option<PendingUserTurn>,
    command_error: Option<String>,
    dialog_request: Option<String>,
    approval_pending: bool,
    settings_open: bool,
    session_management: bool,
    session_edit: Option<session_management::SessionEdit>,
    session_edit_focus: gpui::FocusHandle,
    settings_focus: gpui::FocusHandle,
    // One persistent focus handle per sidebar row id — a session id or a
    // branch id. Populated lazily in the sidebar render and reused across
    // paints so tab focus survives redraws and tests can look a row's
    // handle up by the same key the renderer uses.
    pub(crate) sidebar_row_focus:
        std::cell::RefCell<std::collections::HashMap<String, gpui::FocusHandle>>,
    login_providers: Vec<LoginProvider>,
    settings_error: Option<String>,
    composer_images: Vec<ImageAttachment>,
    composer_image_error: Option<String>,
    composer_empty_hint: bool,
    sent_images: std::collections::BTreeMap<(usize, usize), std::sync::Arc<gpui::Image>>,
    commands: Sender<CommandMessage>,
    _poll_task: Option<Task<()>>,
}

impl ZetaView {
    fn new(window: &mut Window, cx: &mut Context<Self>, commands: Sender<CommandMessage>) -> Self {
        theme::apply(cx);
        let composer = cx.new(|cx| {
            TextareaState::new(window, cx)
                .placeholder("Message zeta")
                .submit_on_enter(true)
        });
        cx.subscribe_in(&composer, window, |view, _, event: &InputEvent, _, cx| {
            if matches!(event, InputEvent::Change) {
                view.composer_empty_hint = false;
                cx.notify();
            }
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
            model_scroll: gpui::ScrollHandle::new(),
            pending_command: false,
            pending_user_turn: None,
            command_error: None,
            dialog_request: None,
            approval_pending: false,
            settings_open: false,
            session_management: false,
            session_edit: None,
            session_edit_focus: cx.focus_handle(),
            settings_focus: cx.focus_handle(),
            sidebar_row_focus: std::cell::RefCell::new(std::collections::HashMap::new()),
            login_providers: Vec::new(),
            settings_error: None,
            composer_images: Vec::new(),
            composer_image_error: None,
            composer_empty_hint: false,
            sent_images: Default::default(),
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
        let login_changed = matches!(
            &message,
            WorkerMessage::Extensions(_)
                | WorkerMessage::LoginProviders(_)
                | WorkerMessage::Login(..)
                | WorkerMessage::Lost(_)
        );
        let previous_session = self.state.active_session.clone();
        let mut edits: Vec<TranscriptEdit> = Vec::new();
        let mut replace = false;
        match message {
            WorkerMessage::LoginProviders(providers) => self.login_providers = providers,
            WorkerMessage::Login(provider, progress) => {
                if let Some(row) = self
                    .login_providers
                    .iter_mut()
                    .find(|row| row.provider == provider)
                {
                    if let Some(url) = row.update(progress) {
                        cx.open_url(&url);
                    }
                }
            }
            WorkerMessage::Extensions(available) => {
                self.state.session_view.available = available;
                self.login_providers.clear();
            }
            WorkerMessage::Tree(tree) => self.state.session_view.branches = tree.branches,
            WorkerMessage::History(history, reset) => {
                self.state.apply_history(history, reset);
                replace = reset;
                self.pending_command = false;
            }
            WorkerMessage::Settings(settings, catalog) => {
                self.pending_command = false;
                self.settings_error = None;
                self.state
                    .session_view
                    .current_model
                    .clone_from(&settings.model);
                self.state.session_view.models = catalog.models;
                self.state.session_view.model_providers = catalog.providers;
                self.state.session_view.selected_model = self
                    .state
                    .session_view
                    .models
                    .iter()
                    .position(|model| model == &settings.model)
                    .unwrap_or(0);
                self.state.session_view.selected_mode = APPROVAL_MODES
                    .iter()
                    .position(|mode| *mode == settings.approval_mode)
                    .unwrap_or(0);
                self.settings_open = true;
                window.focus(&self.settings_focus, cx);
            }
            WorkerMessage::SettingsApplied(settings) => {
                self.pending_command = false;
                self.state.session_view.current_model = settings.model;
                self.close_settings(window, cx);
            }
            WorkerMessage::ImagesSent(text, images) => {
                self.pending_command = false;
                // Both Sent and ImagesSent land the real user turn in the
                // transcript; the queued strip must clear here too or an
                // image-only send leaves a phantom dashed row beside it.
                self.pending_user_turn = None;
                self.state.streaming = true;
                let index = self.state.transcript.len();
                self.state.transcript.push(TranscriptEntry::User(text));
                edits.push(TranscriptEdit::Insert(index));
                self.state.session_view.attachments.insert(
                    index,
                    images
                        .iter()
                        .map(|item| (item.name.clone(), item.size))
                        .collect(),
                );
                for (attachment_index, image) in images.iter().enumerate() {
                    if let Some(image) = polish::image_source(image) {
                        self.sent_images.insert((index, attachment_index), image);
                        if self.sent_images.len() > polish::SENT_IMAGE_LIMIT {
                            if let Some((_, image)) = self.sent_images.pop_first() {
                                image.remove_asset(cx);
                            }
                        }
                    }
                }
                self.composer_images.clear();
                self.composer_image_error = None;
                self.composer
                    .update(cx, |input, cx| input.set_value("", window, cx));
            }
            WorkerMessage::Sent(text) => {
                self.pending_command = false;
                self.pending_user_turn = None;
                self.state.streaming = true;
                let index = self.state.transcript.len();
                self.state.transcript.push(TranscriptEntry::User(text));
                edits.push(TranscriptEdit::Insert(index));
                self.composer
                    .update(cx, |input, cx| input.set_value("", window, cx));
            }
            WorkerMessage::Sessions(list) => {
                self.state.sessions = list.sessions;
                self.state.sessions_truncated = list.truncated;
            }
            WorkerMessage::SessionManagement(available) => self.session_management = available,
            WorkerMessage::Renamed(session) => {
                self.pending_command = false;
                if let Some(row) = self
                    .state
                    .sessions
                    .iter_mut()
                    .find(|row| row.session_id == session.session_id)
                {
                    row.name = session.name;
                    row.updated_at = session.updated_at;
                }
                self.close_session_edit(window, cx);
            }
            WorkerMessage::Deleted(id) => {
                self.pending_command = false;
                self.state.sessions.retain(|row| row.session_id != id);
                self.state.saved_transcripts.remove(&id);
                self.close_session_edit(window, cx);
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
                if let Some(pending) = &mut self.pending_user_turn {
                    pending.failed = true;
                }
                if self.settings_open {
                    self.settings_error = Some(error);
                } else {
                    self.command_error = Some(error);
                }
            }
            WorkerMessage::Connected => self.state.connection = ConnectionState::Connected,
            WorkerMessage::Event(event) => edits = self.state.apply(event),
            WorkerMessage::Lost(error) => {
                if self.session_edit.is_some() {
                    self.command_error = Some(error.clone());
                }
                self.pending_command = false;
                self.approval_pending = false;
                if let Some(pending) = &mut self.pending_user_turn {
                    pending.failed = true;
                }
                for row in &mut self.login_providers {
                    if row.progress.busy() {
                        row.progress = LoginProgress::failed(
                            "Connection lost during sign-in. Reconnect and try again.".into(),
                        );
                    }
                }
                self.state.mark_connection_lost(error);
            }
        }
        if self.state.active_session != previous_session {
            self.composer_images.clear();
            self.composer_image_error = None;
            self.composer_empty_hint = false;
            if self.settings_open {
                self.close_settings(window, cx);
            }
            self.settings_error = None;
            replace = true;
        }
        if replace {
            for image in std::mem::take(&mut self.sent_images).into_values() {
                image.remove_asset(cx);
            }
        }
        let count = self.state.transcript.len();
        self.transcript.update(cx, |scroll, cx| {
            for op in scroll_sync(count, &edits, replace) {
                match op {
                    ScrollSync::Reset(new_count) => scroll.reset(new_count, cx),
                    ScrollSync::Splice(range, insert) => {
                        scroll.splice(range, insert, cx);
                    }
                    ScrollSync::Remeasure(index) => {
                        scroll.remeasure_items(index..index + 1, cx);
                    }
                }
            }
        });
        if login_changed {
            self.remeasure_login_rows(cx);
        }
        self.sync_approval(window, cx);
        cx.notify();
    }

    fn can_change_session(&self) -> bool {
        self.state.connection == ConnectionState::Connected
            && !self.state.streaming
            && self.state.approvals.is_empty()
            && !self.pending_command
            && self.session_edit.is_none()
    }

    fn composer_hint(&self) -> &'static str {
        match &self.state.connection {
            ConnectionState::Lost(_) => "Reconnect to send a message",
            ConnectionState::Reconnecting => "Connecting to zeta…",
            _ if !self.state.approvals.is_empty() => "Approve or deny the tool request to continue",
            _ if self.pending_command => "Waiting for the server…",
            _ if self.state.streaming && self.state.thinking => {
                "zeta is thinking… · Esc stops the turn"
            }
            _ if self.state.streaming => "Responding… · Esc stops the turn",
            _ if self.state.active_session.is_none() => "Create or select a session to begin",
            _ if self.composer_empty_hint => "Type a message or attach an image to send",
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
        let has_images = !self.composer_images.is_empty();
        if text.trim().is_empty() && !has_images {
            self.composer_empty_hint = true;
            cx.notify();
            return;
        }
        self.composer_empty_hint = false;
        self.pending_command = true;
        // Record the queued text so a dashed strip renders under the composer
        // while the server has not yet echoed the turn. `WorkerMessage::Sent`
        // clears it; `Rejected` flips `failed` for the danger rail.
        self.pending_user_turn = Some(PendingUserTurn {
            text: text.clone(),
            failed: false,
        });
        if has_images {
            self.queue(CommandMessage::SendImages(
                text,
                self.composer_images.clone(),
            ));
        } else {
            self.queue(CommandMessage::Send(text));
        }
        cx.notify();
    }

    fn remeasure_login_rows(&self, cx: &mut Context<Self>) {
        self.transcript.update(cx, |scroll, cx| {
            for (index, row) in self.state.transcript.iter().enumerate() {
                if matches!(
                    row,
                    TranscriptEntry::Error {
                        login_provider: Some(_),
                        ..
                    }
                ) {
                    scroll.remeasure_items(index..index + 1, cx);
                }
            }
        });
    }

    fn start_login(&mut self, provider: &str, cx: &mut Context<Self>) {
        if self.state.connection != ConnectionState::Connected {
            return;
        }
        if let Some(row) = self
            .login_providers
            .iter_mut()
            .find(|row| row.provider == provider)
        {
            if row.progress.busy() {
                return;
            }
            row.progress = LoginProgress::Starting;
            self.queue(CommandMessage::LoginStart(provider.to_owned()));
            self.remeasure_login_rows(cx);
            cx.notify();
        }
    }

    fn cancel_login(&mut self, provider: &str, cx: &mut Context<Self>) {
        if let Some(row) = self
            .login_providers
            .iter_mut()
            .find(|row| row.provider == provider)
        {
            if !row.progress.busy() || row.progress == LoginProgress::Cancelling {
                return;
            }
            row.progress = LoginProgress::Cancelling;
            self.queue(CommandMessage::LoginCancel(provider.to_owned()));
            self.remeasure_login_rows(cx);
            cx.notify();
        }
    }

    fn render_login_row(
        &self,
        provider: &LoginProvider,
        prefix: &str,
        view: gpui::WeakEntity<Self>,
        cx: &App,
    ) -> gpui::AnyElement {
        let id = format!("{prefix}-{}", provider.provider);
        let name = provider.provider.clone();
        let cancel = name.clone();
        let cancel_view = view.clone();
        div()
            .id(id.clone())
            .v_flex()
            .gap_2()
            .p_2()
            .child(
                div()
                    .h_flex()
                    .gap_3()
                    .items_center()
                    .justify_between()
                    .child(provider.label().to_owned())
                    .child(
                        Button::new(format!("{id}-start"))
                            .debug_selector({
                                let selector = format!("{id}-start");
                                move || selector.clone()
                            })
                            .h(px(40.))
                            .label(format!("Log in with {}", provider.label()))
                            .disabled(
                                provider.progress.busy()
                                    || self.state.connection != ConnectionState::Connected,
                            )
                            .on_click(move |_, _, cx| {
                                let _ = view.update(cx, |view, cx| view.start_login(&name, cx));
                            }),
                    )
                    .when(provider.progress.busy(), |row| {
                        row.child(
                            Button::new(format!("{id}-cancel"))
                                .debug_selector({
                                    let selector = format!("{id}-cancel");
                                    move || selector.clone()
                                })
                                .h(px(40.))
                                .label("Cancel")
                                .disabled(provider.progress == LoginProgress::Cancelling)
                                .on_click(move |_, _, cx| {
                                    let _ = cancel_view
                                        .update(cx, |view, cx| view.cancel_login(&cancel, cx));
                                }),
                        )
                    }),
            )
            .child(
                div()
                    .text_size(px(12.))
                    .whitespace_normal()
                    .text_color(cx.theme().muted_foreground)
                    .child(provider.status()),
            )
            .when_some(
                match &provider.progress {
                    LoginProgress::Failed { error } => Some(error.message.clone()),
                    _ => None,
                },
                |row, error| row.child(Alert::error(format!("{id}-error"), error)),
            )
            .into_any_element()
    }

    fn new_session(&mut self, cx: &mut Context<Self>) {
        if self.can_change_session() && !self.settings_open {
            self.pending_command = true;
            self.queue(CommandMessage::NewSession);
            cx.notify();
        }
    }

    fn open_settings(&mut self, cx: &mut Context<Self>) {
        if !self.can_change_session()
            || !self.state.session_view.available
            || self.state.active_session.is_none()
            || self.settings_open
        {
            return;
        }
        self.pending_command = true;
        self.settings_error = None;
        self.queue(CommandMessage::LoadSettings);
        cx.notify();
    }

    fn close_settings(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.settings_open = false;
        self.settings_error = None;
        window.focus(&self.composer.focus_handle(cx), cx);
        cx.notify();
    }

    fn apply_settings(&mut self, cx: &mut Context<Self>) {
        if self.pending_command {
            return;
        }
        let view = &self.state.session_view;
        let Some(model) = view.models.get(view.selected_model).cloned() else {
            return;
        };
        let settings = SessionSettings {
            model,
            approval_mode: view.approval_mode().to_owned(),
        };
        self.pending_command = true;
        self.settings_error = None;
        self.queue(CommandMessage::SetSettings(settings));
        cx.notify();
    }

    fn select_settings_model(&mut self, index: usize, cx: &mut Context<Self>) {
        if index < self.state.session_view.models.len() {
            self.state.session_view.selected_model = index;
            cx.notify();
        }
    }

    fn move_settings_model(&mut self, delta: isize, cx: &mut Context<Self>) {
        let count = self.state.session_view.models.len();
        if count == 0 {
            return;
        }
        let current = self.state.session_view.selected_model as isize;
        let next = (current + delta).rem_euclid(count as isize) as usize;
        self.select_settings_model(next, cx);
    }

    fn fork_message(&mut self, id: String, cx: &mut Context<Self>) {
        if !self.can_change_session() || !self.state.session_view.available {
            return;
        }
        self.pending_command = true;
        self.queue(CommandMessage::ForkMessage(id));
        cx.notify();
    }

    fn switch_branch(&mut self, id: String, cx: &mut Context<Self>) {
        if !self.can_change_session() || !self.state.session_view.available {
            return;
        }
        self.pending_command = true;
        self.queue(CommandMessage::SwitchBranch(id));
        cx.notify();
    }

    fn add_attached_images(
        &mut self,
        images: Result<Vec<ImageAttachment>, String>,
        cx: &mut Context<Self>,
    ) -> bool {
        if !self.can_change_session() || self.state.active_session.is_none() || self.settings_open {
            return false;
        }
        let previous_count = self.composer_images.len();
        match images {
            Ok(mut new_images) => {
                let combined = self.composer_images.len() + new_images.len();
                let total_bytes: usize = self
                    .composer_images
                    .iter()
                    .chain(new_images.iter())
                    .map(|image| image.size)
                    .sum();
                if combined > 4 || total_bytes > zeta_gui::session::MAX_IMAGE_BYTES {
                    self.composer_image_error =
                        Some("attach at most 4 images, totaling 512 KiB".into());
                } else {
                    self.composer_images.append(&mut new_images);
                    self.composer_image_error = None;
                    self.composer_empty_hint = false;
                }
            }
            Err(error) => self.composer_image_error = Some(error),
        }
        cx.notify();
        self.composer_images.len() > previous_count
    }

    fn remove_attached_image(&mut self, index: usize, cx: &mut Context<Self>) {
        if index < self.composer_images.len() {
            self.composer_images.remove(index);
            self.composer_image_error = None;
            cx.notify();
        }
    }

    fn attach_from_files(&mut self, _: &mut Window, cx: &mut Context<Self>) {
        if !self.can_change_session() || self.state.active_session.is_none() {
            return;
        }
        let session = self.state.active_session.clone();
        let paths = cx.prompt_for_paths(gpui::PathPromptOptions {
            files: true,
            directories: false,
            multiple: true,
            prompt: None,
        });
        cx.spawn(async move |view, cx| {
            let selection = match paths.await {
                Ok(Ok(Some(paths))) => paths,
                _ => return,
            };
            let images: Result<Vec<ImageAttachment>, String> = selection
                .iter()
                .map(|path| ImageAttachment::from_path(path))
                .collect();
            let _ = view.update(cx, |view, cx| {
                if view.state.active_session == session {
                    view.add_attached_images(images, cx);
                }
            });
        })
        .detach();
    }

    fn attach_from_clipboard(&mut self, cx: &mut Context<Self>) -> bool {
        let Some(item) = cx.read_from_clipboard() else {
            return false;
        };
        for entry in item.entries() {
            match entry {
                gpui::ClipboardEntry::Image(image) => {
                    let name = format!("pasted-image.{}", image.format.extension());
                    return self.add_attached_images(
                        ImageAttachment::from_bytes(name, image.bytes()).map(|image| vec![image]),
                        cx,
                    );
                }
                gpui::ClipboardEntry::ExternalPaths(paths) => {
                    let images: Result<Vec<ImageAttachment>, String> = paths
                        .paths()
                        .iter()
                        .map(|path| ImageAttachment::from_path(path))
                        .collect();
                    return self.add_attached_images(images, cx);
                }
                gpui::ClipboardEntry::String(_) => {}
            }
        }
        false
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
                    .when_some(polish::approval_summary(&tool_call), |dialog, summary| {
                        dialog.child(
                            div()
                                .debug_selector(|| "approval-summary".into())
                                .truncate()
                                .child(summary),
                        )
                    })
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

    fn paste_image(
        &mut self,
        _: &gpui_kit::component::input::Paste,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        // GPUI resolves key bindings before key-down handlers. Capture Paste
        // itself so image acceptance precedes the textarea's text paste.
        if self.session_edit.is_some() {
            cx.propagate();
        } else if self.settings_open {
            cx.stop_propagation();
        } else if self.state.approvals.is_empty() && self.attach_from_clipboard(cx) {
            cx.stop_propagation();
            cx.notify();
        } else {
            cx.propagate();
        }
    }

    fn settings_key(&mut self, event: &KeyDownEvent, window: &mut Window, cx: &mut Context<Self>) {
        if self.session_edit.is_some() {
            match event.keystroke.key.as_str() {
                "escape" => self.close_session_edit(window, cx),
                "enter" => self.commit_session_edit(cx),
                _ => return,
            }
            window.prevent_default();
            cx.stop_propagation();
            return;
        }
        if !self.settings_open {
            return;
        }
        if !event.keystroke.modifiers.modified() {
            match event.keystroke.key.as_str() {
                "escape" => self.close_settings(window, cx),
                "up" => self.move_settings_model(-1, cx),
                "down" => self.move_settings_model(1, cx),
                "enter" => self.apply_settings(cx),
                _ => {}
            }
        }
        window.prevent_default();
        cx.stop_propagation();
        cx.notify();
    }

    fn render_settings_overlay(
        &self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let view = &self.state.session_view;
        let providers = &view.model_providers;
        let group_of = |model: &str| providers.get(model).cloned().unwrap_or_default();
        let mut current_group: Option<String> = None;
        let selected = view.selected_model;
        // The scrollable container's ScrollHandle indexes ALL of its children,
        // including group headings. Precompute each model row's child index so
        // scroll_to_item lands on the selected model regardless of grouping.
        let mut model_child_indices = Vec::with_capacity(view.models.len());
        {
            let mut child_index = 0;
            let mut group: Option<String> = None;
            for model in &view.models {
                let g = group_of(model);
                if group.as_deref() != Some(g.as_str()) && !g.is_empty() {
                    group = Some(g);
                    child_index += 1;
                }
                model_child_indices.push(child_index);
                child_index += 1;
            }
        }
        if let Some(child_ix) = model_child_indices.get(selected).copied() {
            self.model_scroll.scroll_to_item(child_ix);
        }
        let mut list = div()
            .id("model-list")
            .debug_selector(|| "model-list".into())
            .v_flex()
            .flex_1()
            .min_h_0()
            .max_h(px(280.))
            .overflow_y_scroll()
            .track_scroll(&self.model_scroll);
        for (index, model) in view.models.iter().enumerate() {
            let group = group_of(model);
            if current_group.as_deref() != Some(group.as_str()) && !group.is_empty() {
                current_group = Some(group.clone());
                list = list.child(
                    div()
                        .px_3()
                        .pt_2()
                        .pb_1()
                        .text_color(cx.theme().muted_foreground)
                        .child(group),
                );
            }
            list = list.child(
                Button::new(("model", index))
                    .debug_selector(move || format!("model-row-{index}"))
                    .ghost()
                    .selected(index == selected)
                    .w_full()
                    .h(px(32.))
                    .child(
                        div()
                            .h_flex()
                            .w_full()
                            .items_center()
                            .justify_between()
                            .gap_2()
                            .child(div().flex_1().min_w_0().truncate().child(model.clone()))
                            .when(model == &view.current_model, |row| {
                                row.child(
                                    div()
                                        .text_color(cx.theme().muted_foreground)
                                        .child("current"),
                                )
                            }),
                    )
                    .on_click(
                        cx.listener(move |view, _, _, cx| view.select_settings_model(index, cx)),
                    ),
            );
        }
        let mode_row = div()
            .h_flex()
            .gap_2()
            .children(APPROVAL_MODES.iter().enumerate().map(|(index, mode)| {
                Button::new(("mode", index))
                    .debug_selector(move || format!("mode-row-{mode}"))
                    .ghost()
                    .selected(view.selected_mode == index)
                    .label(mode.to_string())
                    .on_click(cx.listener(move |view, _, _, cx| {
                        view.state.session_view.selected_mode = index;
                        cx.notify();
                    }))
            }));
        let pending = self.pending_command;
        let error = self.settings_error.clone();
        div()
            .absolute()
            .inset_0()
            .debug_selector(|| "settings-overlay".into())
            .track_focus(&self.settings_focus)
            .occlude()
            .bg(cx.theme().overlay)
            .v_flex()
            .items_center()
            // Flat panel on scrim: sits at 25% of the viewport HEIGHT rather
            // than centred, matching the wiki modal shape. GPUI's
            // `pt(relative(0.25))` computes a fraction of parent WIDTH
            // (CSS-quirk), which drifts the modal off the shelf on wide
            // windows — measure the height directly and offset in pixels.
            // Contract line 91.
            .pt(window.viewport_size().height * theme::MODAL_TOP_FRACTION)
            .px(px(16.))
            .child(
                div()
                    .debug_selector(|| "settings-panel".into())
                    .v_flex()
                    .w(theme::MODAL_WIDTH)
                    .max_w_full()
                    .max_h(px(560.))
                    .pt(theme::MODAL_PADDING_TOP)
                    .pb(theme::MODAL_PADDING_BOTTOM)
                    .px(theme::MODAL_PADDING_X)
                    .gap_3()
                    .bg(cx.theme().sidebar)
                    .child(modal_title("Session settings"))
                    .child(modal_field_label("Model", cx))
                    .child(list)
                    .child(modal_field_label("Approval mode", cx))
                    .child(mode_row)
                    .children(self.login_providers.iter().map(|provider| {
                        self.render_login_row(
                            provider,
                            "settings-login",
                            cx.entity().downgrade(),
                            cx,
                        )
                    }))
                    .when_some(error, |dialog, error| {
                        dialog.child(Alert::error("settings-error", error))
                    })
                    .child(
                        div()
                            .h_flex()
                            .justify_end()
                            .gap_2()
                            .child(
                                Button::new("settings-close")
                                    .debug_selector(|| "settings-close".into())
                                    .ghost()
                                    .label("Close")
                                    .h(theme::MODAL_BUTTON_HEIGHT)
                                    .on_click(cx.listener(|view, _, window, cx| {
                                        view.close_settings(window, cx)
                                    })),
                            )
                            .child(
                                Button::new("settings-apply")
                                    .debug_selector(|| "settings-apply".into())
                                    .primary()
                                    .label(if pending { "Applying…" } else { "Apply" })
                                    .disabled(pending)
                                    .h(theme::MODAL_BUTTON_HEIGHT)
                                    .on_click(
                                        cx.listener(|view, _, _, cx| view.apply_settings(cx)),
                                    ),
                            ),
                    ),
            )
            .into_any_element()
    }

    fn render_row(&self, index: usize, view: gpui::WeakEntity<Self>, cx: &App) -> gpui::AnyElement {
        let is_first = index == 0;
        let is_last = index + 1 == self.state.transcript.len();
        let this_is_tool = matches!(self.state.transcript[index], TranscriptEntry::Tool { .. });
        let next_is_tool = self
            .state
            .transcript
            .get(index + 1)
            .is_some_and(|entry| matches!(entry, TranscriptEntry::Tool { .. }));
        // Adjacent tool rows collapse the row gap so a run of receipts reads
        // as one column — matches the wiki agent-run rhythm exactly. The last
        // row also carries no gap so column bottom padding lands cleanly.
        let row_gap = if is_last || (this_is_tool && next_is_tool) {
            px(0.)
        } else {
            theme::TRANSCRIPT_ROW_GAP
        };

        let inner = self.render_row_inner(index, view, cx);
        div()
            .debug_selector(|| "transcript-row".into())
            .w_full()
            .min_w_0()
            .flex()
            .flex_col()
            .items_center()
            // Column top/bottom padding lives on the first/last row so it
            // travels with the virtual scroller — a wrapper around the
            // scroller would leave the padding fixed while rows scroll under.
            .when(is_first, |row| row.pt_4())
            .when(is_last, |row| row.pb_3())
            .pb(row_gap)
            .child(
                div()
                    .w_full()
                    .min_w_0()
                    .max_w(theme::TRANSCRIPT_MAX_WIDTH)
                    .px_4()
                    .child(inner),
            )
            .into_any_element()
    }

    // The six transcript render functions below MUST paint every user-
    // visible string via the `RowText` model that `render_row_inner`
    // builds. A destructured `let RowText::<Variant> { … }` at the top of
    // each renderer names every field the model carries — clippy's
    // `unused_variables` under `deny(warnings)` catches a renderer that
    // stops painting a field, and the `renderer_literal_fence` guard
    // test (see tests.rs) rejects any inline user-visible string literal
    // in these fn bodies. Together the two guards make a NEW stray
    // literal impossible: dropped fields fail the build, and new bare
    // literals fail the fence.
    fn render_row_inner(
        &self,
        index: usize,
        view: gpui::WeakEntity<Self>,
        cx: &App,
    ) -> gpui::AnyElement {
        let entry = &self.state.transcript[index];
        // One typed model per row. `row_text::build` is the SINGLE source
        // for every user-visible string a renderer paints, so a sentinel-
        // carrying payload cannot reach any field through a side path.
        let text = row_text::build(
            entry,
            index,
            &self.state.session_view,
            self.state.session_view.available,
        );
        match text {
            RowText::User(text) => self.render_user_row(index, text, view, cx),
            RowText::Assistant(text) => self.render_assistant_row(index, text, cx),
            RowText::Tool(text) => self.render_tool_row(index, text, entry, view, cx),
            RowText::Thinking(text) => self.render_thinking_row(index, text, cx),
            RowText::Error(text) => {
                let TranscriptEntry::Error { login_provider, .. } = entry else {
                    unreachable!("row-text Error variant maps to TranscriptEntry::Error")
                };
                self.render_error_row(index, text, login_provider.as_deref(), view, cx)
            }
        }
    }

    fn render_thinking_row(
        &self,
        index: usize,
        text: ThinkingRowText,
        cx: &App,
    ) -> gpui::AnyElement {
        // Header-only marker at muted-foreground. The header text comes
        // from the typed model; a sentinel-carrying reasoning payload
        // cannot land here because `Thinking` carries no body.
        let ThinkingRowText { header } = text;
        let color = cx.theme().muted_foreground;
        // state_text records (row_id, color) into the render_log at the
        // exact moment the color is applied — a mutation that swaps the
        // color argument at this call site is caught by the sample check.
        state_text(|| format!("thinking-header-{index}"), color)
            .debug_selector(move || format!("thinking-header-{index}"))
            .w_full()
            .min_w_0()
            .py(px(2.))
            .child(header)
            .into_any_element()
    }

    fn render_user_row(
        &self,
        index: usize,
        text: UserRowText<'_>,
        view: gpui::WeakEntity<Self>,
        cx: &App,
    ) -> gpui::AnyElement {
        // Destructure every field so a dropped painter fails the build.
        let UserRowText {
            content,
            attachments,
            fork_label,
        } = text;
        let content = content.to_owned();
        let fork_id =
            fork_label.and_then(|_| self.state.session_view.message_ids.get(&index).cloned());
        let group = format!("user-row-{index}");
        div()
            .group(group.clone())
            .v_flex()
            .child(
                // Rectangle + hover-reveal fork button live in the SAME layout
                // cell (relative parent, absolute button). The invisible button
                // no longer reserves a phantom row that breaks the 14px rhythm.
                div()
                    .relative()
                    .w_full()
                    .min_w_0()
                    .child(
                        div()
                            .w_full()
                            .min_w_0()
                            .py_2()
                            .px_3()
                            .bg(cx.theme().muted)
                            .border_l(theme::RAIL_WIDTH_THICK)
                            .border_color(cx.theme().primary)
                            .whitespace_normal()
                            .child(content),
                    )
                    .when_some(fork_id.zip(fork_label), |row, (id, label)| {
                        let click_id = id.clone();
                        let click_view = view.clone();
                        row.child(
                            div()
                                .absolute()
                                .top_1()
                                .right_1()
                                .opacity(0.)
                                .group_hover(group.clone(), |style| style.opacity(1.))
                                .child(
                                    Button::new(("fork", index))
                                        .debug_selector(move || format!("fork-button-{index}"))
                                        .ghost()
                                        .compact()
                                        .label(label)
                                        .on_click(move |_, _, cx| {
                                            let id = click_id.clone();
                                            let _ = click_view
                                                .update(cx, |view, cx| view.fork_message(id, cx));
                                        }),
                                ),
                        )
                    }),
            )
            .when(!attachments.is_empty(), |row| {
                row.child(
                    div().mt_1().h_flex().flex_wrap().gap_2().children(
                        attachments
                            .into_iter()
                            .enumerate()
                            .map(|(attachment_index, label)| {
                                div()
                                    .debug_selector(|| "attachment-chip".into())
                                    .px_2()
                                    .py_1()
                                    .text_size(px(12.))
                                    .bg(cx.theme().muted)
                                    .h_flex()
                                    .items_center()
                                    .gap_2()
                                    .when_some(
                                        self.sent_images.get(&(index, attachment_index)).cloned(),
                                        |chip, image| chip.child(polish::thumbnail(image, cx)),
                                    )
                                    .child(label)
                            }),
                    ),
                )
            })
            .into_any_element()
    }

    fn render_assistant_row(
        &self,
        index: usize,
        text: AssistantRowText<'_>,
        cx: &App,
    ) -> gpui::AnyElement {
        // Naked assistant turn: 2px vertical breath, no bg, no border, no rail.
        // Hierarchy is carried by weight + color tier + rails on OTHER row types,
        // not by framing the assistant. Prose sits at 1.65 line-height for the
        // wiki reading rhythm; code fences carry 12x16 padding, a 1px border,
        // and soft-wrap so long lines never introduce a horizontal scroll.
        let AssistantRowText {
            source,
            truncated_hint,
        } = text;
        let source = source.to_owned();
        let code_block = gpui::StyleRefinement::default()
            .py(px(12.))
            .px(px(16.))
            .border_1()
            .border_color(cx.theme().border)
            .whitespace_normal();
        let text_style = gpui_kit::component::text::TextViewStyle {
            code_block,
            ..Default::default()
        };
        div()
            .py(px(2.))
            .w_full()
            .min_w_0()
            .line_height(gpui::rems(1.65))
            .when_some(truncated_hint, |row, hint| row.child(hint))
            .child(
                TextView::markdown(format!("message-{index}"), source)
                    .selectable(true)
                    .style(text_style),
            )
            .into_any_element()
    }

    fn render_tool_row(
        &self,
        index: usize,
        text: ToolRowText<'_>,
        entry: &TranscriptEntry,
        view: gpui::WeakEntity<Self>,
        cx: &App,
    ) -> gpui::AnyElement {
        // Destructure every field so a dropped painter fails the build.
        let ToolRowText {
            verb,
            detail,
            output_size_label,
            hover_hint,
            tail_omitted_hint,
            body,
        } = text;
        let verb = verb.to_owned();
        let detail = detail.to_owned();
        let body = body.map(str::to_owned);
        let is_error = entry.unsuccessful();
        // State is signalled by COLOR ONLY. Running sits at normal text tier;
        // done fades to muted; failed/canceled land on danger. Contract line 83
        // forbids any textual "[working]/[done]/[failed]" marker — the color
        // helper hands us the token, and the render below applies it through
        // `state_text` / `record_state` on the verb, detail, and chevron
        // elements. A mutation that swaps the color argument at any call
        // site records the wrong color and fails the sample check.
        let state_color = tool_state_color(entry.tool_state(), cx);
        let group = format!("tool-row-{index}");
        let expanded = body.is_some();

        div()
            .group(group.clone())
            .id(("tool-receipt", index))
            .debug_selector(move || format!("tool-receipt-{index}"))
            .relative()
            .w_full()
            .min_w_0()
            .cursor_pointer()
            .hover(|style| style.bg(cx.theme().list_hover))
            .on_click(move |_, _, cx| {
                let _ = view.update(cx, |view, cx| {
                    view.state.toggle_card(index);
                    view.transcript.update(cx, |scroll, cx| {
                        scroll.remeasure_items(index..index + 1, cx);
                    });
                    cx.notify();
                });
            })
            .child(
                div()
                    .h_flex()
                    .gap_2()
                    .items_center()
                    .min_h(px(20.))
                    .child(
                        Icon::new(if expanded {
                            IconName::ChevronDown
                        } else {
                            IconName::ChevronRight
                        })
                        .size(px(12.))
                        .text_color(record_state(
                            || format!("tool-chevron-{index}"),
                            state_color,
                        )),
                    )
                    .child(
                        // Verb + detail route their state color through the
                        // recorder so a swap on this single call is caught
                        // by the render_log sample check.
                        state_text(|| format!("tool-verb-{index}"), state_color)
                            .debug_selector(move || format!("tool-verb-{index}"))
                            .flex_shrink_0()
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .child(verb),
                    )
                    .child(
                        state_text(|| format!("tool-detail-{index}"), state_color)
                            .debug_selector(move || format!("tool-detail-{index}"))
                            .min_w_0()
                            .flex_1()
                            .truncate()
                            .opacity(0.78)
                            .child(detail),
                    )
                    // Collapsed rows carry the output size at faint tier and a
                    // hover-fade "show output" hint — the visible affordance for
                    // the click-to-expand behaviour.
                    .when_some(output_size_label, |row, label| {
                        row.child(
                            div()
                                .flex_shrink_0()
                                .text_color(cx.theme().muted_foreground)
                                .opacity(0.78)
                                .text_size(px(12.))
                                .child(label),
                        )
                    })
                    .when_some(hover_hint, |row, hint| {
                        row.child(
                            div()
                                .flex_shrink_0()
                                .text_color(cx.theme().muted_foreground)
                                .opacity(0.)
                                .group_hover(group.clone(), |style| style.opacity(0.78))
                                .text_size(px(12.))
                                .child(hint),
                        )
                    }),
            )
            .when_some(body, |row, body| {
                row.child(
                    div()
                        .debug_selector(move || format!("tool-output-{index}"))
                        // Indent rail: margin 3/0/5, padding-left 8, 1px rail,
                        // panel fill — reads as a subordinate body without
                        // fighting the row's leading verb. Vertical padding sits
                        // at 2px per the wiki contract, not the 4px `.py_1()`.
                        .mt(px(3.))
                        .mb(px(5.))
                        .pl_2()
                        .py(px(2.))
                        .border_l(theme::RAIL_WIDTH_THIN)
                        .border_color(if is_error {
                            cx.theme().danger
                        } else {
                            cx.theme().border
                        })
                        .bg(cx.theme().sidebar)
                        .text_color(cx.theme().muted_foreground)
                        .when_some(tail_omitted_hint, |output, hint| {
                            output.child(div().opacity(0.7).child(hint))
                        })
                        .child(div().whitespace_normal().child(body)),
                )
            })
            .into_any_element()
    }

    fn render_error_row(
        &self,
        index: usize,
        text: ErrorRowText<'_>,
        login_provider: Option<&str>,
        view: gpui::WeakEntity<Self>,
        cx: &App,
    ) -> gpui::AnyElement {
        // Destructure every field so a dropped painter fails the build.
        let ErrorRowText {
            header,
            message,
            settings_action_label,
        } = text;
        let message = message.to_owned();
        div()
            .debug_selector(move || format!("error-block-{index}"))
            .v_flex()
            .gap_2()
            .pl_3()
            .py_1()
            .border_l(theme::RAIL_WIDTH_THICK)
            .border_color(cx.theme().danger)
            .child(
                div()
                    .text_color(cx.theme().danger)
                    .font_weight(gpui::FontWeight::SEMIBOLD)
                    .child(header),
            )
            .child(
                div()
                    .debug_selector(move || format!("error-message-{index}"))
                    .whitespace_normal()
                    .child(message),
            )
            .when_some(
                login_provider.and_then(|provider| {
                    self.login_providers
                        .iter()
                        .find(|row| row.provider == provider)
                }),
                |block, provider| {
                    block.child(self.render_login_row(
                        provider,
                        &format!("error-login-{index}"),
                        view.clone(),
                        cx,
                    ))
                },
            )
            .when_some(settings_action_label, |block, label| {
                block.child(
                    Button::new(("error-settings", index))
                        .debug_selector(move || format!("error-settings-{index}"))
                        .label(label)
                        .on_click(move |_, _, cx| {
                            let _ = view.update(cx, |view, cx| view.open_settings(cx));
                        }),
                )
            })
            .into_any_element()
    }
}

impl ZetaView {
    fn footer_mode_word(&self) -> &'static str {
        match &self.state.connection {
            ConnectionState::Lost(_) => "offline",
            ConnectionState::Reconnecting => "connecting",
            _ if !self.state.approvals.is_empty() => "approve",
            _ if self.state.thinking => "thinking",
            _ if self.state.streaming => "streaming",
            _ if self.state.active_session.is_none() => "idle",
            _ => "ready",
        }
    }

    fn render_pending_user_turn(&self, cx: &App) -> Option<gpui::AnyElement> {
        // Queued strip: user turn dashed while awaiting the server's echo;
        // flips to danger rail on `Rejected`/`Lost` so the user sees the send
        // failed without a modal or toast. Clears the moment `Sent` arrives —
        // that same instant the real user turn appears in the transcript.
        let pending = self.pending_user_turn.as_ref()?;
        let (rail_color, opacity) = if pending.failed {
            (cx.theme().danger, 0.85)
        } else {
            (cx.theme().primary, 0.6)
        };
        Some(
            div()
                .w_full()
                .min_w_0()
                .flex()
                .flex_col()
                .items_center()
                .px_4()
                .pb_2()
                .child(
                    div()
                        .w_full()
                        .min_w_0()
                        .max_w(theme::TRANSCRIPT_MAX_WIDTH)
                        .debug_selector(|| "composer-pending".into())
                        .py_2()
                        .px_3()
                        .bg(cx.theme().muted)
                        // Contract line 85 pins the queued strip to a 1px dashed
                        // rail. A thick rail here would read as an active user
                        // turn, not a waiting-for-echo signal.
                        .border_l(theme::RAIL_WIDTH_THIN)
                        .border_dashed()
                        .border_color(rail_color)
                        .opacity(opacity)
                        .whitespace_normal()
                        .child(pending.text.clone()),
                )
                .into_any_element(),
        )
    }

    fn render_composer(
        &self,
        can_send: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let focused = self.composer.focus_handle(cx).is_focused(window);
        // Composer paints its state from the semantic tokens, not from raw
        // palette values — a future theme rethink moves the tokens in one place
        // and every state stays coherent.
        let roles = theme::composer_roles(cx);
        let rail_color = if focused {
            roles.rail_focus
        } else {
            roles.rail_rest
        };
        let fill_color = if focused {
            roles.fill_focus
        } else {
            roles.fill_rest
        };
        // Enabled send: inverted — text color on canvas, hover fades to muted.
        let send_variant = ButtonCustomVariant::new(cx)
            .color(cx.theme().foreground)
            .foreground(cx.theme().background)
            .hover(cx.theme().muted_foreground)
            .active(cx.theme().muted_foreground);
        let model_target = self
            .state
            .metrics
            .model
            .clone()
            .unwrap_or_else(|| "no model".to_owned());

        div()
            .id("composer")
            .debug_selector(|| "composer".into())
            .v_flex()
            .flex_shrink_0()
            .py(theme::COMPOSER_PADDING_Y)
            .px(theme::COMPOSER_PADDING_X)
            .min_h(theme::COMPOSER_MIN_HEIGHT)
            .bg(fill_color)
            .border_l(theme::RAIL_WIDTH_THICK)
            .border_color(rail_color)
            .on_action(|_: &gpui_kit::component::input::Enter, _, _| {})
            .when(!self.composer_images.is_empty(), |composer| {
                composer.child(
                    div().mb_1().h_flex().flex_wrap().gap_2().children(
                        self.composer_images
                            .iter()
                            .enumerate()
                            .map(|(index, image)| {
                                div()
                                    .h_flex()
                                    .items_center()
                                    .gap_2()
                                    .px_2()
                                    .py_1()
                                    .bg(cx.theme().sidebar)
                                    .text_size(px(12.))
                                    .debug_selector(|| "composer-chip".into())
                                    .child(format!("{} · {} bytes", image.name, image.size))
                                    .child(
                                        Button::new(("chip-remove", index))
                                            .ghost()
                                            .compact()
                                            .label("Remove")
                                            .on_click(cx.listener(move |view, _, _, cx| {
                                                view.remove_attached_image(index, cx)
                                            })),
                                    )
                            }),
                    ),
                )
            })
            .when_some(self.composer_image_error.clone(), |composer, error| {
                composer.child(div().mb_1().child(Alert::error("image-error", error)))
            })
            // Target line above the input: mono muted with the model name in
            // the accent tier. Reads as "which target this composer is pointed
            // at" — the same role wiki's composer target-line plays.
            .child(
                div()
                    .h_flex()
                    .items_center()
                    .gap_1()
                    .h(theme::COMPOSER_TARGET_HEIGHT)
                    .text_size(px(12.))
                    .debug_selector(|| "composer-target".into())
                    .child(div().text_color(roles.target_label).child("→"))
                    .child(
                        div()
                            .text_color(roles.target_value)
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .debug_selector(|| "composer-target-name".into())
                            .child(model_target),
                    ),
            )
            // Inline row: textarea flows, action buttons sit inline — a compact
            // 64px grid rather than a 120px v_flex stack. min-height 44px keeps
            // the textarea legible without inflating the composer floor.
            .child(
                div()
                    .h_flex()
                    .items_center()
                    .gap_2()
                    .w_full()
                    .child(
                        div().flex_1().min_w_0().child(
                            Textarea::new(&self.composer)
                                .h(px(44.))
                                .appearance(false)
                                .bordered(false)
                                .disabled(!can_send)
                                .aria_label("Message zeta"),
                        ),
                    )
                    .child(
                        Button::new("attach")
                            .debug_selector(|| "attach-button".into())
                            .ghost()
                            .compact()
                            .label("Attach image")
                            .disabled(!can_send)
                            .h(theme::SEND_BUTTON_HEIGHT)
                            .on_click(cx.listener(|view, _, window, cx| {
                                view.attach_from_files(window, cx)
                            })),
                    )
                    .child(if can_send {
                        Button::new("send")
                            .debug_selector(|| "send-button".into())
                            .custom(send_variant)
                            .label("Send")
                            .h(theme::SEND_BUTTON_HEIGHT)
                            .min_w(theme::SEND_BUTTON_MIN_WIDTH)
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .on_click(cx.listener(|view, _, _, cx| view.send_composer(cx)))
                            .into_any_element()
                    } else {
                        // Disabled: paint the outline ourselves. Kit's Custom
                        // variant derives the border color from the fill color
                        // (button.rs:1011), so a transparent fill kills the
                        // border too. A plain div sets fill and border
                        // independently, with the whole presentation dimmed to
                        // 0.55 opacity per contract line 85.
                        div()
                            .debug_selector(|| "send-button".into())
                            .h_flex()
                            .items_center()
                            .justify_center()
                            .h(theme::SEND_BUTTON_HEIGHT)
                            .min_w(theme::SEND_BUTTON_MIN_WIDTH)
                            .px_3()
                            .bg(gpui::transparent_black())
                            .border_1()
                            .border_color(roles.send_disabled_outline)
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .text_color(roles.send_disabled_outline)
                            .opacity(0.55)
                            .child("Send")
                            .into_any_element()
                    }),
            )
            .into_any_element()
    }

    fn render_run_header(&self, cx: &App) -> gpui::AnyElement {
        // Two-band run header (contract line 83): band 1 (44px) carries the
        // active session label + state pill + step text; band 2 (40px)
        // carries the runtime metadata separated by 1x14 vertical rules.
        // The step text mirrors the composer's own hint so the "what am I
        // waiting for?" answer sits at both the top and the input row.
        div()
            .v_flex()
            .flex_shrink_0()
            .id("run-header")
            .debug_selector(|| "run-header".into())
            .child(self.render_run_header_band1(cx))
            .child(self.render_run_header_band2(cx))
            .into_any_element()
    }

    fn render_run_header_band1(&self, cx: &App) -> gpui::AnyElement {
        // Band 1 shape (contract line 83): min-height 44, padding 7x14, a
        // ticket-labeled title on the left, a state pill in the middle and
        // the step/blocker text filling the rest. Session label = the
        // sidebar's own preview so the operator never loses track of which
        // conversation the pill and metrics belong to.
        let session_label = self
            .state
            .active_session
            .as_ref()
            .and_then(|id| {
                self.state
                    .sessions
                    .iter()
                    .find(|row| &row.session_id == id)
                    .map(|row| sidebar::session_label(row, Some(&self.state.transcript)))
            })
            .unwrap_or_else(|| "No session".to_owned());
        let (pill_bg, pill_fg) = self.status_pill_colors(cx);
        let mode_word = self.footer_mode_word();
        let show_streaming_dot = self.state.streaming || self.state.thinking;
        let (step_text, step_color) = self.run_header_step(cx);
        div()
            .debug_selector(|| "run-header-band1".into())
            .h_flex()
            .items_center()
            .gap(px(10.))
            .flex_shrink_0()
            .min_h(theme::HEADER_BAND1_MIN_HEIGHT)
            .py(px(7.))
            .px(px(14.))
            .border_b_1()
            .border_color(cx.theme().border)
            .child(
                // The session label is the ticket-shaped anchor for the
                // whole header — mono 600 at the normal text tier so it
                // reads as the primary identity of the run.
                div()
                    .debug_selector(|| "run-header-title".into())
                    .flex_shrink_0()
                    .max_w(px(320.))
                    .truncate()
                    .font_weight(gpui::FontWeight::SEMIBOLD)
                    .text_color(cx.theme().foreground)
                    .child(session_label),
            )
            .child(
                // State pill: solid fill + canvas text, mono 600 lowercase,
                // near-square. Neutral states land on accent; the offline
                // mode lands on danger for the scarce, load-bearing alarm
                // signal. Kept as "footer-mode" for test stability.
                div()
                    .debug_selector(|| "footer-mode".into())
                    .flex_shrink_0()
                    .py(theme::STATE_PILL_PADDING_Y)
                    .px(theme::STATE_PILL_PADDING_X)
                    .bg(pill_bg)
                    .text_color(pill_fg)
                    .font_weight(gpui::FontWeight::SEMIBOLD)
                    .child(mode_word),
            )
            .when(show_streaming_dot, |row| {
                row.child(streaming_dot(cx.theme().primary))
            })
            .child(
                // Step text — the one-line explanation of what the run is
                // waiting for. Reuses `composer_hint` so the header and
                // composer never drift out of sync.
                div()
                    .flex_1()
                    .min_w_0()
                    .truncate()
                    .debug_selector(|| "run-header-step".into())
                    .text_color(step_color)
                    .child(step_text),
            )
            .into_any_element()
    }

    fn render_run_header_band2(&self, cx: &App) -> gpui::AnyElement {
        // Band 2 shape (contract line 83): min-height 40, metadata items
        // separated by 1x14 vertical rules at the faint tier. Metrics owns
        // the middle; the trailing chip carries the keybind hint so a
        // returning operator can still find "Enter sends" on the second
        // line rather than only at the composer.
        div()
            .id("status-bar")
            .debug_selector(|| "status-bar".into())
            .h_flex()
            .items_center()
            .flex_shrink_0()
            .gap(px(10.))
            .px(px(14.))
            .min_h(theme::HEADER_BAND2_MIN_HEIGHT)
            .border_b_1()
            .border_color(cx.theme().border)
            .text_color(theme::palette::text_faint())
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .truncate()
                    .debug_selector(|| "composer-hint".into())
                    .child(polish::status_label(&self.state.metrics)),
            )
            .child(status_rule(cx))
            .child(
                div()
                    .flex_shrink_0()
                    .text_color(theme::palette::text_faint())
                    .debug_selector(|| "footer-hints".into())
                    .child(self.composer_hint()),
            )
            .child(status_rule(cx))
            .child(
                // Model name pinned right — the third metadata slice the
                // wiki header carries, kept short so it never crowds out
                // the hint.
                div()
                    .flex_shrink_0()
                    .text_color(theme::palette::text_faint())
                    .debug_selector(|| "run-header-model".into())
                    .child(
                        self.state
                            .metrics
                            .model
                            .clone()
                            .unwrap_or_else(|| "—".into()),
                    ),
            )
            .into_any_element()
    }

    /// The one-line "step" text painted in band 1. Danger tier while the
    /// connection is lost so the header carries its own blocker signal
    /// before the transcript-level banner picks it up.
    fn run_header_step(&self, cx: &App) -> (&'static str, gpui::Hsla) {
        let color = match &self.state.connection {
            ConnectionState::Lost(_) => cx.theme().danger,
            _ => cx.theme().muted_foreground,
        };
        (self.composer_hint(), color)
    }

    fn status_pill_colors(&self, cx: &App) -> (gpui::Hsla, gpui::Hsla) {
        // Offline lands on the negative pill (solid danger); every other
        // mode paints as neutral accent. The wiki "positive" state (success
        // fill) has no zeta equivalent today — the assistant never reports
        // an explicit merge-ready state — so the pill only picks between
        // neutral and negative, never surprising the eye with green chrome.
        let theme = cx.theme();
        match &self.state.connection {
            ConnectionState::Lost(_) => (theme.danger, theme.danger_foreground),
            _ => (theme.primary, theme.primary_foreground),
        }
    }
}

/// Map a tool row's semantic state onto the wiki contract's color-only
/// signal: running=foreground, done=muted_foreground, failed=danger. Kept as
/// a free helper so the render layer and the guard test can only ever read
/// the same mapping.
pub(crate) fn tool_state_color(state: zeta_gui::state::ToolState, cx: &App) -> gpui::Hsla {
    use zeta_gui::state::ToolState;
    let theme = cx.theme();
    match state {
        ToolState::Running => theme.foreground,
        ToolState::Done => theme.muted_foreground,
        ToolState::Failed => theme.danger,
    }
}

/// State-color recorder. Every state-colored text element in the SEAM
/// region routes its color through `record_state` or the div-returning
/// `state_text` shorthand, both of which write `(row_id, color)` to
/// `render_log` under `test` or the `smoke-test` feature. Tests draw, then
/// assert on the recorded samples per row — a mutation that swaps the color
/// argument at any call site records the wrong color and fails the check.
///
/// This is the "call-time" replacement for the round-6 pre-draw probe: the
/// recorder sits inside the color path itself, so the recorded value is by
/// construction the value that reached `.text_color(...)`.
/// The `row_id` closure is called only when the recorder is compiled in
/// (`test` or the `smoke-test` feature); production render never formats a
/// row id string, so the recorder machinery costs zero allocations in
/// release builds.
#[cfg_attr(not(any(test, feature = "smoke-test")), allow(unused_variables))]
pub(crate) fn record_state<F>(row_id: F, color: gpui::Hsla) -> gpui::Hsla
where
    F: FnOnce() -> String,
{
    #[cfg(any(test, feature = "smoke-test"))]
    render_log::record(&row_id(), color);
    color
}

/// Convenience wrapper: creates a `Div` with `.text_color(color)` set and
/// records into the render log in one call. Preferred over `record_state`
/// for divs; the bare recorder covers the icon path where `Icon` needs to
/// receive the color directly.
pub(crate) fn state_text<F>(row_id: F, color: gpui::Hsla) -> gpui::Div
where
    F: FnOnce() -> String,
{
    div().text_color(record_state(row_id, color))
}

#[cfg(any(test, feature = "smoke-test"))]
pub(crate) mod render_log {
    use gpui::Hsla;
    use std::cell::RefCell;

    #[derive(Debug, Clone)]
    pub(crate) struct Sample {
        pub(crate) row_id: String,
        pub(crate) color: Hsla,
    }

    // Thread-local so parallel `cargo test` workers do not cross-contaminate.
    thread_local! {
        static SAMPLES: RefCell<Vec<Sample>> = const { RefCell::new(Vec::new()) };
    }

    pub(crate) fn clear() {
        SAMPLES.with(|slot| slot.borrow_mut().clear());
    }

    pub(crate) fn record(row_id: &str, color: Hsla) {
        SAMPLES.with(|slot| {
            slot.borrow_mut().push(Sample {
                row_id: row_id.to_owned(),
                color,
            })
        });
    }

    pub(crate) fn samples() -> Vec<Sample> {
        SAMPLES.with(|slot| slot.borrow().clone())
    }
}

/// Modal title band: 15px semibold on the left, a plain-text `esc` hint at
/// the right. Contract line 91 pins this shape for every wiki-run modal.
pub(crate) fn modal_title(title: &'static str) -> gpui::AnyElement {
    div()
        .debug_selector(|| "modal-title".into())
        .h_flex()
        .items_center()
        .justify_between()
        .w_full()
        .child(
            div()
                .text_size(theme::FONT_SIZE)
                .font_weight(gpui::FontWeight::SEMIBOLD)
                .child(title),
        )
        .child(div().text_color(theme::palette::text_faint()).child("esc"))
        .into_any_element()
}

/// Modal field caption: muted tier, no uppercase, used to name a control
/// group (model list, approval mode row). Hierarchy comes from color tier
/// alone — contract line 62 pins ONE size across the whole app.
pub(crate) fn modal_field_label(label: &'static str, cx: &App) -> gpui::AnyElement {
    div()
        .text_color(cx.theme().muted_foreground)
        .child(label)
        .into_any_element()
}

/// Thin vertical separator between status-strip items. One-pixel wide, 14px
/// tall — the wiki header pattern for ruling adjacent metadata.
fn status_rule(cx: &App) -> gpui::AnyElement {
    div()
        .debug_selector(|| "status-rule".into())
        .flex_shrink_0()
        .w(px(1.))
        .h(theme::STATUS_RULE_HEIGHT)
        .bg(cx.theme().border)
        .into_any_element()
}

/// Small pulsing dot rendered while the assistant is streaming or thinking.
/// Opacity cycles 0.25 → 1 over ~1.2s in a synced loop so all zeta windows on
/// screen breathe in phase — matches the wiki agent-run indicator.
fn streaming_dot(color: gpui::Hsla) -> gpui::AnyElement {
    div()
        .w(theme::STREAM_DOT_SIZE)
        .h(theme::STREAM_DOT_SIZE)
        .rounded_full()
        .bg(color)
        .debug_selector(|| "streaming-dot".into())
        .with_animation(
            "streaming-dot",
            Animation::new(Duration::from_millis(1200))
                .repeat_synced()
                .with_easing(ease_in_out),
            |el, delta| {
                // Delta 0..1: triangle wave 0..1..0 so we breathe up then down
                // without the pop-back that a sawtooth would show.
                let triangle = 1.0 - (delta * 2.0 - 1.0).abs();
                let alpha = 0.25 + triangle * 0.75;
                el.opacity(alpha)
            },
        )
        .into_any_element()
}

impl Render for ZetaView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // Reset the state-color recorder at the start of every render so
        // tests observe only the samples produced by the draw they trigger,
        // and no test needs a manual `render_log::clear()` before drawing.
        // Compiled out in production alongside the recorder itself.
        #[cfg(any(test, feature = "smoke-test"))]
        render_log::clear();
        let needs_login = !self.login_providers.is_empty()
            && self
                .login_providers
                .iter()
                .all(|row| !row.credentials_present);
        let can_send = self.can_change_session() && self.state.active_session.is_some();
        let view = cx.entity();
        let row_view = view.downgrade();
        let transcript = MessageScroller::new(
            "transcript",
            self.transcript.clone(),
            move |index, _, cx| view.read(cx).render_row(index, row_view.clone(), cx),
        )
        .with_row_style(gpui::StyleRefinement::default().pb_0())
        .flex_1()
        .min_h_0()
        .min_w_0();
        // Connection banner paints as a wiki-run "blocker row": a 2px danger
        // left rail on a 10% danger tint, no framed alert card. Reconnecting
        // borrows the same shape at the accent tier (transitional, not
        // blocking). Contract line 83.
        let banner = match &self.state.connection {
            ConnectionState::Lost(error) => Some(
                div()
                    .debug_selector(|| "connection-lost".into())
                    .v_flex()
                    .gap_2()
                    .py(px(8.))
                    .px(px(14.))
                    .border_l(theme::ATTENTION_RAIL_WIDTH)
                    .border_color(cx.theme().danger)
                    .bg(theme::palette::danger_tint())
                    .child(
                        div()
                            .text_color(cx.theme().danger)
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .child(format!("Connection lost: {error}")),
                    )
                    .child(
                        Button::new("reconnect")
                            .debug_selector(|| "reconnect-button".into())
                            .label("Reconnect")
                            .h(theme::MODAL_BUTTON_HEIGHT)
                            .on_click(cx.listener(|view, _, _, cx| view.reconnect(cx))),
                    ),
            ),
            ConnectionState::Reconnecting => Some(
                div()
                    .debug_selector(|| "connecting".into())
                    .py(px(8.))
                    .px(px(14.))
                    .border_l(theme::ATTENTION_RAIL_WIDTH)
                    .border_color(cx.theme().primary)
                    .text_color(cx.theme().primary)
                    .child("Connecting to zeta…"),
            ),
            ConnectionState::Connected => None,
        };
        let main = div()
            .v_flex()
            .flex_1()
            .min_w_0()
            .h_full()
            // Two-band run header sits above the blocker banner so a
            // connection-lost state shows the danger pill on band 1 first
            // and the tint-rail banner directly below. Contract line 83.
            .child(self.render_run_header(cx))
            .children(banner)
            .when(!self.settings_open, |main| {
                main.children(
                    self.login_providers
                        .iter()
                        .filter(|provider| provider.progress != LoginProgress::Idle)
                        .map(|provider| {
                            self.render_login_row(
                                provider,
                                "login-progress",
                                cx.entity().downgrade(),
                                cx,
                            )
                        }),
                )
            })
            .when_some(self.command_error.clone(), |main, error| {
                // Command-error strip carries the same blocker treatment as
                // the connection-lost banner — one alarm chrome pattern for
                // every top-of-main failure.
                main.child(
                    div()
                        .debug_selector(|| "command-error".into())
                        .py(px(8.))
                        .px(px(14.))
                        .border_l(theme::ATTENTION_RAIL_WIDTH)
                        .border_color(cx.theme().danger)
                        .bg(theme::palette::danger_tint())
                        .text_color(cx.theme().danger)
                        .whitespace_normal()
                        .child(error),
                )
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
                                .v_flex()
                                .gap_3()
                                .child(if needs_login {
                                    "Log in with a provider to start a conversation"
                                } else if self.state.active_session.is_some() {
                                    "Send a message to start a conversation"
                                } else {
                                    "Create a session, then send a message"
                                })
                                .children(
                                    self.login_providers
                                        .iter()
                                        .filter(|row| {
                                            needs_login && row.progress == LoginProgress::Idle
                                        })
                                        .map(|provider| {
                                            self.render_login_row(
                                                provider,
                                                "first-login",
                                                cx.entity().downgrade(),
                                                cx,
                                            )
                                        }),
                                ),
                        )
                    })
                    .child(transcript),
            )
            .children(self.render_pending_user_turn(cx))
            .child(self.render_composer(can_send, window, cx));
        div()
            .size_full()
            .relative()
            .bg(cx.theme().background)
            .text_color(cx.theme().foreground)
            .font_family("JetBrains Mono")
            .text_size(theme::FONT_SIZE)
            .on_action(cx.listener(|view, _: &polish::NewSession, _, cx| view.new_session(cx)))
            .on_action(|_: &polish::About, window, cx| {
                drop(window.prompt(
                    gpui::PromptLevel::Info,
                    "zeta",
                    Some(concat!("Version ", env!("CARGO_PKG_VERSION"))),
                    &["OK"],
                    cx,
                ));
            })
            .on_key_down(cx.listener(Self::control_key))
            .capture_key_down(cx.listener(Self::settings_key))
            .capture_action(
                cx.listener(|view, _: &gpui_kit::component::input::Enter, _, cx| {
                    if view.session_edit.is_some() {
                        view.commit_session_edit(cx);
                        cx.stop_propagation();
                    } else {
                        cx.propagate();
                    }
                }),
            )
            .capture_action(cx.listener(
                |view, _: &gpui_kit::component::input::Escape, window, cx| {
                    if view.session_edit.is_some() {
                        view.close_session_edit(window, cx);
                        cx.stop_propagation();
                    } else {
                        cx.propagate();
                    }
                },
            ))
            .capture_action(cx.listener(Self::paste_image))
            .child(
                div()
                    .h_flex()
                    .size_full()
                    .items_stretch()
                    .child(self.render_sidebar(window, cx))
                    .child(main),
            )
            .child(self.dialogs.clone())
            .when(self.session_edit.is_some(), |view| {
                view.child(self.render_session_edit(window, cx))
            })
            .when(self.settings_open, |view| {
                view.child(self.render_settings_overlay(window, cx))
            })
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
    polish::init_menus(cx);
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
