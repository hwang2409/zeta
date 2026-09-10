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
    session::{ImageAttachment, SessionSettings, APPROVAL_MODES},
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
    model_scroll: gpui::ScrollHandle,
    pending_command: bool,
    command_error: Option<String>,
    dialog_request: Option<String>,
    approval_pending: bool,
    settings_open: bool,
    session_management: bool,
    session_edit: Option<session_management::SessionEdit>,
    session_edit_focus: gpui::FocusHandle,
    settings_focus: gpui::FocusHandle,
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
            command_error: None,
            dialog_request: None,
            approval_pending: false,
            settings_open: false,
            session_management: false,
            session_edit: None,
            session_edit_focus: cx.focus_handle(),
            settings_focus: cx.focus_handle(),
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
        let previous_count = self.state.transcript.len();
        let mut changed_row = None;
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
                self.state.streaming = true;
                let index = self.state.transcript.len();
                self.state.transcript.push(TranscriptEntry::User(text));
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
                self.state.streaming = true;
                self.state.transcript.push(TranscriptEntry::User(text));
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
                if self.settings_open {
                    self.settings_error = Some(error);
                } else {
                    self.command_error = Some(error);
                }
            }
            WorkerMessage::Connected => self.state.connection = ConnectionState::Connected,
            WorkerMessage::Event(event) => changed_row = self.state.apply(event),
            WorkerMessage::Lost(error) => {
                if self.session_edit.is_some() {
                    self.command_error = Some(error.clone());
                }
                self.pending_command = false;
                self.approval_pending = false;
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

    fn render_settings_overlay(&self, cx: &mut Context<Self>) -> gpui::AnyElement {
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
                        .text_size(px(11.))
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
                                        .text_size(px(11.))
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
            .h_flex()
            .items_center()
            .justify_center()
            .child(
                div()
                    .v_flex()
                    .w(px(520.))
                    .max_h(px(560.))
                    .p_5()
                    .gap_3()
                    .bg(cx.theme().background)
                    .border_1()
                    .border_color(cx.theme().border)
                    .child(
                        div()
                            .text_size(px(18.))
                            .font_weight(gpui::FontWeight::BOLD)
                            .child("Session settings"),
                    )
                    .child(
                        div()
                            .text_size(px(12.))
                            .text_color(cx.theme().muted_foreground)
                            .child("Model"),
                    )
                    .child(list)
                    .child(
                        div()
                            .text_size(px(12.))
                            .text_color(cx.theme().muted_foreground)
                            .child("Approval mode"),
                    )
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

    fn render_row_inner(
        &self,
        index: usize,
        view: gpui::WeakEntity<Self>,
        cx: &App,
    ) -> gpui::AnyElement {
        match &self.state.transcript[index] {
            TranscriptEntry::User(text) => self.render_user_row(index, text, view, cx),
            TranscriptEntry::Assistant(doc) => self.render_assistant_row(index, doc, cx),
            entry @ TranscriptEntry::Tool { .. } => self.render_tool_row(index, entry, view, cx),
            TranscriptEntry::Error {
                message,
                settings_action,
                login_provider,
            } => self.render_error_row(
                index,
                message,
                *settings_action,
                login_provider.as_deref(),
                view,
                cx,
            ),
        }
    }

    fn render_user_row(
        &self,
        index: usize,
        text: &str,
        view: gpui::WeakEntity<Self>,
        cx: &App,
    ) -> gpui::AnyElement {
        let fork_id = self
            .state
            .session_view
            .available
            .then(|| self.state.session_view.message_ids.get(&index).cloned())
            .flatten();
        let group = format!("user-row-{index}");
        div()
            .group(group.clone())
            .v_flex()
            .gap_1()
            .child(
                // The user's rectangle: mono, element fill, 3px accent rail —
                // one flat block, no shadow, no rounded chrome.
                div()
                    .w_full()
                    .min_w_0()
                    .py_2()
                    .px_3()
                    .bg(cx.theme().muted)
                    .border_l(theme::RAIL_WIDTH_THICK)
                    .border_color(cx.theme().primary)
                    .whitespace_normal()
                    .child(text.to_owned()),
            )
            .when_some(fork_id, |row, id| {
                let click_id = id.clone();
                let click_view = view.clone();
                row.child(
                    div().h_flex().justify_end().child(
                        Button::new(("fork", index))
                            .debug_selector(move || format!("fork-button-{index}"))
                            .ghost()
                            .compact()
                            .label("Fork here")
                            // Hover-reveal keeps the affordance out of the row's
                            // default reading order (mirrors the wiki pattern).
                            .opacity(0.)
                            .group_hover(group.clone(), |style| style.opacity(1.))
                            .on_click(move |_, _, cx| {
                                let id = click_id.clone();
                                let _ = click_view.update(cx, |view, cx| view.fork_message(id, cx));
                            }),
                    ),
                )
            })
            .children(
                self.state
                    .session_view
                    .attachments
                    .get(&index)
                    .map(|attachments| {
                        div().h_flex().flex_wrap().gap_2().children(
                            attachments.iter().enumerate().map(
                                |(attachment_index, (name, size))| {
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
                                            self.sent_images
                                                .get(&(index, attachment_index))
                                                .cloned(),
                                            |chip, image| chip.child(polish::thumbnail(image, cx)),
                                        )
                                        .child(format!("{name} · {size} bytes"))
                                },
                            ),
                        )
                    }),
            )
            .into_any_element()
    }

    fn render_assistant_row(
        &self,
        index: usize,
        doc: &zeta_gui::markdown::Markdown,
        _: &App,
    ) -> gpui::AnyElement {
        // Naked assistant turn: 2px vertical breath, no bg, no border, no rail.
        // Hierarchy is carried by weight + color tier + rails on OTHER row types,
        // not by framing the assistant.
        div()
            .py(px(2.))
            .w_full()
            .min_w_0()
            .when(doc.preview_truncated, |row| {
                row.child("Showing the latest streamed text…")
            })
            .child(
                TextView::markdown(format!("message-{index}"), doc.source.to_string())
                    .selectable(true),
            )
            .into_any_element()
    }

    fn render_tool_row(
        &self,
        index: usize,
        entry: &TranscriptEntry,
        view: gpui::WeakEntity<Self>,
        cx: &App,
    ) -> gpui::AnyElement {
        let TranscriptEntry::Tool {
            name,
            summary,
            card,
            ..
        } = entry
        else {
            unreachable!("render_tool_row invoked on non-Tool entry");
        };
        // State is signalled by COLOR ONLY. `running` sits at normal text tier;
        // `done` fades to muted; `failed`/`canceled` land on danger.
        let complete = matches!(entry, TranscriptEntry::Tool { complete: true, .. });
        let state_color = if entry.unsuccessful() {
            cx.theme().danger
        } else if complete {
            cx.theme().muted_foreground
        } else {
            cx.theme().foreground
        };
        // Every verb ("read", "bash", …) reads at weight 600 so the grammar
        // "[state] verb detail" is scannable in a run of many receipts.
        let verb = format!("{} {name}", entry.tool_marker());
        let detail = summary.to_owned();

        div()
            .id(("tool-receipt", index))
            .debug_selector(move || format!("tool-receipt-{index}"))
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
                    .text_color(state_color)
                    .child(
                        Icon::new(if card.expanded {
                            IconName::ChevronDown
                        } else {
                            IconName::ChevronRight
                        })
                        .size(px(12.)),
                    )
                    .child(
                        div()
                            .flex_shrink_0()
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .child(verb),
                    )
                    .child(
                        div()
                            .min_w_0()
                            .flex_1()
                            .truncate()
                            .opacity(0.78)
                            .child(detail),
                    ),
            )
            .when(card.expanded, |row| {
                let is_error = entry.unsuccessful();
                row.child(
                    div()
                        .debug_selector(move || format!("tool-output-{index}"))
                        // Indent rail: margin 3/0/5, padding-left 8, 1px rail,
                        // panel fill — reads as a subordinate body without
                        // fighting the row's leading verb.
                        .mt(px(3.))
                        .mb(px(5.))
                        .pl_2()
                        .py_1()
                        .border_l(theme::RAIL_WIDTH_THIN)
                        .border_color(if is_error {
                            cx.theme().danger
                        } else {
                            cx.theme().border
                        })
                        .bg(cx.theme().sidebar)
                        .text_color(cx.theme().muted_foreground)
                        .when(card.tail.truncated, |output| {
                            output.child(div().opacity(0.7).child("Earlier output omitted"))
                        })
                        .child(div().whitespace_normal().child(card.tail.text.clone())),
                )
            })
            .into_any_element()
    }

    fn render_error_row(
        &self,
        index: usize,
        message: &str,
        settings_action: bool,
        login_provider: Option<&str>,
        view: gpui::WeakEntity<Self>,
        cx: &App,
    ) -> gpui::AnyElement {
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
                    .child("Error"),
            )
            .child(
                div()
                    .debug_selector(move || format!("error-message-{index}"))
                    .whitespace_normal()
                    .child(message.to_owned()),
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
            .when(
                settings_action && self.state.session_view.available,
                |block| {
                    block.child(
                        Button::new(("error-settings", index))
                            .debug_selector(move || format!("error-settings-{index}"))
                            .label("Open Settings")
                            .on_click(move |_, _, cx| {
                                let _ = view.update(cx, |view, cx| view.open_settings(cx));
                            }),
                    )
                },
            )
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

    fn footer_mode_color(&self, cx: &App) -> gpui::Hsla {
        match &self.state.connection {
            ConnectionState::Lost(_) => cx.theme().danger,
            ConnectionState::Reconnecting => cx.theme().warning,
            _ if !self.state.approvals.is_empty() => cx.theme().warning,
            _ => cx.theme().primary,
        }
    }

    fn render_composer(
        &self,
        can_send: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let focused = self.composer.focus_handle(cx).is_focused(window);
        // Focus is the ONLY chrome cue on the composer: rail promotes from the
        // dim tint to full accent, and the fill lightens one step. No border,
        // no ring, no outline.
        let rail_color = if focused {
            cx.theme().primary
        } else {
            theme::palette::accent_rail_dim()
        };
        let fill_color = if focused {
            theme::palette::composer_focus_fill()
        } else {
            cx.theme().muted
        };
        let send_variant = ButtonCustomVariant::new(cx)
            .color(cx.theme().foreground)
            .foreground(cx.theme().background)
            .hover(cx.theme().muted_foreground)
            .active(cx.theme().muted_foreground);

        div()
            .id("composer")
            .debug_selector(|| "composer".into())
            .v_flex()
            .flex_shrink_0()
            .gap_2()
            .py(theme::COMPOSER_PADDING_Y)
            .px(theme::COMPOSER_PADDING_X)
            .min_h(theme::COMPOSER_MIN_HEIGHT)
            .bg(fill_color)
            .border_l(theme::RAIL_WIDTH_THICK)
            .border_color(rail_color)
            // The kit emits PressEnter and propagates its action. Consume it
            // here so native text input cannot insert a newline after submit.
            .on_action(|_: &gpui_kit::component::input::Enter, _, _| {})
            .when(!self.composer_images.is_empty(), |composer| {
                composer.child(
                    div().h_flex().flex_wrap().gap_2().children(
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
                composer.child(Alert::error("image-error", error))
            })
            .child(
                // Textarea sits transparent on the composer's fill so the rail
                // + fill is the only visible frame. `appearance(false)` drops
                // Kit's default chrome + focus ring; `bordered(false)` removes
                // the ambient border. The composer div carries the rail alone.
                div().w_full().child(
                    Textarea::new(&self.composer)
                        .h(px(56.))
                        .appearance(false)
                        .bordered(false)
                        .disabled(!can_send)
                        .aria_label("Message zeta"),
                ),
            )
            .child(
                div()
                    .h_flex()
                    .items_center()
                    .justify_end()
                    .gap_2()
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
                    .child(
                        Button::new("send")
                            .debug_selector(|| "send-button".into())
                            .custom(send_variant)
                            .label("Send")
                            .disabled(!can_send)
                            .h(theme::SEND_BUTTON_HEIGHT)
                            .min_w(theme::SEND_BUTTON_MIN_WIDTH)
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .on_click(cx.listener(|view, _, _, cx| view.send_composer(cx))),
                    ),
            )
            .into_any_element()
    }

    fn render_footer(&self, cx: &App) -> gpui::AnyElement {
        // Mode word carries the single load-bearing color on this strip.
        // Metrics and hints stay faint so the mode word wins the eye.
        let mode_color = self.footer_mode_color(cx);
        let show_streaming_dot = self.state.streaming || self.state.thinking;
        div()
            .id("status-bar")
            .h_flex()
            .items_center()
            .flex_shrink_0()
            .gap_4()
            .px_4()
            .py_2()
            .text_size(px(12.))
            .text_color(cx.theme().muted_foreground)
            .debug_selector(|| "status-bar".into())
            .child(
                div()
                    .h_flex()
                    .items_center()
                    .gap_2()
                    .child(
                        div()
                            .text_color(mode_color)
                            .font_weight(gpui::FontWeight::MEDIUM)
                            .debug_selector(|| "footer-mode".into())
                            .child(self.footer_mode_word()),
                    )
                    .when(show_streaming_dot, |row| {
                        row.child(streaming_dot(mode_color))
                    }),
            )
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .truncate()
                    .debug_selector(|| "composer-hint".into())
                    .child(polish::status_label(&self.state.metrics)),
            )
            .child(
                div()
                    .flex_shrink_0()
                    .text_color(cx.theme().muted_foreground)
                    .opacity(0.9)
                    .debug_selector(|| "footer-hints".into())
                    .child(self.composer_hint()),
            )
            .into_any_element()
    }
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
            .child(self.render_composer(can_send, window, cx))
            .child(self.render_footer(cx));
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
                    .child(self.render_sidebar(cx))
                    .child(main),
            )
            .child(self.dialogs.clone())
            .when(self.session_edit.is_some(), |view| {
                view.child(self.render_session_edit(cx))
            })
            .when(self.settings_open, |view| {
                view.child(self.render_settings_overlay(cx))
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
