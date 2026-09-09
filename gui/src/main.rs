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
    ActiveTheme, Disableable, Icon, IconName, Root, Selectable, StyledExt, Theme, WindowExt,
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
    settings_error: Option<String>,
    composer_images: Vec<ImageAttachment>,
    composer_image_error: Option<String>,
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
            model_scroll: gpui::ScrollHandle::new(),
            pending_command: false,
            command_error: None,
            dialog_request: None,
            approval_pending: false,
            settings_open: false,
            settings_error: None,
            composer_images: Vec::new(),
            composer_image_error: None,
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
                self.scroll_model_into_view();
            }
            WorkerMessage::SettingsApplied(settings) => {
                self.pending_command = false;
                self.state.session_view.current_model = settings.model;
                self.settings_error = None;
                self.settings_open = false;
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
        let has_images = !self.composer_images.is_empty();
        if text.trim().is_empty() && !has_images {
            return;
        }
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

    fn close_settings(&mut self, cx: &mut Context<Self>) {
        self.settings_open = false;
        self.settings_error = None;
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
            self.scroll_model_into_view();
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

    fn scroll_model_into_view(&self) {
        let selected = self.state.session_view.selected_model;
        // ScrollHandle indexes the model rows only; group heading indexes are
        // added after each row so the first row is always ix 0.
        self.model_scroll.scroll_to_item(selected);
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
    ) {
        if !self.can_change_session() || self.state.active_session.is_none() {
            return;
        }
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
                }
            }
            Err(error) => self.composer_image_error = Some(error),
        }
        cx.notify();
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
            let _ = view.update(cx, |view, cx| view.add_attached_images(images, cx));
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
                    self.add_attached_images(
                        ImageAttachment::from_bytes(name, image.bytes()).map(|image| vec![image]),
                        cx,
                    );
                    return true;
                }
                gpui::ClipboardEntry::ExternalPaths(paths) => {
                    let images: Result<Vec<ImageAttachment>, String> = paths
                        .paths()
                        .iter()
                        .map(|path| ImageAttachment::from_path(path))
                        .collect();
                    self.add_attached_images(images, cx);
                    return true;
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

    fn paste_key(&mut self, event: &KeyDownEvent, _: &mut Window, cx: &mut Context<Self>) {
        // Intercept image pastes before the Kit textarea consumes Cmd+V as text.
        // Text pastes fall through untouched.
        if event.keystroke.key != "v" {
            return;
        }
        let modifiers = &event.keystroke.modifiers;
        if !(modifiers.platform || modifiers.control) {
            return;
        }
        if self.attach_from_clipboard(cx) {
            cx.stop_propagation();
            cx.notify();
        }
    }

    fn settings_key(&mut self, event: &KeyDownEvent, _: &mut Window, cx: &mut Context<Self>) {
        if !self.settings_open {
            return;
        }
        match event.keystroke.key.as_str() {
            "escape" => self.close_settings(cx),
            "up" => self.move_settings_model(-1, cx),
            "down" => self.move_settings_model(1, cx),
            "enter" => self.apply_settings(cx),
            _ => return,
        }
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
                    .debug_selector(|| "mode-row".into())
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
            .occlude()
            .bg(gpui::black().opacity(0.55))
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
                                    .on_click(
                                        cx.listener(|view, _, _, cx| view.close_settings(cx)),
                                    ),
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
        let row = div()
            .debug_selector(|| "transcript-row".into())
            .w_full()
            .min_w_0()
            .px_6()
            .py_3();
        match &self.state.transcript[index] {
            TranscriptEntry::User(text) => {
                let fork_id = self
                    .state
                    .session_view
                    .available
                    .then(|| self.state.session_view.message_ids.get(&index).cloned())
                    .flatten();
                let group = format!("user-row-{index}");
                row.group(group.clone())
                    .child(
                        div()
                            .h_flex()
                            .items_center()
                            .justify_between()
                            .mb_2()
                            .child(
                                div()
                                    .text_size(px(12.))
                                    .text_color(cx.theme().muted_foreground)
                                    .child("you"),
                            )
                            .when_some(fork_id, |header, id| {
                                let click_id = id.clone();
                                let click_view = view.clone();
                                header.child(
                                    Button::new(("fork", index))
                                        .debug_selector(move || format!("fork-button-{index}"))
                                        .ghost()
                                        .label("Fork here")
                                        // Hover-reveal keeps the affordance out of
                                        // the row's default reading order.
                                        .opacity(0.)
                                        .group_hover(group.clone(), |style| style.opacity(1.))
                                        .on_click(move |_, _, cx| {
                                            let id = click_id.clone();
                                            let _ = click_view
                                                .update(cx, |view, cx| view.fork_message(id, cx));
                                        }),
                                )
                            }),
                    )
                    .child(
                        div()
                            .pl_3()
                            .border_l_2()
                            .border_color(cx.theme().primary)
                            .child(text.clone()),
                    )
                    .children(
                        self.state
                            .session_view
                            .attachments
                            .get(&index)
                            .map(|attachments| {
                                div().h_flex().flex_wrap().gap_2().mt_2().children(
                                    attachments.iter().map(|(name, size)| {
                                        div()
                                            .debug_selector(|| "attachment-chip".into())
                                            .px_2()
                                            .py_1()
                                            .text_size(px(12.))
                                            .bg(cx.theme().muted)
                                            .child(format!("{name} · {size} bytes"))
                                    }),
                                )
                            }),
                    )
                    .into_any_element()
            }
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
            entry @ TranscriptEntry::Tool {
                name,
                summary,
                card,
                ..
            } => row
                .id(("tool-receipt", index))
                .debug_selector(move || format!("tool-receipt-{index}"))
                .py_1()
                .cursor_pointer()
                .hover(|style| style.bg(cx.theme().muted))
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
                        .px_3()
                        .py_2()
                        .min_h(px(40.))
                        .bg(cx.theme().muted)
                        .text_size(px(12.))
                        .text_color(if entry.unsuccessful() {
                            cx.theme().danger
                        } else {
                            cx.theme().muted_foreground
                        })
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
                                .min_w_0()
                                .truncate()
                                .child(format!("{} {name}  {summary}", entry.tool_marker())),
                        ),
                )
                .when(card.expanded, |row| {
                    row.child(
                        div()
                            .debug_selector(move || format!("tool-output-{index}"))
                            .ml_3()
                            .pl_3()
                            .py_2()
                            .border_l_2()
                            .border_color(cx.theme().border)
                            .text_size(px(12.))
                            .text_color(cx.theme().muted_foreground)
                            .when(card.tail.truncated, |output| {
                                output.child(div().child("Earlier output omitted"))
                            })
                            .child(div().whitespace_normal().child(card.tail.text.clone())),
                    )
                })
                .into_any_element(),
            TranscriptEntry::Error {
                message,
                settings_action,
            } => row
                .child(
                    div()
                        .debug_selector(move || format!("error-block-{index}"))
                        .v_flex()
                        .gap_2()
                        .pl_3()
                        .border_l_2()
                        .border_color(cx.theme().danger)
                        .child(div().text_color(cx.theme().danger).child("Error"))
                        .child(
                            div()
                                .debug_selector(move || format!("error-message-{index}"))
                                .whitespace_normal()
                                .child(message.clone()),
                        )
                        .when(
                            *settings_action && self.state.session_view.available,
                            |block| {
                                block.child(
                                    Button::new(("error-settings", index))
                                        .debug_selector(move || format!("error-settings-{index}"))
                                        .label("Open Settings")
                                        .on_click(move |_, _, cx| {
                                            let _ =
                                                view.update(cx, |view, cx| view.open_settings(cx));
                                        }),
                                )
                            },
                        ),
                )
                .into_any_element(),
        }
    }
}

impl Render for ZetaView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
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
                                            .bg(cx.theme().muted)
                                            .text_size(px(12.))
                                            .debug_selector(|| "composer-chip".into())
                                            .child(format!("{} · {} bytes", image.name, image.size))
                                            .child(
                                                Button::new(("chip-remove", index))
                                                    .ghost()
                                                    .label("Remove")
                                                    .on_click(cx.listener(
                                                        move |view, _, _, cx| {
                                                            view.remove_attached_image(index, cx)
                                                        },
                                                    )),
                                            )
                                    }),
                            ),
                        )
                    })
                    .when_some(self.composer_image_error.clone(), |composer, error| {
                        composer.child(Alert::error("image-error", error))
                    })
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
                                div()
                                    .h_flex()
                                    .gap_2()
                                    .child(
                                        Button::new("attach")
                                            .debug_selector(|| "attach-button".into())
                                            .ghost()
                                            .label("Attach image")
                                            .disabled(!can_send)
                                            .h(px(40.))
                                            .on_click(cx.listener(|view, _, window, cx| {
                                                view.attach_from_files(window, cx)
                                            })),
                                    )
                                    .child(
                                        Button::new("send")
                                            .debug_selector(|| "send-button".into())
                                            .primary()
                                            .label("Send")
                                            .disabled(!can_send)
                                            .h(px(40.))
                                            .on_click(
                                                cx.listener(|view, _, _, cx| {
                                                    view.send_composer(cx)
                                                }),
                                            ),
                                    ),
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
            .capture_key_down(cx.listener(Self::settings_key))
            .capture_key_down(cx.listener(Self::paste_key))
            .child(
                div()
                    .h_flex()
                    .size_full()
                    .items_stretch()
                    .child(self.render_sidebar(cx))
                    .child(main),
            )
            .child(self.dialogs.clone())
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
