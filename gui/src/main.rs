extern crate gpui_kit as gpui;

mod polish;
mod prefs;
mod session_management;
mod sidebar;
#[cfg(feature = "smoke-test")]
mod smoke;
mod theme;
mod tool_receipts;
mod transcript_render;

use gpui::{
    div, ease_in_out, prelude::*, px, Animation, AnimationExt, App, Bounds, Context, Entity,
    ExternalPaths, Focusable, KeyDownEvent, Render, Task, Window, WindowBounds, WindowOptions,
};
use gpui_kit::component::{
    alert::Alert,
    button::{Button, ButtonVariants},
    dialog::DialogButtonProps,
    input::{InputEvent, Textarea, TextareaState},
    message_scroller::{MessageScroller, MessageScrollerState},
    ActiveTheme, Disableable, Icon, IconName, Root, Selectable, StyledExt, WindowExt,
};
use gpui_kit::TestSupportExt as _;
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
    row_text,
    session::{ImageAttachment, SessionSettings, APPROVAL_MODES},
    state::{AppState, ConnectionState, TranscriptEdit, TranscriptEntry},
    worker::{CommandMessage, ConnectionWorker, WorkerMessage},
};

/// Composer chip / drop-target chrome literals. Lives at module scope so the
/// composer renderer references named consts rather than bare strings, and a
/// wording change lands in one place. The transcript-render module has its
/// own ZETA-109 typed model + AST fence; composer chrome is view chrome (not
/// a transcript row) and stays outside that fence.
mod chrome {
    pub const ATTACH_ICON_LABEL: &str = "Attach image";
    pub const ATTACH_HINT: &str = "Attach an image · drag files in · Cmd-V pastes";
    pub const DROP_TARGET_TITLE: &str = "Drop image to attach";
    pub const DROP_TARGET_HINT: &str = "PNG · JPEG · GIF · WebP · up to 512 KiB";
    pub const CHIP_REMOVE_LABEL: &str = "Remove attachment";
    pub const ATTACH_LIMIT_ERROR: &str = "Attach up to 4 images, 512 KiB total.";
    pub const ATTACH_DECODE_ERROR: &str = "could not decode this image";
}

/// Cap on pending attachments before a batch trips the size-limit error.
const MAX_ATTACHMENTS: usize = 4;

/// One pending composer attachment. Each entry renders as its own chip so a
/// mixed batch of good and bad files never fails whole-batch — the valid
/// siblings stay attached and each invalid file surfaces its own inline
/// error, never sendable. A valid chip always carries a decoded thumbnail:
/// `add_pending_attachments` runs `polish::image_source` up front and demotes
/// a decode failure straight to `Invalid`, so the render path never sees a
/// half-valid entry.
#[derive(Debug, Clone)]
enum PendingAttachment {
    Valid {
        image: ImageAttachment,
        thumbnail: std::sync::Arc<gpui::Image>,
    },
    Invalid {
        name: String,
        error: String,
    },
}

impl PendingAttachment {
    fn valid_ref(&self) -> Option<&ImageAttachment> {
        match self {
            Self::Valid { image, .. } => Some(image),
            Self::Invalid { .. } => None,
        }
    }
}

#[cfg(any(test, feature = "smoke-test"))]
impl ZetaView {
    /// Filenames for the currently-attached valid images, in chip order.
    pub(crate) fn valid_attachment_names(&self) -> Vec<String> {
        self.composer_attachments
            .iter()
            .filter_map(|p| p.valid_ref().map(|image| image.name.clone()))
            .collect()
    }

    /// Cloned list of the currently-attached valid images.
    pub(crate) fn valid_attachments(&self) -> Vec<ImageAttachment> {
        self.composer_attachments
            .iter()
            .filter_map(|p| p.valid_ref().cloned())
            .collect()
    }

    /// Count of pending chips that would be included in a Send.
    pub(crate) fn valid_attachment_count(&self) -> usize {
        self.composer_attachments
            .iter()
            .filter(|p| matches!(p, PendingAttachment::Valid { .. }))
            .count()
    }

    /// Count of pending chips that surfaced a parse error.
    pub(crate) fn invalid_attachment_count(&self) -> usize {
        self.composer_attachments
            .iter()
            .filter(|p| matches!(p, PendingAttachment::Invalid { .. }))
            .count()
    }
}

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
    /// Focus handle captured when the Settings modal opens — restored on
    /// close so keyboard users land back on the control that opened the
    /// modal (a11y precedent set by ZETA-108/123). `None` when nothing was
    /// focused at open time (e.g. Cmd-shortcut path); `close_settings` then
    /// falls back to the composer, which stays the primary work area.
    pub(crate) settings_return_focus: Option<gpui::FocusHandle>,
    /// Scroll handle for the Settings modal's three-section body. Anchored
    /// so that when Tab lands on a control inside a section that has
    /// scrolled off the top or bottom of the wrapper (18px picker on a
    /// 760px viewport), the render pass calls `scroll_to_item(section_ix)`
    /// and the focused control's parent section swings back into view.
    /// The visible focus ring stays painted inside the viewport — the
    /// safety net paired with cutting Appearance from nine tab stops to
    /// two.
    pub(crate) settings_sections_scroll: gpui::ScrollHandle,
    /// One persistent focus handle per Settings section container ("model",
    /// "behavior", "appearance"). Each container carries `.track_focus`,
    /// and the render pass uses `contains_focused` to route
    /// `settings_sections_scroll.scroll_to_item` to whichever section holds
    /// the current focus.
    pub(crate) settings_section_focus:
        std::cell::RefCell<std::collections::HashMap<&'static str, gpui::FocusHandle>>,
    // One persistent focus handle per sidebar row id — a session id or a
    // branch id. Populated lazily in the sidebar render and reused across
    // paints so tab focus survives redraws and tests can look a row's
    // handle up by the same key the renderer uses.
    pub(crate) sidebar_row_focus:
        std::cell::RefCell<std::collections::HashMap<String, gpui::FocusHandle>>,
    /// One persistent focus handle per tool-group summary row, keyed by the
    /// first-tool-call id of the group (ZETA-125). Populated lazily in the
    /// transcript renderer and reused across paints so keyboard focus and
    /// Enter/Space activation survive redraws.
    pub(crate) tool_group_focus:
        std::cell::RefCell<std::collections::HashMap<String, gpui::FocusHandle>>,
    login_providers: Vec<LoginProvider>,
    settings_error: Option<String>,
    /// Pending composer attachments (valid + invalid). Each entry paints as
    /// its own chip: valid entries carry a cached preview and are included in
    /// Send; invalid entries carry an inline error message and are never
    /// sendable. `add_attachments` / `remove_attachment` / `clear` keep the
    /// vec (and its held asset-cache handles) tidy so repeated attach/remove
    /// cycles never leak GPU image storage.
    composer_attachments: Vec<PendingAttachment>,
    /// Batch-level attach error (over-limit only). Per-file errors travel
    /// inside their own chips, not this banner.
    composer_image_error: Option<String>,
    composer_empty_hint: bool,
    /// True while an external drag is HOVERING the composer element (as
    /// opposed to being active anywhere in the window). Gates the composer's
    /// drop-target overlay so a drag over the sidebar does not light the
    /// composer up. Updated by the composer's `on_drag_move::<ExternalPaths>`
    /// listener BEFORE each render, so a single draw per drag event paints
    /// the correct overlay state; `render_composer` also clears the flag
    /// whenever `has_active_drag()` is false so a fresh drag starts clean.
    /// `Rc<Cell>` because the flag is read from `render_composer` and mutated
    /// from GPUI's event listeners without a `Context<Self>` handle.
    drag_over_composer: std::rc::Rc<std::cell::Cell<bool>>,
    sent_images: std::collections::BTreeMap<(usize, usize), std::sync::Arc<gpui::Image>>,
    commands: Sender<CommandMessage>,
    _poll_task: Option<Task<()>>,
}

impl ZetaView {
    fn new(window: &mut Window, cx: &mut Context<Self>, commands: Sender<CommandMessage>) -> Self {
        // Theme is applied before this call — `main()` reads the persisted
        // appearance through `prefs::load` + `theme::apply_with`, tests run
        // `theme::apply` inside `setup()` for a deterministic baseline. This
        // keeps `ZetaView::new` free of disk I/O so a stray gui-prefs.json
        // written by a peer test never leaks into an unrelated test's
        // fixture.
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
            settings_return_focus: None,
            settings_sections_scroll: gpui::ScrollHandle::new(),
            settings_section_focus: std::cell::RefCell::new(std::collections::HashMap::new()),
            sidebar_row_focus: std::cell::RefCell::new(std::collections::HashMap::new()),
            tool_group_focus: std::cell::RefCell::new(std::collections::HashMap::new()),
            login_providers: Vec::new(),
            settings_error: None,
            composer_attachments: Vec::new(),
            composer_image_error: None,
            composer_empty_hint: false,
            drag_over_composer: std::rc::Rc::new(std::cell::Cell::new(false)),
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
                // Capture whoever had focus at the moment settings actually
                // opens, BEFORE we hand focus to the overlay. `close_settings`
                // restores this handle so keyboard users land back on the
                // control that invoked the modal (a11y precedent set by
                // ZETA-108/123). A click-invoked open often has no focused
                // handle; `close_settings` then falls back to the composer.
                self.settings_return_focus = window.focused(cx);
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
                self.clear_composer_images(cx);
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
                // Text-only send: `send_composer` filters out invalid chips
                // from the outgoing payload, but leaves them in the composer.
                // Clear here (mirroring the `ImagesSent` branch) so a text
                // send never leaves stray error chips beside a landed turn.
                self.clear_composer_images(cx);
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
                edits = self.state.apply_status(status);
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
            self.clear_composer_images(cx);
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
        let valid: Vec<ImageAttachment> = self
            .composer_attachments
            .iter()
            .filter_map(|item| item.valid_ref().cloned())
            .collect();
        if text.trim().is_empty() && valid.is_empty() {
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
        if valid.is_empty() {
            self.queue(CommandMessage::Send(text));
        } else {
            self.queue(CommandMessage::SendImages(text, valid));
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

    /// Resolve a provider to its typed `LoginRowText` and hand it to the
    /// transcript-render module's `render_login_row`. Every login-row call
    /// site — settings overlay, error recovery, in-progress banner,
    /// first-conversation prompt — flows through this pair so the visible
    /// labels are computed OFF the render path and the render module never
    /// sees raw `provider.label()` again (r1 finding 1).
    fn render_login_provider(
        &self,
        provider: &LoginProvider,
        prefix: &str,
        view: gpui::WeakEntity<Self>,
        cx: &App,
    ) -> gpui::AnyElement {
        let text = row_text::build_login(provider, prefix, &self.state.connection);
        self.render_login_row(text, view, cx)
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
        // Focus capture happens later — inside `apply_worker_message` when
        // the `Settings` reply lands. That is the moment we actually flip
        // `settings_open` and hand keyboard focus to the overlay, so it is
        // the correct capture point. Keeping the capture inside the
        // settings path means callers on other surfaces (sidebar,
        // transcript error-hint) do not need to thread a `Window` through.
        self.pending_command = true;
        self.settings_error = None;
        self.queue(CommandMessage::LoadSettings);
        cx.notify();
    }

    fn close_settings(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.settings_open = false;
        self.settings_error = None;
        // Return focus to the invoker (a11y precedent from ZETA-108/123).
        // Fall back to the composer when the modal was opened without a
        // focused element (mouse click), so the caret is never lost.
        if let Some(handle) = self.settings_return_focus.take() {
            window.focus(&handle, cx);
        } else {
            window.focus(&self.composer.focus_handle(cx), cx);
        }
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

    /// Read the currently applied appearance out of the PER-APP theme,
    /// not the process-wide `theme::current_appearance()`. Under parallel
    /// gpui-test workers a peer test can race the ACTIVE slot; anchoring
    /// on `cx.theme()` keeps `set_*`/`adjust_font_size` deterministic in
    /// production AND in tests.
    fn app_appearance(&self, cx: &App) -> theme::Appearance {
        let theme = cx.theme();
        let id = theme::ThemeId::ALL
            .iter()
            .copied()
            .find(|id| id.palette().canvas == theme.background)
            .unwrap_or_default();
        theme::Appearance {
            theme: id,
            font_family: theme.font_family.clone(),
            font_size: theme.font_size,
        }
    }

    /// Apply a new theme id, commit the choice to gui-prefs.json, and
    /// notify so every surface repaints from the fresh palette in the same
    /// frame. The three appearance mutators share one shape so a future
    /// change to persistence lands in one place.
    fn set_theme(&mut self, id: theme::ThemeId, cx: &mut Context<Self>) {
        let mut appearance = self.app_appearance(cx);
        if appearance.theme == id {
            return;
        }
        appearance.theme = id;
        prefs::commit(cx, appearance);
        cx.notify();
    }

    fn set_font_family(&mut self, family: &'static str, cx: &mut Context<Self>) {
        let mut appearance = self.app_appearance(cx);
        if appearance.font_family.as_ref() == family {
            return;
        }
        appearance.font_family = gpui::SharedString::new_static(family);
        prefs::commit(cx, appearance);
        cx.notify();
    }

    /// Advance the current theme by `delta` positions through the shipped
    /// `ThemeId::ALL` list. Cycles wrap so a keyboard cycler never dead-ends
    /// on either edge. Used by the compact single-value theme cycler in
    /// Settings; the previous button-wall exposed FIVE tab-stops offscreen
    /// at 18px — one focusable cycler paints one visible ring instead.
    fn cycle_theme(&mut self, delta: isize, cx: &mut Context<Self>) {
        let all = theme::ThemeId::ALL;
        if all.is_empty() {
            return;
        }
        let current = self.app_appearance(cx).theme;
        let ix = all.iter().position(|id| *id == current).unwrap_or(0) as isize;
        let next = (ix + delta).rem_euclid(all.len() as isize) as usize;
        self.set_theme(all[next], cx);
    }

    /// Advance the current font family by `delta` through `FONT_FAMILIES`.
    /// Same cycler shape as `cycle_theme`; one control, one tab stop.
    fn cycle_font_family(&mut self, delta: isize, cx: &mut Context<Self>) {
        let all = theme::FONT_FAMILIES;
        if all.is_empty() {
            return;
        }
        let current = self.app_appearance(cx).font_family;
        let ix = all
            .iter()
            .position(|family| *family == current.as_ref())
            .unwrap_or(0) as isize;
        let next = (ix + delta).rem_euclid(all.len() as isize) as usize;
        self.set_font_family(all[next], cx);
    }

    fn adjust_font_size(&mut self, delta_px: f32, cx: &mut Context<Self>) {
        let mut appearance = self.app_appearance(cx);
        let target = f32::from(appearance.font_size) + delta_px;
        let next = theme::clamp_font_size(target);
        if appearance.font_size == next {
            return;
        }
        appearance.font_size = next;
        prefs::commit(cx, appearance);
        cx.notify();
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

    /// Push a per-file batch to the pending chip row. Each incoming item is
    /// either a decoded `ImageAttachment` or an `(name, error)` pair; the
    /// batch is rejected wholesale only when it would push the combined
    /// count or byte total past the shared cap. Otherwise every item lands
    /// as its own chip (valid or invalid) — one bad file never rejects its
    /// good siblings.
    fn add_pending_attachments(
        &mut self,
        items: Vec<Result<ImageAttachment, (String, String)>>,
        cx: &mut Context<Self>,
    ) -> bool {
        if !self.can_change_session() || self.state.active_session.is_none() || self.settings_open {
            return false;
        }
        if items.is_empty() {
            return false;
        }
        // Decode thumbnails up-front so a corrupt payload (valid header,
        // undecodable body) demotes the item to an Invalid chip BEFORE the
        // cap check. Otherwise Send would emit an image the preview never
        // rendered, and a batch of five undecodables would blow the cap on
        // paths that were never going to ship.
        let prepared: Vec<PendingAttachment> = items
            .into_iter()
            .map(|item| match item {
                Ok(image) => match polish::image_source(&image) {
                    Some(thumbnail) => PendingAttachment::Valid { image, thumbnail },
                    None => PendingAttachment::Invalid {
                        name: image.name,
                        error: chrome::ATTACH_DECODE_ERROR.to_string(),
                    },
                },
                Err((name, error)) => PendingAttachment::Invalid { name, error },
            })
            .collect();
        // The 4-image and 512 KiB caps count only the payloads that would
        // actually ship. Invalid chips are visible but non-sendable, so a
        // mixed batch of three good + two error files stays under a 4-cap
        // and every valid image attaches.
        let existing_valid_count = self
            .composer_attachments
            .iter()
            .filter(|item| matches!(item, PendingAttachment::Valid { .. }))
            .count();
        let incoming_valid_count = prepared
            .iter()
            .filter(|item| matches!(item, PendingAttachment::Valid { .. }))
            .count();
        let combined_valid = existing_valid_count + incoming_valid_count;
        let incoming_bytes: usize = prepared
            .iter()
            .filter_map(|item| match item {
                PendingAttachment::Valid { image, .. } => Some(image.size),
                PendingAttachment::Invalid { .. } => None,
            })
            .sum();
        let existing_bytes: usize = self
            .composer_attachments
            .iter()
            .filter_map(|item| item.valid_ref().map(|image| image.size))
            .sum();
        if combined_valid > MAX_ATTACHMENTS
            || existing_bytes + incoming_bytes > zeta_gui::session::MAX_IMAGE_BYTES
        {
            self.composer_image_error = Some(chrome::ATTACH_LIMIT_ERROR.into());
            cx.notify();
            return false;
        }
        let appended_any = !prepared.is_empty();
        for pending in prepared {
            self.composer_attachments.push(pending);
        }
        if appended_any {
            self.composer_image_error = None;
            self.composer_empty_hint = false;
        }
        cx.notify();
        appended_any
    }

    /// Single-image entry used by the clipboard-image path. On a valid
    /// decode this appends one chip and returns true. On a decode failure
    /// the batch banner surfaces (single-file clipboard has no siblings to
    /// preserve as chips) and the caller returns false so the paste
    /// propagates to the normal text-paste fallback — critical on macOS,
    /// where the clipboard often carries both an Image and a String entry
    /// for the same paste.
    fn add_single_attachment(
        &mut self,
        parsed: Result<ImageAttachment, String>,
        cx: &mut Context<Self>,
    ) -> bool {
        match parsed {
            Ok(image) => self.add_pending_attachments(vec![Ok(image)], cx),
            Err(error) => {
                self.composer_image_error = Some(error);
                cx.notify();
                false
            }
        }
    }

    fn remove_attached_image(&mut self, index: usize, cx: &mut Context<Self>) {
        if index < self.composer_attachments.len() {
            let removed = self.composer_attachments.remove(index);
            if let PendingAttachment::Valid { thumbnail, .. } = removed {
                thumbnail.remove_asset(cx);
            }
            self.composer_image_error = None;
            cx.notify();
        }
    }

    /// Drop every pending attachment (valid + invalid) and reset the
    /// batch-error banner. Also evicts each cached preview from GPUI's asset
    /// cache so repeated attach → clear cycles do not leak GPU storage.
    fn clear_composer_images(&mut self, cx: &mut Context<Self>) {
        for pending in std::mem::take(&mut self.composer_attachments) {
            if let PendingAttachment::Valid { thumbnail, .. } = pending {
                thumbnail.remove_asset(cx);
            }
        }
        self.composer_image_error = None;
        cx.notify();
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
            let _ = view.update(cx, |view, cx| {
                if view.state.active_session == session {
                    view.attach_from_paths(&selection, cx);
                }
            });
        })
        .detach();
    }

    /// Attach a batch of on-disk paths. Parses each path individually so a
    /// bad file (unreadable, oversize, undecodable, wrong format) becomes
    /// its own error chip while good siblings survive.
    fn attach_from_paths(&mut self, paths: &[PathBuf], cx: &mut Context<Self>) -> bool {
        let items: Vec<_> = paths
            .iter()
            .map(|path| {
                let name = path
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_else(|| path.display().to_string());
                ImageAttachment::from_path(path).map_err(|error| (name, error))
            })
            .collect();
        self.add_pending_attachments(items, cx)
    }

    fn attach_from_clipboard(&mut self, cx: &mut Context<Self>) -> bool {
        let Some(item) = cx.read_from_clipboard() else {
            return false;
        };
        for entry in item.entries() {
            match entry {
                gpui::ClipboardEntry::Image(image) => {
                    let name = format!("pasted-image.{}", image.format.extension());
                    return self.add_single_attachment(
                        ImageAttachment::from_bytes(name, image.bytes()),
                        cx,
                    );
                }
                gpui::ClipboardEntry::ExternalPaths(paths) => {
                    return self.attach_from_paths(paths.paths(), cx);
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
        // Tab / Shift-Tab walk the modal's tab-stop registry so keyboard
        // users reach every control (segmented pickers, stepper, Close,
        // Apply) without touching the mouse. `focus_next` / `focus_prev`
        // are the same helpers Root's Tab/Shift-Tab bindings call — see
        // the sidebar row focus tests. Every other key is swallowed so a
        // typed letter can't fall through to the composer.
        let key = event.keystroke.key.as_str();
        let modifiers = event.keystroke.modifiers;
        if key == "tab" && !modifiers.control && !modifiers.alt && !modifiers.platform {
            if modifiers.shift {
                window.focus_prev(cx);
            } else {
                window.focus_next(cx);
            }
            window.prevent_default();
            cx.stop_propagation();
            cx.notify();
            return;
        }
        if !modifiers.modified() {
            match key {
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

    /// Lazily allocated per-section FocusHandle for the Settings modal.
    /// Same shape as `sidebar_row_focus` in `sidebar.rs`: create once, reuse
    /// across paints so `contains_focused` compares against a stable handle.
    /// Called during render so the borrow scope stays inside one frame.
    fn settings_section_focus_handle(&self, key: &'static str, cx: &App) -> gpui::FocusHandle {
        let mut map = self.settings_section_focus.borrow_mut();
        if let Some(handle) = map.get(key) {
            return handle.clone();
        }
        let handle = cx.focus_handle();
        map.insert(key, handle.clone());
        handle
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
            // Model list cap keeps the whole panel inside the 760px test
            // viewport once the three-section body (Model + Behavior +
            // Appearance) AND an optional credential-error alert are
            // stacked below it. Below this cap the list scrolls; the
            // "current" model is auto-scrolled into view regardless of
            // the visible slice.
            .max_h(theme::SETTINGS_MODEL_LIST_MAX_HEIGHT)
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
                                        .text_size(theme::label_small(cx.theme().font_size))
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
        let mode_segmented = div()
            .debug_selector(|| "settings-approval-segmented".into())
            .h_flex()
            .gap_1()
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
        let appearance = self.app_appearance(cx);
        // Compact single-value cyclers replace the pre-round-2 button
        // walls (five theme buttons + four font buttons). Each cycler is
        // ONE focusable button showing the current selection; clicking it
        // advances to the next value, wrapping at the ends. The row
        // description below explains the interaction. Result: two tab
        // stops in the Appearance section instead of nine, so every focus
        // ring paints inside the viewport at 18px.
        let theme_cycler = Button::new("settings-theme-cycler")
            .debug_selector(|| "settings-theme-cycler".into())
            .ghost()
            .compact()
            .label(appearance.theme.label())
            .on_click(cx.listener(|view, _, _, cx| view.cycle_theme(1, cx)));
        let font_cycler = Button::new("settings-font-cycler")
            .debug_selector(|| "settings-font-cycler".into())
            .ghost()
            .compact()
            .label(gpui::SharedString::from(appearance.font_family.to_string()))
            .on_click(cx.listener(|view, _, _, cx| view.cycle_font_family(1, cx)));
        let size_px = f32::from(appearance.font_size);
        let font_size_px = size_px.round() as i32;
        let can_shrink = size_px > theme::MIN_FONT_SIZE_PX;
        let can_grow = size_px < theme::MAX_FONT_SIZE_PX;
        // Stepper: `−` [value] `+` framed as a single cluster on the right
        // so it reads as ONE control, not three loose buttons. Disabled
        // `−` at MIN and `+` at MAX carry the picker range without a
        // "range 11-18px" caption cluttering the row.
        let size_stepper = div()
            .debug_selector(|| "settings-font-size-stepper".into())
            .h_flex()
            .items_center()
            .gap_1()
            .child(
                Button::new("font-size-shrink")
                    .debug_selector(|| "font-size-shrink".into())
                    .ghost()
                    .compact()
                    .label("−")
                    .disabled(!can_shrink)
                    .on_click(cx.listener(|view, _, _, cx| view.adjust_font_size(-1., cx))),
            )
            .child(
                div()
                    .debug_selector(|| "font-size-value".into())
                    .min_w(px(44.))
                    .text_align(gpui::TextAlign::Center)
                    .text_color(cx.theme().foreground)
                    .child(format!("{font_size_px}px")),
            )
            .child(
                Button::new("font-size-grow")
                    .debug_selector(|| "font-size-grow".into())
                    .ghost()
                    .compact()
                    .label("+")
                    .disabled(!can_grow)
                    .on_click(cx.listener(|view, _, _, cx| view.adjust_font_size(1., cx))),
            );
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
            .px(theme::MODAL_PADDING_X)
            .child(
                div()
                    .debug_selector(|| "settings-panel".into())
                    .v_flex()
                    .w(theme::MODAL_WIDTH)
                    .max_w_full()
                    // Panel caps at 75% of viewport height — the same
                    // budget the 25% modal-top shelf leaves — so at ANY
                    // font-picker base (11px…18px) the title and the
                    // Close/Apply action row stay clickable. The sections
                    // body inside the panel is a scrollable flex slot; any
                    // overflow beyond the cap scrolls through the sections
                    // rather than pushing Close past the viewport bottom.
                    .max_h(
                        window.viewport_size().height * (1.0 - theme::MODAL_TOP_FRACTION)
                            - theme::MODAL_PADDING_X,
                    )
                    .pt(theme::MODAL_PADDING_TOP)
                    .pb(theme::MODAL_PADDING_BOTTOM)
                    .px(theme::MODAL_PADDING_X)
                    .gap_3()
                    .bg(cx.theme().sidebar)
                    .child(modal_title("Session settings", cx))
                    // Three sections stacked with the section-gap between
                    // them so Model / Behavior / Appearance read as three
                    // distinct clusters (Law of Proximity), not one long
                    // strip of muted captions. `flex_1 + min_h_0 +
                    // overflow_y_scroll` lets the sections shrink and
                    // scroll when the panel cap bites (18px picker, tiny
                    // viewport) so Close/Apply stays at the panel bottom.
                    .child({
                        let model_focus = self.settings_section_focus_handle("model", cx);
                        let behavior_focus = self.settings_section_focus_handle("behavior", cx);
                        let appearance_focus = self.settings_section_focus_handle("appearance", cx);
                        // Safety net: if Tab lands on a control inside a
                        // section that has scrolled past the wrapper edge
                        // (18px picker on a 760px viewport), reveal that
                        // section before the frame paints. Each container
                        // carries a stable focus handle via
                        // `.track_focus(&focus)` in `settings_section`;
                        // `scroll_to_item(child_ix)` uses the sibling index
                        // of the section within the sections wrapper.
                        let focused_section = if model_focus.contains_focused(window, cx) {
                            Some(0)
                        } else if behavior_focus.contains_focused(window, cx) {
                            Some(1)
                        } else if appearance_focus.contains_focused(window, cx) {
                            Some(2)
                        } else {
                            None
                        };
                        if let Some(ix) = focused_section {
                            self.settings_sections_scroll.scroll_to_item(ix);
                        }
                        div()
                            .id("settings-sections")
                            .v_flex()
                            .flex_1()
                            .min_h_0()
                            .overflow_y_scroll()
                            .track_scroll(&self.settings_sections_scroll)
                            .gap(theme::SETTINGS_SECTION_GAP)
                            .child(
                                settings_section(
                                    "settings-section-model",
                                    "Model",
                                    &model_focus,
                                    cx,
                                )
                                .child(list),
                            )
                            .child(
                                settings_section(
                                    "settings-section-behavior",
                                    "Behavior",
                                    &behavior_focus,
                                    cx,
                                )
                                .child(settings_row(
                                    "settings-row-approval",
                                    "Approval mode",
                                    Some("How the agent handles risky actions."),
                                    mode_segmented,
                                    cx,
                                )),
                            )
                            .child(
                                settings_section(
                                    "settings-section-appearance",
                                    "Appearance",
                                    &appearance_focus,
                                    cx,
                                )
                                .child(settings_row(
                                    "settings-row-theme",
                                    "Theme",
                                    Some("Click to cycle themes."),
                                    theme_cycler,
                                    cx,
                                ))
                                .child(settings_row(
                                    "settings-row-font",
                                    "Font",
                                    Some("Click to cycle monospace families."),
                                    font_cycler,
                                    cx,
                                ))
                                .child(settings_row(
                                    "settings-row-size",
                                    "Font size",
                                    Some("Whole pixels, 11 to 18."),
                                    size_stepper,
                                    cx,
                                )),
                            )
                    })
                    .children(self.login_providers.iter().map(|provider| {
                        self.render_login_provider(
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
        let model_target = self
            .state
            .metrics
            .model
            .clone()
            .unwrap_or_else(|| "no model".to_owned());
        let composer_hint = self.composer_hint();

        let drop_enabled = can_send;
        // Overlay lights only when the drag is actually over the composer,
        // not any time a drag is active in the window. `on_drag_move` fires
        // in Capture phase on every drag movement and carries the composer
        // hitbox as `event.bounds` — a bounds-vs-position check there
        // updates `drag_over_composer` BEFORE the render that reads it, so
        // one draw per drag event paints the correct overlay state.
        //
        // When the drag ends (`FileDropEvent::Exited`/`Ended` flip
        // `has_active_drag` false and call `refresh()`), we also clear the
        // hover flag here so a subsequent drag starts fresh — otherwise the
        // last-known "inside" state from the prior drag would linger.
        if !cx.has_active_drag() {
            self.drag_over_composer.set(false);
        }
        let drag_active = drop_enabled && cx.has_active_drag() && self.drag_over_composer.get();
        div()
            .id("composer")
            .debug_selector(|| "composer".into())
            .v_flex()
            .flex_shrink_0()
            .relative()
            .py(theme::COMPOSER_PADDING_Y)
            .px(theme::COMPOSER_PADDING_X)
            .min_h(theme::COMPOSER_MIN_HEIGHT)
            .bg(fill_color)
            .border_l(theme::RAIL_WIDTH_THICK)
            .border_color(rail_color)
            .on_action(|_: &gpui_kit::component::input::Enter, _, _| {})
            .when(drop_enabled, |composer| {
                // Hover tracking runs in `on_drag_move`, not the fluent
                // `drag_over` style. GPUI's `drag_over` closure fires per
                // PAINT while the drag is over the hitbox — reading its
                // side-effect the next frame is a stale-state trap: one
                // move outside the composer leaves the overlay visible
                // for a frame, and rapid exit/re-entry flickers. The
                // `on_drag_move` handler receives the current mouse
                // position and this element's bounds every drag move
                // (Capture phase, before the render that reads the flag),
                // so a bounds-vs-position check there updates hover
                // state BEFORE the next paint.
                composer
                    .on_drop::<ExternalPaths>(cx.listener(|view, paths: &ExternalPaths, _, cx| {
                        view.drag_over_composer.set(false);
                        view.attach_from_paths(paths.paths(), cx);
                    }))
                    .on_drag_move::<ExternalPaths>(cx.listener(
                        |view, event: &gpui::DragMoveEvent<ExternalPaths>, _, cx| {
                            let inside = event.bounds.contains(&event.event.position);
                            if view.drag_over_composer.get() != inside {
                                view.drag_over_composer.set(inside);
                                cx.notify();
                            }
                        },
                    ))
                    .drag_over::<ExternalPaths>(|style, _, _, cx| {
                        style.border_color(cx.theme().drag_border)
                    })
            })
            .when(!self.composer_attachments.is_empty(), |composer| {
                composer.child(self.render_attachment_chips(cx))
            })
            .when_some(self.composer_image_error.clone(), |composer, error| {
                composer.child(div().mb_1().child(Alert::error("image-error", error)))
            })
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
                        // Icon-only attach affordance: a `+` glyph sitting in
                        // a square 40x40 hit box (matches SEND_BUTTON_HEIGHT
                        // so the two controls read as one row). The tooltip
                        // spells the full attach surface — click, drag, and
                        // paste all reach the same batch.
                        Button::new("attach")
                            .debug_selector(|| "attach-button".into())
                            .ghost()
                            .compact()
                            .icon(IconName::Plus)
                            .accessibility_label(chrome::ATTACH_ICON_LABEL)
                            .tooltip(chrome::ATTACH_HINT)
                            .disabled(!can_send)
                            .h(theme::SEND_BUTTON_HEIGHT)
                            .w(theme::SEND_BUTTON_HEIGHT)
                            .on_click(cx.listener(|view, _, window, cx| {
                                view.attach_from_files(window, cx)
                            })),
                    )
                    .child(if can_send {
                        // Send now paints as the primary/accent action —
                        // the composer's one bold surface. Sized to its
                        // label with a modest floor so `Send` and, when
                        // Kit paints a loading state, the spinner still
                        // sit inside; paired with the 40x40 attach hit
                        // area on the same row for a clean action row.
                        Button::new("send")
                            .debug_selector(|| "send-button".into())
                            .primary()
                            .label("Send")
                            .h(theme::SEND_BUTTON_HEIGHT)
                            .min_w(theme::SEND_BUTTON_MIN_WIDTH)
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .on_click(cx.listener(|view, _, _, cx| view.send_composer(cx)))
                            .into_any_element()
                    } else {
                        // Disabled: paint the outline ourselves. Kit's
                        // primary variant derives its border from its
                        // fill, so a transparent fill kills the border
                        // too. A plain div sets fill and border
                        // independently, with the whole presentation
                        // dimmed to 0.55 opacity per contract line 85.
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
            // Composer footer: model on the left, kb hint on the right.
            // Both quiet — the composer's role is the input row above;
            // the footer only carries what the operator wants to glance
            // at (the model this send will route to) plus the shortcut
            // reminder. Proximity: metadata sits next to what it
            // describes (laws-of-ux).
            .child(
                div()
                    .h_flex()
                    .items_center()
                    .justify_between()
                    .gap_2()
                    .w_full()
                    .mt_1()
                    .h(theme::COMPOSER_TARGET_HEIGHT)
                    .text_size(theme::label_small(cx.theme().font_size))
                    .debug_selector(|| "composer-footer".into())
                    .child(
                        div()
                            .h_flex()
                            .items_center()
                            .gap_1()
                            .min_w_0()
                            .max_w(theme::COMPOSER_TARGET_MAX_WIDTH)
                            .truncate()
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
                    .child(
                        div()
                            .flex_shrink_0()
                            .text_color(cx.theme().muted_foreground)
                            .debug_selector(|| "composer-hint".into())
                            .child(composer_hint),
                    ),
            )
            .when(drag_active, |composer| {
                composer.child(self.render_drop_target(cx))
            })
            .into_any_element()
    }

    /// Pending-chip row. One chip per `composer_attachments` slot: valid
    /// entries paint a thumbnail, name, size, and
    /// a remove button; invalid entries paint an error icon, name, and the
    /// inline error message. Every dimension routes through theme tokens so
    /// the whole row scales with the appearance picker.
    fn render_attachment_chips(&self, cx: &Context<Self>) -> gpui::AnyElement {
        div()
            .mb_1()
            .h_flex()
            .flex_wrap()
            .gap_2()
            .debug_selector(|| "composer-chip-row".into())
            .children(self.composer_attachments.iter().enumerate().map(
                |(index, item)| match item {
                    PendingAttachment::Valid { image, thumbnail } => {
                        self.render_valid_attachment_chip(index, image, thumbnail.clone(), cx)
                    }
                    PendingAttachment::Invalid { name, error } => {
                        self.render_invalid_attachment_chip(index, name, error, cx)
                    }
                },
            ))
            .into_any_element()
    }

    fn render_valid_attachment_chip(
        &self,
        index: usize,
        image: &ImageAttachment,
        thumbnail: std::sync::Arc<gpui::Image>,
        cx: &Context<Self>,
    ) -> gpui::AnyElement {
        // Thumbnail area: token-sized rectangle with the same subtle
        // black/white outline the transcript thumbnails use, so the pending
        // chip and the sent-turn thumbnail read as one family. `Valid`
        // attachments always carry a decoded thumbnail — a decode failure
        // demotes the entry to `Invalid` at attach time, so this branch never
        // has to reason about a missing preview.
        let base_size = cx.theme().font_size;
        let (thumb_w, thumb_h) = theme::chip_thumbnail_size(base_size);
        let control_size = theme::chip_control_size(base_size);
        let padding_y = theme::chip_padding_y(base_size);
        let label_max = theme::chip_label_max_width(base_size);
        let thumb_border = if cx.theme().is_dark() {
            gpui::white()
        } else {
            gpui::black()
        }
        .opacity(0.1);
        let name = image.name.clone();
        let size_label = polish::format_bytes(image.size);
        // Chip name text routes through `record_state` so the appearance
        // themes test (per ThemeId::ALL) can assert the RENDERED text color
        // at draw time, not the palette-field it points at — a mutation that
        // paints the label with the fill color would slip past a token-only
        // check but fail the render_log sample.
        self.chip_frame(padding_y, cx)
            .debug_selector(|| "composer-chip".into())
            .child(
                gpui::img(thumbnail)
                    .w(thumb_w)
                    .h(thumb_h)
                    .object_fit(gpui::ObjectFit::Cover)
                    .border_1()
                    .border_color(thumb_border)
                    .debug_selector(move || format!("composer-chip-thumbnail-{index}")),
            )
            .child(
                div()
                    .flex()
                    .flex_col()
                    .min_w_0()
                    .max_w(label_max)
                    .child(
                        div()
                            .truncate()
                            .text_size(theme::label_small(base_size))
                            .text_color(record_state(
                                || format!("chip-name-{index}"),
                                cx.theme().foreground,
                            ))
                            .child(name),
                    )
                    .child(
                        div()
                            .text_size(theme::label_micro(base_size))
                            .text_color(cx.theme().muted_foreground)
                            .child(size_label),
                    ),
            )
            .child(self.chip_remove_button(index, control_size, cx))
            .into_any_element()
    }

    fn render_invalid_attachment_chip(
        &self,
        index: usize,
        name: &str,
        error: &str,
        cx: &Context<Self>,
    ) -> gpui::AnyElement {
        // Error chip: same frame as a valid chip so the row rhythm holds,
        // but a danger-tinted glyph + the parser's per-file error message in
        // place of the thumbnail + byte-size line. Never included in Send.
        //
        // Accessibility: the chip advertises the `Alert` role so screen
        // readers announce it as a live error, an `aria_label` that pairs
        // the filename with the full error text (the visible label truncates
        // on narrow chips), and a tooltip carrying the same full error text
        // for sighted users who hover a truncated chip.
        let base_size = cx.theme().font_size;
        let (thumb_w, thumb_h) = theme::chip_thumbnail_size(base_size);
        let control_size = theme::chip_control_size(base_size);
        let padding_y = theme::chip_padding_y(base_size);
        let label_max = theme::chip_label_max_width(base_size);
        let aria_label: gpui::SharedString = format!("Attachment error: {name} — {error}").into();
        let tooltip_text: gpui::SharedString = format!("{name}: {error}").into();
        self.chip_frame(padding_y, cx)
            .id(("composer-chip-error", index))
            .test_support()
            .role(gpui::Role::Alert)
            .aria_label(aria_label)
            .tooltip(move |window, cx| {
                gpui_kit::component::tooltip::Tooltip::new(tooltip_text.clone()).build(window, cx)
            })
            .debug_selector(move || format!("composer-chip-error-{index}"))
            .child(
                div()
                    .w(thumb_w)
                    .h(thumb_h)
                    .flex()
                    .items_center()
                    .justify_center()
                    .bg(theme::palette::danger_tint())
                    .border_1()
                    .border_color(cx.theme().border)
                    .text_color(cx.theme().foreground)
                    .child(Icon::new(IconName::TriangleAlert).size(theme::label_micro(base_size))),
            )
            .child(
                div()
                    .flex()
                    .flex_col()
                    .min_w_0()
                    .max_w(label_max)
                    .child(
                        div()
                            .truncate()
                            .text_size(theme::label_small(base_size))
                            .text_color(cx.theme().foreground)
                            .child(name.to_string()),
                    )
                    .child(
                        div()
                            .truncate()
                            .text_size(theme::label_micro(base_size))
                            .text_color(cx.theme().foreground)
                            .child(error.to_string()),
                    ),
            )
            .child(self.chip_remove_button(index, control_size, cx))
            .into_any_element()
    }

    fn chip_frame(&self, padding_y: gpui::Pixels, cx: &Context<Self>) -> gpui::Div {
        // Shared frame — variant-specific debug selectors (`composer-chip`
        // for valid, `composer-chip-error-{index}` for invalid) live on the
        // call sites so tests can differentiate the two paths.
        div()
            .h_flex()
            .items_center()
            .gap_2()
            .pl_1()
            .pr_1()
            .py(padding_y)
            .bg(cx.theme().muted)
            .border_1()
            .border_color(cx.theme().border)
    }

    fn chip_remove_button(
        &self,
        index: usize,
        size: gpui::Pixels,
        cx: &Context<Self>,
    ) -> impl IntoElement {
        Button::new(("chip-remove", index))
            .debug_selector(move || format!("chip-remove-{index}"))
            .ghost()
            .compact()
            .icon(IconName::Close)
            .accessibility_label(chrome::CHIP_REMOVE_LABEL)
            .tooltip(chrome::CHIP_REMOVE_LABEL)
            .h(size)
            .w(size)
            .on_click(cx.listener(move |view, _, _, cx| view.remove_attached_image(index, cx)))
    }

    /// Drop-target overlay painted while a file drag is active over the
    /// composer. Sits absolutely over the composer bounds so the input
    /// underneath still stops the drag from falling through to the
    /// transcript, and the whole layer reads as one flat drop zone rather
    /// than a per-cell border flicker.
    fn render_drop_target(&self, cx: &Context<Self>) -> gpui::AnyElement {
        let base_size = cx.theme().font_size;
        div()
            .absolute()
            .inset_0()
            .flex()
            .flex_col()
            .items_center()
            .justify_center()
            .gap_1()
            .bg(cx.theme().drop_target)
            .border_1()
            .border_dashed()
            .border_color(cx.theme().drag_border)
            .debug_selector(|| "composer-drop-target".into())
            .child(
                div()
                    .text_size(theme::body(base_size))
                    .text_color(cx.theme().foreground)
                    .font_weight(gpui::FontWeight::SEMIBOLD)
                    .child(chrome::DROP_TARGET_TITLE),
            )
            .child(
                div()
                    .text_size(theme::label_small(base_size))
                    .text_color(cx.theme().muted_foreground)
                    .child(chrome::DROP_TARGET_HINT),
            )
            .into_any_element()
    }

    fn render_run_header(&self, cx: &App) -> gpui::AnyElement {
        // Single-row run header (ZETA-123): session title on the left, a
        // compact metadata cluster on the right — quiet tokens/cache, a
        // dot + state word (glyph, not a filled pill), and the model
        // name. The keyboard hint that used to sit here now lives in the
        // composer footer, next to the send controls, so meta stays
        // close to what it describes (laws-of-ux: Proximity).
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
        let dot_color = self.status_dot_color(cx);
        let mode_word = self.footer_mode_word();
        let busy = self.state.streaming || self.state.thinking;
        let model_name = self
            .state
            .metrics
            .model
            .clone()
            .unwrap_or_else(|| "—".into());
        let base_size = cx.theme().font_size;
        div()
            .id("run-header")
            .debug_selector(|| "run-header".into())
            .h_flex()
            .items_center()
            .flex_shrink_0()
            .gap(px(10.))
            .min_h(theme::HEADER_BAND1_MIN_HEIGHT)
            .py(px(7.))
            .px(px(14.))
            .border_b_1()
            .border_color(cx.theme().border)
            .text_size(theme::label_small(base_size))
            .child(
                // Session title — the primary identity of the run. Grows
                // to eat leftover space so the metadata cluster always
                // hugs the right edge; truncates last when the window
                // narrows because the cluster below shrinks first.
                //
                // No `max_w` cap: with `flex_1` (grow=1, shrink=1,
                // basis=0) the title expands to fill leftover and the
                // cluster hugs the right edge even when the model name
                // is short — a max_w cap would leave dead space between
                // title and metadata and the cluster would float left
                // in the middle of the header.
                //
                // `min_w(px(80.))` keeps a scannable measure of the
                // title on every window width — without it, a title
                // with `flex-basis: 0` collapses to zero when the
                // cluster's shrinkable siblings still add up to more
                // than the row can hold (round-2 finding 2 repro at
                // 760px).
                div()
                    .debug_selector(|| "run-header-title".into())
                    .flex_1()
                    .min_w(theme::HEADER_TITLE_MIN_WIDTH)
                    .truncate()
                    .text_size(theme::title(base_size))
                    .font_weight(gpui::FontWeight::SEMIBOLD)
                    .text_color(cx.theme().foreground)
                    .child(session_label),
            )
            // Right-anchored metadata cluster. The cluster itself carries
            // `min_w_0` so its shrinkable children (tokens, model) can
            // truncate at narrow widths BEFORE the title has to. The
            // status dot + word keeps `flex_shrink_0` — it's the header's
            // single load-bearing signal and must not collapse.
            .child(
                div()
                    .h_flex()
                    .items_center()
                    .gap(px(10.))
                    .min_w_0()
                    .child(
                        // Tokens/cache — quietest metadata. Shrinks and
                        // truncates first when width drops (higher
                        // shrink factor than the model slot).
                        div()
                            .flex_shrink(2.0)
                            .min_w_0()
                            .max_w(theme::HEADER_STATUS_METRICS_MAX_WIDTH)
                            .truncate()
                            .text_color(theme::palette::text_faint())
                            .debug_selector(|| "status-metrics".into())
                            .child(polish::status_label(&self.state.metrics)),
                    )
                    .child(status_rule(cx))
                    .child(
                        // Compact state indicator: dot + word. Neutral
                        // states paint the dot in the accent hue;
                        // offline paints it in danger. When the
                        // assistant is streaming or thinking the SAME
                        // dot pulses so the header carries ONE dot per
                        // state, not a pair — the reader looks at one
                        // place to know what's happening (Selective
                        // Attention). The `footer-mode` selector stays
                        // so existing offline / pill-fill guards
                        // continue to bind here.
                        div()
                            .debug_selector(|| "footer-mode".into())
                            .flex_shrink_0()
                            .h_flex()
                            .items_center()
                            .gap(theme::HEADER_MODE_GAP)
                            .text_color(cx.theme().muted_foreground)
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .child(status_dot(dot_color, busy))
                            .child(mode_word),
                    )
                    .child(status_rule(cx))
                    .child(
                        // Model chip pinned right — same faint tier as
                        // the metrics slot so the two read as one
                        // metadata run. Shrinks and truncates before
                        // the title does; caps at
                        // HEADER_MODEL_MAX_WIDTH so a very long model
                        // id truncates inside the chip.
                        div()
                            .flex_shrink(1.0)
                            .max_w(theme::HEADER_MODEL_MAX_WIDTH)
                            .min_w_0()
                            .truncate()
                            .text_color(theme::palette::text_faint())
                            .debug_selector(|| "run-header-model".into())
                            .child(model_name),
                    ),
            )
            .into_any_element()
    }

    /// Colour for the run-header status dot. Offline paints the dot in
    /// danger — the scarce, load-bearing alarm signal; every other
    /// state paints in the accent hue.
    fn status_dot_color(&self, cx: &App) -> gpui::Hsla {
        match &self.state.connection {
            ConnectionState::Lost(_) => cx.theme().danger,
            _ => cx.theme().primary,
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

/// Text-run geometry recorder. Parallel to the ZETA-107/108 `render_log`
/// color recorder, but captures WRAP GEOMETRY. `painted_quads()` reports
/// rectangles only — background quads, borders, rails; laid-out glyphs
/// paint as sprite primitives that no test accessor exposes. That means
/// a wrapping defect where the text system produces a line whose glyphs
/// shape past the row's inner text column is INVISIBLE to painted_quads.
/// The r0-r2 fix rounds for the ZETA-124 orphan-glyph defect ran blind
/// against exactly that gap.
///
/// The recorder closes it by shaping the source text through
/// `cx.text_system().shape_text` with the same wrap width the renderer
/// hands to the text system, and recording the maximum wrap-line width
/// the shaping produced. A caller that chose a wrap width larger than
/// the column's content box widens `max_line_width` past that column;
/// a caller that chose one smaller shows a `max_line_width` under the
/// column — both are observable failures at the call site.
///
/// The record path is gated behind `cfg(any(test, feature = "smoke-test"))`
/// exactly like `render_log`, so the release-build cost is zero: the
/// hook site in `render_assistant_row` compiles out entirely and the
/// module isn't linked.
#[cfg(any(test, feature = "smoke-test"))]
pub(crate) fn record_text_geometry<F>(
    cx: &App,
    row_id: F,
    source: &str,
    font: gpui::Font,
    font_size: gpui::Pixels,
    wrap_width: gpui::Pixels,
) where
    F: FnOnce() -> String,
{
    let text_system = cx.text_system().clone();
    // `cx.text_system()` returns `Arc<TextSystem>`, which does not own the
    // line-layout cache `shape_text` / `shape_line` need. Every call site
    // of shaping in gpui goes through `WindowTextSystem` — the layout
    // cache lives on the window. `WindowTextSystem::new(text_system)`
    // produces a one-shot layout system that shares the crate-level
    // fonts / metrics and paints identically at the same wrap_width,
    // which is what we need for the recorder: same wrap decisions as
    // the live TextView.
    let window_text_system = gpui::WindowTextSystem::new(text_system.clone());
    let mut max_unwrapped_line_width = gpui::px(0.);
    let mut max_wrap_segment_width = gpui::px(0.);
    let mut wrap_segment_count = 0usize;
    for line_text in source.split('\n') {
        if line_text.is_empty() {
            wrap_segment_count += 1;
            continue;
        }
        let run = gpui::TextRun {
            len: line_text.len(),
            font: font.clone(),
            color: gpui::black(),
            background_color: None,
            underline: None,
            strikethrough: None,
        };
        let line_shared: gpui::SharedString = line_text.to_owned().into();
        let shaped =
            window_text_system.shape_line(line_shared, font_size, std::slice::from_ref(&run), None);
        let unwrapped_width = shaped.width();
        if unwrapped_width > max_unwrapped_line_width {
            max_unwrapped_line_width = unwrapped_width;
        }
        // Compute per-wrap-segment widths using LineWrapper on the same
        // byte offsets, then map each segment's [start_ix..end_ix] to
        // `LineLayout::x_for_index` on the shaped line's unwrapped layout.
        // That gives the SHAPED extent of each wrap segment — the exact
        // measurement `painted_quads()` cannot see for glyphs. A segment
        // whose extent exceeds `wrap_width` is an unbreakable token
        // wider than the column, or a shape-vs-wrap divergence in the
        // caller — both are the class of defect this seam exists to
        // catch.
        let mut handle = text_system.line_wrapper(font.clone(), font_size);
        let mut prev_ix: usize = 0;
        let boundaries: Vec<_> = handle
            .wrap_line(&[gpui::LineFragment::text(line_text)], wrap_width)
            .collect();
        for boundary in &boundaries {
            wrap_segment_count += 1;
            let end_ix = boundary.ix;
            let seg_x_start = shaped.x_for_index(prev_ix);
            let seg_x_end = shaped.x_for_index(end_ix);
            let seg_width = if seg_x_end > seg_x_start {
                seg_x_end - seg_x_start
            } else {
                gpui::px(0.)
            };
            if seg_width > max_wrap_segment_width {
                max_wrap_segment_width = seg_width;
            }
            prev_ix = end_ix;
        }
        // Trailing segment from the last boundary to end of line.
        wrap_segment_count += 1;
        let tail_x_start = shaped.x_for_index(prev_ix);
        let tail_width = if unwrapped_width > tail_x_start {
            unwrapped_width - tail_x_start
        } else {
            gpui::px(0.)
        };
        if tail_width > max_wrap_segment_width {
            max_wrap_segment_width = tail_width;
        }
    }
    text_run_log::record(text_run_log::Sample {
        row_id: row_id(),
        wrap_width,
        max_unwrapped_line_width,
        max_wrap_segment_width,
        wrap_segment_count,
        source_len: source.len(),
    });
}

#[cfg(any(test, feature = "smoke-test"))]
pub(crate) mod text_run_log {
    use gpui::Pixels;
    use std::cell::RefCell;

    #[derive(Debug, Clone)]
    pub(crate) struct Sample {
        pub(crate) row_id: String,
        pub(crate) wrap_width: Pixels,
        pub(crate) max_unwrapped_line_width: Pixels,
        pub(crate) max_wrap_segment_width: Pixels,
        pub(crate) wrap_segment_count: usize,
        pub(crate) source_len: usize,
    }

    thread_local! {
        static SAMPLES: RefCell<Vec<Sample>> = const { RefCell::new(Vec::new()) };
    }

    pub(crate) fn clear() {
        SAMPLES.with(|slot| slot.borrow_mut().clear());
    }

    pub(crate) fn record(sample: Sample) {
        SAMPLES.with(|slot| slot.borrow_mut().push(sample));
    }

    pub(crate) fn samples() -> Vec<Sample> {
        SAMPLES.with(|slot| slot.borrow().clone())
    }
}

/// Modal title band: title-tier semibold on the left, a small-label `esc`
/// hint at the right. Contract line 91 pins this shape for every wiki-run
/// modal; the title role rides the same +2 step every promoted header takes.
///
/// The `esc` hint routes through the theme's `muted_foreground` role rather
/// than the `text_faint` palette accessor — `text_faint` sat at 2.92-4.03:1
/// against the modal panel (WCAG AA needs 4.5:1 for small text) across the
/// five shipped themes; `muted_foreground` clears AA on every theme (see
/// the modal-hint contrast row in `settings_panel_paints_on_tokens_...`).
pub(crate) fn modal_title(title: &'static str, cx: &App) -> gpui::AnyElement {
    let base = cx.theme().font_size;
    div()
        .debug_selector(|| "modal-title".into())
        .h_flex()
        .items_center()
        .justify_between()
        .w_full()
        .child(
            div()
                .text_size(theme::title(base))
                .font_weight(gpui::FontWeight::SEMIBOLD)
                .child(title),
        )
        .child(
            div()
                .debug_selector(|| "modal-title-esc".into())
                .text_size(theme::label_small(base))
                .text_color(cx.theme().muted_foreground)
                .child("esc"),
        )
        .into_any_element()
}

/// Open a Settings section. Returns a `Div` seeded with the section header
/// (body-tier semibold on the panel foreground, one step above field labels
/// so the section reads as a heading) and a thin separator rule. The caller
/// appends its rows with `.child(...)`. Every section paints on tokens so
/// the five themes stay legible.
///
/// The container carries `.track_focus(focus)` so `settings_sections_scroll`
/// can call `scroll_to_item` when Tab lands on a control inside the section
/// — the safety net that keeps focus rings inside the viewport at 18px.
pub(crate) fn settings_section(
    selector: &'static str,
    heading: &'static str,
    focus: &gpui::FocusHandle,
    cx: &App,
) -> gpui::Div {
    let base = cx.theme().font_size;
    div()
        .debug_selector(move || selector.into())
        .track_focus(focus)
        .v_flex()
        .gap(theme::SETTINGS_ROW_GAP)
        .child(
            div()
                .h_flex()
                .items_center()
                .gap_2()
                .child(
                    div()
                        .debug_selector(move || format!("{selector}-heading"))
                        .text_size(theme::body(base))
                        .font_weight(gpui::FontWeight::SEMIBOLD)
                        .text_color(cx.theme().foreground)
                        .child(heading),
                )
                .child(
                    div()
                        .flex_1()
                        .h(theme::RAIL_WIDTH_THIN)
                        .bg(cx.theme().border),
                ),
        )
}

/// One labeled Settings row. `label` sits in a token-derived left column
/// that widens with the base font size (so "Approval mode" fits at every
/// picker base without wrapping); `control` is right-aligned. Every row in
/// every section flows through this so the modal has ONE row anatomy.
///
/// `description` is optional — when present, it paints as a small-label
/// muted caption on the row below the control, aligned under the label. The
/// muted-foreground token stays legible on the sidebar panel across all
/// five themes (the `esc` hint uses the same token — see modal_title).
pub(crate) fn settings_row(
    selector: &'static str,
    label: &'static str,
    description: Option<&'static str>,
    control: impl gpui::IntoElement,
    cx: &App,
) -> gpui::Div {
    let base = cx.theme().font_size;
    let column = theme::settings_label_column(base);
    let header = div()
        .debug_selector(move || format!("{selector}-header"))
        .h_flex()
        .items_center()
        .justify_between()
        .gap_3()
        .child(
            div()
                .debug_selector(move || format!("{selector}-label"))
                .w(column)
                .flex_shrink_0()
                .text_size(theme::body(base))
                .text_color(cx.theme().foreground)
                .child(label),
        )
        .child(
            div()
                .debug_selector(move || format!("{selector}-control"))
                .flex_1()
                .h_flex()
                .justify_end()
                .child(control),
        );
    let mut row = div()
        .debug_selector(move || selector.into())
        .v_flex()
        .gap(theme::SETTINGS_ROW_DESCRIPTION_GAP)
        .child(header);
    if let Some(text) = description {
        row = row.child(
            div()
                .debug_selector(move || format!("{selector}-description"))
                .w(column)
                .text_size(theme::label_small(base))
                .text_color(cx.theme().muted_foreground)
                .child(text),
        );
    }
    row
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

/// Header status dot. Colour carries the run state (accent = neutral,
/// danger = offline) and — when `busy` is true because the assistant is
/// streaming or thinking — the SAME dot pulses its opacity in a synced
/// 1.2s loop. One dot per state keeps the header signal in one place
/// (Selective Attention / Von Restorff) and matches the wiki agent-run
/// indicator that only ever paints a single breathing glyph.
fn status_dot(color: gpui::Hsla, busy: bool) -> gpui::AnyElement {
    let base = div()
        .debug_selector(|| "run-header-status-dot".into())
        .w(theme::STREAM_DOT_SIZE)
        .h(theme::STREAM_DOT_SIZE)
        .rounded_full()
        .bg(color);
    if busy {
        base.with_animation(
            "status-dot-busy",
            Animation::new(Duration::from_millis(1200))
                .repeat_synced()
                .with_easing(ease_in_out),
            |el, delta| {
                // Delta 0..1: triangle wave 0..1..0 so the dot breathes
                // up then down without the pop-back a sawtooth would show.
                let triangle = 1.0 - (delta * 2.0 - 1.0).abs();
                let alpha = 0.25 + triangle * 0.75;
                el.opacity(alpha)
            },
        )
        .into_any_element()
    } else {
        base.into_any_element()
    }
}

impl Render for ZetaView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // Reset the state-color recorder at the start of every render so
        // tests observe only the samples produced by the draw they trigger,
        // and no test needs a manual `render_log::clear()` before drawing.
        // The text-run geometry recorder resets on the same beat so a
        // wrap-containment test reads only this frame's shaped lines.
        // Compiled out in production alongside the recorders themselves.
        #[cfg(any(test, feature = "smoke-test"))]
        {
            render_log::clear();
            text_run_log::clear();
        }
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
                            self.render_login_provider(
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
                                            self.render_login_provider(
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
            .font_family(theme::current_font_family())
            .text_size(theme::body(theme::current_font_size()))
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
            // Load persisted appearance BEFORE opening the window so the
            // first frame paints on the user's picked palette / font. A
            // missing / corrupt prefs file resolves to the shipped default
            // through `prefs::load`.
            theme::apply_with(cx, &prefs::load());
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
