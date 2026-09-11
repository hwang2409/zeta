//! Typed model of every user-visible string a transcript row can paint.
//!
//! One struct per row kind whose fields NAME every visible string the row
//! renders — body text, action/button labels, attachment names and sizes,
//! tool output sizes, chevrons/hints, and fixed chrome. The renderers in
//! `transcript_render.rs` build their visible-text elements by destructuring
//! this model; no visible string is sourced outside it. Together with the
//! `renderer_literal_fence` AST guard — which parses the transcript render
//! module and rejects ANY string or byte-string literal that does not sit
//! inside the tight allowed-context allowlist — the two make a NEW stray
//! literal impossible to add without failing a test.
//!
//! The typed model REPLACES the earlier positional
//! `TranscriptEntry::visible_text -> Vec<&str>` seam. That seam covered row
//! CONTENT only; action labels, attachment names/sizes, tool output sizes,
//! AND the entire login-row surface still flowed from entry fields via
//! inline `format!` calls in the render layer, so a new stray literal on
//! any of those paths bypassed the seam. The typed model closes those
//! paths — the login row now flows through `LoginRowText` too.
//!
//! Borrowing: content-shaped fields (`content`, `source`, `verb`, `detail`,
//! `body`, `message`) hold a `&str` into the transcript entry, so a per-row
//! `build()` allocates only for the composed labels (attachment lines,
//! output-size string, login labels). A per-frame full-transcript clone is
//! not introduced — this matches the ZETA-107 r5 borrowing invariant.

use crate::login::{LoginProgress, LoginProvider};
use crate::session::SessionView;
use crate::state::{ConnectionState, TranscriptEntry, ERROR_HEADER_LABEL, THINKING_HEADER_LABEL};

/// Every fixed literal painted as row chrome (headings, hints, unit
/// suffixes, action labels). Every user-visible string that reaches a row
/// but does not come from an entry field lives here. `ALL` is the exhaustive
/// set the guard tests iterate.
pub mod chrome {
    pub const ASSISTANT_TRUNCATED: &str = "Showing the latest streamed text…";
    pub const TOOL_HOVER_HINT: &str = "show output";
    pub const TOOL_TAIL_OMITTED: &str = "Earlier output omitted";
    pub const OUTPUT_SIZE_UNIT_B: &str = "B";
    pub const OUTPUT_SIZE_UNIT_KB: &str = "KB";
    pub const OUTPUT_SIZE_UNIT_MB: &str = "MB";
    pub const ATTACHMENT_SIZE_SEPARATOR: &str = " · ";
    pub const ATTACHMENT_SIZE_SUFFIX: &str = " bytes";
    pub const FORK_HERE: &str = "Fork here";
    pub const OPEN_SETTINGS: &str = "Open Settings";
    pub const LOGIN_START_PREFIX: &str = "Log in with ";
    pub const LOGIN_CANCEL: &str = "Cancel";

    /// Exhaustive set the guard tests iterate. Adding a new chrome literal
    /// without adding it here fails the fence's chrome-coverage check.
    pub const ALL: &[&str] = &[
        ASSISTANT_TRUNCATED,
        TOOL_HOVER_HINT,
        TOOL_TAIL_OMITTED,
        OUTPUT_SIZE_UNIT_B,
        OUTPUT_SIZE_UNIT_KB,
        OUTPUT_SIZE_UNIT_MB,
        ATTACHMENT_SIZE_SEPARATOR,
        ATTACHMENT_SIZE_SUFFIX,
        FORK_HERE,
        OPEN_SETTINGS,
        LOGIN_START_PREFIX,
        LOGIN_CANCEL,
    ];
}

/// Widget-ID / debug-selector composers. Every selector the render module
/// paints is produced by one of these helpers, so the render module itself
/// has no bare `format!` fragments in its bodies — the AST fence stays
/// strict on "no literal outside debug_selector/id/aria_label/role args".
pub mod sel {
    pub const TRANSCRIPT_ROW: &str = "transcript-row";
    pub const ATTACHMENT_CHIP: &str = "attachment-chip";
    pub const FORK_BUTTON_TAG: &str = "fork";
    pub const TOOL_RECEIPT_TAG: &str = "tool-receipt";
    pub const ERROR_SETTINGS_TAG: &str = "error-settings";

    pub fn thinking_header(i: usize) -> String {
        format!("thinking-header-{i}")
    }
    pub fn user_row_group(i: usize) -> String {
        format!("user-row-{i}")
    }
    pub fn fork_button(i: usize) -> String {
        format!("fork-button-{i}")
    }
    pub fn message(i: usize) -> String {
        format!("message-{i}")
    }
    pub fn tool_row_group(i: usize) -> String {
        format!("tool-row-{i}")
    }
    pub fn tool_receipt(i: usize) -> String {
        format!("tool-receipt-{i}")
    }
    pub fn tool_chevron(i: usize) -> String {
        format!("tool-chevron-{i}")
    }
    pub fn tool_verb(i: usize) -> String {
        format!("tool-verb-{i}")
    }
    pub fn tool_detail(i: usize) -> String {
        format!("tool-detail-{i}")
    }
    pub fn tool_output(i: usize) -> String {
        format!("tool-output-{i}")
    }
    pub fn error_block(i: usize) -> String {
        format!("error-block-{i}")
    }
    pub fn error_message(i: usize) -> String {
        format!("error-message-{i}")
    }
    pub fn error_settings(i: usize) -> String {
        format!("error-settings-{i}")
    }
    pub fn error_login_prefix(i: usize) -> String {
        format!("error-login-{i}")
    }
    pub fn login_base_id(prefix: &str, provider: &str) -> String {
        format!("{prefix}-{provider}")
    }
    pub fn login_start(base: &str) -> String {
        format!("{base}-start")
    }
    pub fn login_cancel(base: &str) -> String {
        format!("{base}-cancel")
    }
    pub fn login_error(base: &str) -> String {
        format!("{base}-error")
    }
}

/// Typed model of the user text a single transcript row paints. One variant
/// per `TranscriptEntry` kind. Fields are named after the visible role each
/// string plays; the renderer destructures the struct and consumes every
/// field — an unused field trips clippy's `unused_variables` under the
/// crate's `deny(warnings)` lint, so a renderer that stops painting a field
/// fails to build.
#[derive(Debug, Clone)]
pub enum RowText<'a> {
    User(UserRowText<'a>),
    Assistant(AssistantRowText<'a>),
    Thinking(ThinkingRowText),
    Tool(ToolRowText<'a>),
    Error(ErrorRowText<'a>),
}

/// User row: the prompt body plus optional attachment chips and the
/// hover-revealed fork action. The attachment field is tri-state on purpose:
/// text-only history rows carry `Some(vec![])` (attachment key present but
/// empty) while pre-history rows carry `None`; the old positional seam
/// collapsed both to "empty", which changed the row height when a user turn
/// restored from history — the empty container's `mt_1` margin disappeared.
#[derive(Debug, Clone)]
pub struct UserRowText<'a> {
    /// The prompt body — the row's only dynamic body string.
    pub content: &'a str,
    /// `Some(list)` when the session view has an attachment entry for this
    /// index — even if `list` is empty. The renderer paints the mt_1
    /// container whenever the value is `Some`, matching the pre-typed-seam
    /// row height for text-only history rows. `None` skips the container.
    pub attachments: Option<Vec<String>>,
    /// `Some(chrome::FORK_HERE)` when the row is fork-eligible; `None`
    /// otherwise (no button paints).
    pub fork_label: Option<&'static str>,
}

/// Assistant row: the rendered markdown plus the truncated-preview hint.
#[derive(Debug, Clone)]
pub struct AssistantRowText<'a> {
    /// The rendered markdown source — the row's only dynamic body string.
    pub source: &'a str,
    /// `Some(chrome::ASSISTANT_TRUNCATED)` when the streamed preview was
    /// truncated; `None` when the full text is on screen.
    pub truncated_hint: Option<&'static str>,
}

/// Thinking row: the generic header. The provider protocol carries no
/// display-safe summary channel, so the row never paints body text.
#[derive(Debug, Clone)]
pub struct ThinkingRowText {
    /// Exactly `THINKING_HEADER_LABEL`. A sentinel-carrying reasoning
    /// payload contributes NOTHING to this field.
    pub header: &'static str,
}

/// Tool row: the receipt line plus optional collapsed peek and expanded
/// body. Every state variant flows through the same model — running, done,
/// failed, canceled — because state is signalled by COLOR only, not text.
#[derive(Debug, Clone)]
pub struct ToolRowText<'a> {
    /// Tool name — the leading semibold verb.
    pub verb: &'a str,
    /// One-line argument summary — the dim detail after the verb.
    pub detail: &'a str,
    /// `Some(...)` while the row is collapsed AND the tail carries bytes:
    /// the compact byte-count peek. `None` when expanded or empty.
    pub output_size_label: Option<String>,
    /// `Some(chrome::TOOL_HOVER_HINT)` under the same condition as
    /// `output_size_label`; `None` otherwise.
    pub hover_hint: Option<&'static str>,
    /// `Some(chrome::TOOL_TAIL_OMITTED)` when the expanded body was
    /// clipped; `None` when the tail is complete or the row is collapsed.
    pub tail_omitted_hint: Option<&'static str>,
    /// The expanded output body. `Some(&tail)` when the row is expanded,
    /// `None` when collapsed.
    pub body: Option<&'a str>,
}

/// Error row: the header, the message body, and the optional settings
/// recovery button.
#[derive(Debug, Clone)]
pub struct ErrorRowText<'a> {
    /// Exactly `ERROR_HEADER_LABEL`.
    pub header: &'static str,
    /// The provider's error text.
    pub message: &'a str,
    /// `Some(chrome::OPEN_SETTINGS)` when the recovery button should
    /// paint (settings_action AND session_view.available); `None`
    /// otherwise.
    pub settings_action_label: Option<&'static str>,
}

/// Typed model of the login-row surface. Rendered from four call sites in
/// the app — settings overlay, inline error recovery, in-progress banner,
/// first-conversation prompt — every one of which paints the same six
/// visible strings. Resolving the provider label BEFORE render kills the
/// "raw provider text handed to the renderer" bypass the r1 review found.
#[derive(Debug, Clone)]
pub struct LoginRowText {
    /// Outer widget id — used for `.id(...)` and as the base for the
    /// button selectors below. Composed once in `build_login`, never in
    /// the render body.
    pub outer_id: String,
    /// Provider header label — dynamic ("Claude", "ChatGPT", ...).
    pub header_label: String,
    /// Provider slug — carried alongside the label so the click callback
    /// can identify the provider without seeing the visible text. Not
    /// painted as visible text — it is the opaque handle passed to
    /// `start_login`/`cancel_login`.
    pub provider_slug: String,
    /// Start button model — id + composed label + disabled flag.
    pub start: LoginActionText,
    /// Cancel button model — present only while the login is busy.
    pub cancel: Option<LoginActionText>,
    /// Status text — one of a fixed set of `&'static str` values sourced
    /// from `LoginProvider::status()`.
    pub status_text: &'static str,
    /// Error alert model — present when the last attempt failed.
    pub error: Option<LoginErrorText>,
}

/// One button on the login row.
#[derive(Debug, Clone)]
pub struct LoginActionText {
    /// Composed widget id (also used for the debug selector).
    pub id: String,
    /// Composed visible label — always begins with `chrome::LOGIN_START_PREFIX`
    /// for the start action, or equals `chrome::LOGIN_CANCEL` for cancel.
    pub label: String,
    /// Whether the button paints as disabled.
    pub disabled: bool,
}

/// Error alert on the login row.
#[derive(Debug, Clone)]
pub struct LoginErrorText {
    /// Composed widget id — `format!("{outer_id}-error")`.
    pub id: String,
    /// The provider's error text.
    pub message: String,
}

impl<'a> RowText<'a> {
    /// Iterate every visible string the row paints — used by guard tests
    /// to sweep for stray markers or sentinel leaks. Chrome-only fields
    /// (headers, hints, action labels) are yielded alongside content
    /// fields so a marker anywhere in the row's paint set fails the
    /// sweep.
    pub fn visible_strings(&self) -> Vec<&str> {
        let mut out: Vec<&str> = Vec::new();
        match self {
            Self::User(text) => {
                let UserRowText {
                    content,
                    attachments,
                    fork_label,
                } = text;
                out.push(content);
                if let Some(list) = attachments {
                    out.extend(list.iter().map(String::as_str));
                }
                out.extend(fork_label.iter().copied());
            }
            Self::Assistant(text) => {
                let AssistantRowText {
                    source,
                    truncated_hint,
                } = text;
                out.push(source);
                out.extend(truncated_hint.iter().copied());
            }
            Self::Thinking(text) => {
                let ThinkingRowText { header } = text;
                out.push(header);
            }
            Self::Tool(text) => {
                let ToolRowText {
                    verb,
                    detail,
                    output_size_label,
                    hover_hint,
                    tail_omitted_hint,
                    body,
                } = text;
                out.push(verb);
                out.push(detail);
                out.extend(output_size_label.as_deref());
                out.extend(hover_hint.iter().copied());
                out.extend(tail_omitted_hint.iter().copied());
                out.extend(body.iter().copied());
            }
            Self::Error(text) => {
                let ErrorRowText {
                    header,
                    message,
                    settings_action_label,
                } = text;
                out.push(header);
                out.push(message);
                out.extend(settings_action_label.iter().copied());
            }
        }
        out
    }
}

impl LoginRowText {
    /// Sentinel/marker sweep for guard tests — yields every visible string
    /// a login row paints.
    pub fn visible_strings(&self) -> Vec<&str> {
        let mut out: Vec<&str> = vec![
            self.header_label.as_str(),
            self.start.label.as_str(),
            self.status_text,
        ];
        if let Some(cancel) = &self.cancel {
            out.push(cancel.label.as_str());
        }
        if let Some(error) = &self.error {
            out.push(error.message.as_str());
        }
        out
    }
}

/// Build the row-text model for one transcript row. Everything the
/// renderer will paint as user-visible text is computed here — never in
/// the render body.
pub fn build<'a>(
    entry: &'a TranscriptEntry,
    index: usize,
    session_view: &SessionView,
    settings_action_available: bool,
) -> RowText<'a> {
    match entry {
        TranscriptEntry::User(text) => RowText::User(UserRowText {
            content: text,
            attachments: session_view.attachments.get(&index).map(|list| {
                list.iter()
                    .map(|(name, size)| format_attachment(name, *size))
                    .collect()
            }),
            fork_label: fork_label_for(index, session_view),
        }),
        TranscriptEntry::Assistant(doc) => RowText::Assistant(AssistantRowText {
            source: doc.source.as_ref(),
            truncated_hint: doc.preview_truncated.then_some(chrome::ASSISTANT_TRUNCATED),
        }),
        TranscriptEntry::Thinking => RowText::Thinking(ThinkingRowText {
            header: THINKING_HEADER_LABEL,
        }),
        TranscriptEntry::Tool {
            name,
            summary,
            card,
            ..
        } => {
            let output_size = card.tail.text.len();
            let has_output = output_size > 0;
            let collapsed_with_output = !card.expanded && has_output;
            RowText::Tool(ToolRowText {
                verb: name,
                detail: summary,
                output_size_label: collapsed_with_output.then(|| format_output_size(output_size)),
                hover_hint: collapsed_with_output.then_some(chrome::TOOL_HOVER_HINT),
                tail_omitted_hint: (card.expanded && card.tail.truncated)
                    .then_some(chrome::TOOL_TAIL_OMITTED),
                body: card.expanded.then_some(card.tail.text.as_str()),
            })
        }
        TranscriptEntry::Error {
            message,
            settings_action,
            ..
        } => RowText::Error(ErrorRowText {
            header: ERROR_HEADER_LABEL,
            message,
            settings_action_label: (*settings_action && settings_action_available)
                .then_some(chrome::OPEN_SETTINGS),
        }),
    }
}

/// Build the login-row model from a provider record and its outer-id
/// prefix. The provider label is resolved here — the renderer receives a
/// composed `header_label`/`start.label` and never touches `provider.label()`
/// again. This is the fix for the r1 review finding at main.rs:1564: the
/// login renderer bypassed the typed model by reading `provider.label()`
/// directly on every paint.
pub fn build_login(
    provider: &LoginProvider,
    prefix: &str,
    connection: &ConnectionState,
) -> LoginRowText {
    let outer_id = sel::login_base_id(prefix, &provider.provider);
    let label = provider.label().to_owned();
    let start_id = sel::login_start(&outer_id);
    let cancel_id = sel::login_cancel(&outer_id);
    let error_id = sel::login_error(&outer_id);
    let start_disabled =
        provider.progress.busy() || !matches!(connection, ConnectionState::Connected);
    let cancel = provider.progress.busy().then(|| LoginActionText {
        id: cancel_id,
        label: chrome::LOGIN_CANCEL.to_owned(),
        disabled: provider.progress == LoginProgress::Cancelling,
    });
    let error = match &provider.progress {
        LoginProgress::Failed { error } => Some(LoginErrorText {
            id: error_id,
            message: error.message.clone(),
        }),
        _ => None,
    };
    LoginRowText {
        outer_id,
        header_label: label.clone(),
        provider_slug: provider.provider.clone(),
        start: LoginActionText {
            id: start_id,
            label: format!("{}{}", chrome::LOGIN_START_PREFIX, label),
            disabled: start_disabled,
        },
        cancel,
        status_text: provider.status(),
        error,
    }
}

fn fork_label_for(index: usize, session_view: &SessionView) -> Option<&'static str> {
    if !session_view.available {
        return None;
    }
    session_view
        .message_ids
        .get(&index)
        .map(|_| chrome::FORK_HERE)
}

/// One attachment chip label: `"{name}{sep}{size}{suffix}"`. The renderer
/// draws these verbatim — every visible piece comes from `chrome`.
fn format_attachment(name: &str, size: usize) -> String {
    format!(
        "{name}{sep}{size}{suffix}",
        sep = chrome::ATTACHMENT_SIZE_SEPARATOR,
        suffix = chrome::ATTACHMENT_SIZE_SUFFIX,
    )
}

/// Compact byte-size label for the collapsed tool-row output peek. Kept
/// short so the receipt still fits on one line at the wiki 1024px column.
/// Unit suffixes come from `chrome` so wording changes have one home.
fn format_output_size(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes}{}", chrome::OUTPUT_SIZE_UNIT_B)
    } else if bytes < 1024 * 1024 {
        format!(
            "{:.1}{}",
            bytes as f64 / 1024.0,
            chrome::OUTPUT_SIZE_UNIT_KB
        )
    } else {
        format!(
            "{:.1}{}",
            bytes as f64 / (1024.0 * 1024.0),
            chrome::OUTPUT_SIZE_UNIT_MB
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::Card;
    use crate::login::{LoginError, LoginProgress, LoginProvider};
    use crate::state::ToolReceiptKey;

    fn empty_view() -> SessionView {
        SessionView::default()
    }

    #[test]
    fn thinking_row_header_never_carries_body_text() {
        let entry = TranscriptEntry::Thinking;
        let row = build(&entry, 0, &empty_view(), true);
        let RowText::Thinking(text) = row else {
            panic!("thinking entry must build a Thinking row")
        };
        assert_eq!(text.header, THINKING_HEADER_LABEL);
    }

    #[test]
    fn sentinel_in_thinking_source_never_lands_on_the_row_model() {
        // The Thinking entry carries no body — a sentinel-shaped provider
        // payload cannot flow into any RowText field. This mirrors the
        // ZETA-107 privacy invariant.
        let entry = TranscriptEntry::Thinking;
        let row = build(&entry, 0, &empty_view(), true);
        for text in row.visible_strings() {
            assert!(
                !text.contains("SENTINEL"),
                "thinking row text contained sentinel: {text:?}"
            );
        }
    }

    #[test]
    fn tool_row_never_emits_state_markers_on_the_visible_seam() {
        // Contract line 83: state is COLOR only. No textual
        // "[working]/[done]/[failed]/[canceled]" marker may reach a
        // visible field on any tool row, in any state combination.
        let markers = ["[working]", "[done]", "[failed]", "[canceled]"];
        for (complete, error, canceled) in [
            (false, false, false),
            (true, false, false),
            (true, true, false),
            (true, false, true),
        ] {
            let entry = TranscriptEntry::Tool {
                key: ToolReceiptKey {
                    session_id: None,
                    agent_instance_id: None,
                    tool_call_id: "id".into(),
                },
                name: "bash".into(),
                summary: "echo".into(),
                complete,
                error,
                canceled,
                card: Card {
                    expanded: true,
                    ..Default::default()
                },
            };
            let row = build(&entry, 0, &empty_view(), true);
            for text in row.visible_strings() {
                for marker in markers {
                    assert!(
                        !text.contains(marker),
                        "tool row visible text carried {marker} for (complete={complete}, error={error}, canceled={canceled}): {text:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn chrome_literals_never_carry_state_markers() {
        // Adding a new chrome literal that happens to name a state
        // marker regresses the contract. Exhaustive sweep over the ALL
        // set (grep proof that ALL contains every chrome constant lives
        // in the fence's chrome-coverage check).
        for literal in chrome::ALL {
            for marker in ["[working]", "[done]", "[failed]", "[canceled]"] {
                assert!(
                    !literal.contains(marker),
                    "chrome literal {literal:?} carried state marker {marker}"
                );
            }
        }
    }

    #[test]
    fn user_row_model_carries_attachment_labels_and_optional_fork() {
        use crate::state::AppState;
        let mut state = AppState::default();
        state.session_view.available = true;
        state.session_view.message_ids.insert(0, "msg-1".into());
        state
            .session_view
            .attachments
            .insert(0, vec![("hero.png".into(), 4096)]);
        let entry = TranscriptEntry::User("hello".into());
        let row = build(&entry, 0, &state.session_view, true);
        let RowText::User(text) = row else {
            panic!("user entry must build a User row")
        };
        assert_eq!(text.content, "hello");
        assert_eq!(
            text.attachments,
            Some(vec!["hero.png · 4096 bytes".to_owned()])
        );
        assert_eq!(text.fork_label, Some(chrome::FORK_HERE));
    }

    #[test]
    fn user_row_without_message_id_hides_the_fork_label() {
        let entry = TranscriptEntry::User("hello".into());
        let row = build(&entry, 0, &empty_view(), true);
        let RowText::User(text) = row else {
            panic!("user entry must build a User row")
        };
        assert_eq!(text.fork_label, None);
    }

    #[test]
    fn user_row_distinguishes_absent_key_from_present_empty_attachments() {
        // Regression guard for the r1 review finding: text-only history
        // rows store `Some(vec![])` in `session_view.attachments`, and the
        // pre-typed-seam renderer painted the mt_1 container in that case.
        // The typed model must preserve the tri-state so the renderer can
        // keep those rows' height identical to the pre-seam behaviour.
        use crate::state::AppState;
        let entry = TranscriptEntry::User("hi".into());
        // Absent key: renderer must skip the container.
        let no_key = build(&entry, 0, &AppState::default().session_view, true);
        let RowText::User(text) = no_key else {
            unreachable!()
        };
        assert!(text.attachments.is_none(), "absent key must map to None");
        // Present empty: renderer must paint the (empty) container so the
        // row height matches text-only history rows before the refactor.
        let mut state = AppState::default();
        state.session_view.attachments.insert(0, Vec::new());
        let present_empty = build(&entry, 0, &state.session_view, true);
        let RowText::User(text) = present_empty else {
            unreachable!()
        };
        assert_eq!(
            text.attachments,
            Some(Vec::<String>::new()),
            "present-empty key must map to Some(empty)",
        );
    }

    #[test]
    fn tool_output_size_label_scales_across_unit_boundaries() {
        for (bytes, expected) in [
            (0usize, None),
            (900, Some("900B".to_owned())),
            (2048, Some("2.0KB".to_owned())),
            (3 * 1024 * 1024, Some("3.0MB".to_owned())),
        ] {
            let mut card = Card {
                expanded: false,
                ..Default::default()
            };
            card.tail.text = "x".repeat(bytes);
            let entry = TranscriptEntry::Tool {
                key: ToolReceiptKey {
                    session_id: None,
                    agent_instance_id: None,
                    tool_call_id: "id".into(),
                },
                name: "bash".into(),
                summary: "".into(),
                complete: true,
                error: false,
                canceled: false,
                card,
            };
            let row = build(&entry, 0, &empty_view(), true);
            let RowText::Tool(text) = row else {
                panic!("tool entry must build a Tool row")
            };
            assert_eq!(text.output_size_label, expected, "at {bytes}B");
        }
    }

    fn provider(progress: LoginProgress) -> LoginProvider {
        LoginProvider {
            provider: "claude".into(),
            credentials_present: false,
            progress,
        }
    }

    #[test]
    fn login_model_composes_start_label_from_chrome_prefix_and_provider_label() {
        let text = build_login(
            &provider(LoginProgress::Idle),
            "settings-login",
            &ConnectionState::Connected,
        );
        assert_eq!(text.header_label, "Claude");
        assert_eq!(text.start.label, "Log in with Claude");
        assert!(!text.start.disabled);
        assert_eq!(text.status_text, "Not logged in");
        assert!(text.cancel.is_none());
        assert!(text.error.is_none());
        // The composed IDs derive from the prefix + provider slug.
        assert_eq!(text.outer_id, "settings-login-claude");
        assert_eq!(text.start.id, "settings-login-claude-start");
    }

    #[test]
    fn login_model_disables_start_when_disconnected_and_shows_cancel_when_busy() {
        let text = build_login(
            &provider(LoginProgress::Idle),
            "settings-login",
            &ConnectionState::Reconnecting,
        );
        assert!(text.start.disabled, "disconnected must disable start");
        let busy = build_login(
            &provider(LoginProgress::Starting),
            "settings-login",
            &ConnectionState::Connected,
        );
        let cancel = busy.cancel.expect("busy provider paints a cancel button");
        assert_eq!(cancel.label, "Cancel");
        assert!(!cancel.disabled);
    }

    #[test]
    fn login_model_surfaces_the_failure_message() {
        let mut prov = provider(LoginProgress::Failed {
            error: LoginError {
                code: "boom".into(),
                message: "no browser".into(),
            },
        });
        prov.credentials_present = false;
        let text = build_login(&prov, "first-login", &ConnectionState::Connected);
        let err = text.error.expect("failed progress must produce an error");
        assert_eq!(err.message, "no browser");
        assert_eq!(err.id, "first-login-claude-error");
    }
}
