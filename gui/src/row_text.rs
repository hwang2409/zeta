//! Typed model of every user-visible string a transcript row can paint.
//!
//! One struct per row kind whose fields NAME every visible string the row
//! renders — body text, action/button labels, attachment names and sizes,
//! tool output sizes, chevrons/hints, and fixed chrome. The renderers in
//! `main.rs` build their visible-text elements by destructuring this model;
//! no visible string is sourced outside it. Together with the
//! `renderer_literal_fence` guard test — which trips on any inline user-
//! visible string literal in the six transcript render functions — the two
//! together make a NEW stray literal impossible to add without failing a
//! test.
//!
//! The typed model REPLACES the earlier positional
//! `TranscriptEntry::visible_text -> Vec<&str>` seam. That seam covered row
//! CONTENT only; action labels, attachment names/sizes, and tool output
//! sizes still flowed from entry fields via inline `format!` calls in the
//! render layer, so a new stray literal on any of those paths bypassed the
//! seam. The typed model closes those paths.
//!
//! Borrowing: content-shaped fields (`content`, `source`, `verb`, `detail`,
//! `body`, `message`) hold a `&str` into the transcript entry, so a per-row
//! `build()` allocates only for the composed labels (attachment lines,
//! output-size string). A per-frame full-transcript clone is not
//! introduced — this matches the ZETA-107 r5 borrowing invariant.

use crate::session::SessionView;
use crate::state::{TranscriptEntry, ERROR_HEADER_LABEL, THINKING_HEADER_LABEL};

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
    ];
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
/// hover-revealed fork action.
#[derive(Debug, Clone)]
pub struct UserRowText<'a> {
    /// The prompt body — the row's only dynamic body string.
    pub content: &'a str,
    /// One label per attachment chip, pre-composed with the chrome
    /// separator and unit suffix so the renderer never formats visible
    /// text inline.
    pub attachments: Vec<String>,
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
                out.extend(attachments.iter().map(String::as_str));
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
            attachments: session_view
                .attachments
                .get(&index)
                .map(|list| {
                    list.iter()
                        .map(|(name, size)| format_attachment(name, *size))
                        .collect()
                })
                .unwrap_or_default(),
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
        assert_eq!(text.attachments, vec!["hero.png · 4096 bytes".to_owned()]);
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
}
