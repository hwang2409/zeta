use crate::{
    cards::{Card, OutputTail},
    markdown::Markdown,
};
use std::collections::HashMap;

use crate::client::{
    Approval, ContentBlock, Message, ServerEvent, SessionMetadata, StatusResult, SubAgentReceipt,
    SubAgentStatus, ToolCall,
};

#[derive(Debug, Clone, PartialEq)]
pub struct ToolReceiptKey {
    pub session_id: Option<String>,
    pub agent_instance_id: Option<String>,
    pub tool_call_id: String,
}

impl ToolReceiptKey {
    fn new(session_id: Option<String>, data: &serde_json::Value, tool_call: &ToolCall) -> Self {
        Self {
            session_id,
            agent_instance_id: data["agent_instance_id"].as_str().map(str::to_owned),
            tool_call_id: tool_call.id.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TranscriptEntry {
    User(String),
    Assistant(Markdown),
    Error {
        message: String,
        settings_action: bool,
        login_provider: Option<String>,
    },
    Tool {
        key: ToolReceiptKey,
        name: String,
        summary: String,
        complete: bool,
        error: bool,
        canceled: bool,
        card: Card,
    },
    /// Generic thinking marker. Zeta's provider protocol has no display-safe
    /// summary channel — `ContentBlock::Thinking` mixes raw reasoning with any
    /// model-emitted summary — so the GUI never renders body text. The entry is
    /// purely a header ("+ Thought"), signalling that the model thought without
    /// leaking what.
    Thinking,
}

impl TranscriptEntry {
    pub fn unsuccessful(&self) -> bool {
        matches!(
            self,
            Self::Tool { error: true, .. } | Self::Tool { canceled: true, .. }
        )
    }

    /// Semantic state carried by a tool row. Contract line 83: state is signalled by
    /// COLOR ONLY — no textual "[working]/[done]/[failed]" markers land on the
    /// visible row. The render layer maps each state to a theme token.
    pub fn tool_state(&self) -> ToolState {
        if self.unsuccessful() {
            ToolState::Failed
        } else if matches!(self, Self::Tool { complete: true, .. }) {
            ToolState::Done
        } else {
            ToolState::Running
        }
    }

    /// Ordered visible strings the render layer paints for this row.
    /// Single seam: any string that reaches the user through the row's own
    /// text elements (not framing chrome like a chevron icon or a hover hint)
    /// flows through this collection. Borrowed so a full-transcript pass
    /// does not clone every source string on every draw — renderers convert
    /// to owned SharedStrings only at the leaf gpui element that needs them.
    pub fn visible_text(&self) -> Vec<&str> {
        match self {
            Self::User(text) => vec![text.as_str()],
            Self::Assistant(doc) => vec![doc.source.as_ref()],
            Self::Error { message, .. } => vec![ERROR_HEADER_LABEL, message.as_str()],
            Self::Thinking => vec![THINKING_HEADER_LABEL],
            Self::Tool {
                name,
                summary,
                card,
                ..
            } => {
                // Dynamic body text only: name at 0, summary at 1, and the
                // expanded output tail at 2 when the card is open. Fixed
                // chrome (headings, hints, unit suffixes) lives in the
                // render-layer chrome module — one home per literal.
                let mut strings = vec![name.as_str(), summary.as_str()];
                if card.expanded {
                    strings.push(card.tail.text.as_str());
                }
                strings
            }
        }
    }
}

/// Single source of truth for the thinking marker text. Both `visible_text`
/// and the render layer read this constant so no wording lives on both sides
/// of the seam.
pub const THINKING_HEADER_LABEL: &str = "+ Thought";

/// Single source of truth for the error row header text. Kept alongside the
/// thinking label so both sides of the seam read one constant.
pub const ERROR_HEADER_LABEL: &str = "Error";

/// Semantic tool row state; the render layer maps each variant to a theme
/// token per the wiki contract (running=foreground, done=muted_foreground,
/// failed/canceled=danger).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolState {
    Running,
    Done,
    Failed,
}

/// What `AppState::apply` did to the transcript, for the view to splice the
/// virtual-list metadata cache in step. A bare count delta cannot say WHERE
/// a row was removed, so the tail-splice branch would drop the cache entry
/// for the last row while the surviving rows kept stale heights and scroll
/// offsets from their pre-removal positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscriptChange {
    /// Row at this index was appended or mutated in place. The view remeasures
    /// exactly that row.
    Row(usize),
    /// Row at this index was removed from the transcript. The view splices
    /// `index..index + 1` out of its metadata cache so the shifted rows keep
    /// their measured heights aligned to their new positions.
    Removed(usize),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConnectionState {
    Connected,
    Reconnecting,
    Lost(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct AppState {
    pub session_view: crate::session::SessionView,
    pub sessions: Vec<SessionMetadata>,
    pub active_session: Option<String>,
    pub sessions_truncated: bool,
    pub saved_transcripts: HashMap<String, Vec<TranscriptEntry>>,
    pub transcript: Vec<TranscriptEntry>,
    pub approvals: Vec<Approval>,
    pub connection: ConnectionState,
    pub streaming: bool,
    pub thinking: bool,
    assistant_started: bool,
    /// Index into `transcript` where the current server turn began. Set to
    /// `transcript.len()` on every `TurnStart`, session switch, and history
    /// replay (each of those rebuilds the transcript, then anchors here to
    /// the new tail). `commit_assistant` reconciles only against rows at or
    /// after this index, so a spontaneous second turn (no user row of its
    /// own) never merges into the previous turn's rows, and a resumed
    /// in-flight message after a history replay never edits restored rows.
    turn_start: usize,
    pub metrics: StatusMetrics,
    pub metrics_boundary: bool,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            session_view: Default::default(),
            sessions: Vec::new(),
            active_session: None,
            sessions_truncated: false,
            saved_transcripts: HashMap::new(),
            transcript: Vec::new(),
            approvals: Vec::new(),
            connection: ConnectionState::Reconnecting,
            streaming: false,
            thinking: false,
            assistant_started: false,
            turn_start: 0,
            metrics: StatusMetrics::default(),
            metrics_boundary: true,
        }
    }
}

impl AppState {
    pub fn apply_history(&mut self, messages: Vec<crate::client::HistoryMessage>, replace: bool) {
        use crate::client::HistoryContent;
        if replace {
            self.transcript.clear();
            // Reset temporarily; the rebuilt tail becomes the anchor at the
            // end of this method so a resumed in-flight `AssistantMessage`
            // never folds into a pre-history row.
            self.turn_start = 0;
        }
        self.session_view.message_ids.clear();
        self.session_view.attachments.clear();
        let user_rows: Vec<_> = self
            .transcript
            .iter()
            .enumerate()
            .filter_map(|(index, row)| matches!(row, TranscriptEntry::User(_)).then_some(index))
            .collect();
        let mut users = 0;
        let mut calls = HashMap::new();
        for message in messages {
            let text: String = message
                .content
                .iter()
                .filter_map(|block| match block {
                    HistoryContent::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            if message.role == "user" {
                let index = if replace {
                    let index = self.transcript.len();
                    self.transcript.push(TranscriptEntry::User(text));
                    Some(index)
                } else {
                    user_rows.get(users).copied()
                };
                users += 1;
                if let Some(index) = index {
                    self.session_view.message_ids.insert(index, message.id);
                    let attachments = message
                        .content
                        .iter()
                        .filter_map(|block| match block {
                            HistoryContent::Attachment { name, size } => {
                                Some((name.clone(), *size))
                            }
                            _ => None,
                        })
                        .collect();
                    self.session_view.attachments.insert(index, attachments);
                }
            } else if replace && message.role == "assistant" {
                if !text.is_empty() {
                    self.transcript
                        .push(TranscriptEntry::Assistant(text.into()));
                }
                for block in message.content {
                    if let HistoryContent::ToolUse { tool_call } = block {
                        self.apply(ServerEvent::ToolStart {
                            session_id: self.active_session.clone(),
                            tool_call: tool_call.clone(),
                            data: Default::default(),
                        });
                        calls.insert(tool_call.id.clone(), tool_call);
                    }
                }
            } else if replace {
                if let Some(result) = message.tool_result {
                    if let Some(call) = calls.remove(&result.tool_call_id) {
                        self.apply(ServerEvent::ToolEnd {
                            session_id: self.active_session.clone(),
                            tool_call: call,
                            tool_result: Some(result),
                            data: Default::default(),
                        });
                    }
                }
            }
        }
        if replace {
            // Anchor the next turn's reconciliation to the rebuilt tail. A
            // resumed in-flight `AssistantMessage` following history replay
            // would otherwise scan from index 0 and replace an old assistant
            // row from the restored history. Regression test:
            // `resumed_stream_after_history_replay_does_not_edit_older_rows`.
            self.turn_start = self.transcript.len();
        }
    }

    pub fn mark_connection_lost(&mut self, error: impl Into<String>) {
        self.connection = ConnectionState::Lost(error.into());
        self.streaming = false;
        self.thinking = false;
        self.approvals.clear();
    }

    pub fn begin_reconnect(&mut self) {
        self.connection = ConnectionState::Reconnecting;
    }

    pub fn select_session(&mut self, session_id: Option<String>) {
        if self.active_session == session_id {
            return;
        }
        let transcript = std::mem::take(&mut self.transcript);
        if let Some(previous) = self.active_session.take() {
            self.saved_transcripts.insert(previous, transcript);
        }
        self.transcript = session_id
            .as_ref()
            .and_then(|id| self.saved_transcripts.remove(id))
            .unwrap_or_default();
        self.active_session = session_id;
        self.thinking = false;
        self.assistant_started = false;
        self.turn_start = self.transcript.len();
        self.session_view = crate::session::SessionView {
            available: self.session_view.available,
            ..Default::default()
        };
        self.metrics = StatusMetrics::default();
        self.metrics_boundary = true;
    }

    pub fn apply_status(&mut self, status: StatusResult) {
        self.select_session(
            status
                .session
                .as_ref()
                .map(|session| session.session_id.clone()),
        );
        if self.metrics_boundary || (!self.streaming && status.state == "idle") {
            self.metrics = StatusMetrics::from_status(&status);
            self.metrics_boundary = false;
        }
        self.streaming = status.state != "idle";
        if !self.streaming {
            self.thinking = false;
        }
        self.approvals = status.pending_approvals;
    }

    pub fn apply(&mut self, event: ServerEvent) -> Option<TranscriptChange> {
        let mut changed: Option<TranscriptChange> = None;
        match event {
            ServerEvent::TurnStart { .. } => {
                self.streaming = true;
                self.thinking = false;
                self.assistant_started = false;
                self.metrics_boundary = false;
                // Anchor reconciliation for this turn to the current tail so
                // streamed rows and the final AssistantMessage merge here
                // without folding into the previous turn's rows.
                self.turn_start = self.transcript.len();
            }
            ServerEvent::AgentEnd { .. } | ServerEvent::TurnAborted { .. } => {
                self.streaming = false;
                self.thinking = false;
                self.metrics_boundary = true;
                self.approvals.clear();
            }
            ServerEvent::TurnEnd { .. } => {
                self.thinking = false;
                self.metrics_boundary = true;
            }
            ServerEvent::AssistantDelta { delta, kind, .. }
                if kind == "assistant" && !delta.is_empty() =>
            {
                self.thinking = false;
                self.assistant_started = true;
                match self.transcript.last_mut() {
                    Some(TranscriptEntry::Assistant(text)) => text.push_str(&delta),
                    _ => self
                        .transcript
                        .push(TranscriptEntry::Assistant(Markdown::streaming(delta))),
                }
                changed = self
                    .transcript
                    .len()
                    .checked_sub(1)
                    .map(TranscriptChange::Row);
            }
            ServerEvent::AssistantDelta { kind, delta, .. }
                if kind == "thinking" && !delta.is_empty() =>
            {
                self.thinking = !self.assistant_started;
                // Emit the header-only thinking marker the first time we see
                // thinking in this turn. The reasoning delta itself is never
                // stored — the provider protocol has no display-safe summary
                // channel, so the row carries no body.
                if self.thinking
                    && !matches!(self.transcript.last(), Some(TranscriptEntry::Thinking))
                {
                    self.transcript.push(TranscriptEntry::Thinking);
                    changed = self
                        .transcript
                        .len()
                        .checked_sub(1)
                        .map(TranscriptChange::Row);
                }
            }
            ServerEvent::AssistantDelta { .. } => {}
            ServerEvent::AssistantMessage { message, .. } => {
                self.thinking = false;
                self.assistant_started |= !message.text().is_empty();
                changed = self.commit_assistant(message);
            }
            ServerEvent::ToolStart {
                session_id,
                tool_call,
                data,
            } => {
                self.thinking = false;
                self.transcript.push(tool_entry(
                    &tool_call,
                    ToolReceiptKey::new(session_id, &data, &tool_call),
                ));
                changed = self
                    .transcript
                    .len()
                    .checked_sub(1)
                    .map(TranscriptChange::Row);
            }
            ServerEvent::ToolOutput {
                session_id,
                tool_call,
                output,
                data,
            } => {
                let (index, entry) = self.tool_receipt(
                    &tool_call,
                    ToolReceiptKey::new(session_id, &data, &tool_call),
                );
                changed = Some(TranscriptChange::Row(index));
                if let TranscriptEntry::Tool { summary, card, .. } = entry {
                    card.tail.append(&output);
                    if let Some(line) = card
                        .tail
                        .text
                        .lines()
                        .rev()
                        .find(|line| !line.trim().is_empty())
                    {
                        *summary = bounded_summary(line);
                    }
                }
            }
            ServerEvent::ToolEnd {
                tool_call,
                tool_result,
                session_id,
                data,
            } => {
                let (index, entry) = self.tool_receipt(
                    &tool_call,
                    ToolReceiptKey::new(session_id, &data, &tool_call),
                );
                changed = Some(TranscriptChange::Row(index));
                if let TranscriptEntry::Tool {
                    complete,
                    error,
                    canceled,
                    summary,
                    card,
                    name,
                    ..
                } = entry
                {
                    if name.eq_ignore_ascii_case("agent") {
                        if let Some(child) = tool_result
                            .as_ref()
                            .and_then(|result| result.structured_content.as_ref())
                            .and_then(|data| data["child_instance_id"].as_str())
                        {
                            card.child_instance_id = Some(child.to_owned());
                        }
                    }
                    *canceled = tool_result
                        .as_ref()
                        .is_some_and(|result| result.is_canceled);
                    *complete = *canceled
                        || !name.eq_ignore_ascii_case("agent")
                        || !tool_result
                            .as_ref()
                            .and_then(|result| result.structured_content.as_ref())
                            .is_some_and(|data| data["status"] == "running");
                    let failed = tool_result.as_ref().is_some_and(|result| result.is_error);
                    if failed && !*error {
                        card.expanded = true;
                    }
                    *error = failed;
                    if let Some(result) = tool_result.filter(|result| !result.content.is_empty()) {
                        if card.tail.text != result.content {
                            if !card.tail.text.is_empty() && !card.tail.text.ends_with('\n') {
                                card.tail.append("\n");
                            }
                            card.tail.append(&result.content);
                        }
                        *summary = bounded_summary(
                            result
                                .content
                                .lines()
                                .find(|line| !line.trim().is_empty())
                                .unwrap_or(""),
                        );
                    }
                }
            }
            ServerEvent::SubAgentReceipt {
                session_id,
                receipt,
            } => {
                changed = Some(TranscriptChange::Row(
                    self.commit_sub_agent(session_id, receipt),
                ));
            }
            ServerEvent::ApprovalRequest { approval, .. } => {
                if !self
                    .approvals
                    .iter()
                    .any(|item| item.request_id == approval.request_id)
                {
                    self.approvals.push(approval);
                }
            }
            // The protocol has no request ID here. The worker follows this event
            // with authoritative status, which preserves other delegated approvals.
            ServerEvent::ApprovalEnd { .. } => {}
            ServerEvent::Error { error, data, .. } => {
                self.streaming = false;
                self.thinking = false;
                self.metrics_boundary = true;
                self.approvals.clear();
                let settings_action =
                    matches!(error.code.as_str(), "model_access_error" | "model_reverted");
                self.transcript.push(TranscriptEntry::Error {
                    message: error.message,
                    settings_action,
                    login_provider: data["login_provider"].as_str().map(str::to_owned),
                });
                changed = self
                    .transcript
                    .len()
                    .checked_sub(1)
                    .map(TranscriptChange::Row);
            }
            ServerEvent::Other { .. } => {}
        }
        changed
    }

    pub fn toggle_card(&mut self, index: usize) {
        if let Some(TranscriptEntry::Tool { card, .. }) = self.transcript.get_mut(index) {
            card.toggle();
        }
    }

    fn tool_receipt(
        &mut self,
        tool_call: &ToolCall,
        key: ToolReceiptKey,
    ) -> (usize, &mut TranscriptEntry) {
        let index = self
            .transcript
            .iter()
            .position(
                |entry| matches!(entry, TranscriptEntry::Tool { key: existing, .. } if existing == &key),
            )
            .unwrap_or_else(|| {
                self.transcript.push(tool_entry(tool_call, key));
                self.transcript.len() - 1
            });
        (index, &mut self.transcript[index])
    }

    fn commit_sub_agent(&mut self, session_id: Option<String>, receipt: SubAgentReceipt) -> usize {
        // Durable notifications have no raw tool ID. The launch result supplies
        // the child identity when we saw it; reconnect drains can create it alone.
        let index = self
            .transcript
            .iter()
            .position(|entry| {
                matches!(entry, TranscriptEntry::Tool { key, card, .. }
                if key.session_id == session_id
                    && card.child_instance_id.as_deref() == Some(&receipt.child_instance_id))
            })
            .unwrap_or_else(|| {
                self.transcript.push(TranscriptEntry::Tool {
                    key: ToolReceiptKey {
                        session_id,
                        agent_instance_id: Some(receipt.child_instance_id.clone()),
                        tool_call_id: String::new(),
                    },
                    name: "agent".into(),
                    summary: String::new(),
                    complete: false,
                    error: false,
                    canceled: false,
                    card: Card {
                        child_instance_id: Some(receipt.child_instance_id.clone()),
                        ..Default::default()
                    },
                });
                self.transcript.len() - 1
            });
        if let TranscriptEntry::Tool {
            summary,
            complete,
            error,
            canceled,
            card,
            ..
        } = &mut self.transcript[index]
        {
            *complete = true;
            let failed = receipt.status != SubAgentStatus::Completed;
            if failed && receipt.status != SubAgentStatus::Canceled && !*error {
                card.expanded = true;
            }
            *error = failed;
            *canceled = receipt.status == SubAgentStatus::Canceled;
            *summary = bounded_summary(
                receipt
                    .text
                    .lines()
                    .find(|line| !line.trim().is_empty())
                    .unwrap_or(""),
            );
            card.agent_label = Some(bounded_summary(&receipt.description));
            // Replace the authoritative receipt tail so replay is idempotent.
            card.tail = OutputTail::default();
            card.tail.append(&receipt.text);
        }
        index
    }

    fn commit_assistant(&mut self, message: Message) -> Option<TranscriptChange> {
        let has_thinking = message
            .content
            .iter()
            .any(|block| matches!(block, ContentBlock::Thinking { .. }));
        let mut assistant_text = String::new();
        for block in &message.content {
            if let ContentBlock::Text { text } = block {
                assistant_text.push_str(text);
            }
        }
        // Reconcile against the active turn only. `self.turn_start` is set to
        // `transcript.len()` on every `TurnStart`, so streamed rows plus the
        // final `AssistantMessage` in the SAME turn merge here — while a
        // spontaneous second turn (no fresh user row) never folds into the
        // previous turn's Thinking or Assistant rows. Clamped defensively in
        // case a mutation trimmed the transcript after the last TurnStart.
        let turn_start = self.turn_start.min(self.transcript.len());
        let has_thinking_marker = self.transcript[turn_start..]
            .iter()
            .any(|entry| matches!(entry, TranscriptEntry::Thinking));
        let mut changed: Option<TranscriptChange> = None;
        // Thinking body text is never retained — the provider protocol mixes
        // raw reasoning with any summary, so we only guarantee that a
        // header-only marker exists when the turn thought at all.
        if has_thinking && !has_thinking_marker {
            self.transcript.push(TranscriptEntry::Thinking);
            changed = self
                .transcript
                .len()
                .checked_sub(1)
                .map(TranscriptChange::Row);
        }
        if !assistant_text.is_empty() {
            // Every assistant row in the current turn, in transcript order.
            // An interleaved stream — AssistantDelta("pre"), ToolStart,
            // AssistantDelta("post") — leaves TWO assistant rows around the
            // tool row. Reconciling only the last row (the old `rposition`
            // path) replaced it with the full final "prepost" and left "pre"
            // in the earlier row, duplicating the prefix on screen.
            //
            // The correct behaviour preserves row order around tools: keep
            // the leading assistant rows exactly as streamed and put the
            // trailing remainder in the last row so the concatenation
            // equals the final text. If the streamed rows are not a real
            // prefix of the final (a rare drop/reorder), collapse the
            // earlier ones into the last row rather than paint stale text.
            let assistant_indices: Vec<usize> = self.transcript[turn_start..]
                .iter()
                .enumerate()
                .filter_map(|(offset, entry)| {
                    matches!(entry, TranscriptEntry::Assistant(_)).then_some(turn_start + offset)
                })
                .collect();
            if assistant_indices.is_empty() {
                self.transcript
                    .push(TranscriptEntry::Assistant(assistant_text.into()));
                changed = self
                    .transcript
                    .len()
                    .checked_sub(1)
                    .map(TranscriptChange::Row);
            } else {
                let leading = &assistant_indices[..assistant_indices.len() - 1];
                let mut cursor = 0usize;
                let mut prefix_ok = true;
                for &idx in leading {
                    let TranscriptEntry::Assistant(doc) = &self.transcript[idx] else {
                        unreachable!("assistant_indices filter matched this row")
                    };
                    let src = doc.source.as_ref();
                    if assistant_text[cursor..].starts_with(src) {
                        cursor += src.len();
                    } else {
                        prefix_ok = false;
                        break;
                    }
                }
                if prefix_ok {
                    let last_idx = *assistant_indices.last().unwrap();
                    let tail = assistant_text[cursor..].to_owned();
                    // Empty tail means the leading rows already cover the full
                    // final text — a shorter-than-streamed final or a final
                    // equal to a mid-tool prefix would otherwise leave an
                    // empty assistant row that still consumes transcript
                    // rhythm. Drop the trailing row instead of blanking it,
                    // and report the exact removed index so the view splices
                    // metadata at that position rather than at the tail.
                    if tail.is_empty() {
                        self.transcript.remove(last_idx);
                        changed = Some(TranscriptChange::Removed(last_idx));
                    } else {
                        if let TranscriptEntry::Assistant(current) = &mut self.transcript[last_idx]
                        {
                            *current = tail.into();
                        }
                        changed = Some(TranscriptChange::Row(last_idx));
                    }
                } else {
                    for &idx in leading.iter().rev() {
                        self.transcript.remove(idx);
                    }
                    let last_idx = self.transcript[turn_start..]
                        .iter()
                        .rposition(|entry| matches!(entry, TranscriptEntry::Assistant(_)))
                        .map(|offset| turn_start + offset)
                        .expect("collapsed assistant row still present");
                    if let TranscriptEntry::Assistant(current) = &mut self.transcript[last_idx] {
                        *current = assistant_text.into();
                    }
                    changed = Some(TranscriptChange::Row(last_idx));
                }
            }
        }
        changed
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct StatusMetrics {
    pub model: Option<String>,
    pub tokens: Option<u64>,
    pub cache_hit_rate: Option<f64>,
}

impl StatusMetrics {
    fn from_status(status: &StatusResult) -> Self {
        let usage = &status.usage;
        let input = usage["input_tokens"].as_u64();
        let output = usage["output_tokens"].as_u64();
        let read = usage["cache_read_input_tokens"].as_u64();
        let write = usage["cache_creation_input_tokens"].as_u64().unwrap_or(0);
        let prompt =
            input.and_then(|input| input.checked_add(read.unwrap_or(0))?.checked_add(write));
        Self {
            model: status
                .session
                .as_ref()
                .map(|s| s.model.clone())
                .filter(|s| !s.is_empty()),
            tokens: prompt
                .zip(output)
                .and_then(|(input, output)| input.checked_add(output)),
            cache_hit_rate: read
                .zip(prompt)
                .filter(|(read, total)| *total > 0 && read <= total)
                .map(|(read, total)| read as f64 / total as f64 * 100.),
        }
    }
    pub fn model_label(&self) -> &str {
        self.model.as_deref().unwrap_or("—")
    }
    pub fn tokens_label(&self) -> String {
        self.tokens.map_or_else(|| "—".into(), |n| n.to_string())
    }
    pub fn cache_label(&self) -> String {
        self.cache_hit_rate
            .map_or_else(|| "—".into(), |n| format!("{n:.1}%"))
    }
}

const SUMMARY_CHARS: usize = 240;

fn bounded_summary(text: &str) -> String {
    text.chars()
        .map(|ch| if ch.is_control() { ' ' } else { ch })
        .take(SUMMARY_CHARS)
        .collect()
}

fn tool_entry(tool_call: &ToolCall, key: ToolReceiptKey) -> TranscriptEntry {
    let agent_label = if tool_call.name.eq_ignore_ascii_case("agent") {
        Some(
            tool_call
                .arguments
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("agent")
                .to_owned(),
        )
    } else {
        key.agent_instance_id.clone()
    };
    let agent_label = agent_label.map(|label| bounded_summary(&label));
    TranscriptEntry::Tool {
        card: Card {
            agent_label,
            ..Default::default()
        },
        key,
        name: tool_call.name.clone(),
        summary: bounded_summary(
            &serde_json::to_string(&tool_call.arguments)
                .unwrap_or_else(|_| "arguments unavailable".to_owned()),
        ),
        complete: false,
        error: false,
        canceled: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::{EventError, ToolResult};
    use serde_json::json;

    #[test]
    fn canceled_server_results_never_show_success() {
        for is_error in [false, true] {
            let mut state = AppState::default();
            // server/server.py forwards ToolResult.to_dict() unchanged. Agent
            // cancellations use false/true; approval cancellations use true/true.
            let result = serde_json::from_value(json!({
                "tool_call_id": "tool-1", "content": "tool execution canceled",
                "is_error": is_error, "is_canceled": true,
                "structured_content": {"status": "running"}
            }))
            .unwrap();
            let changed = state.apply(ServerEvent::ToolEnd {
                session_id: None,
                tool_call: ToolCall {
                    name: "agent".into(),
                    ..call()
                },
                tool_result: Some(result),
                data: json!({}),
            });
            assert_eq!(changed, Some(TranscriptChange::Row(0)));
            let entry = &state.transcript[0];
            assert!(entry.unsuccessful());
            assert!(matches!(
                entry,
                TranscriptEntry::Tool {
                    complete: true,
                    canceled: true,
                    error,
                    ..
                } if *error == is_error
            ));
        }
    }

    #[test]
    fn metrics_ignore_midstream_status_and_clear_on_session_switch() {
        let status = |usage| {
            serde_json::from_value(json!({"session":{"session_id":"one","model":"test"},"state":"running","usage":usage})).unwrap()
        };
        let mut state = AppState::default();
        state.apply_status(status(json!({"input_tokens":10,"output_tokens":4})));
        assert_eq!(state.metrics.tokens_label(), "14");
        assert_eq!(state.metrics.cache_label(), "—");
        state.apply(ServerEvent::TurnStart {
            session_id: Some("one".into()),
            data: json!({}),
        });
        state.apply_status(status(json!({"input_tokens":900,"output_tokens":100})));
        assert_eq!(state.metrics.tokens_label(), "14");
        state.apply(ServerEvent::TurnEnd {
            session_id: Some("one".into()),
            data: json!({}),
        });
        state.apply_status(status(json!({"input_tokens":900,"output_tokens":100})));
        assert_eq!(state.metrics.tokens_label(), "1000");
        state.select_session(Some("two".into()));
        assert_eq!(state.metrics.model_label(), "—");
        assert_eq!(state.metrics.tokens_label(), "—");
    }

    #[test]
    fn background_agent_receipt_stays_running_until_final_result() {
        let mut state = AppState::default();
        let agent = ToolCall {
            name: "agent".into(),
            arguments: [("description".into(), json!("review code"))]
                .into_iter()
                .collect(),
            ..call()
        };
        state.apply(ServerEvent::ToolStart {
            session_id: None,
            tool_call: agent.clone(),
            data: json!({}),
        });
        for (status, text, complete) in [
            ("running", "started", false),
            ("completed", "review passed\nfull summary", true),
        ] {
            state.apply(ServerEvent::ToolEnd {
                session_id: None,
                tool_call: agent.clone(),
                data: json!({}),
                tool_result: Some(ToolResult {
                    tool_call_id: agent.id.clone(),
                    content: text.into(),
                    is_error: false,
                    is_canceled: false,
                    content_blocks: vec![],
                    structured_content: Some(json!({"status":status})),
                }),
            });
            assert!(
                matches!(&state.transcript[0], TranscriptEntry::Tool { complete: actual, card, .. } if *actual == complete && card.agent_label.as_deref() == Some("review code"))
            );
        }
        assert_eq!(state.transcript.len(), 1);
        assert!(
            matches!(&state.transcript[0], TranscriptEntry::Tool { summary, card, .. } if summary == "review passed" && card.tail.text.ends_with("full summary"))
        );
    }

    fn call() -> ToolCall {
        ToolCall {
            id: "tool-1".to_owned(),
            name: "read".to_owned(),
            arguments: [("path".to_owned(), json!("README.md"))]
                .into_iter()
                .collect(),
        }
    }

    #[test]
    fn reassembles_stream_and_committed_message() {
        let mut state = AppState::default();
        state.apply(ServerEvent::TurnStart {
            session_id: None,
            data: json!({}),
        });
        state.apply(ServerEvent::AssistantDelta {
            session_id: None,
            delta: "hel".to_owned(),
            kind: "assistant".to_owned(),
        });
        state.apply(ServerEvent::AssistantDelta {
            session_id: None,
            delta: "lo".to_owned(),
            kind: "assistant".to_owned(),
        });
        state.apply(ServerEvent::AssistantMessage {
            session_id: None,
            message: Message {
                role: "assistant".to_owned(),
                content: vec![crate::client::ContentBlock::Text {
                    text: "hello".to_owned(),
                }],
            },
        });
        assert_eq!(
            describe_transcript(&state.transcript),
            vec![r#"Assistant("hello")"#.to_owned()]
        );
    }

    #[test]
    fn streamed_mixed_turn_reconciles_without_duplicating_rows() {
        // Regression: with mixed content, the stream emits a thinking delta
        // (which pushes the Thinking marker) and a text delta (which pushes the
        // Assistant row). The final AssistantMessage carries BOTH blocks. The
        // old reconciler only checked `last()`, so it saw the Assistant on top
        // and pushed a second Thinking, then found the second Thinking on top
        // and pushed a second Assistant — [Thinking, Assistant, Thinking,
        // Assistant]. Reconciliation must update the existing rows in place.
        let mut state = AppState::default();
        state.apply(ServerEvent::TurnStart {
            session_id: None,
            data: json!({}),
        });
        state.apply(ServerEvent::AssistantDelta {
            session_id: None,
            delta: "reasoning trace".into(),
            kind: "thinking".into(),
        });
        state.apply(ServerEvent::AssistantDelta {
            session_id: None,
            delta: "hello".into(),
            kind: "assistant".into(),
        });
        state.apply(ServerEvent::AssistantMessage {
            session_id: None,
            message: Message {
                role: "assistant".into(),
                content: vec![
                    crate::client::ContentBlock::Thinking {
                        text: "reasoning trace".into(),
                    },
                    crate::client::ContentBlock::Text {
                        text: "hello".into(),
                    },
                ],
            },
        });
        assert_eq!(
            describe_transcript(&state.transcript),
            vec!["Thinking".to_owned(), r#"Assistant("hello")"#.to_owned(),]
        );
    }

    #[test]
    fn multi_frame_fence_bounds_stream_then_restores_complete_source() {
        let mut state = AppState::default();
        let mut source = String::new();
        for delta in std::iter::once("```rust\n").chain(std::iter::repeat_n("let x = 1;\n", 2000)) {
            source.push_str(delta);
            state.apply(ServerEvent::AssistantDelta {
                session_id: None,
                delta: delta.into(),
                kind: "assistant".into(),
            });
        }
        assert!(
            matches!(&state.transcript[0], TranscriptEntry::Assistant(doc) if source.ends_with(doc.source.as_ref()) && doc.preview_truncated)
        );
        source.push_str("```\n");
        state.apply(ServerEvent::AssistantMessage {
            session_id: None,
            message: Message {
                role: "assistant".into(),
                content: vec![crate::client::ContentBlock::Text {
                    text: source.clone(),
                }],
            },
        });
        let TranscriptEntry::Assistant(doc) = &state.transcript[0] else {
            panic!("missing assistant")
        };
        assert_eq!(doc.source.as_ref(), source);
        assert!(!doc.preview_truncated);
    }

    #[test]
    fn approval_end_waits_for_authoritative_status() {
        let mut state = AppState::default();
        let tool = call();
        state.apply(ServerEvent::ApprovalRequest {
            session_id: None,
            approval: Approval {
                request_id: "approval-1".to_owned(),
                tool_call: tool.clone(),
            },
        });
        assert!(!state.approvals.is_empty());
        state.apply(ServerEvent::ApprovalEnd {
            session_id: None,
            tool_call: tool,
            data: json!({ "decision": "deny" }),
        });
        assert_eq!(state.approvals[0].request_id, "approval-1");
    }

    #[test]
    fn tool_receipts_preserve_one_line_summary_and_error_state() {
        let mut state = AppState::default();
        let tool = call();
        state.apply(ServerEvent::ToolStart {
            session_id: None,
            tool_call: tool.clone(),
            data: json!({}),
        });
        state.apply(ServerEvent::ToolOutput {
            session_id: None,
            tool_call: tool.clone(),
            output: "ok".to_owned(),
            data: json!({}),
        });
        state.apply(ServerEvent::ToolEnd {
            session_id: None,
            tool_call: tool,
            tool_result: Some(ToolResult {
                tool_call_id: "tool-1".to_owned(),
                content: "no".to_owned(),
                is_error: true,
                is_canceled: false,
                content_blocks: Vec::new(),
                structured_content: None,
            }),
            data: json!({}),
        });
        assert!(matches!(
            state.transcript[0],
            TranscriptEntry::Tool {
                complete: true,
                error: true,
                ..
            }
        ));
        assert!(
            matches!(&state.transcript[0], TranscriptEntry::Tool { summary, .. } if summary == "no")
        );
    }

    #[test]
    fn parallel_same_name_receipts_use_ids_and_bound_all_output() {
        let mut state = AppState::default();
        let first = call();
        let second = ToolCall {
            id: "tool-2".into(),
            ..call()
        };
        for tool in [&first, &second] {
            state.apply(ServerEvent::ToolStart {
                session_id: None,
                tool_call: tool.clone(),
                data: json!({}),
            });
        }
        state.apply(ServerEvent::ToolOutput {
            session_id: None,
            tool_call: first.clone(),
            output: "\nα".repeat(1000),
            data: json!({}),
        });
        state.apply(ServerEvent::ToolOutput {
            session_id: None,
            tool_call: second.clone(),
            output: "second".into(),
            data: json!({}),
        });
        for (tool, error) in [(second, true), (first, false)] {
            state.apply(ServerEvent::ToolEnd {
                session_id: None,
                tool_call: tool.clone(),
                tool_result: Some(ToolResult {
                    tool_call_id: tool.id,
                    content: String::new(),
                    is_error: error,
                    is_canceled: false,
                    content_blocks: vec![],
                    structured_content: None,
                }),
                data: json!({}),
            });
        }
        assert!(
            matches!(&state.transcript[0], TranscriptEntry::Tool { key, summary, complete: true, error: false, .. }
            if key.tool_call_id == "tool-1" && summary == "α" && !summary.contains('\n'))
        );
        assert!(
            matches!(&state.transcript[1], TranscriptEntry::Tool { key, summary, complete: true, error: true, .. }
            if key.tool_call_id == "tool-2" && summary.ends_with("second"))
        );
        let huge = ToolCall {
            arguments: [("arg".into(), json!("x".repeat(1000)))]
                .into_iter()
                .collect(),
            ..call()
        };
        assert!(
            matches!(tool_entry(&huge, ToolReceiptKey::new(None, &json!({}), &huge)), TranscriptEntry::Tool { summary, .. } if summary.chars().count() == SUMMARY_CHARS)
        );
    }

    #[test]
    fn delegated_receipts_with_duplicate_raw_ids_keep_their_own_output_and_result() {
        let mut state = AppState::default();
        // Match the server's delegated lifecycle shape: the session is the parent,
        // while data.agent_instance_id identifies each child.
        for agent in [None, Some("child-one"), Some("child-two")] {
            state.apply(ServerEvent::ToolStart {
                session_id: Some("parent".into()),
                tool_call: call(),
                data: json!({"agent_instance_id":agent}),
            });
        }
        // Finish out of order, updating exactly one receipt each time.
        for (agent, output, error) in [
            (Some("child-two"), "second", true),
            (None, "parent", false),
            (Some("child-one"), "first", false),
        ] {
            state.apply(ServerEvent::ToolOutput {
                session_id: Some("parent".into()),
                tool_call: call(),
                output: output.into(),
                data: json!({"agent_instance_id":agent}),
            });
            state.apply(ServerEvent::ToolEnd {
                session_id: Some("parent".into()),
                tool_call: call(),
                tool_result: Some(ToolResult {
                    tool_call_id: call().id,
                    content: output.into(),
                    is_error: error,
                    is_canceled: false,
                    content_blocks: vec![],
                    structured_content: None,
                }),
                data: json!({"agent_instance_id":agent}),
            });
        }
        assert_eq!(state.transcript.len(), 3);
        for (index, output, expected_error) in [
            (0, "parent", false),
            (1, "first", false),
            (2, "second", true),
        ] {
            assert!(
                matches!(&state.transcript[index], TranscriptEntry::Tool { summary, complete: true, error, .. }
                if summary.ends_with(output) && *error == expected_error)
            );
        }
        // A different parent session must not reuse an existing receipt either.
        state.apply(ServerEvent::ToolOutput {
            session_id: Some("other-parent".into()),
            tool_call: call(),
            output: "other session".into(),
            data: json!({"agent_instance_id":"child-one"}),
        });
        assert_eq!(state.transcript.len(), 4);
        assert!(
            matches!(&state.transcript[3], TranscriptEntry::Tool { complete: false, summary, .. }
            if summary.ends_with("other session"))
        );
    }

    #[test]
    fn errors_stop_streaming_and_render_as_distinct_block() {
        let mut state = AppState {
            streaming: true,
            ..Default::default()
        };
        state.apply(ServerEvent::Error {
            session_id: None,
            error: EventError {
                code: "backend_error".to_owned(),
                message: "provider failed".to_owned(),
            },
            data: json!({}),
        });
        assert!(!state.streaming);
        assert!(matches!(state.transcript[0], TranscriptEntry::Error { .. }));
    }

    #[test]
    fn settings_recovery_uses_codes_not_error_text() {
        for (code, message, expected) in [
            ("model_access_error", "Access denied", true),
            ("model_reverted", "Restored previous selection", true),
            ("auth_error", "MCP OAuth credentials expired", false),
            ("backend_error", "MCP OAuth credentials expired", false),
            ("http_error", "Model not found", false),
        ] {
            let mut state = AppState::default();
            state.apply(ServerEvent::Error {
                session_id: None,
                error: EventError {
                    code: code.into(),
                    message: message.into(),
                },
                data: json!({}),
            });
            assert!(matches!(&state.transcript[0], TranscriptEntry::Error {
                message: text, settings_action, ..
            } if text == message && *settings_action == expected));
        }
    }

    /// Assemble a stream: TurnStart, one or more AssistantDelta chunks, then
    /// the final `AssistantMessage`. Returns the resulting transcript so the
    /// test can compare live and replay shapes byte for byte.
    fn stream_turn(state: &mut AppState, deltas: &[(&str, &str)], final_blocks: Vec<ContentBlock>) {
        state.apply(ServerEvent::TurnStart {
            session_id: None,
            data: json!({}),
        });
        for (kind, delta) in deltas {
            state.apply(ServerEvent::AssistantDelta {
                session_id: None,
                delta: (*delta).into(),
                kind: (*kind).into(),
            });
        }
        state.apply(ServerEvent::AssistantMessage {
            session_id: None,
            message: Message {
                role: "assistant".into(),
                content: final_blocks,
            },
        });
        state.apply(ServerEvent::TurnEnd {
            session_id: None,
            data: json!({}),
        });
    }

    #[test]
    fn final_only_second_turn_appends_without_editing_the_first_turn() {
        // Turn 1 leaves an Assistant row on the transcript. Turn 2 emits no
        // user prompt and no streamed deltas — only a final AssistantMessage.
        // The old User/Error boundary scan treated the previous turn's row as
        // inside the current turn and overwrote it. Reconciliation must anchor
        // on the TurnStart index so Turn 2's text lands on a NEW row.
        let mut state = AppState::default();
        stream_turn(
            &mut state,
            &[],
            vec![ContentBlock::Text {
                text: "first turn".into(),
            }],
        );
        stream_turn(
            &mut state,
            &[],
            vec![ContentBlock::Text {
                text: "second turn".into(),
            }],
        );
        assert_eq!(
            describe_transcript(&state.transcript),
            vec![
                r#"Assistant("first turn")"#.to_owned(),
                r#"Assistant("second turn")"#.to_owned(),
            ]
        );
    }

    #[test]
    fn thinking_only_second_turn_emits_its_own_marker_not_the_previous_one() {
        // A pure-thinking turn following an assistant turn must push its own
        // marker rather than reuse the earlier turn's Thinking row (which is
        // what happens if reconciliation scans without a turn boundary).
        let mut state = AppState::default();
        stream_turn(
            &mut state,
            &[("thinking", "reasoning"), ("assistant", "hello")],
            vec![
                ContentBlock::Thinking {
                    text: "reasoning".into(),
                },
                ContentBlock::Text {
                    text: "hello".into(),
                },
            ],
        );
        stream_turn(
            &mut state,
            &[],
            vec![ContentBlock::Thinking {
                text: "silent turn".into(),
            }],
        );
        assert_eq!(
            describe_transcript(&state.transcript),
            vec![
                "Thinking".to_owned(),
                r#"Assistant("hello")"#.to_owned(),
                "Thinking".to_owned(),
            ],
            "third row is a fresh Thinking marker, not a merge into the first turn's marker"
        );
    }

    #[test]
    fn text_only_final_message_carries_over_one_row() {
        // Baseline: a turn with no thinking and no streamed deltas produces a
        // single Assistant row.
        let mut state = AppState::default();
        stream_turn(
            &mut state,
            &[],
            vec![ContentBlock::Text {
                text: "answer".into(),
            }],
        );
        assert_eq!(
            describe_transcript(&state.transcript),
            vec![r#"Assistant("answer")"#.to_owned()]
        );
    }

    #[test]
    fn mixed_streamed_then_final_matches_replay_of_committed_events_only() {
        // The live state receives EVERY event — TurnStart, streamed deltas,
        // the final AssistantMessage, TurnEnd. The replay state receives
        // ONLY the committed events (TurnStart / AssistantMessage / TurnEnd)
        // that history playback carries. If both produce the same
        // transcript, reconciliation is idempotent under history replay —
        // this is the round-6 bug the earlier test hid by feeding identical
        // event streams to both states.
        let session = "session";
        let live_events = vec![
            ServerEvent::TurnStart {
                session_id: Some(session.to_owned()),
                data: json!({}),
            },
            ServerEvent::AssistantDelta {
                session_id: Some(session.to_owned()),
                delta: "trace".into(),
                kind: "thinking".into(),
            },
            ServerEvent::AssistantDelta {
                session_id: Some(session.to_owned()),
                delta: "hi".into(),
                kind: "assistant".into(),
            },
            ServerEvent::AssistantMessage {
                session_id: Some(session.to_owned()),
                message: Message {
                    role: "assistant".into(),
                    content: vec![
                        ContentBlock::Thinking {
                            text: "trace".into(),
                        },
                        ContentBlock::Text { text: "hi".into() },
                    ],
                },
            },
            ServerEvent::TurnEnd {
                session_id: Some(session.to_owned()),
                data: json!({}),
            },
            ServerEvent::TurnStart {
                session_id: Some(session.to_owned()),
                data: json!({}),
            },
            ServerEvent::AssistantMessage {
                session_id: Some(session.to_owned()),
                message: Message {
                    role: "assistant".into(),
                    content: vec![ContentBlock::Text {
                        text: "followup".into(),
                    }],
                },
            },
            ServerEvent::TurnEnd {
                session_id: Some(session.to_owned()),
                data: json!({}),
            },
        ];
        // Replay drops the streamed deltas — a rebuilt session hydrates
        // through the committed messages only, then this event stream fires.
        let replay_events: Vec<ServerEvent> = live_events
            .iter()
            .filter(|event| !matches!(event, ServerEvent::AssistantDelta { .. }))
            .cloned()
            .collect();
        let mut live = AppState::default();
        for event in live_events {
            live.apply(event);
        }
        let mut replay = AppState::default();
        for event in replay_events {
            replay.apply(event);
        }
        let live_shape = describe_transcript(&live.transcript);
        let replay_shape = describe_transcript(&replay.transcript);
        assert_eq!(
            live_shape,
            vec![
                "Thinking".to_owned(),
                r#"Assistant("hi")"#.to_owned(),
                r#"Assistant("followup")"#.to_owned(),
            ],
            "live transcript regressed: {live_shape:?}"
        );
        assert_eq!(
            live_shape, replay_shape,
            "live vs replay diverge — reconciliation is not idempotent"
        );
    }

    /// Compact per-row descriptor: kind + the row's own dynamic text. The
    /// five reconciliation tests below compare full vectors of descriptors
    /// so a regression that flips a row kind OR a body string fails with a
    /// readable diff, not a `matches!` slot check.
    #[cfg(test)]
    fn describe_transcript(rows: &[TranscriptEntry]) -> Vec<String> {
        rows.iter()
            .map(|entry| match entry {
                TranscriptEntry::User(text) => format!("User({text:?})"),
                TranscriptEntry::Assistant(doc) => {
                    format!("Assistant({:?})", doc.source.as_ref())
                }
                TranscriptEntry::Thinking => "Thinking".to_owned(),
                TranscriptEntry::Tool { name, summary, .. } => {
                    format!("Tool({name:?}, {summary:?})")
                }
                TranscriptEntry::Error { message, .. } => format!("Error({message:?})"),
            })
            .collect()
    }

    #[test]
    fn interleaved_assistant_and_tool_row_reconcile_without_duplicating_prefix() {
        // ROUND-7 regression: streamed sequence AssistantDelta("pre"),
        // ToolStart, AssistantDelta("post"), then the final AssistantMessage
        // carries "prepost". The old `rposition` reconciler replaced the
        // LAST assistant row with the full "prepost" and left the earlier
        // "pre" row untouched — "pre" appeared twice on screen. The new
        // reconciler preserves row order around tools and puts the
        // remainder in the trailing row so their concatenation matches.
        let mut state = AppState::default();
        state.apply(ServerEvent::TurnStart {
            session_id: None,
            data: json!({}),
        });
        state.apply(ServerEvent::AssistantDelta {
            session_id: None,
            delta: "pre".into(),
            kind: "assistant".into(),
        });
        state.apply(ServerEvent::ToolStart {
            session_id: None,
            tool_call: call(),
            data: json!({}),
        });
        state.apply(ServerEvent::AssistantDelta {
            session_id: None,
            delta: "post".into(),
            kind: "assistant".into(),
        });
        state.apply(ServerEvent::AssistantMessage {
            session_id: None,
            message: Message {
                role: "assistant".into(),
                content: vec![ContentBlock::Text {
                    text: "prepost".into(),
                }],
            },
        });
        assert_eq!(
            describe_transcript(&state.transcript),
            vec![
                r#"Assistant("pre")"#.to_owned(),
                r#"Tool("read", "{\"path\":\"README.md\"}")"#.to_owned(),
                r#"Assistant("post")"#.to_owned(),
            ]
        );
    }

    #[test]
    fn resumed_stream_after_history_replay_does_not_edit_older_rows() {
        // ROUND-7 regression: `apply_history(replace=true)` used to leave
        // `turn_start` at 0. A resumed in-flight AssistantMessage arriving
        // just after history replay would then scan from index 0 and
        // overwrite an OLD assistant row from the restored history. Anchor
        // must move to the rebuilt tail so the resumed message appends.
        use crate::client::{HistoryContent, HistoryMessage};
        let mut state = AppState::default();
        let history = vec![
            HistoryMessage {
                id: "u1".into(),
                role: "user".into(),
                content: vec![HistoryContent::Text {
                    text: "old question".into(),
                }],
                tool_result: None,
            },
            HistoryMessage {
                id: "a1".into(),
                role: "assistant".into(),
                content: vec![HistoryContent::Text {
                    text: "old answer".into(),
                }],
                tool_result: None,
            },
        ];
        state.apply_history(history, true);
        // Resumed in-flight message arrives without a fresh TurnStart —
        // exactly what the server emits when the client reconnects mid-turn.
        state.apply(ServerEvent::AssistantMessage {
            session_id: None,
            message: Message {
                role: "assistant".into(),
                content: vec![ContentBlock::Text {
                    text: "resumed answer".into(),
                }],
            },
        });
        assert_eq!(
            describe_transcript(&state.transcript),
            vec![
                r#"User("old question")"#.to_owned(),
                r#"Assistant("old answer")"#.to_owned(),
                r#"Assistant("resumed answer")"#.to_owned(),
            ]
        );
    }

    #[test]
    fn aborted_turn_leaves_the_streamed_row_intact() {
        // TurnAborted stops streaming without a final AssistantMessage.
        // Any streamed assistant text must remain visible so the user sees
        // what the model said before the cancel. This test also proves
        // aborted turns do not corrupt turn_start for a subsequent turn.
        let mut state = AppState::default();
        state.apply(ServerEvent::TurnStart {
            session_id: None,
            data: json!({}),
        });
        state.apply(ServerEvent::AssistantDelta {
            session_id: None,
            delta: "partial".into(),
            kind: "assistant".into(),
        });
        state.apply(ServerEvent::TurnAborted {
            session_id: None,
            data: json!({}),
        });
        state.apply(ServerEvent::TurnStart {
            session_id: None,
            data: json!({}),
        });
        state.apply(ServerEvent::AssistantMessage {
            session_id: None,
            message: Message {
                role: "assistant".into(),
                content: vec![ContentBlock::Text {
                    text: "next turn".into(),
                }],
            },
        });
        assert_eq!(
            describe_transcript(&state.transcript),
            vec![
                r#"Assistant("partial")"#.to_owned(),
                r#"Assistant("next turn")"#.to_owned(),
            ]
        );
    }

    #[test]
    fn empty_final_message_after_streamed_deltas_keeps_streamed_text() {
        // A final AssistantMessage with empty text (or only-whitespace)
        // must not erase or duplicate the streamed row. Some providers emit
        // an empty final envelope when the response is tool-only.
        let mut state = AppState::default();
        state.apply(ServerEvent::TurnStart {
            session_id: None,
            data: json!({}),
        });
        state.apply(ServerEvent::AssistantDelta {
            session_id: None,
            delta: "streamed".into(),
            kind: "assistant".into(),
        });
        state.apply(ServerEvent::AssistantMessage {
            session_id: None,
            message: Message {
                role: "assistant".into(),
                content: vec![ContentBlock::Text { text: "".into() }],
            },
        });
        assert_eq!(
            describe_transcript(&state.transcript),
            vec![r#"Assistant("streamed")"#.to_owned()]
        );
    }

    #[test]
    fn final_shorter_than_streamed_drops_the_trailing_blank_row() {
        // Round-8 fix: streamed "pre", ToolStart, streamed "post", final
        // AssistantMessage("pre"). The leading row already carries "pre",
        // so the trailing assistant row's tail is empty. Blanking it would
        // leave an empty row that still consumes transcript rhythm — drop
        // the row instead.
        let mut state = AppState::default();
        state.apply(ServerEvent::TurnStart {
            session_id: None,
            data: json!({}),
        });
        state.apply(ServerEvent::AssistantDelta {
            session_id: None,
            delta: "pre".into(),
            kind: "assistant".into(),
        });
        state.apply(ServerEvent::ToolStart {
            session_id: None,
            tool_call: call(),
            data: json!({}),
        });
        state.apply(ServerEvent::AssistantDelta {
            session_id: None,
            delta: "post".into(),
            kind: "assistant".into(),
        });
        state.apply(ServerEvent::AssistantMessage {
            session_id: None,
            message: Message {
                role: "assistant".into(),
                content: vec![ContentBlock::Text { text: "pre".into() }],
            },
        });
        assert_eq!(
            describe_transcript(&state.transcript),
            vec![
                r#"Assistant("pre")"#.to_owned(),
                r#"Tool("read", "{\"path\":\"README.md\"}")"#.to_owned(),
            ],
            "reconciliation must never leave an empty assistant row"
        );
    }

    #[test]
    fn middle_row_removal_reports_the_exact_removed_index() {
        // Round-9: a blank-row removal at index 2 leaves rows AFTER it in
        // the transcript (a trailing Tool row here at old index 3). The
        // change signal must carry that removed index so the view splices
        // the virtual list at slot 2 — not at the tail, which would leave
        // shifted rows anchored to their pre-removal cached heights.
        let mut state = AppState::default();
        state.apply(ServerEvent::TurnStart {
            session_id: None,
            data: json!({}),
        });
        state.apply(ServerEvent::AssistantDelta {
            session_id: None,
            delta: "pre".into(),
            kind: "assistant".into(),
        });
        state.apply(ServerEvent::ToolStart {
            session_id: None,
            tool_call: call(),
            data: json!({}),
        });
        state.apply(ServerEvent::AssistantDelta {
            session_id: None,
            delta: "post".into(),
            kind: "assistant".into(),
        });
        state.apply(ServerEvent::ToolStart {
            session_id: None,
            tool_call: call(),
            data: json!({}),
        });
        // Transcript before the final message:
        //   [Assistant("pre"), Tool, Assistant("post"), Tool]
        // Final "pre" reconciles by dropping Assistant("post") at index 2.
        let change = state.apply(ServerEvent::AssistantMessage {
            session_id: None,
            message: Message {
                role: "assistant".into(),
                content: vec![ContentBlock::Text { text: "pre".into() }],
            },
        });
        assert_eq!(
            change,
            Some(TranscriptChange::Removed(2)),
            "the exact removed index must reach the view — the tail delta alone would splice the wrong slot"
        );
        assert_eq!(
            describe_transcript(&state.transcript),
            vec![
                r#"Assistant("pre")"#.to_owned(),
                r#"Tool("read", "{\"path\":\"README.md\"}")"#.to_owned(),
                r#"Tool("read", "{\"path\":\"README.md\"}")"#.to_owned(),
            ],
        );
    }

    #[test]
    fn multiple_tools_with_shorter_final_drop_every_blank_trailing_row() {
        // Two tools bracket streamed text on both sides; the final message
        // matches only the first fragment. Every empty tail must be
        // removed — a lingering blank row breaks transcript rhythm.
        let mut state = AppState::default();
        state.apply(ServerEvent::TurnStart {
            session_id: None,
            data: json!({}),
        });
        state.apply(ServerEvent::AssistantDelta {
            session_id: None,
            delta: "alpha".into(),
            kind: "assistant".into(),
        });
        state.apply(ServerEvent::ToolStart {
            session_id: None,
            tool_call: call(),
            data: json!({}),
        });
        state.apply(ServerEvent::AssistantDelta {
            session_id: None,
            delta: "beta".into(),
            kind: "assistant".into(),
        });
        state.apply(ServerEvent::ToolStart {
            session_id: None,
            tool_call: call(),
            data: json!({}),
        });
        state.apply(ServerEvent::AssistantDelta {
            session_id: None,
            delta: "gamma".into(),
            kind: "assistant".into(),
        });
        state.apply(ServerEvent::AssistantMessage {
            session_id: None,
            message: Message {
                role: "assistant".into(),
                content: vec![ContentBlock::Text {
                    text: "alphabeta".into(),
                }],
            },
        });
        assert_eq!(
            describe_transcript(&state.transcript),
            vec![
                r#"Assistant("alpha")"#.to_owned(),
                r#"Tool("read", "{\"path\":\"README.md\"}")"#.to_owned(),
                r#"Assistant("beta")"#.to_owned(),
                r#"Tool("read", "{\"path\":\"README.md\"}")"#.to_owned(),
            ],
            "trailing empty assistant row after multiple tools must be dropped"
        );
    }

    #[test]
    fn empty_delta_around_tool_leaves_no_blank_row() {
        // Empty deltas ("") never push their own assistant row (see the
        // `AssistantDelta` handler's `!delta.is_empty()` guard), so the
        // reconciler only ever sees the non-empty streamed fragments plus
        // the final message. Even if a provider bookends a tool with empty
        // deltas around a real fragment, the transcript still ends with no
        // blank assistant row when the final text equals the streamed
        // prefix.
        let mut state = AppState::default();
        state.apply(ServerEvent::TurnStart {
            session_id: None,
            data: json!({}),
        });
        state.apply(ServerEvent::AssistantDelta {
            session_id: None,
            delta: "".into(),
            kind: "assistant".into(),
        });
        state.apply(ServerEvent::AssistantDelta {
            session_id: None,
            delta: "hello".into(),
            kind: "assistant".into(),
        });
        state.apply(ServerEvent::ToolStart {
            session_id: None,
            tool_call: call(),
            data: json!({}),
        });
        state.apply(ServerEvent::AssistantDelta {
            session_id: None,
            delta: "".into(),
            kind: "assistant".into(),
        });
        state.apply(ServerEvent::AssistantMessage {
            session_id: None,
            message: Message {
                role: "assistant".into(),
                content: vec![ContentBlock::Text {
                    text: "hello".into(),
                }],
            },
        });
        assert_eq!(
            describe_transcript(&state.transcript),
            vec![
                r#"Assistant("hello")"#.to_owned(),
                r#"Tool("read", "{\"path\":\"README.md\"}")"#.to_owned(),
            ]
        );
    }

    #[test]
    fn reconnect_state_is_visible_and_explicit() {
        let mut state = AppState::default();
        state.mark_connection_lost("server closed the connection");
        assert!(matches!(state.connection, ConnectionState::Lost(_)));
        state.begin_reconnect();
        assert_eq!(state.connection, ConnectionState::Reconnecting);
    }
}
