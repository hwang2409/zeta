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

// The `Tool` variant carries a `Card` inline — expanded state, output
// tail (bounded at 16 KiB), agent label, child id, and now a per-turn
// stamp. The size disparity between variants is acknowledged: boxing
// the card would ripple auto-deref through dozens of pattern matches
// across the state / render / row_text seams and pay heap traffic for
// every receipt in the transcript. The enum is not clone-hot on any
// path (transcripts hold entries by owned index, not by value copies),
// so the inline layout stays.
#[allow(clippy::large_enum_variant)]
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
        /// One-line excerpt of what the tool actually ran, derived from
        /// `tool_call.arguments` at construction and stable for the lifetime
        /// of the receipt. Bash/exec → first line of the command; read/write
        /// /edit → the file path; fetch → the URL; anything else → the first
        /// primitive argument, else the tool name. This becomes the row's
        /// primary text; the tool name becomes a small leading label.
        excerpt: String,
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
}

/// Minimum consecutive tool receipts that collapse into a single group
/// summary row. Below this threshold the receipts render one per row. The
/// contract pins this at "3+ consecutive"; keeping the constant on
/// `state.rs` places it next to the code that reads it.
pub const TOOL_GROUP_MIN_LEN: usize = 3;

/// Cap on the number of excerpts previewed on a collapsed group's summary
/// row. The contract asks for "the first 1-2 excerpts previewed"; two
/// keeps the row readable at the 11px scale.
pub const TOOL_GROUP_PREVIEW_MAX: usize = 2;

/// Where in a run of consecutive tool receipts a given transcript index
/// sits. Returned by `AppState::tool_group_position` and consumed by the
/// render layer, which dispatches the group summary row on `Start` and
/// skips interior rows on `Interior` when the group is collapsed. `None`
/// means the row is not part of a 3+ run (either not a tool row, or in a
/// shorter run that renders one row per receipt).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolGroupPosition {
    pub first_index: usize,
    pub last_index: usize,
    pub first_id: String,
    /// The turn every receipt in this group belongs to. Grouping never
    /// crosses turns (see `AppState::tool_group_position`), so a receipt
    /// from a later turn opens a NEW group even when it sits directly
    /// after this one in the transcript.
    pub turn: u64,
}

impl ToolGroupPosition {
    pub fn count(&self) -> usize {
        self.last_index - self.first_index + 1
    }

    pub fn is_start(&self, index: usize) -> bool {
        index == self.first_index
    }
}

/// Single source of truth for the thinking marker text. Read by
/// `row_text::build` when composing the model for a `Thinking` row and by
/// the guard tests that sweep for stray markers.
pub const THINKING_HEADER_LABEL: &str = "+ Thought";

/// Single source of truth for the error row header text. Read by
/// `row_text::build` and by the guard tests.
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

/// One primitive edit performed on the transcript. `AppState::apply` returns
/// the ORDERED list of edits that describe a reconcile completely — a single
/// event may append a row AND drop another row AND remeasure a third, and
/// the view applies each edit in order to keep its virtual-list metadata
/// cache aligned to the transcript. A one-action signal cannot describe a
/// compound reconcile (append Thinking + remove middle assistant), so the
/// view would splice only one operation and drift out of sync.
///
/// Every index is expressed against the transcript state the view holds at
/// the moment that edit is applied — earlier edits in the same batch have
/// already been applied. State.rs emits removals in descending order so an
/// earlier `Remove(hi)` never invalidates a later `Remove(lo)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscriptEdit {
    /// A new row was inserted at `index`. The view splices `index..index`
    /// with one new row, growing the list by one.
    Insert(usize),
    /// The row at `index` was removed. The view splices `index..index + 1`
    /// out, shrinking the list by one; downstream rows shift left.
    Remove(usize),
    /// The row at `index` was mutated in place. The view remeasures exactly
    /// that row; count is unchanged.
    Remeasure(usize),
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
    /// Monotonic counter that increments on every `TurnStart`. Every tool
    /// receipt created inside a turn stamps its `Card::turn` with this
    /// value; `tool_group_position` compares turns before joining
    /// receipts into a group so a run of tool rows that spans a turn
    /// boundary paints as two groups instead of one. Streaming forces
    /// the CURRENT turn's groups expanded — every group whose turn
    /// matches `current_turn` while `streaming` is true, not just the
    /// trailing one — so a mid-turn assistant message that lands between
    /// two tool bursts still leaves both bursts on screen without a click.
    pub current_turn: u64,
    pub metrics: StatusMetrics,
    pub metrics_boundary: bool,
    /// Per-group expansion override, keyed by the `tool_call_id` of the
    /// group's FIRST tool receipt. Default is collapsed; the map holds
    /// `true` when the user has explicitly expanded a group. Groups form
    /// dynamically as consecutive tool rows accumulate (see
    /// `AppState::tool_group_of`), so keying by the first row's stable id
    /// keeps state pinned across a group that grows a new tail.
    /// Streaming turns force every group in the active turn expanded
    /// regardless of the map — see `AppState::tool_group_expanded`.
    pub tool_group_expanded: HashMap<String, bool>,
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
            current_turn: 0,
            metrics: StatusMetrics::default(),
            metrics_boundary: true,
            tool_group_expanded: HashMap::new(),
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
                // Each replayed assistant message opens a HISTORICAL turn so
                // its tool receipts land under a `card.turn` that no future
                // streamed `TurnStart` will re-use. Without this bump the
                // restored receipts stamp `current_turn` verbatim — a
                // mid-turn resume then treats every historical group as
                // the active turn and paints it force-expanded. Bump
                // BEFORE the tool_use loop so its `apply(ToolStart)`
                // stamps the fresh historical value.
                self.current_turn = self.current_turn.saturating_add(1);
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
            // Advance past every historical turn stamp we handed out during
            // this replay. `is_tool_group_expanded` uses `group.turn ==
            // current_turn` to decide whether streaming should force a
            // group expanded; leaving `current_turn` on the LAST historical
            // stamp would let a mid-turn resume paint that group expanded
            // even though it belongs to completed history. This bump keeps
            // historical turns strictly below `current_turn` so only
            // GENUINELY new streamed turns force expansion.
            self.current_turn = self.current_turn.saturating_add(1);
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

    pub fn apply(&mut self, event: ServerEvent) -> Vec<TranscriptEdit> {
        let mut edits: Vec<TranscriptEdit> = Vec::new();
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
                // Every tool receipt in the new turn stamps this counter on
                // its `Card::turn`, so `tool_group_position` refuses to fold
                // a tool row from the previous turn into a group with the
                // fresh turn's tools. `saturating_add` keeps the counter safe
                // under a synthetic pathological session.
                self.current_turn = self.current_turn.saturating_add(1);
            }
            ServerEvent::AgentEnd { .. } | ServerEvent::TurnAborted { .. } => {
                // Streaming force-expanded every group in the LIVE turn;
                // ending the stream flips those groups back to their map
                // default (collapsed unless the user opened them). Row
                // shapes change — the interior rows collapse to zero
                // height and the summary row swaps in — so ROUTE the
                // shape change through the ordered edit list so the
                // virtual-list remeasures each affected row. ZETA-107
                // invariant: ALL shape changes travel `TranscriptEdit`.
                edits.extend(self.remeasure_current_turn_groups());
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
                let appended = match self.transcript.last_mut() {
                    Some(TranscriptEntry::Assistant(text)) => {
                        text.push_str(&delta);
                        false
                    }
                    _ => {
                        self.transcript
                            .push(TranscriptEntry::Assistant(Markdown::streaming(delta)));
                        true
                    }
                };
                if let Some(index) = self.transcript.len().checked_sub(1) {
                    edits.push(if appended {
                        TranscriptEdit::Insert(index)
                    } else {
                        TranscriptEdit::Remeasure(index)
                    });
                }
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
                    if let Some(index) = self.transcript.len().checked_sub(1) {
                        edits.push(TranscriptEdit::Insert(index));
                    }
                }
            }
            ServerEvent::AssistantDelta { .. } => {}
            ServerEvent::AssistantMessage { message, .. } => {
                self.thinking = false;
                self.assistant_started |= !message.text().is_empty();
                edits.extend(self.commit_assistant(message));
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
                    self.current_turn,
                ));
                if let Some(index) = self.transcript.len().checked_sub(1) {
                    edits.push(TranscriptEdit::Insert(index));
                    // When the freshly inserted receipt is the one that
                    // pushes an adjacent same-turn run over
                    // TOOL_GROUP_MIN_LEN, the pre-existing rows in that
                    // run change render shape: the first row gains a
                    // group header (wrapped above the receipt) and the
                    // interiors switch from single-receipt paint to
                    // group-interior paint. Route these shape changes
                    // through the ordered edit list so the virtual-list
                    // remeasures each row. Only the transition round
                    // (size == MIN_LEN) needs the sweep — subsequent
                    // arrivals grow the group but do not re-shape the
                    // earlier members.
                    if let Some(group) = self.tool_group_position(index) {
                        if group.count() == TOOL_GROUP_MIN_LEN {
                            for i in group.first_index..index {
                                edits.push(TranscriptEdit::Remeasure(i));
                            }
                        }
                    }
                }
            }
            ServerEvent::ToolOutput {
                session_id,
                tool_call,
                output,
                data,
            } => {
                let (index, inserted, entry) = self.tool_receipt(
                    &tool_call,
                    ToolReceiptKey::new(session_id, &data, &tool_call),
                );
                edits.push(if inserted {
                    TranscriptEdit::Insert(index)
                } else {
                    TranscriptEdit::Remeasure(index)
                });
                if let TranscriptEntry::Tool { summary, card, .. } = entry {
                    card.tail.append(&output);
                    card.streamed = true;
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
                let (index, inserted, entry) = self.tool_receipt(
                    &tool_call,
                    ToolReceiptKey::new(session_id, &data, &tool_call),
                );
                edits.push(if inserted {
                    TranscriptEdit::Insert(index)
                } else {
                    TranscriptEdit::Remeasure(index)
                });
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
                        // Bash-shaped tools stream stdout via `ToolOutput`
                        // AND repeat the whole thing in the final result
                        // (src/zeta/tools/bash.py:202). Once `streamed`
                        // is set, the streamed chunks are already the
                        // authoritative content — skip the append so
                        // `bytes_seen` and the on-screen tail don't
                        // double-count. Agent-style tools that only
                        // report through `ToolEnd` keep `streamed=false`
                        // and still append here.
                        if !card.streamed && card.tail.text != result.content {
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
                edits.push(self.commit_sub_agent(session_id, receipt));
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
                if let Some(index) = self.transcript.len().checked_sub(1) {
                    edits.push(TranscriptEdit::Insert(index));
                }
            }
            ServerEvent::Other { .. } => {}
        }
        edits
    }

    pub fn toggle_card(&mut self, index: usize) {
        if let Some(TranscriptEntry::Tool { card, .. }) = self.transcript.get_mut(index) {
            card.toggle();
        }
    }

    /// Locate the tool-receipt group that contains `index`, if any. A run
    /// of `TOOL_GROUP_MIN_LEN` or more consecutive `TranscriptEntry::Tool`
    /// rows FROM THE SAME TURN counts as a group; below that threshold
    /// each receipt renders on its own row. A turn boundary between two
    /// adjacent tool rows breaks the run — the reviewer's r2 scenario
    /// (2 tools in turn A followed by 1 tool in turn B) paints as two
    /// separate units, never a 3-group. Returns `None` when the row is
    /// not a Tool, or when the surrounding same-turn run is too short to
    /// group.
    pub fn tool_group_position(&self, index: usize) -> Option<ToolGroupPosition> {
        let turn = match self.transcript.get(index) {
            Some(TranscriptEntry::Tool { card, .. }) => card.turn,
            _ => return None,
        };
        let same_turn_tool = |i: usize| -> bool {
            matches!(
                self.transcript.get(i),
                Some(TranscriptEntry::Tool { card, .. }) if card.turn == turn
            )
        };
        let mut first = index;
        while first > 0 && same_turn_tool(first - 1) {
            first -= 1;
        }
        let mut last = index;
        while last + 1 < self.transcript.len() && same_turn_tool(last + 1) {
            last += 1;
        }
        if last - first + 1 < TOOL_GROUP_MIN_LEN {
            return None;
        }
        let first_id = match &self.transcript[first] {
            TranscriptEntry::Tool { key, .. } => key.tool_call_id.clone(),
            _ => return None,
        };
        Some(ToolGroupPosition {
            first_index: first,
            last_index: last,
            first_id,
            turn,
        })
    }

    /// Is the given group currently expanded on screen? Groups default to
    /// collapsed; the map holds `true` for groups the user opened. While
    /// streaming, EVERY group in the current turn (`group.turn ==
    /// self.current_turn`) is forced expanded so live tool activity stays
    /// visible without a click — the reviewer's r2 scenario (mid-turn
    /// assistant text splits one turn into two tool bursts) expands both
    /// bursts, not only the trailing one.
    pub fn is_tool_group_expanded(&self, group: &ToolGroupPosition) -> bool {
        if self.streaming && group.turn == self.current_turn {
            return true;
        }
        self.tool_group_expanded
            .get(&group.first_id)
            .copied()
            .unwrap_or(false)
    }

    /// Flip a group's expansion override. Groups default to collapsed
    /// (absent from the map); the first toggle stores `true`, the next
    /// removes the entry, keeping the map small over a long session.
    pub fn toggle_tool_group(&mut self, first_id: &str) {
        if self
            .tool_group_expanded
            .get(first_id)
            .copied()
            .unwrap_or(false)
        {
            self.tool_group_expanded.remove(first_id);
        } else {
            self.tool_group_expanded.insert(first_id.to_owned(), true);
        }
    }

    /// Emit `Remeasure` edits for every row that belongs to a tool group
    /// whose `turn` matches the LIVE `current_turn`. Called from streaming
    /// termination handlers (`AgentEnd`, `TurnAborted`) — streaming forces
    /// the current turn's groups expanded, so ending the stream flips
    /// them back to their map default (collapsed unless the user opened
    /// them). Every row in each affected group changes render shape and
    /// must ride the ordered edit list per the ZETA-107 invariant.
    fn remeasure_current_turn_groups(&self) -> Vec<TranscriptEdit> {
        let mut edits = Vec::new();
        let mut cursor = 0;
        while cursor < self.transcript.len() {
            if let Some(group) = self.tool_group_position(cursor) {
                if group.turn == self.current_turn {
                    for i in group.first_index..=group.last_index {
                        edits.push(TranscriptEdit::Remeasure(i));
                    }
                }
                cursor = group.last_index + 1;
            } else {
                cursor += 1;
            }
        }
        edits
    }

    /// Sum the cumulative bytes flowed through every receipt in the group.
    /// Used by the render layer to compose the summary row's total-size
    /// metadata. Reads `bytes_seen` on each receipt's tail — the total
    /// output the tool produced — so the summary stays consistent with
    /// each row's own size label even after the per-tail 20-line /
    /// 16k-char truncation clipped some bytes off screen.
    pub fn tool_group_output_bytes(&self, group: &ToolGroupPosition) -> usize {
        (group.first_index..=group.last_index)
            .filter_map(|i| match self.transcript.get(i) {
                Some(TranscriptEntry::Tool { card, .. }) => Some(card.tail.bytes_seen),
                _ => None,
            })
            .sum()
    }

    fn tool_receipt(
        &mut self,
        tool_call: &ToolCall,
        key: ToolReceiptKey,
    ) -> (usize, bool, &mut TranscriptEntry) {
        let existing = self.transcript.iter().position(
            |entry| matches!(entry, TranscriptEntry::Tool { key: found, .. } if found == &key),
        );
        let (index, inserted) = match existing {
            Some(index) => (index, false),
            None => {
                self.transcript
                    .push(tool_entry(tool_call, key, self.current_turn));
                (self.transcript.len() - 1, true)
            }
        };
        (index, inserted, &mut self.transcript[index])
    }

    fn commit_sub_agent(
        &mut self,
        session_id: Option<String>,
        receipt: SubAgentReceipt,
    ) -> TranscriptEdit {
        // Durable notifications have no raw tool ID. The launch result supplies
        // the child identity when we saw it; reconnect drains can create it alone.
        let existing = self.transcript.iter().position(|entry| {
            matches!(entry, TranscriptEntry::Tool { key, card, .. }
                if key.session_id == session_id
                    && card.child_instance_id.as_deref() == Some(&receipt.child_instance_id))
        });
        let (index, inserted) = match existing {
            Some(index) => (index, false),
            None => {
                self.transcript.push(TranscriptEntry::Tool {
                    key: ToolReceiptKey {
                        session_id,
                        agent_instance_id: Some(receipt.child_instance_id.clone()),
                        tool_call_id: String::new(),
                    },
                    name: "agent".into(),
                    excerpt: "agent".into(),
                    summary: String::new(),
                    complete: false,
                    error: false,
                    canceled: false,
                    card: Card {
                        child_instance_id: Some(receipt.child_instance_id.clone()),
                        turn: self.current_turn,
                        ..Default::default()
                    },
                });
                (self.transcript.len() - 1, true)
            }
        };
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
        if inserted {
            TranscriptEdit::Insert(index)
        } else {
            TranscriptEdit::Remeasure(index)
        }
    }

    fn commit_assistant(&mut self, message: Message) -> Vec<TranscriptEdit> {
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
        let mut edits: Vec<TranscriptEdit> = Vec::new();
        // Thinking body text is never retained — the provider protocol mixes
        // raw reasoning with any summary, so we only guarantee that a
        // header-only marker exists when the turn thought at all. A compound
        // reconcile (Thinking marker append + trailing assistant row drop)
        // MUST emit both edits — the view applies them in order so the
        // virtual-list count and cached heights end aligned to the
        // transcript. A one-action signal would splice only one operation.
        if has_thinking && !has_thinking_marker {
            self.transcript.push(TranscriptEntry::Thinking);
            if let Some(index) = self.transcript.len().checked_sub(1) {
                edits.push(TranscriptEdit::Insert(index));
            }
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
                if let Some(index) = self.transcript.len().checked_sub(1) {
                    edits.push(TranscriptEdit::Insert(index));
                }
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
                    // rhythm. Drop the trailing row instead of blanking it.
                    if tail.is_empty() {
                        self.transcript.remove(last_idx);
                        edits.push(TranscriptEdit::Remove(last_idx));
                    } else {
                        if let TranscriptEntry::Assistant(current) = &mut self.transcript[last_idx]
                        {
                            *current = tail.into();
                        }
                        edits.push(TranscriptEdit::Remeasure(last_idx));
                    }
                } else {
                    // Divergent prefix — remove leading rows in DESCENDING
                    // index order so an earlier `Remove(hi)` never invalidates
                    // a later `Remove(lo)`. The view applies each edit in
                    // sequence: after Remove(hi), the row previously at `lo`
                    // is still at `lo`, so the next Remove is well-formed.
                    for &idx in leading.iter().rev() {
                        self.transcript.remove(idx);
                        edits.push(TranscriptEdit::Remove(idx));
                    }
                    let last_idx = self.transcript[turn_start..]
                        .iter()
                        .rposition(|entry| matches!(entry, TranscriptEntry::Assistant(_)))
                        .map(|offset| turn_start + offset)
                        .expect("collapsed assistant row still present");
                    if let TranscriptEntry::Assistant(current) = &mut self.transcript[last_idx] {
                        *current = assistant_text.into();
                    }
                    edits.push(TranscriptEdit::Remeasure(last_idx));
                }
            }
        }
        edits
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

/// Character cap on the excerpt shown on a tool receipt. Anything longer
/// truncates with a single-character ellipsis so a wide command still fits
/// on one row at the transcript's reading measure.
pub const EXCERPT_CHARS: usize = 160;

fn bounded_summary(text: &str) -> String {
    text.chars()
        .map(|ch| if ch.is_control() { ' ' } else { ch })
        .take(SUMMARY_CHARS)
        .collect()
}

/// One-line excerpt of what a tool call actually ran. Contract line "each
/// tool receipt shows what actually ran": bash/exec use the command's first
/// line; read/write/edit use the file path; fetch uses the URL; other tools
/// fall through to the first primitive argument, else the tool name alone.
/// The excerpt is stripped of control chars, run through `redact_secrets`
/// so no secret material ever lands in the persisted transcript, and
/// truncated at `EXCERPT_CHARS` with a horizontal-ellipsis marker. Never
/// empty. Redaction happens BEFORE truncation so a secret that would sit
/// beyond the cap is still masked in the retained prefix rather than
/// preserved in whatever ends up displayed.
pub fn tool_excerpt(name: &str, arguments: &serde_json::Map<String, serde_json::Value>) -> String {
    let key = match name.to_ascii_lowercase().as_str() {
        "bash" | "exec" | "shell" => Some("command"),
        "read" | "write" | "edit" | "list" => Some("path"),
        "fetch" | "webfetch" => Some("url"),
        _ => None,
    };
    let raw = key
        .and_then(|k| arguments.get(k))
        .and_then(argument_text)
        .or_else(|| arguments.values().find_map(argument_text));
    let first_line = raw
        .as_deref()
        .and_then(|text| text.lines().find(|line| !line.trim().is_empty()))
        .unwrap_or("");
    if first_line.is_empty() {
        return name.to_owned();
    }
    let cleaned: String = first_line
        .chars()
        .map(|ch| if ch.is_control() { ' ' } else { ch })
        .collect();
    let redacted = redact_secrets(&cleaned);
    truncate_excerpt(&redacted)
}

/// Placeholder that replaces a redacted secret in a stored excerpt. Not a
/// chrome literal — the redaction runs BEFORE the row_text model composes
/// its visible strings, so this string travels inside `excerpt` and rides
/// the same fence path the raw excerpt does. Kept short so an assignment
/// like `KEY=[redacted]` remains legible.
pub const REDACTED_MARKER: &str = "[redacted]";

/// Word-boundary secret-name tokens matched against the segments of an
/// assignment key or URL query-param key. Keys are split on `_`, `-`, `.`,
/// and camelCase transitions BEFORE matching; a segment must equal one of
/// these tokens in full for the value to be redacted. Matching by segment
/// keeps `monkey=banana` and `design=modern` intact (`monkey` and `design`
/// are not in the set) while `api_key`, `AUTH_TOKEN`, `x-sig`, and
/// `AWS_SECRET_ACCESS_KEY` all light up because at least one segment
/// (`key`, `auth`/`token`, `sig`, `secret`/`key`) equals a listed token.
const SECRET_KEY_SEGMENTS: &[&str] = &[
    "password",
    "passwd",
    "secret",
    "secrets",
    "token",
    "tokens",
    "credential",
    "credentials",
    "auth",
    "bearer",
    "apikey",
    "key",
    "keys",
    "sig",
    "signature",
    "session",
];

/// Split a key on non-alphanumeric separators AND camelCase transitions,
/// lowercase every segment, and return the pieces. `ANTHROPIC_API_KEY` →
/// `["anthropic", "api", "key"]`; `x-github-token` → `["x", "github",
/// "token"]`; `apiKey` → `["api", "key"]`; `monkey` → `["monkey"]`.
fn split_key_segments(name: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let mut current = String::new();
    let mut prev_was_lower_or_digit = false;
    for ch in name.chars() {
        if ch.is_alphanumeric() {
            if ch.is_ascii_uppercase() && prev_was_lower_or_digit && !current.is_empty() {
                segments.push(std::mem::take(&mut current));
            }
            for lower in ch.to_lowercase() {
                current.push(lower);
            }
            prev_was_lower_or_digit = ch.is_ascii_lowercase() || ch.is_ascii_digit();
        } else {
            if !current.is_empty() {
                segments.push(std::mem::take(&mut current));
            }
            prev_was_lower_or_digit = false;
        }
    }
    if !current.is_empty() {
        segments.push(current);
    }
    segments
}

/// Does an assignment key or query-param key smell like a secret? A segment
/// of the split key must EQUAL one of `SECRET_KEY_SEGMENTS`. Empty keys are
/// never secrets (an equals sign with no left side is not an assignment).
fn key_is_secret(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    split_key_segments(name)
        .iter()
        .any(|segment| SECRET_KEY_SEGMENTS.contains(&segment.as_str()))
}

/// Redact secret material from a one-line command / URL / path excerpt
/// BEFORE it lands in `TranscriptEntry::Tool::excerpt`. Three shapes are
/// covered:
///
///   * URL userinfo: `scheme://user:pass@host/...` -> `scheme://[redacted]@host/...`
///   * URL query params whose key smells like a secret:
///     `?token=abc&filter=x` -> `?token=[redacted]&filter=x`
///   * Env-token assignments in any shell word whose key smells like a
///     secret: `ANTHROPIC_API_KEY=sk-ant-xxx` -> `ANTHROPIC_API_KEY=[redacted]`
///
/// Tokenization is shell-aware — single and double quotes bind everything
/// through the matching close quote into one word, so
/// `PASSWORD='correct horse battery'` becomes ONE token whose value (quotes
/// included) redacts in full. Naive whitespace splitting would leak the
/// tail of a quoted secret; that was the r2 blocker. The excerpt is a
/// display preview, not a runnable command, so this pass collapses runs of
/// whitespace to a single space — acceptable because `EXCERPT_CHARS`
/// truncates the excerpt and control chars are scrubbed by the caller
/// before redaction runs.
pub(crate) fn redact_secrets(input: &str) -> String {
    let mut pieces: Vec<String> = Vec::new();
    for token in shell_split(input) {
        pieces.push(redact_token(&token));
    }
    pieces.join(" ")
}

/// Split on shell-word boundaries: whitespace separates words, but single
/// or double quotes bind everything through the matching close quote (or
/// end-of-string if the quote is never closed) into one word. Quote
/// characters are RETAINED in the output word — the excerpt is a preview,
/// not a runnable command, and keeping the quotes makes it obvious that a
/// value spanned whitespace before redaction ran.
fn shell_split(input: &str) -> Vec<String> {
    let mut words: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    for ch in input.chars() {
        match (quote, ch) {
            (None, c) if c.is_whitespace() => {
                if !current.is_empty() {
                    words.push(std::mem::take(&mut current));
                }
            }
            (None, '\'' | '"') => {
                quote = Some(ch);
                current.push(ch);
            }
            (Some(q), c) if c == q => {
                quote = None;
                current.push(ch);
            }
            _ => current.push(ch),
        }
    }
    if !current.is_empty() {
        words.push(current);
    }
    words
}

fn redact_token(token: &str) -> String {
    if token.contains("://") {
        return redact_url_token(token);
    }
    if let Some(eq_ix) = token.find('=') {
        let key = &token[..eq_ix];
        if key_is_secret(key) {
            let mut out = String::with_capacity(key.len() + REDACTED_MARKER.len() + 1);
            out.push_str(key);
            out.push('=');
            out.push_str(REDACTED_MARKER);
            return out;
        }
    }
    token.to_owned()
}

fn redact_url_token(token: &str) -> String {
    let scheme_end = match token.find("://") {
        Some(ix) => ix,
        None => return token.to_owned(),
    };
    let auth_start = scheme_end + 3;
    let rest = &token[auth_start..];
    let auth_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let authority = &rest[..auth_end];
    let after_auth = &rest[auth_end..];
    let mut out = String::with_capacity(token.len() + REDACTED_MARKER.len());
    out.push_str(&token[..auth_start]);
    if let Some(at_ix) = authority.rfind('@') {
        out.push_str(REDACTED_MARKER);
        out.push('@');
        out.push_str(&authority[at_ix + 1..]);
    } else {
        out.push_str(authority);
    }
    // Query and fragment: redact each `?key=value` / `&key=value` pair whose
    // key matches the secret-shaped set. Fragment (`#...`) is preserved
    // as-is; secrets in a fragment are unusual and the fragment layout
    // varies too much across services for a per-pair rewrite to be safe.
    if let Some(q_rel) = after_auth.find('?') {
        out.push_str(&after_auth[..=q_rel]);
        let tail = &after_auth[q_rel + 1..];
        let (query, fragment) = match tail.find('#') {
            Some(hash) => (&tail[..hash], &tail[hash..]),
            None => (tail, ""),
        };
        let mut first_pair = true;
        for pair in query.split('&') {
            if !first_pair {
                out.push('&');
            }
            first_pair = false;
            if let Some(eq_ix) = pair.find('=') {
                let key = &pair[..eq_ix];
                if key_is_secret(key) {
                    out.push_str(key);
                    out.push('=');
                    out.push_str(REDACTED_MARKER);
                    continue;
                }
            }
            out.push_str(pair);
        }
        out.push_str(fragment);
    } else {
        out.push_str(after_auth);
    }
    out
}

fn argument_text(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(s) => (!s.is_empty()).then(|| s.clone()),
        serde_json::Value::Number(n) => Some(n.to_string()),
        serde_json::Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

fn truncate_excerpt(text: &str) -> String {
    for (char_count, (index, _)) in text.char_indices().enumerate() {
        if char_count == EXCERPT_CHARS {
            let mut out = text[..index].trim_end().to_owned();
            out.push('…');
            return out;
        }
    }
    text.to_owned()
}

fn tool_entry(tool_call: &ToolCall, key: ToolReceiptKey, turn: u64) -> TranscriptEntry {
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
            turn,
            ..Default::default()
        },
        key,
        name: tool_call.name.clone(),
        excerpt: tool_excerpt(&tool_call.name, &tool_call.arguments),
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
            assert_eq!(changed, vec![TranscriptEdit::Insert(0)]);
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
            matches!(tool_entry(&huge, ToolReceiptKey::new(None, &json!({}), &huge), 0), TranscriptEntry::Tool { summary, .. } if summary.chars().count() == SUMMARY_CHARS)
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
            vec![TranscriptEdit::Remove(2)],
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
    fn compound_reconcile_emits_thinking_insert_and_middle_removal_in_order() {
        // A single AssistantMessage can both (a) append a Thinking marker
        // AND (b) drop a middle assistant row when the final text is a
        // prefix of the streamed fragments bracketed by tools. `apply` MUST
        // return BOTH edits, in order — a single-action signal would splice
        // only one operation and the virtual list would desync from the
        // transcript. This test fails any mutation that collapses
        // `commit_assistant` to a one-edit return.
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
        // Transcript: [Assistant("pre"), Tool, Assistant("post"), Tool].
        // The final message brings a Thinking block AND text that matches
        // only the first fragment, so Thinking appends at index 4 and
        // Assistant("post") at index 2 is dropped.
        let edits = state.apply(ServerEvent::AssistantMessage {
            session_id: None,
            message: Message {
                role: "assistant".into(),
                content: vec![
                    ContentBlock::Thinking {
                        text: "hidden".into(),
                    },
                    ContentBlock::Text { text: "pre".into() },
                ],
            },
        });
        assert_eq!(
            edits,
            vec![TranscriptEdit::Insert(4), TranscriptEdit::Remove(2)],
            "compound reconcile MUST emit every edit in order — one-edit \
             collapses would desync the virtual list from the transcript"
        );
        assert_eq!(
            describe_transcript(&state.transcript),
            vec![
                r#"Assistant("pre")"#.to_owned(),
                r#"Tool("read", "{\"path\":\"README.md\"}")"#.to_owned(),
                r#"Tool("read", "{\"path\":\"README.md\"}")"#.to_owned(),
                "Thinking".to_owned(),
            ],
        );
    }

    #[test]
    fn divergent_prefix_emits_leading_removals_in_descending_index_order() {
        // When the streamed fragments are NOT a prefix of the final text,
        // `commit_assistant` collapses the leading assistant rows into
        // the last one. The emitted edits MUST list every removal (in
        // descending index order so an earlier `Remove(hi)` never
        // invalidates a later `Remove(lo)`) and a final `Remeasure` for
        // the surviving row. A mutation that returns only the final
        // `Remeasure` leaves the leading rows still occupying scroller
        // slots — this test fails it.
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
        // Transcript: [Assistant("alpha"), Tool, Assistant("beta"), Tool,
        //   Assistant("gamma")]. Final text "zzz" is NOT a prefix of any
        // streamed fragment — the reconciler removes the two leading
        // assistant rows and rewrites the tail assistant row.
        let edits = state.apply(ServerEvent::AssistantMessage {
            session_id: None,
            message: Message {
                role: "assistant".into(),
                content: vec![ContentBlock::Text { text: "zzz".into() }],
            },
        });
        assert_eq!(
            edits,
            vec![
                TranscriptEdit::Remove(2),
                TranscriptEdit::Remove(0),
                TranscriptEdit::Remeasure(2),
            ],
            "divergent-prefix collapse MUST report every leading-row \
             removal (in descending order) plus the surviving-row remeasure"
        );
        assert_eq!(
            describe_transcript(&state.transcript),
            vec![
                r#"Tool("read", "{\"path\":\"README.md\"}")"#.to_owned(),
                r#"Tool("read", "{\"path\":\"README.md\"}")"#.to_owned(),
                r#"Assistant("zzz")"#.to_owned(),
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

    // ---------- excerpt redaction (ZETA-125 privacy) ------------------------

    #[test]
    fn excerpt_redacts_env_token_assignments_by_key_shape() {
        // Every secret-shaped assignment MUST land in the persisted excerpt
        // with the value replaced by REDACTED_MARKER. Values differ across
        // real sessions but the key names do not; matching on the key
        // catches the whole class. A friendly, non-secret assignment like
        // NODE_ENV=production stays intact so the receipt still names what
        // ran.
        for command in [
            "export ANTHROPIC_API_KEY=sk-ant-abc123 && node build.js",
            "GITHUB_TOKEN=ghp_xyz curl example.com",
            "AWS_SECRET_ACCESS_KEY=wJalrXUt aws s3 ls",
            "PASSWORD=hunter2 psql",
            "X_AUTH=bearer-xyz curl",
            "STRIPE_SIGNATURE=v1,t=1 verify.sh",
        ] {
            let excerpt = tool_excerpt(
                "bash",
                &[("command".to_owned(), json!(command))]
                    .into_iter()
                    .collect(),
            );
            assert!(
                excerpt.contains(REDACTED_MARKER),
                "excerpt for {command:?} did not redact: {excerpt:?}"
            );
            for secret in [
                "sk-ant-abc123",
                "ghp_xyz",
                "wJalrXUt",
                "hunter2",
                "bearer-xyz",
                "v1,t=1",
            ] {
                assert!(
                    !excerpt.contains(secret),
                    "excerpt for {command:?} leaked {secret:?}: {excerpt:?}"
                );
            }
        }
    }

    #[test]
    fn excerpt_preserves_non_secret_assignments() {
        // A false-positive redaction would blank a benign env var and make
        // the receipt harder to read. NODE_ENV / DEBUG / VERBOSE all stay
        // intact — none of them match the secret-shaped patterns.
        let excerpt = tool_excerpt(
            "bash",
            &[(
                "command".to_owned(),
                json!("NODE_ENV=production DEBUG=1 node app.js"),
            )]
            .into_iter()
            .collect(),
        );
        assert!(
            !excerpt.contains(REDACTED_MARKER),
            "false positive: {excerpt:?}"
        );
        assert!(excerpt.contains("NODE_ENV=production"));
        assert!(excerpt.contains("DEBUG=1"));
    }

    #[test]
    fn excerpt_redacts_url_userinfo() {
        // Credentials in the userinfo portion (`user:pass@host`) MUST never
        // land in the persisted excerpt. The rest of the URL — scheme,
        // host, path, non-secret query — remains legible so the receipt
        // still identifies what got hit.
        let excerpt = tool_excerpt(
            "fetch",
            &[(
                "url".to_owned(),
                json!("https://alice:hunter2@api.example.com/v1/data"),
            )]
            .into_iter()
            .collect(),
        );
        assert!(
            !excerpt.contains("alice") && !excerpt.contains("hunter2"),
            "userinfo leaked: {excerpt:?}"
        );
        assert!(excerpt.contains("api.example.com/v1/data"));
        assert!(excerpt.contains("[redacted]@"));
    }

    #[test]
    fn excerpt_redacts_secret_query_params() {
        // Token-bearing query params (token=, api_key=, secret=, sig=, ...)
        // land redacted while benign params stay intact. Values differ per
        // session; keys do not.
        let url = "https://api.example.com/read?filter=all&token=secret123&limit=10&api_key=xyz";
        let excerpt = tool_excerpt(
            "fetch",
            &[("url".to_owned(), json!(url))].into_iter().collect(),
        );
        assert!(!excerpt.contains("secret123"), "token leaked: {excerpt:?}");
        assert!(!excerpt.contains("=xyz"), "api_key leaked: {excerpt:?}");
        assert!(excerpt.contains("filter=all"));
        assert!(excerpt.contains("limit=10"));
        assert!(excerpt.contains("token=[redacted]"));
        assert!(excerpt.contains("api_key=[redacted]"));
    }

    #[test]
    fn excerpt_redacts_secrets_from_a_curl_bash_line() {
        // Composite scenario from the r2 review: an inline curl with the
        // credentials in userinfo AND a token query param AND a leading
        // env-var assignment. All three redactions must fire together.
        let cmd = "GITHUB_TOKEN=ghp_secret curl https://alice:hunter2@api.example.com/data?token=abc&filter=x";
        let excerpt = tool_excerpt(
            "bash",
            &[("command".to_owned(), json!(cmd))].into_iter().collect(),
        );
        for leak in ["ghp_secret", "alice", "hunter2", "token=abc"] {
            assert!(
                !excerpt.contains(leak),
                "excerpt leaked {leak:?}: {excerpt:?}"
            );
        }
    }

    #[test]
    fn excerpt_redacts_quoted_multi_word_secret_values() {
        // r3 BLOCKER: a naive whitespace split would slice a quoted secret
        // in half and leak the tail. Single-quote, double-quote, and
        // unquoted trailing-space cases must all land fully redacted —
        // no fragment of the value survives in the excerpt.
        for (cmd, leaks) in [
            (
                "PASSWORD='correct horse battery' psql",
                &["correct", "horse", "battery"][..],
            ),
            (
                "GITHUB_TOKEN=\"ghp secret staple\" gh api",
                &["ghp", "secret", "staple"][..],
            ),
            (
                "AUTH='alpha bravo' PASSWORD=\"charlie delta\" node app.js",
                &["alpha", "bravo", "charlie", "delta"][..],
            ),
        ] {
            let excerpt = tool_excerpt(
                "bash",
                &[("command".to_owned(), json!(cmd))].into_iter().collect(),
            );
            assert!(
                excerpt.contains(REDACTED_MARKER),
                "excerpt for {cmd:?} did not redact: {excerpt:?}"
            );
            for leak in leaks {
                assert!(
                    !excerpt.contains(leak),
                    "excerpt for {cmd:?} leaked {leak:?}: {excerpt:?}"
                );
            }
        }
    }

    #[test]
    fn excerpt_secret_key_match_is_word_boundary_not_substring() {
        // r3 finding 2: the substring predicate matched `monkey=banana`
        // (contains "key") and `design=modern` (contains "sig"). Both are
        // benign env-var-shaped tokens and must persist unredacted so the
        // receipt still reads as what actually ran. Word-boundary matching
        // splits the key on `_`/`-`/`.` and camelCase transitions before
        // matching against the secret-name set.
        for (cmd, value) in [
            ("monkey=banana ls", "banana"),
            ("design=modern build.sh", "modern"),
            ("keychain_hint=off setup", "off"),
        ] {
            let excerpt = tool_excerpt(
                "bash",
                &[("command".to_owned(), json!(cmd))].into_iter().collect(),
            );
            assert!(
                !excerpt.contains(REDACTED_MARKER),
                "false-positive redaction for {cmd:?}: {excerpt:?}"
            );
            assert!(
                excerpt.contains(value),
                "excerpt for {cmd:?} lost the benign value: {excerpt:?}"
            );
        }
        // The genuine secret-key shapes STILL redact — the tightened match
        // must not become so strict it lets a real secret through.
        for cmd in [
            "api_key=abc123 curl",
            "AUTH_TOKEN=xyz gh api",
            "x-sig=zzz verify",
            "apiKey=camel node app.js",
        ] {
            let excerpt = tool_excerpt(
                "bash",
                &[("command".to_owned(), json!(cmd))].into_iter().collect(),
            );
            assert!(
                excerpt.contains(REDACTED_MARKER),
                "excerpt for {cmd:?} failed to redact: {excerpt:?}"
            );
        }
    }

    // ---------- turn-scoped tool groups (ZETA-125 r2 review) ---------------

    fn ordered_call(id: &str) -> ToolCall {
        ToolCall {
            id: id.to_owned(),
            name: "bash".to_owned(),
            arguments: [("command".to_owned(), json!(format!("echo {id}")))]
                .into_iter()
                .collect(),
        }
    }

    #[test]
    fn tool_groups_never_span_a_turn_boundary() {
        // r2 reviewer's regression probe: two tools in turn A followed by
        // ONE tool in turn B must paint as two separate units — never a
        // three-member group. `tool_group_position` refuses to cross a
        // turn boundary even when the receipts sit adjacent in the
        // transcript.
        let mut state = AppState::default();
        state.apply(ServerEvent::TurnStart {
            session_id: None,
            data: json!({}),
        });
        for id in ["a", "b"] {
            state.apply(ServerEvent::ToolStart {
                session_id: None,
                tool_call: ordered_call(id),
                data: json!({}),
            });
        }
        state.apply(ServerEvent::TurnEnd {
            session_id: None,
            data: json!({}),
        });
        state.apply(ServerEvent::TurnStart {
            session_id: None,
            data: json!({}),
        });
        state.apply(ServerEvent::ToolStart {
            session_id: None,
            tool_call: ordered_call("c"),
            data: json!({}),
        });
        // 3 receipts total, but split 2 + 1 across two turns. Neither run
        // clears the TOOL_GROUP_MIN_LEN=3 floor within its own turn, so
        // NO group forms.
        for i in 0..3 {
            assert_eq!(
                state.tool_group_position(i),
                None,
                "receipt {i} joined a cross-turn group"
            );
        }
        // Confirm the turns on the receipts differ — the guarantee the
        // grouping guard rides on.
        let turns: Vec<u64> = state
            .transcript
            .iter()
            .filter_map(|entry| match entry {
                TranscriptEntry::Tool { card, .. } => Some(card.turn),
                _ => None,
            })
            .collect();
        assert_eq!(turns[0], turns[1], "turn A receipts share a turn id");
        assert_ne!(turns[1], turns[2], "turn B receipt has a fresh turn id");
    }

    #[test]
    fn streaming_expands_every_group_in_the_active_turn() {
        // r2 reviewer's scenario: mid-turn assistant text splits one turn
        // into two runs of tool receipts. While the turn streams, BOTH
        // groups must stay expanded — not just the trailing one — so live
        // tool activity in either burst is visible without a click.
        let mut state = AppState::default();
        state.apply(ServerEvent::TurnStart {
            session_id: None,
            data: json!({}),
        });
        for id in ["a", "b", "c"] {
            state.apply(ServerEvent::ToolStart {
                session_id: None,
                tool_call: ordered_call(id),
                data: json!({}),
            });
        }
        // A mid-turn assistant message splits the run of receipts...
        state.apply(ServerEvent::AssistantMessage {
            session_id: None,
            message: Message {
                role: "assistant".into(),
                content: vec![ContentBlock::Text {
                    text: "checking...".into(),
                }],
            },
        });
        for id in ["d", "e", "f"] {
            state.apply(ServerEvent::ToolStart {
                session_id: None,
                tool_call: ordered_call(id),
                data: json!({}),
            });
        }
        // Two 3-member groups now exist within the same turn.
        let first = state.tool_group_position(0).expect("first group");
        let second = state
            .tool_group_position(state.transcript.len() - 1)
            .expect("second group");
        assert_ne!(first.first_index, second.first_index);
        assert_eq!(first.turn, second.turn, "both groups share the turn");
        // With streaming ON, both should read as expanded.
        assert!(state.streaming);
        assert!(state.is_tool_group_expanded(&first));
        assert!(state.is_tool_group_expanded(&second));
        // Turn ends -> streaming stops -> both fall back to the map (empty)
        // and read as collapsed.
        state.apply(ServerEvent::AgentEnd {
            session_id: None,
            data: json!({}),
        });
        assert!(!state.streaming);
        assert!(!state.is_tool_group_expanded(&first));
        assert!(!state.is_tool_group_expanded(&second));
    }

    #[test]
    fn group_shape_changes_emit_remeasure_edits() {
        // r3 finding 4: streaming-state group transitions used to skip
        // the ordered edit list, leaving rows painted at their pre-group
        // (or pre-collapse) heights until the next unrelated event
        // forced a remeasure. Every SHAPE change must ride
        // TranscriptEdit — this test drives the two hot cases: a
        // 3rd-tool arrival that first FORMS a group, and a streaming
        // termination that flips force-expanded groups back to their
        // default.
        let mut state = AppState::default();
        state.apply(ServerEvent::TurnStart {
            session_id: None,
            data: json!({}),
        });
        // First two tools: no group yet, no shape change on earlier rows.
        for id in ["a", "b"] {
            let edits = state.apply(ServerEvent::ToolStart {
                session_id: None,
                tool_call: ordered_call(id),
                data: json!({}),
            });
            assert!(
                edits
                    .iter()
                    .all(|edit| matches!(edit, TranscriptEdit::Insert(_))),
                "pre-threshold ToolStart emits only Insert, got {edits:?}"
            );
        }
        // Third tool: this is the arrival that FORMS the group. Rows 0
        // and 1 change render shape (row 0 gains a header wrapper, row
        // 1 becomes interior) and must remeasure. Row 2's own Insert
        // covers its remeasure.
        let edits = state.apply(ServerEvent::ToolStart {
            session_id: None,
            tool_call: ordered_call("c"),
            data: json!({}),
        });
        assert!(
            edits.contains(&TranscriptEdit::Insert(2)),
            "third tool inserts at index 2: {edits:?}"
        );
        assert!(
            edits.contains(&TranscriptEdit::Remeasure(0)),
            "group formation must remeasure row 0 (gains header): {edits:?}"
        );
        assert!(
            edits.contains(&TranscriptEdit::Remeasure(1)),
            "group formation must remeasure row 1 (becomes interior): {edits:?}"
        );
        // Streaming ends: every current-turn group flips out of
        // force-expanded, so every row in the group needs remeasure.
        let end_edits = state.apply(ServerEvent::AgentEnd {
            session_id: None,
            data: json!({}),
        });
        for i in 0..=2 {
            assert!(
                end_edits.contains(&TranscriptEdit::Remeasure(i)),
                "streaming end must remeasure row {i} (group collapses): {end_edits:?}"
            );
        }
    }

    #[test]
    fn resumed_history_groups_stay_collapsed_even_while_streaming() {
        // r3 finding 6: `apply_history` used to stamp restored tool
        // receipts with the LIVE `current_turn`. If the client resumed
        // mid-turn (streaming=true), every restored group matched
        // `current_turn` and painted force-expanded as if it were the
        // active turn. The fix stamps each replayed assistant message
        // with its own historical turn AND bumps `current_turn` past
        // every replayed turn at end-of-replay, so a subsequent
        // streaming pass forces expansion ONLY on genuinely new turns.
        use crate::client::{HistoryContent, HistoryMessage};
        let history = vec![
            HistoryMessage {
                id: "u1".into(),
                role: "user".into(),
                content: vec![HistoryContent::Text {
                    text: "look".into(),
                }],
                tool_result: None,
            },
            HistoryMessage {
                id: "a1".into(),
                role: "assistant".into(),
                content: vec![
                    HistoryContent::ToolUse {
                        tool_call: ordered_call("h1"),
                    },
                    HistoryContent::ToolUse {
                        tool_call: ordered_call("h2"),
                    },
                    HistoryContent::ToolUse {
                        tool_call: ordered_call("h3"),
                    },
                ],
                tool_result: None,
            },
        ];
        let mut state = AppState::default();
        // Simulate the mid-turn resume the finding calls out.
        state.streaming = true;
        state.apply_history(history, true);
        assert!(state.streaming, "resume keeps streaming=true");
        let group = state
            .tool_group_position(state.transcript.len() - 1)
            .expect("three tools form a historical group");
        assert!(
            !state.is_tool_group_expanded(&group),
            "historical group must not paint force-expanded under a mid-turn resume — \
             streaming expansion is reserved for the LIVE turn",
        );
        // A genuinely new streamed turn still forces its own group open.
        state.apply(ServerEvent::TurnStart {
            session_id: None,
            data: json!({}),
        });
        for id in ["n1", "n2", "n3"] {
            state.apply(ServerEvent::ToolStart {
                session_id: None,
                tool_call: ordered_call(id),
                data: json!({}),
            });
        }
        let new_group = state
            .tool_group_position(state.transcript.len() - 1)
            .expect("new streamed group");
        assert!(
            state.is_tool_group_expanded(&new_group),
            "streaming still expands the LIVE turn's group",
        );
        assert!(
            !state.is_tool_group_expanded(&group),
            "the historical group stays collapsed even after new streaming",
        );
    }

    #[test]
    fn streamed_bash_result_bytes_are_counted_exactly_once() {
        // r3 finding 5: bash streams stdout via `ToolOutput` and repeats
        // the whole thing in the final `ToolEnd` payload. Before the fix
        // the tail's `bytes_seen` counted BOTH sources — the size label
        // on the receipt overstated the stdout by 2x. Counting once means
        // the streamed bytes are authoritative and the final payload
        // does not re-increment `bytes_seen`.
        let mut state = AppState::default();
        state.apply(ServerEvent::TurnStart {
            session_id: None,
            data: json!({}),
        });
        let call = ordered_call("a");
        state.apply(ServerEvent::ToolStart {
            session_id: None,
            tool_call: call.clone(),
            data: json!({}),
        });
        let stdout: String = "hello world\n".repeat(50);
        state.apply(ServerEvent::ToolOutput {
            session_id: None,
            tool_call: call.clone(),
            output: stdout.clone(),
            data: json!({}),
        });
        state.apply(ServerEvent::ToolEnd {
            session_id: None,
            tool_call: call.clone(),
            tool_result: Some(crate::client::ToolResult {
                tool_call_id: call.id.clone(),
                content: stdout.clone(),
                is_error: false,
                is_canceled: false,
                content_blocks: vec![],
                structured_content: None,
            }),
            data: json!({}),
        });
        let bytes = match &state.transcript[0] {
            TranscriptEntry::Tool { card, .. } => card.tail.bytes_seen,
            _ => panic!("expected a tool row"),
        };
        assert_eq!(
            bytes,
            stdout.len(),
            "streamed bash stdout must count once, not twice",
        );
    }

    #[test]
    fn tool_group_total_bytes_reads_cumulative_output() {
        // The group summary total tracks `bytes_seen` on each receipt's
        // tail, not the on-screen retained bytes. A tail that truncated
        // some of its history STILL contributes every byte that ever
        // flowed through it, so the total matches "total output produced"
        // — what an operator counting bytes would expect — rather than
        // "bytes currently retained on screen".
        let mut state = AppState::default();
        state.apply(ServerEvent::TurnStart {
            session_id: None,
            data: json!({}),
        });
        for id in ["a", "b", "c"] {
            state.apply(ServerEvent::ToolStart {
                session_id: None,
                tool_call: ordered_call(id),
                data: json!({}),
            });
            state.apply(ServerEvent::ToolOutput {
                session_id: None,
                tool_call: ordered_call(id),
                output: "x".repeat(100),
                data: json!({}),
            });
        }
        let group = state.tool_group_position(0).expect("group");
        assert_eq!(state.tool_group_output_bytes(&group), 300);
    }
}
