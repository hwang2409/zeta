use crate::{
    cards::{Card, OutputTail},
    markdown::Markdown,
};
use std::collections::HashMap;

use crate::client::{
    Approval, Message, ServerEvent, SessionMetadata, StatusResult, SubAgentReceipt, SubAgentStatus,
    ToolCall,
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
    Tool {
        key: ToolReceiptKey,
        name: String,
        summary: String,
        complete: bool,
        error: bool,
        card: Card,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConnectionState {
    Connected,
    Reconnecting,
    Lost(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct AppState {
    pub sessions: Vec<SessionMetadata>,
    pub active_session: Option<String>,
    pub sessions_truncated: bool,
    pub saved_transcripts: HashMap<String, Vec<TranscriptEntry>>,
    pub transcript: Vec<TranscriptEntry>,
    pub approvals: Vec<Approval>,
    pub connection: ConnectionState,
    pub streaming: bool,
    pub metrics: StatusMetrics,
    pub metrics_boundary: bool,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            sessions: Vec::new(),
            active_session: None,
            sessions_truncated: false,
            saved_transcripts: HashMap::new(),
            transcript: Vec::new(),
            approvals: Vec::new(),
            connection: ConnectionState::Reconnecting,
            streaming: false,
            metrics: StatusMetrics::default(),
            metrics_boundary: true,
        }
    }
}

impl AppState {
    pub fn mark_connection_lost(&mut self, error: impl Into<String>) {
        self.connection = ConnectionState::Lost(error.into());
        self.streaming = false;
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
        self.approvals = status.pending_approvals;
    }

    pub fn apply(&mut self, event: ServerEvent) {
        match event {
            ServerEvent::TurnStart { .. } => {
                self.streaming = true;
                self.metrics_boundary = false;
            }
            ServerEvent::AgentEnd { .. } | ServerEvent::TurnAborted { .. } => {
                self.streaming = false;
                self.metrics_boundary = true;
                self.approvals.clear();
            }
            ServerEvent::TurnEnd { .. } => self.metrics_boundary = true,
            ServerEvent::AssistantDelta { delta, kind, .. } if kind == "assistant" => {
                match self.transcript.last_mut() {
                    Some(TranscriptEntry::Assistant(text)) => text.push_str(&delta),
                    _ => self
                        .transcript
                        .push(TranscriptEntry::Assistant(Markdown::streaming(delta))),
                }
            }
            ServerEvent::AssistantDelta { .. } => {}
            ServerEvent::AssistantMessage { message, .. } => self.commit_assistant(message),
            ServerEvent::ToolStart {
                session_id,
                tool_call,
                data,
            } => self.transcript.push(tool_entry(
                &tool_call,
                ToolReceiptKey::new(session_id, &data, &tool_call),
            )),
            ServerEvent::ToolOutput {
                session_id,
                tool_call,
                output,
                data,
            } => {
                if let TranscriptEntry::Tool { summary, card, .. } = self.tool_receipt(
                    &tool_call,
                    ToolReceiptKey::new(session_id, &data, &tool_call),
                ) {
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
                if let TranscriptEntry::Tool {
                    complete,
                    error,
                    summary,
                    card,
                    name,
                    ..
                } = self.tool_receipt(
                    &tool_call,
                    ToolReceiptKey::new(session_id, &data, &tool_call),
                ) {
                    if name.eq_ignore_ascii_case("agent") {
                        if let Some(child) = tool_result
                            .as_ref()
                            .and_then(|result| result.structured_content.as_ref())
                            .and_then(|data| data["child_instance_id"].as_str())
                        {
                            card.child_instance_id = Some(child.to_owned());
                        }
                    }
                    *complete = !name.eq_ignore_ascii_case("agent")
                        || !tool_result
                            .as_ref()
                            .and_then(|result| result.structured_content.as_ref())
                            .is_some_and(|data| data["status"] == "running");
                    *error = tool_result.as_ref().is_some_and(|result| result.is_error);
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
                self.commit_sub_agent(session_id, receipt);
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
            ServerEvent::Error { error, .. } => {
                self.streaming = false;
                self.metrics_boundary = true;
                self.approvals.clear();
                self.transcript.push(TranscriptEntry::Tool {
                    key: ToolReceiptKey {
                        session_id: None,
                        agent_instance_id: None,
                        tool_call_id: String::new(),
                    },
                    name: "error".to_owned(),
                    summary: bounded_summary(&error.message),
                    card: Card {
                        tail: {
                            let mut tail = OutputTail::default();
                            tail.append(&error.message);
                            tail
                        },
                        ..Default::default()
                    },
                    complete: true,
                    error: true,
                });
            }
            ServerEvent::Other { .. } => {}
        }
    }

    pub fn toggle_card(&mut self, index: usize) {
        if let Some(TranscriptEntry::Tool { card, .. }) = self.transcript.get_mut(index) {
            card.toggle();
        }
    }

    fn tool_receipt(&mut self, tool_call: &ToolCall, key: ToolReceiptKey) -> &mut TranscriptEntry {
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
        &mut self.transcript[index]
    }

    fn commit_sub_agent(&mut self, session_id: Option<String>, receipt: SubAgentReceipt) {
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
            card,
            ..
        } = &mut self.transcript[index]
        {
            *complete = true;
            *error = receipt.status != SubAgentStatus::Completed;
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
    }

    fn commit_assistant(&mut self, message: Message) {
        let text = message.text();
        if text.is_empty() {
            return;
        }
        match self.transcript.last_mut() {
            Some(TranscriptEntry::Assistant(current)) => *current = text.into(),
            _ => self
                .transcript
                .push(TranscriptEntry::Assistant(text.into())),
        }
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::{EventError, ToolResult};
    use serde_json::json;

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
            state.transcript,
            vec![TranscriptEntry::Assistant("hello".into())]
        );
    }

    #[test]
    fn multi_frame_fence_stream_never_builds_a_markdown_tree() {
        let mut state = AppState::default();
        let mut source = String::new();
        for delta in std::iter::once("```rust\n").chain(std::iter::repeat_n("let x = 1;\n", 2000)) {
            source.push_str(delta);
            state.apply(ServerEvent::AssistantDelta {
                session_id: None,
                delta: delta.into(),
                kind: "assistant".into(),
            });
            assert!(
                matches!(&state.transcript[0], TranscriptEntry::Assistant(doc) if doc.root.is_none())
            );
        }
        assert!(
            matches!(&state.transcript[0], TranscriptEntry::Assistant(doc) if doc.source == source)
        );
        source.push_str("```\n");
        state.apply(ServerEvent::AssistantMessage {
            session_id: None,
            message: Message {
                role: "assistant".into(),
                content: vec![crate::client::ContentBlock::Text { text: source }],
            },
        });
        let TranscriptEntry::Assistant(doc) = &state.transcript[0] else {
            panic!("missing assistant")
        };
        let code = &doc.root.as_ref().unwrap().children[0];
        assert_eq!(code.kind, crate::markdown::BlockKind::Code("rust".into()));
        assert!(code.syntax.is_empty());
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
    fn errors_stop_streaming_and_render_as_distinct_receipt() {
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
        assert!(matches!(
            state.transcript[0],
            TranscriptEntry::Tool { error: true, .. }
        ));
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
