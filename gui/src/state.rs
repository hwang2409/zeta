use std::collections::HashMap;

use crate::client::{Approval, Message, ServerEvent, SessionMetadata, StatusResult, ToolCall};

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
    Assistant(String),
    Tool {
        key: ToolReceiptKey,
        name: String,
        summary: String,
        complete: bool,
        error: bool,
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
    }

    pub fn apply_status(&mut self, status: StatusResult) {
        self.select_session(status.session.map(|session| session.session_id));
        self.streaming = status.state != "idle";
        self.approvals = status.pending_approvals;
    }

    pub fn apply(&mut self, event: ServerEvent) {
        match event {
            ServerEvent::TurnStart { .. } => self.streaming = true,
            ServerEvent::AgentEnd { .. } | ServerEvent::TurnAborted { .. } => {
                self.streaming = false;
                self.approvals.clear();
            }
            ServerEvent::TurnEnd { .. } => {}
            ServerEvent::AssistantDelta { delta, kind, .. } if kind == "assistant" => {
                match self.transcript.last_mut() {
                    Some(TranscriptEntry::Assistant(text)) => text.push_str(&delta),
                    _ => self.transcript.push(TranscriptEntry::Assistant(delta)),
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
                if let TranscriptEntry::Tool { summary, .. } = self.tool_receipt(
                    &tool_call,
                    ToolReceiptKey::new(session_id, &data, &tool_call),
                ) {
                    *summary = bounded_summary(&format!("{summary}{output}"));
                }
            }
            ServerEvent::ToolEnd {
                tool_call,
                tool_result,
                session_id,
                data,
            } => {
                if let TranscriptEntry::Tool {
                    complete, error, ..
                } = self.tool_receipt(
                    &tool_call,
                    ToolReceiptKey::new(session_id, &data, &tool_call),
                ) {
                    *complete = true;
                    *error = tool_result.as_ref().is_some_and(|result| result.is_error);
                }
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
                self.approvals.clear();
                self.transcript.push(TranscriptEntry::Tool {
                    key: ToolReceiptKey {
                        session_id: None,
                        agent_instance_id: None,
                        tool_call_id: String::new(),
                    },
                    name: "error".to_owned(),
                    summary: bounded_summary(&error.message),
                    complete: true,
                    error: true,
                });
            }
            ServerEvent::Other { .. } => {}
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

    fn commit_assistant(&mut self, message: Message) {
        let text = message.text();
        if text.is_empty() {
            return;
        }
        match self.transcript.last_mut() {
            Some(TranscriptEntry::Assistant(current)) => *current = text,
            _ => self.transcript.push(TranscriptEntry::Assistant(text)),
        }
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
    TranscriptEntry::Tool {
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
            vec![TranscriptEntry::Assistant("hello".to_owned())]
        );
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
            matches!(&state.transcript[0], TranscriptEntry::Tool { summary, .. } if summary.contains("README.md"))
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
            if key.tool_call_id == "tool-1" && summary.chars().count() == SUMMARY_CHARS && !summary.contains('\n'))
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
