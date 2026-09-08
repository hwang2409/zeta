use crate::client::{Approval, Message, ServerEvent, SessionMetadata, StatusResult, ToolCall};

#[derive(Debug, Clone, PartialEq)]
pub enum TranscriptEntry {
    User(String),
    Assistant(String),
    Tool {
        id: String,
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

    pub fn apply_status(&mut self, status: StatusResult) {
        self.active_session = status.session.map(|session| session.session_id);
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
            ServerEvent::ToolStart { tool_call, .. } => {
                self.transcript.push(tool_entry(&tool_call))
            }
            ServerEvent::ToolOutput {
                tool_call, output, ..
            } => {
                if let TranscriptEntry::Tool { summary, .. } = self.tool_receipt(&tool_call) {
                    *summary = bounded_summary(&format!("{summary}{output}"));
                }
            }
            ServerEvent::ToolEnd {
                tool_call,
                tool_result,
                ..
            } => {
                if let TranscriptEntry::Tool {
                    complete, error, ..
                } = self.tool_receipt(&tool_call)
                {
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
            ServerEvent::ApprovalEnd { tool_call, .. } => self
                .approvals
                .retain(|item| item.tool_call.id != tool_call.id),
            ServerEvent::Error { error, .. } => {
                self.streaming = false;
                self.approvals.clear();
                self.transcript.push(TranscriptEntry::Tool {
                    id: String::new(),
                    name: "error".to_owned(),
                    summary: bounded_summary(&error.message),
                    complete: true,
                    error: true,
                });
            }
            ServerEvent::Other { .. } => {}
        }
    }

    fn tool_receipt(&mut self, tool_call: &ToolCall) -> &mut TranscriptEntry {
        let index = self
            .transcript
            .iter()
            .position(
                |entry| matches!(entry, TranscriptEntry::Tool { id, .. } if id == &tool_call.id),
            )
            .unwrap_or_else(|| {
                self.transcript.push(tool_entry(tool_call));
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

fn tool_entry(tool_call: &ToolCall) -> TranscriptEntry {
    TranscriptEntry::Tool {
        id: tool_call.id.clone(),
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
    fn approval_and_deny_clear_the_modal() {
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
        assert!(state.approvals.is_empty());
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
            matches!(&state.transcript[0], TranscriptEntry::Tool { id, summary, complete: true, error: false, .. }
            if id == "tool-1" && summary.chars().count() == SUMMARY_CHARS && !summary.contains('\n'))
        );
        assert!(
            matches!(&state.transcript[1], TranscriptEntry::Tool { id, summary, complete: true, error: true, .. }
            if id == "tool-2" && summary.ends_with("second"))
        );
        let huge = ToolCall {
            arguments: [("arg".into(), json!("x".repeat(1000)))]
                .into_iter()
                .collect(),
            ..call()
        };
        assert!(
            matches!(tool_entry(&huge), TranscriptEntry::Tool { summary, .. } if summary.chars().count() == SUMMARY_CHARS)
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
