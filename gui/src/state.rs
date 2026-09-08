use crate::client::{Approval, Message, ServerEvent, SessionMetadata, ToolCall};

#[derive(Debug, Clone, PartialEq)]
pub enum TranscriptEntry {
    User(String),
    Assistant(String),
    Tool {
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
    pub composer: String,
    pub approval: Option<Approval>,
    pub connection: ConnectionState,
    pub streaming: bool,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            sessions: Vec::new(),
            active_session: None,
            transcript: Vec::new(),
            composer: String::new(),
            approval: None,
            connection: ConnectionState::Reconnecting,
            streaming: false,
        }
    }
}

impl AppState {
    pub fn mark_connection_lost(&mut self, error: impl Into<String>) {
        self.connection = ConnectionState::Lost(error.into());
    }

    pub fn begin_reconnect(&mut self) {
        self.connection = ConnectionState::Reconnecting;
    }

    pub fn apply(&mut self, event: ServerEvent) {
        match event {
            ServerEvent::TurnStart { .. } => self.streaming = true,
            ServerEvent::TurnEnd { .. } | ServerEvent::TurnAborted { .. } => self.streaming = false,
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
                if let Some(TranscriptEntry::Tool { name, summary, .. }) = self
                    .transcript
                    .iter_mut()
                    .rev()
                    .find(|entry| matches!(entry, TranscriptEntry::Tool { .. }))
                {
                    if *name == tool_call.name {
                        summary.push_str(&output.replace(['\n', '\r'], " "));
                    }
                }
            }
            ServerEvent::ToolEnd {
                tool_call,
                tool_result,
                ..
            } => {
                if let Some(TranscriptEntry::Tool {
                    complete, error, ..
                }) = self
                    .transcript
                    .iter_mut()
                    .rev()
                    .find(|entry| matches!(entry, TranscriptEntry::Tool { .. }))
                {
                    *complete = true;
                    *error = tool_result.as_ref().is_some_and(|result| result.is_error);
                } else {
                    self.transcript.push(tool_entry(&tool_call));
                }
            }
            ServerEvent::ApprovalRequest { approval, .. } => self.approval = Some(approval),
            ServerEvent::ApprovalEnd { .. } => self.approval = None,
            ServerEvent::Error { error, .. } => {
                self.streaming = false;
                self.transcript.push(TranscriptEntry::Tool {
                    name: "error".to_owned(),
                    summary: error.message,
                    complete: true,
                    error: true,
                });
            }
            ServerEvent::Other { .. } => {}
        }
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

fn tool_entry(tool_call: &ToolCall) -> TranscriptEntry {
    TranscriptEntry::Tool {
        name: tool_call.name.clone(),
        summary: serde_json::to_string(&tool_call.arguments)
            .unwrap_or_else(|_| "arguments unavailable".to_owned()),
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
        assert!(state.approval.is_some());
        state.apply(ServerEvent::ApprovalEnd {
            session_id: None,
            tool_call: tool,
            data: json!({ "decision": "deny" }),
        });
        assert!(state.approval.is_none());
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
