//! A gpui-independent client for the zeta serve 1.0 protocol.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::VecDeque;
use std::io::{self, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::Path;
use std::time::Duration;
use thiserror::Error;

pub const PROTOCOL_VERSION: &str = "1.0";
pub const MAX_FRAME_BYTES: usize = 1024 * 1024;
const IO_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Error)]
pub enum ClientError {
    #[error("connection failed: {0}")]
    Io(#[from] io::Error),
    #[error("invalid protocol frame: {0}")]
    Json(#[from] serde_json::Error),
    #[error("server error {code}: {message}")]
    Rpc {
        code: i64,
        message: String,
        data: Option<Value>,
    },
    #[error("protocol version mismatch: requested {requested}, server supports {supported}")]
    VersionMismatch {
        requested: String,
        supported: String,
    },
    #[error("server returned an unexpected response")]
    UnexpectedResponse,
    #[error("protocol frame exceeds {MAX_FRAME_BYTES} bytes")]
    FrameTooLarge,
    #[error("unix sockets are not supported on this platform")]
    UnixSocketUnavailable,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionMetadata {
    #[serde(default)]
    pub version: u64,
    pub session_id: String,
    #[serde(default)]
    pub created_at: String,
    #[serde(default)]
    pub updated_at: String,
    #[serde(default)]
    pub provider: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub cwd: String,
    #[serde(default)]
    pub retained_tail: u64,
    #[serde(default)]
    pub compaction_budget: u64,
    #[serde(default)]
    pub override_audit: Vec<Value>,
    #[serde(default)]
    pub system_prompt: String,
    #[serde(default)]
    pub context_files: Vec<String>,
    #[serde(default)]
    pub vim_mode: bool,
    #[serde(default)]
    pub budget_pinned: bool,
    #[serde(default)]
    pub plan_mode: bool,
    #[serde(default)]
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub arguments: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Message {
    pub role: String,
    #[serde(default)]
    pub content: Vec<ContentBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "thinking")]
    Thinking { text: String },
    #[serde(rename = "tool_use")]
    ToolUse { tool_call: ToolCall },
    #[serde(rename = "image")]
    Image {
        data: String,
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
}

impl Message {
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub content: String,
    pub is_error: bool,
    #[serde(default)]
    pub is_canceled: bool,
    #[serde(default)]
    pub content_blocks: Vec<Value>,
    #[serde(default)]
    pub structured_content: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Approval {
    pub request_id: String,
    pub tool_call: ToolCall,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct EventNotification {
    pub params: EventParams,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct EventParams {
    pub event: String,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(flatten)]
    pub fields: Map<String, Value>,
}

impl EventParams {
    fn field<T: for<'de> Deserialize<'de>>(&self, name: &str) -> Result<T, ClientError> {
        self.fields
            .get(name)
            .ok_or(ClientError::UnexpectedResponse)
            .and_then(|value| serde_json::from_value(value.clone()).map_err(ClientError::Json))
    }

    pub fn into_event(self) -> Result<ServerEvent, ClientError> {
        let session_id = self.session_id.clone();
        let event = match self.event.as_str() {
            "turn_start" => ServerEvent::TurnStart {
                session_id,
                data: self.field_or_empty("data")?,
            },
            "turn_end" => ServerEvent::TurnEnd {
                session_id,
                data: self.field_or_empty("data")?,
            },
            "turn_aborted" => ServerEvent::TurnAborted {
                session_id,
                data: self.field_or_empty("data")?,
            },
            "assistant_delta" => ServerEvent::AssistantDelta {
                session_id,
                delta: self.field("delta")?,
                kind: self.field("kind")?,
            },
            "assistant_message" => ServerEvent::AssistantMessage {
                session_id,
                message: self.field("message")?,
            },
            "tool_start" => ServerEvent::ToolStart {
                session_id,
                tool_call: self.field("tool_call")?,
                data: self.field_or_empty("data")?,
            },
            "tool_output" => ServerEvent::ToolOutput {
                session_id,
                tool_call: self.field("tool_call")?,
                output: self.field("output")?,
                data: self.field_or_empty("data")?,
            },
            "tool_end" => ServerEvent::ToolEnd {
                session_id,
                tool_call: self.field("tool_call")?,
                tool_result: self.field("tool_result")?,
                data: self.field_or_empty("data")?,
            },
            "approval_request" => ServerEvent::ApprovalRequest {
                session_id,
                approval: Approval {
                    request_id: self.field("request_id")?,
                    tool_call: self.field("tool_call")?,
                },
            },
            "approval_end" => ServerEvent::ApprovalEnd {
                session_id,
                tool_call: self.field("tool_call")?,
                data: self.field_or_empty("data")?,
            },
            "error" => ServerEvent::Error {
                session_id,
                error: self.field("error")?,
                data: self.field_or_empty("data")?,
            },
            _ => ServerEvent::Other {
                session_id,
                name: self.event,
                fields: self.fields,
            },
        };
        Ok(event)
    }

    fn field_or_empty<T: Default + for<'de> Deserialize<'de>>(
        &self,
        name: &str,
    ) -> Result<T, ClientError> {
        self.fields.get(name).map_or_else(
            || Ok(T::default()),
            |value| serde_json::from_value(value.clone()).map_err(ClientError::Json),
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ServerEvent {
    TurnStart {
        session_id: Option<String>,
        data: Value,
    },
    TurnEnd {
        session_id: Option<String>,
        data: Value,
    },
    TurnAborted {
        session_id: Option<String>,
        data: Value,
    },
    AssistantDelta {
        session_id: Option<String>,
        delta: String,
        kind: String,
    },
    AssistantMessage {
        session_id: Option<String>,
        message: Message,
    },
    ToolStart {
        session_id: Option<String>,
        tool_call: ToolCall,
        data: Value,
    },
    ToolOutput {
        session_id: Option<String>,
        tool_call: ToolCall,
        output: String,
        data: Value,
    },
    ToolEnd {
        session_id: Option<String>,
        tool_call: ToolCall,
        tool_result: Option<ToolResult>,
        data: Value,
    },
    ApprovalRequest {
        session_id: Option<String>,
        approval: Approval,
    },
    ApprovalEnd {
        session_id: Option<String>,
        tool_call: ToolCall,
        data: Value,
    },
    Error {
        session_id: Option<String>,
        error: EventError,
        data: Value,
    },
    Other {
        session_id: Option<String>,
        name: String,
        fields: Map<String, Value>,
    },
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct EventError {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct HelloResult {
    pub protocol_version: String,
    pub server: String,
    #[serde(default)]
    pub capabilities: Value,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct StatusResult {
    pub session: Option<SessionMetadata>,
    pub state: String,
    #[serde(default)]
    pub pending_approvals: Vec<Approval>,
    #[serde(default)]
    pub usage: Value,
    #[serde(default)]
    pub compaction_markers: u64,
}

#[derive(Debug)]
enum Connection {
    Tcp(TcpStream),
    #[cfg(unix)]
    Unix(std::os::unix::net::UnixStream),
}

impl Read for Connection {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        match self {
            Self::Tcp(stream) => stream.read(buffer),
            #[cfg(unix)]
            Self::Unix(stream) => stream.read(buffer),
        }
    }
}

impl Write for Connection {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        match self {
            Self::Tcp(stream) => stream.write(buffer),
            #[cfg(unix)]
            Self::Unix(stream) => stream.write(buffer),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            Self::Tcp(stream) => stream.flush(),
            #[cfg(unix)]
            Self::Unix(stream) => stream.flush(),
        }
    }
}

impl Connection {
    fn set_timeouts(&self) -> io::Result<()> {
        match self {
            Self::Tcp(stream) => {
                stream.set_read_timeout(Some(IO_TIMEOUT))?;
                stream.set_write_timeout(Some(IO_TIMEOUT))
            }
            #[cfg(unix)]
            Self::Unix(stream) => {
                stream.set_read_timeout(Some(IO_TIMEOUT))?;
                stream.set_write_timeout(Some(IO_TIMEOUT))
            }
        }
    }
}

pub struct ProtocolClient {
    connection: Connection,
    read_buffer: Vec<u8>,
    events: VecDeque<ServerEvent>,
    next_id: u64,
}

impl ProtocolClient {
    #[cfg(unix)]
    pub fn connect_socket(path: impl AsRef<Path>) -> Result<Self, ClientError> {
        let connection = Connection::Unix(std::os::unix::net::UnixStream::connect(path)?);
        Self::from_connection(connection)
    }

    #[cfg(not(unix))]
    pub fn connect_socket(_path: impl AsRef<Path>) -> Result<Self, ClientError> {
        Err(ClientError::UnixSocketUnavailable)
    }

    pub fn connect_tcp(address: impl ToSocketAddrs) -> Result<Self, ClientError> {
        let address = address
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "no socket address"))?;
        let stream = TcpStream::connect_timeout(&address, IO_TIMEOUT)?;
        Self::from_connection(Connection::Tcp(stream))
    }

    fn from_connection(connection: Connection) -> Result<Self, ClientError> {
        connection.set_timeouts()?;
        Ok(Self {
            connection,
            read_buffer: Vec::new(),
            events: VecDeque::new(),
            next_id: 1,
        })
    }

    pub fn set_read_timeout(&self, timeout: Option<Duration>) -> Result<(), ClientError> {
        match &self.connection {
            Connection::Tcp(stream) => stream.set_read_timeout(timeout),
            #[cfg(unix)]
            Connection::Unix(stream) => stream.set_read_timeout(timeout),
        }
        .map_err(ClientError::Io)
    }

    pub fn handshake(&mut self) -> Result<HelloResult, ClientError> {
        let hello: HelloResult = self.request(
            "hello",
            serde_json::json!({ "protocol_version": PROTOCOL_VERSION }),
        )?;
        if hello.protocol_version != PROTOCOL_VERSION {
            return Err(ClientError::VersionMismatch {
                requested: PROTOCOL_VERSION.to_owned(),
                supported: hello.protocol_version,
            });
        }
        Ok(hello)
    }

    pub fn list_sessions(&mut self) -> Result<Vec<SessionMetadata>, ClientError> {
        let result: Value = self.request("list_sessions", Value::Object(Map::new()))?;
        serde_json::from_value(result.get("sessions").cloned().unwrap_or_default())
            .map_err(ClientError::Json)
    }

    pub fn new_session(
        &mut self,
        provider: Option<&str>,
        model: Option<&str>,
    ) -> Result<SessionMetadata, ClientError> {
        let mut params = Map::new();
        if let Some(provider) = provider {
            params.insert("provider".to_owned(), Value::String(provider.to_owned()));
        }
        if let Some(model) = model {
            params.insert("model".to_owned(), Value::String(model.to_owned()));
        }
        self.request_session("new_session", Value::Object(params))
    }

    pub fn resume(&mut self, session_id: &str) -> Result<SessionMetadata, ClientError> {
        self.request_session("resume", serde_json::json!({ "session_id": session_id }))
    }

    fn request_session(
        &mut self,
        method: &str,
        params: Value,
    ) -> Result<SessionMetadata, ClientError> {
        let result: Value = self.request(method, params)?;
        serde_json::from_value(
            result
                .get("session")
                .cloned()
                .ok_or(ClientError::UnexpectedResponse)?,
        )
        .map_err(ClientError::Json)
    }

    pub fn send(&mut self, text: &str) -> Result<bool, ClientError> {
        let result: Value = self.request("send", serde_json::json!({ "text": text }))?;
        Ok(result
            .get("accepted")
            .and_then(Value::as_bool)
            .unwrap_or(false))
    }

    pub fn steer(&mut self, text: &str) -> Result<bool, ClientError> {
        let result: Value = self.request("steer", serde_json::json!({ "text": text }))?;
        Ok(result
            .get("accepted")
            .and_then(Value::as_bool)
            .unwrap_or(false))
    }

    pub fn approve(&mut self, request_id: &str) -> Result<bool, ClientError> {
        self.decision("approve", request_id)
    }

    pub fn deny(&mut self, request_id: &str) -> Result<bool, ClientError> {
        self.decision("deny", request_id)
    }

    fn decision(&mut self, method: &str, request_id: &str) -> Result<bool, ClientError> {
        let result: Value =
            self.request(method, serde_json::json!({ "request_id": request_id }))?;
        Ok(result
            .get("accepted")
            .and_then(Value::as_bool)
            .unwrap_or(false))
    }

    pub fn abort(&mut self) -> Result<bool, ClientError> {
        let result: Value = self.request("abort", Value::Object(Map::new()))?;
        Ok(result
            .get("aborted")
            .and_then(Value::as_bool)
            .unwrap_or(false))
    }

    pub fn status(&mut self) -> Result<StatusResult, ClientError> {
        self.request("status", Value::Object(Map::new()))
    }

    pub fn next_event(&mut self) -> Result<ServerEvent, ClientError> {
        if let Some(event) = self.events.pop_front() {
            return Ok(event);
        }
        loop {
            let value = self.read_frame()?;
            if let Some(event) = parse_event(value)? {
                return Ok(event);
            }
        }
    }

    pub fn try_event(&mut self) -> Option<ServerEvent> {
        self.events.pop_front()
    }

    fn request<T: for<'de> Deserialize<'de>>(
        &mut self,
        method: &str,
        params: Value,
    ) -> Result<T, ClientError> {
        let id = self.next_id;
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or(ClientError::UnexpectedResponse)?;
        let request =
            serde_json::json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        self.write_frame(&request)?;
        loop {
            let value = self.read_frame()?;
            if let Some(event) = parse_event(value.clone())? {
                self.events.push_back(event);
                continue;
            }
            let response: RpcResponse = serde_json::from_value(value)?;
            if response.id != Some(id) {
                return Err(ClientError::UnexpectedResponse);
            }
            if let Some(error) = response.error {
                if error.code == -32002 {
                    let requested = error
                        .data
                        .as_ref()
                        .and_then(|data| data.get("requested"))
                        .and_then(Value::as_str)
                        .unwrap_or(PROTOCOL_VERSION);
                    let supported = error
                        .data
                        .as_ref()
                        .and_then(|data| data.get("supported"))
                        .and_then(Value::as_str)
                        .unwrap_or("unknown");
                    return Err(ClientError::VersionMismatch {
                        requested: requested.to_owned(),
                        supported: supported.to_owned(),
                    });
                }
                return Err(ClientError::Rpc {
                    code: error.code,
                    message: error.message,
                    data: error.data,
                });
            }
            return serde_json::from_value(response.result.ok_or(ClientError::UnexpectedResponse)?)
                .map_err(ClientError::Json);
        }
    }

    fn write_frame(&mut self, value: &Value) -> Result<(), ClientError> {
        let mut frame = serde_json::to_vec(value)?;
        frame.push(b'\n');
        if frame.len() > MAX_FRAME_BYTES {
            return Err(ClientError::FrameTooLarge);
        }
        self.connection.write_all(&frame)?;
        self.connection.flush()?;
        Ok(())
    }

    fn read_frame(&mut self) -> Result<Value, ClientError> {
        loop {
            if let Some(position) = self.read_buffer.iter().position(|byte| *byte == b'\n') {
                let frame = self.read_buffer.drain(..=position).collect::<Vec<_>>();
                if frame.len() > MAX_FRAME_BYTES {
                    return Err(ClientError::FrameTooLarge);
                }
                return serde_json::from_slice(&frame[..frame.len() - 1])
                    .map_err(ClientError::Json);
            }
            let mut chunk = [0_u8; 4096];
            let count = self.connection.read(&mut chunk)?;
            if count == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "server closed the connection",
                )
                .into());
            }
            self.read_buffer.extend_from_slice(&chunk[..count]);
            if self.read_buffer.len() > MAX_FRAME_BYTES {
                return Err(ClientError::FrameTooLarge);
            }
        }
    }
}

#[derive(Debug, Deserialize)]
struct RpcResponse {
    id: Option<u64>,
    result: Option<Value>,
    error: Option<RpcError>,
}

#[derive(Debug, Deserialize)]
struct RpcError {
    code: i64,
    message: String,
    data: Option<Value>,
}

fn parse_event(value: Value) -> Result<Option<ServerEvent>, ClientError> {
    let is_event = value.get("method").and_then(Value::as_str) == Some("event");
    if !is_event {
        return Ok(None);
    }
    let event: EventNotification = serde_json::from_value(value)?;
    Ok(Some(event.params.into_event()?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::os::unix::net::UnixListener;
    use std::thread;

    fn fake_server(path: &Path, response: Value, event: Option<Value>) -> thread::JoinHandle<()> {
        let _ = std::fs::remove_file(path);
        let listener = UnixListener::bind(path).expect("bind fake server");
        thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept fake client");
            let mut reader = BufReader::new(stream.try_clone().expect("clone fake stream"));
            let mut line = String::new();
            reader.read_line(&mut line).expect("read hello");
            if let Some(event) = event {
                serde_json::to_writer(&mut stream, &event).expect("write event");
                stream.write_all(b"\n").expect("write newline");
            }
            serde_json::to_writer(&mut stream, &response).expect("write response");
            stream.write_all(b"\n").expect("write newline");
        })
    }

    #[test]
    fn parses_assistant_delta_and_tool_receipts() {
        let value = serde_json::json!({ "jsonrpc": "2.0", "method": "event", "params": { "event": "assistant_delta", "delta": "hi", "kind": "assistant" } });
        let event = parse_event(value)
            .expect("event parses")
            .expect("event exists");
        assert_eq!(
            event,
            ServerEvent::AssistantDelta {
                session_id: None,
                delta: "hi".to_owned(),
                kind: "assistant".to_owned()
            }
        );
    }

    #[test]
    fn handshake_fails_loudly_on_version_mismatch() {
        let path = std::env::temp_dir().join(format!("zeta-gui-test-{}", std::process::id()));
        let server = fake_server(
            &path,
            serde_json::json!({ "jsonrpc": "2.0", "id": 1, "error": { "code": -32002, "message": "version mismatch", "data": { "requested": "1.0", "supported": "2.0" } } }),
            None,
        );
        let mut client = ProtocolClient::connect_socket(&path).expect("connect fake server");
        let error = client.handshake().expect_err("mismatch should fail");
        assert!(error.to_string().contains("protocol version mismatch"));
        server.join().expect("fake server exits");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn round_trips_every_m1_request_and_notification_family() {
        let path = std::env::temp_dir().join(format!("zeta-gui-round-trip-{}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path).expect("bind scripted server");
        let server = thread::spawn({
            move || {
                let (mut stream, _) = listener.accept().expect("accept scripted client");
                let mut reader = BufReader::new(stream.try_clone().expect("clone scripted stream"));
                for expected_method in [
                    "hello",
                    "list_sessions",
                    "new_session",
                    "resume",
                    "send",
                    "steer",
                    "approve",
                    "deny",
                    "abort",
                    "status",
                ] {
                    let mut line = String::new();
                    reader.read_line(&mut line).expect("read request");
                    let request: Value = serde_json::from_str(&line).expect("request is json");
                    assert_eq!(request["method"], expected_method);
                    let id = request["id"].clone();
                    let result = match expected_method {
                        "hello" => {
                            serde_json::json!({ "protocol_version": "1.0", "server": "zeta", "capabilities": {} })
                        }
                        "list_sessions" => serde_json::json!({ "sessions": [] }),
                        "new_session" | "resume" => {
                            serde_json::json!({ "session": { "session_id": "session-1" } })
                        }
                        "send" | "steer" => serde_json::json!({ "accepted": true }),
                        "approve" => {
                            serde_json::json!({ "accepted": true, "request_id": "approval-1", "decision": "approve" })
                        }
                        "deny" => {
                            serde_json::json!({ "accepted": true, "request_id": "approval-1", "decision": "deny" })
                        }
                        "abort" => serde_json::json!({ "aborted": true }),
                        "status" => {
                            serde_json::json!({ "session": null, "state": "idle", "pending_approvals": [], "usage": {}, "compaction_markers": 0 })
                        }
                        _ => unreachable!(),
                    };
                    serde_json::to_writer(
                        &mut stream,
                        &serde_json::json!({ "jsonrpc": "2.0", "id": id, "result": result }),
                    )
                    .expect("write response");
                    stream.write_all(b"\n").expect("write response newline");
                    if expected_method == "send" {
                        for event in [
                            serde_json::json!({ "event": "turn_start", "data": {} }),
                            serde_json::json!({ "event": "assistant_delta", "delta": "hel", "kind": "assistant" }),
                            serde_json::json!({ "event": "assistant_delta", "delta": "lo", "kind": "assistant" }),
                            serde_json::json!({ "event": "tool_start", "tool_call": { "id": "tool-1", "name": "read", "arguments": {} }, "data": {} }),
                            serde_json::json!({ "event": "tool_output", "tool_call": { "id": "tool-1", "name": "read", "arguments": {} }, "output": "ok", "data": {} }),
                            serde_json::json!({ "event": "tool_end", "tool_call": { "id": "tool-1", "name": "read", "arguments": {} }, "tool_result": { "tool_call_id": "tool-1", "content": "ok", "is_error": false }, "data": {} }),
                            serde_json::json!({ "event": "approval_request", "request_id": "approval-1", "tool_call": { "id": "tool-2", "name": "bash", "arguments": {} } }),
                            serde_json::json!({ "event": "approval_end", "tool_call": { "id": "tool-2", "name": "bash", "arguments": {} }, "data": {} }),
                            serde_json::json!({ "event": "error", "error": { "code": "backend_error", "message": "provider failed" }, "data": {} }),
                            serde_json::json!({ "event": "turn_end", "data": {} }),
                        ] {
                            let envelope = serde_json::json!({ "jsonrpc": "2.0", "method": "event", "params": event });
                            serde_json::to_writer(&mut stream, &envelope).expect("write event");
                            stream.write_all(b"\n").expect("write event newline");
                        }
                    }
                }
            }
        });
        let mut client = ProtocolClient::connect_socket(&path).expect("connect scripted server");
        assert_eq!(client.handshake().expect("hello").protocol_version, "1.0");
        assert!(client.list_sessions().expect("list").is_empty());
        assert_eq!(
            client.new_session(None, None).expect("new").session_id,
            "session-1"
        );
        assert_eq!(
            client.resume("session-1").expect("resume").session_id,
            "session-1"
        );
        assert!(client.send("hello").expect("send"));
        let mut events = Vec::new();
        for _ in 0..10 {
            events.push(client.next_event().expect("event"));
        }
        assert!(
            matches!(events[1], ServerEvent::AssistantDelta { ref delta, .. } if delta == "hel")
        );
        assert!(matches!(events[4], ServerEvent::ToolOutput { ref output, .. } if output == "ok"));
        assert!(matches!(events[6], ServerEvent::ApprovalRequest { .. }));
        assert!(matches!(events[8], ServerEvent::Error { .. }));
        assert!(client.steer("later").expect("steer"));
        assert!(client.approve("approval-1").expect("approve"));
        assert!(client.deny("approval-1").expect("deny"));
        assert!(client.abort().expect("abort"));
        assert_eq!(client.status().expect("status").state, "idle");
        server.join().expect("scripted server exits");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn structured_rpc_errors_are_returned_without_hanging() {
        let path = std::env::temp_dir().join(format!("zeta-gui-error-{}", std::process::id()));
        let server = fake_server(
            &path,
            serde_json::json!({ "jsonrpc": "2.0", "id": 1, "error": { "code": -32003, "message": "no active session", "data": { "method": "send" } } }),
            None,
        );
        let mut client = ProtocolClient::connect_socket(&path).expect("connect fake server");
        let error = client
            .handshake()
            .expect_err("server error should be returned");
        assert!(matches!(error, ClientError::Rpc { code: -32003, .. }));
        server.join().expect("fake server exits");
        let _ = std::fs::remove_file(path);
    }
}
