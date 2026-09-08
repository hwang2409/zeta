//! The connection actor owns the socket, session identity, and command ordering.
use crate::client::{ClientError, ProtocolClient, ServerEvent, SessionMetadata, StatusResult};
use std::env;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Debug)]
pub enum CommandMessage {
    NewSession,
    Resume(String),
    Send(String),
    Approve(String),
    Deny(String),
    Abort,
    Reconnect,
}

#[derive(Debug)]
pub enum WorkerMessage {
    Sessions(Vec<SessionMetadata>),
    Session(SessionMetadata),
    Status(StatusResult),
    Sent(String),
    Connected,
    Event(ServerEvent),
    Rejected(String),
    Lost(String),
}

pub struct ConnectionWorker {
    pub commands: Receiver<CommandMessage>,
    pub messages: Sender<WorkerMessage>,
    pub socket: Option<PathBuf>,
}

impl ConnectionWorker {
    pub fn run(self) {
        let mut process = None;
        let mut selected = None;
        loop {
            let result = self.connected(&mut process, &mut selected);
            if let Err(error) = result {
                let _ = self.messages.send(WorkerMessage::Lost(error.to_string()));
            } else if matches!(result, Ok(true)) {
                continue;
            } else {
                return; // The UI closed its command channel.
            }
            loop {
                match self.commands.recv() {
                    Ok(CommandMessage::Reconnect) => break,
                    Ok(_) => self.reject("connection lost; reconnect before sending commands"),
                    Err(_) => return,
                }
            }
        }
    }

    fn reject(&self, reason: &str) {
        let _ = self
            .messages
            .send(WorkerMessage::Rejected(reason.to_owned()));
    }

    fn status(&self, client: &mut ProtocolClient) -> Result<StatusResult, ClientError> {
        let status = client.status()?;
        let _ = self.messages.send(WorkerMessage::Status(status.clone()));
        Ok(status)
    }

    fn connected(
        &self,
        process: &mut Option<Child>,
        selected: &mut Option<String>,
    ) -> Result<bool, ClientError> {
        let mut client = connect(&self.socket, process)?;
        client.handshake()?;
        let sessions = client.list_sessions()?;
        let _ = self.messages.send(WorkerMessage::Sessions(sessions));
        if let Some(id) = selected.as_deref() {
            client.resume(id)?;
        }
        let status = self.status(&mut client)?;
        *selected = status
            .session
            .as_ref()
            .map(|session| session.session_id.clone());
        let mut busy = status.state != "idle";
        let mut pending_approvals = !status.pending_approvals.is_empty();
        let _ = self.messages.send(WorkerMessage::Connected);
        let mut resumed_tool = false;
        let mut last_status = Instant::now();
        loop {
            // Commands get a chance between every event, even with continuous output.
            client.set_read_timeout(Some(Duration::from_secs(5)))?;
            match self.commands.try_recv() {
                Ok(command) => {
                    let approve = matches!(&command, CommandMessage::Approve(_));
                    let result = match command {
                        CommandMessage::NewSession
                        | CommandMessage::Resume(_)
                        | CommandMessage::Send(_)
                        | CommandMessage::Reconnect
                            if busy || pending_approvals =>
                        {
                            self.reject("finish or abort the current operation before changing sessions or sending");
                            Ok(())
                        }
                        CommandMessage::Reconnect => return Ok(true),
                        CommandMessage::NewSession | CommandMessage::Resume(_) => {
                            let session = match command {
                                CommandMessage::Resume(id) => client.resume(&id),
                                _ => client.new_session(None, None),
                            };
                            session.and_then(|session| {
                                *selected = Some(session.session_id.clone());
                                let _ = self.messages.send(WorkerMessage::Session(session));
                                let status = self.status(&mut client)?;
                                busy = status.state != "idle";
                                pending_approvals = !status.pending_approvals.is_empty();
                                Ok(())
                            })
                        }
                        CommandMessage::Send(text) => client.send(&text).map(|accepted| {
                            if accepted {
                                busy = true;
                                let _ = self.messages.send(WorkerMessage::Sent(text));
                            } else {
                                self.reject("server did not accept the message");
                            }
                        }),
                        CommandMessage::Approve(id) | CommandMessage::Deny(id) => {
                            let was_idle = !busy;
                            let result = if approve {
                                client.approve(&id)
                            } else {
                                client.deny(&id)
                            };
                            result.and_then(|_| {
                                let status = self.status(&mut client)?;
                                busy = status.state != "idle";
                                pending_approvals = !status.pending_approvals.is_empty();
                                resumed_tool |= was_idle && busy;
                                Ok(())
                            })
                        }
                        CommandMessage::Abort => client.abort().map(|_| ()),
                    };
                    match result {
                        Err(error @ ClientError::Rpc { .. }) => self.reject(&error.to_string()),
                        Err(error) => return Err(error),
                        Ok(()) => {}
                    }
                }
                Err(TryRecvError::Disconnected) => return Ok(false),
                Err(TryRecvError::Empty) => {}
            }
            // Resumed approvals run a tool without an agent_end event.
            if resumed_tool && last_status.elapsed() >= Duration::from_millis(100) {
                let status = self.status(&mut client)?;
                busy = status.state != "idle";
                pending_approvals = !status.pending_approvals.is_empty();
                resumed_tool = busy;
                last_status = Instant::now();
            }
            client.set_read_timeout(Some(Duration::from_millis(20)))?;
            match client.next_event() {
                Ok(event) => {
                    match &event {
                        ServerEvent::AgentEnd { .. }
                        | ServerEvent::TurnAborted { .. }
                        | ServerEvent::Error { .. } => {
                            busy = false;
                            pending_approvals = false;
                        }
                        ServerEvent::TurnStart { .. } => busy = true,
                        ServerEvent::ApprovalRequest { .. } => pending_approvals = true,
                        ServerEvent::ApprovalEnd { .. } => {
                            // status handles multiple pending approvals and resumed tools.
                            client.set_read_timeout(Some(Duration::from_secs(5)))?;
                            let status = self.status(&mut client)?;
                            pending_approvals = !status.pending_approvals.is_empty();
                        }
                        _ => {}
                    }
                    let _ = self.messages.send(WorkerMessage::Event(event));
                }
                Err(ClientError::Io(error))
                    if matches!(
                        error.kind(),
                        std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
                    ) => {}
                Err(error) => return Err(error),
            }
        }
    }
}

fn connect(
    socket: &Option<PathBuf>,
    process: &mut Option<Child>,
) -> Result<ProtocolClient, ClientError> {
    let path = socket.clone().unwrap_or_else(default_socket);
    if socket.is_none() && !path.exists() {
        let mut command = Command::new(zeta_binary());
        command.arg("serve").arg("--socket").arg(&path);
        *process = Some(command.spawn().map_err(ClientError::Io)?);
        for _ in 0..50 {
            if path.exists() {
                break;
            }
            thread::sleep(Duration::from_millis(100));
        }
    }
    ProtocolClient::connect_socket(path)
}

fn zeta_binary() -> String {
    env::var("ZETA_BIN").unwrap_or_else(|_| "zeta".to_owned())
}

fn default_socket() -> PathBuf {
    if let Some(home) = env::var_os("ZETA_HOME") {
        return PathBuf::from(home).join("run/serve.sock");
    }
    PathBuf::from(env::var_os("HOME").unwrap_or_default()).join(".zeta/run/serve.sock")
}
