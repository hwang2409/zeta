//! The connection actor owns the socket, session identity, and command ordering.
use crate::client::{
    ClientError, ModelCatalog, ProtocolClient, ServerEvent, SessionList, SessionMetadata,
    StatusResult,
};
use crate::client::{HistoryMessage, TreeResult};
use crate::login::{LoginProgress, LoginProvider};
use crate::session::{ImageAttachment, SessionSettings};
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
    RenameSession(String, String),
    DeleteSession(String),
    Send(String),
    SendImages(String, Vec<ImageAttachment>),
    SwitchBranch(String),
    ForkMessage(String),
    LoadSettings,
    LoginStart(String),
    LoginCancel(String),
    SetSettings(SessionSettings),
    Approve(String),
    Deny(String),
    Abort,
    Reconnect,
}

#[derive(Debug)]
pub enum WorkerMessage {
    Sessions(SessionList),
    Session(SessionMetadata),
    Status(StatusResult),
    Sent(String),
    Connected,
    Extensions(bool),
    SessionManagement(bool),
    Renamed(SessionMetadata),
    Deleted(String),
    LoginProviders(Vec<LoginProvider>),
    Login(String, LoginProgress),
    Tree(TreeResult),
    History(Vec<HistoryMessage>, bool),
    Settings(SessionSettings, ModelCatalog),
    SettingsApplied(SessionSettings),
    ImagesSent(String, Vec<ImageAttachment>),
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

    fn status(
        &self,
        client: &mut ProtocolClient,
        selected: &mut Option<String>,
        refresh: Option<bool>,
    ) -> Result<StatusResult, ClientError> {
        let result = (|| {
            let status = client.status()?;
            *selected = status
                .session
                .as_ref()
                .map(|session| session.session_id.clone());
            let _ = self.messages.send(WorkerMessage::Status(status.clone()));
            if let Some(replace) = refresh {
                if replace || status.state == "idle" {
                    self.refresh_session(client, selected.as_deref(), replace)?;
                }
            }
            Ok(status)
        })();
        match result {
            Err(error @ ClientError::Rpc { .. }) => {
                *selected = None;
                self.reject(&error.to_string());
                let status = StatusResult {
                    session: None,
                    state: "idle".into(),
                    pending_approvals: Vec::new(),
                    usage: serde_json::Value::Null,
                    compaction_markers: 0,
                };
                let _ = self.messages.send(WorkerMessage::Status(status.clone()));
                Ok(status)
            }
            result => result,
        }
    }

    fn refresh_session(
        &self,
        client: &mut ProtocolClient,
        selected: Option<&str>,
        replace: bool,
    ) -> Result<(), ClientError> {
        if let Some(id) = selected.filter(|_| client.session_extensions) {
            let tree = client.tree(id)?;
            let _ = self.messages.send(WorkerMessage::Tree(tree));
            let history = client.history(id)?;
            let _ = self.messages.send(WorkerMessage::History(history, replace));
        }
        Ok(())
    }

    fn login_update(
        &self,
        client: &mut ProtocolClient,
        method: &str,
        provider: &str,
        pending: &mut std::collections::HashSet<String>,
    ) -> Result<(), ClientError> {
        let progress = match client.login(method, provider) {
            Ok(progress) => progress,
            Err(error @ ClientError::Rpc { .. }) => LoginProgress::failed(error.to_string()),
            Err(error) => return Err(error),
        };
        if progress.busy() {
            pending.insert(provider.to_owned());
        } else {
            pending.remove(provider);
        }
        let _ = self
            .messages
            .send(WorkerMessage::Login(provider.to_owned(), progress));
        Ok(())
    }

    fn connected(
        &self,
        process: &mut Option<Child>,
        selected: &mut Option<String>,
    ) -> Result<bool, ClientError> {
        let mut client = connect(&self.socket, process)?;
        client.handshake()?;
        let _ = self
            .messages
            .send(WorkerMessage::Extensions(client.session_extensions));
        let _ = self
            .messages
            .send(WorkerMessage::SessionManagement(client.session_management));
        if client.login_extensions {
            let providers = client.login_providers()?.providers;
            let _ = self.messages.send(WorkerMessage::LoginProviders(providers));
        }
        let mut pending_logins = std::collections::HashSet::new();
        let mut last_login_status = Instant::now();
        match client.list_sessions() {
            Ok(sessions) => {
                let _ = self.messages.send(WorkerMessage::Sessions(sessions));
            }
            Err(error @ ClientError::Rpc { .. }) => self.reject(&error.to_string()),
            Err(error) => return Err(error),
        }
        if let Some(id) = selected.as_deref() {
            match client.resume(id) {
                Ok(_) => {}
                Err(error @ ClientError::Rpc { .. }) => {
                    *selected = None;
                    self.reject(&format!("could not resume previous session: {error}"));
                }
                Err(error) => return Err(error),
            }
        }
        let status = self.status(&mut client, selected, Some(true))?;
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
                    let login_start = matches!(&command, CommandMessage::LoginStart(_));
                    let approve = matches!(&command, CommandMessage::Approve(_));
                    let result = match command {
                        CommandMessage::NewSession
                        | CommandMessage::Resume(_)
                        | CommandMessage::DeleteSession(_)
                        | CommandMessage::Send(_)
                        | CommandMessage::SendImages(..)
                        | CommandMessage::SwitchBranch(_)
                        | CommandMessage::ForkMessage(_)
                        | CommandMessage::LoadSettings
                        | CommandMessage::SetSettings(_)
                        | CommandMessage::Reconnect
                            if busy || pending_approvals =>
                        {
                            self.reject("finish or abort the current operation before changing sessions or sending");
                            Ok(())
                        }
                        CommandMessage::LoginStart(provider)
                        | CommandMessage::LoginCancel(provider) => {
                            let method = if login_start {
                                "login_start"
                            } else {
                                "login_cancel"
                            };
                            self.login_update(&mut client, method, &provider, &mut pending_logins)
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
                                let status = self.status(&mut client, selected, Some(true))?;
                                busy = status.state != "idle";
                                pending_approvals = !status.pending_approvals.is_empty();
                                Ok(())
                            })
                        }
                        CommandMessage::RenameSession(id, name) => {
                            client.rename_session(&id, &name).map(|session| {
                                let _ = self.messages.send(WorkerMessage::Renamed(session));
                            })
                        }
                        CommandMessage::DeleteSession(id) => client.delete_session(&id).map(|()| {
                            let _ = self.messages.send(WorkerMessage::Deleted(id));
                        }),
                        CommandMessage::Send(text) => client.send(&text).map(|accepted| {
                            if accepted {
                                busy = true;
                                let _ = self.messages.send(WorkerMessage::Sent(text));
                            } else {
                                self.reject("server did not accept the message");
                            }
                        }),
                        CommandMessage::SendImages(text, images) => client
                            .send_images(selected.as_deref().unwrap_or(""), &text, &images)
                            .map(|accepted| {
                                if accepted {
                                    busy = true;
                                    let _ =
                                        self.messages.send(WorkerMessage::ImagesSent(text, images));
                                } else {
                                    self.reject("server did not accept the message");
                                }
                            }),
                        CommandMessage::SwitchBranch(_) | CommandMessage::ForkMessage(_) => {
                            let id = selected.as_deref().unwrap_or("");
                            let result = match command {
                                CommandMessage::SwitchBranch(head) => {
                                    client.switch_branch(id, &head)
                                }
                                CommandMessage::ForkMessage(message) => {
                                    client.fork_message(id, &message)
                                }
                                _ => unreachable!(),
                            };
                            result.and_then(|_| {
                                let status = self.status(&mut client, selected, Some(true))?;
                                busy = status.state != "idle";
                                pending_approvals = !status.pending_approvals.is_empty();
                                Ok(())
                            })
                        }
                        CommandMessage::LoadSettings => {
                            let id = selected.as_deref().unwrap_or("");
                            client.settings(id).and_then(|settings| {
                                let models = client.models(id)?;
                                let _ = self
                                    .messages
                                    .send(WorkerMessage::Settings(settings, models));
                                Ok(())
                            })
                        }
                        CommandMessage::SetSettings(settings) => client
                            .set_settings(selected.as_deref().unwrap_or(""), &settings)
                            .and_then(|settings| {
                                let _ =
                                    self.messages.send(WorkerMessage::SettingsApplied(settings));
                                self.status(&mut client, selected, None)?;
                                Ok(())
                            }),
                        CommandMessage::Approve(id) | CommandMessage::Deny(id) => {
                            let was_idle = !busy;
                            let result = if approve {
                                client.approve(&id)
                            } else {
                                client.deny(&id)
                            };
                            result.and_then(|_| {
                                let status = self.status(&mut client, selected, None)?;
                                busy = status.state != "idle";
                                pending_approvals = !status.pending_approvals.is_empty();
                                resumed_tool |= was_idle && busy;
                                Ok(())
                            })
                        }
                        CommandMessage::Abort => client.abort().map(|_| ()),
                    };
                    match result {
                        Err(error @ (ClientError::Rpc { .. } | ClientError::RequestTooLarge)) => {
                            self.reject(&error.to_string());
                        }
                        Err(error) => return Err(error),
                        Ok(()) => {}
                    }
                }
                Err(TryRecvError::Disconnected) => return Ok(false),
                Err(TryRecvError::Empty) => {}
            }
            if !pending_logins.is_empty()
                && last_login_status.elapsed() >= Duration::from_millis(500)
            {
                for provider in pending_logins.clone() {
                    self.login_update(&mut client, "login_status", &provider, &mut pending_logins)?;
                }
                last_login_status = Instant::now();
            }
            // Resumed approvals run a tool without an agent_end event.
            if resumed_tool && last_status.elapsed() >= Duration::from_millis(100) {
                let status = self.status(&mut client, selected, None)?;
                busy = status.state != "idle";
                pending_approvals = !status.pending_approvals.is_empty();
                resumed_tool = busy;
                last_status = Instant::now();
            }
            client.set_read_timeout(Some(Duration::from_millis(20)))?;
            match client.next_event() {
                Ok(event) => {
                    if event.session_id() != selected.as_deref() {
                        continue;
                    }
                    let refresh_status = matches!(
                        event,
                        ServerEvent::ApprovalEnd { .. }
                            | ServerEvent::TurnEnd { .. }
                            | ServerEvent::AgentEnd { .. }
                            | ServerEvent::TurnAborted { .. }
                            | ServerEvent::Error { .. }
                    );
                    match &event {
                        ServerEvent::AgentEnd { .. }
                        | ServerEvent::TurnAborted { .. }
                        | ServerEvent::Error { .. } => {
                            busy = false;
                            pending_approvals = false;
                        }
                        ServerEvent::TurnStart { .. } => busy = true,
                        ServerEvent::ApprovalRequest { .. } => pending_approvals = true,
                        _ => {}
                    }
                    let _ = self.messages.send(WorkerMessage::Event(event));
                    if refresh_status {
                        client.set_read_timeout(Some(Duration::from_secs(5)))?;
                        let status = self.status(&mut client, selected, Some(false))?;
                        busy = status.state != "idle";
                        pending_approvals = !status.pending_approvals.is_empty();
                    }
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
    connect_or_spawn(&path, socket.is_none(), process, || {
        server_command(&path).spawn()
    })
}

fn server_command(path: &std::path::Path) -> Command {
    let mut command = Command::new(zeta_binary());
    command
        .arg("serve")
        .args(["--provider", "claude", "--model", "claude-sonnet-4-6"])
        .arg("--socket")
        .arg(path);
    command
}

fn connect_or_spawn(
    path: &std::path::Path,
    spawn_allowed: bool,
    process: &mut Option<Child>,
    spawn: impl FnOnce() -> std::io::Result<Child>,
) -> Result<ProtocolClient, ClientError> {
    match ProtocolClient::connect_socket(path) {
        Err(ClientError::Io(error))
            if spawn_allowed
                && matches!(
                    error.kind(),
                    std::io::ErrorKind::NotFound | std::io::ErrorKind::ConnectionRefused
                ) => {}
        result => return result,
    }
    *process = Some(spawn()?);
    for _ in 0..50 {
        match ProtocolClient::connect_socket(path) {
            Err(ClientError::Io(error))
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::NotFound | std::io::ErrorKind::ConnectionRefused
                ) =>
            {
                thread::sleep(Duration::from_millis(100));
            }
            result => return result,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::net::UnixListener;

    #[test]
    fn gui_server_uses_real_default() {
        let command = server_command(std::path::Path::new("/tmp/zeta-test.sock"));
        let args: Vec<_> = command
            .get_args()
            .map(|arg| arg.to_str().unwrap())
            .collect();
        assert!(args.windows(2).any(|pair| pair == ["--provider", "claude"]));
        assert!(args
            .windows(2)
            .any(|pair| pair == ["--model", "claude-sonnet-4-6"]));
        assert!(!args
            .iter()
            .any(|arg| ["fake", "offline", "faster"].contains(arg)));
    }

    #[test]
    fn session_rpc_errors_keep_commands_alive() {
        use serde_json::{json, Value};
        use std::io::{BufRead, BufReader, Write};
        use std::sync::mpsc;

        for (failed_method, after_connect) in [
            ("list_sessions", false),
            ("status", false),
            ("session_tree", false),
            ("session_history", false),
            ("status", true),
            ("session_tree", true),
            ("session_history", true),
        ] {
            let path = env::temp_dir().join(format!(
                "zg-recovery-{}-{failed_method}-{after_connect}.sock",
                std::process::id()
            ));
            let listener = UnixListener::bind(&path).unwrap();
            let server = thread::spawn(move || {
                let (mut socket, _) = listener.accept().unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut reader = BufReader::new(socket.try_clone().unwrap());
                let mut failed = false;
                let mut connected = false;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap() == 0 {
                        break;
                    }
                    let request: Value = serde_json::from_str(&line).unwrap();
                    let method = request["method"].as_str().unwrap();
                    let response =
                        if method == failed_method && !failed && (!after_connect || connected) {
                            failed = true;
                            json!({"jsonrpc":"2.0", "id":request["id"],
                            "error":{"code":-32602,"message":"session metadata is missing"}})
                        } else {
                            let result = match method {
                                "hello" => json!({"protocol_version":"1.1","server":"zeta"}),
                                "list_sessions" => json!({"sessions":[]}),
                                "status" => {
                                    json!({"session":{"session_id":"session-1"},"state":"idle"})
                                }
                                "session_tree" => json!({"branches":[]}),
                                "session_history" => json!({"messages":[],"next_offset":null}),
                                "new_session" => json!({"session":{"session_id":"session-1"}}),
                                "send" => json!({"accepted":true,"session_id":"session-1"}),
                                other => panic!("unexpected request: {other}"),
                            };
                            json!({"jsonrpc":"2.0","id":request["id"],"result":result})
                        };
                    writeln!(socket, "{response}").unwrap();
                    if after_connect && !connected && method == "session_history" {
                        connected = true;
                        writeln!(
                            socket,
                            "{}",
                            json!({"jsonrpc":"2.0", "method":"event",
                            "params":{"event":"agent_end","session_id":"session-1","data":{}}})
                        )
                        .unwrap();
                    }
                }
                assert!(failed);
            });
            let (commands, receiver) = mpsc::channel();
            let (sender, messages) = mpsc::channel();
            let worker_path = path.clone();
            let worker = thread::spawn(move || {
                ConnectionWorker {
                    commands: receiver,
                    messages: sender,
                    socket: Some(worker_path),
                }
                .run()
            });
            let mut rejected = false;
            let mut connected = false;
            let mut cleared = failed_method == "list_sessions";
            while !rejected || !connected || !cleared {
                match messages.recv_timeout(Duration::from_secs(5)).unwrap() {
                    WorkerMessage::Rejected(error) => {
                        assert!(error.contains("session metadata is missing"));
                        rejected = true;
                    }
                    WorkerMessage::Connected => connected = true,
                    WorkerMessage::Status(status)
                        if rejected && failed_method != "list_sessions" =>
                    {
                        cleared = status.session.is_none();
                    }
                    WorkerMessage::Lost(error) => {
                        panic!("{failed_method} lost connection: {error}")
                    }
                    _ => {}
                }
            }
            commands.send(CommandMessage::NewSession).unwrap();
            loop {
                match messages.recv_timeout(Duration::from_secs(5)).unwrap() {
                    WorkerMessage::Session(_) => break,
                    WorkerMessage::Lost(error) | WorkerMessage::Rejected(error) => {
                        panic!("{error}")
                    }
                    _ => {}
                }
            }
            commands
                .send(CommandMessage::Send("still connected".into()))
                .unwrap();
            loop {
                match messages.recv_timeout(Duration::from_secs(5)).unwrap() {
                    WorkerMessage::Sent(_) => break,
                    WorkerMessage::Lost(error) | WorkerMessage::Rejected(error) => {
                        panic!("{error}")
                    }
                    _ => {}
                }
            }
            drop(commands);
            worker.join().unwrap();
            server.join().unwrap();
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn session_management_actor_renames_clears_and_recovers_from_delete_error() {
        use serde_json::{json, Value};
        use std::io::{BufRead, BufReader, Write};
        use std::sync::mpsc;
        let path =
            env::temp_dir().join(format!("zg-management-worker-{}.sock", std::process::id()));
        let listener = UnixListener::bind(&path).unwrap();
        let server = thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut reader = BufReader::new(socket.try_clone().unwrap());
            let mut mutations = 0;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap() == 0 {
                    break;
                }
                let request: Value = serde_json::from_str(&line).unwrap();
                let method = request["method"].as_str().unwrap();
                let result = match method {
                    "hello" => {
                        json!({"protocol_version":"1.1","server":"zeta","capabilities":{"requests":["rename_session","delete_session"]}})
                    }
                    "list_sessions" => json!({"sessions":[{"session_id":"stored"}]}),
                    "status" => json!({"session":null,"state":"idle"}),
                    "rename_session" => {
                        assert_eq!(request["params"]["session_id"], "stored");
                        assert_eq!(
                            request["params"]["name"],
                            if mutations == 0 { "name" } else { "" }
                        );
                        mutations += 1;
                        json!({"session":{"session_id":"stored","name":request["params"]["name"]}})
                    }
                    "delete_session" => {
                        mutations += 1;
                        if mutations == 3 {
                            writeln!(socket, "{}", json!({"id":request["id"],"error":{"code":-32005,"message":"session is open"}})).unwrap();
                            continue;
                        }
                        assert_eq!(request["params"]["session_id"], "stored");
                        json!({"session_id":"stored"})
                    }
                    other => panic!("unexpected request: {other}"),
                };
                writeln!(socket, "{}", json!({"id":request["id"],"result":result})).unwrap();
            }
            assert_eq!(mutations, 4);
        });
        let (commands, receiver) = mpsc::channel();
        let (sender, messages) = mpsc::channel();
        let worker_path = path.clone();
        let worker = thread::spawn(move || {
            ConnectionWorker {
                commands: receiver,
                messages: sender,
                socket: Some(worker_path),
            }
            .run()
        });
        let mut available = false;
        loop {
            match messages.recv_timeout(Duration::from_secs(5)).unwrap() {
                WorkerMessage::SessionManagement(enabled) => available = enabled,
                WorkerMessage::Connected => break,
                WorkerMessage::Lost(error) | WorkerMessage::Rejected(error) => panic!("{error}"),
                _ => {}
            }
        }
        assert!(available);
        for name in ["name", ""] {
            commands
                .send(CommandMessage::RenameSession("stored".into(), name.into()))
                .unwrap();
            assert!(
                matches!(messages.recv_timeout(Duration::from_secs(5)).unwrap(), WorkerMessage::Renamed(session) if session.name == name)
            );
        }
        commands
            .send(CommandMessage::DeleteSession("stored".into()))
            .unwrap();
        assert!(
            matches!(messages.recv_timeout(Duration::from_secs(5)).unwrap(), WorkerMessage::Rejected(error) if error.contains("session is open"))
        );
        commands
            .send(CommandMessage::DeleteSession("stored".into()))
            .unwrap();
        assert!(
            matches!(messages.recv_timeout(Duration::from_secs(5)).unwrap(), WorkerMessage::Deleted(id) if id == "stored")
        );
        drop(commands);
        worker.join().unwrap();
        server.join().unwrap();
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn login_actor_polls_completion_and_cancels_without_blocking_commands() {
        use serde_json::{json, Value};
        use std::io::{BufRead, BufReader, Write};
        use std::sync::mpsc;
        let path = env::temp_dir().join(format!("zg-login-worker-{}.sock", std::process::id()));
        let listener = UnixListener::bind(&path).unwrap();
        let server = thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut reader = BufReader::new(socket.try_clone().unwrap());
            let mut methods = Vec::new();
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap() == 0 {
                    break;
                }
                let request: Value = serde_json::from_str(&line).unwrap();
                let method = request["method"].as_str().unwrap();
                methods.push(method.to_owned());
                let result = match method {
                    "hello" => {
                        json!({"protocol_version":"1.1","server":"zeta","capabilities":{"requests":["login_start","login_status","login_cancel","login_providers"]}})
                    }
                    "login_providers" => {
                        json!({"providers":[{"provider":"claude","credentials_present":false},{"provider":"codex","credentials_present":false}]})
                    }
                    "list_sessions" => json!({"sessions":[]}),
                    "status" => json!({"session":null,"state":"idle"}),
                    "login_start" => {
                        json!({"state":"pending","authorization_url":"https://authorize.invalid/"})
                    }
                    "login_status" => {
                        assert_eq!(request["params"]["provider"], "claude");
                        json!({"state":"succeeded"})
                    }
                    "login_cancel" => {
                        assert_eq!(request["params"]["provider"], "codex");
                        json!({"state":"cancelled"})
                    }
                    other => panic!("unexpected request {other}"),
                };
                writeln!(socket, "{}", json!({"id":request["id"],"result":result})).unwrap();
            }
            assert!(methods.iter().any(|method| method == "login_status"));
            assert!(methods.iter().any(|method| method == "login_cancel"));
        });
        let (commands, receiver) = mpsc::channel();
        let (sender, messages) = mpsc::channel();
        let worker_path = path.clone();
        let worker = thread::spawn(move || {
            ConnectionWorker {
                commands: receiver,
                messages: sender,
                socket: Some(worker_path),
            }
            .run()
        });
        loop {
            if matches!(
                messages.recv_timeout(Duration::from_secs(5)).unwrap(),
                WorkerMessage::Connected
            ) {
                break;
            }
        }
        commands
            .send(CommandMessage::LoginStart("claude".into()))
            .unwrap();
        let next_login = || loop {
            match messages.recv_timeout(Duration::from_secs(5)).unwrap() {
                WorkerMessage::Login(provider, progress) => break (provider, progress),
                WorkerMessage::Lost(error) | WorkerMessage::Rejected(error) => panic!("{error}"),
                _ => {}
            }
        };
        assert!(
            matches!(next_login(), (provider, LoginProgress::Pending { authorization_url: Some(url) }) if provider == "claude" && url == "https://authorize.invalid/")
        );
        assert_eq!(next_login(), ("claude".into(), LoginProgress::Succeeded));
        commands
            .send(CommandMessage::LoginStart("codex".into()))
            .unwrap();
        assert!(
            matches!(next_login(), (provider, LoginProgress::Pending { .. }) if provider == "codex")
        );
        commands
            .send(CommandMessage::LoginCancel("codex".into()))
            .unwrap();
        assert_eq!(next_login(), ("codex".into(), LoginProgress::Cancelled));
        drop(commands);
        worker.join().unwrap();
        server.join().unwrap();
        std::fs::remove_file(path).unwrap();
    }

    fn stale_socket(path: &std::path::Path) {
        // Bind without listening: the path exists but cannot accept connections,
        // regardless of when macOS finishes closing the process's socket.
        assert!(Command::new("python3")
            .arg("-c")
            .arg("import socket,sys; s=socket.socket(socket.AF_UNIX); s.bind(sys.argv[1])")
            .arg(path)
            .status()
            .unwrap()
            .success());
    }

    #[test]
    fn default_connection_spawns_for_missing_and_stale_sockets() {
        for stale in [false, true] {
            let path =
                env::temp_dir().join(format!("zg-spawn-{}-{stale}.sock", std::process::id()));
            if stale {
                stale_socket(&path);
                assert!(path.exists());
            }
            let mut process = None;
            let client = connect_or_spawn(&path, true, &mut process, || {
                Command::new("python3")
                    .arg("-c")
                    .arg("import os,socket,sys,time; p=sys.argv[1]; time.sleep(0.15); os.path.exists(p) and os.unlink(p); s=socket.socket(socket.AF_UNIX); s.bind(p); s.listen(1); c,_=s.accept(); c.recv(1)")
                    .arg(&path)
                    .spawn()
            }).unwrap();
            drop(client);
            assert!(process.unwrap().wait().unwrap().success());
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn existing_server_connects_without_spawning_and_explicit_stale_socket_fails() {
        let path = env::temp_dir().join(format!("zg-existing-{}.sock", std::process::id()));
        let listener = UnixListener::bind(&path).unwrap();
        let mut process = None;
        let client = connect_or_spawn(&path, true, &mut process, || {
            panic!("must reuse the server")
        })
        .unwrap();
        drop(listener.accept().unwrap());
        drop(client);
        drop(listener);
        std::fs::remove_file(&path).unwrap();
        stale_socket(&path);
        assert!(matches!(
            connect_or_spawn(&path, false, &mut process, || panic!(
                "explicit paths never spawn"
            )),
            Err(ClientError::Io(_))
        ));
        assert!(
            path.exists(),
            "the GUI must not remove stale socket files itself"
        );
        assert!(process.is_none());
        std::fs::remove_file(path).unwrap();
    }
}
