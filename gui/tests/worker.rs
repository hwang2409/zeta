//! Scripted sockets exercise the same connection actor used by the GUI.
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use zeta_gui::client::ServerEvent;
use zeta_gui::state::{AppState, TranscriptEntry};
use zeta_gui::worker::{CommandMessage, ConnectionWorker, WorkerMessage};

static NEXT_SOCKET: AtomicUsize = AtomicUsize::new(0);

struct Peer {
    reader: BufReader<UnixStream>,
    writer: UnixStream,
}

impl Peer {
    fn accept(listener: &UnixListener) -> Self {
        let (writer, _) = listener.accept().unwrap();
        writer
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        writer
            .set_write_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        Self {
            reader: BufReader::new(writer.try_clone().unwrap()),
            writer,
        }
    }

    fn write(&mut self, value: Value) {
        writeln!(self.writer, "{value}").unwrap();
    }

    fn request(&mut self, method: &str) -> Value {
        let mut line = String::new();
        self.reader.read_line(&mut line).unwrap();
        let value: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(value["method"], method);
        assert_eq!(value["jsonrpc"], "2.0");
        value
    }

    fn respond(&mut self, method: &str, result: Value) -> Value {
        let request = self.request(method);
        self.write(json!({"jsonrpc":"2.0", "id":request["id"], "result":result}));
        request
    }

    fn event(&mut self, mut event: Value) {
        event["session_id"] = json!("session-1");
        self.write(json!({"jsonrpc":"2.0", "method":"event", "params":event}));
    }

    fn hello(&mut self) {
        let request = self.respond("hello", json!({"protocol_version":"1.0", "server":"zeta", "capabilities":{"requests":["list_sessions","new_session","resume","send","steer","approve","deny","abort","status"],"notifications":["event"]}}));
        assert_eq!(request["params"]["protocol_version"], "1.0");
        self.respond("list_sessions", json!({"sessions":[session()]}));
    }

    fn status(&mut self, active: bool, state: &str, approvals: Value) {
        self.respond("status", json!({"session":if active { session() } else { Value::Null }, "state":state,"pending_approvals":approvals,"usage":{},"compaction_markers":0}));
    }

    fn send(&mut self) {
        self.respond("send", json!({"accepted":true,"session_id":"session-1"}));
    }
}

fn session() -> Value {
    json!({"version":1,"session_id":"session-1","created_at":"2026-09-08T12:00:00","updated_at":"2026-09-08T12:00:00","provider":"fake","model":"offline","cwd":"/tmp","retained_tail":10,"compaction_budget":1000,"override_audit":[],"system_prompt":"","context_files":[],"vim_mode":false,"budget_pinned":false,"plan_mode":false,"name":"test"})
}

fn call(id: &str) -> Value {
    json!({"id":id,"name":"read","arguments":{}})
}

struct Harness {
    commands: Sender<CommandMessage>,
    messages: Receiver<WorkerMessage>,
    worker: JoinHandle<()>,
    server: JoinHandle<()>,
    path: PathBuf,
}

impl Harness {
    fn new(script: impl FnOnce(UnixListener) + Send + 'static) -> Self {
        let path = std::env::temp_dir().join(format!(
            "zg-{}-{}.sock",
            std::process::id(),
            NEXT_SOCKET.fetch_add(1, Ordering::Relaxed)
        ));
        let listener = UnixListener::bind(&path).unwrap();
        let server = thread::spawn(move || script(listener));
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
        Self {
            commands,
            messages,
            worker,
            server,
            path,
        }
    }

    fn next(&self) -> WorkerMessage {
        self.messages
            .recv_timeout(Duration::from_secs(5))
            .expect("worker must make progress")
    }

    fn connected(&self) {
        assert!(matches!(self.next(), WorkerMessage::Sessions(_)));
        assert!(matches!(self.next(), WorkerMessage::Status(_)));
        assert!(matches!(self.next(), WorkerMessage::Connected));
    }

    fn command(&self, command: CommandMessage) {
        self.commands.send(command).unwrap();
    }

    fn event(&self) -> ServerEvent {
        match self.next() {
            WorkerMessage::Event(event) => event,
            other => panic!("expected event, got {other:?}"),
        }
    }

    fn finish(self) {
        drop(self.commands);
        self.worker.join().unwrap();
        self.server.join().unwrap();
        std::fs::remove_file(self.path).unwrap();
    }
}

#[test]
fn tool_turn_then_final_answer_and_terminal_error_preserve_order() {
    let (release, wait) = mpsc::channel();
    let harness = Harness::new(move |listener| {
        let mut peer = Peer::accept(&listener);
        peer.hello();
        peer.status(true, "idle", json!([]));
        peer.send();
        for event in [
            json!({"event":"agent_start","data":{}}),
            json!({"event":"turn_start","data":{"turn":1}}),
            json!({"event":"assistant_delta","delta":"checking","kind":"assistant"}),
            json!({"event":"assistant_message","message":{"role":"assistant","content":[{"type":"text","text":"checking"}]}}),
            json!({"event":"tool_start","tool_call":call("one"),"data":{}}),
            json!({"event":"tool_output","tool_call":call("one"),"output":"ok","data":{}}),
            json!({"event":"tool_end","tool_call":call("one"),"tool_result":{"tool_call_id":"one","content":"ok","is_error":false},"data":{}}),
            json!({"event":"turn_end","data":{"turn":1,"tool_calls":1}}),
        ] {
            peer.event(event);
        }
        wait.recv_timeout(Duration::from_secs(5)).unwrap();
        for event in [
            json!({"event":"turn_start","data":{"turn":2}}),
            json!({"event":"assistant_delta","delta":"final answer","kind":"assistant"}),
            json!({"event":"assistant_message","message":{"role":"assistant","content":[{"type":"text","text":"final answer"}]}}),
            json!({"event":"turn_end","data":{"turn":2,"tool_calls":0}}),
            json!({"event":"agent_end","data":{}}),
        ] {
            peer.event(event);
        }
        peer.send();
        peer.event(json!({"event":"error","error":{"code":"server_error","message":"provider failed"},"data":{}}));
        peer.respond("new_session", json!({"session":session()}));
        peer.status(true, "idle", json!([]));
    });
    harness.connected();
    harness.command(CommandMessage::Send("question".into()));
    assert!(matches!(harness.next(), WorkerMessage::Sent(_)));
    let mut state = AppState::default();
    let mut events = Vec::new();
    for _ in 0..8 {
        let event = harness.event();
        state.apply(event.clone());
        events.push(event);
    }
    assert!(matches!(&events[0], ServerEvent::Other { name, .. } if name == "agent_start"));
    assert!(matches!(events[1], ServerEvent::TurnStart { .. }));
    assert!(matches!(events[2], ServerEvent::AssistantDelta { .. }));
    assert!(matches!(events[3], ServerEvent::AssistantMessage { .. }));
    assert!(matches!(events[4], ServerEvent::ToolStart { .. }));
    assert!(matches!(events[5], ServerEvent::ToolOutput { .. }));
    assert!(matches!(events[6], ServerEvent::ToolEnd { .. }));
    assert!(matches!(events[7], ServerEvent::TurnEnd { .. }));
    assert!(
        state.streaming,
        "a model step is not the whole agent request"
    );
    for command in [
        CommandMessage::Send("extra".into()),
        CommandMessage::NewSession,
        CommandMessage::Resume("other".into()),
        CommandMessage::Reconnect,
    ] {
        harness.command(command);
        assert!(matches!(harness.next(), WorkerMessage::Rejected(_)));
    }
    release.send(()).unwrap();
    let final_events: Vec<_> = (0..5).map(|_| harness.event()).collect();
    assert!(matches!(final_events[0], ServerEvent::TurnStart { .. }));
    assert!(matches!(
        final_events[1],
        ServerEvent::AssistantDelta { .. }
    ));
    assert!(matches!(
        final_events[2],
        ServerEvent::AssistantMessage { .. }
    ));
    assert!(matches!(final_events[3], ServerEvent::TurnEnd { .. }));
    assert!(matches!(final_events[4], ServerEvent::AgentEnd { .. }));
    for event in final_events {
        state.apply(event);
    }
    assert!(!state.streaming);
    assert!(
        matches!(state.transcript.last(), Some(TranscriptEntry::Assistant(text)) if text == "final answer")
    );
    harness.command(CommandMessage::Send("fail".into()));
    assert!(matches!(harness.next(), WorkerMessage::Sent(_)));
    assert!(matches!(harness.event(), ServerEvent::Error { .. }));
    harness.command(CommandMessage::NewSession);
    assert!(matches!(harness.next(), WorkerMessage::Session(_)));
    assert!(matches!(harness.next(), WorkerMessage::Status(_)));
    harness.finish();
}

#[test]
fn live_abort_is_processed_between_continuous_events() {
    let harness = Harness::new(|listener| {
        let mut peer = Peer::accept(&listener);
        peer.hello();
        peer.status(true, "idle", json!([]));
        peer.send();
        peer.event(json!({"event":"turn_start","data":{}}));
        // Each incoming abort must be read while this independent writer keeps streaming.
        let mut writer = peer.writer.try_clone().unwrap();
        let (stop, stopped) = mpsc::channel();
        let producer = thread::spawn(move || loop {
            writeln!(writer, "{}", json!({"jsonrpc":"2.0","method":"event","params":{"event":"assistant_delta","delta":"x","kind":"assistant","session_id":"session-1"}})).unwrap();
            if stopped.recv_timeout(Duration::from_millis(1)).is_ok() {
                break;
            }
        });
        let abort = peer.request("abort");
        stop.send(()).unwrap();
        producer.join().unwrap();
        // The real server sends turn_aborted before its RPC acknowledgement.
        peer.event(json!({"event":"turn_aborted","data":{}}));
        peer.write(json!({"jsonrpc":"2.0","id":abort["id"],"result":{"aborted":true}}));
        peer.respond("new_session", json!({"session":session()}));
        peer.status(true, "idle", json!([]));
    });
    harness.connected();
    harness.command(CommandMessage::Send("stream".into()));
    assert!(matches!(harness.next(), WorkerMessage::Sent(_)));
    assert!(matches!(harness.event(), ServerEvent::TurnStart { .. }));
    assert!(matches!(
        harness.event(),
        ServerEvent::AssistantDelta { .. }
    ));
    let start = std::time::Instant::now();
    harness.command(CommandMessage::Abort);
    loop {
        match harness.event() {
            ServerEvent::AssistantDelta { .. } => {}
            ServerEvent::TurnAborted { .. } => break,
            event => panic!("unexpected event: {event:?}"),
        }
    }
    assert!(
        start.elapsed() < Duration::from_secs(1),
        "abort must not wait for output to stop"
    );
    harness.command(CommandMessage::NewSession);
    assert!(matches!(harness.next(), WorkerMessage::Session(_)));
    assert!(matches!(harness.next(), WorkerMessage::Status(_)));
    harness.finish();
}

#[test]
fn reconnect_resumes_selected_session_and_restores_pending_approvals() {
    let (disconnect, wait) = mpsc::channel();
    let harness = Harness::new(move |listener| {
        let mut peer = Peer::accept(&listener);
        peer.hello();
        peer.status(false, "idle", json!([]));
        let request = peer.respond("resume", json!({"session":session()}));
        assert_eq!(request["params"]["session_id"], "session-1");
        peer.status(
            true,
            "idle",
            json!([{ "request_id":"one", "tool_call":call("one") }]),
        );
        let request = peer.respond(
            "deny",
            json!({"accepted":true,"request_id":"one","decision":"deny"}),
        );
        assert_eq!(request["params"]["request_id"], "one");
        peer.status(true, "tool", json!([]));
        peer.event(json!({"event":"approval_end","tool_call":call("one"),"data":{}}));
        peer.event(json!({"event":"tool_end","tool_call":call("one"),"tool_result":{"tool_call_id":"one","content":"denied","is_error":true},"data":{}}));
        peer.status(true, "idle", json!([]));
        // Resuming a pending tool emits no agent_end. The worker polls status.
        peer.status(true, "idle", json!([]));
        wait.recv_timeout(Duration::from_secs(5)).unwrap();
        drop(peer);
        // A restarted server requires resume before the next send.
        let mut peer = Peer::accept(&listener);
        peer.hello();
        let request = peer.respond("resume", json!({"session":session()}));
        assert_eq!(request["params"]["session_id"], "session-1");
        peer.status(true, "idle", json!([]));
        peer.send();
        peer.event(json!({"event":"agent_end","data":{}}));
    });
    harness.connected();
    harness.command(CommandMessage::Resume("session-1".into()));
    assert!(matches!(harness.next(), WorkerMessage::Session(_)));
    let WorkerMessage::Status(status) = harness.next() else {
        panic!("missing status");
    };
    let mut state = AppState::default();
    state.apply_status(status);
    assert_eq!(state.active_session.as_deref(), Some("session-1"));
    assert_eq!(state.approvals[0].request_id, "one");
    harness.command(CommandMessage::Deny("one".into()));
    assert!(matches!(harness.next(), WorkerMessage::Status(status) if status.state == "tool"));
    assert!(matches!(harness.next(), WorkerMessage::Status(status) if status.state == "idle"));
    assert!(matches!(harness.event(), ServerEvent::ApprovalEnd { .. }));
    assert!(matches!(harness.event(), ServerEvent::ToolEnd { .. }));
    assert!(matches!(harness.next(), WorkerMessage::Status(status) if status.state == "idle"));
    disconnect.send(()).unwrap();
    assert!(matches!(harness.next(), WorkerMessage::Lost(_)));
    harness.command(CommandMessage::Reconnect);
    assert!(matches!(harness.next(), WorkerMessage::Sessions(_)));
    assert!(
        matches!(harness.next(), WorkerMessage::Status(status) if status.session.as_ref().unwrap().session_id == "session-1" && status.pending_approvals.is_empty())
    );
    assert!(matches!(harness.next(), WorkerMessage::Connected));
    harness.command(CommandMessage::Send("after restart".into()));
    assert!(matches!(harness.next(), WorkerMessage::Sent(_)));
    assert!(matches!(harness.event(), ServerEvent::AgentEnd { .. }));
    harness.finish();
}

#[test]
fn missing_explicit_socket_fails_without_spawning() {
    let path = std::env::temp_dir().join(format!("zg-missing-{}.sock", std::process::id()));
    let (commands, receiver) = mpsc::channel();
    let (sender, messages) = mpsc::channel();
    let worker = thread::spawn(move || {
        ConnectionWorker {
            commands: receiver,
            messages: sender,
            socket: Some(path),
        }
        .run()
    });
    assert!(matches!(
        messages.recv_timeout(Duration::from_secs(1)).unwrap(),
        WorkerMessage::Lost(_)
    ));
    // Lost remains stable until reconnect. Old code retried every 250ms.
    assert!(messages.recv_timeout(Duration::from_millis(300)).is_err());
    drop(commands);
    worker.join().unwrap();
}
