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

    fn wait_for_close(&mut self) {
        // Keep the server alive until the actor consumes its last event/response.
        // Closing earlier can make macOS reject the actor's timeout update.
        assert_eq!(self.reader.read_line(&mut String::new()).unwrap(), 0);
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
        if event.get("session_id").is_none() {
            event["session_id"] = json!("session-1");
        }
        self.write(json!({"jsonrpc":"2.0", "method":"event", "params":event}));
    }

    fn boundary(&mut self, event: Value) {
        let running = event["event"] == "turn_end";
        self.event(event);
        self.status(true, if running { "running" } else { "idle" }, json!([]));
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
        let message = self
            .messages
            .recv_timeout(Duration::from_secs(5))
            .expect("worker must make progress");
        if matches!(message, WorkerMessage::Extensions(false)) {
            self.next()
        } else {
            message
        }
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
            WorkerMessage::Event(event) => {
                if matches!(
                    event,
                    ServerEvent::TurnEnd { .. }
                        | ServerEvent::AgentEnd { .. }
                        | ServerEvent::TurnAborted { .. }
                        | ServerEvent::Error { .. }
                ) {
                    assert!(matches!(self.next(), WorkerMessage::Status(_)));
                }
                event
            }
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
            if matches!(event["event"].as_str(), Some("turn_end" | "agent_end")) {
                peer.boundary(event);
            } else {
                peer.event(event);
            }
        }
        wait.recv_timeout(Duration::from_secs(5)).unwrap();
        for event in [
            json!({"event":"turn_start","data":{"turn":2}}),
            json!({"event":"assistant_delta","delta":"final answer","kind":"assistant"}),
            json!({"event":"assistant_message","message":{"role":"assistant","content":[{"type":"text","text":"final answer"}]}}),
            json!({"event":"turn_end","data":{"turn":2,"tool_calls":0}}),
            json!({"event":"agent_end","data":{}}),
        ] {
            if matches!(event["event"].as_str(), Some("turn_end" | "agent_end")) {
                peer.boundary(event);
            } else {
                peer.event(event);
            }
        }
        peer.send();
        peer.boundary(json!({"event":"error","error":{"code":"server_error","message":"provider failed"},"data":{}}));
        peer.respond("new_session", json!({"session":session()}));
        peer.status(true, "idle", json!([]));
        peer.wait_for_close();
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
        matches!(state.transcript.last(), Some(TranscriptEntry::Assistant(text)) if text.source.as_ref() == "final answer")
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
        peer.status(true, "idle", json!([]));
        peer.respond("new_session", json!({"session":session()}));
        peer.status(true, "idle", json!([]));
        peer.wait_for_close();
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
        peer.boundary(json!({"event":"agent_end","data":{}}));
        peer.wait_for_close();
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
    assert!(matches!(harness.event(), ServerEvent::ApprovalEnd { .. }));
    assert!(matches!(harness.next(), WorkerMessage::Status(status) if status.state == "idle"));
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
fn stale_remembered_session_reconnects_and_allows_a_new_session() {
    let harness = Harness::new(|listener| {
        let mut peer = Peer::accept(&listener);
        peer.hello();
        peer.status(true, "idle", json!([]));
        peer.wait_for_close();

        let mut peer = Peer::accept(&listener);
        peer.hello();
        let request = peer.request("resume");
        assert_eq!(request["params"]["session_id"], "session-1");
        peer.write(json!({"jsonrpc":"2.0","id":request["id"],"error":{"code":-32602,"message":"session session-1 was not found"}}));
        peer.status(false, "idle", json!([]));
        peer.respond("new_session", json!({"session":session()}));
        peer.status(true, "idle", json!([]));
        peer.wait_for_close();
    });
    harness.connected();
    harness.command(CommandMessage::Reconnect);
    assert!(matches!(harness.next(), WorkerMessage::Sessions(list) if list.sessions.len() == 1));
    assert!(
        matches!(harness.next(), WorkerMessage::Rejected(error) if error == "could not resume previous session: server error -32602: session session-1 was not found")
    );
    let WorkerMessage::Status(status) = harness.next() else {
        panic!("expected cleared session status");
    };
    assert!(status.session.is_none());
    let mut state = AppState::default();
    state.select_session(Some("session-1".into()));
    state.apply_status(status);
    assert!(state.active_session.is_none());
    assert!(matches!(harness.next(), WorkerMessage::Connected));
    harness.command(CommandMessage::NewSession);
    assert!(
        matches!(harness.next(), WorkerMessage::Session(session) if session.session_id == "session-1")
    );
    assert!(
        matches!(harness.next(), WorkerMessage::Status(status) if status.session.as_ref().unwrap().session_id == "session-1")
    );
    harness.finish();
}

#[test]
fn explicit_missing_session_is_rejected_without_losing_the_connection() {
    let harness = Harness::new(|listener| {
        let mut peer = Peer::accept(&listener);
        peer.hello();
        peer.status(true, "idle", json!([]));
        let request = peer.request("resume");
        assert_eq!(request["params"]["session_id"], "missing");
        peer.write(json!({"jsonrpc":"2.0","id":request["id"],"error":{"code":-32602,"message":"session missing was not found"}}));
        peer.send();
        peer.boundary(json!({"event":"agent_end","data":{}}));
        peer.wait_for_close();
    });
    harness.connected();
    harness.command(CommandMessage::Resume("missing".into()));
    assert!(
        matches!(harness.next(), WorkerMessage::Rejected(error) if error == "server error -32602: session missing was not found")
    );
    harness.command(CommandMessage::Send("still connected".into()));
    assert!(matches!(harness.next(), WorkerMessage::Sent(text) if text == "still connected"));
    assert!(
        matches!(harness.event(), ServerEvent::AgentEnd { session_id, .. } if session_id.as_deref() == Some("session-1"))
    );
    harness.finish();
}

#[test]
fn transport_failure_during_resume_loses_the_connection() {
    for remembered in [false, true] {
        let harness = Harness::new(move |listener| {
            let mut peer = Peer::accept(&listener);
            peer.hello();
            peer.status(true, "idle", json!([]));
            if remembered {
                peer.wait_for_close();
                peer = Peer::accept(&listener);
                peer.hello();
            }
            let request = peer.request("resume");
            assert_eq!(request["params"]["session_id"], "session-1");
            // Close the socket before replying to the resume request.
        });
        harness.connected();
        if remembered {
            harness.command(CommandMessage::Reconnect);
            assert!(matches!(harness.next(), WorkerMessage::Sessions(_)));
        } else {
            harness.command(CommandMessage::Resume("session-1".into()));
        }
        assert!(matches!(harness.next(), WorkerMessage::Lost(_)));
        harness.command(CommandMessage::NewSession);
        assert!(
            matches!(harness.next(), WorkerMessage::Rejected(error) if error == "connection lost; reconnect before sending commands")
        );
        harness.finish();
    }
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

#[test]
fn background_tool_events_do_not_block_a_foreground_send() {
    let harness = Harness::new(|listener| {
        let mut peer = Peer::accept(&listener);
        peer.hello();
        peer.status(true, "idle", json!([]));
        peer.event(json!({"event":"tool_start","tool_call":call("same-provider-id"),"data":{"agent_instance_id":"child-one"}}));
        peer.event(json!({"event":"tool_end","tool_call":call("same-provider-id"),"tool_result":null,"data":{"agent_instance_id":"child-one"}}));
        peer.send();
        peer.boundary(json!({"event":"agent_end","data":{}}));
        peer.wait_for_close();
    });
    harness.connected();
    let mut state = AppState::default();
    state.apply(harness.event());
    state.apply(harness.event());
    assert!(!state.streaming);
    harness.command(CommandMessage::Send("after background tool".into()));
    assert!(matches!(harness.next(), WorkerMessage::Sent(_)));
    assert!(matches!(harness.event(), ServerEvent::AgentEnd { .. }));
    harness.finish();
}

#[test]
fn old_session_events_queued_during_a_switch_do_not_leak_or_block_sends() {
    let harness = Harness::new(|listener| {
        let mut peer = Peer::accept(&listener);
        peer.hello();
        peer.status(true, "idle", json!([]));
        let request = peer.request("new_session");
        // These arrive before the RPC response, but the client delivers them
        // after the actor switches to the new session.
        for event in [
            json!({"event":"turn_start","data":{}}),
            json!({"event":"assistant_delta","delta":"old text","kind":"assistant"}),
            json!({"event":"tool_start","tool_call":call("old"),"data":{}}),
            json!({"event":"approval_request","tool_call":call("old"),"request_id":"old"}),
            json!({"event":"approval_end","tool_call":call("old"),"data":{}}),
            json!({"event":"error","error":{"code":"old","message":"old failure"},"data":{}}),
        ] {
            peer.event(event);
        }
        let mut new_session = session();
        new_session["session_id"] = json!("session-2");
        peer.write(json!({"jsonrpc":"2.0","id":request["id"],"result":{"session":new_session}}));
        peer.respond(
            "status",
            json!({"session":new_session,"state":"idle","pending_approvals":[]}),
        );
        peer.event(json!({"event":"assistant_delta","delta":"new text","kind":"assistant","session_id":"session-2"}));
        peer.respond("send", json!({"accepted":true,"session_id":"session-2"}));
        peer.boundary(json!({"event":"agent_end","data":{},"session_id":"session-2"}));
        peer.wait_for_close();
    });
    harness.connected();
    harness.command(CommandMessage::NewSession);
    assert!(matches!(harness.next(), WorkerMessage::Session(_)));
    let WorkerMessage::Status(status) = harness.next() else {
        panic!("missing status");
    };
    let mut state = AppState::default();
    state.apply_status(status);
    state.apply(harness.event());
    assert_eq!(
        state.transcript,
        vec![TranscriptEntry::Assistant(
            zeta_gui::markdown::Markdown::streaming("new text".into())
        )]
    );
    assert!(!state.streaming);
    assert!(state.approvals.is_empty());
    harness.command(CommandMessage::Send("new question".into()));
    assert!(matches!(harness.next(), WorkerMessage::Sent(_)));
    assert!(matches!(harness.event(), ServerEvent::AgentEnd { .. }));
    harness.finish();
}

#[test]
fn approval_end_preserves_the_other_delegated_request_with_the_same_raw_id() {
    let first = json!({"request_id":"delegated-one","tool_call":call("same-provider-id")});
    let second = json!({"request_id":"delegated-two","tool_call":call("same-provider-id")});
    let harness = Harness::new(move |listener| {
        let mut peer = Peer::accept(&listener);
        peer.hello();
        peer.status(true, "idle", json!([first, second]));
        for (request_id, agent, remaining) in [
            ("delegated-one", "child-one", json!([second])),
            ("delegated-two", "child-two", json!([])),
        ] {
            let request = peer.request("deny");
            assert_eq!(request["params"]["request_id"], request_id);
            // The event may already be queued when the command's status arrives.
            peer.event(json!({"event":"approval_end","tool_call":call("same-provider-id"),"data":{"agent_instance_id":agent}}));
            peer.write(json!({"jsonrpc":"2.0","id":request["id"],"result":{"accepted":true}}));
            peer.status(true, "idle", remaining.clone());
            peer.status(true, "idle", remaining);
        }
        peer.send();
        peer.boundary(json!({"event":"agent_end","data":{}}));
        peer.wait_for_close();
    });
    assert!(matches!(harness.next(), WorkerMessage::Sessions(_)));
    let WorkerMessage::Status(status) = harness.next() else {
        panic!("missing status");
    };
    let mut state = AppState::default();
    state.apply_status(status);
    assert!(matches!(harness.next(), WorkerMessage::Connected));
    assert_eq!(state.approvals.len(), 2);
    for (request, remaining) in [("delegated-one", 1), ("delegated-two", 0)] {
        harness.command(CommandMessage::Deny(request.into()));
        let WorkerMessage::Status(status) = harness.next() else {
            panic!("missing status");
        };
        state.apply_status(status);
        state.apply(harness.event());
        assert_eq!(state.approvals.len(), remaining);
        let WorkerMessage::Status(status) = harness.next() else {
            panic!("missing status");
        };
        state.apply_status(status);
        assert_eq!(state.approvals.len(), remaining);
        if remaining == 1 {
            assert_eq!(state.approvals[0].request_id, "delegated-two");
        }
    }
    harness.command(CommandMessage::Send("after approvals".into()));
    assert!(matches!(harness.next(), WorkerMessage::Sent(_)));
    assert!(matches!(harness.event(), ServerEvent::AgentEnd { .. }));
    harness.finish();
}

#[test]
fn oversized_send_is_rejected_without_losing_the_connection() {
    use zeta_gui::client::MAX_FRAME_BYTES;
    let harness = Harness::new(|listener| {
        let mut peer = Peer::accept(&listener);
        peer.hello();
        peer.status(true, "idle", json!([]));
        // Neither oversized request may reach this healthy socket.
        let request = peer.respond("send", json!({"accepted":true}));
        assert_eq!(request["params"]["text"], "short message");
        peer.boundary(json!({"event":"agent_end","data":{}}));
        peer.wait_for_close();
    });
    harness.connected();
    for text in [
        "x".repeat(MAX_FRAME_BYTES + 1),
        "\n".repeat(MAX_FRAME_BYTES / 2),
    ] {
        harness.command(CommandMessage::Send(text));
        match harness.next() {
            WorkerMessage::Rejected(error) => {
                assert!(error.contains("1048576-byte (1 MiB)"));
                assert!(error.contains("encoded request"));
            }
            other => panic!("expected command rejection, got {other:?}"),
        }
    }
    harness.command(CommandMessage::Send("short message".into()));
    assert!(matches!(harness.next(), WorkerMessage::Sent(text) if text == "short message"));
    assert!(matches!(harness.event(), ServerEvent::AgentEnd { .. }));
    harness.finish();
}

#[test]
fn delegated_cards_keep_separate_tails_disclosure_and_failures() {
    let harness = Harness::new(|listener| {
        let mut peer = Peer::accept(&listener);
        peer.hello();
        peer.status(true, "idle", json!([]));
        for agent in [Value::Null, json!("child-one"), json!("child-two")] {
            peer.event(json!({"event":"tool_start","tool_call":call("duplicate"),"data":{"agent_instance_id":agent}}));
        }
        for (agent, output, error) in [
            (json!("child-two"), "failed child\n".repeat(30), true),
            (Value::Null, "parent output\n".repeat(25), false),
            (json!("child-one"), "first child output".into(), false),
        ] {
            peer.event(json!({"event":"tool_output","tool_call":call("duplicate"),"output":output,"data":{"agent_instance_id":agent}}));
            peer.event(json!({"event":"tool_end","tool_call":call("duplicate"),"tool_result":{"tool_call_id":"duplicate","content":if agent.is_null() { "parent done" } else { &output },"is_error":error},"data":{"agent_instance_id":agent}}));
        }
        peer.wait_for_close();
    });
    harness.connected();
    let mut state = AppState::default();
    for _ in 0..3 {
        state.apply(harness.event());
    }
    for entry in &state.transcript {
        assert!(
            matches!(entry, TranscriptEntry::Tool { complete: false, card, .. } if !card.expanded && card.tail.text.is_empty())
        );
    }
    state.toggle_card(2);
    for _ in 0..6 {
        state.apply(harness.event());
    }
    for (index, expected, failed) in [
        (0, "parent output", false),
        (1, "first child output", false),
        (2, "failed child", true),
    ] {
        let TranscriptEntry::Tool {
            key,
            complete,
            error,
            card,
            summary,
            ..
        } = &state.transcript[index]
        else {
            panic!("missing tool");
        };
        assert_eq!(key.tool_call_id, "duplicate");
        assert!(*complete);
        assert_eq!(*error, failed);
        assert!(card.tail.text.contains(expected));
        assert_eq!(card.expanded, index == 2);
        assert_eq!(card.agent_label.is_some(), index > 0);
        if index == 0 {
            assert_eq!(summary, "parent done");
            assert!(card.tail.truncated);
            assert!(card.tail.text.ends_with("parent done"));
            assert_eq!(card.tail.text.lines().count(), zeta_gui::cards::TAIL_LINES);
        }
        if failed {
            assert_eq!(summary, "failed child");
            assert!(card.tail.truncated);
            assert_eq!(card.tail.text.lines().count(), zeta_gui::cards::TAIL_LINES);
        }
    }
    state.toggle_card(2);
    assert!(matches!(&state.transcript[2], TranscriptEntry::Tool { card, .. } if !card.expanded));
    harness.finish();
}

#[test]
fn status_metrics_bind_only_at_boundaries_and_ignore_streamed_usage() {
    let harness = Harness::new(|listener| {
        let mut peer = Peer::accept(&listener);
        peer.hello();
        peer.status(true, "idle", json!([]));
        peer.send();
        peer.event(json!({"event":"turn_start","data":{}}));
        peer.event(json!({"event":"usage","usage":{"input_tokens":999,"output_tokens":999}}));
        peer.event(json!({"event":"assistant_delta","kind":"assistant","delta":"hello"}));
        peer.event(json!({"event":"turn_end","data":{}}));
        // The next request must be status. No request is issued during deltas.
        peer.respond("status", json!({"session":session(),"state":"running","usage":{"input_tokens":20,"output_tokens":10,"cache_read_input_tokens":60,"cache_creation_input_tokens":20}}));
        peer.event(json!({"event":"agent_end","data":{}}));
        peer.status(true, "idle", json!([]));
        peer.wait_for_close();
    });
    assert!(matches!(harness.next(), WorkerMessage::Sessions(_)));
    let mut state = AppState::default();
    let WorkerMessage::Status(status) = harness.next() else {
        panic!("missing status");
    };
    state.apply_status(status);
    assert_eq!(state.metrics.model_label(), "offline");
    assert_eq!(state.metrics.tokens_label(), "—");
    assert_eq!(state.metrics.cache_label(), "—");
    assert!(matches!(harness.next(), WorkerMessage::Connected));
    harness.command(CommandMessage::Send("question".into()));
    assert!(matches!(harness.next(), WorkerMessage::Sent(_)));
    let before = state.metrics.clone();
    for _ in 0..3 {
        state.apply(harness.event());
        assert_eq!(state.metrics, before);
    }
    let WorkerMessage::Event(event) = harness.next() else {
        panic!("missing turn end");
    };
    assert!(matches!(event, ServerEvent::TurnEnd { .. }));
    state.apply(event);
    let WorkerMessage::Status(status) = harness.next() else {
        panic!("missing boundary status");
    };
    state.apply_status(status);
    assert_eq!(state.metrics.tokens_label(), "110");
    assert_eq!(state.metrics.cache_label(), "60.0%");
    state.apply(harness.event());
    harness.finish();
}

// Matches loop.py's next-turn drain and core/store.py's persisted notification.
fn durable_receipt(child: &str, status: &str, text: &str) -> Value {
    json!({"event":"sub_agent_receipt","data":{
        "notification_id":format!("notification-{child}"),
        "child_instance_id":child,
        "child_session_path":format!("/tmp/children/{child}.jsonl"),
        "description":format!("review {child}"),
        "status":status,
        "text":text,
        "stats":{"turns_used":2,"elapsed":1.2,"tool_calls":3,"error":status == "error","canceled":status == "canceled"}
    }})
}

#[test]
fn reconnect_next_turn_drain_creates_a_durable_agent_card_without_tool_events() {
    let (disconnect, wait) = mpsc::channel();
    let harness = Harness::new(move |listener| {
        let mut peer = Peer::accept(&listener);
        peer.hello();
        peer.status(true, "idle", json!([]));
        wait.recv_timeout(Duration::from_secs(5)).unwrap();
        drop(peer);
        let mut peer = Peer::accept(&listener);
        peer.hello();
        peer.respond("resume", json!({"session":session()}));
        peer.status(true, "idle", json!([]));
        peer.send();
        // Completion occurred while disconnected. No tool_start/tool_end replay.
        peer.event(durable_receipt(
            "child-one",
            "completed",
            &"review passed\n".repeat(30),
        ));
        peer.boundary(json!({"event":"agent_end","data":{}}));
        peer.wait_for_close();
    });
    harness.connected();
    let mut state = AppState::default();
    state.select_session(Some("session-1".into()));
    disconnect.send(()).unwrap();
    let WorkerMessage::Lost(reason) = harness.next() else {
        panic!("missing disconnect")
    };
    state.mark_connection_lost(reason);
    state.begin_reconnect();
    harness.command(CommandMessage::Reconnect);
    assert!(matches!(harness.next(), WorkerMessage::Sessions(_)));
    let WorkerMessage::Status(status) = harness.next() else {
        panic!("missing status")
    };
    state.apply_status(status);
    assert!(matches!(harness.next(), WorkerMessage::Connected));
    harness.command(CommandMessage::Send("next turn".into()));
    assert!(matches!(harness.next(), WorkerMessage::Sent(_)));
    let receipt = harness.event();
    assert!(matches!(receipt, ServerEvent::SubAgentReceipt { .. }));
    state.apply(receipt.clone());
    state.toggle_card(0);
    state.apply(receipt);
    assert_eq!(state.transcript.len(), 1);
    assert!(matches!(&state.transcript[0], TranscriptEntry::Tool {
        complete: true, error: false, summary, card, ..
    } if summary == "review passed"
        && card.agent_label.as_deref() == Some("review child-one")
        && card.child_instance_id.as_deref() == Some("child-one")
        && card.expanded && card.tail.truncated
        && card.tail.text.lines().count() == zeta_gui::cards::TAIL_LINES));
    state.apply(harness.event());
    harness.finish();
}

#[test]
fn durable_receipts_update_launch_cards_with_duplicate_raw_tool_ids() {
    let harness = Harness::new(|listener| {
        let mut peer = Peer::accept(&listener);
        peer.hello();
        peer.status(true, "idle", json!([]));
        peer.event(json!({"event":"tool_start","tool_call":call("duplicate"),"data":{}}));
        for (parent, child) in [("parent-one", "child-one"), ("parent-two", "child-two")] {
            let call = json!({"id":"duplicate","name":"agent","arguments":{"description":child}});
            peer.event(
                json!({"event":"tool_start","tool_call":call,"data":{"agent_instance_id":parent}}),
            );
            peer.event(json!({"event":"tool_end","tool_call":call,"tool_result":{
                "tool_call_id":"duplicate","content":"background agent started","is_error":false,
                "structured_content":{"status":"running","child_instance_id":child}
            },"data":{"agent_instance_id":parent}}));
        }
        for (child, status, text) in [
            ("child-two", "error", "review failed\nerror details"),
            ("child-one", "completed", "review passed\nsummary"),
            ("child-two", "error", "review failed\nerror details"),
        ] {
            peer.event(durable_receipt(child, status, text));
        }
        peer.event(durable_receipt(
            "child-three",
            "canceled",
            "background child canceled",
        ));
        peer.wait_for_close();
    });
    harness.connected();
    let mut state = AppState::default();
    for _ in 0..5 {
        state.apply(harness.event());
    }
    state.toggle_card(2);
    for _ in 0..4 {
        state.apply(harness.event());
    }
    assert_eq!(state.transcript.len(), 4);
    assert!(
        matches!(&state.transcript[0], TranscriptEntry::Tool { complete: false, card, .. } if card.child_instance_id.is_none())
    );
    for (index, child, summary, error) in [
        (1, "child-one", "review passed", false),
        (2, "child-two", "review failed", true),
        (3, "child-three", "background child canceled", true),
    ] {
        assert!(matches!(&state.transcript[index], TranscriptEntry::Tool {
            complete: true, error: actual_error, summary: actual_summary, card, ..
        } if *actual_error == error && actual_summary == summary
            && card.child_instance_id.as_deref() == Some(child)
            && card.expanded == (index == 2)));
    }
    assert!(
        matches!(&state.transcript[2], TranscriptEntry::Tool { card, .. } if card.tail.text == "review failed\nerror details")
    );
    assert_eq!(state.transcript[1].tool_marker(), "[done]");
    assert_eq!(state.transcript[2].tool_marker(), "[failed]");
    assert_eq!(state.transcript[3].tool_marker(), "[canceled]");
    assert!(state.transcript[3].unsuccessful());
    harness.finish();
}

#[test]
fn extensions_fetch_switch_fork_apply_settings_and_send_images() {
    use zeta_gui::session::{ImageAttachment, SessionSettings};
    let harness = Harness::new(|listener| {
        let mut peer = Peer::accept(&listener);
        let hello = peer.respond("hello", json!({"protocol_version":"1.1", "server":"zeta"}));
        assert_eq!(
            hello["params"],
            json!({"protocol_version":"1.0", "client_version":"1.1"})
        );
        peer.respond("list_sessions", json!({"sessions":[session()]}));
        peer.status(true, "idle", json!([]));
        let tree =
            json!({"branches":[{"id":"head","label":"first branch","depth":2,"current":true}]});
        let history = json!({"messages":[{"id":"user-1","role":"user","content":[{"type":"text","text":"history"}]}],"next_offset":null});
        assert_eq!(
            peer.respond("session_tree", tree.clone())["params"]["session_id"],
            "session-1"
        );
        peer.respond("session_history", history.clone());
        assert_eq!(
            peer.respond("switch_branch", tree.clone())["params"],
            json!({"session_id":"session-1","head_id":"other-head"})
        );
        peer.status(true, "idle", json!([]));
        peer.respond("session_tree", tree.clone());
        peer.respond("session_history", history.clone());
        assert_eq!(
            peer.respond("fork_message", tree.clone())["params"],
            json!({"session_id":"session-1","message_id":"user-1"})
        );
        peer.status(true, "idle", json!([]));
        peer.respond("session_tree", tree);
        peer.respond("session_history", history);
        peer.respond(
            "session_settings",
            json!({"model":"claude-sonnet-4-6","approval_mode":"ask"}),
        );
        peer.respond("model_catalog", json!({"models":["claude-sonnet-4-6","gpt-5.4"],"providers":{"claude-sonnet-4-6":"claude","gpt-5.4":"codex"}}));
        assert_eq!(
            peer.respond(
                "set_settings",
                json!({"model":"gpt-5.4","approval_mode":"deny"})
            )["params"],
            json!({"session_id":"session-1","model":"gpt-5.4","approval_mode":"deny"})
        );
        peer.status(true, "idle", json!([]));
        let send = peer.respond("send_images", json!({"accepted":true}));
        assert_eq!(send["params"]["session_id"], "session-1");
        assert_eq!(send["params"]["text"], "inspect");
        assert_eq!(
            send["params"]["images"][0],
            json!({"name":"shot.png","mime_type":"image/png","data":"iVBORw0KGgo="})
        );
        peer.wait_for_close();
    });
    assert!(matches!(harness.next(), WorkerMessage::Extensions(true)));
    assert!(matches!(harness.next(), WorkerMessage::Sessions(_)));
    assert!(matches!(harness.next(), WorkerMessage::Status(_)));
    assert!(matches!(harness.next(), WorkerMessage::Tree(tree) if tree.branches[0].depth == 2));
    assert!(
        matches!(harness.next(), WorkerMessage::History(history, true) if history[0].id == "user-1")
    );
    assert!(matches!(harness.next(), WorkerMessage::Connected));
    for command in [
        CommandMessage::SwitchBranch("other-head".into()),
        CommandMessage::ForkMessage("user-1".into()),
    ] {
        harness.command(command);
        assert!(matches!(harness.next(), WorkerMessage::Status(_)));
        assert!(matches!(harness.next(), WorkerMessage::Tree(_)));
        assert!(matches!(harness.next(), WorkerMessage::History(_, true)));
    }
    harness.command(CommandMessage::LoadSettings);
    assert!(
        matches!(harness.next(), WorkerMessage::Settings(settings, models) if settings.approval_mode == "ask" && models.models == ["claude-sonnet-4-6", "gpt-5.4"] && models.providers.get("gpt-5.4").map(String::as_str) == Some("codex"))
    );
    harness.command(CommandMessage::SetSettings(SessionSettings {
        model: "gpt-5.4".into(),
        approval_mode: "deny".into(),
    }));
    assert!(
        matches!(harness.next(), WorkerMessage::SettingsApplied(settings) if settings.model == "gpt-5.4" && settings.approval_mode == "deny")
    );
    assert!(matches!(harness.next(), WorkerMessage::Status(_)));
    harness.command(CommandMessage::SendImages(
        "inspect".into(),
        vec![ImageAttachment::from_bytes("shot.png".into(), b"\x89PNG\r\n\x1a\n").unwrap()],
    ));
    assert!(
        matches!(harness.next(), WorkerMessage::ImagesSent(text, images) if text == "inspect" && images[0].size == 8)
    );
    harness.finish();
}

#[test]
fn refresh_large_history_follows_variable_page_offsets_without_losing_connection() {
    use zeta_gui::client::MAX_FRAME_BYTES;
    let harness = Harness::new(|listener| {
        let mut peer = Peer::accept(&listener);
        peer.respond("hello", json!({"protocol_version":"1.1", "server":"zeta"}));
        peer.respond("list_sessions", json!({"sessions":[session()]}));
        peer.status(true, "idle", json!([]));
        peer.respond("session_tree", json!({"branches":[]}));
        // Short pages are not the end of history. The first message alone is
        // near the frame limit; tool receipts carry no persisted arguments.
        for (offset, count, next_offset, blocks) in
            [(0, 1, Some(1), 130), (1, 3, Some(4), 20), (4, 2, None, 20)]
        {
            let messages: Vec<Value> = (offset..offset + count)
                .map(|index| {
                    let mut content = vec![json!({"type":"text","text":"x".repeat(8000)}); blocks];
                    content.push(json!({"type":"tool_use", "tool_call":{
                        "id":format!("call-{index}"),"name":"write","arguments":{}
                    }}));
                    json!({"id":format!("message-{index}"),"role":"assistant","content":content})
                })
                .collect();
            let request = peer.request("session_history");
            assert_eq!(
                request["params"],
                json!({"session_id":"session-1","offset":offset})
            );
            let response = json!({"jsonrpc":"2.0","id":request["id"],"result":{
                "messages":messages,"next_offset":next_offset
            }});
            let size = response.to_string().len() + 1;
            assert!(size <= MAX_FRAME_BYTES);
            if offset == 0 {
                assert!(size > MAX_FRAME_BYTES - 10_000);
            }
            peer.write(response);
        }
        peer.send();
        peer.wait_for_close();
    });
    assert!(matches!(harness.next(), WorkerMessage::Extensions(true)));
    assert!(matches!(harness.next(), WorkerMessage::Sessions(_)));
    assert!(matches!(harness.next(), WorkerMessage::Status(_)));
    assert!(matches!(harness.next(), WorkerMessage::Tree(_)));
    match harness.next() {
        WorkerMessage::History(history, true) => {
            assert_eq!(history.len(), 6);
            for (index, message) in history.iter().enumerate() {
                assert_eq!(message.id, format!("message-{index}"));
            }
        }
        other => panic!("expected complete history, got {other:?}"),
    }
    assert!(matches!(harness.next(), WorkerMessage::Connected));
    harness.command(CommandMessage::Send("continue".into()));
    assert!(matches!(harness.next(), WorkerMessage::Sent(text) if text == "continue"));
    harness.finish();
}

#[test]
fn old_server_disables_extensions_without_sending_new_requests() {
    let harness = Harness::new(|listener| {
        let mut peer = Peer::accept(&listener);
        peer.hello();
        peer.status(true, "idle", json!([]));
        peer.send();
        peer.wait_for_close();
    });
    assert!(matches!(
        harness
            .messages
            .recv_timeout(Duration::from_secs(5))
            .unwrap(),
        WorkerMessage::Extensions(false)
    ));
    harness.connected();
    for command in [
        CommandMessage::SwitchBranch("head".into()),
        CommandMessage::ForkMessage("message".into()),
        CommandMessage::LoadSettings,
    ] {
        harness.command(command);
        assert!(
            matches!(harness.next(), WorkerMessage::Rejected(error) if error.contains("unavailable"))
        );
    }
    harness.command(CommandMessage::Send("old server still works".into()));
    assert!(matches!(harness.next(), WorkerMessage::Sent(_)));
    harness.finish();
}

fn branch_history_failure(command: CommandMessage, method: &'static str, rpc: bool) {
    let harness = Harness::new(move |listener| {
        let mut peer = Peer::accept(&listener);
        peer.respond("hello", json!({"protocol_version":"1.1", "server":"zeta"}));
        peer.respond("list_sessions", json!({"sessions":[session()]}));
        peer.status(true, "idle", json!([]));
        peer.respond("session_tree", json!({"branches":[]}));
        peer.respond("session_history", json!({"messages":[{"id":"old","role":"user","content":[{"type":"text","text":"old transcript"}]}],"next_offset":null}));
        peer.respond(
            method,
            json!({"branches":[{"id":"new","label":"new branch","depth":1,"current":true}]}),
        );
        // Permit a status/tree refresh before the history fetch. The branch
        // mutation has already succeeded, so old history is no longer current.
        loop {
            let mut line = String::new();
            peer.reader.read_line(&mut line).unwrap();
            let request: Value = serde_json::from_str(&line).unwrap();
            let result = match request["method"].as_str().unwrap() {
                "session_history" => {
                    if rpc {
                        peer.write(json!({"jsonrpc":"2.0","id":request["id"],"error":{"code":-32602,"message":"history is corrupt"}}));
                        peer.wait_for_close();
                    }
                    return;
                }
                "status" => json!({"session":session(),"state":"idle"}),
                "session_tree" => {
                    json!({"branches":[{"id":"new","label":"new branch","depth":1,"current":true}]})
                }
                other => panic!("unexpected request: {other}"),
            };
            peer.write(json!({"jsonrpc":"2.0","id":request["id"],"result":result}));
        }
    });
    let mut state = AppState::default();
    loop {
        match harness.next() {
            WorkerMessage::Status(status) => state.apply_status(status),
            WorkerMessage::History(history, replace) => state.apply_history(history, replace),
            WorkerMessage::Connected => break,
            WorkerMessage::Extensions(_) | WorkerMessage::Sessions(_) | WorkerMessage::Tree(_) => {}
            other => panic!("unexpected startup message: {other:?}"),
        }
    }
    assert_eq!(state.active_session.as_deref(), Some("session-1"));
    assert!(!state.transcript.is_empty());
    harness.command(command);
    let mut rejected = false;
    let mut lost = false;
    while let Ok(message) = harness.messages.recv_timeout(Duration::from_secs(5)) {
        match message {
            WorkerMessage::Status(status) => {
                state.apply_status(status);
                if state.active_session.is_none() {
                    break;
                }
            }
            WorkerMessage::Tree(tree) => state.session_view.branches = tree.branches,
            WorkerMessage::Rejected(error) => {
                assert!(error.contains("history is corrupt"));
                rejected = true;
            }
            WorkerMessage::Lost(error) => {
                state.mark_connection_lost(error);
                lost = true;
                break;
            }
            other => panic!("unexpected recovery message: {other:?}"),
        }
    }
    harness.finish();
    if rpc {
        assert!(rejected);
        assert!(!lost, "RPC failures must retain the connection");
        assert!(
            state.transcript.is_empty(),
            "old transcript remained current"
        );
        assert!(state.active_session.is_none());
        assert!(state.session_view.branches.is_empty());
    } else {
        assert!(lost, "transport failures must lose the connection");
        assert!(!rejected);
    }
}

#[test]
fn switch_history_rpc_failure_clears_old_transcript() {
    branch_history_failure(
        CommandMessage::SwitchBranch("new".into()),
        "switch_branch",
        true,
    );
}

#[test]
fn fork_history_rpc_failure_clears_old_transcript() {
    branch_history_failure(
        CommandMessage::ForkMessage("old".into()),
        "fork_message",
        true,
    );
}

#[test]
fn switch_history_transport_failure_loses_connection() {
    branch_history_failure(
        CommandMessage::SwitchBranch("new".into()),
        "switch_branch",
        false,
    );
}

#[test]
fn fork_history_transport_failure_loses_connection() {
    branch_history_failure(
        CommandMessage::ForkMessage("old".into()),
        "fork_message",
        false,
    );
}
