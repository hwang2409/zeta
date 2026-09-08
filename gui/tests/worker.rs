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
        peer.event(json!({"event":"agent_end","data":{}}));
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
        peer.wait_for_close();

        // The cleared selection must not trigger another resume on reconnect.
        let mut peer = Peer::accept(&listener);
        peer.hello();
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
    harness.command(CommandMessage::Reconnect);
    harness.connected();
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
        peer.event(json!({"event":"agent_end","data":{}}));
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
        peer.event(json!({"event":"agent_end","data":{}}));
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
        peer.event(json!({"event":"agent_end","data":{},"session_id":"session-2"}));
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
        vec![TranscriptEntry::Assistant("new text".into())]
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
        peer.event(json!({"event":"agent_end","data":{}}));
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
        peer.event(json!({"event":"agent_end","data":{}}));
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
