from __future__ import annotations

import base64
import json
import sqlite3
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE viewer_meta (singleton INTEGER PRIMARY KEY CHECK (singleton = 1), schema_version INTEGER NOT NULL CHECK (schema_version >= 1), committed_sequence INTEGER NOT NULL DEFAULT 0 CHECK (committed_sequence >= 0)) STRICT;
CREATE TABLE spans (trace_id BLOB NOT NULL CHECK (length(trace_id) = 16), span_id BLOB NOT NULL CHECK (length(span_id) = 8), parent_span_id BLOB CHECK (parent_span_id IS NULL OR length(parent_span_id) = 8), operation TEXT NOT NULL CHECK (operation IN ('invoke_agent','turn','chat','execute_tool','context_compact','serve','mcp')), kind INTEGER NOT NULL CHECK (kind BETWEEN 0 AND 5), start_ns INTEGER NOT NULL CHECK (start_ns >= 0), end_ns INTEGER NOT NULL CHECK (end_ns >= start_ns), duration_ns INTEGER NOT NULL CHECK (duration_ns = end_ns - start_ns), status TEXT NOT NULL CHECK (status IN ('unset','ok','error')),  error_type TEXT, session_id TEXT, root_session_id TEXT, service_version TEXT NOT NULL, scope_version TEXT NOT NULL, agent_name TEXT, agent_depth INTEGER, background INTEGER, turn_number INTEGER, notification INTEGER, provider TEXT, request_model TEXT, response_model TEXT, stream INTEGER, input_tokens INTEGER, output_tokens INTEGER, total_tokens INTEGER, finish_reason TEXT, tool_name TEXT, tool_type TEXT, tool_call_id TEXT, tool_argument_bytes INTEGER, tool_result_bytes INTEGER, tool_count INTEGER, emergency INTEGER, tokens_before INTEGER, tokens_after INTEGER, rpc_method TEXT, mcp_method TEXT, mcp_server TEXT, mcp_tool TEXT, mcp_prompt TEXT, mcp_protocol TEXT, request_id TEXT, PRIMARY KEY (trace_id, span_id)) STRICT;
CREATE TABLE span_events (trace_id BLOB NOT NULL, span_id BLOB NOT NULL, event_index INTEGER NOT NULL, name TEXT NOT NULL, time_ns INTEGER NOT NULL, retry_attempt INTEGER, retry_reason TEXT, retry_delay INTEGER, retry_after INTEGER, decision TEXT, scope TEXT, duration_ns INTEGER, PRIMARY KEY (trace_id, span_id, event_index), FOREIGN KEY (trace_id, span_id) REFERENCES spans(trace_id, span_id) ON DELETE CASCADE) STRICT;
CREATE TABLE traces (trace_id BLOB PRIMARY KEY, first_ns INTEGER NOT NULL, last_ns INTEGER NOT NULL, root_span_id BLOB, root_session_id TEXT, status TEXT NOT NULL, span_count INTEGER NOT NULL DEFAULT 0, error_count INTEGER NOT NULL DEFAULT 0) STRICT;
CREATE TABLE sessions (root_session_id TEXT PRIMARY KEY, first_ns INTEGER NOT NULL, last_ns INTEGER NOT NULL, last_trace_id BLOB NOT NULL, span_count INTEGER NOT NULL DEFAULT 0, turn_count INTEGER NOT NULL DEFAULT 0, model_count INTEGER NOT NULL DEFAULT 0, tool_count INTEGER NOT NULL DEFAULT 0, error_count INTEGER NOT NULL DEFAULT 0, input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0, total_tokens INTEGER NOT NULL DEFAULT 0, profile TEXT, model TEXT, last_span_id BLOB NOT NULL, profile_last_ns INTEGER, profile_trace_id BLOB, profile_span_id BLOB, model_last_ns INTEGER, model_trace_id BLOB, model_span_id BLOB) STRICT;
CREATE INDEX spans_session_start ON spans(root_session_id, start_ns); CREATE INDEX spans_trace_start ON spans(trace_id, start_ns); CREATE INDEX sessions_last ON sessions(last_ns DESC); CREATE INDEX traces_last ON traces(last_ns DESC);
"""

SPAN_COLUMNS = "trace_id,span_id,parent_span_id,operation,kind,start_ns,end_ns,duration_ns,status,error_type,session_id,root_session_id,service_version,scope_version,agent_name,agent_depth,background,turn_number,notification,provider,request_model,response_model,stream,input_tokens,output_tokens,total_tokens,finish_reason,tool_name,tool_type,tool_call_id,tool_argument_bytes,tool_result_bytes,tool_count,emergency,tokens_before,tokens_after,rpc_method,mcp_method,mcp_server,mcp_tool,mcp_prompt,mcp_protocol,request_id"

def _hex(value: bytes | None) -> str | None:
    return value.hex() if value is not None else None

def _cursor(value: str) -> tuple[int, str]:
    try:
        raw = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
        item = json.loads(raw)
        if not isinstance(item, list) or len(item) != 2 or type(item[0]) is not int or not isinstance(item[1], str): raise ValueError
        return item[0], item[1]
    except (ValueError, TypeError, json.JSONDecodeError, UnicodeError):
        raise ValueError("invalid pagination") from None

class ViewerStore:
    def __init__(self, path: Path):
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        self.conn.execute("INSERT OR IGNORE INTO viewer_meta(singleton,schema_version) VALUES (1,1)")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def write(self, spans: list[dict[str, Any]]) -> int:
        inserted = 0
        with self.conn:
            for span in spans:
                values = [span.get(c) for c in SPAN_COLUMNS.split(",")]
                self.conn.execute(f"INSERT INTO spans ({SPAN_COLUMNS}) VALUES ({','.join('?' for _ in values)}) ON CONFLICT(trace_id,span_id) DO NOTHING", values)
                if self.conn.execute("SELECT changes()").fetchone()[0] != 1: continue
                inserted += 1
                for event in span.get("events", []):
                    self.conn.execute("INSERT INTO span_events(trace_id,span_id,event_index,name,time_ns,retry_attempt,retry_reason,retry_delay,retry_after,decision,scope,duration_ns) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [span["trace_id"], span["span_id"], event.get("event_index", 0), event["name"], event["time_ns"], event.get("retry_attempt"), event.get("retry_reason"), event.get("retry_delay"), event.get("retry_after"), event.get("decision"), event.get("scope"), event.get("duration_ns")])
                self._project(span)
            if inserted:
                self.conn.execute("UPDATE viewer_meta SET committed_sequence=committed_sequence+1 WHERE singleton=1")
        return inserted

    def _project(self, s: dict[str, Any]) -> None:
        tid, sid, root = s["trace_id"], s["span_id"], s.get("root_session_id")
        self.conn.execute("INSERT INTO traces(trace_id,first_ns,last_ns,root_span_id,root_session_id,status,span_count,error_count) VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(trace_id) DO UPDATE SET first_ns=MIN(traces.first_ns,excluded.first_ns),last_ns=MAX(traces.last_ns,excluded.last_ns),root_span_id=COALESCE(traces.root_span_id,excluded.root_span_id),root_session_id=COALESCE(traces.root_session_id,excluded.root_session_id),status=CASE WHEN traces.status='error' OR excluded.status='error' THEN 'error' WHEN traces.status='ok' OR excluded.status='ok' THEN 'ok' ELSE 'unset' END,span_count=traces.span_count+1,error_count=traces.error_count+excluded.error_count", (tid,s["start_ns"],s["end_ns"],sid if s.get("parent_span_id") is None else None,root,s["status"],1,int(s["status"] == "error")))
        if root is None: return
        operation = s["operation"]
        profile = s.get("agent_name") if operation == "invoke_agent" else None
        model = (s.get("response_model") or s.get("request_model")) if operation == "chat" else None
        self.conn.execute("""INSERT INTO sessions(root_session_id,first_ns,last_ns,last_trace_id,span_count,turn_count,model_count,tool_count,error_count,input_tokens,output_tokens,total_tokens,profile,model,last_span_id,profile_last_ns,profile_trace_id,profile_span_id,model_last_ns,model_trace_id,model_span_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(root_session_id) DO UPDATE SET first_ns=MIN(sessions.first_ns,excluded.first_ns),last_ns=MAX(sessions.last_ns,excluded.last_ns),last_trace_id=CASE WHEN (excluded.last_ns,excluded.last_trace_id,excluded.last_span_id)>(sessions.last_ns,sessions.last_trace_id,sessions.last_span_id) THEN excluded.last_trace_id ELSE sessions.last_trace_id END,last_span_id=CASE WHEN (excluded.last_ns,excluded.last_trace_id,excluded.last_span_id)>(sessions.last_ns,sessions.last_trace_id,sessions.last_span_id) THEN excluded.last_span_id ELSE sessions.last_span_id END,span_count=sessions.span_count+1,turn_count=sessions.turn_count+excluded.turn_count,model_count=sessions.model_count+excluded.model_count,tool_count=sessions.tool_count+excluded.tool_count,error_count=sessions.error_count+excluded.error_count,input_tokens=sessions.input_tokens+excluded.input_tokens,output_tokens=sessions.output_tokens+excluded.output_tokens,total_tokens=sessions.total_tokens+excluded.total_tokens,profile=CASE WHEN excluded.profile IS NOT NULL AND (sessions.profile IS NULL OR (excluded.profile_last_ns,excluded.profile_trace_id,excluded.profile_span_id)>(sessions.profile_last_ns,sessions.profile_trace_id,sessions.profile_span_id)) THEN excluded.profile ELSE sessions.profile END,profile_last_ns=CASE WHEN excluded.profile IS NOT NULL AND (sessions.profile IS NULL OR (excluded.profile_last_ns,excluded.profile_trace_id,excluded.profile_span_id)>(sessions.profile_last_ns,sessions.profile_trace_id,sessions.profile_span_id)) THEN excluded.profile_last_ns ELSE sessions.profile_last_ns END,profile_trace_id=CASE WHEN excluded.profile IS NOT NULL AND (sessions.profile IS NULL OR (excluded.profile_last_ns,excluded.profile_trace_id,excluded.profile_span_id)>(sessions.profile_last_ns,sessions.profile_trace_id,sessions.profile_span_id)) THEN excluded.profile_trace_id ELSE sessions.profile_trace_id END,profile_span_id=CASE WHEN excluded.profile IS NOT NULL AND (sessions.profile IS NULL OR (excluded.profile_last_ns,excluded.profile_trace_id,excluded.profile_span_id)>(sessions.profile_last_ns,sessions.profile_trace_id,sessions.profile_span_id)) THEN excluded.profile_span_id ELSE sessions.profile_span_id END,model=CASE WHEN excluded.model IS NOT NULL AND (sessions.model IS NULL OR (excluded.model_last_ns,excluded.model_trace_id,excluded.model_span_id)>(sessions.model_last_ns,sessions.model_trace_id,sessions.model_span_id)) THEN excluded.model ELSE sessions.model END,model_last_ns=CASE WHEN excluded.model IS NOT NULL AND (sessions.model IS NULL OR (excluded.model_last_ns,excluded.model_trace_id,excluded.model_span_id)>(sessions.model_last_ns,sessions.model_trace_id,sessions.model_span_id)) THEN excluded.model_last_ns ELSE sessions.model_last_ns END,model_trace_id=CASE WHEN excluded.model IS NOT NULL AND (sessions.model IS NULL OR (excluded.model_last_ns,excluded.model_trace_id,excluded.model_span_id)>(sessions.model_last_ns,sessions.model_trace_id,sessions.model_span_id)) THEN excluded.model_trace_id ELSE sessions.model_trace_id END,model_span_id=CASE WHEN excluded.model IS NOT NULL AND (sessions.model IS NULL OR (excluded.model_last_ns,excluded.model_trace_id,excluded.model_span_id)>(sessions.model_last_ns,sessions.model_trace_id,sessions.model_span_id)) THEN excluded.model_span_id ELSE sessions.model_span_id END""", (root,s["start_ns"],s["end_ns"],tid,1,int(operation=="turn"),int(operation=="chat"),int(operation=="execute_tool"),int(s["status"]=="error"),s.get("input_tokens") or 0,s.get("output_tokens") or 0,s.get("total_tokens") or 0,profile,model,sid,s["end_ns"] if profile else None,tid if profile else None,sid if profile else None,s["end_ns"] if model else None,tid if model else None,sid if model else None))

    def sessions(self, limit: int = 50, cursor: str | None = None) -> dict[str, Any]:
        if not 1 <= limit <= 100: raise ValueError("invalid pagination")
        args: list[Any] = []
        where = ""
        if cursor:
            last, root = _cursor(cursor); where = "WHERE (last_ns < ? OR (last_ns = ? AND root_session_id > ?))"; args += [last,last,root]
        rows = self.conn.execute(f"SELECT root_session_id,first_ns,last_ns,lower(hex(last_trace_id)) last_trace_id,span_count,turn_count,model_count,tool_count,error_count,input_tokens,output_tokens,total_tokens,profile,model FROM sessions {where} ORDER BY last_ns DESC,root_session_id ASC LIMIT ?", (*args,limit+1)).fetchall()
        next_cursor = None
        if len(rows) > limit:
            row = rows.pop(); raw = json.dumps([row["last_ns"],row["root_session_id"]],separators=(",",":")).encode(); next_cursor = base64.urlsafe_b64encode(raw).decode().rstrip("=")
        return {"sessions":[dict(r) for r in rows],"next_cursor":next_cursor}

    def trace(self, trace_id: str) -> dict[str, Any] | None:
        raw = bytes.fromhex(trace_id)
        trace = self.conn.execute("SELECT lower(hex(trace_id)) trace_id,first_ns,last_ns,lower(hex(root_span_id)) root_span_id,root_session_id,status,span_count,error_count FROM traces WHERE trace_id=?",(raw,)).fetchone()
        if trace is None: return None
        spans=[]
        for s in self.conn.execute("SELECT * FROM spans WHERE trace_id=? ORDER BY start_ns,span_id",(raw,)):
            item={k:s[k] for k in ("span_id","parent_span_id","operation","kind","start_ns","end_ns","duration_ns","status","error_type","session_id","agent_name","provider","request_model","response_model","tool_name","tool_type","input_tokens","output_tokens","total_tokens")}
            item["span_id"]=_hex(item["span_id"]); item["parent_span_id"]=_hex(item["parent_span_id"])
            item["events"]=[dict(e) for e in self.conn.execute("SELECT event_index,name,time_ns,retry_attempt,retry_reason,retry_delay,retry_after,decision,scope,duration_ns FROM span_events WHERE trace_id=? AND span_id=? ORDER BY time_ns,event_index",(raw,s["span_id"]))]
            spans.append(item)
        return {"trace":dict(trace),"spans":spans}
