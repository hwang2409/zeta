from __future__ import annotations

import asyncio
import fcntl
import os
import secrets
from pathlib import Path

from .receiver import MAX_BODY, Reject, project
from .store import ViewerStore

HOST="127.0.0.1"; PORT=4323

def _home_id(directory: Path) -> str:
    path=directory/"home-id"
    try: value=path.read_text().strip()
    except FileNotFoundError:
        value=secrets.token_hex(16); path.write_text(value); os.chmod(path,0o600)
    if len(value)!=32: raise RuntimeError("invalid viewer home id")
    return value

def create_app(home: Path):
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, PlainTextResponse, Response
    from starlette.routing import Route
    directory=home/"telemetry"; directory.mkdir(parents=True,exist_ok=True)
    identity=_home_id(directory); store=ViewerStore(directory/"telemetry.sqlite3")
    async def health(request): return JSONResponse({"home_id":identity,"schema_version":1,"otlp_url":"http://127.0.0.1:4323/v1/traces","url":"http://127.0.0.1:4323/","ready":True})
    async def ingest(request):
        if request.url.path != "/v1/traces" or request.headers.get("content-type","").split(";",1)[0] != "application/x-protobuf": return JSONResponse({"error":"unsupported request"},status_code=415)
        if request.headers.get("host") not in {"127.0.0.1:4323","localhost:4323"}: return JSONResponse({"error":"invalid host"},status_code=400)
        body=await request.body()
        if len(body)>MAX_BODY: return JSONResponse({"error":"body too large"},status_code=413)
        try: rows=project(body); await asyncio.to_thread(store.write,rows)
        except Reject as exc: return JSONResponse({"error":str(exc)},status_code=400)
        except (OSError, RuntimeError, ValueError): return JSONResponse({"error":"receiver unavailable"},status_code=503)
        return Response(status_code=200)
    async def sessions(request):
        try: result=await asyncio.to_thread(store.sessions,int(request.query_params.get("limit","50")),request.query_params.get("cursor"))
        except (ValueError,TypeError): return JSONResponse({"error":"invalid pagination"},status_code=400)
        return JSONResponse(result)
    async def trace(request):
        value=request.path_params["trace_id"]
        if len(value)!=32 or any(c not in "0123456789abcdef" for c in value): return JSONResponse({"error":"trace not found"},status_code=404)
        result=await asyncio.to_thread(store.trace,value)
        return JSONResponse(result or {"error":"trace not found"},status_code=200 if result else 404)
    async def page(request): return PlainTextResponse(PAGE,media_type="text/html")
    return Starlette(routes=[Route("/",page),Route("/health",health),Route("/v1/traces",ingest,methods=["POST"]),Route("/api/sessions",sessions),Route("/api/traces/{trace_id}",trace)])

def run_viewer(home: Path, no_open: bool=False) -> int:
    directory=home/"telemetry"; directory.mkdir(parents=True,exist_ok=True)
    lock_path=directory/"viewer.lock"; lock=lock_path.open("a+")
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print("already running at http://127.0.0.1:4323",file=__import__("sys").stderr); return 2
    try:
        import uvicorn
        uvicorn.run(create_app(home),host=HOST,port=PORT,log_level="error")
    except OSError as exc:
        print(f"zeta otel view: {exc}",file=__import__("sys").stderr); return 2
    return 0

PAGE='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>zeta telemetry</title><style>:root{color-scheme:dark;--bg:#282828;--fg:#ebdbb2;--muted:#a89984;--accent:#fabd2f;--panel:#3c3836}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px system-ui,sans-serif}main{max-width:1200px;margin:auto;padding:32px}h1{color:var(--accent)}a{color:var(--accent)}table{width:100%;border-collapse:collapse;background:var(--panel)}th,td{text-align:left;padding:12px;border-bottom:1px solid #504945}button{padding:8px 14px}#status{color:var(--muted)}@media(max-width:600px){main{padding:16px}table{font-size:12px;display:block;overflow:auto;white-space:nowrap}}</style></head><body><main><h1>zeta telemetry</h1><div id="status">Loading telemetry…</div><section id="content"></section></main><script type="module">const s=document.querySelector('#status'),c=document.querySelector('#content');const help='How to start telemetry: set [otel].endpoint = "http://127.0.0.1:4323/v1/traces", restart zeta, and run one turn.';async function load(){s.textContent='Loading telemetry…';c.replaceChildren();try{const r=await fetch('/api/sessions');if(!r.ok)throw Error();const d=await r.json();if(!d.sessions.length){s.textContent='No telemetry yet';c.textContent=help;return}s.textContent='';const t=document.createElement('table');t.innerHTML='<tr><th>session</th><th>range</th><th>spans</th><th>turns</th><th>models</th><th>tools</th><th>errors</th><th>profile</th><th>model</th><th></th></tr>';for(const x of d.sessions){const row=t.insertRow();row.innerHTML=`<td>${x.root_session_id}</td><td>${x.first_ns} – ${x.last_ns}</td><td>${x.span_count}</td><td>${x.turn_count}</td><td>${x.model_count}</td><td>${x.tool_count}</td><td>${x.error_count}</td><td>${x.profile??''}</td><td>${x.model??''}</td><td><a href="#trace/${x.last_trace_id}">trace</a></td>`}c.append(t)}catch(e){s.textContent='Telemetry could not be loaded.';const b=document.createElement('button');b.textContent='Retry';b.onclick=load;c.append(b)}}addEventListener('hashchange',async()=>{if(!location.hash){load();return}const id=location.hash.slice(7),r=await fetch('/api/traces/'+id),d=await r.json();s.textContent='';c.innerHTML='<h2>trace '+id+'</h2>';const t=document.createElement('table');t.innerHTML='<tr><th>operation</th><th>status</th><th>duration</th><th>span</th><th>parent</th></tr>';for(const x of d.spans){const row=t.insertRow();row.innerHTML=`<td>${x.operation}</td><td>${x.status}</td><td>${x.duration_ns}</td><td>${x.span_id}</td><td>${x.parent_span_id??''}</td>`}c.append(t)});load();</script></body></html>'''
