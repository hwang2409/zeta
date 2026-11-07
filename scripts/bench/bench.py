#!/usr/bin/env python3
"""Small, repeatable performance probe for copied zeta sessions."""
from __future__ import annotations

import argparse
import json
import os
import pty
import shutil
import signal
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

from zeta.core.session import SessionManager
from zeta.storage_format import advance

ROOT = Path(__file__).resolve().parents[3]
DEFAULTS = {"phoebe": Path.home()/".zeta/sessions/7897c1aed2e5444caba907f462306b10", "orchestrator": Path.home()/".zeta/sessions/86e3d775a40b42ceaa76ca571e27ff1b"}
def clock(fn):
    start=time.perf_counter_ns(); value=fn(); return value,(time.perf_counter_ns()-start)/1e6
def load():
    try:return os.getloadavg()
    except OSError:return (float("nan"),)*3
def stats(values):
    values=sorted(values); return {"p50_ms":round(statistics.median(values),3),"p95_ms":round(values[max(0,int(len(values)*.95)-1)],3),"n":len(values)}
def rows_and_bytes(path):
    conversation=path/"conversation.jsonl"; return sum(1 for _ in conversation.open()),conversation.stat().st_size
def probe_store(home,sid,repeats=3):
    manager=SessionManager(home); opened,first=clock(lambda:manager.open(sid,_read_only=True)); store=opened.store; replay_times=[]; replay_rows=[]
    for _ in range(repeats):
        entries,elapsed=clock(store.replay); replay_times.append(elapsed); replay_rows.append(len(entries))
    messages,message_ms=clock(store.messages); store.close()
    return {"startup_resume":stats([first]),"replay":stats(replay_times),"rows":replay_rows[-1],"messages":len(messages),"message_ms":message_ms}
def probe_pty(home,sid):
    env=os.environ.copy(); env.update(ZETA_HOME=str(home),ZETA_ACCOUNT_HOME=str(home),TERM="xterm-256color")
    master,slave=pty.openpty(); started=time.perf_counter_ns(); proc=subprocess.Popen(["zeta","--resume",sid,"--no-auto-memory","--provider","codex"],stdin=slave,stdout=slave,stderr=slave,env=env,cwd=ROOT,start_new_session=True); os.close(slave)
    try:
        os.set_blocking(master,False); deadline=time.monotonic()+12; data=b""
        while time.monotonic()<deadline and b"zeta" not in data.lower():
            try:data+=os.read(master,65536)
            except BlockingIOError:time.sleep(.01)
        ready_ms=(time.perf_counter_ns()-started)/1e6; before=len(data); key_start=time.perf_counter_ns(); os.write(master,b"x"); echoed=False; deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            try:
                data+=os.read(master,65536)
                if len(data)>before:echoed=True;break
            except BlockingIOError:time.sleep(.005)
        return {"pty_start_ms":ready_ms,"pty_key_to_render_ms":(time.perf_counter_ns()-key_start)/1e6 if echoed else None,"bytes":len(data)}
    finally:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=3)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        os.close(master)
def clone(src,destination):
    destination.mkdir(parents=True,exist_ok=True); target=destination/src.name; subprocess.run(["cp","-cR",str(src),str(target)],check=True); return target
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--session",action="append",choices=[*DEFAULTS,"fresh"]); ap.add_argument("--keep",action="store_true"); args=ap.parse_args(); names=args.session or ["phoebe","orchestrator","fresh"]
    base=Path(tempfile.mkdtemp(prefix="zeta-perf-",dir=Path.home()/".zeta/scratch/perf")); home=base/"home"; (home/"sessions").mkdir(parents=True); advance(home, expected_epoch=0); results={"generated_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"load_average":load(),"home":str(base)}
    try:
        for name in names:
            if name=="fresh":
                manager=SessionManager(home); opened=manager.create(provider="codex",model="offline",cwd=ROOT,auto_project=False); sid=opened.metadata.session_id; opened.store.close()
            else:clone(DEFAULTS[name],home/"sessions"); sid=DEFAULTS[name].name
            rows,size=rows_and_bytes(home/"sessions"/sid); result={"scale":{"rows":rows,"conversation_bytes":size,"children":len([p for p in (home/"sessions"/sid/"agents").iterdir() if p.is_dir()]) if (home/"sessions"/sid/"agents").is_dir() else 0},"load_average":load()}; result.update(probe_store(home,sid)); result["load_average_after_store"]=load(); result["pty"]=probe_pty(home,sid) if name!="fresh" else None; result["load_average_after_pty"]=load(); results[name]=result
        print(json.dumps(results,indent=2))
    finally:
        if not args.keep:shutil.rmtree(base,ignore_errors=True)
if __name__=="__main__":main()
