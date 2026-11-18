from __future__ import annotations

import importlib.metadata
import re
from typing import Any

MAX_BODY = 8 * 1024 * 1024
MAX_SPANS = 10_000
OPS = {"invoke_agent", "turn", "chat", "execute_tool", "context_compact", "serve", "mcp"}
SESSION = re.compile(r"[0-9a-f]{32}(?::[1-9][0-9]{0,8}){0,5}\Z")

class Reject(ValueError): pass

def _attrs(items: Any) -> dict[str, Any]:
    result={}
    for item in items:
        value=item.value
        kind=value.WhichOneof("value")
        result[item.key] = getattr(value, kind) if kind and kind != "kvlist_value" else None
    return result

def _op(name: str, attrs: dict[str, Any]) -> str:
    if name == "zeta.turn": return "turn"
    if name == "zeta.context.compact": return "context_compact"
    for prefix, op in (("invoke_agent ","invoke_agent"),("chat ","chat"),("execute_tool ","execute_tool"),("serve ","serve")):
        if name == prefix.rstrip() or name.startswith(prefix): return op
    if name in OPS: return name
    if attrs.get("gen_ai.operation.name") == "invoke_agent": return "invoke_agent"
    raise Reject("unknown operation")

def project(body: bytes) -> list[dict[str, Any]]:
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )
    request=ExportTraceServiceRequest()
    try: request.ParseFromString(body)
    except Exception as exc: raise Reject("malformed protobuf") from exc
    rows=[]
    version=importlib.metadata.version("zeta")
    for resource in request.resource_spans:
        ra=_attrs(resource.resource.attributes)
        if ra.get("service.name") != "zeta": raise Reject("invalid service")
        service_version=ra.get("service.version", version)
        if service_version != version: service_version="_invalid"
        for scope in resource.scope_spans:
            scope_version=scope.scope.version or version
            if scope.scope.name not in ("", "zeta"): raise Reject("invalid scope")
            if scope_version != version: scope_version="_invalid"
            for span in scope.spans:
                if len(span.trace_id)!=16 or not any(span.trace_id) or len(span.span_id)!=8 or not any(span.span_id): raise Reject("invalid id")
                if span.parent_span_id and (len(span.parent_span_id)!=8 or not any(span.parent_span_id)): raise Reject("invalid parent")
                if span.end_time_unix_nano < span.start_time_unix_nano: raise Reject("invalid time")
                attrs=_attrs(span.attributes); operation=_op(span.name,attrs)
                session=attrs.get("gen_ai.conversation.id")
                if session is not None and (not isinstance(session,str) or not SESSION.fullmatch(session)): session=None
                root=session.split(":",1)[0] if session else None
                status={0:"unset",1:"ok",2:"error"}.get(span.status.code,"unset")
                row={"trace_id":bytes(span.trace_id),"span_id":bytes(span.span_id),"parent_span_id":bytes(span.parent_span_id) if span.parent_span_id else None,"operation":operation,"kind":int(span.kind),"start_ns":span.start_time_unix_nano,"end_ns":span.end_time_unix_nano,"duration_ns":span.end_time_unix_nano-span.start_time_unix_nano,"status":status,"error_type":attrs.get("error.type") if status=="error" else None,"session_id":session,"root_session_id":root,"service_version":service_version,"scope_version":scope_version,"events":[]}
                mapping={"zeta.agent.name":"agent_name","zeta.agent.depth":"agent_depth","zeta.background":"background","zeta.turn.number":"turn_number","zeta.notification":"notification","gen_ai.provider.name":"provider","gen_ai.request.model":"request_model","gen_ai.response.model":"response_model","zeta.stream":"stream","gen_ai.usage.input_tokens":"input_tokens","gen_ai.usage.output_tokens":"output_tokens","gen_ai.usage.total_tokens":"total_tokens","gen_ai.response.finish_reasons":"finish_reason","zeta.tool.name":"tool_name","zeta.tool.type":"tool_type","zeta.tool.call.id":"tool_call_id","zeta.tool.call.arguments.size":"tool_argument_bytes","zeta.tool.call.result.size":"tool_result_bytes","zeta.tool.count":"tool_count","zeta.emergency":"emergency","zeta.tokens.before":"tokens_before","zeta.tokens.after":"tokens_after","rpc.method":"rpc_method","mcp.method.name":"mcp_method","zeta.mcp.server.name":"mcp_server","gen_ai.tool.name":"mcp_tool","gen_ai.prompt.name":"mcp_prompt","mcp.protocol.version":"mcp_protocol","jsonrpc.request.id":"request_id"}
                row.update({dest:attrs[key] for key,dest in mapping.items() if key in attrs and isinstance(attrs[key],(str,int,bool))})
                for index,event in enumerate(span.events):
                    if event.name not in {"gen_ai.client.retry","zeta.approval.requested","zeta.approval.resolved","zeta.serve.invalid_propagation"}: continue
                    ea=_attrs(event.attributes); row["events"].append({"event_index":index,"name":event.name,"time_ns":event.time_unix_nano,"retry_attempt":ea.get("gen_ai.retry.attempt"),"retry_reason":ea.get("gen_ai.retry.reason"),"retry_delay":ea.get("gen_ai.retry.delay"),"retry_after":ea.get("gen_ai.retry.after"),"decision":ea.get("decision"),"scope":ea.get("scope"),"duration_ns":ea.get("duration")})
                rows.append(row)
                if len(rows)>MAX_SPANS: raise Reject("too many spans")
    return rows
