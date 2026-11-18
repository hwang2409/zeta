// zeta otel viewer: session list and one trace's span waterfall.
// Every received value is rendered as text through DOM APIs; no HTML is parsed from data.

import { formatCount, formatDuration, formatFullTime, formatTime, formatTokens, shortId } from "./format.js";

const view = document.getElementById("view");
const TRACE_ROUTE = /^#\/trace\/([0-9a-f]{32})$/;
const HELP = 'How to start telemetry: set [otel].endpoint = "http://127.0.0.1:4323/v1/traces", restart zeta, and run one turn.';
let generation = 0;

function el(tag, attributes = {}, ...children) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) {
    if (value !== null && value !== undefined && value !== false) node.setAttribute(name, value === true ? "" : String(value));
  }
  for (const child of children) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : String(child));
  }
  return node;
}

function show(...nodes) {
  view.replaceChildren(...nodes);
}

function showLoading() {
  show(el("p", { class: "state", role: "status" }, "Loading telemetry…"));
}

function showError(retry) {
  const button = el("button", { type: "button" }, "Retry");
  button.addEventListener("click", retry);
  show(el("div", { class: "state error", role: "alert" }, el("p", {}, "Telemetry could not be loaded."), button));
  button.focus();
}

async function fetchJson(url) {
  const response = await fetch(url, { headers: { Accept: "application/json" } });
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  return response.json();
}

// ---- session list ----

function statusMark(errors) {
  return errors > 0 ? el("span", { class: "num bad" }, formatCount(errors)) : el("span", { class: "num dim" }, "0");
}

function sessionRow(session) {
  const link = el("a", { href: `#/trace/${session.last_trace_id}`, class: "trace-link" }, "Latest trace");
  link.setAttribute("aria-label", `Latest trace of session ${session.root_session_id}`);
  const cells = [
    ["Last activity", el("time", { class: "mono dim", title: formatFullTime(session.last_ns) }, formatTime(session.last_ns))],
    ["Session", el("div", { class: "who" },
      el("span", { class: "profile" }, session.profile ?? "unknown profile"),
      el("span", { class: "mono id" }, session.root_session_id))],
    ["Model", el("span", { class: "mono" }, session.model ?? "–")],
    ["Started", el("time", { class: "mono dim", title: formatFullTime(session.first_ns) }, formatTime(session.first_ns))],
    ["Duration", el("span", { class: "num" }, formatDuration(session.last_ns - session.first_ns))],
    ["Turns", el("span", { class: "num" }, formatCount(session.turn_count))],
    ["Model calls", el("span", { class: "num" }, formatCount(session.model_count))],
    ["Tool calls", el("span", { class: "num" }, formatCount(session.tool_count))],
    ["Spans", el("span", { class: "num" }, formatCount(session.span_count))],
    ["Errors", statusMark(session.error_count)],
    ["Input", el("span", { class: "num", title: formatCount(session.input_tokens) }, formatTokens(session.input_tokens))],
    ["Output", el("span", { class: "num", title: formatCount(session.output_tokens) }, formatTokens(session.output_tokens))],
    ["Trace", link],
  ];
  return el("tr", {}, ...cells.map(([label, content]) => el("td", { "data-label": label }, content)));
}

function sessionTable() {
  const labels = ["Last activity", "Session", "Model", "Started", "Duration", "Turns", "Model calls", "Tool calls", "Spans", "Errors", "Input", "Output", "Trace"];
  const numeric = new Set(["Duration", "Turns", "Model calls", "Tool calls", "Spans", "Errors", "Input", "Output"]);
  const head = el("tr", {}, ...labels.map((label) => el("th", { scope: "col", class: numeric.has(label) ? "num" : null }, label)));
  return el("table", { class: "sessions" }, el("caption", { class: "visually-hidden" }, "Sessions, newest activity first"), el("thead", {}, head), el("tbody"));
}

async function renderSessions() {
  const current = ++generation;
  showLoading();
  let page;
  try {
    page = await fetchJson("/api/sessions?limit=50");
  } catch {
    if (current === generation) showError(renderSessions);
    return;
  }
  if (current !== generation) return;
  if (page.sessions.length === 0) {
    show(el("div", { class: "state empty" }, el("h1", {}, "No telemetry yet"), el("p", {}, HELP)));
    return;
  }
  const table = sessionTable();
  const body = table.tBodies[0];
  const count = el("span", { class: "count" });
  const more = el("button", { type: "button", class: "more" }, "Load more");
  let shown = 0;
  const append = (result) => {
    for (const session of result.sessions) body.append(sessionRow(session));
    shown += result.sessions.length;
    count.textContent = `${formatCount(shown)} ${shown === 1 ? "session" : "sessions"}${result.next_cursor ? " shown" : ""}`;
    more.hidden = !result.next_cursor;
    more.dataset.cursor = result.next_cursor ?? "";
  };
  more.addEventListener("click", async () => {
    more.disabled = true;
    try {
      append(await fetchJson(`/api/sessions?limit=50&cursor=${encodeURIComponent(more.dataset.cursor)}`));
    } catch {
      showError(renderSessions);
    } finally {
      more.disabled = false;
    }
  });
  append(page);
  show(el("div", { class: "heading" }, el("h1", {}, "Sessions"), count), el("div", { class: "table-wrap" }, table), more);
}

// ---- trace waterfall ----

function spanLabel(span) {
  const detail = {
    invoke_agent: span.agent_name,
    chat: span.response_model ?? span.request_model,
    execute_tool: span.tool_name,
  }[span.operation];
  return detail ? `${span.operation} ${detail}` : span.operation;
}

// Parents before children; siblings by start time. Spans whose parent is absent start a new root.
function treeOrder(spans) {
  const ids = new Set(spans.map((span) => span.span_id));
  const children = new Map();
  const roots = [];
  for (const span of spans) {
    if (span.parent_span_id && ids.has(span.parent_span_id)) {
      if (!children.has(span.parent_span_id)) children.set(span.parent_span_id, []);
      children.get(span.parent_span_id).push(span);
    } else {
      roots.push(span);
    }
  }
  const ordered = [];
  const stack = roots.reverse().map((span) => [span, 0]);
  while (stack.length) {
    const [span, depth] = stack.pop();
    ordered.push([span, depth]);
    for (const child of (children.get(span.span_id) ?? []).slice().reverse()) stack.push([child, depth + 1]);
  }
  return ordered;
}

function statusBadge(status) {
  const mark = { ok: "✓", error: "×", unset: "·" }[status];
  return el("span", { class: `status ${status}` }, el("span", { "aria-hidden": "true" }, mark), ` ${status}`);
}

function dimensions(span, rootSession) {
  const parts = [`span ${span.span_id}`, span.parent_span_id ? `parent ${span.parent_span_id}` : "root span"];
  if (span.provider) parts.push(`provider ${span.provider}`);
  if (span.request_model && span.response_model && span.request_model !== span.response_model) parts.push(`requested ${span.request_model}`);
  if (span.tool_type) parts.push(`type ${span.tool_type}`);
  if (span.input_tokens !== null) parts.push(`in ${formatCount(span.input_tokens)}`);
  if (span.output_tokens !== null) parts.push(`out ${formatCount(span.output_tokens)}`);
  if (span.total_tokens !== null) parts.push(`total ${formatCount(span.total_tokens)} tok`);
  if (span.error_type) parts.push(`error ${span.error_type}`);
  if (span.session_id && span.session_id !== rootSession) parts.push(`session ${span.session_id}`);
  return parts.join(" · ");
}

function eventText(event, span) {
  const parts = [event.name, `+${formatDuration(Math.max(0, event.time_ns - span.start_ns))}`];
  if (event.retry_attempt !== null) parts.push(`attempt ${event.retry_attempt}`);
  if (event.retry_reason) parts.push(event.retry_reason);
  if (event.retry_delay !== null) parts.push(`delay ${formatCount(event.retry_delay)} ms`);
  if (event.retry_after) parts.push("retry-after");
  if (event.decision) parts.push(event.decision);
  if (event.scope) parts.push(event.scope);
  if (event.duration_ns !== null) parts.push(`waited ${formatDuration(event.duration_ns)}`);
  return parts.join(" · ");
}

function spanRow(span, depth, trace) {
  const total = Math.max(1, trace.last_ns - trace.first_ns);
  const left = ((span.start_ns - trace.first_ns) / total) * 100;
  const width = Math.max(0.4, (span.duration_ns / total) * 100);
  const bar = el("div", { class: `bar op-${span.operation}${span.status === "error" ? " failed" : ""}`, "aria-hidden": "true" });
  bar.style.left = `${Math.min(left, 99.6)}%`;
  bar.style.width = `${Math.min(width, 100 - Math.min(left, 99.6))}%`;
  const name = el("div", { class: "span-name" },
    el("span", { class: `dot op-${span.operation}`, "aria-hidden": "true" }),
    el("span", { class: "mono" }, spanLabel(span)));
  const details = el("div", { class: "span-meta mono" }, dimensions(span, trace.root_session_id));
  const cell = el("td", { class: "span-cell", "data-label": "Span" }, name, details);
  cell.style.setProperty("--depth", String(Math.min(depth, 12)));
  if (span.events.length) {
    cell.append(el("ul", { class: "events mono" }, ...span.events.map((event) => el("li", {}, eventText(event, span)))));
  }
  return el("tr", { class: span.status === "error" ? "is-error" : null },
    cell,
    el("td", { class: "timeline", "data-label": "Timeline" }, el("div", { class: "track" }, bar)),
    el("td", { class: "num", "data-label": "Duration" }, formatDuration(span.duration_ns)),
    el("td", { "data-label": "Status" }, statusBadge(span.status)));
}

async function renderTrace(traceId) {
  const current = ++generation;
  showLoading();
  let result;
  try {
    result = await fetchJson(`/api/traces/${traceId}`);
  } catch {
    if (current === generation) showError(() => renderTrace(traceId));
    return;
  }
  if (current !== generation) return;
  const { trace, spans } = result;
  const crumbs = el("nav", { class: "crumbs", "aria-label": "Breadcrumb" },
    el("a", { href: "#/" }, "Sessions"), el("span", { "aria-hidden": "true" }, " › "),
    el("span", { class: "mono" }, trace.root_session_id ?? "no session"));
  const stats = el("p", { class: "stats mono" },
    `${formatCount(trace.span_count)} spans · `,
    el("span", { class: trace.error_count ? "bad" : null }, `${formatCount(trace.error_count)} errors`),
    ` · ${formatDuration(trace.last_ns - trace.first_ns)} · started `,
    el("time", { title: formatFullTime(trace.first_ns) }, formatTime(trace.first_ns)));
  const heading = el("div", { class: "heading" },
    el("h1", {}, "Trace ", el("span", { class: "mono id" }, shortId(trace.trace_id))),
    statusBadge(trace.status));
  const head = el("tr", {},
    el("th", { scope: "col" }, "Span"),
    el("th", { scope: "col", class: "timeline" }, el("span", {}, "0"), el("span", {}, formatDuration(trace.last_ns - trace.first_ns))),
    el("th", { scope: "col", class: "num" }, "Duration"),
    el("th", { scope: "col" }, "Status"));
  const table = el("table", { class: "waterfall" },
    el("caption", { class: "visually-hidden" }, `Spans of trace ${trace.trace_id}, parents before children`),
    el("thead", {}, head),
    el("tbody", {}, ...treeOrder(spans).map(([span, depth]) => spanRow(span, depth, trace))));
  const legend = el("ul", { class: "legend", "aria-label": "Legend" },
    ...[["turn", "Turn"], ["invoke_agent", "Agent"], ["chat", "Model call"], ["execute_tool", "Tool"], ["context_compact", "Compaction"], ["mcp", "MCP / serve"]]
      .map(([operation, label]) => el("li", {}, el("span", { class: `dot op-${operation}`, "aria-hidden": "true" }), label)));
  show(crumbs, heading, stats, el("div", { class: "table-wrap" }, table), legend,
    el("p", { class: "dim mono trace-id" }, `trace ${trace.trace_id}`));
}

function route() {
  const match = TRACE_ROUTE.exec(location.hash);
  if (match) renderTrace(match[1]);
  else renderSessions();
  view.focus({ preventScroll: true });
}

addEventListener("hashchange", route);
route();
