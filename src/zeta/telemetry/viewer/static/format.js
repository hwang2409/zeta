// Display formatting for nanosecond timestamps, durations, and counts.

const NS_PER_MS = 1e6;
const NS_PER_S = 1e9;

export function formatDuration(ns) {
  if (ns < 1e3) return `${ns} ns`;
  if (ns < NS_PER_MS) return `${Math.round(ns / 1e3)} µs`;
  if (ns < NS_PER_S) {
    const ms = ns / NS_PER_MS;
    return `${ms < 10 ? ms.toFixed(1) : Math.round(ms)} ms`;
  }
  const seconds = ns / NS_PER_S;
  if (seconds < 60) return `${seconds < 10 ? seconds.toFixed(2) : seconds.toFixed(1)} s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ${String(Math.floor(seconds % 60)).padStart(2, "0")}s`;
  return `${Math.floor(minutes / 60)}h ${String(minutes % 60).padStart(2, "0")}m`;
}

// Today's times show the clock only; older ones add the date.
export function formatTime(ns) {
  const date = new Date(ns / NS_PER_MS);
  const clock = date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
  if (date.toDateString() === new Date().toDateString()) return clock;
  return `${date.toLocaleDateString([], { month: "short", day: "numeric" })} ${clock}`;
}

export function formatFullTime(ns) {
  return new Date(ns / NS_PER_MS).toLocaleString([], { hour12: false });
}

export function formatCount(value) {
  return value.toLocaleString("en-US");
}

export function formatTokens(value) {
  if (value < 10_000) return formatCount(value);
  if (value < 1_000_000) return `${Math.round(value / 1_000)}K`;
  return `${(value / 1_000_000).toFixed(1)}M`;
}

export function shortId(id) {
  return id.slice(0, 8);
}
