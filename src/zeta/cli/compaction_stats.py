"""Read-only compaction report over stored session logs.

The scanner reads ``meta.json`` and ``conversation.jsonl`` files directly. It
never opens a ``ConversationStore`` or ``SessionManager`` because those can
take append locks and repair torn tails. Each log is read only up to the size
seen at open, so concurrent appends do not change one scan. Memory per log is
bounded by the largest single row plus two integers per message row.
"""

from __future__ import annotations

import json
import os
import re
from array import array
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ..compaction import FALLBACK_SUMMARY_PREFIX, LEGACY_SESSION_COMPACTION
from ..context_eviction import (
    EVICTION_KIND,
    RECALL_NO_QUERY_MATCH,
    RECALL_NO_RANGE_MATCH,
)

DEFAULT_SINCE = "7d"
_MESSAGE_SUFFIX = b'"type":"message"}'
_SEQ_KEY = b'"seq":'
_PARSE_TRIGGERS = (b'"turn_error"', b'"recall_history"')
_DURATION = re.compile(r"^(\d+)([mhdw])$")
_DURATION_UNITS = {"m": "minutes", "h": "hours", "d": "days", "w": "weeks"}
# Durable turn errors carry only the exception text, so classification
# matches the texts raised by core.context.BudgetExceeded and the compaction
# and eviction paths.
_BUDGET_TEXTS = ("exceeds budget", "exceed the token budget", "exceeds the token budget")
_COMPACTION_TEXTS = (
    "during compaction",
    "during eviction",
    "summary source",
    "compaction summar",
    "compaction requires",
)
_CONTEXT_LENGTH_TEXTS = ("context_length_exceeded", "prompt is too long")


class ReportError(ValueError):
    """The report arguments do not select a valid scan."""


def parse_since(value: str, *, now: datetime) -> datetime | None:
    """Return the scan cutoff for ``7d``/``12h``/``30m``/``2w``, an ISO date, or ``all``."""

    if value == "all":
        return None
    match = _DURATION.match(value)
    if match is not None:
        amount, unit = match.groups()
        return now - timedelta(**{_DURATION_UNITS[unit]: int(amount)})
    try:
        cutoff = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ReportError(
            f"invalid --since {value!r}: use 7d, 12h, 30m, 2w, an ISO date, or all"
        ) from exc
    return cutoff if cutoff.tzinfo is not None else cutoff.replace(tzinfo=UTC)


@dataclass(slots=True)
class LogTally:
    """Counters for one conversation log (root session or child agent)."""

    rows: int = 0
    torn_final_lines: int = 0
    unparseable_lines: int = 0
    history_est_tokens: int = 0
    evictions: int = 0
    evictions_without_telemetry: int = 0
    items_evicted: int = 0
    evict_tokens_before: int = 0
    evict_tokens_after: int = 0
    summary_compactions: int = 0
    empty_summary_fallbacks: int = 0
    summary_est_tokens_before: int = 0
    summary_est_tokens_after: int = 0
    recall_calls: int = 0
    recall_with_content: int = 0
    recall_no_match: int = 0
    recall_errors: int = 0
    recall_unanswered: int = 0
    budget_exceeded: int = 0
    compaction_errors: int = 0
    context_length_errors: int = 0

    @property
    def markers(self) -> int:
        return self.evictions + self.summary_compactions

    @property
    def errors(self) -> int:
        return self.budget_exceeded + self.compaction_errors + self.context_length_errors

    def add(self, other: LogTally) -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))


@dataclass(slots=True)
class SessionTally:
    session_id: str
    mode: str
    budget: int | None
    updated_at: datetime
    root: LogTally
    children: list[LogTally] = field(default_factory=list)

    @property
    def total(self) -> LogTally:
        total = LogTally()
        for tally in (self.root, *self.children):
            total.add(tally)
        return total

    @property
    def compacted(self) -> bool:
        return any(tally.markers for tally in (self.root, *self.children))

    @property
    def peak_history_est_tokens(self) -> int:
        return max(tally.history_est_tokens for tally in (self.root, *self.children))


def _est_tokens(byte_count: int) -> int:
    # Context accounting charges ceil(len(compact message JSON) / 4); a raw row
    # is that JSON plus a small fixed wrapper, so this is a close estimate.
    return -(-byte_count // 4)


def _bounded_lines(path: Path) -> Iterator[tuple[bytes, bool]]:
    """Yield ``(line, complete)`` up to the file size seen at open."""

    with open(path, "rb") as handle:
        remaining = os.fstat(handle.fileno()).st_size
        while remaining > 0:
            line = handle.readline()
            if not line:
                return
            line = line[:remaining]
            remaining -= len(line)
            complete = line.endswith(b"\n")
            yield line.rstrip(b"\r\n"), complete


class _LogScanner:
    def __init__(self) -> None:
        self.tally = LogTally()
        self.seqs = array("q")
        self.cumulative = array("q")
        self.markers: dict[str, tuple[int, int, int]] = {}
        self.pending_recalls: set[str] = set()

    def scan(self, path: Path) -> LogTally:
        for line, complete in _bounded_lines(path):
            if not line.strip():
                continue
            self.tally.rows += 1
            if self._fast_message(line):
                continue
            try:
                row = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                row = None
            if not isinstance(row, dict):
                if complete:
                    self.tally.unparseable_lines += 1
                else:
                    self.tally.torn_final_lines += 1
                continue
            self._row(row, len(line))
        self.tally.recall_unanswered = len(self.pending_recalls)
        return self.tally

    def _fast_message(self, line: bytes) -> bool:
        if not line.endswith(_MESSAGE_SUFFIX) or any(
            trigger in line for trigger in _PARSE_TRIGGERS
        ):
            return False
        if self.pending_recalls and b'"tool_result"' in line:
            return False
        # Row keys are sorted, so the row's own seq is the last top-level key
        # before "type"; nested "seq" keys inside data come earlier.
        start = line.rfind(_SEQ_KEY)
        if start < 0:
            return False
        digits = line[start + len(_SEQ_KEY) : line.rfind(b",")]
        if not digits.isdigit():
            return False
        self._message_size(int(digits), len(line))
        return True

    def _message_size(self, seq: int, size: int) -> None:
        tokens = _est_tokens(size)
        self.tally.history_est_tokens += tokens
        if self.seqs and seq <= self.seqs[-1]:
            return
        self.seqs.append(seq)
        self.cumulative.append((self.cumulative[-1] if self.cumulative else 0) + tokens)

    def _range_tokens(self, start: int, end: int) -> int:
        low = bisect_left(self.seqs, start)
        high = bisect_right(self.seqs, end)
        if high <= low:
            return 0
        return self.cumulative[high - 1] - (self.cumulative[low - 1] if low else 0)

    def _row(self, row: dict[str, Any], size: int) -> None:
        data = row.get("data")
        if not isinstance(data, dict):
            return
        kind = row.get("type")
        if kind == "message":
            seq = row.get("seq")
            if type(seq) is int:
                self._message_size(seq, size)
            message = data.get("message")
            if isinstance(message, dict):
                self._message(message)
        elif kind == "compaction":
            self._marker(row.get("id"), data)

    def _message(self, message: dict[str, Any]) -> None:
        metadata = message.get("metadata")
        if isinstance(metadata, dict) and isinstance(metadata.get("turn_error"), dict):
            self._turn_error(metadata["turn_error"])
        content = message.get("content")
        for block in content if isinstance(content, list) else ():
            call = block.get("tool_call") if isinstance(block, dict) else None
            if isinstance(call, dict) and call.get("name") == "recall_history":
                self.tally.recall_calls += 1
                if isinstance(call.get("id"), str):
                    self.pending_recalls.add(call["id"])
        result = message.get("tool_result")
        if not isinstance(result, dict):
            return
        call_id = result.get("tool_call_id")
        if call_id not in self.pending_recalls:
            return
        self.pending_recalls.discard(call_id)
        text = result.get("content")
        if result.get("is_error") is True or not isinstance(text, str):
            self.tally.recall_errors += 1
        elif text.startswith((RECALL_NO_QUERY_MATCH, RECALL_NO_RANGE_MATCH)):
            self.tally.recall_no_match += 1
        else:
            self.tally.recall_with_content += 1

    def _turn_error(self, error: dict[str, Any]) -> None:
        code = str(error.get("code", ""))
        text = f"{code} {error.get('message', '')}".lower()
        if any(item in text for item in _BUDGET_TEXTS):
            self.tally.budget_exceeded += 1
        elif any(item in text for item in _COMPACTION_TEXTS):
            self.tally.compaction_errors += 1
        elif any(item in text for item in _CONTEXT_LENGTH_TEXTS):
            self.tally.context_length_errors += 1

    def _marker(self, marker_id: object, data: dict[str, Any]) -> None:
        start, end = data.get("source_seq_start"), data.get("source_seq_end")
        summary = data.get("summary")
        if type(start) is not int or type(end) is not int or not isinstance(summary, str):
            self.tally.unparseable_lines += 1
            return
        summary_tokens = _est_tokens(len(summary.encode()))
        if isinstance(marker_id, str):
            self.markers[marker_id] = (start, end, summary_tokens)
        if data.get("kind", LEGACY_SESSION_COMPACTION) == EVICTION_KIND:
            self.tally.evictions += 1
            telemetry = data.get("telemetry")
            values = (
                [telemetry.get(key) for key in ("items_evicted", "tokens_before", "tokens_after")]
                if isinstance(telemetry, dict)
                else []
            )
            if len(values) != 3 or any(type(value) is not int for value in values):
                self.tally.evictions_without_telemetry += 1
                return
            self.tally.items_evicted += values[0]
            self.tally.evict_tokens_before += values[1]
            self.tally.evict_tokens_after += values[2]
            return
        self.tally.summary_compactions += 1
        if summary.startswith(FALLBACK_SUMMARY_PREFIX):
            self.tally.empty_summary_fallbacks += 1
        replaced = [
            self.markers[item]
            for item in data.get("replaces", [])
            if isinstance(item, str) and item in self.markers
        ]
        before = self._range_tokens(start, end) + sum(item[2] for item in replaced)
        for low, high in _merged((item[0], item[1]) for item in replaced):
            before -= self._range_tokens(max(low, start), min(high, end))
        self.tally.summary_est_tokens_before += max(before, 0)
        self.tally.summary_est_tokens_after += summary_tokens


def _merged(ranges: Iterator[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for low, high in sorted(ranges):
        if merged and low <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], high))
        else:
            merged.append((low, high))
    return merged


def scan_log(path: Path) -> LogTally:
    """Scan one ``conversation.jsonl`` without writing or locking anything."""

    return _LogScanner().scan(path)


def _child_logs(session_dir: Path) -> Iterator[Path]:
    stack = [session_dir / "agents"]
    while stack:
        directory = stack.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError:
            continue
        for entry in entries:
            if not entry.is_dir(follow_symlinks=False):
                continue
            child = Path(entry.path)
            if (child / "conversation.jsonl").is_file():
                yield child / "conversation.jsonl"
            stack.append(child / "agents")


def _read_meta(session_dir: Path) -> dict[str, Any]:
    try:
        with open(session_dir / "meta.json", "rb") as handle:
            value = json.load(handle)
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _updated_at(meta: dict[str, Any], log: Path) -> datetime:
    value = meta.get("updated_at")
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
        except ValueError:
            pass
    return datetime.fromtimestamp(log.stat().st_mtime, UTC)


def _session_dirs(home: Path, session: str | None) -> list[Path]:
    root = home / "sessions"
    try:
        candidates = sorted(
            Path(entry.path)
            for entry in os.scandir(root)
            if entry.is_dir(follow_symlinks=False)
        )
    except FileNotFoundError:
        candidates = []
    if session is None:
        return candidates
    matches = [path for path in candidates if path.name.startswith(session)]
    if len(matches) != 1:
        problem = "no session matches" if not matches else "ambiguous session prefix"
        raise ReportError(f"{problem}: {session}")
    return matches


def collect_sessions(
    home: Path,
    *,
    since: datetime | None,
    session: str | None = None,
) -> tuple[list[SessionTally], int]:
    """Return tallies for selected sessions and the count of unreadable logs.

    An explicit ``session`` prefix ignores the ``since`` window.
    """

    tallies: list[SessionTally] = []
    unreadable = 0
    for session_dir in _session_dirs(home, session):
        log = session_dir / "conversation.jsonl"
        if not log.is_file():
            continue
        meta = _read_meta(session_dir)
        try:
            updated_at = _updated_at(meta, log)
        except OSError:
            unreadable += 1
            continue
        if session is None and since is not None and updated_at < since:
            continue
        try:
            root = scan_log(log)
        except OSError:
            unreadable += 1
            continue
        children = []
        for child in _child_logs(session_dir):
            try:
                children.append(scan_log(child))
            except OSError:
                unreadable += 1
        mode = meta.get("compaction", LEGACY_SESSION_COMPACTION) if meta else "unknown"
        budget = meta.get("compaction_budget")
        tallies.append(
            SessionTally(
                session_id=session_dir.name,
                mode=mode if isinstance(mode, str) else "unknown",
                budget=budget if type(budget) is int else None,
                updated_at=updated_at,
                root=root,
                children=children,
            )
        )
    return tallies, unreadable


def _saved_pct(before: int, after: int) -> float | None:
    return round(100 * (before - after) / before, 1) if before else None


def _mode_metrics(mode: str, sessions: list[SessionTally]) -> dict[str, Any]:
    total = LogTally()
    for session in sessions:
        total.add(session.total)
    child_logs = [child for session in sessions for child in session.children]
    return {
        "sessions": len(sessions),
        "sessions_compacted": sum(session.compacted for session in sessions),
        "child_logs": len(child_logs),
        "child_logs_compacted": sum(bool(child.markers) for child in child_logs),
        "evictions": total.evictions,
        "evictions_without_telemetry": total.evictions_without_telemetry,
        "items_evicted": total.items_evicted,
        "evict_tokens_before": total.evict_tokens_before,
        "evict_tokens_after": total.evict_tokens_after,
        "evict_saved_pct": _saved_pct(total.evict_tokens_before, total.evict_tokens_after),
        "summary_compactions": total.summary_compactions,
        "summary_fallbacks_from_evict": (
            total.summary_compactions if mode == EVICTION_KIND else None
        ),
        "empty_summary_fallbacks": total.empty_summary_fallbacks,
        "summary_est_tokens_before": total.summary_est_tokens_before,
        "summary_est_tokens_after": total.summary_est_tokens_after,
        "summary_est_saved_pct": _saved_pct(
            total.summary_est_tokens_before, total.summary_est_tokens_after
        ),
        "recall_calls": total.recall_calls,
        "recall_with_content": total.recall_with_content,
        "recall_no_match": total.recall_no_match,
        "recall_errors": total.recall_errors,
        "recall_unanswered": total.recall_unanswered,
        "budget_exceeded": total.budget_exceeded,
        "compaction_errors": total.compaction_errors,
        "context_length_errors": total.context_length_errors,
    }


def build_report(
    sessions: list[SessionTally],
    *,
    home: Path,
    since: datetime | None,
    unreadable: int = 0,
    top: int = 10,
) -> dict[str, Any]:
    """Aggregate session tallies into the JSON-ready report."""

    by_mode: dict[str, list[SessionTally]] = {}
    for session in sessions:
        by_mode.setdefault(session.mode, []).append(session)
    budgets: dict[tuple[str, int | None], list[SessionTally]] = {}
    for session in sessions:
        budgets.setdefault((session.mode, session.budget), []).append(session)
    ranked = sorted(
        sessions,
        key=lambda item: (
            -(item.total.markers + item.total.recall_calls + item.total.errors),
            -item.updated_at.timestamp(),
        ),
    )
    integrity = LogTally()
    for session in sessions:
        integrity.add(session.total)
    return {
        "home": str(home),
        "since": None if since is None else since.isoformat(),
        "sessions_scanned": len(sessions),
        "logs_scanned": sum(1 + len(session.children) for session in sessions),
        "modes": {
            mode: _mode_metrics(mode, items)
            for mode, items in sorted(by_mode.items(), key=lambda pair: _mode_order(pair[0]))
        },
        "budgets": [
            {
                "mode": mode,
                "budget": budget,
                "sessions": len(items),
                "sessions_compacted": sum(item.compacted for item in items),
                "max_est_history_tokens": max(item.peak_history_est_tokens for item in items),
                "sessions_history_over_budget": sum(
                    budget is not None and item.peak_history_est_tokens > budget
                    for item in items
                ),
            }
            for (mode, budget), items in sorted(
                budgets.items(),
                key=lambda pair: (_mode_order(pair[0][0]), -(pair[0][1] or 0)),
            )
        ],
        "top_sessions": [
            {
                "session_id": item.session_id,
                "mode": item.mode,
                "budget": item.budget,
                "updated_at": item.updated_at.isoformat(),
                "child_logs": len(item.children),
                "evictions": item.total.evictions,
                "summary_compactions": item.total.summary_compactions,
                "empty_summary_fallbacks": item.total.empty_summary_fallbacks,
                "recall_calls": item.total.recall_calls,
                "errors": item.total.errors,
                "max_est_history_tokens": item.peak_history_est_tokens,
            }
            for item in ranked[:top]
            if item.total.markers + item.total.recall_calls + item.total.errors
        ],
        "integrity": {
            "torn_final_lines": integrity.torn_final_lines,
            "unparseable_lines": integrity.unparseable_lines,
            "unreadable_logs": unreadable,
        },
    }


def _mode_order(mode: str) -> tuple[int, str]:
    order = {EVICTION_KIND: 0, LEGACY_SESSION_COMPACTION: 1}
    return order.get(mode, 2), mode


def _count(value: int | None) -> str:
    if value is None:
        return "-"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 10_000:
        return f"{value / 1_000:.0f}k"
    return str(value)


def _budget(value: int | None) -> str:
    return "-" if value is None else f"{value:,}"


def _tokens(before: int, after: int, saved: float | None, approx: str = "") -> str:
    if not before:
        return "-"
    return f"{approx}{_count(before)} -> {approx}{_count(after)} ({saved:.1f}% saved)"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    widths = [max(len(row[index]) for row in (header, *rows)) for index in range(len(header))]
    lines = []
    for row in (header, *rows):
        cells = [row[0].ljust(widths[0])]
        cells.extend(cell.rjust(width) for cell, width in zip(row[1:], widths[1:], strict=True))
        lines.append("  ".join(cells).rstrip())
    return lines


def render_report(report: dict[str, Any]) -> str:
    """Render the compact human table for one report."""

    window = "all time" if report["since"] is None else f"updated since {report['since'][:16]}"
    lines = [
        (
            f"compaction report: {report['sessions_scanned']} sessions, "
            f"{report['logs_scanned']} logs, {window} ({report['home']})"
        )
    ]
    modes = report["modes"]
    if not modes:
        lines.append("no sessions in range")
        return "\n".join(lines) + "\n"
    names = list(modes)

    def row(label: str, render: Callable[[dict[str, Any]], str]) -> list[str]:
        return [label, *(render(modes[name]) for name in names)]

    lines.append("")
    lines.extend(
        _table(
            ["", *names],
            [
                row("sessions (compacted)", lambda m: f"{m['sessions']} ({m['sessions_compacted']})"),
                row("child logs (compacted)", lambda m: f"{m['child_logs']} ({m['child_logs_compacted']})"),
                row("evictions", lambda m: str(m["evictions"])),
                row("  items evicted", lambda m: str(m["items_evicted"])),
                row(
                    "  tokens before -> after",
                    lambda m: _tokens(m["evict_tokens_before"], m["evict_tokens_after"], m["evict_saved_pct"]),
                ),
                row("summary compactions", lambda m: str(m["summary_compactions"])),
                row(
                    "  from evict fallback",
                    lambda m: _count(m["summary_fallbacks_from_evict"]),
                ),
                row("  empty-summary fallbacks", lambda m: str(m["empty_summary_fallbacks"])),
                row(
                    "  est tokens before -> after",
                    lambda m: _tokens(
                        m["summary_est_tokens_before"],
                        m["summary_est_tokens_after"],
                        m["summary_est_saved_pct"],
                        "~",
                    ),
                ),
                row(
                    "recall_history (hit/miss/err)",
                    lambda m: f"{m['recall_calls']} ({m['recall_with_content']}/"
                    f"{m['recall_no_match']}/{m['recall_errors']})",
                ),
                row("BudgetExceeded", lambda m: str(m["budget_exceeded"])),
                row("compaction errors", lambda m: str(m["compaction_errors"])),
                row("context-length errors", lambda m: str(m["context_length_errors"])),
            ],
        )
    )
    lines.extend(["", "effective budgets"])
    lines.extend(
        _table(
            ["mode", "budget", "sessions", "compacted", "max est history", "history > budget"],
            [
                [
                    item["mode"],
                    _budget(item["budget"]),
                    str(item["sessions"]),
                    str(item["sessions_compacted"]),
                    _count(item["max_est_history_tokens"]),
                    str(item["sessions_history_over_budget"]),
                ]
                for item in report["budgets"]
            ],
        )
    )
    lines.extend(["", "top sessions by compaction activity"])
    if report["top_sessions"]:
        lines.extend(
            _table(
                ["session", "mode", "budget", "children", "evict", "summary", "fallback", "recall", "errors"],
                [
                    [
                        item["session_id"][:8],
                        item["mode"],
                        _budget(item["budget"]),
                        str(item["child_logs"]),
                        str(item["evictions"]),
                        str(item["summary_compactions"]),
                        str(item["empty_summary_fallbacks"]),
                        str(item["recall_calls"]),
                        str(item["errors"]),
                    ]
                    for item in report["top_sessions"]
                ],
            )
        )
    else:
        lines.append("(no compaction activity)")
    integrity = report["integrity"]
    if any(integrity.values()):
        lines.extend(
            [
                "",
                (
                    f"skipped: {integrity['torn_final_lines']} torn final lines, "
                    f"{integrity['unparseable_lines']} unparseable lines, "
                    f"{integrity['unreadable_logs']} unreadable logs"
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def compaction_report(
    home: Path,
    *,
    since: str = DEFAULT_SINCE,
    session: str | None = None,
    top: int = 10,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Scan ``home/sessions`` read-only and return the aggregated report."""

    if top < 1:
        raise ReportError("--top must be at least 1")
    cutoff = None if session is not None else parse_since(since, now=now or datetime.now(UTC))
    sessions, unreadable = collect_sessions(home, since=cutoff, session=session)
    return build_report(sessions, home=home, since=cutoff, unreadable=unreadable, top=top)
