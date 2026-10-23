#!/usr/bin/env python3
"""Read-only compatibility scan for step-delivery session records."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from zeta.agent.pending_delivery import notification_batch_id
from zeta.core.checkpoints import ConversationEntry, active_branch
from zeta.core.store._validation import validate_notification_data

SCHEMA = "zeta.step-delivery-scan.v1"


class ScanError(Exception):
    def __init__(self, code: str, *, fatal: bool = False) -> None:
        self.code = code
        self.fatal = fatal


def _diagnose(path: Path, line: int, code: str) -> None:
    print(f"{path}:{line}: {code}"[:1000], file=sys.stderr)


def _metadata(entry: ConversationEntry) -> dict[str, Any] | None:
    message = entry.data.get("message") if entry.type == "message" else None
    metadata = message.get("metadata", {}) if type(message) is dict else None
    return metadata if type(metadata) is dict else None


def _identity(metadata: dict[str, Any], payload: list[Any]) -> tuple[str, ...] | None:
    value = metadata.get("zeta.step_delivery")
    if value is None:
        return None
    if type(value) is not dict or set(value) != {"version", "batch_id", "notification_ids"}:
        raise ScanError("malformed_step_delivery_identity")
    if value.get("version") != 1:
        raise ScanError("unknown_step_delivery_version")
    ids = value.get("notification_ids")
    if type(ids) is not list or any(type(item) is not str or not item for item in ids) or len(ids) != len(set(ids)):
        raise ScanError("malformed_step_delivery_identity")
    represented = {item.get("notification_id") for item in payload if type(item) is dict}
    if any(item not in represented for item in ids):
        raise ScanError("dangling_batch_notification_id")
    if notification_batch_id(payload, ids) != value.get("batch_id"):
        raise ScanError("bad_batch_digest")
    return tuple(ids)


def _scan_file(path: Path, counts: Counter[str], anomalies: list[dict[str, Any]]) -> None:
    try:
        lines = path.read_bytes().splitlines()
    except OSError:
        _diagnose(path, 0, "unreadable_file")
        anomalies.append({"path": str(path), "line": 0, "code": "unreadable_file"})
        raise ScanError("unreadable_file", fatal=True) from None
    entries: list[ConversationEntry] = []
    line_by_id: dict[str, int] = {}
    for number, raw in enumerate(lines, 1):
        counts["rows_scanned"] += 1
        try:
            value = json.loads(raw)
            if type(value) is not dict:
                raise ValueError
            if value.get("type") == "header":
                continue
            entry = ConversationEntry.from_dict(value)
        except (ValueError, TypeError):
            _diagnose(path, number, "invalid_json")
            anomalies.append({"path": str(path), "line": number, "code": "invalid_json"})
            raise ScanError("invalid_json", fatal=True) from None
        entries.append(entry)
        line_by_id[entry.id] = number
    try:
        branch = active_branch(entries)
    except ValueError:
        _diagnose(path, 0, "invalid_branch")
        anomalies.append({"path": str(path), "line": 0, "code": "invalid_branch"})
        raise ScanError("invalid_branch", fatal=True) from None
    prompts = {entry.id: entry for entry in branch if entry.type == "pending_prompt"}
    prompt_acks: set[str] = set()
    sources = {entry.id: entry for entry in branch if entry.type == "notification"}
    notification_acks: set[str] = set()
    prompt_tags: Counter[str] = Counter()
    active_ids: set[str] = set()

    def anomaly(entry: ConversationEntry, code: str) -> None:
        line = line_by_id[entry.id]
        counts[code] += 1
        anomalies.append({"path": str(path), "line": line, "code": code})
        _diagnose(path, line, code)

    for entry in branch:
        if entry.type == "pending_prompt_ack":
            prompt_id = entry.data.get("prompt_id")
            if type(prompt_id) is not str or prompt_id not in prompts:
                anomaly(entry, "dangling_prompt_ack")
            elif prompt_id in prompt_acks:
                anomaly(entry, "duplicate_prompt_ack")
            else:
                prompt_acks.add(prompt_id)
        elif entry.type == "notification_ack":
            notification_id = entry.data.get("notification_id")
            if type(notification_id) is not str or notification_id not in sources:
                anomaly(entry, "dangling_notification_ack")
            elif notification_id in notification_acks:
                anomaly(entry, "duplicate_notification_ack")
            else:
                notification_acks.add(notification_id)
        elif entry.type == "notification":
            normalized_source = dict(entry.data)
            if "origin" not in normalized_source:
                normalized_source["origin"] = "notification"
                counts["notification_sources_legacy_origin"] += 1
            try:
                validate_notification_data(normalized_source)
            except ValueError:
                anomaly(entry, "malformed_notification")
        metadata = _metadata(entry)
        if metadata is None:
            continue
        prompt_id = metadata.get("zeta.pending_prompt_id")
        if prompt_id is not None:
            if type(prompt_id) is not str or not prompt_id:
                anomaly(entry, "malformed_prompt_tag")
            else:
                prompt_tags[prompt_id] += 1
        if metadata.get("zeta_event") == "agent_notifications":
            payload = metadata.get("notifications")
            if type(payload) is not list:
                anomaly(entry, "malformed_notification_metadata")
                continue
            for item in payload:
                if type(item) is not dict or type(item.get("notification_id")) is not str or not item["notification_id"]:
                    anomaly(entry, "malformed_notification_metadata")
                    break
                normalized = {key: value for key, value in item.items() if key != "notification_id"}
                if "origin" not in normalized:
                    normalized["origin"] = "notification"
                    counts["notification_payloads_legacy_origin"] += 1
                try:
                    validate_notification_data(normalized)
                except ValueError:
                    anomaly(entry, "malformed_notification_metadata")
                    break
            try:
                ids = _identity(metadata, payload)
            except ScanError as exc:
                anomaly(entry, exc.code)
                continue
            if ids is None:
                counts["notification_batches_legacy"] += 1
                continue
            counts["notification_batches_version_one"] += 1
            acknowledged = sum(item in notification_acks for item in ids)
            state = "fully_acknowledged" if acknowledged == len(ids) else "partially_acknowledged" if acknowledged else "unacknowledged"
            counts[f"notification_batches_{state}"] += 1
            if state != "fully_acknowledged":
                for item in ids:
                    if item in active_ids:
                        anomaly(entry, "overlapping_notification_id")
                    active_ids.add(item)
        if metadata.get("eviction_receipt") == "notification":
            fields = metadata.get("eviction_receipt_fields")
            receipt_payload = fields.get("payload") if type(fields) is dict else None
            if type(fields) is not dict or type(fields.get("source_seq")) is not int or type(metadata.get("eviction_content_digest")) is not str or type(receipt_payload) is not dict or type(receipt_payload.get("notifications")) is not list:
                anomaly(entry, "malformed_notification_receipt")
            else:
                counts["notification_receipts"] += 1
    for prompt_id, seen in prompt_tags.items():
        tagged = next(item for item in reversed(branch) if (_metadata(item) or {}).get("zeta.pending_prompt_id") == prompt_id)
        if prompt_id not in prompts:
            anomaly(tagged, "dangling_prompt_tag")
        elif seen > 1:
            anomaly(tagged, "duplicate_prompt_tag")
        else:
            counts["prompt_tagged"] += 1
    counts["pending_prompts"] += len(prompts)
    counts["pending_prompt_acks"] += len(prompt_acks)
    counts["notification_sources"] += len(sources)
    counts["notification_acks"] += len(notification_acks)


def scan(root: Path) -> tuple[dict[str, Any], int]:
    files = sorted(root.rglob("conversation.jsonl"))
    counts: Counter[str] = Counter()
    anomalies: list[dict[str, Any]] = []
    fatal = False
    for path in files:
        try:
            _scan_file(path, counts, anomalies)
        except ScanError as exc:
            fatal = fatal or exc.fatal
    rows_scanned = counts.pop("rows_scanned", 0)
    summary = {
        "schema": SCHEMA,
        "version": 1,
        "root": str(root),
        "files_scanned": len(files),
        "rows_scanned": rows_scanned,
        "child_session_files": sum("agents" in path.relative_to(root).parts for path in files),
        "counts": dict(sorted(counts.items())),
        "legacy": {"notification_batches": counts.get("notification_batches_legacy", 0), "untagged_prompts": counts.get("prompt_legacy_untagged", 0)},
        "anomalies": anomalies,
    }
    return summary, 2 if fatal else 1 if anomalies else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--format", choices=("json",), required=True)
    args = parser.parse_args()
    summary, status = scan(args.root)
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
