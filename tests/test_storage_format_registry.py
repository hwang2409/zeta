from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import pytest

from zeta.accounts.store import _read_publication_pending
from zeta.attention_forks import read_attention_fork
from zeta.attention_records import AttentionRecord
from zeta.context_eviction._fact_floor import validate_fact_floor
from zeta.context_eviction.receipt_constructors import (
    _valid_persisted_reasoning_receipt_metadata,
)
from zeta.core.checkpoints import ConversationEntry
from zeta.core.hooks import _parse_hooks
from zeta.core.store import ConversationStore
from zeta.core.store._approval_display import (
    validated_approval_display,
    validated_approval_facts,
)
from zeta.core.store._child_runs import validate_child_run_started_data
from zeta.core.store._client_delivery import validate_client_delivery_data
from zeta.core.store._turn_history import validate_turn_end_data
from zeta.core.store._validation import (
    validate_child_idle_notification_data,
    validate_terminal_transition_data,
)
from zeta.core.store.prompt_composition import parse_prompt_recipe
from zeta.core.todo import parse_todo_items
from zeta.core.transcript_recovery import (
    parse_evidence_manifest,
    parse_recovery_record,
)
from zeta.memory import entry_store
from zeta.memory.entry_history import EntryMemoryHistoryMixin
from zeta.memory.entry_store import _exact_dict
from zeta.memory.entry_sync import parse_resolution_candidates
from zeta.project_errors import ProjectRegistryError
from zeta.project_memory_history import ProjectMemoryHistoryMixin
from zeta.project_registry import ProjectRegistry
from zeta.project_schema import validate_project_record
from zeta.protocol.types import (
    _validate_annotations,
    _validate_image_block,
    _validate_resource_block,
    _validate_text_block,
)
from zeta.remote_sync.memory import (
    _valid_entry_sync_state,
    _valid_entry_transition,
    _valid_state,
)
from zeta.remote_sync.project_publish import (
    ProjectPublicationError,
    _manifest_payloads,
    _validate_format_two_payload,
)
from zeta.storage_format import (
    ACCEPTED_KEY_SETS,
    _read_manifest,
    compare_registry_dumps,
)
from zeta.tools._shared.background_output_archive import _BackgroundOutputArchive

Validator = Callable[[dict[str, object]], object]

# Registered before their writers exist so older releases can detect the future
# authoritative shapes. Each named slice adds the production validator and case.
_DORMANT_UNTIL_SLICE = {
    "retention_state": 2,
    "retention_preview": 2,
    "retention_reservation": 3,
    "retention_manifest": 3,
    "retention_arm": 4,
}


@dataclass(frozen=True)
class ShapeCase:
    name: str
    validator: Validator
    valid: dict[str, object]
    required: frozenset[str]


def _assert_valid(result: object) -> None:
    assert result is not False


def _assert_invalid(
    validator: Validator, payload: dict[str, object], context: str
) -> None:
    try:
        result = validator(payload)
    except (KeyError, OSError, ProjectPublicationError, TypeError, ValueError):
        return
    assert result is False, f"{context}: validator accepted an invalid persisted shape"


def test_every_registry_name_has_one_production_owner() -> None:
    owners: dict[str, tuple[str, int]] = {}
    for path in Path("src/zeta").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            name_node: ast.expr | None = None
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "reject_unknown_keys"
            ):
                assert len(node.args) >= 2, (
                    f"{path}:{node.lineno} must use a literal stored-shape name"
                )
                name_node = node.args[1]
            elif (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id == "ACCEPTED_KEY_SETS"
            ):
                name_node = node.slice
            if name_node is None:
                continue
            assert isinstance(name_node, ast.Constant) and isinstance(
                name_node.value, str
            ), f"{path}:{node.lineno} must use a literal stored-shape name"
            owners[name_node.value] = (str(path), node.lineno)
    assert set(_DORMANT_UNTIL_SLICE) <= set(ACCEPTED_KEY_SETS)
    assert all(slice_number in {2, 3, 4} for slice_number in _DORMANT_UNTIL_SLICE.values())
    expected_owners = set(ACCEPTED_KEY_SETS) - set(_DORMANT_UNTIL_SLICE)
    assert set(owners) == expected_owners, {
        "missing_owners": sorted(expected_owners - set(owners)),
        "unregistered_owners": sorted(set(owners) - expected_owners),
    }


def test_strict_production_validators_are_bound_to_registry(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "session")

    def store_entry(entry_type: str) -> Validator:
        def validate(payload: dict[str, object]) -> object:
            entry = ConversationEntry(1, "e", None, "main", entry_type, payload)
            store._validate_entry_payload(entry)
            return True

        return validate

    def hook_root(payload: dict[str, object]) -> object:
        return _parse_hooks(payload, tmp_path / "hooks.toml")

    def hook_record(payload: dict[str, object]) -> object:
        return _parse_hooks({"hook": [payload]}, tmp_path / "hooks.toml")

    def todo(payload: dict[str, object]) -> object:
        return parse_todo_items([payload])

    def prompt_component(payload: dict[str, object]) -> object:
        return parse_prompt_recipe(
            {"prompt_recipe": "default", "prompt_components": {"x": payload}},
            system_prompt="",
            has_context_snapshot=True,
            path=tmp_path / "meta.json",
        )

    def client(payload: dict[str, object]) -> object:
        validate_client_delivery_data(payload, "session")
        return True

    def nested_client(outer: dict[str, object], key: str) -> Validator:
        def validate(payload: dict[str, object]) -> object:
            value = dict(outer)
            value[key] = payload
            return client(value)

        return validate

    def exact(label: str, fields: set[str]) -> Validator:
        return lambda payload: _exact_dict(payload, fields, label)

    legacy_entry_fields = frozenset(
        {
            "status",
            "supersedes",
            "accepted_by",
            "project_id",
            "valid_from",
            "created_at",
            "superseded_by",
            "representation",
            "id",
            "kind",
            "text",
            "section",
            "sources",
            "migration_order",
            "accepted_at",
            "seen_at",
            "valid_until",
            "last_operation_id",
            "expires_at",
            "migration_source",
            "automatic",
            "updated_at",
        }
    )

    def persisted_legacy_entry(payload: dict[str, object]) -> object:
        callers: list[str] = []
        reject_unknown_keys = entry_store.reject_unknown_keys

        def track_reject_unknown_keys(value: object, name: str) -> None:
            if name == "memory_entry":
                callers.append(inspect.currentframe().f_back.f_code.co_name)
            reject_unknown_keys(value, name)

        state = {
            "format": 2,
            "project_id": "p_" + "a" * 32,
            "generation": 0,
            "schema": {
                "version": 1,
                "profile": "zeta",
                "kinds": [
                    {
                        "key": "state",
                        "name": "State",
                        "description": "State",
                        "prompt_mode": "always",
                        "prompt_priority": 1,
                        "prompt_max_entries": 1,
                        "default_expiry_days": None,
                    }
                ],
            },
            "entries": {"m_" + "a" * 32: payload},
            "compacted_through_version": None,
        }
        try:
            with mock.patch.object(
                entry_store, "reject_unknown_keys", track_reject_unknown_keys
            ):
                return entry_store.state_from_bytes(json.dumps(state).encode())
        except Exception as exc:
            if "_registry_probe_extra" in payload:
                assert callers == ["state_from_bytes"], (
                    "legacy memory entry extra key bypassed the direct registry check"
                )
            missing_fields = legacy_entry_fields - payload.keys()
            if missing_fields:
                assert isinstance(exc, ProjectRegistryError) and str(exc) == (
                    "invalid memory entry"
                ), f"legacy memory entry missing {sorted(missing_fields)} bypassed the shape check"
            raise

    def fact_root(payload: dict[str, object]) -> object:
        return validate_fact_floor(payload)

    def fact_aggregate(payload: dict[str, object]) -> object:
        return validate_fact_floor({"version": 1, "oldest": [payload], "newest": []})

    def fact_record(payload: dict[str, object]) -> object:
        aggregate = {
            "fact": payload,
            "first_seq": 1,
            "last_seq": 1,
            "value_rank": [1, 0],
        }
        return validate_fact_floor({"version": 1, "oldest": [aggregate], "newest": []})

    manifest_home = tmp_path / "manifest"
    manifest_home.mkdir()

    def manifest(payload: dict[str, object]) -> object:
        (manifest_home / "storage-format.json").write_text(json.dumps(payload))
        return _read_manifest(manifest_home)

    def session_reference(payload: dict[str, object]) -> object:
        return ProjectRegistry._decode_session_link(payload)

    def remote_conflict(payload: dict[str, object]) -> object:
        state = {
            "schema": 2,
            "project_id": "p_x",
            "files": {},
            "conflicts": {"brief.md": payload},
        }
        return _valid_state(state, "p_x")

    def entry_conflict(payload: dict[str, object]) -> object:
        state = {
            "schema": 3,
            "project_id": "p_x",
            "format": 2,
            "entries": {},
            "schema_digest": "missing",
            "conflicts": {
                "schema" if payload.get("kind") == "schema" else "m_a": payload
            },
        }
        return _valid_entry_sync_state(state, "p_x")

    def candidate(payload: dict[str, object]) -> object:
        key = "schema" if payload.get("kind") == "schema" else "m_a"
        return parse_resolution_candidates({key: payload}) is not None

    account_dir = tmp_path / "account"
    account_dir.mkdir()
    account_fd = os.open(account_dir, os.O_RDONLY | os.O_DIRECTORY)

    def publication(payload: dict[str, object]) -> object:
        (account_dir / "publish-pending.json").write_text(json.dumps(payload))
        return bool(_read_publication_pending(account_fd))

    def attention_fork(payload: dict[str, object]) -> object:
        session_dir = tmp_path / "fork"
        session_dir.mkdir(exist_ok=True)
        (session_dir / "attention_fork.json").write_text(json.dumps(payload))
        return read_attention_fork(session_dir)

    def snapshot_reference(payload: dict[str, object]) -> object:
        try:
            EntryMemoryHistoryMixin._entry_blob(-1, payload)
        except ValueError as exc:
            if "incomplete" in str(exc):
                return True
            raise
        return True

    publication_root = tmp_path / "publication"
    publication_blobs = publication_root / "memory-versions" / "blobs"
    publication_blobs.mkdir(parents=True)
    legacy_payload = b""
    legacy_digest = hashlib.sha256(legacy_payload).hexdigest()
    (publication_blobs / legacy_digest).write_bytes(legacy_payload)
    state = {
        "format": 2,
        "project_id": "p_" + "a" * 32,
        "generation": 0,
        "schema": {},
        "entries": {},
        "compacted_through_version": None,
    }
    state_payload = json.dumps(state).encode()
    state_digest = hashlib.sha256(state_payload).hexdigest()
    (publication_blobs / state_digest).write_bytes(state_payload)

    def published_snapshot_reference(payload: dict[str, object]) -> object:
        manifest = {
            "format": 2,
            "snapshot": {key: state_digest for key in payload},
        }
        return _manifest_payloads(publication_root, manifest, "snapshot")

    def published_legacy_snapshot(payload: dict[str, object]) -> object:
        manifest = {
            "format": 1,
            "snapshot": {key: legacy_digest for key in payload},
        }
        return _manifest_payloads(publication_root, manifest, "snapshot")

    def history_legacy_snapshot(payload: dict[str, object]) -> object:
        directory_fd = os.open(publication_blobs, os.O_RDONLY | os.O_DIRECTORY)
        try:
            manifest = {"snapshot": {key: legacy_digest for key in payload}}
            return ProjectMemoryHistoryMixin._contents_from_manifest(
                None, directory_fd, manifest
            )
        finally:
            os.close(directory_fd)

    def published_state_root(payload: dict[str, object]) -> object:
        _validate_format_two_payload(json.dumps(payload).encode())
        return True

    digest = hashlib.sha256(b"").hexdigest()
    timestamp = "2026-01-01T00:00:00.000000Z"
    missing = "missing"
    send = {
        "delivery_id": "d" * 32,
        "method": "send",
        "status": "delivered",
        "outcome": {"accepted": True, "session_id": "session"},
    }
    steer = {
        "delivery_id": "d" * 32,
        "method": "steer",
        "status": "queued",
        "outcome": {"accepted": True},
    }
    user_message = {
        "role": "user",
        "content": [{"type": "text", "text": "x"}],
        "metadata": {"zeta.origin": "user"},
    }
    project = {
        "schema_version": 1,
        "project_id": "p_" + "a" * 32,
        "name": "x",
        "scope": "x",
        "created_at": timestamp,
        "updated_at": timestamp,
        "canonical_integration_root": None,
        "repo_agents": "include",
        "lanes": [],
    }
    recovery_target = {"root_session_id": "session", "child_run_id": None}
    recovery_evidence = {
        "schema": "zeta.transcript-quarantine.v1",
        "quarantine_id": "a" * 32,
        "target": recovery_target,
        "source_digest": digest,
        "source_size": 1,
        "first_bad_row": 2,
        "first_bad_offset": 1,
        "valid_prefix_end": 1,
        "retained_last_sequence": 1,
        "repaired_digest": digest,
        "diagnosed_at": timestamp,
        "payload_digest": digest,
        "payload_size": 1,
    }
    recovery_state = {
        "schema": "zeta.transcript-recovery.v1",
        "quarantine_id": "a" * 32,
        "target": recovery_target,
        "state": "committed",
        "expected_source_digest": digest,
        "repaired_digest": digest,
        "cutoff_sequence": 1,
        "diagnosis_timestamp": timestamp,
        "memory_step": "not_required",
    }

    def recovery_target_parser(payload: dict[str, object]) -> object:
        return parse_evidence_manifest({**recovery_evidence, "target": payload})

    attention = {
        "id": "a" * 32,
        "created_at": timestamp,
        "session_id": "s",
        "project_id": None,
        "entry_id": None,
        "entry_seq": None,
        "lane": "main",
        "title": "title",
        "why": "why",
        "options": [],
        "recommendation": None,
        "status": "open",
        "resolved_at": None,
        "fork_session_id": None,
        "decision": None,
    }

    cases = [
        ShapeCase("hook_config", hook_root, {"hook": []}, frozenset()),
        ShapeCase(
            "hook_record",
            hook_record,
            {
                "event": "pre_tool",
                "command": "true",
                "timeout_seconds": 1,
                "tools": [],
                "on_error": "allow",
            },
            frozenset({"event", "command"}),
        ),
        ShapeCase(
            "todo_item",
            todo,
            {"content": "ship", "status": "pending"},
            frozenset({"content", "status"}),
        ),
        ShapeCase(
            "notification_marker",
            store_entry("notification_tui_presented"),
            {"notification_id": "n"},
            frozenset({"notification_id"}),
        ),
        ShapeCase(
            "turn_history",
            validate_turn_end_data,
            {"turn_id": "a" * 32, "reason": "completed"},
            frozenset({"turn_id", "reason"}),
        ),
        ShapeCase(
            "prompt_component",
            prompt_component,
            {"offset": 0, "length": 0, "digest": digest},
            frozenset({"offset", "length", "digest"}),
        ),
        ShapeCase(
            "approval_fact", validated_approval_facts, {"action": "run"}, frozenset()
        ),
        ShapeCase(
            "approval_display",
            validated_approval_display,
            {"effective_cwd": "/x", "resolved_path": "/x/y", "utf8_bytes": 1},
            frozenset(),
        ),
        ShapeCase(
            "child_run",
            validate_child_run_started_data,
            {
                "run_id": "session:1",
                "parent_run_id": None,
                "name": "worker",
                "task": "task",
                "started_at": timestamp,
            },
            frozenset({"run_id", "task", "started_at"}),
        ),
        ShapeCase(
            "pending_queue_closed", store_entry("pending_queue_closed"), {}, frozenset()
        ),
        ShapeCase(
            "client_delivery_accepted_outcome",
            nested_client({k: v for k, v in send.items() if k != "outcome"}, "outcome"),
            {"accepted": True, "session_id": "session"},
            frozenset({"accepted", "session_id"}),
        ),
        ShapeCase(
            "client_delivery_steer_accepted_outcome",
            nested_client(
                {k: v for k, v in steer.items() if k != "outcome"}, "outcome"
            ),
            {"accepted": True},
            frozenset({"accepted"}),
        ),
        ShapeCase(
            "fact_floor_root",
            fact_root,
            {"version": 1, "oldest": [], "newest": []},
            frozenset({"version", "oldest", "newest"}),
        ),
        ShapeCase(
            "fact_floor_aggregate",
            fact_aggregate,
            {
                "fact": {"type": "pr", "pr": 1},
                "first_seq": 1,
                "last_seq": 1,
                "value_rank": [1, 0],
            },
            frozenset({"fact", "first_seq", "last_seq", "value_rank"}),
        ),
        ShapeCase(
            "fact_pr", fact_record, {"type": "pr", "pr": 1}, frozenset({"type", "pr"})
        ),
        ShapeCase(
            "fact_pr_verdict",
            fact_record,
            {"type": "pr_verdict", "pr": 1, "verdict": "pass"},
            frozenset({"type", "pr", "verdict"}),
        ),
        ShapeCase(
            "fact_pr_head",
            fact_record,
            {"type": "pr_head", "pr": 1, "sha": "a" * 40},
            frozenset({"type", "pr", "sha"}),
        ),
        ShapeCase(
            "fact_child_status",
            fact_record,
            {"type": "child_status", "handle": "h", "status": "completed"},
            frozenset({"type", "handle", "status"}),
        ),
        ShapeCase(
            "fact_child_head",
            fact_record,
            {"type": "child_head", "handle": "h", "sha": "a" * 40},
            frozenset({"type", "handle", "sha"}),
        ),
        ShapeCase(
            "storage_manifest",
            manifest,
            {"writer_epoch": 3},
            frozenset({"writer_epoch"}),
        ),
        ShapeCase(
            "project_session_reference",
            session_reference,
            {
                "session_id": "s",
                "transcript_path": "sessions/s",
                "role": "session",
                "parent_session_id": None,
                "recorded_at": timestamp,
            },
            frozenset(
                {
                    "session_id",
                    "transcript_path",
                    "role",
                    "parent_session_id",
                    "recorded_at",
                }
            ),
        ),
        ShapeCase(
            "remote_memory_sync_state",
            lambda value: _valid_state(value, "p_x"),
            {"schema": 2, "project_id": "p_x", "files": {}, "conflicts": {}},
            frozenset({"schema", "project_id", "files", "conflicts"}),
        ),
        ShapeCase(
            "remote_memory_sync_conflict",
            remote_conflict,
            {"digests": ["a" * 64, missing]},
            frozenset({"digests"}),
        ),
        ShapeCase(
            "entry_sync_state",
            lambda value: _valid_entry_sync_state(value, "p_x"),
            {
                "schema": 3,
                "project_id": "p_x",
                "format": 2,
                "entries": {},
                "schema_digest": missing,
                "conflicts": {},
                "transition": None,
            },
            frozenset(
                {
                    "schema",
                    "project_id",
                    "format",
                    "entries",
                    "schema_digest",
                    "conflicts",
                }
            ),
        ),
        ShapeCase(
            "entry_sync_schema_conflict",
            entry_conflict,
            {"kind": "schema", "digests": [missing]},
            frozenset({"kind", "digests"}),
        ),
        ShapeCase(
            "entry_sync_entry_conflict",
            entry_conflict,
            {"kind": "entries", "entry_ids": ["m_a"], "digests": [missing]},
            frozenset({"kind", "entry_ids", "digests"}),
        ),
        ShapeCase(
            "entry_sync_transition",
            _valid_entry_transition,
            {
                "id": "0" * 32,
                "previous_state": missing,
                "resolutions": {},
                "candidates": {},
            },
            frozenset({"id", "previous_state", "resolutions", "candidates"}),
        ),
        ShapeCase(
            "entry_sync_schema_resolution_candidate",
            candidate,
            {
                "kind": "schema",
                "local": missing,
                "remote": missing,
                "prepared": missing,
            },
            frozenset({"kind", "local", "remote", "prepared"}),
        ),
        ShapeCase(
            "entry_sync_entry_resolution_candidate",
            candidate,
            {
                "kind": "entries",
                "entry_ids": ["m_a"],
                "local": missing,
                "remote": missing,
                "prepared": missing,
            },
            frozenset({"kind", "entry_ids", "local", "remote", "prepared"}),
        ),
        ShapeCase(
            "account_publication_marker",
            publication,
            {"provider": "codex", "name": "default", "sha256": "a" * 64},
            frozenset({"provider", "name", "sha256"}),
        ),
        ShapeCase(
            "memory_history_provenance",
            ProjectMemoryHistoryMixin._validate_provenance,
            {
                "session_id": "s",
                "seq_start": 1,
                "seq_end": 1,
                "fragment_start": 0,
                "fragment_end": 1,
                "model": "m",
                "usage": {},
            },
            frozenset({"session_id", "seq_start", "seq_end"}),
        ),
        ShapeCase(
            "entry_memory_snapshot_reference",
            snapshot_reference,
            {"state": "a" * 64},
            frozenset({"state"}),
        ),
        ShapeCase(
            "entry_memory_snapshot_reference",
            published_snapshot_reference,
            {"state": state_digest},
            frozenset({"state"}),
        ),
        ShapeCase(
            "legacy_memory_snapshot",
            published_legacy_snapshot,
            {name: legacy_digest for name in ("brief.md", "state.md", "backlog.md", "changelog.md", "decisions.md")},
            frozenset({"brief.md", "state.md", "backlog.md", "changelog.md", "decisions.md"}),
        ),
        ShapeCase(
            "legacy_memory_snapshot",
            history_legacy_snapshot,
            {name: legacy_digest for name in ("brief.md", "state.md", "backlog.md", "changelog.md", "decisions.md")},
            frozenset({"brief.md", "state.md", "backlog.md", "changelog.md", "decisions.md"}),
        ),
        ShapeCase(
            "eviction_reasoning_receipt_metadata",
            _valid_persisted_reasoning_receipt_metadata,
            {"context_evicted": True, "source_seq": 1},
            frozenset({"context_evicted", "source_seq"}),
        ),
        ShapeCase(
            "response_action",
            validate_terminal_transition_data,
            {"action": "cancel"},
            frozenset({"action"}),
        ),
        ShapeCase(
            "response_action_report",
            validate_terminal_transition_data,
            {"action": "finish", "report": {"status": "done", "summary": "ok"}},
            frozenset({"action", "report"}),
        ),
        ShapeCase(
            "response_report",
            lambda value: validate_terminal_transition_data(
                {"action": "finish", "report": value}
            ),
            {"status": "done", "summary": "ok"},
            frozenset({"status", "summary"}),
        ),
        ShapeCase(
            "child_idle_notification",
            validate_child_idle_notification_data,
            {
                "origin": "notification",
                "kind": "child_idle",
                "child_instance_id": "child",
                "child_session_path": "/sessions/child",
                "description": "waiting",
                "status": "child_idle",
                "text": "waiting",
                "ask_parent_question_id": None,
                "display_only": True,
            },
            frozenset(),
        ),
        ShapeCase(
            "project_registry_record",
            validate_project_record,
            project,
            frozenset(
                {
                    "schema_version",
                    "project_id",
                    "name",
                    "scope",
                    "created_at",
                    "updated_at",
                    "canonical_integration_root",
                }
            ),
        ),
        ShapeCase(
            "memory_state_root",
            exact(
                "format-2 memory state",
                {
                    "entries",
                    "format",
                    "schema",
                    "generation",
                    "project_id",
                    "compacted_through_version",
                },
            ),
            {
                key: None
                for key in (
                    "entries",
                    "format",
                    "schema",
                    "generation",
                    "project_id",
                    "compacted_through_version",
                )
            },
            frozenset(
                {
                    "entries",
                    "format",
                    "schema",
                    "generation",
                    "project_id",
                    "compacted_through_version",
                }
            ),
        ),
        ShapeCase(
            "memory_state_root",
            published_state_root,
            state,
            frozenset(state),
        ),
        ShapeCase(
            "memory_schema",
            exact("memory schema", {"profile", "version", "kinds"}),
            {key: None for key in ("profile", "version", "kinds")},
            frozenset({"profile", "version", "kinds"}),
        ),
        ShapeCase(
            "memory_kind",
            exact(
                "memory kind",
                {
                    "default_expiry_days",
                    "description",
                    "key",
                    "name",
                    "prompt_max_entries",
                    "prompt_mode",
                    "prompt_priority",
                },
            ),
            {
                key: None
                for key in (
                    "default_expiry_days",
                    "description",
                    "key",
                    "name",
                    "prompt_max_entries",
                    "prompt_mode",
                    "prompt_priority",
                )
            },
            frozenset(
                {
                    "default_expiry_days",
                    "description",
                    "key",
                    "name",
                    "prompt_max_entries",
                    "prompt_mode",
                    "prompt_priority",
                }
            ),
        ),
        ShapeCase(
            "memory_entry",
            exact(
                "memory entry",
                {
                    "status",
                    "supersedes",
                    "accepted_by",
                    "project_id",
                    "valid_from",
                    "created_at",
                    "superseded_by",
                    "representation",
                    "id",
                    "kind",
                    "text",
                    "section",
                    "sources",
                    "migration_order",
                    "accepted_at",
                    "seen_at",
                    "expiry_is_default",
                    "valid_until",
                    "last_operation_id",
                    "expires_at",
                    "migration_source",
                    "automatic",
                    "updated_at",
                },
            ),
            {
                key: None
                for key in (
                    "status",
                    "supersedes",
                    "accepted_by",
                    "project_id",
                    "valid_from",
                    "created_at",
                    "superseded_by",
                    "representation",
                    "id",
                    "kind",
                    "text",
                    "section",
                    "sources",
                    "migration_order",
                    "accepted_at",
                    "seen_at",
                    "expiry_is_default",
                    "valid_until",
                    "last_operation_id",
                    "expires_at",
                    "migration_source",
                    "automatic",
                    "updated_at",
                )
            },
            frozenset(
                {
                    "status",
                    "supersedes",
                    "accepted_by",
                    "project_id",
                    "valid_from",
                    "created_at",
                    "superseded_by",
                    "representation",
                    "id",
                    "kind",
                    "text",
                    "section",
                    "sources",
                    "migration_order",
                    "accepted_at",
                    "seen_at",
                    "expiry_is_default",
                    "valid_until",
                    "last_operation_id",
                    "expires_at",
                    "migration_source",
                    "automatic",
                    "updated_at",
                }
            ),
        ),
        ShapeCase(
            "memory_entry",
            persisted_legacy_entry,
            {
                "id": "m_" + "a" * 32,
                "project_id": "p_" + "a" * 32,
                "kind": "state",
                "text": "State",
                "representation": "entry",
                "status": "active",
                "created_at": timestamp,
                "updated_at": timestamp,
                "seen_at": timestamp,
                "expires_at": None,
                "valid_from": timestamp,
                "valid_until": None,
                "supersedes": [],
                "superseded_by": [],
                "sources": [],
                "automatic": False,
                "accepted_at": None,
                "accepted_by": None,
                "last_operation_id": "op_" + "b" * 32,
                "section": None,
                "migration_order": None,
                "migration_source": None,
            },
            legacy_entry_fields,
        ),
        ShapeCase(
            "memory_missing_entry",
            exact("missing memory entry", {"missing", "id"}),
            {key: None for key in ("missing", "id")},
            frozenset({"missing", "id"}),
        ),
        ShapeCase(
            "memory_source",
            exact(
                "memory entry source",
                {
                    "seq_start",
                    "seq_end",
                    "observed_at",
                    "origins",
                    "session_id",
                    "evidence_rank",
                },
            ),
            {
                key: None
                for key in (
                    "seq_start",
                    "seq_end",
                    "observed_at",
                    "origins",
                    "session_id",
                    "evidence_rank",
                )
            },
            frozenset(
                {
                    "seq_start",
                    "seq_end",
                    "observed_at",
                    "origins",
                    "session_id",
                    "evidence_rank",
                }
            ),
        ),
        ShapeCase(
            "memory_migration_source",
            exact("memory entry migration source", {"source_digest", "source_version"}),
            {key: None for key in ("source_digest", "source_version")},
            frozenset({"source_digest", "source_version"}),
        ),
        ShapeCase(
            "memory_operation_receipt",
            exact(
                "memory operation receipt",
                {
                    "result_ids",
                    "reconciliation_key",
                    "type",
                    "target_ids",
                    "automatic",
                    "reason",
                    "operation_id",
                },
            ),
            {
                key: None
                for key in (
                    "result_ids",
                    "reconciliation_key",
                    "type",
                    "target_ids",
                    "automatic",
                    "reason",
                    "operation_id",
                )
            },
            frozenset(
                {
                    "result_ids",
                    "reconciliation_key",
                    "type",
                    "target_ids",
                    "automatic",
                    "reason",
                    "operation_id",
                }
            ),
        ),
        ShapeCase(
            "background_output_journal_commit",
            _BackgroundOutputArchive._validate_commit,
            {"task_id": "t", "offset": 0, "length": 1},
            frozenset({"task_id", "offset", "length"}),
        ),
        ShapeCase("client_delivery_send_acceptance", client, send, frozenset(send)),
        ShapeCase("client_delivery_steer_acceptance", client, steer, frozenset(steer)),
        ShapeCase(
            "client_delivery_queued_steer_acceptance_message",
            client,
            {**steer, "accepted_message": user_message},
            frozenset(steer),
        ),
        ShapeCase(
            "client_delivery_delivered_transition",
            client,
            {"delivery_id": "d" * 32, "method": "steer", "status": "delivered"},
            frozenset({"delivery_id", "method", "status"}),
        ),
        ShapeCase(
            "client_delivery_dropped_transition",
            client,
            {
                "delivery_id": "d" * 32,
                "method": "steer",
                "status": "dropped",
                "reason": "restart",
            },
            frozenset({"delivery_id", "method", "status", "reason"}),
        ),
        ShapeCase(
            "tool_result_text",
            lambda value: _validate_text_block("x", value),
            {
                "type": "text",
                "text": "x",
                "truncated": False,
                "full_size": 1,
                "full_size_chars": 1,
                "next_offset": 1,
                "spill_path": "/x",
                "annotations": {},
            },
            frozenset({"type", "text", "truncated", "full_size"}),
        ),
        ShapeCase(
            "tool_result_image",
            lambda value: _validate_image_block("x", value, complete_image=False),
            {
                "type": "image",
                "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
                "mimeType": "image/png",
                "caption": "x",
                "path": "/x",
                "width": 1,
                "height": 1,
                "size": 1,
                "annotations": {},
            },
            frozenset({"type", "data", "mimeType"}),
        ),
        ShapeCase(
            "tool_result_resource",
            lambda value: _validate_resource_block("x", value),
            {
                "type": "resource",
                "resource": {"uri": "x", "text": "x"},
                "annotations": {},
            },
            frozenset({"type", "resource"}),
        ),
        ShapeCase(
            "tool_result_resource_contents",
            lambda value: _validate_resource_block(
                "x", {"type": "resource", "resource": value}
            ),
            {"uri": "x", "mimeType": "text/plain", "text": "x"},
            frozenset({"uri", "text"}),
        ),
        ShapeCase(
            "tool_result_annotations",
            lambda value: _validate_annotations("x", value),
            {"audience": ["user"], "priority": 1.0, "lastModified": timestamp},
            frozenset(),
        ),
        ShapeCase(
            "attention_record",
            AttentionRecord.from_dict,
            attention,
            frozenset(
                {
                    "id",
                    "created_at",
                    "session_id",
                    "project_id",
                    "entry_id",
                    "entry_seq",
                    "lane",
                    "title",
                    "why",
                    "recommendation",
                    "status",
                }
            ),
        ),
        ShapeCase(
            "attention_fork_meta",
            attention_fork,
            {"forked_from_session": "s", "forked_at_entry": "e", "attention_id": "a"},
            frozenset({"forked_from_session", "forked_at_entry", "attention_id"}),
        ),
        ShapeCase(
            "transcript_recovery.evidence",
            parse_evidence_manifest,
            recovery_evidence,
            frozenset(recovery_evidence),
        ),
        ShapeCase(
            "transcript_recovery.state",
            parse_recovery_record,
            recovery_state,
            frozenset(recovery_state),
        ),
        ShapeCase(
            "transcript_recovery.target",
            recovery_target_parser,
            recovery_target,
            frozenset(recovery_target),
        ),
    ]

    try:
        assert {case.name for case in cases} == (
            set(ACCEPTED_KEY_SETS) - set(_DORMANT_UNTIL_SLICE)
        )
        for case in cases:
            result = case.validator(dict(case.valid))
            assert result is not False, case.name
            _assert_invalid(
                case.validator,
                {**case.valid, "_registry_probe_extra": True},
                f"{case.name} extra",
            )
            for key in case.required:
                missing_key = dict(case.valid)
                del missing_key[key]
                _assert_invalid(
                    case.validator, missing_key, f"{case.name} missing {key}"
                )
    finally:
        os.close(account_fd)
        store.close()


def test_registry_delta_requires_epoch_and_names_the_fix() -> None:
    predecessor = {
        "writer_epoch": 3,
        "accepted_key_sets": {"todo_item": ["content", "status"]},
    }
    candidate = {
        "writer_epoch": 3,
        "accepted_key_sets": {"todo_item": ["content", "status", "extra"]},
    }
    with pytest.raises(
        ValueError, match="bump the epoch in src/zeta/storage_format.py"
    ):
        compare_registry_dumps(predecessor, candidate)
