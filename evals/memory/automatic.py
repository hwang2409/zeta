"""Benchmark adapter for automatic, incremental memory reconciliation.

The product implementation is developed separately. This adapter models its required
observable behavior: reconcile a live transcript range, apply it automatically, and
persist both a sequence cursor and an append-only version receipt before another
session continues.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from zeta.memory.reconciler import (
    Proposal,
    ReconciliationError,
    Transcript,
    build_prompt,
    memory_digest,
    parse_proposal,
    read_transcript,
)
from zeta.project_errors import ProjectRegistryError
from zeta.project_registry import ProjectRegistry


@dataclass(frozen=True, slots=True)
class AutoReconcileReceipt:
    session_id: str
    seq_start: int
    seq_end: int
    trigger: str
    files: tuple[str, ...]
    rejected_files: tuple[str, ...]
    before_digest: str
    after_digest: str


def reconcile_session(
    transcript_path: Path,
    session_id: str,
    memory: Mapping[str, str],
    invoke: Callable[[str], str],
    *,
    as_of: date,
) -> Proposal:
    """Produce one filtered legacy benchmark proposal."""

    transcript = read_transcript(transcript_path, session_id)
    raw = invoke(build_prompt(transcript, memory, as_of=as_of))
    return parse_proposal(
        raw,
        expected_digest=memory_digest(memory),
        transcript=transcript,
        as_of=as_of,
    )


def apply_proposal(
    registry: ProjectRegistry, project_id: str, proposal: Proposal
) -> list[tuple[str, str]]:
    """Apply a legacy benchmark proposal with the product registry CAS."""

    updates = {item.name: item.content for item in proposal.replacements}
    if not updates:
        return registry.load_memory(project_id)
    try:
        result = registry.compare_and_swap_memory(
            project_id, expected_digest=proposal.base_digest, updates=updates
        )
        return result.contents
    except ProjectRegistryError as exc:
        raise ReconciliationError("project memory changed before approval") from exc


class AutomaticReconciler:
    """Incrementally reconcile transcript rows through one crash-safe interface."""

    def __init__(self, state_dir: Path) -> None:
        self._state_dir = state_dir
        self._cursor_path = state_dir / "cursor.json"
        self._versions_path = state_dir / "versions.jsonl"

    def reconcile_available(
        self,
        *,
        transcript_path: Path,
        session_id: str,
        memory: Mapping[str, str],
        registry: ProjectRegistry,
        project_id: str,
        invoke: Callable[[str], str],
        as_of: date,
        trigger: str,
    ) -> AutoReconcileReceipt | None:
        """Apply all rows after the durable cursor and advance it after the write."""
        transcript = read_transcript(transcript_path, session_id)
        cursor = self._load_cursor().get(session_id, 0)
        rows = tuple(
            row
            for row in transcript.rows
            if type(row.get("seq")) is int and row["seq"] > cursor
        )
        if not rows:
            return None
        incremental = Transcript(session_id, rows)
        raw = invoke(build_prompt(incremental, memory, as_of=as_of))
        proposal = parse_proposal(
            raw,
            expected_digest=memory_digest(memory),
            transcript=incremental,
            as_of=as_of,
        )
        before_digest = proposal.base_digest
        apply_proposal(registry, project_id, proposal)
        after_memory = dict(registry.load_memory(project_id))
        receipt = AutoReconcileReceipt(
            session_id=session_id,
            seq_start=min(incremental.sequences),
            seq_end=max(incremental.sequences),
            trigger=trigger,
            files=tuple(item.name for item in proposal.replacements),
            rejected_files=proposal.rejected_files,
            before_digest=before_digest,
            after_digest=memory_digest(after_memory),
        )
        self._persist(receipt)
        return receipt

    def versions(self) -> tuple[AutoReconcileReceipt, ...]:
        if not self._versions_path.is_file():
            return ()
        return tuple(
            AutoReconcileReceipt(**json.loads(line))
            for line in self._versions_path.read_text().splitlines()
            if line.strip()
        )

    def _load_cursor(self) -> dict[str, int]:
        if not self._cursor_path.is_file():
            return {}
        value = json.loads(self._cursor_path.read_text())
        if not isinstance(value, dict) or any(
            not isinstance(key, str) or type(position) is not int
            for key, position in value.items()
        ):
            raise ValueError("invalid automatic reconciliation cursor")
        return value

    def _persist(self, receipt: AutoReconcileReceipt) -> None:
        self._state_dir.mkdir(parents=True, exist_ok=True)
        with self._versions_path.open("a") as output:
            output.write(json.dumps(asdict(receipt), sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        cursor = self._load_cursor()
        cursor[receipt.session_id] = receipt.seq_end
        temporary = self._cursor_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(cursor, sort_keys=True) + "\n")
        os.replace(temporary, self._cursor_path)
