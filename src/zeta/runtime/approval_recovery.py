"""Durable restart recovery for approvals a resumed runtime will not re-present.

Rule (lead decision, 2026-11-11): abort every durable unresolved approval that
the resumed runtime does NOT re-present, and only those.

- The resumed session's own foreground requests ARE re-presented: the serve
  ``status`` RPC and the TUI both rebuild the pending set from the store
  (``_pending_approvals_projection`` in ``server/approval_lifecycle.py``, read at
  ``server/handlers/lifecycle.py`` before any join). Recovery leaves them open.
- A delegated child is canceled on resume and is never re-presented, so its
  unresolved approvals are stale: recovery aborts them and writes the missing
  delegated end on the root session history (serve writes every delegated
  approval row to the root store at any depth).

One ``restart_abort_reason(home_epoch)`` selector picks the reason for every
store in a recovered tree. All stores share the root writer lease, so they share
one observed home epoch.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..core.store import ConversationStore
from ..storage_format import restart_abort_reason


def recover_store_approvals(store: ConversationStore, home_epoch: int) -> None:
    """Abort this store's own unresolved approvals, idempotently.

    Already-resolved requests are a no-op: ``resolve_approval`` checks the state
    under the append lock, so a repeated restart writes nothing.
    """
    reason = restart_abort_reason(home_epoch)
    # Every unresolved approval in this store, same semantics as main's removed
    # ``pending_approvals()``: a resident non-blocking read, paged so no count
    # is dropped. Resolving a page removes it from the next page's pending set.
    after_seq = 0
    while True:
        page = store.pending_approval_page_now(after_seq)
        for request_id, _tool_call in page.items:
            store.resolve_approval(request_id, "abort", reason=reason)
        if page.next_seq is None:
            break
        after_seq = page.next_seq


def repair_delegated_child(
    history_store: ConversationStore,
    child_store: ConversationStore,
    child_instance_id: str,
    home_epoch: int,
) -> None:
    """Close a child's unresolved approvals and project any missing end.

    ``history_store`` is the root session store: serve writes every delegated
    approval row there, at any depth, keyed by the child's own
    ``agent_instance_id``. Post-order: the caller recurses into deeper children
    first, so a child's authoritative resolution exists before its delegated end
    is projected. The end carries the child's actual durable decision; a user
    ``allow`` or ``deny`` is never rewritten to a restart abort.
    """
    recover_store_approvals(child_store, home_epoch)
    for wire_id, tool_call in history_store.delegated_unmatched_requests(
        child_instance_id
    ):
        store_request_id = tool_call.get("id")
        if not isinstance(store_request_id, str):
            continue
        # Reread the authoritative child resolution. A false resolve result (for
        # example another owner committed first) is never a reason to skip this.
        resolution = child_store.approval_resolution_now(store_request_id)
        if resolution is None:
            continue
        history_store.append_approval_history_end_if_absent(
            _end_payload(wire_id, tool_call, resolution)
        )


def _end_payload(
    wire_id: str, tool_call: Mapping[str, object], resolution: Mapping[str, str]
) -> dict[str, object]:
    data: dict[str, object] = {"decision": resolution["decision"]}
    if "reason" in resolution:
        data["reason"] = resolution["reason"]
    payload: dict[str, object] = {
        "request_id": wire_id,
        "tool_call": dict(tool_call),
        "data": data,
    }
    if "reason" in data:
        payload["reason"] = data["reason"]
    return payload


__all__ = ["recover_store_approvals", "repair_delegated_child"]
