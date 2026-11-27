"""Reconstruct suspended background runtimes under their existing subtree lease."""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.abort import AbortSignal
from ..core.checkpoints import _now
from ..core.store._async_writes import (
    _AppendChildRecoveredCommand,
    _CloseChildApprovalHistoryCommand,
    _RepairChildInterruptionCommand,
    _SelectChildAccountsCommand,
)
from ..protocol.types import StreamEvent, StreamEventType, ToolCall
from ..runtime.approval_recovery import _end_payload
from ..runtime.offload import await_joined_offload
from .background_lifecycle import _recoverable_marker
from .child_context import resolve_child_backend
from .recovery_lease import acquire_recovery_subtree
from .resume_descriptor import restore_preset
from .runner import _build_child_loop, _run_installed_child
from .tool_results import validated_tool_result
from .turn_limit import turn_control_for

if TYPE_CHECKING:
    from ..core.store import ConversationStore
    from ..runtime.loop import AgentLoop


def _resolve_backend(parent: AgentLoop, saved: dict[str, Any], context: Any) -> Any:
    backend, error = resolve_child_backend(parent, saved["model"], pinned_context=context)
    if error or backend is None or getattr(backend, "provider", None) != saved["provider"]:
        raise ValueError("saved model or account is unavailable")
    return backend


async def repair_interrupted_child(root: ConversationStore, store: ConversationStore, run_id: str) -> None:
    await store._commit_storage_command(_RepairChildInterruptionCommand())
    for wire_id, tool_call in root.delegated_unmatched_requests(run_id):
        request_id = tool_call.get("id")
        if isinstance(request_id, str) and (resolution := store.approval_resolution_now(request_id)) is not None:
            await root._commit_storage_command(_CloseChildApprovalHistoryCommand(
                payload=_end_payload(wire_id, tool_call, resolution),
            ))


async def resume_agent_children(loop: AgentLoop) -> None:
    for marker in loop.store.agent_children().values():
        if not marker.get("background") or "resume" not in marker:
            continue
        path = Path(marker["child_session_path"])
        leases = ExitStack()
        stores: dict[Path, ConversationStore] = {}
        retained: set[ConversationStore] = set()
        try:
            acquired = await await_joined_offload(
                acquire_recovery_subtree, loop.store, path, leases, stores, retained=retained,
            )
            if acquired and path in stores:
                await _resume_child(loop, marker, stores[path], stores, retained)
        finally:
            await await_joined_offload(leases.close)


async def _resume_child(
    parent: AgentLoop, marker: dict[str, Any], store: ConversationStore,
    stores: dict[Path, ConversationStore], retained: set[ConversationStore],
) -> None:
    tree = parent._background_owner
    run_id = marker["child_instance_id"]
    if tree.owns_running(run_id) or not await await_joined_offload(_recoverable_marker, marker, store, parent.store):
        return
    saved = marker["resume"]
    preset = restore_preset(saved["preset"])
    tool_call = ToolCall.from_dict(marker["tool_call"])
    pins = saved["selected_accounts"]
    description = saved["description"]
    turns = max(marker["turns_used"], (store.agent_lifecycle() or {}).get("turns_used", 0))
    turns = max(turns, *(entry.data.get("settlement_turn", 0) for entry in store._delivery_fold.child_questions.values()), 0)
    control = turn_control_for(saved["effective_max_turns"], str(store.session_dir))
    if control is not None:
        control.restore_used(turns)

    async def initialize() -> None:
        if store.agent_lifecycle() is None:
            await store._to_thread_store(lambda: store.start_agent_lifecycle(
                handle=run_id, started_at=_now(), depth=saved["depth"], agent_type=preset.name,
                description=description, cwd=saved["cwd"], selected_accounts=pins or None,
            ))

    task = asyncio.current_task()
    assert task is not None
    owner = tree.begin_run(
        run_id, store, tool_call.id, task, initialize, parent_store=parent.store,
        parent_instance_id=parent.agent_instance_id, description=description,
        explicit_finish=saved["explicit_finish"], background=True,
    )
    activation = asyncio.Event()
    installed = False
    try:
        state = await owner.persist_suspension("owner_lost")
        if state is None:
            return

        def load_backend() -> tuple[Any, Any, dict[str, str] | None]:
            context = getattr(parent, "account_context", None)
            if saved["account_pins"] == "pending":
                context = context.child_snapshot(saved["provider"]) if context is not None else None
                selected = {name: str(pin) for name, pin in context.snapshot().items()} if context is not None else {}
                return _resolve_backend(parent, saved, context), context, selected
            elif context is not None:
                context = context.restore_child_snapshot(saved["provider"], pins)
            elif pins:
                raise ValueError("saved account context is unavailable")
            return _resolve_backend(parent, saved, context), context, None

        try:
            backend, context, selected = await await_joined_offload(load_backend)
            if selected is not None:
                await parent.store._commit_storage_command(_SelectChildAccountsCommand(run_id=run_id, pins=selected))
        except (OSError, ValueError, RuntimeError):
            await owner.block_recovery("saved model or account is unavailable")
            return
        if state["resume_mode"] == "active":
            await repair_interrupted_child(tree.notification_store, store, run_id)
        store.bind_owner_thread()
        try:
            child, policy = await _build_child_loop(
                parent, child_backend=backend, child_store=store, child_instance_id=run_id,
                child_depth=saved["depth"], preset=preset, model=saved["model"],
                cwd_override=saved["cwd"], follow_up_loop=saved["follow_up"],
                description=description, child_account_context=context, run_owner=owner,
                child_turn_control=control, recovered_explicit_finish=saved["explicit_finish"],
            )
        except (OSError, ValueError, RuntimeError):
            await owner.block_recovery("saved child runtime is unavailable under current policy")
            return
        store.accepts_follow_ups = saved["follow_up"]
        child.agent_background = True
        tree.project_context_loops[store.session_id] = child
        tree.track_store(store)
        retained.add(store)
        parent._agent_child_stores[tool_call.id] = store
        parent._agent_child_turns[tool_call.id] = turns
        parent._agent_child_types[tool_call.id] = preset.name
        was = "suspended" if state["suspension_reason"] == "normal_exit" else "interrupted because the parent crashed"
        held = control is not None and control.exhausted and (await owner.hold_at_limit() or owner.settlement_due())
        await _run_installed_child(
            parent, child, store, owner, tool_call,
            prompt=f"Execution was {was}. Review the durable context and verify any interrupted work before continuing.",
            description=description, preset=preset, child_depth=saved["depth"],
            child_instance_id=run_id, background=True, follow_up_loop=saved["follow_up"],
            child_policy=policy, child_account_pins=context, external=False, abort_signal=AbortSignal(),
            publisher=None, validate_result=validated_tool_result, error_message=str,
            execution_context=None, activation=activation,
            resumed_idle=held or state["resume_mode"] == "idle" and (control is None or not control.exhausted),
        )
        installed = True
        episode = state["suspension_episode"]
        notice, created = await parent.store._commit_storage_command(_AppendChildRecoveredCommand(
            run_id=run_id, episode=episode, path=str(store.session_dir), description=description,
        ))
        if created:
            parent._publish_background_event(StreamEvent(
                StreamEventType.AGENT_NOTIFICATION,
                data={"notification_id": notice.id, **notice.data}, row_seq=notice.seq,
            ))
        for descendant in store.agent_children().values():
            if nested := stores.get(Path(descendant["child_session_path"])):
                await _resume_child(child, descendant, nested, stores, retained)
        await owner.reconcile_questions()
        await owner.activate_recovered(episode)
        activation.set()
    finally:
        if not installed:
            tree.conversation_channel.unregister_loop(run_id)
            tree.unregister(run_id)
        elif not activation.is_set():
            await owner.suspend()
            await tree.wait_for_descendant_watchers(run_id)
