"""Checkpoint commands and transcript rebuilding for the terminal UI."""

from __future__ import annotations

from datetime import UTC, datetime

from ..core.checkpoints import BranchInfo, ConversationIntegrityError
from ..core.checkpoints.workspace import (
    SIZE_NOTICE_THRESHOLD_BYTES,
    SNAPSHOT_MODE_GIT,
    WorkspaceSnapshot,
    WorkspaceSnapshotError,
    WorkspaceSnapshotStore,
    git_repo_root,
)
from ..types import (
    FAILED_TURN_ERROR,
    FAILED_TURN_MARKER,
    ErrorInfo,
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    ToolCall,
    ToolUseContent,
    assistant_text,
)
from .render import is_retryable_error, render_event, render_markdown

FORCE_FLAGS = frozenset({"--force", "-f", "!"})


def _age(created_at: str) -> str:
    try:
        seconds = max(
            0,
            int(
                (datetime.now(UTC) - datetime.fromisoformat(created_at)).total_seconds()
            ),
        )
    except ValueError:
        return "age unknown"
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3_600:
        return f"{seconds // 60}m ago"
    if seconds < 86_400:
        return f"{seconds // 3_600}h ago"
    return f"{seconds // 86_400}d ago"


def _parse_force(args: str) -> tuple[str, bool]:
    """Return ``(remainder, forced)`` after peeling any force flag."""

    tokens = args.split()
    forced = False
    remaining: list[str] = []
    for token in tokens:
        if token in FORCE_FLAGS:
            forced = True
            continue
        remaining.append(token)
    return " ".join(remaining), forced


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes}B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f}MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.2f}GB"


class CheckpointTranscriptMixin:
    """Add checkpoint commands and active-branch transcript rebuilding."""

    _workspace_snapshot_store: WorkspaceSnapshotStore | None = None
    _workspace_snapshot_cap: int | None = None

    def _snapshots(self) -> WorkspaceSnapshotStore:
        if self._workspace_snapshot_store is None:
            self._workspace_snapshot_store = WorkspaceSnapshotStore(
                self.loop.store.session_dir,
                self.loop.store.session_id,
                cap=self._workspace_snapshot_cap,
            )
        return self._workspace_snapshot_store

    def slash_checkpoint(self, args: str) -> str:
        if self.active or self.loop.store.turn_in_flight():
            return "checkpoint unavailable while a turn is running"
        if args.strip() == "--reset":
            try:
                self._snapshots().reset_corruption()
            except (WorkspaceSnapshotError, OSError) as exc:
                return f"snapshot state reset failed: {exc}"
            return "workspace snapshot state reset; valid snapshots retained"
        try:
            entry = self.loop.store.append_checkpoint(args.strip() or None)
        except ValueError as exc:
            return f"checkpoint failed: {exc}"
        label = entry.data["label"]
        try:
            snapshots = self._snapshots()
            snapshot = snapshots.take(
                self.loop.store.bash_cwd,
                label=label,
                checkpoint_entry_id=entry.id,
            )
        except (WorkspaceSnapshotError, OSError) as exc:
            return (
                f"checkpoint '{label}' at seq {entry.seq} "
                f"(workspace snapshot skipped: {exc})"
            )
        return _format_checkpoint_result(entry.seq, label, snapshot)

    async def slash_compact(self) -> str:
        """Force one compaction through the context assembler."""

        if self.active:
            return "compact unavailable while a turn is running"
        before = self.loop.store.compaction_marker_count()
        try:
            context = await self.loop.context_assembler.assemble_context(
                backend=self.loop.backend,
                force=True,
            )
        except Exception as exc:  # noqa: BLE001 - context assembler surface is broad
            return f"compact failed: {exc}"
        finally:
            self._usage_tracker.record_compaction(self.model)
        after = self.loop.store.compaction_marker_count()
        if after == before:
            return "compact: nothing to compact"
        marker = next(
            entry
            for entry in reversed(self.loop.store.replay())
            if entry.type == "compaction"
        )
        return (
            "compacted entries "
            f"{marker.data['source_seq_start']}–{marker.data['source_seq_end']}; "
            f"tokens after: {context.token_count}"
        )

    def slash_fork(self, args: str) -> str:
        if self.active:
            return "fork unavailable while a turn is running"
        if self.loop.store.has_outstanding_tool_calls(self.loop.store.replay()):
            return "fork unavailable while a tool call is pending"
        if self.loop.background_children_running:
            return "fork unavailable while background agents are running"
        selector, forced = _parse_force(args)
        selector = selector.strip()
        if not selector:
            return self._render_fork_picker()
        if selector.isdigit():
            return self._fork_from_message_index(int(selector), forced)
        return self._fork_from_checkpoint_label(selector, forced)

    def _render_fork_picker(self) -> str:
        forkpoints = self.loop.store.list_user_message_forkpoints()
        checkpoints = self.loop.store.list_checkpoints()
        if not forkpoints and not checkpoints:
            return "no user messages to fork from; run a turn first"
        snapshots = self._snapshots()
        lines: list[str] = []
        if forkpoints:
            lines.append("prior user messages on the active branch:")
            for index, entry, preview in forkpoints:
                snap_note = _snapshot_note(snapshots.by_checkpoint(entry.id))
                lines.append(
                    f"{index} · seq {entry.seq} · {snap_note} · "
                    f"{preview or '(empty)'}"
                )
        if checkpoints:
            lines.append("explicit checkpoint labels:")
            for entry, preview in checkpoints:
                snap_note = _snapshot_note(snapshots.by_checkpoint(entry.id))
                next_message = preview or "(no message after checkpoint)"
                lines.append(
                    f"{entry.data['label']} · seq {entry.seq} · "
                    f"{_age(entry.data['created_at'])} · {snap_note} · "
                    f"{next_message}"
                )
        lines.append(
            "use /fork <n> for a message index or /fork <label> for a checkpoint"
        )
        return "\n".join(lines)

    def _fork_from_message_index(self, index: int, forced: bool) -> str:
        forkpoints = self.loop.store.list_user_message_forkpoints()
        if index < 1 or index > len(forkpoints):
            return (
                "fork failed: user message index out of range "
                f"(1..{len(forkpoints)})"
            )
        _, source, _ = forkpoints[index - 1]
        snapshots = self._snapshots()
        target_snapshot = snapshots.by_checkpoint(source.id)
        if target_snapshot is not None:
            dirty_warning = _dirty_guard(
                snapshots, self.loop.store.bash_cwd, forced
            )
            if dirty_warning is not None:
                return dirty_warning
        try:
            entry = self.loop.store.append_message_fork(source.id)
        except (ValueError, ConversationIntegrityError) as exc:
            return f"fork failed: {exc}"
        _, restore_note = _restore_workspace(
            snapshots, target_snapshot, self.loop.store.bash_cwd, forced=forced
        )
        self._rebuild_transcript()
        self._fork_rebuilt = True
        header = (
            f"forked to user message {index} at seq {source.seq} "
            f"(label '{entry.data['label']}')"
        )
        if restore_note:
            return f"{header} ({restore_note})"
        return f"{header} · conversation-only, no workspace snapshot at this message"

    def _fork_from_checkpoint_label(self, selector: str, forced: bool) -> str:
        snapshots = self._snapshots()
        # With no snapshots this fork only changes the conversation.
        if snapshots.snapshots:
            dirty_warning = _dirty_guard(snapshots, self.loop.store.bash_cwd, forced)
            if dirty_warning is not None:
                return dirty_warning
        try:
            entry = self.loop.store.append_fork(selector)
        except ValueError as exc:
            return f"fork failed: {exc}"
        target_snapshot = _resolve_fork_snapshot(snapshots, entry)
        _, restore_note = _restore_workspace(
            snapshots, target_snapshot, self.loop.store.bash_cwd, forced=forced
        )
        self._rebuild_transcript()
        self._fork_rebuilt = True
        message = (
            f"forked to checkpoint '{entry.data['label']}' at seq "
            f"{entry.data['from_seq']}"
        )
        if restore_note:
            return f"{message} ({restore_note})"
        return message

    def slash_tree(self, args: str) -> str:
        if self.active:
            return "tree unavailable while a turn is running"
        tokens = args.split()
        branches = self.loop.store.list_branches()
        if not tokens:
            return _render_branch_tree(branches)
        if len(tokens) != 1 or not tokens[0].isdigit():
            return "tree: usage /tree [n]"
        if self.loop.store.has_outstanding_tool_calls(self.loop.store.replay()):
            return "tree unavailable while a tool call is pending"
        if self.loop.background_children_running:
            return "tree unavailable while background agents are running"
        index = int(tokens[0])
        if index < 1 or index > len(branches):
            return f"tree: branch index out of range (1..{len(branches)})"
        target = branches[index - 1]
        if target.is_current:
            return f"tree: already on branch {index}"
        try:
            self.loop.store.switch_to_branch(target.head.id)
        except (ValueError, ConversationIntegrityError) as exc:
            return f"tree: switch failed: {exc}"
        self._rebuild_transcript()
        self._fork_rebuilt = True
        return f"switched to branch {index} (head seq {target.head.seq})"

    def slash_undo(self, args: str) -> str:
        return self._navigate_snapshot(args, verb="undo", past="undone")

    def slash_redo(self, args: str) -> str:
        return self._navigate_snapshot(args, verb="redo", past="redone")

    def _navigate_snapshot(self, args: str, *, verb: str, past: str) -> str:
        remainder, forced = _parse_force(args)
        if remainder.strip():
            return f"{verb} unchanged: /{verb} accepts only --force"
        if self.active or self.loop.store.turn_in_flight():
            return f"{verb} unavailable while a turn is running"
        if self.loop.background_children_running:
            return f"{verb} unavailable while background agents are running"
        snapshots = self._snapshots()
        if snapshots.is_corrupt and not forced:
            return snapshots.corruption_message
        target = (
            snapshots.undo_target() if verb == "undo" else snapshots.redo_target()
        )
        if target is None:
            if snapshots.is_corrupt:
                return snapshots.corruption_message
            direction = "earlier" if verb == "undo" else "later"
            return f"{verb} unavailable: no {direction} workspace snapshot"
        dirty_warning = _dirty_guard(snapshots, self.loop.store.bash_cwd, forced)
        if dirty_warning is not None:
            return dirty_warning
        restored, restore_note = _restore_workspace(
            snapshots, target, self.loop.store.bash_cwd, forced=forced
        )
        if not restored:
            return f"{verb} failed: {restore_note}"
        label = target.label or target.id[:8]
        message = f"{past} to workspace snapshot '{label}'"
        if restore_note:
            return f"{message} ({restore_note})"
        return message

    def _rebuild_transcript(self) -> None:
        """Re-render the visible transcript from the active durable branch."""

        self._presenter.clear()
        self._failed_turn = None
        tool_calls: dict[str, ToolCall] = {}
        last_user: Message | None = None
        pending_notifications = {
            entry.id for entry in self.loop.store.agent_notifications()
        }
        for entry in self.loop.store.replay():
            if entry.type == "checkpoint":
                self._print_system(
                    f"checkpoint '{entry.data['label']}' at seq {entry.seq}"
                )
                continue
            if entry.type == "fork":
                self._print_system(_fork_banner(entry))
                continue
            if entry.type == "compaction":
                self._print_system(
                    "[compaction marker: entries "
                    f"{entry.data['source_seq_start']}–{entry.data['source_seq_end']}]"
                )
                continue
            if entry.type != "message":
                if entry.type == "notification" and entry.id in pending_notifications:
                    self._print_unit(
                        render_event(
                            StreamEvent(
                                StreamEventType.AGENT_NOTIFICATION,
                                data={"notification_id": entry.id, **entry.data},
                            )
                        )
                    )
                    self.loop.store.acknowledge_agent_notification(entry.id)
                continue
            message = Message.from_dict(entry.data["message"])
            if message.role is MessageRole.USER:
                self._failed_turn = None
                last_user = message
                self._print_user(message)
            elif message.role is MessageRole.ASSISTANT:
                text = assistant_text(message)
                if text:
                    self._print_unit(render_markdown(text))
                if not message.metadata.get(FAILED_TURN_MARKER):
                    self._failed_turn = None
                else:
                    error_value = message.metadata.get(FAILED_TURN_ERROR)
                    error = (
                        ErrorInfo.from_dict(error_value)
                        if isinstance(error_value, dict)
                        else ErrorInfo("backend_error", "provider failure")
                    )
                    if is_retryable_error(error) and last_user is not None:
                        self._failed_turn = (assistant_text(last_user), last_user)
                    else:
                        self._failed_turn = None
                    self._print_unit(
                        render_event(StreamEvent(StreamEventType.ERROR, error=error))
                    )
                for block in message.content:
                    if isinstance(block, ToolUseContent):
                        tool_calls[block.tool_call.id] = block.tool_call
            elif (
                message.role is MessageRole.TOOL_RESULT
                and message.tool_result is not None
            ):
                self._print_unit(
                    render_event(
                        StreamEvent(
                            StreamEventType.TOOL_EXECUTION_END,
                            tool_call=tool_calls.get(message.tool_result.tool_call_id),
                            tool_result=message.tool_result,
                        )
                    )
                )


def _fork_banner(entry: object) -> str:
    """Format the transcript banner for a fork entry by source type."""

    data = entry.data  # type: ignore[attr-defined]
    label = data.get("label", "")
    from_seq = data.get("from_seq", "?")
    source_type = data.get("source_type", "checkpoint")
    if source_type == "message":
        return f"forked to user message '{label}' at seq {from_seq}"
    if source_type == "branch":
        return f"switched to branch '{label}' at seq {from_seq}"
    return f"forked to checkpoint '{label}' at seq {from_seq}"


def _format_checkpoint_result(
    seq: int, label: str, snapshot: WorkspaceSnapshot
) -> str:
    if snapshot.mode != SNAPSHOT_MODE_GIT:
        return (
            f"checkpoint '{label}' at seq {seq} "
            "(workspace snapshot skipped: not a git repository)"
        )
    size = _format_size(snapshot.size_bytes)
    base = (
        f"checkpoint '{label}' at seq {seq} "
        f"(workspace snapshot: {snapshot.file_count} files, {size})"
    )
    if snapshot.size_bytes >= SIZE_NOTICE_THRESHOLD_BYTES:
        return (
            f"{base} — WARNING: workspace snapshot is large ({size}); "
            "consider adding heavy paths to .gitignore"
        )
    return base


TREE_SOFT_CAP = 20


def _render_branch_tree(branches: list[BranchInfo]) -> str:
    if not branches:
        return "no branches"
    lines = ["branches on this session:"]
    # Soft cap: long branch lists pushed the "* current" marker and the
    # /tree usage footer off the visible transcript. Keep the first and last
    # halves so the current branch marker stays visible.
    head_count = TREE_SOFT_CAP // 2
    hidden_start: int | None = None
    hidden_end: int | None = None
    if len(branches) > TREE_SOFT_CAP:
        hidden_start = head_count
        hidden_end = len(branches) - head_count
    for pos, branch in enumerate(branches):
        if (
            hidden_start is not None
            and hidden_end is not None
            and hidden_start <= pos < hidden_end
        ):
            if pos == hidden_start:
                lines.append(f"  ... {hidden_end - hidden_start} more branches ...")
            continue
        index = pos + 1
        marker = "*" if branch.is_current else " "
        divergence = (
            f" · from seq {branch.divergence.seq}"
            if branch.divergence is not None
            else ""
        )
        lines.append(
            f"{marker} {index}. head seq {branch.head.seq}{divergence} · "
            f"{branch.message_count} user messages · {branch.preview}"
        )
    lines.append("use /tree <n> to switch branches")
    return "\n".join(lines)


def _snapshot_note(snapshot: WorkspaceSnapshot | None) -> str:
    if snapshot is None or snapshot.mode != SNAPSHOT_MODE_GIT:
        return "no workspace snapshot"
    return f"workspace: {snapshot.file_count} files"


def _resolve_fork_snapshot(
    snapshots: WorkspaceSnapshotStore, fork_entry: object
) -> WorkspaceSnapshot | None:
    from_id = fork_entry.data.get("from_entry_id")  # type: ignore[attr-defined]
    if not isinstance(from_id, str):
        return None
    return snapshots.by_checkpoint(from_id)


def _dirty_guard(
    snapshots: WorkspaceSnapshotStore, cwd: str, forced: bool
) -> str | None:
    if forced:
        return None
    if snapshots.is_corrupt:
        return snapshots.corruption_message
    if git_repo_root(cwd) is None:
        return None
    if not snapshots.is_dirty(cwd):
        return None
    return (
        "workspace has uncommitted edits since the last snapshot; "
        "re-run with --force to overwrite"
    )


def _restore_workspace(
    snapshots: WorkspaceSnapshotStore,
    target: WorkspaceSnapshot | None,
    cwd: str,
    *,
    forced: bool = False,
) -> tuple[bool, str]:
    if target is None:
        return False, ""
    try:
        restored = snapshots.restore(target.id, cwd, force=forced)
    except WorkspaceSnapshotError as exc:
        return False, f"workspace restore failed: {exc}"
    if restored.mode != SNAPSHOT_MODE_GIT:
        return False, "no restorable workspace snapshot"
    return True, (
        f"workspace restored: {restored.file_count} files, "
        f"{_format_size(restored.size_bytes)}"
    )
