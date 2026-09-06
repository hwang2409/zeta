"""Checkpoint commands and transcript rebuilding for the terminal UI."""

from __future__ import annotations

from datetime import UTC, datetime

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
        try:
            entry = self.loop.store.append_checkpoint(args.strip() or None)
        except ValueError as exc:
            return f"checkpoint failed: {exc}"
        label = entry.data["label"]
        snapshots = self._snapshots()
        try:
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
        if self.active or self.loop.store.turn_in_flight():
            return "fork unavailable while a turn is running"
        if self.loop.background_children_running:
            return "fork unavailable while background agents are running"
        selector, forced = _parse_force(args)
        selector = selector.strip()
        if not selector:
            checkpoints = self.loop.store.list_checkpoints()
            if not checkpoints:
                return "no checkpoints on the active branch; use /checkpoint [label]"
            lines = ["checkpoints on the active branch:"]
            snapshots = self._snapshots()
            for entry, preview in checkpoints:
                next_message = preview or "(no message after checkpoint)"
                workspace_snapshot = snapshots.by_checkpoint(entry.id)
                snap_note = _snapshot_note(workspace_snapshot)
                lines.append(
                    f"seq {entry.seq} · {entry.data['label']} · "
                    f"{_age(entry.data['created_at'])} · {snap_note} · {next_message}"
                )
            return "\n".join(lines)
        snapshots = self._snapshots()
        dirty_warning = _dirty_guard(snapshots, self.loop.store.bash_cwd, forced)
        if dirty_warning is not None:
            return dirty_warning
        try:
            entry = self.loop.store.append_fork(selector)
        except ValueError as exc:
            return f"fork failed: {exc}"
        target_snapshot = _resolve_fork_snapshot(snapshots, entry)
        restore_note = _restore_workspace(
            snapshots, target_snapshot, self.loop.store.bash_cwd
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
        target = (
            snapshots.undo_target() if verb == "undo" else snapshots.redo_target()
        )
        if target is None:
            direction = "earlier" if verb == "undo" else "later"
            return f"{verb} unavailable: no {direction} workspace snapshot"
        dirty_warning = _dirty_guard(snapshots, self.loop.store.bash_cwd, forced)
        if dirty_warning is not None:
            return dirty_warning
        restore_note = _restore_workspace(snapshots, target, self.loop.store.bash_cwd)
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
                self._print_system(
                    f"forked to checkpoint '{entry.data['label']}' at seq "
                    f"{entry.data['from_seq']}"
                )
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
    if git_repo_root(cwd) is None:
        return None
    # Gate on "any restorable snapshot exists" so a corrupt or missing
    # current_id in the state file cannot silently bypass the confirm.
    if not snapshots.has_restorable_snapshot():
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
) -> str:
    if target is None:
        return ""
    try:
        restored = snapshots.restore(target.id, cwd)
    except WorkspaceSnapshotError as exc:
        return f"workspace unchanged: {exc}"
    if restored.mode != SNAPSHOT_MODE_GIT:
        return ""
    return (
        f"workspace restored: {restored.file_count} files, "
        f"{_format_size(restored.size_bytes)}"
    )
