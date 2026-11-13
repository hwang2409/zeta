"""Slash-command handlers for the full-screen terminal UI.

Extracted from ``tui.app`` so the composition root stays under the module line
cap. ``TUIApp`` mixes these in, so they run against its attributes.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
from typing import Any

from prompt_toolkit.enums import EditingMode

from ...accounts import (
    AccountError,
    AccountId,
    AccountListSnapshot,
    AccountUsage,
    account_provider,
)
from ...accounts.login import login_failure_message, login_named_account
from ...accounts.store import rename_account_command
from ...config.settings import load_settings
from ...core.approval import ApprovalDecision
from ...core.commands.plan_commands import run_implement_command, run_plan_command
from ...core.commands.slash_registry import SlashLock, waiting_message
from ...core.project_context import project_command_view
from ...core.session import (
    SessionError,
    env_home,
    normalize_session_name,
    persist_metadata_change,
)
from ...core.slash import SlashStatus, SlashStatusSnapshot, build_slash_status
from ...core.transcript_recovery.runtime import (
    TranscriptRepairRequiresUpgrade,
    repair_transcript_for_runtime,
)
from ...memory.user_authorization import MemoryMutationAuthorization
from ...model_input import ModelInputEnvelope
from ...project_inbox import InboxError, ProjectInbox
from ...project_memory_commands import run_memory_command
from ...project_registry import ProjectRegistryError
from ...protocol.types import StreamEvent, StreamEventType
from ...providers.factory import build_backend
from ...runtime.status import capture_slash_status
from ...session_retention import (
    disarm_retention,
    read_retention_history,
    retention_arm_warning,
    retention_authority,
)
from ...session_retention.pin import set_retention_pin
from ...tools._shared.user_discovery import trust_project_tools
from .. import theme as _theme
from ..models import known_models, match_models, validate_model_name
from ..render import render_event
from .accounts_picker import AccountsPicker
from .model_picker import ModelPicker
from .permissions import format_permissions, revoke_permissions


def _validate_model_name(provider: str, model: str) -> None:
    validate_model_name(provider, model)


class SlashHandlerMixin:
    """Serve the session-state slash commands the registry dispatches."""

    @functools.cached_property
    def slash_lock(self) -> SlashLock:
        """The TUI waits visibly behind the running command."""

        return SlashLock(
            contended=lambda holder: self._print_system(waiting_message(holder)),
            waiting=lambda holder: self._print_system(waiting_message(holder)),
            late_output=lambda output: (
                self._handle_slash_output(output) if isinstance(output, str) else None
            ),
        )

    async def slash_approve(self, args: str) -> str:
        """Approve the first pending request, or the request with this key."""

        await self._resolve_approval(ApprovalDecision.ALLOW, args.strip() or None)
        return ""

    async def slash_always(self, args: str) -> str:
        """Allow this tool/action, with any arguments, for the running session."""

        await self._resolve_approval(
            ApprovalDecision.ALLOW, args.strip() or None, always=True
        )
        return ""

    def slash_permissions(self, args: str) -> str:
        """Inspect the active policy or explicitly revoke session grants."""

        registry = self.loop.tool_registry
        if args.strip():
            return revoke_permissions(registry.approval_policy, args)
        return format_permissions(
            registry.approval_policy,
            registry.tool_policy,
            settings_path=(self._zeta_home or env_home()) / "settings.toml",
            plan_mode=self.loop.plan_mode,
        )

    async def slash_deny(self, args: str) -> str:
        """Deny the first pending request, or the request with this key."""

        await self._resolve_approval(ApprovalDecision.DENY, args.strip() or None)
        return ""

    async def _resolve_approval(
        self,
        decision: ApprovalDecision,
        requested_key: str | None,
        *,
        always: bool = False,
    ) -> None:
        controller = self._fork_controller
        if controller is None:
            await self.resolve_local_approval(decision, requested_key, always=always)
            return
        await controller.resolve_approval(decision, requested_key, always=always)

    def slash_status_input(self, args: str) -> tuple[bool, SlashStatusSnapshot]:
        card = not args.strip() and self._full_screen_active()
        if card:
            self._consume_overlay_submission()
        return card, self._status_snapshot()

    def slash_status(self) -> SlashStatus:
        return build_slash_status(self._status_snapshot())

    def _status_snapshot(self) -> SlashStatusSnapshot:
        pending = tuple(
            f"{request.key} ({request.label or request.tool_call.name})"
            for request in self.pending_approvals
        )
        viewed_path = self._viewed_session_path()
        if viewed_path == self.loop.store.session_dir:
            viewed_health = self.loop.project_context_health
        else:
            viewed_loop = self.loop._background_owner.project_context_loops.get(
                viewed_path.name
            )
            viewed_health = (
                viewed_loop.project_context_health if viewed_loop is not None else None
            )
        return capture_slash_status(
            self.loop,
            session_id=self.loop.store.session_id,
            provider=self.provider,
            model=self.model,
            context_files=self._context_files,
            vim_mode=self.vim_mode,
            pending_approvals=pending,
            usage_tracker=self._usage_tracker,
            project_context_health=(
                viewed_health.summary if viewed_health is not None else None
            ),
            transcript_session_id=viewed_path.name,
        )

    def model_choices(self) -> tuple[str, ...]:
        """Models the picker and the composer completer offer for this provider."""

        return known_models(self.provider, self._model_catalog)

    @property
    def _picker_active(self) -> bool:
        return self.accounts_picker_active or self.model_picker_active or self.remote_picker_active

    def _picker_move(self, delta: int) -> None:
        if self.accounts_picker_active:
            self.accounts_picker_move(delta)
        elif self.model_picker_active:
            self.model_picker_move(delta)
        else:
            self.remote_picker_move(delta)

    def _picker_select(self) -> None:
        if self.accounts_picker_active:
            self.accounts_picker_select()
        elif self.model_picker_active:
            self.model_picker_select()
        else:
            self.remote_picker_select()

    def _picker_cancel(self) -> None:
        if self.accounts_picker_active:
            self.accounts_picker_cancel()
        elif self.model_picker_active:
            self.model_picker_cancel()
        else:
            self.remote_picker_cancel()

    @property
    def model_picker_active(self) -> bool:
        return self._model_picker is not None

    def _show_accounts_picker(self, snapshot: AccountListSnapshot) -> str:
        displays = snapshot.displays
        self._dismiss_accounts_picker()
        picker = AccountsPicker(
            tuple(item.account for item in displays),
            snapshot.selected,
            snapshot.defaults,
            {item.account: item.detail for item in displays if item.detail},
            {item.account: item.label for item in displays},
            {item.account: item.usage for item in displays},
            {item.account for item in displays if not item.available},
            self._accounts_switching_unavailable(),
            background_notice=self._accounts_background_notice(),
        )
        self._accounts_picker = picker
        self._accounts_picker_unit = self._presenter.print_unit(picker.render())
        self._invalidate_prompt()
        return ""

    async def _login_account(self, requested: list[str]) -> str:
        unavailable = self._accounts_switching_unavailable()
        if unavailable is not None:
            return f"account unchanged: {unavailable}"
        context = getattr(self.loop, "account_context", None)
        if context is None:
            return "accounts unavailable for this session"
        if len(requested) != 3 or requested[1] not in {"anthropic", "codex"}:
            return "usage: /accounts login <anthropic|codex> <label>"
        try:
            await asyncio.to_thread(
                login_named_account, context.home, requested[1], requested[2]
            )
        except Exception as exc:
            return login_failure_message(exc)
        return f"logged in as {requested[1]}:{requested[2].lower()}"

    @property
    def account_switch_pending(self) -> bool:
        return getattr(self, "_account_switch_pending", False)

    async def wait_for_account_switch(self) -> None:
        await self._account_switch_finished.wait()

    def _accounts_switching_unavailable(self) -> str | None:
        if getattr(self, "active", False) or getattr(self, "pending_approvals", ()):
            return "switching waits until the turn ends"
        context = self.loop.account_context
        return context.switching_unavailable

    def _accounts_background_notice(self) -> str | None:
        if getattr(self.loop, "background_children_running", False):
            return "running children keep their account"
        return None

    def _block_account_mutation(self) -> bool:
        picker = self._accounts_picker
        if picker is None:
            return True
        unavailable = self._accounts_switching_unavailable()
        if unavailable is None:
            return False
        picker.switching_unavailable = unavailable
        self._repaint_accounts_picker()
        return True

    async def slash_accounts(self, args: str) -> str:
        """List named accounts and open the current-session switcher."""

        context = getattr(self.loop, "account_context", None)
        requested = args.split()
        if requested and requested[0] == "login":
            return await self._login_account(requested)
        if requested:
            unavailable = self._accounts_switching_unavailable()
            if unavailable is not None:
                return f"account unchanged: {unavailable}"
            if context is None:
                return "accounts unavailable for this session"
            return await asyncio.to_thread(rename_account_command, context.store, args)
        if context is None:
            return "accounts unavailable for this session"
        snapshot = await context.list_snapshot()
        if not snapshot.displays:
            unavailable = self._accounts_switching_unavailable()
            return unavailable or "no accounts; run zeta login <provider> --account <name>"
        self._show_accounts_picker(snapshot)
        return ""

    def _handle_account_limit(self, event: StreamEvent) -> asyncio.Task[None]:
        context = self.loop.account_context

        async def show() -> None:
            snapshot = await context.list_snapshot()
            alternative = event.data.get("alternative")
            if event.data.get("action") == "switch" and isinstance(alternative, str):
                snapshot = AccountListSnapshot(
                    tuple(
                        item
                        for item in snapshot.displays
                        if str(item.account) == alternative
                    ),
                    snapshot.selected,
                    snapshot.defaults,
                )
            self._show_accounts_picker(snapshot)

        return asyncio.create_task(show())

    @property
    def accounts_picker_active(self) -> bool:
        return self._accounts_picker is not None

    def accounts_picker_move(self, delta: int) -> None:
        picker = self._accounts_picker
        if picker is None:
            return
        picker.move(delta)
        self._repaint_accounts_picker()

    def _repaint_accounts_picker(self) -> None:
        picker = self._accounts_picker
        if picker is None:
            return
        picker.switching_unavailable = self._accounts_switching_unavailable()
        picker.background_notice = self._accounts_background_notice()
        unit = self._accounts_picker_unit
        if unit is not None:
            self._accounts_picker_unit = self._transcript.replace(unit, picker.render())
        self._invalidate_prompt()

    def accounts_picker_refresh(self) -> None:
        picker = self._accounts_picker
        if picker is None or picker.refreshing:
            return
        picker.refreshing = True
        self._repaint_accounts_picker()

        async def refresh() -> None:
            try:
                refreshed = await asyncio.to_thread(
                    lambda: asyncio.run(
                        AccountUsage(self.loop.account_context.store).refresh_stale()
                    )
                )
            except (AccountError, OSError, RuntimeError, ValueError):
                current = self._accounts_picker
                if current is not None:
                    current.refreshing = False
                    self._repaint_accounts_picker()
                self._print_system("account usage refresh failed")
                return
            current = self._accounts_picker
            if current is None:
                return
            current.refreshing = False
            displays = await asyncio.to_thread(
                self.loop.account_context.store.list_display
            )
            current.usages = {item.account: item.usage for item in displays}
            current.usages.update(refreshed)
            self._repaint_accounts_picker()

        asyncio.create_task(refresh())

    def accounts_picker_default(self) -> asyncio.Task[None] | None:
        if self._block_account_mutation():
            return None
        picker = self._accounts_picker
        if picker is None:
            return None
        account = picker.highlighted

        async def set_default() -> None:
            if self._block_account_mutation():
                return
            try:
                await asyncio.to_thread(
                    self.loop.account_context.store.set_default, account
                )
            except (AccountError, OSError, ValueError) as exc:
                self._print_system(f"default unchanged: {exc}")
                return
            picker.defaults[account.provider] = account
            self._repaint_accounts_picker()

        return asyncio.create_task(set_default())

    def accounts_picker_login(self) -> None:
        if self._block_account_mutation():
            return
        picker = self._accounts_picker
        session = self._active_session
        if picker is None or session is None:
            return
        provider = picker.highlighted.provider
        self._dismiss_accounts_picker()
        session.default_buffer.reset()
        session.default_buffer.insert_text(f"/accounts login {provider} ")
        self._invalidate_prompt()

    def accounts_picker_logout(self) -> asyncio.Task[None] | None:
        if self._block_account_mutation():
            return None
        picker = self._accounts_picker
        if picker is None:
            return None
        account = picker.highlighted
        if picker.logout_confirmation != account:
            picker.logout_confirmation = account
            self._repaint_accounts_picker()
            return None

        async def logout() -> None:
            if self._block_account_mutation():
                return
            try:
                message = await asyncio.to_thread(
                    self.loop.account_context.store.logout,
                    account,
                    sessions_root=self.loop.account_context.home / "sessions",
                )
            except (AccountError, OSError, ValueError) as exc:
                picker.logout_confirmation = None
                self._repaint_accounts_picker()
                self._print_system(f"account unchanged: {exc}")
                return
            snapshot = await self.loop.account_context.list_snapshot()
            if snapshot.displays:
                self._show_accounts_picker(snapshot)
            else:
                self._dismiss_accounts_picker()
            self._print_system(message)

        return asyncio.create_task(logout())

    def accounts_picker_rename(self) -> None:
        if self._block_account_mutation():
            return
        picker = self._accounts_picker
        session = self._active_session
        if picker is None or session is None:
            return
        account = picker.highlighted
        self._dismiss_accounts_picker()
        session.default_buffer.reset()
        session.default_buffer.insert_text(f"/accounts rename {account} ")
        self._invalidate_prompt()

    def accounts_picker_select(self) -> asyncio.Task[None] | None:
        blocker = getattr(self, "_block_account_mutation", None)
        if blocker is not None and blocker():
            return None
        picker = self._accounts_picker
        current_task = getattr(self, "_account_switch_task", None)
        if picker is None or (current_task is not None and not current_task.done()):
            return None
        account = picker.highlighted
        context = self.loop.account_context

        current_provider = account_provider(self.provider)

        def prepare_backend(selected: AccountId) -> Any:
            if selected.provider != current_provider:
                return None
            backend, _ = build_backend(
                self.provider,
                self.model,
                home=context.home,
                account=selected,
                token_budget=self.loop.context_assembler.token_budget,
            )
            return backend

        async def switch() -> None:
            switch_account = getattr(self, "_switch_account", None)
            if switch_account is None:
                await SlashHandlerMixin._switch_account(
                    self, account, context, prepare_backend
                )
            else:
                await switch_account(account, context, prepare_backend)

        self._account_switch_pending = True
        finished = getattr(self, "_account_switch_finished", None)
        if finished is not None:
            finished.clear()
        task = asyncio.create_task(switch())
        self._account_switch_task = task
        done = getattr(self, "_account_switch_done", None)
        if done is not None:
            task.add_done_callback(done)
        return task

    async def _switch_account(
        self, account: AccountId, context: Any, prepare_backend: Any
    ) -> None:
        picker = self._accounts_picker
        if picker is None:
            return
        if getattr(self, "active", False) or getattr(self, "pending_approvals", ()):
            unavailable = "switching waits until the turn ends"
        else:
            unavailable = getattr(context, "switching_unavailable", None)
        if unavailable is not None:
            picker.switching_unavailable = unavailable
            self._repaint_accounts_picker()
            return
        try:
            await context.switch_active(
                self.loop,
                account,
                boundary=self.loop.store.active_branch_head_sequence(),
                provider=self.provider,
                build_backend=prepare_backend,
            )
        except (AccountError, OSError, RuntimeError, ValueError) as exc:
            self._print_system(f"account unchanged: {exc}")
            return
        finally:
            self._account_switch_task = None
        self._dismiss_accounts_picker()
        if context.account_state_unknown:
            self._presenter.print_unit(
                render_event(StreamEvent(StreamEventType.ACCOUNT_STATE_UNKNOWN))
            )
        else:
            self._print_system(f"account: {account}")

    def _account_switch_done(self, task: asyncio.Task[None]) -> None:
        del task
        self._account_switch_pending = False
        finished = getattr(self, "_account_switch_finished", None)
        if finished is not None:
            finished.set()
        invalidate = getattr(self, "_invalidate_prompt", None)
        if invalidate is not None:
            invalidate()

    def accounts_picker_cancel(self) -> None:
        self._dismiss_accounts_picker()
        self._invalidate_prompt()

    def _dismiss_accounts_picker(self) -> None:
        unit = self._accounts_picker_unit
        self._accounts_picker = None
        self._accounts_picker_unit = None
        if unit is not None:
            self._transcript.remove(unit, leading_blank=True)

    async def slash_pin(self, args: str) -> str:
        if args.strip():
            return "usage: /pin"
        return await asyncio.to_thread(
            set_retention_pin,
            self._zeta_home or env_home(),
            self.loop.store.session_id,
            self.loop.store._storage_lease,
            pinned=True,
        )

    async def slash_unpin(self, args: str) -> str:
        if args.strip():
            return "usage: /unpin"
        return await asyncio.to_thread(
            set_retention_pin,
            self._zeta_home or env_home(),
            self.loop.store.session_id,
            self.loop.store._storage_lease,
            pinned=False,
        )

    async def slash_retention(self, args: str) -> str:
        command = args.strip()
        home = self._zeta_home or env_home()
        if command == "":
            history = await asyncio.to_thread(read_retention_history, home)
            return "retention: none" if not history else f"retention: {history[-1]}"
        if command == "disarm":
            return await asyncio.to_thread(disarm_retention, home, self.loop.store._storage_lease)
        if command == "arm":
            days = load_settings(home=home).settings.retention_days
            result = await asyncio.to_thread(
                retention_authority, home, self.loop.store._storage_lease,
                days=days, zeta_version="unknown",
            )
            return f"{result}\n{retention_arm_warning()}" if result.startswith("retention arm") else result
        return "usage: /retention [arm|disarm]"

    async def slash_model(self, args: str) -> str:
        """Pick a model from a card, or switch straight to a named one."""

        requested = args.strip()
        self._start_model_catalog_load()
        busy = bool(self.active or self.pending_approvals)
        if not requested:
            if busy:
                return (
                    f"model: {self.model} (cannot change model while a turn or "
                    "approval is active)"
                )
            return self._open_model_picker(self.model_choices(), query="")
        if busy:
            return "model unchanged: cannot change model while a turn or approval is active"
        choices = self.model_choices()
        if requested not in choices:
            matches = match_models(requested, choices)
            if len(matches) == 1:
                requested = matches[0]
            elif matches:
                return self._open_model_picker(matches, query=requested)
        return await self._switch_model(requested)

    def _open_model_picker(self, choices: tuple[str, ...], *, query: str) -> str:
        """Draw the picker card; its keys take over while the composer is empty."""

        self._dismiss_model_picker()
        picker = ModelPicker(self.provider, choices, self.model, query)
        self._model_picker = picker
        self._model_picker_unit = self._presenter.print_unit(picker.render())
        self._invalidate_prompt()
        return ""

    def _repaint_model_picker(self) -> None:
        picker = self._model_picker
        if picker is None:
            return
        unit = self._model_picker_unit
        if unit is not None:
            self._model_picker_unit = self._transcript.replace(unit, picker.render())
        self._invalidate_prompt()

    def _dismiss_model_picker(self) -> None:
        unit = self._model_picker_unit
        self._model_picker = None
        self._model_picker_unit = None
        if unit is not None:
            self._transcript.remove(unit, leading_blank=True)

    def model_picker_move(self, delta: int) -> None:
        picker = self._model_picker
        if picker is None:
            return
        picker.move(delta)
        self._repaint_model_picker()

    def model_picker_select(self) -> None:
        """Run the highlighted choice through the normal ``/model`` path."""

        picker = self._model_picker
        if picker is None:
            return
        choice = picker.selected
        self._dismiss_model_picker()
        self._submit_input(f"/model {choice}")

    def model_picker_cancel(self) -> None:
        if self._model_picker is None:
            return
        self._dismiss_model_picker()
        self._invalidate_prompt()

    def refresh_model_picker(self) -> None:
        """Fold a freshly loaded catalog into an open picker."""

        picker = self._model_picker
        if picker is None:
            return
        choices = self.model_choices()
        if picker.query:
            choices = match_models(picker.query, choices)
        if not choices or choices == picker.choices:
            return
        self._model_picker = picker.with_choices(choices)
        self._repaint_model_picker()

    async def _switch_model(self, model: str) -> str:
        try:
            _validate_model_name(self.provider, model)
        except ValueError as exc:
            return f"model unchanged: {exc}"
        if self._model_catalog is None:
            catalog_warning = (
                f"model catalog unavailable for {self.provider} — using anyway"
            )
        elif model not in self._model_catalog:
            catalog_warning = (
                f"model not found in {self.provider} catalog — using anyway"
            )
        else:
            catalog_warning = None
        if model == self.model:
            return f"model: {model}"
        if self._on_model_change is None:
            return "model unchanged: model transition is unavailable"
        previous_budget = self.loop.context_assembler.token_budget
        try:
            authoritative = await self._on_model_change(model)
        except Exception as exc:
            return f"model unchanged: {exc}"
        self.model = authoritative.model
        current_budget = self.loop.context_assembler.token_budget
        budget_note = (
            f"context budget {previous_budget:,} -> {current_budget:,}"
            if current_budget != previous_budget
            else None
        )
        notes = [note for note in (catalog_warning, budget_note) if note]
        if notes:
            return f"model: {self.model} ({'; '.join(notes)})"
        return f"model: {self.model}"

    @property
    def plan_mode(self) -> bool:
        override = getattr(self, "_slash_plan_mode", None)
        if override is not None:
            return override
        return self.loop.plan_mode

    @plan_mode.setter
    def plan_mode(self, enabled: bool) -> None:
        self._slash_plan_mode = enabled

    def set_plan_mode(self, enabled: bool):
        if self.loop is self:
            return super().set_plan_mode(enabled)
        return self.loop.set_plan_mode(enabled)

    def invalidate_prompt(self) -> None:
        self._invalidate_prompt()

    def plan_mode_background_blocker(self) -> str | None:
        return self._plan_mode_background_blocker()

    async def slash_plan(self, args: str) -> str | ModelInputEnvelope:
        return await run_plan_command(self, args)

    async def slash_implement(self, args: str) -> str | ModelInputEnvelope:
        return await run_implement_command(self, args)

    def _plan_mode_background_blocker(self) -> str | None:
        work = tuple(getattr(self.loop, "background_work_descriptions", ()))
        if not work:
            return None
        return (
            "plan mode unchanged: background work is active: "
            f"{', '.join(work)}; stop it or wait"
        )

    def toggle_plan_mode(self) -> None:
        """Toggle plan mode from the keyboard, reporting the same notice."""

        async def toggle() -> None:
            self._print_system(await self.slash_plan("toggle"))

        asyncio.create_task(toggle())

    def slash_tools(self, args: str) -> str:
        """List registered tools or trust pending project tools."""

        requested = args.strip().lower()
        discovery = self._external_tools
        pending = discovery.pending_project_tools if discovery is not None else ()
        if requested in {"", "list"}:
            names = sorted(self.loop.tool_registry.registered_names)
            summary = f"tools: {len(names)} registered"
            if names:
                summary = f"{summary} ({', '.join(names)})"
            policy = self.loop.tool_registry.tool_policy
            if policy.allow_layers:
                layers = [", ".join(layer) or "none" for layer in policy.allow_layers]
                summary += f"; allow: {' AND '.join(layers)}"
            if policy.deny:
                summary += f"; deny: {', '.join(policy.deny)}"
            if pending:
                waiting = ", ".join(sorted({tool.module_stem for tool in pending}))
                summary = (
                    f"{summary}; untrusted project tools waiting: {waiting}; "
                    "run /tools trust to enable"
                )
            return summary
        if requested != "trust":
            return "tools: use /tools or /tools trust"
        if self.active or self.pending_approvals:
            return (
                "tools unchanged: cannot trust project tools while a turn or "
                "approval is active"
            )
        if discovery is None or not pending:
            return "tools: no project tools waiting for trust"
        notices, trusted = trust_project_tools(self.loop.tool_registry, discovery)
        self.loop.tool_schemas = tuple(self.loop.tool_registry.schemas)
        for notice in notices:
            self._print_system(notice)
        names = sorted({tool.module_stem for tool in trusted})
        return f"tools: trusted {len(names)} project tool(s): {', '.join(names)}"

    @property
    def new_session_requested(self) -> bool:
        return self._new_session_requested

    def request_new_session(self) -> None:
        self._new_session_requested = True
        self.request_exit()

    @property
    def restart_has_running_work(self) -> bool:
        return (
            self.loop.background_children_running
            or self.loop.tool_registry.background_tasks.running_count > 0
        )

    @property
    def composer_empty(self) -> bool:
        session = self._active_session
        application = None if session is None else getattr(session, "app", None)
        buffer = getattr(application, "current_buffer", None)
        return buffer is None or not buffer.text

    @property
    def restart_pending(self) -> bool:
        return self._fork_controller is not None and self._fork_controller.restart_pending

    def _maybe_restart_when_idle(self) -> None:
        if self._fork_controller is not None:
            self._fork_controller.notify_restart_idle_check()

    def slash_restart(self, args: str) -> str:
        requested = args.strip()
        if requested not in {"", "now", "cancel"}:
            return "restart unchanged: use /restart, /restart now, or /restart cancel"
        if self._ephemeral_root is not None:
            return "restart unavailable: this session is not persistent"
        if self._fork_controller is None:
            return "restart unavailable: no TUI session controller"
        if requested == "cancel":
            self._fork_controller.cancel_restart()
            return "restart canceled"
        if requested == "" and self._fork_controller.restart_has_running_work:
            self._fork_controller.request_restart_when_idle()
            return "restart pending · waiting for running work · /restart now to cancel it"
        self._fork_controller.request_restart()
        return "restarting from the local checkout..."

    def slash_new(self, args: str) -> str:
        """Signal the CLI wrapper to start a fresh session in this window."""

        if args.strip():
            return "new unchanged: /new does not accept arguments"
        if self.active or self.pending_approvals:
            return (
                "new unchanged: cannot start a new session while a turn or "
                "approval is active"
            )
        self.request_new_session()
        return "starting a fresh session..."

    async def slash_name(self, args: str) -> str:
        """Persist a session label shown in the resume picker."""

        raw = args.strip()
        if not raw:
            current = getattr(self, "_session_name", "")
            return f"session name: {current}" if current else "session name: (unnamed)"
        if self._on_name_change is None:
            return "session name unchanged: names are unavailable for this session"
        try:
            label = normalize_session_name(raw)
        except SessionError as exc:
            return f"session name unchanged: {exc}"
        try:
            await self._on_name_change(label)
        except SessionError as exc:
            return f"session name unchanged: {exc}"
        self._session_name = label
        return f"session name: {label}"

    async def slash_project(self, args: str) -> str:
        registry = self.loop.project_registry
        if registry is None:
            return "project registry unavailable"
        initialize = args.strip() == "init"
        try:
            view = await asyncio.to_thread(
                project_command_view,
                registry,
                self.loop.store.cwd,
                self.loop.session_metadata.project_id,
                initialize=initialize,
            )
        except (ProjectRegistryError, OSError, SessionError) as exc:
            if initialize:
                return f"could not initialize project: {exc}"
            raise
        if initialize:
            assert view.project is not None
            await persist_metadata_change(
                self.loop.manager,
                self.loop.session_metadata,
                lambda candidate: self.loop.manager.associate_project(
                    candidate, view.project.project_id
                ),
            )
            self.loop._set_runtime_project(self.loop.session_metadata)
        return view.text

    async def slash_memory(self, args: str) -> str:
        """Show, retry, or undo versioned automatic project-memory changes."""
        registry = self.loop.project_registry
        project_id = self.loop.session_metadata.project_id
        if registry is None or project_id is None:
            return "memory: no associated project"
        result = await asyncio.to_thread(
            run_memory_command,
            registry,
            project_id,
            args,
            getattr(self.loop, "memory_reconciler", None),
            MemoryMutationAuthorization.direct_slash(),
        )
        return await result if inspect.isawaitable(result) else result

    async def slash_repair_transcript(self, args: str) -> str:
        """Repair one corrupt root transcript after explicit confirmation."""
        parts = args.split()
        if len(parts) not in {2, 3} or parts[-1] != "confirm":
            return "usage: /repair-transcript <session-id> [<child-run-id>] confirm"
        child_run_id = parts[1] if len(parts) == 3 else None
        try:
            result = await repair_transcript_for_runtime(
                self.loop.manager,
                parts[0],
                child_run_id,
                parent_store=self.loop.store,
                memory_reconciler=getattr(self.loop, "memory_reconciler", None),
            )
        except TranscriptRepairRequiresUpgrade as exc:
            return str(exc)
        return (
            f"repaired transcript {parts[0]}"
            + (f" child {child_run_id}" if child_run_id is not None else "")
            + "; quarantine "
            f"{result.quarantine_id}; retained through sequence "
            f"{result.retained_last_sequence}"
        )

    async def slash_inbox(self, args: str) -> str:
        registry = self.loop.project_registry
        project_id = self.loop.session_metadata.project_id
        if registry is None or project_id is None:
            return "project inbox unavailable: session is not associated with a project"
        try:
            state = await asyncio.to_thread(
                ProjectInbox(
                    registry,
                    sessions_root=registry.root.parent / "sessions",
                ).command,
                project_id,
                args,
                from_session=self.loop.session_metadata.session_id,
            )
        except (InboxError, OSError) as exc:
            return f"project inbox unavailable: {exc}"
        return state

    def slash_theme(self, args: str) -> str:
        """Show, list, or switch the active TUI theme."""

        home = getattr(self, "_zeta_home", None)
        requested = args.strip()
        if not requested or requested.lower() == "list":
            available = ", ".join(_theme.list_available(home))
            return f"theme: {_theme.active_palette().name} · available: {available}"
        palette, notice = _theme.resolve_palette(requested, home=home)
        if palette is None:
            if notice is not None:
                return f"theme unchanged: {notice}"
            available = ", ".join(_theme.list_available(home))
            return (
                f"theme unchanged: unknown theme {requested!r}; available: {available}"
            )
        _theme.set_active_palette(palette)
        self._on_theme_change()
        message = f"theme: {palette.name}"
        return message if notice is None else f"{message} ({notice})"

    def _on_theme_change(self) -> None:
        """Refresh the prompt style cache and re-render the transcript."""

        # Cached DynamicStyle values were built from the previous palette; drop
        # them so the next _prompt_style() call rebuilds against the new one.
        prompt_styles = getattr(self, "_prompt_styles", None)
        if isinstance(prompt_styles, dict):
            prompt_styles.clear()
        self._rebuild_transcript()
        self._invalidate_prompt()

    async def slash_vim(self, args: str) -> str:
        requested = args.strip().lower()
        if not args:
            return f"vim mode: {'on' if self.vim_mode else 'off'}"
        if requested not in {"on", "off", "toggle"}:
            return "vim mode unchanged: use /vim on, /vim off, or /vim toggle"
        enabled = not self.vim_mode if requested == "toggle" else requested == "on"
        was_enabled = self.vim_mode
        if self._on_vim_mode_change is not None:
            await self._on_vim_mode_change(enabled)
        self.vim_mode = enabled
        if (session := self._active_session or self._session) is not None:
            if was_enabled and not enabled:
                session.app.output.reset_cursor_shape()
                session.app.output.flush()
            session.editing_mode = EditingMode.VI if enabled else EditingMode.EMACS
            session.app.vi_state.reset()
        self._invalidate_prompt()
        return f"vim mode: {'on' if self.vim_mode else 'off'}"


__all__ = ["SlashHandlerMixin"]
