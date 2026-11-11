"""Approval lifecycle and action handling for the submission pipeline.

These handlers run on the single pipeline consumer and read approval state
off-loop through the store's async approval index. They are a mixin so the
pipeline class stays within the module-size limit while sharing its state.
"""

from __future__ import annotations

import asyncio

from ..core.approval import ApprovalDecision, ApprovalRequest
from .messages import _ApprovalAction, _ApprovalLifecycle, _DurableToolDone
from .model import SubmissionState, _Entry


class ApprovalPipelineMixin:
    """Consumer-side approval handlers for ``SubmissionPipeline``."""

    async def _on_approval_lifecycle(
        self, message: _ApprovalLifecycle
    ) -> None:
        entry = self._entry_for(message.submission)
        if entry is None:
            return
        if message.kind == "approval_start":
            self._approval_owners[message.call.id] = entry.submission
            if entry.state is SubmissionState.PREPROCESSING:
                entry.state = SubmissionState.AWAITING_APPROVAL
            await self.refresh_pending_approvals()
            self._host._present_pending_approvals()
            return
        self._approval_owners = {
            key: owner
            for key, owner in self._approval_owners.items()
            if owner is not entry.submission or key != message.call.id
        }
        if entry.state is SubmissionState.AWAITING_APPROVAL:
            entry.state = SubmissionState.PREPROCESSING
        await self.refresh_pending_approvals()
        self._host._present_pending_approvals()

    async def refresh_pending_approvals(self) -> None:
        policy = self._host._approval_policy
        self._pending_approvals = (
            ()
            if policy is None
            else tuple(await policy.pending_requests_for_display())
        )

    async def _on_approval_action(self, message: _ApprovalAction) -> None:
        policy = self._host._approval_policy
        if policy is None:
            self._ack_action(message)
            return
        await self.refresh_pending_approvals()
        request = self._find_request(message.requested_key)
        if request is None:
            self._host._print_system("approval · no matching request")
            self._ack_action(message)
            return
        owner = self._approval_owners.get(request.key)
        durable = (
            owner is None
            and isinstance(request.key, str)
            and await self._host.loop.store.approval_state(request.request_id)
            == (request.tool_call, None)
        )
        if message.decision is ApprovalDecision.ALLOW:
            if message.always:
                try:
                    rule = policy.remember_allow(request)
                except ValueError as exc:
                    self._host._print_system(
                        f"approval · cannot always allow {request.key}: {exc}"
                    )
                    self._ack_action(message)
                    return
            policy.approve(request.key)
            verb = (
                f"always allowed {rule} for this session · request" if message.always else "approved"
            )
        else:
            policy.deny(request.key)
            verb = "denied"
            entry = self._entry_for(owner)
            if entry is not None:
                entry.denied = True
                entry.state = SubmissionState.DENIED
        self._host._print_system(f"approval · {verb} {request.key}")
        await self.refresh_pending_approvals()
        self._host._present_pending_approvals()
        self._host._invalidate_prompt()
        if durable and self._host.loop.prepare_resume_pending_tool(
            request.tool_call
        ):
            task = asyncio.create_task(
                self._host.loop.resume_pending_tool(
                    request.request_id,
                    prepared=True,
                    event_sink=self._host._handle_tool_event,
                )
            )
            self._durable_tasks[request.request_id] = task
            self._durable_tool_calls[request.request_id] = request.tool_call
            task.add_done_callback(
                lambda completed, request_id=request.request_id: self._send(
                    _DurableToolDone(request_id, completed)
                )
            )
        self._ack_action(message)

    def _find_request(self, requested_key: str | None) -> ApprovalRequest | None:
        requests = self.pending_approvals
        if requested_key is None:
            return requests[0] if requests else None
        return next(
            (request for request in requests if str(request.key) == requested_key),
            None,
        )

    def _abort_approvals(self, entry: _Entry) -> None:
        policy = self._host._approval_policy
        if policy is None:
            return
        # Cancellation is a decision: use the strict pending view and record a
        # canceled result only when this abort actually won.
        for request in self._pending_approvals:
            if self._approval_owners.get(request.key) is not entry.submission:
                continue
            if not policy.abort(request.key):
                continue
            if entry is self._provider_entry and isinstance(request.key, str):
                self._host.loop.finalize_canceled(request.tool_call)
