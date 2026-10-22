"""Revert unverified frontend client model choices without issuing a paid completion probe.

Our Claude/Codex adapters use a static catalog, not an account entitlement API.
Keep a durable fallback until the selected model completes its first response.
This uses the existing protocol 1.1 settings RPC and ordinary error events.
"""

import asyncio

from ..accounts import AccountId, account_provider, api_key_mode
from ..core.approval import ApprovalDecision
from ..core.session import SessionMetadata
from ..core.slash import resolve_session_budget
from ..models.catalog import provider_for_model
from ..protocol.types import ErrorInfo
from .runtime import ServerRuntime


async def apply(
    runtime: ServerRuntime, model: str, mode: str, *, reverting: bool = False
) -> None:
    loop = runtime.loop
    context = loop.account_context

    def prepare(metadata: SessionMetadata) -> SessionMetadata:
        fallback = metadata.model_fallback
        if reverting:
            provider, selected_model, budget = fallback
            selected_fallback = None
        else:
            provider = provider_for_model(model)
            selected_model = model
            budget, _ = resolve_session_budget(
                metadata.compaction_budget,
                metadata.budget_pinned,
                provider,
                selected_model,
                None,
            )
            selected_fallback = fallback
        if (
            not reverting
            and selected_model != metadata.model
            and selected_fallback is None
        ):
            selected_fallback = (
                metadata.provider,
                metadata.model,
                loop.context_assembler.token_budget,
            )
        if selected_fallback is not None and (
            provider,
            selected_model,
        ) == selected_fallback[:2]:
            selected_fallback = None
        account_provider_name = account_provider(provider)
        if (
            account_provider_name is not None
            and not api_key_mode(provider)
            and account_provider_name not in metadata.selected_accounts
        ):
            account = context.store.resolve_account(provider)
            if context.store.contains(account):
                metadata = runtime.manager._apply_account_selection(
                    metadata, account, boundary=0, switched=False
                )
        return runtime.manager._apply_session_settings(
            metadata,
            model=selected_model,
            approval_mode=mode,
            budget=budget,
            provider=provider,
            model_fallback=selected_fallback,
        )

    def build_backend(metadata: SessionMetadata):
        account = None
        selected = metadata.selected_accounts.get(account_provider(metadata.provider) or "")
        if selected is not None:
            account = AccountId(*selected.split(":", 1))
        return runtime.backend_for_model(
            metadata.provider,
            metadata.model,
            token_budget=metadata.compaction_budget,
            account=account,
        )

    authoritative = await context.switch_model(
        loop, prepare=prepare, build_backend=build_backend
    )
    expected_provider, expected_model = (
        runtime.metadata.model_fallback[:2]
        if reverting and runtime.metadata.model_fallback is not None
        else (provider_for_model(model), model)
    )
    if (
        authoritative.provider != expected_provider
        or authoritative.model != expected_model
        or authoritative.approval_mode != mode
    ):
        raise OSError("session settings were not persisted")
    approval_mode = authoritative.approval_mode or mode
    runtime.policy.default = ApprovalDecision(approval_mode)

async def confirm(runtime: ServerRuntime) -> None:
    metadata = runtime.metadata
    if metadata.model_fallback is not None:
        await asyncio.to_thread(
            runtime.manager.record_session_settings,
            metadata,
            model=metadata.model,
            approval_mode=runtime.policy.default.value,
            budget=metadata.compaction_budget,
            provider=metadata.provider,
        )


def entitlement_error(error: ErrorInfo) -> bool:
    return error.provider_error and (
        error.status_code in {400, 401, 403, 404}
        or error.code in {"auth_error", "model_not_found", "permission_denied"}
    )


async def recover_async(runtime: ServerRuntime, info: ErrorInfo) -> dict:
    error = {"code": info.code, "message": info.message}
    if not entitlement_error(info):
        return error
    error["code"] = "model_access_error"
    fallback = runtime.metadata.model_fallback
    if fallback is None:
        return error
    provider, model, _ = fallback
    try:
        await apply(runtime, model, runtime.policy.default.value, reverting=True)
    except (OSError, ValueError, RuntimeError) as exc:
        return {
            **error,
            "message": f"{error['message']}\n\nCould not restore {model}: {exc}. Open Settings to choose a model.",
        }
    return {
        "code": "model_reverted",
        "message": f"Model reverted to {model} ({provider}). Open Settings to choose another model.\n\n{error['message']}",
    }
