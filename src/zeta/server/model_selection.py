"""Revert unverified GUI model choices without issuing a paid completion probe.

Our Claude/Codex adapters use a static catalog, not an account entitlement API.
Keep a durable fallback until the selected model completes its first response.
This uses the existing protocol 1.1 settings RPC and ordinary error events.
"""

from ..core.approval import ApprovalDecision
from ..core.slash import resolve_session_budget
from ..model_catalog import provider_for_model
from ..types import ErrorInfo
from .runtime import ServerRuntime


def apply(
    runtime: ServerRuntime, model: str, mode: str, *, reverting: bool = False
) -> None:
    fallback = runtime.metadata.model_fallback
    if reverting:
        provider, model, budget = fallback
        fallback = None
    else:
        provider = "fake" if runtime.fake_catalog else provider_for_model(model)
        budget, _ = resolve_session_budget(
            runtime.metadata.compaction_budget,
            runtime.metadata.budget_pinned,
            provider,
            model,
            None,
        )
    previous = runtime.model
    loop = runtime.loop
    previous_backend = loop.backend
    backend = (
        runtime.backend_for_model(provider, model)
        if provider != runtime.provider
        else previous_backend
    )
    assembler = loop.context_assembler
    previous_budget = assembler.token_budget
    if (
        not reverting
        and not runtime.fake_catalog
        and model != previous
        and fallback is None
    ):
        fallback = (runtime.provider, previous, previous_budget)
    if fallback is not None and (provider, model) == fallback[:2]:
        fallback = None
    try:
        loop.backend = backend
        loop.set_model(model)
        assembler.token_budget = budget
        runtime.manager.record_session_settings(
            runtime.metadata,
            model=model,
            approval_mode=mode,
            budget=budget,
            provider=provider,
            model_fallback=fallback,
        )
    except Exception:
        assembler.token_budget = previous_budget
        loop.backend = previous_backend
        loop.set_model(previous)
        raise
    assembler.backend = backend
    assembler.compaction_policy.backend = backend
    runtime.policy.default = ApprovalDecision(mode)


def confirm(runtime: ServerRuntime) -> None:
    metadata = runtime.metadata
    if metadata.model_fallback is not None:
        runtime.manager.record_session_settings(
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


def recover(runtime: ServerRuntime, info: ErrorInfo) -> dict:
    error = {"code": info.code, "message": info.message}
    if not entitlement_error(info):
        return error
    error["code"] = "model_access_error"
    fallback = runtime.metadata.model_fallback
    if fallback is None:
        return error
    provider, model, _ = fallback
    try:
        apply(runtime, model, runtime.policy.default.value, reverting=True)
    except (OSError, ValueError, RuntimeError) as exc:
        return {
            **error,
            "message": f"{error['message']}\n\nCould not restore {model}: {exc}. Open Settings to choose a model.",
        }
    return {
        "code": "model_reverted",
        "message": f"Model reverted to {model} ({provider}). Open Settings to choose another model.\n\n{error['message']}",
    }
