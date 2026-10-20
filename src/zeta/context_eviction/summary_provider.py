from __future__ import annotations

from collections.abc import Callable

from ..protocol.types import CompletionBackend
from .summaries import SummaryProvider, SummaryResponse

BackendFactory = Callable[[], CompletionBackend]


def build_summary_provider(
    backend_or_factory: CompletionBackend | BackendFactory,
) -> SummaryProvider:
    """Build the one-dispatch provider used by every eviction-summary caller."""
    backend = None if callable(backend_or_factory) else backend_or_factory

    async def invoke(prompt: str) -> SummaryResponse:
        nonlocal backend
        from ..memory.provider import complete_reconciliation, use_response_byte_limit
        from ..providers.retry_policy import ProviderRetryBudget, use_retry_budget

        if backend is None:
            backend = backend_or_factory()
        with (
            use_response_byte_limit(512),
            use_retry_budget(ProviderRetryBudget(max_attempts=1)),
        ):
            response = await complete_reconciliation(backend, prompt)
        return SummaryResponse(
            response.text,
            input_tokens=sum(
                int(response.usage.get(key, 0) or 0)
                for key in (
                    "input_tokens",
                    "cache_read_input_tokens",
                    "cache_creation_input_tokens",
                )
            ),
            output_tokens=int(response.usage.get("output_tokens", 0) or 0),
        )

    return invoke
