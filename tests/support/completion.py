"""Explicit completion controls for tests that call providers directly."""

from zeta.providers.completion import (
    CompletionControl,
    CompletionRetryBudget,
    ResetPolicy,
)


def completion_control(
    *, reset: ResetPolicy = ResetPolicy.NO_RESET
) -> CompletionControl:
    return CompletionControl(
        abort=None,
        reset=reset,
        budget=CompletionRetryBudget(),
    )
