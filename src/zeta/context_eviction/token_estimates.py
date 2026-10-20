"""Stable token estimates used by context eviction."""

from math import ceil

from ..context_accounting import message_token_count
from ..protocol.types import Message


def estimated_text_tokens(text: str) -> int:
    """Use the project's stable token approximation for persisted text."""
    return max(1, ceil(len(text) / 4))


def estimated_tokens(message: Message) -> int:
    """Use the same stable approximation as normal context accounting."""
    return message_token_count(message)
