"""test-only readers for the store's public row-reference interface."""

import copy

from zeta.core.checkpoints import ConversationEntry
from zeta.core.store import ConversationStore
from zeta.protocol.types import Message


def branch_entries(store: ConversationStore) -> list[ConversationEntry]:
    return [copy.deepcopy(store.entry(ref)) for ref in store.branch_refs()]


def branch_messages(store: ConversationStore) -> list[Message]:
    return [
        Message.from_dict(entry.data["message"])
        for entry in branch_entries(store)
        if entry.type == "message"
    ]
