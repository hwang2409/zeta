"""Custom agent harness."""

from importlib import import_module

from .core.approval import ApprovalDecision, ApprovalPolicy, ApprovalRequest
from .core.context import (
    AssembledContext,
    BudgetExceeded,
    CompactionPolicy,
    ContextAssembler,
    StaleBranchError,
    SummaryCompletionError,
    SummaryInputTooLarge,
)
from .core.session import (
    META_VERSION,
    OpenedSession,
    SessionError,
    SessionManager,
    SessionMetadata,
    env_home,
    find_most_recent,
    list_sessions,
)
from .core.store import ConversationEntry, ConversationIntegrityError, ConversationStore
from .tools import AbortSignal, ToolAbortSignal, ToolDefinition, ToolRegistry
from .protocol.types import *

_LAZY_EXPORTS = {
    "AgentLoop": (".runtime.loop", "AgentLoop"),
    "AnthropicAuthError": (".providers.anthropic", "AnthropicAuthError"),
    "AnthropicBackend": (".providers.anthropic", "AnthropicBackend"),
    "AnthropicBackendError": (".providers.anthropic", "AnthropicBackendError"),
    "AnthropicCredentialStore": (".providers.anthropic", "AnthropicCredentialStore"),
    "AnthropicHTTPError": (".providers.anthropic", "AnthropicHTTPError"),
    "AnthropicStreamError": (".providers.anthropic", "AnthropicStreamError"),
    "OAuthTokens": (".oauth", "OAuthTokens"),
    "build_messages_payload": (".providers.anthropic", "build_messages_payload"),
    "DEFAULT_CODEX_MODEL": (".codex", "DEFAULT_CODEX_MODEL"),
    "CodexAuthError": (".codex", "CodexAuthError"),
    "CodexBackend": (".providers.codex", "CodexBackend"),
    "CodexBackendError": (".codex", "CodexBackendError"),
    "CodexCredentialStore": (".codex", "CodexCredentialStore"),
    "CodexHTTPError": (".codex", "CodexHTTPError"),
    "CodexStreamError": (".codex", "CodexStreamError"),
    "build_authorization_url": (".codex", "build_authorization_url"),
    "build_responses_payload": (".providers.codex", "build_responses_payload"),
    "exchange_authorization_code": (".codex", "exchange_authorization_code"),
    "extract_account_id": (".codex", "extract_account_id"),
}


def __getattr__(name: str):
    """Load provider exports only when a caller asks for one."""

    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


__all__ = [
    "DEFAULT_CODEX_MODEL",
    "AgentLoop",
    "ApprovalDecision",
    "ApprovalPolicy",
    "ApprovalRequest",
    "AssembledContext",
    "AbortSignal",
    "AnthropicAuthError",
    "AnthropicBackend",
    "AnthropicBackendError",
    "AnthropicCredentialStore",
    "AnthropicHTTPError",
    "AnthropicStreamError",
    "CodexAuthError",
    "CodexBackend",
    "CodexBackendError",
    "CodexCredentialStore",
    "CodexHTTPError",
    "CodexStreamError",
    "BudgetExceeded",
    "CompactionPolicy",
    "CompletionBackend",
    "ContentBlock",
    "ContentType",
    "ConversationEntry",
    "ConversationIntegrityError",
    "ConversationStore",
    "ContextAssembler",
    "ErrorInfo",
    "ImageContent",
    "Message",
    "MessageRole",
    "META_VERSION",
    "OpenedSession",
    "OAuthTokens",
    "RedactedThinkingContent",
    "SessionError",
    "SessionManager",
    "SessionMetadata",
    "StaleBranchError",
    "SummaryCompletionError",
    "SummaryInputTooLarge",
    "StreamEvent",
    "StreamEventType",
    "TextContent",
    "ThinkingContent",
    "ToolAbortSignal",
    "ToolCall",
    "ToolDefinition",
    "ToolRegistry",
    "ToolResult",
    "ToolUseContent",
    "build_authorization_url",
    "build_messages_payload",
    "build_responses_payload",
    "exchange_authorization_code",
    "env_home",
    "extract_account_id",
    "find_most_recent",
    "list_sessions",
]
