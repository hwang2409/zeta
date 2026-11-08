"""Custom agent harness."""

from importlib import import_module

_LAZY_EXPORTS = {
    "AbortSignal": (".tools", "AbortSignal"),
    "ApprovalDecision": (".core.approval", "ApprovalDecision"),
    "ApprovalPolicy": (".core.approval", "ApprovalPolicy"),
    "ApprovalRequest": (".core.approval", "ApprovalRequest"),
    "AssembledContext": (".core.context", "AssembledContext"),
    "BudgetExceeded": (".core.context", "BudgetExceeded"),
    "CompactionPolicy": (".core.context", "CompactionPolicy"),
    "ContextAssembler": (".core.context", "ContextAssembler"),
    "ConversationEntry": (".core.store", "ConversationEntry"),
    "ConversationIntegrityError": (".core.store", "ConversationIntegrityError"),
    "ConversationStore": (".core.store", "ConversationStore"),
    "META_VERSION": (".core.session", "META_VERSION"),
    "OpenedSession": (".core.session", "OpenedSession"),
    "SessionError": (".core.session", "SessionError"),
    "SessionManager": (".core.session", "SessionManager"),
    "SessionMetadata": (".core.session", "SessionMetadata"),
    "StaleBranchError": (".core.context", "StaleBranchError"),
    "SummaryCompletionError": (".core.context", "SummaryCompletionError"),
    "SummaryInputTooLarge": (".core.context", "SummaryInputTooLarge"),
    "ToolAbortSignal": (".tools", "ToolAbortSignal"),
    "ToolDefinition": (".tools", "ToolDefinition"),
    "ToolRegistry": (".tools", "ToolRegistry"),
    "env_home": (".core.session", "env_home"),
    "find_most_recent": (".core.session", "find_most_recent"),
    "list_sessions": (".core.session", "list_sessions"),
    "CompletionBackend": (".protocol.types", "CompletionBackend"),
    "ContentBlock": (".protocol.types", "ContentBlock"),
    "ContentType": (".protocol.types", "ContentType"),
    "ContextWindowBackend": (".protocol.types", "ContextWindowBackend"),
    "ErrorInfo": (".protocol.types", "ErrorInfo"),
    "ImageContent": (".protocol.types", "ImageContent"),
    "Message": (".protocol.types", "Message"),
    "MessageRole": (".protocol.types", "MessageRole"),
    "RedactedThinkingContent": (".protocol.types", "RedactedThinkingContent"),
    "StreamEvent": (".protocol.types", "StreamEvent"),
    "StreamEventType": (".protocol.types", "StreamEventType"),
    "TextContent": (".protocol.types", "TextContent"),
    "ThinkingContent": (".protocol.types", "ThinkingContent"),
    "ToolCall": (".protocol.types", "ToolCall"),
    "ToolResult": (".protocol.types", "ToolResult"),
    "ToolUseContent": (".protocol.types", "ToolUseContent"),
    "TextBlock": (".protocol.types", "TextBlock"),
    "ThinkingBlock": (".protocol.types", "ThinkingBlock"),
    "RedactedThinkingBlock": (".protocol.types", "RedactedThinkingBlock"),
    "ToolUseBlock": (".protocol.types", "ToolUseBlock"),
    "ToolContentBlock": (".protocol.types", "ToolContentBlock"),
    "StructuredContentValue": (".protocol.types", "StructuredContentValue"),
    "validate_tool_content_block": (".protocol.types", "validate_tool_content_block"),
    "ASSISTANT_RESPONSE_ABORTED": (".protocol.types", "ASSISTANT_RESPONSE_ABORTED"),
    "ASSISTANT_RESPONSE_COMPLETED": (".protocol.types", "ASSISTANT_RESPONSE_COMPLETED"),
    "ASSISTANT_RESPONSE_FAILED": (".protocol.types", "ASSISTANT_RESPONSE_FAILED"),
    "ASSISTANT_RESPONSE_STATE": (".protocol.types", "ASSISTANT_RESPONSE_STATE"),
    "ASSISTANT_RESPONSE_SYNTHETIC": (".protocol.types", "ASSISTANT_RESPONSE_SYNTHETIC"),
    "FAILED_TURN_ERROR": (".protocol.types", "FAILED_TURN_ERROR"),
    "FAILED_TURN_MARKER": (".protocol.types", "FAILED_TURN_MARKER"),
    "MESSAGE_ORIGIN_METADATA": (".protocol.types", "MESSAGE_ORIGIN_METADATA"),
    "MessageOrigin": (".protocol.types", "MessageOrigin"),
    "OwnedCompletionBackend": (".protocol.types", "OwnedCompletionBackend"),
    "StructuredToolResult": (".protocol.types", "StructuredToolResult"),
    "ToolAnnotations": (".protocol.types", "ToolAnnotations"),
    "ToolBlobResource": (".protocol.types", "ToolBlobResource"),
    "ToolBlobResourceBlock": (".protocol.types", "ToolBlobResourceBlock"),
    "ToolImageBlock": (".protocol.types", "ToolImageBlock"),
    "ToolResourceBlock": (".protocol.types", "ToolResourceBlock"),
    "ToolSchema": (".protocol.types", "ToolSchema"),
    "ToolTextBlock": (".protocol.types", "ToolTextBlock"),
    "ToolTextResource": (".protocol.types", "ToolTextResource"),
    "ToolTextResourceBlock": (".protocol.types", "ToolTextResourceBlock"),
    "assistant_text": (".protocol.types", "assistant_text"),
    "content_from_dict": (".protocol.types", "content_from_dict"),
    "flatten_tool_content": (".protocol.types", "flatten_tool_content"),
    "harness_notice": (".protocol.types", "harness_notice"),
    "is_owned_completion_backend": (".protocol.types", "is_owned_completion_backend"),
    "is_passive_harness_message": (".protocol.types", "is_passive_harness_message"),
    "is_visible_harness_notice": (".protocol.types", "is_visible_harness_notice"),
    "require_new_message_origin": (".protocol.types", "require_new_message_origin"),
    "user_message_for_turn": (".protocol.types", "user_message_for_turn"),
    "with_message_origin": (".protocol.types", "with_message_origin"),
    "AgentLoop": (".runtime.loop", "AgentLoop"),
    "AnthropicAuthError": (".providers.anthropic", "AnthropicAuthError"),
    "AnthropicBackend": (".providers.anthropic", "AnthropicBackend"),
    "AnthropicBackendError": (".providers.anthropic", "AnthropicBackendError"),
    "AnthropicHTTPError": (".providers.anthropic", "AnthropicHTTPError"),
    "AnthropicStreamError": (".providers.anthropic", "AnthropicStreamError"),
    "OAuthTokens": (".oauth", "OAuthTokens"),
    "build_messages_payload": (".providers.anthropic", "build_messages_payload"),
    "DEFAULT_CODEX_MODEL": (".codex", "DEFAULT_CODEX_MODEL"),
    "CodexAuthError": (".codex", "CodexAuthError"),
    "CodexBackend": (".providers.codex", "CodexBackend"),
    "CodexBackendError": (".codex", "CodexBackendError"),
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


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = list(_LAZY_EXPORTS)
