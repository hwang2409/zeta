from __future__ import annotations

import importlib
import subprocess
import sys


def test_importing_cli_does_not_load_harness_modules() -> None:
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import zeta.cli; "
                "assert 'zeta.core.context' not in sys.modules; "
                "assert 'zeta.core.store' not in sys.modules; "
                "assert 'zeta.protocol.types' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, check.stderr or check.stdout


def test_all_exports_resolve_and_cache() -> None:
    import zeta

    for name in zeta.__all__:
        value = getattr(zeta, name)
        assert value is getattr(zeta, name)


def test_stable_exports_keep_their_original_objects() -> None:
    import zeta

    modules = {
        "core.approval": {
            "ApprovalDecision",
            "ApprovalPolicy",
            "ApprovalRequest",
        },
        "core.context": {
            "AssembledContext",
            "BudgetExceeded",
            "CompactionPolicy",
            "ContextAssembler",
            "StaleBranchError",
            "SummaryCompletionError",
            "SummaryInputTooLarge",
        },
        "core.session": {
            "META_VERSION",
            "OpenedSession",
            "SessionError",
            "SessionManager",
            "SessionMetadata",
            "env_home",
            "find_most_recent",
            "list_sessions",
        },
        "core.store": {
            "ConversationEntry",
            "ConversationIntegrityError",
            "ConversationStore",
        },
        "protocol.types": {
            "CompletionBackend",
            "ContentBlock",
            "ContentType",
            "ErrorInfo",
            "ImageContent",
            "Message",
            "MessageRole",
            "RedactedThinkingContent",
            "StreamEvent",
            "StreamEventType",
            "TextContent",
            "ThinkingContent",
            "ToolCall",
            "ToolResult",
            "ToolUseContent",
        },
        "tools": {"AbortSignal", "ToolAbortSignal", "ToolDefinition", "ToolRegistry"},
        "providers.anthropic": {
            "AnthropicAuthError",
            "AnthropicBackend",
            "AnthropicBackendError",
            "AnthropicHTTPError",
            "AnthropicStreamError",
            "build_messages_payload",
        },
        "providers.codex": {"CodexBackend", "build_responses_payload"},
        "codex": {
            "DEFAULT_CODEX_MODEL",
            "CodexAuthError",
            "CodexBackendError",
            "CodexHTTPError",
            "CodexStreamError",
            "build_authorization_url",
            "exchange_authorization_code",
            "extract_account_id",
        },
        "oauth": {"OAuthTokens"},
        "runtime.loop": {"AgentLoop"},
    }
    expected = {
        name: getattr(importlib.import_module(f"zeta.{module}"), name)
        for module, names in modules.items()
        for name in names
    }
    assert set(expected) == set(zeta.__all__)
    assert all(getattr(zeta, name) is value for name, value in expected.items())

    script = (
        "import zeta; from zeta import *; "
        + "; ".join(f"assert {name} is zeta.{name}" for name in expected)
    )
    check = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, check.stderr or check.stdout


def test_public_interface_preserves_star_import_and_protocol_surface() -> None:
    import zeta

    stable_exports = (
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
        "AnthropicHTTPError",
        "AnthropicStreamError",
        "CodexAuthError",
        "CodexBackend",
        "CodexBackendError",
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
    )
    assert zeta.__all__ == list(stable_exports)

    protocol_names = {
        "ASSISTANT_RESPONSE_ABORTED",
        "ASSISTANT_RESPONSE_COMPLETED",
        "ASSISTANT_RESPONSE_FAILED",
        "ASSISTANT_RESPONSE_STATE",
        "ASSISTANT_RESPONSE_SYNTHETIC",
        "ContentBlock",
        "ContentType",
        "ContextWindowBackend",
        "CompletionBackend",
        "ErrorInfo",
        "FAILED_TURN_ERROR",
        "FAILED_TURN_MARKER",
        "ImageContent",
        "MESSAGE_ORIGIN_METADATA",
        "Message",
        "MessageOrigin",
        "MessageRole",
        "OwnedCompletionBackend",
        "RedactedThinkingBlock",
        "RedactedThinkingContent",
        "StreamEvent",
        "StreamEventType",
        "StructuredContentValue",
        "StructuredToolResult",
        "TextBlock",
        "TextContent",
        "ThinkingBlock",
        "ThinkingContent",
        "ToolAnnotations",
        "ToolBlobResource",
        "ToolBlobResourceBlock",
        "ToolCall",
        "ToolContentBlock",
        "ToolImageBlock",
        "ToolResourceBlock",
        "ToolResult",
        "ToolSchema",
        "ToolTextBlock",
        "ToolTextResource",
        "ToolTextResourceBlock",
        "ToolUseBlock",
        "ToolUseContent",
        "assistant_text",
        "content_from_dict",
        "flatten_tool_content",
        "harness_notice",
        "is_owned_completion_backend",
        "is_passive_harness_message",
        "is_visible_harness_notice",
        "require_new_message_origin",
        "user_message_for_turn",
        "validate_tool_content_block",
        "with_message_origin",
    }
    protocol = importlib.import_module("zeta.protocol.types")

    assert protocol_names <= set(dir(zeta))
    assert set(stable_exports) <= set(dir(zeta))
    for name in protocol_names:
        value = getattr(zeta, name)
        assert value is getattr(protocol, name)
        assert value is getattr(zeta, name)
