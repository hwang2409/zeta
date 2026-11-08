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


def test_protocol_exports_preserve_root_surface_and_discoverability() -> None:
    import zeta

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

    assert protocol_names <= set(zeta.__all__)
    assert protocol_names <= set(dir(zeta))
    for name in protocol_names:
        value = getattr(zeta, name)
        assert value is getattr(protocol, name)
        assert value is getattr(zeta, name)
