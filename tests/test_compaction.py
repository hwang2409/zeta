import json

from zeta.compaction import FALLBACK_SUMMARY_PREFIX, fallback_summary


def test_fallback_summary_strictly_enforces_output_bound() -> None:
    source = json.dumps(
        [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "tool_call": {
                            "id": "call-1",
                            "name": "x" * 5_000,
                            "arguments": {},
                        },
                    }
                ],
            }
        ]
    )

    short_summary = fallback_summary(source, max_chars=80)
    truncated_summary = fallback_summary(source, max_chars=100)
    field_bounded_summary = fallback_summary(source, max_chars=1_000)

    assert short_summary.startswith(FALLBACK_SUMMARY_PREFIX)
    assert len(short_summary) <= 80
    assert len(truncated_summary) == 100
    assert truncated_summary.endswith("[fallback summary truncated]")
    assert "tool: x" in truncated_summary
    assert "x" * 401 not in field_bounded_summary


def test_fallback_summary_keeps_bounded_tool_result_excerpts() -> None:
    source = json.dumps(
        [
            {
                "role": "tool_result",
                "content": [],
                "tool_result": {
                    "tool_call_id": "call-1",
                    "content": "UNIQUE-TOOL-RESULT",
                    "is_error": False,
                },
            },
            {
                "role": "tool_result",
                "content": [],
                "tool_result": {
                    "tool_call_id": "call-2",
                    "content": [
                        {"type": "text", "text": "HEAD-" + "x" * 5_000 + "-TAIL"}
                    ],
                    "is_error": True,
                },
            },
        ]
    )

    summary = fallback_summary(source, max_chars=2_000)

    assert summary.startswith(FALLBACK_SUMMARY_PREFIX)
    assert "UNIQUE-TOOL-RESULT" in summary
    assert "error" in summary
    assert "HEAD-" in summary and "-TAIL" in summary
    assert len(summary) <= 2_000
    assert fallback_summary(source, max_chars=2_000) == summary
