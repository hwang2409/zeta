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
