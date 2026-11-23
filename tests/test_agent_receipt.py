from zeta.agent.receipt import (
    MAX_AGENT_RESULT_BYTES,
    build_explicit_finish_receipt,
    encode_json,
)


def test_explicit_finish_multibyte_summary_fits_exact_envelope_limit() -> None:
    report = {
        "status": "done",
        "summary": "界 " * 100 + "界",
        "head_sha": "abcdef0",
        "verdict": "pass",
    }
    full = build_explicit_finish_receipt(report, {}, child_instance_id="child-1")
    exact_limit = len(encode_json(full))
    assert exact_limit < MAX_AGENT_RESULT_BYTES
    assert build_explicit_finish_receipt(
        report, {}, child_instance_id="child-1", max_bytes=exact_limit
    )["structuredContent"]["summary"] == report["summary"]

    limited = build_explicit_finish_receipt(
        report, {}, child_instance_id="child-1", max_bytes=exact_limit - 1
    )
    summary = limited["structuredContent"]["summary"]
    assert "[truncated]" in summary
    assert limited["content"][0]["text"] == "done: " + summary
    assert len(encode_json(limited)) < exact_limit
