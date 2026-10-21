from zeta.agent.receipt import (
    MAX_AGENT_RESULT_BYTES,
    build_explicit_finish_receipt,
    encode_json,
)


def test_explicit_finish_multibyte_summary_fits_exact_envelope_limit() -> None:
    report = {
        "status": "done",
        "summary": "界" * 8000,
        "head_sha": "abcdef0",
        "verdict": "pass",
    }
    receipt = build_explicit_finish_receipt(
        report, {}, tool_call_id="finish-1", max_bytes=MAX_AGENT_RESULT_BYTES
    )
    summary = receipt["structuredContent"]["summary"]
    assert summary
    assert receipt["content"][0]["text"].startswith("done: " + summary)
    assert len(encode_json(receipt)) <= MAX_AGENT_RESULT_BYTES
    longer = {**report, "summary": summary + "界"}
    longer_receipt = build_explicit_finish_receipt(
        longer, {}, tool_call_id="finish-1", max_bytes=MAX_AGENT_RESULT_BYTES
    )
    assert longer_receipt["structuredContent"]["summary"] == summary
