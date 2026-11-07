from pathlib import Path

from scripts.lint_cancel_ownership import HISTORICAL_PROOF, RATCHET, findings


def test_cancel_ownership_lint_flags_single_shield_reawait() -> None:
    source = """async def own():
    worker = make_task()
    try:
        await asyncio.shield(worker)
    except asyncio.CancelledError:
        await asyncio.shield(worker)
"""

    result = findings(Path("src/example.py"), source)

    assert result == [(5, "src/example.py:own:worker")]
    assert HISTORICAL_PROOF == "ad1cfd43:accounts/session.py:_run_locked_apply:worker"


def test_cancel_ownership_lint_accepts_looped_owner() -> None:
    source = """async def own():
    worker = make_task()
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancelled = True
    return worker.result()
"""

    assert findings(Path("src/example.py"), source) == []


def test_cancel_ownership_lint_ratchet_matches_every_triaged_site() -> None:
    observed = {
        key
        for path in Path("src").rglob("*.py")
        for _line, key in findings(path, path.read_text(encoding="utf-8"))
    }

    assert observed == RATCHET


def test_cancel_ownership_lint_preserves_historical_account_session_proof() -> None:
    source = """async def _run_locked_apply():
    worker = make_task()
    try:
        await asyncio.shield(worker)
    except asyncio.CancelledError:
        await asyncio.shield(worker)
"""

    result = findings(Path("accounts/session.py"), source)

    assert result == [(5, "accounts/session.py:_run_locked_apply:worker")]
    assert HISTORICAL_PROOF == f"ad1cfd43:{result[0][1]}"
