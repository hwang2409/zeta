from pathlib import Path

from scripts.lint_cancel_ownership import HISTORICAL_PROOF, findings


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
