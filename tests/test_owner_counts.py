from __future__ import annotations

import asyncio
import json
import threading

from tests.support.owner_counts import OwnerCounts


def _row(tmp_path, action):
    recorder = OwnerCounts(str(tmp_path / "counts.jsonl"))
    recorder.node_start("self")
    action(recorder)
    recorder.node_finish("self")
    return [
        json.loads(line)
        for line in (tmp_path / "counts.jsonl").read_text().splitlines()
    ]


def test_explicit_close_has_exact_row(tmp_path):
    def action(recorder):
        owner = object()
        recorder.constructed(owner, "Explicit")
        recorder.closed(owner)

    rows = _row(tmp_path, action)
    assert rows[0]["nodeid"] == "self"
    assert rows[0]["kind"] == "Explicit"
    assert rows[0]["constructed"] == 1
    assert rows[0]["explicit"] == 1
    assert rows[0]["forced"] == 0
    assert rows[0]["unclosed"] == 0


def test_intentionally_open_has_exact_row(tmp_path):
    def action(recorder):
        recorder.constructed(object(), "Open")

    rows = _row(tmp_path, action)
    assert rows[0]["kind"] == "Open"
    assert rows[0]["constructed"] == 1
    assert rows[0]["explicit"] == 0
    assert rows[0]["forced"] == 0
    assert rows[0]["unclosed"] == 1


def test_thread_open_close_has_exact_row(tmp_path):
    def action(recorder):
        def worker():
            owner = object()
            recorder.constructed(owner, "Thread")
            recorder.closed(owner)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(10)
        assert not thread.is_alive()

    rows = _row(tmp_path, action)
    assert rows[0]["kind"] == "Thread"
    assert rows[0]["constructed"] == 1
    assert rows[0]["explicit"] == 1
    assert rows[0]["forced"] == 0
    assert rows[0]["unclosed"] == 0


def test_async_close_has_exact_row(tmp_path):
    async def run(recorder):
        owner = object()
        recorder.constructed(owner, "Async")
        await asyncio.sleep(0)
        recorder.closed(owner)

    rows = _row(tmp_path, lambda recorder: asyncio.run(run(recorder)))
    assert rows[0]["kind"] == "Async"
    assert rows[0]["constructed"] == 1
    assert rows[0]["explicit"] == 1
    assert rows[0]["forced"] == 0
    assert rows[0]["unclosed"] == 0
