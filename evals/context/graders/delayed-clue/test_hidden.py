import clockqueue.scheduler as module
from clockqueue import Job, Scheduler


def test_injected_millisecond_clock():
    scheduler = Scheduler(now=lambda: 10_999)
    scheduler.add(Job("fresh", 10_000, 1_000))
    assert scheduler.ready() == []
    scheduler = Scheduler(now=lambda: 11_000)
    job = Job("ready", 10_000, 1_000)
    scheduler.add(job)
    assert scheduler.ready() == [job]


def test_default_uses_monotonic_milliseconds(monkeypatch):
    monkeypatch.setattr(module.time, "monotonic_ns", lambda: 12_345_000_000)
    scheduler = Scheduler()
    fresh = Job("fresh", 12_000, 346)
    ready = Job("ready", 12_000, 345)
    scheduler.add(fresh)
    scheduler.add(ready)
    assert scheduler.ready() == [ready]
