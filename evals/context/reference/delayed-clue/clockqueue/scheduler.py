import time

from .cache import Cache


class Scheduler:
    def __init__(self, now=None):
        self.cache = Cache()
        self.now = now or (lambda: time.monotonic_ns() // 1_000_000)

    def add(self, job):
        self.cache.put(job)

    def ready(self):
        current = self.now()
        return [
            job for job in self.cache.values() if current >= job.created_at + job.ttl
        ]
