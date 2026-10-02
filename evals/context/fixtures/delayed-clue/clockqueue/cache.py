class Cache:
    """Red herring: cache ordering is unrelated to readiness."""

    def __init__(self):
        self._data = {}

    def put(self, job):
        self._data[job.key] = job

    def values(self):
        return list(self._data.values())
