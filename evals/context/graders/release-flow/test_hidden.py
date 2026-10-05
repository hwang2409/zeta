import pytest
from release_flow import run_release


class Sink:
    def __init__(self, fail=None):
        self.calls = []
        self.rolled = None
        self.fail = fail

    def apply(self, name, payload):
        if name == self.fail:
            raise RuntimeError("boom")
        self.calls.append((name, payload))

    def rollback(self, names):
        self.rolled = list(names)


RAW = {
    "steps": [
        {"name": "ship", "dependencies": ["build"], "payload": "s"},
        {"name": "audit", "payload": "a"},
        {"name": "build", "payload": "b"},
    ]
}


def test_plan_and_dry_run():
    sink = Sink()
    assert run_release(RAW, sink, dry_run=True) == ["audit", "build", "ship"]
    assert sink.calls == []


def test_execute_and_rollback():
    sink = Sink("ship")
    with pytest.raises(RuntimeError):
        run_release(RAW, sink)
    assert sink.calls == [("audit", "a"), ("build", "b")]
    assert sink.rolled == ["audit", "build"]


def test_rejects_bad_graph_before_effects():
    sink = Sink()
    with pytest.raises(ValueError):
        run_release({"steps": [{"name": "x", "dependencies": ["missing"]}]}, sink)
    assert sink.calls == []
    with pytest.raises(ValueError):
        run_release({"steps": [{"name": "x", "dependencies": ["x"]}]}, sink)
