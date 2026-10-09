import json
from pathlib import Path


def _answer():
    return json.loads(Path("answer.json").read_text())


def test_action():
    assert (
        _answer().get("action")
        == {"action": "run", "value": "PROC-QUARTZ-8N3F"}["action"]
    )


def test_value():
    assert (
        _answer().get("value")
        == {"action": "run", "value": "PROC-QUARTZ-8N3F"}["value"]
    )
