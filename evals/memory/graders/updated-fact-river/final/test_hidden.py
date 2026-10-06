import json
from pathlib import Path


def _answer():
    return json.loads(Path("answer.json").read_text())


def test_action():
    assert (
        _answer().get("action")
        == {"action": "use", "value": "ENDPOINT-RIVER-Y-7T5B"}["action"]
    )


def test_value():
    assert (
        _answer().get("value")
        == {"action": "use", "value": "ENDPOINT-RIVER-Y-7T5B"}["value"]
    )
