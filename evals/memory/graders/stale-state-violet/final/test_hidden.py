import json
from pathlib import Path


def _answer():
    return json.loads(Path("answer.json").read_text())


def test_action():
    assert (
        _answer().get("action")
        == {"action": "stop", "value": "DONE-VIOLET-8K2F"}["action"]
    )


def test_value():
    assert (
        _answer().get("value")
        == {"action": "stop", "value": "DONE-VIOLET-8K2F"}["value"]
    )
