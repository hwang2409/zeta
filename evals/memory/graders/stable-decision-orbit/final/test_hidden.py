import json
from pathlib import Path


def _answer():
    return json.loads(Path("answer.json").read_text())


def test_action():
    assert (
        _answer().get("action")
        == {"action": "apply", "value": "SER-ORBIT-2V9K"}["action"]
    )


def test_value():
    assert (
        _answer().get("value")
        == {"action": "apply", "value": "SER-ORBIT-2V9K"}["value"]
    )
