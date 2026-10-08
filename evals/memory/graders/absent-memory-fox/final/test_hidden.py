import json
from pathlib import Path


def _answer():
    return json.loads(Path("answer.json").read_text())


def test_action():
    assert _answer().get("action") == {"action": "abstain", "value": None}["action"]


def test_value():
    assert _answer().get("value") == {"action": "abstain", "value": None}["value"]
