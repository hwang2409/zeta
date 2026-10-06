import json
from pathlib import Path


def _answer():
    return json.loads(Path("answer.json").read_text())


def test_action():
    assert (
        _answer().get("action")
        == {"action": "avoid", "value": "TRY-EMBER-NEW-4H8S"}["action"]
    )


def test_value():
    assert (
        _answer().get("value")
        == {"action": "avoid", "value": "TRY-EMBER-NEW-4H8S"}["value"]
    )
