import json
from pathlib import Path


def _answer():
    return json.loads(Path("answer.json").read_text())


def test_action():
    assert _answer().get("action") == "apply"


def test_value():
    assert _answer().get("value") == "PREF-GLOBAL-6T2W"
