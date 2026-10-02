from pathlib import Path

from event_export import Event, export_events

EXPECTED = "from dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass Event:\n    id: str\n    type: str\n    payload: dict[str, object]\n"


def test_exact_ndjson():
    events = [Event("1", "café", {"z": 1}), Event("2", "done", {})]
    assert (
        export_events(events)
        == '{"id":"1","type":"café","payload":{"z":1}}\n{"id":"2","type":"done","payload":{}}\n'
    )
    assert export_events([]) == ""


def test_frozen_schema_file():
    import event_export.schema

    assert Path(event_export.schema.__file__).read_text() == EXPECTED
