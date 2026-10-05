import importlib
import pathlib

import signal_report


def test_new_api_contract():
    assert signal_report.__all__ == ["build_report"]
    assert (
        signal_report.build_report({"b": 2, "a": 1}, heading="Exact Heading")
        == "Exact Heading\na: 1\nb: 2\n"
    )


def test_plugins_and_no_stale_name():
    for i in range(7):
        plugin = importlib.import_module(f"signal_report.plugins.p{i}")
        assert plugin.run({"x": i}).startswith(f"Plugin {i}\n")
    root = pathlib.Path(signal_report.__file__).parent
    assert "render_summary" not in "".join(p.read_text() for p in root.rglob("*.py"))
