from pathlib import Path

from notebook_service import Notebook

EXPECTED = "from dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass Note:\n    slug: str\n    title: str\n    body: str\n"


def test_exact_export():
    book = Notebook()
    book.create("Zed", "café")
    book.create("Alpha", "x")
    assert (
        book.export()
        == '{"slug":"alpha","title":"Alpha","body":"x"}\n{"slug":"zed","title":"Zed","body":"café"}\n'
    )


def test_empty():
    assert Notebook().export() == ""


def test_models_unchanged():
    import notebook_service.models

    assert Path(notebook_service.models.__file__).read_text() == EXPECTED
