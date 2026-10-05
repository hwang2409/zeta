import pytest
from notebook_service import Notebook


def test_round_trip():
    book = Notebook()
    book.create("B", "β")
    book.create("A", "a")
    text = book.export()
    assert Notebook.load(text).export() == text


def test_rejects_invalid_records():
    bad = [
        '{"slug":"a","title":"A","body":"x","extra":1}\n',
        '{"slug":"A","title":"A","body":"x"}\n',
        '{"slug":"root","title":"Root","body":"x"}\n',
        '{"slug":"a","title":"A","body":"x"}\n{"slug":"a","title":"Again","body":"y"}\n',
    ]
    for text in bad:
        with pytest.raises(ValueError):
            Notebook.load(text)


def test_prior_constraints_remain():
    with pytest.raises(ValueError):
        Notebook().create("root", "x")
