import pytest
from notebook_service import Note, Notebook


def test_create_and_duplicate():
    book = Notebook()
    note = book.create("My Note", "body")
    assert note == Note("my-note", "My Note", "body")
    with pytest.raises(ValueError):
        book.create("my note", "again")


def test_reserved_still_applies():
    with pytest.raises(ValueError):
        Notebook().create("Root", "x")
