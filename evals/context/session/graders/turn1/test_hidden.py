import pytest
from notebook_service import normalize


def test_slug_policy():
    assert normalize("  Héllo,   WORLD! ") == "hello-world"
    assert normalize("a---b") == "a-b"


def test_reserved():
    with pytest.raises(ValueError):
        normalize("ROOT")
    with pytest.raises(ValueError):
        normalize("!!!")
