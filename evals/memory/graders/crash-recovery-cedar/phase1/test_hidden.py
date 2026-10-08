from pathlib import Path


def test_setup_did_not_answer_early():
    assert not Path("answer.json").exists()
