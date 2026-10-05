from parcel_audit import summarize


def test_first_duplicate_wins_and_sorting():
    lines = [
        "t|same|zeta|2.5",
        "t|a|alpha|1",
        "t|same|zeta|900",
        "bad",
        "t|b|alpha|-0.25",
    ]
    assert summarize(lines) == "alpha=0.75\nzeta=2.5\n"


def test_malformed_and_empty():
    assert summarize(["x|y|z|nope", "", "too|short"]) == "\n"
