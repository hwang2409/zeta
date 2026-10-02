from gate_policy import Rule, decide


def test_deny_overrides_earlier_allow():
    rules = [
        Rule("team/", "allow", "broad"),
        Rule("team/red", "deny", "blocked"),
        Rule("team/", "deny", "later deny"),
    ]
    assert decide(rules, "team/red/1") == (False, "blocked")


def test_first_within_effect_and_defaults():
    assert decide(
        [Rule("a", "allow", "first"), Rule("a", "allow", "second")], "abc"
    ) == (True, "first")
    assert decide([], "none") == (False, "default deny")
