from .models import Rule


def decide(rules: list[Rule], subject: str) -> tuple[bool, str]:
    matches = [rule for rule in rules if subject.startswith(rule.prefix)]
    for rule in matches:
        if rule.effect == "deny":
            return False, rule.reason
    for rule in matches:
        if rule.effect == "allow":
            return True, rule.reason
    return False, "default deny"
