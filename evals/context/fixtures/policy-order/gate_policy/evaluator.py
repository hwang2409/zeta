from .models import Rule


def decide(rules: list[Rule], subject: str) -> tuple[bool, str]:
    for rule in rules:
        if subject.startswith(rule.prefix):
            return rule.effect == "allow", rule.reason
    return False, "default deny"
