def format_decision(subject, decision):
    allowed, reason = decision
    return f"{subject}|{'ALLOW' if allowed else 'DENY'}|{reason}\n"
