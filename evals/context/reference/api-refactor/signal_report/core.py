def build_report(records: dict[str, int], *, heading: str) -> str:
    body = [heading, *(f"{key}: {records[key]}" for key in sorted(records))]
    return "\n".join(body) + "\n"
