def render_summary(records: dict[str, int]) -> str:
    return "\n".join(f"{key}: {records[key]}" for key in sorted(records)) + "\n"
