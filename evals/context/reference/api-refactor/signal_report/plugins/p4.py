from signal_report import build_report


def run(records: dict[str, int]) -> str:
    return build_report(records, heading="Plugin 4")
