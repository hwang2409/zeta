from signal_report import render_summary


def run(records: dict[str, int]) -> str:
    return render_summary(records)
