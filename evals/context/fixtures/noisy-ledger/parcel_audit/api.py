from collections.abc import Iterable

from .ledger import totals
from .parser import parse
from .report import render


def summarize(lines: Iterable[str]) -> str:
    return render(totals(event for line in lines if (event := parse(line)) is not None))
