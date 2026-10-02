from .config import parse
from .executor import execute
from .planner import plan


def run_release(raw, sink, *, dry_run=False):
    steps = plan(parse(raw))
    return [step.name for step in steps] if dry_run else execute(steps, sink)
