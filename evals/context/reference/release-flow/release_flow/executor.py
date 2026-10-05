def execute(steps, sink):
    completed = []
    try:
        for step in steps:
            sink.apply(step.name, step.payload)
            completed.append(step.name)
    except Exception:
        sink.rollback(completed)
        raise
    return completed
