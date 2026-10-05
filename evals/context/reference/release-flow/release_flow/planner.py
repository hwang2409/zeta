def plan(steps):
    by_name = {}
    for step in steps:
        if not step.name or step.name in by_name:
            raise ValueError("duplicate step")
        by_name[step.name] = step
    if any(dep not in by_name for step in steps for dep in step.dependencies):
        raise ValueError("unknown dependency")
    result = []
    remaining = set(by_name)
    while remaining:
        ready = sorted(
            name
            for name in remaining
            if set(by_name[name].dependencies) <= {x.name for x in result}
        )
        if not ready:
            raise ValueError("dependency cycle")
        for name in ready:
            result.append(by_name[name])
            remaining.remove(name)
    return result
