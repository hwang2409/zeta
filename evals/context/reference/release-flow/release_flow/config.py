from .models import Step


def parse(raw):
    if not isinstance(raw, dict) or not isinstance(raw.get("steps"), list):
        raise TypeError("steps must be a list")
    steps = []
    for item in raw["steps"]:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise TypeError("invalid step")
        deps = item.get("dependencies", [])
        if not isinstance(deps, list) or not all(isinstance(x, str) for x in deps):
            raise ValueError("invalid dependencies")
        steps.append(Step(item["name"], tuple(deps), str(item.get("payload", ""))))
    return steps
