from decimal import Decimal


def render(values: dict[str, Decimal]) -> str:
    return "\n".join(f"{key}={values[key]}" for key in sorted(values)) + "\n"
