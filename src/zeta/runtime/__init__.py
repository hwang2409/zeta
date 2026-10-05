"""Frontend-neutral runtime composition for zeta."""

from importlib import import_module

_LAZY_EXPORTS = {
    "DENIAL_MARKER": (".driver", "DENIAL_MARKER"),
    "TOOL_RESULT_MAX_BYTES": (".driver", "TOOL_RESULT_MAX_BYTES"),
    "RuntimeComposition": (".composition", "RuntimeComposition"),
    "build_unattended_loop": (".unattended", "build_unattended_loop"),
    "compose_runtime": (".composition", "compose_runtime"),
    "drive_turn": (".driver", "drive_turn"),
}


def __getattr__(name: str):
    """Load runtime modules only when a caller asks for an export."""

    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


__all__ = list(_LAZY_EXPORTS)
