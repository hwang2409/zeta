"""Validate the immutable spawn inputs kept in a recoverable child marker."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..storage_format import ACCEPTED_KEY_SETS
from .presets import AgentPreset, valid_max_turns


def preset_snapshot(preset: AgentPreset) -> dict[str, Any]:
    return {
        key: preset.tools if key == "tool_names" else getattr(preset, key)
        for key in ACCEPTED_KEY_SETS["child_resume_preset"]
    }


def restore_preset(data: dict[str, Any]) -> AgentPreset:
    return AgentPreset(
        **{**data, "tool_names": None if data["tool_names"] is None else frozenset(data["tool_names"])},
        selection_guidance="",
    )


def validate_child_resume(value: object) -> None:
    def require(valid: bool) -> None:
        if not valid:
            raise ValueError("invalid child resume descriptor")

    require(type(value) is dict and set(value) == ACCEPTED_KEY_SETS["child_resume"])
    assert isinstance(value, dict)
    preset = value["preset"]
    require(type(preset) is dict and set(preset) == ACCEPTED_KEY_SETS["child_resume_preset"])
    for key in ("name", "source", "preamble", "prompt_suffix"):
        require(type(preset[key]) is str and (bool(preset[key]) or key in {"preamble", "prompt_suffix"}))
    for key in ("allow_delegation", "accepts_follow_ups", "explicit_finish"):
        require(type(preset[key]) is bool)
    names = preset["tool_names"]
    require(names is None or type(names) is list and all(type(name) is str and name for name in names))
    require(preset["max_child_depth"] is None or type(preset["max_child_depth"]) is int and preset["max_child_depth"] >= 0)
    for turns in (preset["max_turns"], value["requested_max_turns"], value["effective_max_turns"]):
        require(turns is None or valid_max_turns(turns))
    for key in ("model", "provider", "cwd", "description"):
        require(type(value[key]) is str and bool(value[key]))
    require(value["requested_model"] is None or type(value["requested_model"]) is str and bool(value["requested_model"]))
    require(type(value["depth"]) is int and value["depth"] > 0)
    require(Path(value["cwd"]).is_absolute())
    require(value["background"] is True)
    require(type(value["follow_up"]) is bool and type(value["explicit_finish"]) is bool)
    phase = value["account_pins"]
    pins = value["selected_accounts"]
    require(phase in ("pending", "selected") and type(pins) is dict)
    require(phase != "pending" or not pins)
    require(all(type(key) is str and bool(key) and type(pin) is str and pin.startswith(key + ":") and len(pin) > len(key) + 1 for key, pin in pins.items()))
