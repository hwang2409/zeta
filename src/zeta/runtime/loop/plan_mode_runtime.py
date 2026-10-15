"""Plan-mode state transitions for the agent loop."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ...agent.plan_mode import plan_mode_prompt
from ...protocol.types import ToolCall


class PlanModeRuntimeMixin:
    """Apply and remove the plan prompt and tool policy as one transition."""

    _plan_mode: bool
    _plan_mode_prior_prompt: Any
    _plan_mode_policy: Any
    _on_plan_mode_change: Callable[[bool], None] | None
    context_assembler: Any
    tool_registry: Any

    @property
    def plan_mode(self) -> bool:
        return self._plan_mode

    def set_plan_mode(self, enabled: bool) -> None:
        """Restrict the assistant to read-only tools, or lift the restriction.

        GPT-5.6 keeps tool schemas stable; other models advertise a subset.
        """

        if enabled == self._plan_mode:
            return
        assembler = self.context_assembler
        if enabled:
            self._plan_mode_prior_prompt = assembler.system_prompt
            assembler.system_prompt = plan_mode_prompt(assembler.system_prompt)
        elif self._plan_mode_prior_prompt is not None:
            assembler.system_prompt = self._plan_mode_prior_prompt
            self._plan_mode_prior_prompt = None
        self._plan_mode = enabled
        if self._on_plan_mode_change is not None:
            self._on_plan_mode_change(enabled)

    def plan_mode_allows(self, tool_call: ToolCall) -> bool:
        """Authorize one registry-resolved capability in the plan-mode layer."""

        return not self._plan_mode or self.tool_registry.call_allowed_by(
            tool_call, self._plan_mode_policy
        )


__all__ = ["PlanModeRuntimeMixin"]
