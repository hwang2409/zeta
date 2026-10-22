"""Plan-mode state transitions for the agent loop."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from ...agent.plan_mode import plan_mode_prompt
from ...protocol.types import ToolCall
from .skill_refresh import RuntimePrompt


class PlanModeRuntimeMixin:
    """Apply and remove the plan prompt and tool policy as one transition."""

    _plan_mode: bool
    _plan_mode_prior_prompt: Any
    _plan_mode_policy: Any
    _on_plan_mode_change: Callable[[bool], Awaitable[None] | None] | None
    context_assembler: Any
    tool_registry: Any

    @property
    def plan_mode(self) -> bool:
        return self._plan_mode

    def set_plan_mode(self, enabled: bool) -> Awaitable[None] | None:
        """Restrict the assistant to read-only tools, or lift the restriction.

        GPT-5.6 keeps tool schemas stable; other models advertise a subset.
        """

        if enabled == self._plan_mode:
            return None
        assembler = self.context_assembler
        runtime_prompt = getattr(self, "_runtime_prompt", None)
        if enabled:
            if runtime_prompt is not None:
                self._plan_mode_prior_prompt = runtime_prompt
                self._set_runtime_prompt(
                    RuntimePrompt(
                        plan_mode_prompt(runtime_prompt.system_prompt),
                        runtime_prompt.prompt_components,
                    )
                )
            else:
                self._plan_mode_prior_prompt = assembler.system_prompt
                assembler.system_prompt = plan_mode_prompt(assembler.system_prompt)
        elif self._plan_mode_prior_prompt is not None:
            if runtime_prompt is not None:
                self._set_runtime_prompt(self._plan_mode_prior_prompt)
            else:
                assembler.system_prompt = self._plan_mode_prior_prompt
            self._plan_mode_prior_prompt = None
        self._plan_mode = enabled
        if self._on_plan_mode_change is not None:
            return self._on_plan_mode_change(enabled)
        return None

    def plan_mode_allows(self, tool_call: ToolCall) -> bool:
        """Authorize one registry-resolved capability in the plan-mode layer."""

        return not self._plan_mode or self.tool_registry.call_allowed_by(
            tool_call, self._plan_mode_policy
        )


__all__ = ["PlanModeRuntimeMixin"]
