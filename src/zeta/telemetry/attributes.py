"""Semantic-convention names owned by zeta telemetry."""

# reserved content names: never emit or reuse these in v1.
RESERVED_CONTENT_ATTRIBUTES = frozenset(
    {
        "gen_ai.input.messages",
        "gen_ai.output.messages",
        "gen_ai.system_instructions",
        "gen_ai.tool.call.arguments",
        "gen_ai.tool.call.result",
    }
)

# zeta provider id -> GenAI ``gen_ai.provider.name`` well-known value.
GEN_AI_PROVIDER_NAMES = {"claude": "anthropic", "codex": "openai", "ollama": "ollama"}
