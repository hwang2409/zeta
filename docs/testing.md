# Testing

Production provider names are `claude`, `codex`, and `ollama`. Unit tests inject
deterministic backends through the runtime backend-factory seam.

External UI harnesses that must launch `zeta serve` can set the test-only
`ZETA_TEST_SCRIPTED_PROVIDER=1` environment variable and select any real provider
and model from the catalog. The server then uses a deterministic echo backend
while retaining that real provider and model in protocol data. This variable is
not a supported user configuration surface.
