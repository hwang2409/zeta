# Ollama

Zeta includes an Ollama-native provider for text and function tools using
Ollama 0.12.11's streaming `/api/chat` endpoint. It defaults to the loopback
endpoint `http://127.0.0.1:11434` and model `qwen3:4b` (no credentials are
used or persisted).

Configure an explicit endpoint with `ZETA_OLLAMA_BASE_URL`, or set
`ollama_base_url = "http://127.0.0.1:11434"` in `~/.zeta/settings.toml`.
Project settings cannot configure the endpoint. The environment variable wins.
Do not point this at a shared or untrusted network service: prompts and tool
results may contain private source code.

For a remote Ollama instance, use a local tunnel to the endpoint you control,
then run `zeta --provider ollama`. Zeta derives each known model's context
allocation from its model-window table; the default `qwen3:4b` window is 40,960
tokens. Unknown or locally created Ollama models use a conservative 8,192-token
default (still larger than Ollama's own default). Zeta resolves one effective
budget—the smaller of the requested/session budget and the model window—and
uses it for session metadata, context assembly, child loops, and Ollama's
`options.num_ctx` on every chat request. Larger `num_ctx` values use more memory
on the Ollama host; on smaller hardware, set `token_budget` lower in settings or
with the CLI option.

Ollama automations are not yet supported; Ollama is currently available for
interactive TUI sessions only. Image input and reasoning blocks are
intentionally unsupported in this initial release. Malformed tool arguments
fail the stream clearly rather than being guessed.
