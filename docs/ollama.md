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
then run `zeta --provider ollama`. Image input and reasoning blocks are
intentionally unsupported in this initial release. Malformed tool arguments
fail the stream clearly rather than being guessed.
