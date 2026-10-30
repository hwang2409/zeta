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
allocation from its model-window table:

| Model | Zeta context window | Behavior |
| --- | ---: | --- |
| `qwen3:4b` | 40,960 tokens | Thinking model; may spend substantial time reasoning before visible text or a tool call. |
| `qwen3:4b-instruct` | 32,768 tokens | Non-thinking instruct model; generally responds much faster and is useful for headless tool calls. |

The `qwen3:4b-instruct` allocation is deliberately below the model's native
window to bound Ollama memory use. For an unknown or locally created model,
Zeta calls Ollama's `/api/show` once when it builds the backend. It reads the
architecture's `context_length` and uses a lower `num_ctx` from the model's
parameters when present. A missing or invalid response, server error, or short
request timeout falls back to a conservative 8,192-token window.

Zeta resolves one effective budget—the smaller of the requested/session budget
and the model window—and uses it for session metadata, context assembly, child
loops, and Ollama's `options.num_ctx` on every chat request. If the model window
is too small for Zeta's prompt and tools, increase the model's `num_ctx` in
Ollama up to its architecture limit, or use a model with a larger context
window. Larger `num_ctx` values use more memory on the Ollama host; on smaller
hardware, set `token_budget` lower in settings or with the CLI option. For
example:

```console
zeta --provider ollama --model qwen3:4b-instruct --token-budget 32768
zeta --provider ollama --model qwen3:4b-instruct -p "Summarize this repository"
zeta --provider ollama --model qwen3:4b-instruct -p "Summarize this repository" --format json
```

The same model can be selected in `~/.zeta/settings.toml`:

```toml
provider = "ollama"
model = "qwen3:4b-instruct"
```

Ollama `-p`/`--print` headless runs,
`--format json` output, interactive TUI, and `zeta serve` sessions are supported.
Image input is intentionally unsupported. Native thinking is shown live in the
TUI, but is not stored or replayed; headless/print output excludes it. Ollama
also has a server-side think toggle, but Zeta does not wire that toggle yet.
Malformed tool arguments fail the stream clearly rather than being guessed.
