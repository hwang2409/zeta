# Scripted fake provider

The `fake` provider works offline. By default it answers every message with
`you said: <message>`. To test tool cards, approvals, thinking, errors, and
streaming without a network provider, give it a script.

## Turn it on

Set `ZETA_FAKE_SCRIPT` to the path of a JSON script **and** select the fake
provider:

```sh
ZETA_FAKE_SCRIPT=docs/fake-scripts/read-then-bash.json zeta serve --provider fake
ZETA_FAKE_SCRIPT=docs/fake-scripts/failures.json zeta --provider fake -p "missing file"
ZETA_FAKE_SCRIPT=docs/fake-scripts/long-markdown.json zeta --provider fake
```

- The script applies only when the provider is `fake`. Real providers ignore
  `ZETA_FAKE_SCRIPT`.
- Without `ZETA_FAKE_SCRIPT`, the fake provider behaves as before.
- Zeta reads and validates the script at startup. An invalid script stops
  `zeta serve` and `zeta -p` with an error that names the bad field, for
  example `script.json.rules[0].responses[1].steps[2].type: must be one of ...`.
  `zeta serve` reads the script once; restart it to load changes.

## Safety

Scripted tool calls go through the normal agent loop. The tool registry, the
approval policy, and `--tools`/`--disallowed-tools` apply as for a
real provider. A scripted `bash` call asks for approval the same way. Tool
results are real: a `read` reads the file, a denied call returns the denial.
In headless mode (`-p`), tools that need approval are denied unless you pass
`--yolo`.

## Format

```json
{
  "version": 1,
  "rules": [
    {
      "match": {"contains": "inspect"},
      "responses": [
        {
          "steps": [
            {"type": "thinking", "text": "Plan the work.", "chunk_size": 4, "delay": 0.02},
            {"type": "text", "text": "Reading the file."},
            {"type": "tool_call", "name": "read", "arguments": {"path": "README.md"}}
          ],
          "usage": {"input_tokens": 100, "output_tokens": 20}
        },
        {"steps": [{"type": "text", "text": "Done."}]}
      ]
    }
  ]
}
```

- `version`: must be `1`.
- `rules`: a non-empty list. For each user message, Zeta uses the first rule
  that matches.
  - `match` (optional): exactly one of `equals` (the user text with outer
    whitespace removed is equal), `contains` (substring), or `regex` (Python
    `re.search`). A rule without `match` matches all messages.
  - `responses`: a non-empty list. Each entry is one model call. The first
    model call after the user message uses `responses[0]`. After tool results,
    the next model call uses `responses[1]`, and so on.
- A response has `steps` (non-empty), and optional `usage` (object of
  non-negative integers, sent as provider usage) and `stop_reason` (default:
  `tool_use` when the response has a tool call, else `end_turn`).
- Steps:
  - `{"type": "text", "text": ..., "chunk_size"?: N, "delay"?: seconds}`
    streams text deltas. Without `chunk_size`, the text is one delta. `delay`
    is the wait before each delta.
  - `{"type": "thinking", ...}` has the same fields and streams thinking.
  - `{"type": "tool_call", "name": ..., "arguments"?: {...}, "id"?: ..., "delay"?: seconds}`
    requests a tool call. The default ID is `fake_<user message number>_<response index>_<step index>`,
    so replays give the same IDs.
  - `{"type": "error", "code": ..., "message"?: ..., "status"?: HTTP status, "delay"?: seconds}`
    ends the response with a provider error. It must be the last step.
- Unknown fields are errors.

When no rule matches, the turn fails with the provider error
`fake_script_no_match`. When a rule has no response left for a model call, the
turn fails with `fake_script_exhausted`.

The backend has no hidden state. It selects the response from the request
messages (the last user message and the number of assistant messages after
it). The same conversation gives the same events. A steering message starts a
new match.

## Examples

- [`fake-scripts/read-then-bash.json`](fake-scripts/read-then-bash.json): send
  a message that contains `inspect`. The model reads `README.md`, then asks
  approval for `bash`.
- [`fake-scripts/failures.json`](fake-scripts/failures.json): `missing ...`
  gives a failing `read`; `rate limit` streams text and then fails with a 429
  provider error; `overloaded` fails at once.
- [`fake-scripts/long-markdown.json`](fake-scripts/long-markdown.json): a long
  streamed answer with thinking, markdown, a table, and a code block.
