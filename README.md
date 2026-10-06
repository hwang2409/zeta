# Zeta

Zeta is a terminal agent harness. It owns the agent loop, conversation state,
context management, tool dispatch, approvals, session persistence, and provider
connections for Claude, Codex, Ollama, and a fake provider.

## Install and first run

Zeta requires Python 3.12 or newer. From a checkout, install the editable tool:

```sh
uv tool install --editable .
```

Log in to a provider, then start the TUI:

```sh
zeta login                  # Anthropic OAuth
zeta login --provider codex # or Codex OAuth
zeta
```

Use `zeta --help` for the complete list of options and subcommands.

## Commands and modes

- `zeta` — interactive terminal UI (TUI).
- `zeta -p "prompt"` — run one headless turn; add `--format json` for JSONL.
- `zeta serve` — serve one local frontend over a Unix socket; use `--port N`
  for localhost TCP. See [the serve protocol](docs/serve-protocol.md).
- `zeta automation` — list, approve, disable, import, and run automations;
  `zeta automation daemon` runs the foreground daemon. See
  [automations](docs/automations.md).
- `zeta project init` and `zeta project memory` — associate a directory and
  inspect or update its bounded project memory.
- `zeta inbox` — list messages for the current project inbox. See
  [project inboxes](docs/project-inbox.md).
- `zeta mcp` — add, list, test, trust, and manage MCP servers. Definitions live
  in `~/.zeta/mcp.json` and `<project>/.zeta/mcp.json`. See
  [MCP management](docs/mcp-management.md).
- `zeta session` — list, rename, export, delete, and inspect stored sessions.
- `zeta completion zsh` or `zeta completion bash` — print shell completion.

Inside the TUI, `/help` lists slash commands. Skills can be invoked with
`/skill-name` or `$skill-name`. Skills and agents are Markdown definitions
loaded from packaged files, `~/.zeta/skills/` or `~/.zeta/agents/`, and the
current project's `.zeta/skills/` or `.zeta/agents/`. Custom slash commands are
loaded from `~/.zeta/commands/` and `.zeta/commands/`.

## Configuration

The default data directory is `~/.zeta/`; set `ZETA_HOME` to use another one.
Global settings are in `~/.zeta/settings.toml`; project overrides are in
`<project>/.zeta/settings.toml`. MCP configuration uses `~/.zeta/mcp.json` and
`<project>/.zeta/mcp.json`; set `ZETA_MCP_CONFIG` to select another MCP file.
The user system prompt is `~/.zeta/AGENTS.md`. Repository `AGENTS.md` files
add project instructions as Zeta walks from the repository root to the current
directory.

## Docs

- [Automations](docs/automations.md)
- [Computer use](docs/computer-use.md)
- [Design notes](docs/design.md)
- [Fake provider](docs/fake-provider.md)
- [MCP management](docs/mcp-management.md)
- [Ollama](docs/ollama.md)
- [Project inbox](docs/project-inbox.md)
- [Serve protocol](docs/serve-protocol.md)

## Development

Install the development environment and run the test suite and linter:

```sh
uv sync
uv run --frozen pytest -q
uv run --frozen ruff check .
```

The package is under `src/zeta/`; tests are under `tests/`. Optional browser
dependencies use the `browser` extra. See the relevant document above before
running browser or computer-use workflows.
