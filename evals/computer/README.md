# Disposable computer eval

An opt-in local eval that gives Zeta **one** generic MCP tool, `computer__bash`,
inside a disposable Docker container. The host runs the model client; the
container receives only the task fixture, never host credentials or a host
directory mount. The runner rejects a container whose inspected configuration
has a mount, network access, writable root, wrong user, or retained capabilities.
It exports named artifacts through tar, and tests repaired code in a separate
container. Success also requires a final assistant message. It uses the
existing `evals/tasks.jsonl` fixtures.

```sh
uv run python evals/computer/run.py --docker-context colima-zeta-eval --task count-and-write
uv run python evals/computer/run.py --docker-context colima-zeta-eval --task csv-parser-repair
```

Pass `--provider claude --model claude-opus-5-5` to test another backend;
`--keep-artifacts DIR` retains checked outputs. A Docker context and working
provider credentials are required. Use a dedicated Docker VM whose own mounts
are checked separately: container isolation does not audit the VM host. In
Colima 0.9.1, `mounts: []` unexpectedly mounted the host home read-write, so
the local `zeta-eval` profile uses a dedicated empty read-only dummy mount.
This is a command-only mock, not a browser-equipped computer or a production
sandbox. The image currently follows the `python:3.12-alpine` tag.
