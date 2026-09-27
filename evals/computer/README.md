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
uv run python evals/computer/run.py --docker-context colima-zeta-eval --task browser-todo-repair
```

Pass `--provider claude --model claude-opus-5-5` to test another backend;
`--keep-artifacts DIR` retains checked outputs. A Docker context and working
provider credentials are required. Use a dedicated Docker VM whose own mounts
are checked separately: container isolation does not audit the VM host. In
Colima 0.9.1, `mounts: []` unexpectedly mounted the host home read-write, so
the local `zeta-eval` profile uses a dedicated empty read-only dummy mount.
The browser task uses a separate, version-pinned [Playwright Python image](https://playwright.dev/python/docs/docker)
with Chromium and a seeded local page. It keeps the same one `bash` tool; the
agent writes and runs browser checks in the guest, then a fresh browser
container independently tests the exported page. It has no outbound network.
Chromium's own sandbox does not launch under this local container profile, so
do **not** use this mock for untrusted public sites. Browser action routing,
controlled site access, and Jev comparison remain separate experiments. The
command-only image still follows the `python:3.12-alpine` tag.
