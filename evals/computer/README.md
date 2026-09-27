# Disposable computer eval

An opt-in local eval that gives Zeta **one** generic MCP tool, `computer__bash`,
inside a disposable Docker container. The host runs the model client; the
container receives only the task fixture, never host credentials or a host
directory mount. The runner rejects a container whose inspected configuration
has a mount, network access, writable root, wrong user, or retained capabilities.
It exports named artifacts through tar, and tests repaired code in a separate
container. Success also requires a final assistant message. It uses the
existing `evals/tasks.jsonl` fixtures and one browser-only fixture.

```sh
uv run python evals/computer/run.py --docker-context colima-zeta-eval --task count-and-write
uv run python evals/computer/run.py --docker-context colima-zeta-eval --task csv-parser-repair
uv run python evals/computer/run.py --docker-context colima-zeta-eval --task browser-todo-repair
uv run python evals/computer/run.py --docker-context colima-zeta-eval --task browser-issue-triage
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
The browser container uses [Playwright v1.63.0's seccomp profile](https://github.com/microsoft/playwright/blob/v1.63.0/utils/docker/seccomp_profile.json),
with only its `chroot` rule changed to allow that syscall after dropping all
container capabilities; see its [license](LICENSE-PLAYWRIGHT) and
[notice](NOTICE-PLAYWRIGHT). In the local VM,
Chromium launched with its sandbox enabled and its renderer had separate user
and PID namespaces, active seccomp filtering, and zero effective capabilities.
The regression requires sandboxed Chromium, but the generic `bash` tool cannot
force every agent-authored browser script to do so. The `browser-issue-triage`
task instead mounts only `computer__browser`: the guest owns the Chromium launch
with `chromium_sandbox=True`, accepts typed page actions but no shell commands,
and allows only `/workspace` file URLs. Its grader checks the last actual
browser snapshot, not the agent's final claim, and confirms the seeded file was
unchanged. It remains networkless. This does not establish production safety:
public-site access still needs controlled egress and a separate eval. Jev
comparison remains a separate experiment. The command-only image still follows the
`python:3.12-alpine` tag.
