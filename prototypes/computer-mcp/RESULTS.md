# Prototype results

Date: 2026-10-03

## Lima isolation

The VM was created with:

```text
limactl create --name=zeta-sandbox --mount-none --cpus=4 --memory=6 --tty=false template:docker
```

Verification before Docker use:

```text
$ limactl shell zeta-sandbox -- mount | grep -E 'virtiofs|9p|sshfs|/Users'
<no output>
exit=1

$ limactl shell zeta-sandbox -- ls /Users
ls: cannot access '/Users': No such file or directory
exit=2
```

The effective configuration reported `mounts: None`. Its only configured port
forward was the rootless Docker socket:

```text
guest /run/user/501/docker.sock -> host /Users/henry/.lima/zeta-sandbox/sock/docker.sock
```

All build, test, backend, and demo operations selected that socket explicitly.

## Image

A clean, no-cache arm64 build took 23.07 seconds. The final image was
774,614,304 bytes (738.7 MiB) and reported `linux/arm64`.

## Three final demo runs

All final runs used a temporary `ZETA_HOME`, Codex `gpt-5.6-luna`, `--yolo`, a
60-turn limit, and temporary settings that hard-denied every host built-in tool.
The grader read `/home/zeta/notes/demo.txt` before container destruction. Each
file contained the required two lines; Mousepad omitted the optional trailing
newline. Every observed tool call was a `computer__*` MCP call.

| Run | Pass | Model steps / tool calls | Model screenshots / JPEG bytes | Tokens (input + cache read + output) | Wall | Cleanup | Final screenshot |
|---|---:|---:|---:|---:|---:|---:|---|
| final-1 | yes | 10 / 9 | 9 / 184,550 | 26,070 + 52,992 + 372 | 31.58 s | yes | `/tmp/computer-demo/final-1/artifacts/final.jpg` (12,494 B) |
| final-2 | yes | 10 / 9 | 9 / 184,960 | 28,501 + 50,560 + 388 | 42.39 s | yes | `/tmp/computer-demo/final-2/artifacts/final.jpg` (12,904 B) |
| final-3 | yes | 10 / 9 | 9 / 185,150 | 28,863 + 50,560 + 434 | 35.61 s | yes | `/tmp/computer-demo/final-3/artifacts/final.jpg` (12,904 B) |

The final screenshots are valid 1024x640 JPEG files. Across the three runs, the
model received 27 screenshots and 554,660 compressed bytes.

### Action latency

Measured Docker exec latency across the three final runs:

| Action | Calls | Mean | Min | Max |
|---|---:|---:|---:|---:|
| screenshot | 27 | 56.4 ms | 52.4 ms | 86.8 ms |
| click | 10 | 130.6 ms | 128.0 ms | 133.6 ms |
| double click | 3 | 231.1 ms | 228.8 ms | 232.7 ms |
| type | 6 | 56.3 ms | 34.5 ms | 104.7 ms |
| key | 4 | 56.4 ms | 26.3 ms | 67.3 ms |

One run used `computer_wait`; that run predates the final wait-specific metric,
so its requested delay is included in wall time but not in the table. The final
backend now records wait latency directly.

### Screenshot token cost

A 1024x640 image contains 640 32x32 image patches. The three runs sent 5,760
patch units each and 17,280 total. The private `gpt-5.6-luna` endpoint does not
report its image-token multiplier separately, so an exact billed token count per
screenshot cannot be derived from aggregate usage. Base64 transport expanded
the three runs to 246,068, 246,616, and 246,868 bytes, respectively.

## What broke

1. The existing Colima VM exposed `/Users/henry` read-write despite
   `mounts: []`. Work stopped there. The dedicated mountless Lima VM fixed the
   host-filesystem boundary.
2. The first image pulls timed out because the host Docker CLI configuration
   invoked the Docker Desktop credential helper. An empty per-run
   `DOCKER_CONFIG` fixed the pull and also removed dependence on host contexts.
3. The isolated CLI configuration did not expose BuildKit, so `COPY --chmod`
   failed under the legacy builder. The image now invokes the copied entrypoint
   through `/bin/sh` and does not require that extension.
4. A first temporary `ZETA_HOME` had no Codex credential. The demo now copies
   only the OAuth file into the temporary home; refresh writes remain temporary.
5. One exploratory pair of concurrent demos caused MCP initialization to time
   out. Zeta then advertised its normal host built-ins, and the model used host
   `bash` and `read`. The final demo starts the desktop lazily after MCP setup,
   hard-denies all host built-ins, and fails grading if any non-computer tool is
   called. The three final runs were sequential and used only computer tools.
6. Zeta has no CLI tool allowlist, so built-ins remain visible even when they are
   hard-denied. A first-class allowlist would make “only sandbox tools exist”
   literal instead of an enforced execution policy.

## Recommendations

- Implement `DesktopBackend` directly with per-session Lima guests cloned from
  an immutable desktop template. Send screenshots and input over Lima SSH and
  destroy the guest at the lifecycle boundary. This removes the Docker daemon
  from the trusted runtime path.
- Add a Zeta headless tool allowlist. It should omit disallowed schemas, not only
  deny execution. This is necessary before treating `--yolo` as safe by
  construction.
- Add Anthropic's native computer tool for supported models and map its actions
  to the same backend. Keep these MCP functions as the portable Codex fallback.
- Reduce screenshot frequency or add safe action batching if vision cost is a
  problem. Keep grading based on guest state, not model prose.
